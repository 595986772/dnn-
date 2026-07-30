"""Concurrent multi-DNN route-and-batch environment with stateful interference.

The legacy environment serializes every device.  This independent variant
keeps the same action library but lets configured devices execute multiple
batches concurrently.  Active batches consume isolated service work at a
state-dependent rate, so adding a co-runner can delay every batch already on
that device.  The analytic interference proxy is intentionally configurable;
final paper experiments should replace or calibrate it with co-run profiles.
"""

from collections import deque
import heapq
import itertools

import numpy as np

from collab_batching.sequential_route_batch_env import SequentialRouteBatchEnv


class ConcurrentInterferenceRouteBatchEnv(SequentialRouteBatchEnv):
    """Event-driven multi-device inference with concurrent batch interference."""

    def __init__(
        self,
        *args,
        history_length=16,
        server_concurrency=None,
        interference_strength=0.0,
        cross_model_factor=1.15,
        max_slowdown=3.0,
        energy_interference_ratio=0.35,
        deadline_multiplier_by_model=None,
        **kwargs,
    ):
        self.history_length = int(history_length)
        if self.history_length <= 0:
            raise ValueError("history_length must be positive")
        super().__init__(*args, **kwargs)
        if self.sla_penalty_timing != "deadline_event":
            raise ValueError(
                "concurrent interference requires deadline-event SLA penalties"
            )
        capacities = (
            np.ones(self.N, dtype=np.int64)
            if server_concurrency is None
            else np.asarray(server_concurrency, dtype=np.int64)
        )
        if capacities.shape != (self.N,) or np.any(capacities <= 0):
            raise ValueError("server concurrency must provide one positive slot count")
        self.server_concurrency = capacities
        self.interference_strength = float(interference_strength)
        self.cross_model_factor = float(cross_model_factor)
        self.max_slowdown = float(max_slowdown)
        self.energy_interference_ratio = float(energy_interference_ratio)
        if self.interference_strength < 0.0:
            raise ValueError("interference strength must be nonnegative")
        if self.cross_model_factor <= 0.0 or self.max_slowdown < 1.0:
            raise ValueError("invalid interference multiplier bounds")
        if self.energy_interference_ratio < 0.0:
            raise ValueError("energy interference ratio must be nonnegative")

        multipliers = np.ones(self.n_models, dtype=np.float64)
        for model, value in (deadline_multiplier_by_model or {}).items():
            if str(model) not in self.models:
                raise ValueError("unknown deadline model: %s" % model)
            multipliers[self.models.index(str(model))] = float(value)
        if np.any(multipliers <= 0.0):
            raise ValueError("deadline multipliers must be positive")
        self.model_deadline_multipliers = multipliers
        self._complexity = self._build_interference_complexity()
        self._reset_storage()

    def _build_interference_complexity(self):
        values = np.zeros((self.N, self.n_models), dtype=np.float64)
        for server, provider in enumerate(self.providers):
            for model_index, model in enumerate(self.models):
                values[server, model_index] = self.catalog.table(model).latency_ms(
                    provider, 1
                )
            values[server] /= max(float(values[server].max()), 1e-9)
        return np.clip(values, 0.05, 1.0)

    def _reset_storage(self):
        super()._reset_storage()
        self._active_batches = []
        self._active_last_update_s = 0.0
        self._batch_counter = itertools.count()
        self._request_deadline_s = np.zeros(0, dtype=np.float64)
        self._feedback_trace = []
        self._feedback_events = deque(maxlen=self.history_length)
        self._peak_active_by_server = np.zeros(self.N, dtype=np.int64)
        self._concurrent_dispatches = 0
        self._causal_arrival_events = []

    def causal_arrival_events(self, start=0):
        """Return only arrivals already revealed to the online scheduler."""

        return tuple(self._causal_arrival_events[int(start) :])

    def reset(self):
        self._reset_storage()
        self._rng = np.random.default_rng(self.seed)
        self._latency_rng = np.random.default_rng(self.seed + 1_000_003)
        self._arrival_times, self._request_models = self._arrival_trace(self._rng)
        request_count = len(self._arrival_times)
        self._request_servers = np.full(request_count, -1, dtype=np.int64)
        self._dispatch_times = np.full(request_count, np.nan, dtype=np.float64)
        self._completion_times = np.full(request_count, np.nan, dtype=np.float64)
        self._sla_penalized = np.zeros(request_count, dtype=bool)
        self._request_deadline_s = (
            self.deadline_s * self.model_deadline_multipliers[self._request_models]
        )
        for request_id, arrival_s in enumerate(self._arrival_times):
            self._push(arrival_s, "arrival", int(request_id))
        return self._advance_to_decision(force_time_advance=True)

    def deadline_s_for_model(self, model_index):
        return float(
            self.deadline_s * self.model_deadline_multipliers[int(model_index)]
        )

    def deadline_s_for_request(self, request_id):
        return float(self._request_deadline_s[int(request_id)])

    def _deadline_times(self):
        return self._arrival_times + self._request_deadline_s

    def observation_size(self):
        return int(
            self._base_observation_size()
            + 2 * self.N
            + self.N * self.n_models
        )

    @property
    def history_feature_dim(self):
        return int(5 + self.n_models + self.N)

    def _active_for_server(self, server):
        return [
            row for row in self._active_batches if int(row["server"]) == int(server)
        ]

    def _record_demand(self, row):
        server = int(row["server"])
        model = int(row["model_index"])
        batch_fraction = np.sqrt(
            float(row["batch_size"])
            / max(float(max(self.action_space.batch_sizes)), 1.0)
        )
        return float(
            np.clip(
                0.20
                + 0.80 * np.sqrt(self._complexity[server, model] * batch_fraction),
                0.20,
                1.0,
            )
        )

    def _slowdown(self, row, active_rows=None):
        active_rows = (
            self._active_for_server(row["server"])
            if active_rows is None
            else list(active_rows)
        )
        penalty = 0.0
        for other in active_rows:
            if int(other["batch_id"]) == int(row["batch_id"]):
                continue
            cross = (
                self.cross_model_factor
                if int(other["model_index"]) != int(row["model_index"])
                else 1.0
            )
            penalty += cross * self._record_demand(other)
        return float(
            np.clip(
                1.0 + self.interference_strength * penalty,
                1.0,
                self.max_slowdown,
            )
        )

    def predict_runtime_s(self, model_index, server, batch_size):
        isolated = (
            self.catalog.table(self.models[int(model_index)]).latency_ms(
                self.providers[int(server)], int(batch_size)
            )
            / 1000.0
        )
        candidate = {
            "batch_id": -1,
            "server": int(server),
            "model_index": int(model_index),
            "batch_size": int(batch_size),
            "isolated_runtime_s": float(isolated),
        }
        active = self._active_for_server(server) + [candidate]
        return float(isolated * self._slowdown(candidate, active))

    def _advance_active_work(self, target_s):
        target_s = float(target_s)
        elapsed = target_s - float(self._active_last_update_s)
        if elapsed < -1e-12:
            raise RuntimeError("active service clock moved backwards")
        if elapsed > 0.0:
            for server in range(self.N):
                active = self._active_for_server(server)
                for row in active:
                    slowdown = self._slowdown(row, active)
                    row["remaining_work_s"] -= elapsed / slowdown
                    row["slowdown_time_integral"] += elapsed * slowdown
        self._active_last_update_s = target_s

    def _next_completion_time(self):
        candidates = []
        for server in range(self.N):
            active = self._active_for_server(server)
            for row in active:
                candidates.append(
                    self.now_s
                    + max(float(row["remaining_work_s"]), 0.0)
                    * self._slowdown(row, active)
                )
        return min(candidates) if candidates else None

    def _refresh_busy_compatibility(self):
        for server in range(self.N):
            active = self._active_for_server(server)
            self._busy[server] = bool(active)
            if not active:
                self._busy_until_s[server] = self.now_s
                self._busy_model[server] = -1
                continue
            finish_estimates = [
                self.now_s
                + max(float(row["remaining_work_s"]), 0.0)
                * self._slowdown(row, active)
                for row in active
            ]
            self._busy_until_s[server] = min(finish_estimates)
            models = {int(row["model_index"]) for row in active}
            self._busy_model[server] = next(iter(models)) if len(models) == 1 else -2

    def _finish_ready_batches(self):
        ready = [
            row for row in self._active_batches if row["remaining_work_s"] <= 1e-9
        ]
        if not ready:
            return
        ready_ids = {int(row["batch_id"]) for row in ready}
        self._active_batches = [
            row
            for row in self._active_batches
            if int(row["batch_id"]) not in ready_ids
        ]
        for row in ready:
            request_ids = row["request_ids"]
            self._completion_times[request_ids] = self.now_s
            self._completion_history.append((self.now_s, int(len(request_ids))))
            service_s = max(self.now_s - float(row["dispatch_s"]), 0.0)
            realized_slowdown = service_s / max(
                float(row["isolated_runtime_s"]), 1e-9
            )
            row["batch_row"]["finish_s"] = float(self.now_s)
            row["batch_row"]["realized_slowdown"] = float(realized_slowdown)
            deadlines = self._arrival_times[request_ids] + self._request_deadline_s[
                request_ids
            ]
            row["batch_row"]["violation_count"] = int(
                np.count_nonzero(self.now_s > deadlines + 1e-12)
            )
            self._feedback_trace.append(
                {
                    "completion_s": float(self.now_s),
                    "model_index": int(row["model_index"]),
                    "server": int(row["server"]),
                    "batch_size": int(row["batch_size"]),
                    "latency_ratio": float(realized_slowdown),
                    "energy_ratio": float(row["energy_ratio"]),
                }
            )
            self._feedback_events.append(self._feedback_trace[-1])
        self._refresh_busy_compatibility()

    def _process_events_at(self, target_s):
        self._advance_active_work(target_s)
        self.now_s = float(target_s)
        while self._events and self._events[0][0] <= self.now_s + 1e-12:
            _, _, event_type, payload = heapq.heappop(self._events)
            if event_type != "arrival":
                raise AssertionError("concurrent environment only heaps arrivals")
            request_id = int(payload)
            model_index = int(self._request_models[request_id])
            self._pending[model_index].append(request_id)
            self._causal_arrival_events.append(
                (float(self.now_s), int(model_index))
            )
            self._recent_arrivals[model_index] += 1.0
            self._arrival_history.append((self.now_s, 1))
        self._finish_ready_batches()

    def _collect_new_deadline_violation_counts(self):
        if not len(self._arrival_times):
            return np.zeros(self.n_models, dtype=np.int64)
        deadlines = self._deadline_times()
        newly_due = (deadlines <= self.now_s + 1e-12) & (~self._sla_penalized)
        if not np.any(newly_due):
            return np.zeros(self.n_models, dtype=np.int64)
        completed_on_time = (
            np.isfinite(self._completion_times)
            & (self._completion_times <= deadlines + 1e-12)
        )
        violations = newly_due & (~completed_on_time)
        self._sla_penalized[newly_due] = True
        return np.bincount(
            self._request_models[violations], minlength=self.n_models
        ).astype(np.int64)

    def _physical_and_safe_masks(self):
        physical = np.zeros(self.action_dim, dtype=bool)
        safe = np.zeros(self.action_dim, dtype=bool)
        active_counts = np.asarray(
            [len(self._active_for_server(server)) for server in range(self.N)]
        )
        for index in range(1, self.action_dim):
            model_index, server, batch_size = self.action_space.decode(index)
            if (
                self.allowed_batch_sizes is not None
                and batch_size not in self.allowed_batch_sizes
            ):
                continue
            fixed_server = self.fixed_provider_by_model.get(model_index)
            if fixed_server is not None and server != fixed_server:
                continue
            queue = self._pending[model_index]
            minimum_requests = 1 if self.partial_batch_flush else batch_size
            if (
                active_counts[server] >= self.server_concurrency[server]
                or len(queue) < minimum_requests
            ):
                continue
            physical[index] = True
            oldest = int(queue[0])
            runtime_s = self.predict_runtime_s(model_index, server, batch_size)
            safe[index] = (
                self.now_s + runtime_s
                <= self._arrival_times[oldest]
                + self._request_deadline_s[oldest]
                + 1e-12
            )
        return physical, safe

    def _latest_safe_wait_time(self, physical):
        latest_times = []
        for model_index, queue in enumerate(self._pending):
            model_actions = [
                index
                for index in np.flatnonzero(physical)
                if self.action_space.decode(index)[0] == model_index
            ]
            if not queue or not model_actions:
                continue
            best_runtime_s = min(
                self.predict_runtime_s(
                    model_index,
                    self.action_space.decode(index)[1],
                    self.action_space.decode(index)[2],
                )
                for index in model_actions
            )
            oldest = int(queue[0])
            latest_times.append(
                self._arrival_times[oldest]
                + self._request_deadline_s[oldest]
                - best_runtime_s
            )
        return min(latest_times) if latest_times else None

    def _terminal(self):
        return bool(
            not self._events
            and not self._active_batches
            and all(not queue for queue in self._pending)
        )

    def _advance_one_time_point(self):
        physical, _ = self._physical_and_safe_masks()
        urgency_s = self._latest_safe_wait_time(physical)
        candidates = []
        if self._events:
            candidates.append(float(self._events[0][0]))
        completion_s = self._next_completion_time()
        if completion_s is not None:
            candidates.append(float(completion_s))
        if urgency_s is not None and urgency_s > self.now_s + 1e-12:
            candidates.append(float(urgency_s))
        if not candidates:
            if self._terminal():
                self._done = True
                return
            raise RuntimeError("concurrent event simulator cannot make progress")
        self._process_events_at(min(candidates))

    def _concurrency_observation(self):
        active_fraction = np.zeros(self.N, dtype=np.float64)
        interference = np.zeros(self.N, dtype=np.float64)
        model_fraction = np.zeros((self.N, self.n_models), dtype=np.float64)
        for server in range(self.N):
            active = self._active_for_server(server)
            capacity = max(float(self.server_concurrency[server]), 1.0)
            active_fraction[server] = len(active) / capacity
            if active:
                interference[server] = np.mean(
                    [self._slowdown(row, active) - 1.0 for row in active]
                ) / max(self.max_slowdown - 1.0, 1e-9)
            for row in active:
                model_fraction[server, int(row["model_index"])] += 1.0 / capacity
        return np.concatenate(
            [active_fraction, interference, model_fraction.reshape(-1)]
        ).astype(np.float32)

    def _history_matrix(self):
        matrix = np.zeros(
            (self.history_length, self.history_feature_dim), dtype=np.float32
        )
        events = list(self._feedback_events)
        for row_index, event in enumerate(events):
            cursor = 0
            age = max(float(self.now_s) - float(event["completion_s"]), 0.0)
            matrix[row_index, cursor] = np.clip(
                age / max(10.0 * self.deadline_s, 1e-9), 0.0, 1.0
            )
            cursor += 1
            matrix[row_index, cursor] = (
                np.log2(max(int(event["batch_size"]), 1)) / 5.0
            )
            cursor += 1
            matrix[row_index, cursor] = np.clip(
                np.log(max(float(event["latency_ratio"]), 1e-9)), -1.5, 1.5
            )
            cursor += 1
            matrix[row_index, cursor] = np.clip(
                np.log(max(float(event["energy_ratio"]), 1e-9)), -1.5, 1.5
            )
            cursor += 1
            matrix[row_index, cursor] = 1.0
            cursor += 1
            matrix[row_index, cursor + int(event["model_index"])] = 1.0
            cursor += self.n_models
            matrix[row_index, cursor + int(event["server"])] = 1.0
        return matrix, int(len(events))

    def _observation(self):
        observation = super()._observation()
        vector = np.asarray(observation["vector"], dtype=np.float32).copy()
        oldest_slack_raw = np.ones(self.n_models, dtype=np.float64)
        for model_index, queue in enumerate(self._pending):
            if queue:
                oldest = int(queue[0])
                oldest_slack_raw[model_index] = (
                    self._arrival_times[oldest]
                    + self._request_deadline_s[oldest]
                    - self.now_s
                ) / self._request_deadline_s[oldest]
        oldest_start = 4 + 3 * self.n_models
        vector[oldest_start : oldest_start + self.n_models] = np.clip(
            oldest_slack_raw, -4.0, 1.0
        )
        observation["oldest_slack"] = np.clip(
            oldest_slack_raw, -4.0, 1.0
        ).astype(np.float32)

        if self.queue_distribution_observation:
            rows = []
            for queue in self._pending:
                if queue:
                    request_ids = np.fromiter(queue, dtype=np.int64)
                    slacks = (
                        self._arrival_times[request_ids]
                        + self._request_deadline_s[request_ids]
                        - self.now_s
                    ) / self._request_deadline_s[request_ids]
                    rows.append(
                        [
                            *np.clip(np.quantile(slacks, [0.25, 0.50, 0.75]), -4.0, 1.0),
                            float(np.mean(slacks <= 0.25)),
                            float(np.mean(slacks <= 0.0)),
                        ]
                    )
                else:
                    rows.append([1.0, 1.0, 1.0, 0.0, 0.0])
            queue_features = np.asarray(rows, dtype=np.float32).reshape(-1)
            vector[-5 * self.n_models :] = queue_features
            observation["queue_distribution_features"] = queue_features

        concurrency = self._concurrency_observation()
        vector = np.concatenate([vector, concurrency]).astype(np.float32)
        if len(vector) != self.observation_size():
            raise AssertionError("concurrent observation size mismatch")
        observation["vector"] = vector
        observation["active_batch_counts"] = np.asarray(
            [len(self._active_for_server(server)) for server in range(self.N)],
            dtype=np.int64,
        )
        observation["concurrency_features"] = concurrency
        history, length = self._history_matrix()
        observation["history"] = history
        observation["history_length"] = length
        return observation

    def step(self, action_index):
        if self._done:
            raise RuntimeError("step called after episode termination")
        action_index = int(action_index)
        current_mask = self.action_mask()
        if (
            action_index < 0
            or action_index >= self.action_dim
            or not current_mask[action_index]
        ):
            raise ValueError("selected action is masked or out of range")
        start_s = float(self.now_s)
        energy_cost = 0.0
        request_count = 0
        request_count_by_model = np.zeros(self.n_models, dtype=np.int64)
        model_index, server, batch_size = self.action_space.decode(action_index)
        dispatch_batch_id = -1

        if model_index is None:
            kind = "wait"
            blocking_batch_ids = [
                int(row["batch_id"]) for row in self._active_batches
            ]
            next_observation, next_mask = self._advance_to_decision(
                force_time_advance=True
            )
        else:
            kind = "dispatch"
            queue = self._pending[model_index]
            dispatch_size = (
                min(len(queue), batch_size)
                if self.partial_batch_flush
                else batch_size
            )
            request_ids = np.asarray(
                [int(queue.popleft()) for _ in range(dispatch_size)],
                dtype=np.int64,
            )
            table = self.catalog.table(self.models[model_index])
            provider = self.providers[server]
            profile_batch = table.effective_batch_size(provider, batch_size)
            isolated_runtime_s = table.runtime_latency_ms(
                provider, profile_batch, rng=self._latency_rng
            ) / 1000.0
            batch_id = int(next(self._batch_counter))
            dispatch_batch_id = batch_id
            batch_row = {
                "batch_id": batch_id,
                "server": int(server),
                "model": self.models[model_index],
                "batch_size": int(dispatch_size),
                "batch_cap": int(batch_size),
                "profile_batch_size": int(profile_batch),
                "dispatch_s": float(self.now_s),
                "finish_s": np.nan,
                "energy_j": 0.0,
                "violation_count": 0,
                "isolated_runtime_s": float(isolated_runtime_s),
                "realized_slowdown": np.nan,
            }
            active_row = {
                "batch_id": batch_id,
                "server": int(server),
                "model_index": int(model_index),
                "batch_size": int(batch_size),
                "request_ids": request_ids,
                "dispatch_s": float(self.now_s),
                "isolated_runtime_s": float(isolated_runtime_s),
                "remaining_work_s": float(isolated_runtime_s),
                "slowdown_time_integral": 0.0,
                "batch_row": batch_row,
            }
            active_with_candidate = self._active_for_server(server) + [active_row]
            dispatch_slowdown = self._slowdown(active_row, active_with_candidate)
            energy_ratio = 1.0 + self.energy_interference_ratio * (
                dispatch_slowdown - 1.0
            )
            energy_j = float(table.energy_j(provider, batch_size) * energy_ratio)
            active_row["energy_ratio"] = float(energy_ratio)
            batch_row["energy_j"] = energy_j
            if len(active_with_candidate) > 1:
                self._concurrent_dispatches += 1
            self._active_batches.append(active_row)
            active_count = len(self._active_for_server(server))
            self._peak_active_by_server[server] = max(
                self._peak_active_by_server[server], active_count
            )
            self._request_servers[request_ids] = int(server)
            self._dispatch_times[request_ids] = self.now_s
            self._energy_by_server[server] += energy_j
            self._total_padding += int(profile_batch - dispatch_size)
            request_count = int(len(request_ids))
            request_count_by_model[int(model_index)] = request_count
            self._dispatch_history.append((self.now_s, request_count))
            self._batches.append(batch_row)
            self._refresh_busy_compatibility()
            blocking_batch_ids = [
                int(row["batch_id"]) for row in self._active_batches
            ]
            next_observation, next_mask = self._advance_to_decision(
                force_time_advance=False
            )

        violation_count_by_model = self._collect_new_deadline_violation_counts()
        violation_count = int(violation_count_by_model.sum())
        sla_cost_by_model = violation_count_by_model.astype(np.float64)
        sla_cost = float(violation_count)
        energy_cost = (
            0.0
            if model_index is None
            else float(self._batches[-1]["energy_j"] / self.energy_ref_per_request)
        )
        reward = self.reward_scale * (
            -energy_cost - self.sla_lambda * sla_cost
        )
        trace_row = {
            "time_s": start_s,
            "action_index": action_index,
            "action_label": self.action_space.label(action_index),
            "kind": kind,
            "dispatch_batch_id": int(dispatch_batch_id),
            "blocking_batch_ids": list(blocking_batch_ids),
            "reward": float(reward),
            "energy_cost": float(energy_cost),
            "sla_cost": float(sla_cost),
            "request_count": int(request_count),
            "violation_count": int(violation_count),
            "request_count_by_model": request_count_by_model.tolist(),
            "violation_count_by_model": violation_count_by_model.tolist(),
            "sla_cost_by_model": sla_cost_by_model.tolist(),
            "valid_action_count": int(current_mask.sum()),
            "elapsed_s": float(self.now_s - start_s),
            "active_batch_counts": [
                len(self._active_for_server(server)) for server in range(self.N)
            ],
        }
        self._decision_trace.append(trace_row)
        return next_observation, next_mask, float(reward), bool(self._done), dict(
            trace_row
        )

    def result(self, record_requests=False):
        result = super().result(record_requests=record_requests)
        deadlines = self._deadline_times()
        violations = self._completion_times > deadlines + 1e-12
        requests_by_model = np.bincount(
            self._request_models, minlength=self.n_models
        ).astype(np.int64)
        violations_by_model = np.bincount(
            self._request_models[violations], minlength=self.n_models
        ).astype(np.int64)
        result.update(
            {
                "violation_count": int(violations.sum()),
                "violation_rate": float(violations.mean()) if len(violations) else 0.0,
                "requests_by_model": requests_by_model,
                "violation_count_by_model": violations_by_model,
                "violation_rate_by_model": np.divide(
                    violations_by_model,
                    requests_by_model,
                    out=np.zeros(self.n_models, dtype=np.float64),
                    where=requests_by_model > 0,
                ),
                "server_concurrency": self.server_concurrency.copy(),
                "peak_active_by_server": self._peak_active_by_server.copy(),
                "concurrent_dispatch_fraction": float(
                    self._concurrent_dispatches / max(len(self._batches), 1)
                ),
                "mean_realized_slowdown": float(
                    np.nanmean(
                        [row.get("realized_slowdown", np.nan) for row in self._batches]
                    )
                )
                if self._batches
                else 1.0,
            }
        )
        if record_requests:
            result["request_deadline_s"] = self._request_deadline_s.copy()
            result["request_deadline_times_s"] = deadlines.copy()
        return result
