"""Event-level sequential routing and batching for heterogeneous inference.

Requests wait in one FIFO queue per DNN model.  At each scheduling event the
policy either dispatches one concrete ``(model, server, batch_size)`` action or
waits.  Routing therefore happens when a batch is dispatched, which gives one
action a directly attributable energy and request-level SLA outcome.
"""

from collections import deque
import heapq
import itertools

import numpy as np

from collab_batching.online_batch_env import (
    MultiModelProfileCatalog,
    _normalize_model_mix,
)
from collab_batching.openb_arrival_source import OpenBArrivalSource


class AtomicRouteBatchActionSpace:
    """Finite WAIT plus model-server-batch action library."""

    def __init__(self, models, providers, batch_sizes=(1, 2, 4, 8, 16, 32)):
        self.models = tuple(str(value) for value in models)
        self.providers = tuple(str(value) for value in providers)
        self.batch_sizes = tuple(sorted({int(value) for value in batch_sizes}))
        if not self.batch_sizes or min(self.batch_sizes) <= 0:
            raise ValueError("batch_sizes must contain positive integers")
        self.actions = [(None, None, None)]
        for model_index in range(len(self.models)):
            for server in range(len(self.providers)):
                for batch_size in self.batch_sizes:
                    self.actions.append((model_index, server, batch_size))

    def __len__(self):
        return len(self.actions)

    @property
    def wait_index(self):
        return 0

    def decode(self, index):
        index = int(index)
        if index < 0 or index >= len(self.actions):
            raise ValueError("action index is out of range")
        return self.actions[index]

    def label(self, index):
        model_index, server, batch_size = self.decode(index)
        if model_index is None:
            return "WAIT"
        return "%s@%s:b%d" % (
            self.models[model_index],
            self.providers[server],
            batch_size,
        )


class SequentialRouteBatchEnv:
    """Central-queue, event-driven route-and-batch scheduling environment.

    The actor never observes future arrivals or runtime draws.  A lightweight
    safety mask uses only current queues, request deadlines, idle servers, and
    the measured mean profile.  If a model has at least one predicted-feasible
    dispatch, predicted-infeasible alternatives are hidden; when no feasible
    action exists, physical actions remain available so overload can drain.
    """

    def __init__(
        self,
        profile_csv,
        server_profiles,
        arrival_rate_rps,
        horizon_s=20.0,
        deadline_s=0.10,
        model_mix=None,
        seed=0,
        batch_sizes=(1, 2, 4, 8, 16, 32),
        allowed_batch_sizes=None,
        partial_batch_flush=False,
        fixed_provider_by_model=None,
        optional_wait=True,
        energy_ref_per_request=0.08,
        sla_lambda=2.0,
        excess_beta=0.25,
        excess_clip=1.0,
        sla_penalty_timing="dispatch",
        reward_scale=0.05,
        predicted_safety_mask=True,
        overload_observation=False,
        causal_workload_features=False,
        queue_distribution_observation=False,
        recovery_window_s=0.0,
        arrival_mode="poisson",
        burst_strength=0.0,
        openb_trace_csv=None,
        openb_split="train",
        openb_lookback_days=30.0,
        openb_raw_bin_s=3600.0,
        openb_sim_bin_s=0.25,
        openb_sampling_mode="uniform",
        openb_hard_fraction=0.5,
        openb_hard_quantile=0.8,
        catalog=None,
    ):
        self.catalog = catalog or MultiModelProfileCatalog(profile_csv)
        self.models = list(self.catalog.models)
        self.providers = list(server_profiles)
        unknown = sorted(set(self.providers) - set(self.catalog.providers))
        if unknown:
            raise ValueError("unknown server profiles: %s" % unknown)
        self.N = len(self.providers)
        self.n_models = len(self.models)
        self.arrival_rate_rps = float(arrival_rate_rps)
        self.horizon_s = float(horizon_s)
        self.deadline_s = float(deadline_s)
        self.model_mix = _normalize_model_mix(self.models, model_mix)
        self.seed = int(seed)
        self.energy_ref_per_request = float(energy_ref_per_request)
        self.sla_lambda = float(sla_lambda)
        self.excess_beta = float(excess_beta)
        self.excess_clip = float(excess_clip)
        if self.excess_clip <= 0.0:
            raise ValueError("excess_clip must be positive")
        self.sla_penalty_timing = str(sla_penalty_timing).strip().lower()
        if self.sla_penalty_timing not in {"dispatch", "deadline_event"}:
            raise ValueError("sla_penalty_timing must be dispatch or deadline_event")
        self.reward_scale = float(reward_scale)
        self.optional_wait = bool(optional_wait)
        self.partial_batch_flush = bool(partial_batch_flush)
        self.predicted_safety_mask = bool(predicted_safety_mask)
        self.overload_observation = bool(overload_observation)
        self.causal_workload_features = bool(causal_workload_features)
        self.queue_distribution_observation = bool(
            queue_distribution_observation
        )
        self.arrival_mode = str(arrival_mode).lower()
        self.burst_strength = float(burst_strength)
        if self.arrival_mode not in {"poisson", "hyperexponential", "openb_guided"}:
            raise ValueError(
                "arrival_mode must be poisson, hyperexponential, or openb_guided"
            )
        if not 0.0 <= self.burst_strength < 1.0:
            raise ValueError("burst_strength must be in [0, 1)")
        self.recovery_window_s = (
            float(recovery_window_s)
            if float(recovery_window_s) > 0.0
            else max(5.0 * self.deadline_s, 0.20)
        )
        if self.arrival_rate_rps < 0.0 or self.horizon_s <= 0.0:
            raise ValueError("arrival rate and horizon must be valid")
        if self.deadline_s <= 0.0 or self.energy_ref_per_request <= 0.0:
            raise ValueError("deadline and energy reference must be positive")
        self.openb_split = str(openb_split).strip().lower()
        self.openb_sampling_mode = str(openb_sampling_mode).strip().lower()
        self.openb_hard_fraction = float(openb_hard_fraction)
        self.openb_hard_quantile = float(openb_hard_quantile)
        self._openb_arrival_source = None
        if self.arrival_mode == "openb_guided":
            if not str(openb_trace_csv or "").strip():
                raise ValueError("openb_trace_csv is required for openb_guided arrivals")
            self._openb_arrival_source = OpenBArrivalSource(
                openb_trace_csv,
                lookback_days=openb_lookback_days,
                raw_bin_s=openb_raw_bin_s,
                sim_bin_s=openb_sim_bin_s,
            )
            self._openb_arrival_source.stats(self.openb_split)
        valid_batches = [
            value for value in batch_sizes if int(value) <= self.catalog.max_batch_size
        ]
        self.action_space = AtomicRouteBatchActionSpace(
            self.models, self.providers, valid_batches
        )
        self.allowed_batch_sizes = (
            None
            if allowed_batch_sizes is None
            else frozenset(int(value) for value in allowed_batch_sizes)
        )
        if self.allowed_batch_sizes is not None:
            unknown_batches = sorted(
                self.allowed_batch_sizes - set(self.action_space.batch_sizes)
            )
            if unknown_batches or not self.allowed_batch_sizes:
                raise ValueError(
                    "allowed_batch_sizes must be a nonempty subset of the action library"
                )
        self.fixed_provider_by_model = {}
        for model, provider in dict(fixed_provider_by_model or {}).items():
            if str(model) not in self.models:
                raise ValueError("unknown model in fixed provider map: %s" % model)
            if str(provider) not in self.providers:
                raise ValueError("unknown provider in fixed provider map: %s" % provider)
            self.fixed_provider_by_model[self.models.index(str(model))] = (
                self.providers.index(str(provider))
            )
        self._aggregate_peak_rps = self.catalog.aggregate_peak_rps(
            self.providers, model_mix=self.model_mix
        )
        self._scheduler_observation_capacity_rps = self.catalog.aggregate_peak_rps(
            self.providers,
            model_mix=np.full(self.n_models, 1.0 / self.n_models),
        )
        self._scheduler_arrival_rates_rps = None
        self._scheduler_arrival_long_counts = np.zeros(
            self.n_models, dtype=np.float64
        )
        self._scheduler_arrival_rate_discrepancy = np.zeros(
            self.n_models, dtype=np.float64
        )
        self._scheduler_arrival_confidence = np.zeros(
            self.n_models, dtype=np.float64
        )
        self._profile_features = self._build_profile_features()
        self._reset_storage()

    @property
    def action_dim(self):
        return len(self.action_space)

    def _base_observation_size(self):
        base = int(
            4
            + 4 * self.n_models
            + self.N
            + self.N * self.n_models
            + 5 * self.n_models * self.N
        )
        if self.overload_observation:
            # Per-model uncapped queue/overdue pressure plus four recent-rate
            # features: arrivals, completions, dispatches, and queue drift.
            base += 2 * self.n_models + 4
        if self.causal_workload_features:
            # Per-model long-window count, short/long rate discrepancy,
            # and confidence from revealed arrivals only.
            base += 3 * self.n_models
        if self.queue_distribution_observation:
            # Three current-slack quantiles, near-deadline ratio, overdue ratio.
            base += 5 * self.n_models
        return base

    def observation_size(self):
        return self._base_observation_size()

    def set_scheduler_arrival_rate_estimate(
        self,
        rates_rps,
        long_window_counts=None,
        rate_discrepancy_rps=None,
        arrival_confidence=None,
    ):
        """Set causal per-DNN rates used only to build scheduler observations."""

        rates = np.asarray(rates_rps, dtype=np.float64)
        if rates.shape != (self.n_models,) or np.any(rates < 0.0):
            raise ValueError("scheduler arrival-rate estimate has invalid shape")
        self._scheduler_arrival_rates_rps = rates.copy()
        optional = {
            "long_window_counts": long_window_counts,
            "rate_discrepancy_rps": rate_discrepancy_rps,
            "arrival_confidence": arrival_confidence,
        }
        parsed = {}
        for name, values in optional.items():
            if values is None:
                parsed[name] = np.zeros(self.n_models, dtype=np.float64)
                continue
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (self.n_models,) or not np.all(np.isfinite(array)):
                raise ValueError("scheduler %s has invalid shape" % name)
            parsed[name] = array.copy()
        if np.any(parsed["long_window_counts"] < 0.0):
            raise ValueError("scheduler long-window counts must be nonnegative")
        if np.any(
            (parsed["arrival_confidence"] < 0.0)
            | (parsed["arrival_confidence"] > 1.0)
        ):
            raise ValueError("scheduler arrival confidence must be in [0, 1]")
        self._scheduler_arrival_long_counts = parsed["long_window_counts"]
        self._scheduler_arrival_rate_discrepancy = parsed[
            "rate_discrepancy_rps"
        ]
        self._scheduler_arrival_confidence = parsed["arrival_confidence"]

    def clear_scheduler_arrival_rate_estimate(self):
        self._scheduler_arrival_rates_rps = None
        self._scheduler_arrival_long_counts.fill(0.0)
        self._scheduler_arrival_rate_discrepancy.fill(0.0)
        self._scheduler_arrival_confidence.fill(0.0)

    def scheduler_observation(self):
        """Rebuild the current state after an online estimator update."""

        return self._observation()

    def _observation_workload_parameters(self):
        if self._scheduler_arrival_rates_rps is None:
            return (
                float(self.arrival_rate_rps),
                np.asarray(self.model_mix, dtype=np.float64),
                float(self._aggregate_peak_rps),
            )
        rates = np.maximum(
            np.asarray(self._scheduler_arrival_rates_rps, dtype=np.float64), 0.0
        )
        total = float(rates.sum())
        mix = (
            rates / total
            if total > 1e-12
            else np.full(self.n_models, 1.0 / self.n_models)
        )
        return total, mix, float(self._scheduler_observation_capacity_rps)

    def _build_profile_features(self):
        energy_values = []
        for model in self.models:
            table = self.catalog.table(model)
            for provider in self.providers:
                for batch_size in (1, 4, 16):
                    energy_values.append(
                        table.energy_j(provider, batch_size) / float(batch_size)
                    )
        energy_scale = max(max(energy_values), 1e-9)
        rows = []
        for model in self.models:
            table = self.catalog.table(model)
            model_rows = []
            for provider in self.providers:
                model_rows.append(
                    [
                        table.latency_ms(provider, 1) / (1000.0 * self.deadline_s),
                        table.latency_ms(provider, 4) / (1000.0 * self.deadline_s),
                        table.latency_ms(provider, 16) / (1000.0 * self.deadline_s),
                        table.energy_j(provider, 1) / energy_scale,
                        (table.energy_j(provider, 16) / 16.0) / energy_scale,
                    ]
                )
            rows.append(model_rows)
        return np.asarray(rows, dtype=np.float64)

    def _reset_storage(self):
        self.now_s = 0.0
        self._rng = None
        self._latency_rng = None
        self._events = []
        self._event_counter = itertools.count()
        self._arrival_times = np.zeros(0, dtype=np.float64)
        self._request_models = np.zeros(0, dtype=np.int64)
        self._request_servers = np.zeros(0, dtype=np.int64)
        self._dispatch_times = np.zeros(0, dtype=np.float64)
        self._completion_times = np.zeros(0, dtype=np.float64)
        self._sla_penalized = np.zeros(0, dtype=bool)
        self._pending = [deque() for _ in range(self.n_models)]
        self._busy = np.zeros(self.N, dtype=bool)
        self._busy_until_s = np.zeros(self.N, dtype=np.float64)
        self._busy_model = np.full(self.N, -1, dtype=np.int64)
        self._recent_arrivals = np.zeros(self.n_models, dtype=np.float64)
        self._arrival_history = deque()
        self._completion_history = deque()
        self._dispatch_history = deque()
        self._energy_by_server = np.zeros(self.N, dtype=np.float64)
        self._batches = []
        self._decision_trace = []
        self._total_padding = 0
        self._done = False
        self._last_mask = None

    def _push(self, time_s, event_type, payload):
        heapq.heappush(
            self._events,
            (float(time_s), next(self._event_counter), str(event_type), payload),
        )

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
        for request_id, arrival_s in enumerate(self._arrival_times):
            self._push(arrival_s, "arrival", int(request_id))
        return self._advance_to_decision(force_time_advance=True)

    def _arrival_trace(self, rng):
        if self.arrival_mode == "openb_guided":
            return self._openb_arrival_source.sample(
                self.openb_split,
                self.arrival_rate_rps,
                self.horizon_s,
                self.model_mix,
                rng,
                sampling_mode=self.openb_sampling_mode,
                hard_fraction=self.openb_hard_fraction,
                hard_quantile=self.openb_hard_quantile,
            )
        if self.arrival_mode == "poisson" or self.burst_strength <= 1e-12:
            count = int(rng.poisson(self.arrival_rate_rps * self.horizon_s))
            times = np.sort(rng.uniform(0.0, self.horizon_s, size=count))
        elif self.arrival_rate_rps <= 0.0:
            times = np.zeros(0, dtype=np.float64)
            count = 0
        else:
            # This balanced hyper-exponential renewal process preserves the
            # expected inter-arrival time while adding short burst intervals
            # and compensating quiet intervals.
            multipliers = np.asarray(
                [1.0 - self.burst_strength, 1.0 + self.burst_strength],
                dtype=np.float64,
            )
            arrivals = []
            now_s = 0.0
            while True:
                multiplier = float(multipliers[int(rng.integers(0, 2))])
                now_s += float(
                    rng.exponential(multiplier / self.arrival_rate_rps)
                )
                if now_s > self.horizon_s:
                    break
                arrivals.append(now_s)
            times = np.asarray(arrivals, dtype=np.float64)
            count = len(times)
        models = rng.choice(self.n_models, size=count, p=self.model_mix)
        return times, np.asarray(models, dtype=np.int64)

    def arrival_metadata(self):
        if self._openb_arrival_source is None:
            return {"mode": self.arrival_mode}
        metadata = self._openb_arrival_source.metadata(self.openb_split)
        metadata.update(
            {
                "sampling_mode": self.openb_sampling_mode,
                "hard_fraction": self.openb_hard_fraction,
                "hard_quantile": self.openb_hard_quantile,
            }
        )
        return metadata

    def _observation(self):
        pending_counts = np.asarray(
            [len(queue) for queue in self._pending], dtype=np.float64
        )
        observed_total_rate, observed_mix, observed_capacity = (
            self._observation_workload_parameters()
        )
        queue_scale = np.maximum(
            observed_total_rate * self.deadline_s * observed_mix, 1.0
        )
        pending_norm = np.clip(pending_counts / queue_scale, 0.0, 8.0)
        oldest_slack_raw = np.ones(self.n_models, dtype=np.float64)
        for model_index, queue in enumerate(self._pending):
            if queue:
                oldest = int(queue[0])
                oldest_slack_raw[model_index] = (
                    self._arrival_times[oldest]
                    + self.deadline_s
                    - self.now_s
                ) / self.deadline_s
        oldest_slack = np.clip(oldest_slack_raw, -4.0, 1.0)
        recent_scale = np.maximum(
            observed_total_rate * self.deadline_s * observed_mix, 1.0
        )
        recent_norm = np.clip(self._recent_arrivals / recent_scale, 0.0, 8.0)
        busy_remaining = np.where(
            self._busy,
            np.maximum(self._busy_until_s - self.now_s, 0.0) / self.deadline_s,
            0.0,
        )
        busy_remaining = np.clip(busy_remaining, 0.0, 8.0)
        busy_one_hot = np.zeros((self.N, self.n_models), dtype=np.float64)
        for server, model_index in enumerate(self._busy_model):
            if self._busy[server] and int(model_index) >= 0:
                busy_one_hot[server, int(model_index)] = 1.0
        global_features = np.asarray(
            [
                np.clip(self.now_s / max(self.horizon_s, 1e-9), 0.0, 2.0),
                observed_total_rate / max(observed_capacity, 1e-9),
                self.deadline_s / 0.05,
                float(self._busy.mean()),
            ],
            dtype=np.float64,
        )
        parts = [
            global_features,
            observed_mix,
            recent_norm,
            pending_norm,
            oldest_slack,
            busy_remaining,
            busy_one_hot.reshape(-1),
            self._profile_features.reshape(-1),
        ]
        overload_features = np.zeros(0, dtype=np.float64)
        if self.overload_observation:
            pending_ratio = pending_counts / queue_scale
            pending_log = np.log1p(pending_ratio)
            overdue_log = np.log1p(np.maximum(-oldest_slack_raw, 0.0))
            duration = max(
                min(self.now_s, self.recovery_window_s), self.deadline_s
            )
            arrivals = self._history_count(self._arrival_history)
            completions = self._history_count(self._completion_history)
            dispatches = self._history_count(self._dispatch_history)
            rate_scale = max(duration * self._aggregate_peak_rps, 1e-9)
            recent_rates = np.asarray(
                [
                    arrivals / rate_scale,
                    completions / rate_scale,
                    dispatches / rate_scale,
                    (arrivals - dispatches) / rate_scale,
                ],
                dtype=np.float64,
            )
            recent_rates[:3] = np.clip(recent_rates[:3], 0.0, 4.0)
            recent_rates[3] = np.clip(recent_rates[3], -4.0, 4.0)
            overload_features = np.concatenate(
                [pending_log, overdue_log, recent_rates]
            )
            parts.append(overload_features)
        causal_workload_features = np.zeros(0, dtype=np.float64)
        if self.causal_workload_features:
            count_norm = np.clip(
                np.log1p(self._scheduler_arrival_long_counts) / np.log1p(32.0),
                0.0,
                1.0,
            )
            per_model_capacity = max(
                self._scheduler_observation_capacity_rps / self.n_models,
                1e-9,
            )
            discrepancy_norm = np.clip(
                self._scheduler_arrival_rate_discrepancy / per_model_capacity,
                -2.0,
                2.0,
            )
            confidence = np.clip(
                self._scheduler_arrival_confidence, 0.0, 1.0
            )
            causal_workload_features = np.concatenate(
                [count_norm, discrepancy_norm, confidence]
            )
            parts.append(causal_workload_features)
        queue_distribution_features = np.zeros(0, dtype=np.float64)
        if self.queue_distribution_observation:
            rows = []
            for queue in self._pending:
                if queue:
                    request_ids = np.fromiter(queue, dtype=np.int64)
                    slacks = (
                        self._arrival_times[request_ids]
                        + self.deadline_s
                        - self.now_s
                    ) / self.deadline_s
                    quantiles = np.quantile(slacks, [0.25, 0.50, 0.75])
                    rows.append(
                        [
                            *np.clip(quantiles, -4.0, 1.0),
                            float(np.mean(slacks <= 0.25)),
                            float(np.mean(slacks <= 0.0)),
                        ]
                    )
                else:
                    rows.append([1.0, 1.0, 1.0, 0.0, 0.0])
            queue_distribution_features = np.asarray(
                rows, dtype=np.float64
            ).reshape(-1)
            parts.append(queue_distribution_features)
        vector = np.concatenate(parts).astype(np.float32)
        if len(vector) != self._base_observation_size():
            raise AssertionError("observation size mismatch")
        return {
            "vector": vector,
            "time_s": float(self.now_s),
            "pending_counts": pending_counts.astype(np.int64),
            "oldest_slack": oldest_slack.astype(np.float32),
            "busy_remaining": busy_remaining.astype(np.float32),
            "busy_model": self._busy_model.copy(),
            "overload_features": overload_features.astype(np.float32),
            "causal_workload_features": causal_workload_features.astype(
                np.float32
            ),
            "queue_distribution_features": queue_distribution_features.astype(
                np.float32
            ),
        }

    def _history_count(self, history):
        cutoff = self.now_s - self.recovery_window_s
        while history and history[0][0] < cutoff - 1e-12:
            history.popleft()
        return float(sum(count for _, count in history))

    def _physical_and_safe_masks(self):
        physical = np.zeros(self.action_dim, dtype=bool)
        safe = np.zeros(self.action_dim, dtype=bool)
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
            if self._busy[server] or len(queue) < minimum_requests:
                continue
            physical[index] = True
            oldest = int(queue[0])
            runtime_s = (
                self.catalog.table(self.models[model_index]).latency_ms(
                    self.providers[server], batch_size
                )
                / 1000.0
            )
            safe[index] = (
                self.now_s + runtime_s
                <= self._arrival_times[oldest] + self.deadline_s + 1e-12
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
                self.catalog.table(self.models[model_index]).latency_ms(
                    self.providers[self.action_space.decode(index)[1]],
                    self.action_space.decode(index)[2],
                )
                / 1000.0
                for index in model_actions
            )
            oldest = int(queue[0])
            latest_times.append(
                self._arrival_times[oldest] + self.deadline_s - best_runtime_s
            )
        return min(latest_times) if latest_times else None

    def action_mask(self):
        physical, safe = self._physical_and_safe_masks()
        mask = physical.copy()
        if self.predicted_safety_mask:
            for model_index in range(self.n_models):
                model_indices = np.asarray(
                    [
                        index
                        for index in range(1, self.action_dim)
                        if self.action_space.decode(index)[0] == model_index
                    ],
                    dtype=np.int64,
                )
                if safe[model_indices].any():
                    mask[model_indices] = safe[model_indices]
        latest_safe = self._latest_safe_wait_time(physical)
        mask[self.action_space.wait_index] = bool(
            self.optional_wait
            and physical.any()
            and latest_safe is not None
            and self.now_s < latest_safe - 1e-12
        )
        return mask

    def _process_events_at(self, target_s):
        self.now_s = float(target_s)
        while self._events and self._events[0][0] <= self.now_s + 1e-12:
            _, _, event_type, payload = heapq.heappop(self._events)
            if event_type == "arrival":
                request_id = int(payload)
                model_index = int(self._request_models[request_id])
                self._pending[model_index].append(request_id)
                self._recent_arrivals[model_index] += 1.0
                self._arrival_history.append((self.now_s, 1))
            elif event_type == "completion":
                server, model_index, request_ids = payload
                self._completion_times[request_ids] = self.now_s
                self._completion_history.append(
                    (self.now_s, int(len(request_ids)))
                )
                self._busy[server] = False
                self._busy_until_s[server] = self.now_s
                self._busy_model[server] = -1
            else:
                raise AssertionError("unknown event type")

    def _collect_new_deadline_violation_counts(self):
        if not len(self._arrival_times):
            return np.zeros(self.n_models, dtype=np.int64)
        deadlines = self._arrival_times + self.deadline_s
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

    def _collect_new_deadline_violations(self):
        """Backward-compatible scalar deadline-event counter."""

        return int(self._collect_new_deadline_violation_counts().sum())

    def _terminal(self):
        return bool(
            not self._events
            and not self._busy.any()
            and all(not queue for queue in self._pending)
        )

    def _advance_one_time_point(self):
        physical, _ = self._physical_and_safe_masks()
        urgency_s = self._latest_safe_wait_time(physical)
        candidates = []
        if self._events:
            candidates.append(float(self._events[0][0]))
        if urgency_s is not None and urgency_s > self.now_s + 1e-12:
            candidates.append(float(urgency_s))
        if not candidates:
            if self._terminal():
                self._done = True
                return
            raise RuntimeError("event simulator cannot make progress")
        self._process_events_at(min(candidates))

    def _emit_decision(self):
        observation = self._observation()
        mask = self.action_mask()
        if not mask.any():
            raise AssertionError("a decision must expose at least one action")
        self._recent_arrivals.fill(0.0)
        self._last_mask = mask.copy()
        return observation, mask

    def _advance_to_decision(self, force_time_advance=False):
        if force_time_advance:
            self._advance_one_time_point()
        while not self._done:
            mask = self.action_mask()
            if mask[1:].any():
                return self._emit_decision()
            if self._terminal():
                self._done = True
                break
            self._advance_one_time_point()
        return None, np.zeros(self.action_dim, dtype=bool)

    def step(self, action_index):
        if self._done:
            raise RuntimeError("step called after episode termination")
        action_index = int(action_index)
        current_mask = self.action_mask()
        if action_index < 0 or action_index >= self.action_dim or not current_mask[action_index]:
            raise ValueError("selected action is masked or out of range")
        start_s = float(self.now_s)
        reward = 0.0
        energy_cost = 0.0
        sla_cost = 0.0
        violation_count = 0
        request_count = 0
        request_count_by_model = np.zeros(self.n_models, dtype=np.int64)
        violation_count_by_model = np.zeros(self.n_models, dtype=np.int64)
        sla_cost_by_model = np.zeros(self.n_models, dtype=np.float64)
        model_index, server, batch_size = self.action_space.decode(action_index)

        if model_index is None:
            kind = "wait"
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
            model = self.models[model_index]
            provider = self.providers[server]
            table = self.catalog.table(model)
            profile_batch = table.effective_batch_size(provider, batch_size)
            runtime_s = table.runtime_latency_ms(
                provider, profile_batch, rng=self._latency_rng
            ) / 1000.0
            finish_s = float(self.now_s + runtime_s)
            self._request_servers[request_ids] = int(server)
            self._dispatch_times[request_ids] = self.now_s
            self._busy[server] = True
            self._busy_until_s[server] = finish_s
            self._busy_model[server] = int(model_index)
            self._push(
                finish_s,
                "completion",
                (int(server), int(model_index), request_ids),
            )
            energy_j = float(table.energy_j(provider, batch_size))
            self._energy_by_server[server] += energy_j
            self._total_padding += int(profile_batch - dispatch_size)
            latencies = finish_s - self._arrival_times[request_ids]
            violations = latencies > self.deadline_s
            excess = np.clip(
                np.maximum(latencies - self.deadline_s, 0.0) / self.deadline_s,
                0.0,
                self.excess_clip,
            )
            request_count = int(len(request_ids))
            request_count_by_model[int(model_index)] = request_count
            self._dispatch_history.append((self.now_s, request_count))
            predicted_violation_count = int(violations.sum())
            violation_count = predicted_violation_count
            violation_count_by_model[int(model_index)] = violation_count
            energy_cost = energy_j / self.energy_ref_per_request
            sla_cost = violation_count + self.excess_beta * float(excess.sum())
            sla_cost_by_model[int(model_index)] = sla_cost
            if self.sla_penalty_timing == "deadline_event":
                violation_count = 0
                sla_cost = 0.0
                violation_count_by_model.fill(0)
                sla_cost_by_model.fill(0.0)
            reward = self.reward_scale * (
                -energy_cost - self.sla_lambda * sla_cost
            )
            self._batches.append(
                {
                    "server": int(server),
                    "model": model,
                    "batch_size": int(dispatch_size),
                    "batch_cap": int(batch_size),
                    "profile_batch_size": int(profile_batch),
                    "dispatch_s": float(self.now_s),
                    "finish_s": finish_s,
                    "energy_j": energy_j,
                    "violation_count": predicted_violation_count,
                }
            )
            next_observation, next_mask = self._advance_to_decision(
                force_time_advance=False
            )

        if self.sla_penalty_timing == "deadline_event":
            violation_count_by_model = (
                self._collect_new_deadline_violation_counts()
            )
            violation_count = int(violation_count_by_model.sum())
            sla_cost_by_model = violation_count_by_model.astype(np.float64)
            sla_cost = float(violation_count)
            reward = self.reward_scale * (
                -energy_cost - self.sla_lambda * sla_cost
            )

        self._decision_trace.append(
            {
                "time_s": start_s,
                "action_index": action_index,
                "action_label": self.action_space.label(action_index),
                "kind": kind,
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
            }
        )
        info = dict(self._decision_trace[-1])
        return next_observation, next_mask, float(reward), bool(self._done), info

    def result(self, record_requests=False):
        if not self._done:
            raise RuntimeError("result is available only after episode completion")
        request_count = len(self._arrival_times)
        if request_count and (
            np.any(self._request_servers < 0)
            or np.any(~np.isfinite(self._dispatch_times))
            or np.any(~np.isfinite(self._completion_times))
        ):
            raise AssertionError("every request must be dispatched and completed")
        latencies = self._completion_times - self._arrival_times
        waits = self._dispatch_times - self._arrival_times
        violations = latencies > self.deadline_s
        requests_by_model = np.bincount(
            self._request_models, minlength=self.n_models
        ).astype(np.int64)
        violations_by_model = np.bincount(
            self._request_models[violations], minlength=self.n_models
        ).astype(np.int64)
        violation_rate_by_model = np.divide(
            violations_by_model,
            requests_by_model,
            out=np.zeros(self.n_models, dtype=np.float64),
            where=requests_by_model > 0,
        )
        batch_sizes = np.asarray(
            [row["batch_size"] for row in self._batches], dtype=np.float64
        )
        dispatch_actions = [
            row for row in self._decision_trace if row["kind"] == "dispatch"
        ]
        stability = self._stability_metrics()
        result = {
            "requests": int(request_count),
            "completed_requests": int(np.isfinite(self._completion_times).sum()),
            "mean_latency_s": float(latencies.mean()) if request_count else 0.0,
            "p50_latency_s": float(np.percentile(latencies, 50)) if request_count else 0.0,
            "p95_latency_s": float(np.percentile(latencies, 95)) if request_count else 0.0,
            "p99_latency_s": float(np.percentile(latencies, 99)) if request_count else 0.0,
            "max_latency_s": float(latencies.max()) if request_count else 0.0,
            "mean_wait_s": float(waits.mean()) if request_count else 0.0,
            "p95_wait_s": float(np.percentile(waits, 95)) if request_count else 0.0,
            "violation_count": int(violations.sum()),
            "violation_rate": float(violations.mean()) if request_count else 0.0,
            "requests_by_model": requests_by_model,
            "violation_count_by_model": violations_by_model,
            "violation_rate_by_model": violation_rate_by_model,
            "energy_total_j": float(self._energy_by_server.sum()),
            "energy_per_request_j": float(
                self._energy_by_server.sum() / max(request_count, 1)
            ),
            "throughput_rps": float(request_count / self.horizon_s),
            "mean_batch_size": float(batch_sizes.mean()) if len(batch_sizes) else 0.0,
            "p95_batch_size": float(np.percentile(batch_sizes, 95)) if len(batch_sizes) else 0.0,
            "batch_count": int(len(self._batches)),
            "total_padded_requests": int(self._total_padding),
            "routing_counts": self._routing_counts(),
            "energy_by_server_j": self._energy_by_server.copy(),
            "decision_count": int(len(self._decision_trace)),
            "wait_actions": int(
                sum(row["kind"] == "wait" for row in self._decision_trace)
            ),
            "dispatch_actions": int(len(dispatch_actions)),
            "distinct_dispatch_actions": int(
                len({row["action_index"] for row in dispatch_actions})
            ),
            "decision_trace": list(self._decision_trace),
            **stability,
        }
        if record_requests:
            result.update(
                {
                    "arrival_times_s": self._arrival_times.copy(),
                    "request_model_indices": self._request_models.copy(),
                    "request_servers": self._request_servers.copy(),
                    "dispatch_times_s": self._dispatch_times.copy(),
                    "completion_times_s": self._completion_times.copy(),
                    "request_latencies_s": latencies.copy(),
                    "batches": list(self._batches),
                }
            )
        return result

    def _stability_metrics(self):
        """Queue and drain diagnostics at the end of the arrival window."""
        request_count = len(self._arrival_times)
        if not request_count:
            return {
                "completed_by_horizon": 0,
                "service_throughput_rps": 0.0,
                "pending_at_horizon": 0,
                "inflight_at_horizon": 0,
                "backlog_at_horizon": 0,
                "backlog_fraction_at_horizon": 0.0,
                "drain_time_s": 0.0,
                "max_pending_requests": 0,
                "mean_pending_requests": 0.0,
                "queue_slope_rps": 0.0,
                "mean_device_utilization": 0.0,
                "max_device_utilization": 0.0,
                "device_utilization": np.zeros(self.N, dtype=np.float64),
            }

        horizon = float(self.horizon_s)
        completed_by_horizon = int(
            np.count_nonzero(self._completion_times <= horizon + 1e-12)
        )
        pending_at_horizon = int(
            np.count_nonzero(self._dispatch_times > horizon + 1e-12)
        )
        inflight_at_horizon = int(
            np.count_nonzero(
                (self._dispatch_times <= horizon + 1e-12)
                & (self._completion_times > horizon + 1e-12)
            )
        )
        backlog_at_horizon = int(
            np.count_nonzero(self._completion_times > horizon + 1e-12)
        )
        drain_time_s = max(float(self._completion_times.max()) - horizon, 0.0)

        grid = np.linspace(0.0, horizon, 101)
        pending_curve = np.asarray(
            [
                np.count_nonzero(
                    (self._arrival_times <= time_s + 1e-12)
                    & (self._dispatch_times > time_s + 1e-12)
                )
                for time_s in grid
            ],
            dtype=np.float64,
        )
        second_half = grid >= 0.5 * horizon
        queue_slope = float(
            np.polyfit(grid[second_half], pending_curve[second_half], 1)[0]
        )

        utilization = np.zeros(self.N, dtype=np.float64)
        for batch in self._batches:
            overlap = max(
                min(float(batch["finish_s"]), horizon)
                - min(max(float(batch["dispatch_s"]), 0.0), horizon),
                0.0,
            )
            utilization[int(batch["server"])] += overlap
        utilization = np.clip(utilization / max(horizon, 1e-12), 0.0, 1.0)
        return {
            "completed_by_horizon": completed_by_horizon,
            "service_throughput_rps": completed_by_horizon / max(horizon, 1e-12),
            "pending_at_horizon": pending_at_horizon,
            "inflight_at_horizon": inflight_at_horizon,
            "backlog_at_horizon": backlog_at_horizon,
            "backlog_fraction_at_horizon": backlog_at_horizon
            / max(request_count, 1),
            "drain_time_s": drain_time_s,
            "max_pending_requests": int(pending_curve.max()),
            "mean_pending_requests": float(np.trapz(pending_curve, grid) / horizon),
            "queue_slope_rps": queue_slope,
            "mean_device_utilization": float(utilization.mean()),
            "max_device_utilization": float(utilization.max()),
            "device_utilization": utilization,
        }

    def _routing_counts(self):
        counts = np.zeros((self.n_models, self.N), dtype=np.int64)
        for request_id, server in enumerate(self._request_servers):
            if int(server) >= 0:
                counts[int(self._request_models[request_id]), int(server)] += 1
        return counts
