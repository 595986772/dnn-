"""Request-level windowed micro-batch environment for P1 correctness tests."""

from pathlib import Path

import numpy as np
import pandas as pd

from chunk_assignment import round_to_chunks_largest_remainder


def normalize_available(action, available):
    action = np.clip(np.asarray(action, dtype=np.float64), 0.0, None)
    available = np.asarray(available, dtype=bool)
    if action.shape != available.shape:
        raise ValueError("action and availability shapes differ")
    action = action * available.astype(np.float64)
    total = float(action.sum())
    if total <= 1e-12:
        count = int(available.sum())
        if count <= 0:
            raise ValueError("at least one server must be available")
        action = available.astype(np.float64) / count
    else:
        action /= total
    return action


def largest_remainder_counts(action, total_requests, available=None):
    total_requests = int(total_requests)
    if total_requests < 0:
        raise ValueError("total_requests must be non-negative")
    action = np.asarray(action, dtype=np.float64)
    if available is None:
        available = np.ones_like(action, dtype=bool)
    probs = normalize_available(action, available)
    if total_requests == 0:
        return np.zeros_like(action, dtype=np.int64)
    quotas = probs * total_requests
    counts = np.floor(quotas).astype(np.int64)
    remaining = total_requests - int(counts.sum())
    if remaining > 0:
        fractions = quotas - counts
        order = np.lexsort((np.arange(len(action)), -fractions))
        counts[order[:remaining]] += 1
    counts[~np.asarray(available, dtype=bool)] = 0
    if int(counts.sum()) != total_requests:
        raise AssertionError("largest remainder violated request conservation")
    return counts


def split_largest_fit(count, max_batch_size, forced_batch_size=None):
    count = int(count)
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return []
    limit = int(forced_batch_size or max_batch_size)
    if limit <= 0:
        raise ValueError("batch size limit must be positive")
    full, remainder = divmod(count, limit)
    batches = [limit] * full
    if remainder:
        batches.append(remainder)
    return batches


def resolve_server_batch_caps(forced_batch_size, n_servers):
    """Expand one legacy batch cap or validate one cap per server."""
    n_servers = int(n_servers)
    if n_servers <= 0:
        raise ValueError("n_servers must be positive")
    if forced_batch_size is None:
        return [None] * n_servers
    if np.isscalar(forced_batch_size):
        cap = int(forced_batch_size)
        if cap <= 0:
            raise ValueError("batch size limit must be positive")
        return [cap] * n_servers
    values = np.asarray(forced_batch_size, dtype=np.int64).reshape(-1)
    if len(values) != n_servers:
        raise ValueError("one batch cap is required for each server")
    if np.any(values <= 0):
        raise ValueError("batch size limits must be positive")
    return [int(value) for value in values]


class LatencyProfileTable:
    LATENCY_COLUMNS = {
        "mean": "mean_ms",
        "p95": "p95_ms",
        "p99": "p99_ms",
        "stochastic": "mean_ms",
    }

    def __init__(
        self,
        csv_path,
        lookup_mode="linear",
        latency_mode="mean",
        model_name=None,
    ):
        lookup_mode = str(lookup_mode)
        if lookup_mode not in {"linear", "measured_bucket"}:
            raise ValueError("lookup_mode must be linear or measured_bucket")
        latency_mode = str(latency_mode).lower()
        if latency_mode not in self.LATENCY_COLUMNS:
            raise ValueError("latency_mode must be mean, p95, p99, or stochastic")
        self.lookup_mode = lookup_mode
        self.latency_mode = latency_mode
        frame = pd.read_csv(csv_path)
        if "model" in frame.columns:
            available_models = sorted(frame["model"].dropna().astype(str).unique().tolist())
            if model_name is None:
                if len(available_models) > 1:
                    raise ValueError(
                        "profile contains multiple models; model_name must be selected explicitly"
                    )
                model_name = available_models[0] if available_models else None
            elif str(model_name) not in available_models:
                raise ValueError("unknown model profile: %s" % model_name)
            if model_name is not None:
                frame = frame[frame["model"].astype(str) == str(model_name)].copy()
        elif model_name is not None:
            raise ValueError("model_name requires a profile CSV with a model column")
        self.model_name = None if model_name is None else str(model_name)
        frame = frame[(frame["status"] == "ok") & frame["mean_ms"].notna()].copy()
        if frame.empty:
            raise ValueError("profile contains no valid rows")
        latency_column = self.LATENCY_COLUMNS[latency_mode]
        if latency_column not in frame.columns or frame[latency_column].isna().any():
            raise ValueError("profile does not contain complete %s values" % latency_column)
        if latency_mode == "stochastic":
            required = {"p50_ms", "p95_ms", "p99_ms"}
            missing = sorted(required - set(frame.columns))
            if missing or frame[list(required)].isna().any().any():
                raise ValueError(
                    "stochastic latency requires complete p50_ms, p95_ms, and p99_ms values"
                )
        self.providers = sorted(
            frame["provider"].unique().tolist(),
            key=lambda value: int(str(value).replace("cpu", "")) if str(value).startswith("cpu") else 10_000,
        )
        self.curves = {}
        self.quantile_curves = {}
        self.energy_curves = {}
        for provider in self.providers:
            rows = frame[frame["provider"] == provider].sort_values("batch_size")
            self.curves[provider] = (
                rows["batch_size"].to_numpy(dtype=np.int64),
                rows[latency_column].to_numpy(dtype=np.float64),
            )
            if latency_mode == "stochastic":
                self.quantile_curves[provider] = {
                    column: rows[column].to_numpy(dtype=np.float64)
                    for column in ("p50_ms", "p95_ms", "p99_ms")
                }
            if "energy_per_batch_j" in rows.columns and rows["energy_per_batch_j"].notna().all():
                self.energy_curves[provider] = (
                    rows["batch_size"].to_numpy(dtype=np.int64),
                    rows["energy_per_batch_j"].to_numpy(dtype=np.float64),
                )
        self.max_batch_size = min(int(self.curves[p][0].max()) for p in self.providers)

    def effective_batch_size(self, provider, batch_size):
        batch_size = int(batch_size)
        if batch_size <= 0:
            return 0
        x, _ = self.curves[str(provider)]
        if batch_size > int(x.max()):
            raise ValueError("effective_batch_size only accepts one physical batch")
        if self.lookup_mode == "linear" and batch_size >= int(x.min()):
            return batch_size
        index = int(np.searchsorted(x, batch_size, side="left"))
        index = min(index, len(x) - 1)
        return int(x[index])

    def execution_batches(self, provider, count, forced_batch_size=None):
        requested = split_largest_fit(count, self.max_batch_size, forced_batch_size)
        effective = [self.effective_batch_size(provider, batch) for batch in requested]
        return requested, effective

    def latency_ms(self, provider, batch_size):
        batch_size = int(batch_size)
        if batch_size <= 0:
            return 0.0
        x, y = self.curves[str(provider)]
        if batch_size > int(x.max()):
            return sum(self.latency_ms(provider, b) for b in split_largest_fit(batch_size, int(x.max())))
        if self.lookup_mode == "measured_bucket" or batch_size < int(x.min()):
            batch_size = self.effective_batch_size(provider, batch_size)
            return float(y[np.where(x == batch_size)[0][0]])
        return float(np.interp(batch_size, x, y))

    def _profile_value(self, provider, batch_size, column):
        batch_size = int(batch_size)
        if batch_size <= 0:
            return 0.0
        x, _ = self.curves[str(provider)]
        if batch_size > int(x.max()):
            return sum(
                self._profile_value(provider, part, column)
                for part in split_largest_fit(batch_size, int(x.max()))
            )
        values = self.quantile_curves[str(provider)][column]
        if self.lookup_mode == "measured_bucket" or batch_size < int(x.min()):
            effective = self.effective_batch_size(provider, batch_size)
            return float(values[np.where(x == effective)[0][0]])
        return float(np.interp(batch_size, x, values))

    def runtime_latency_ms(self, provider, batch_size, rng=None):
        """Return one runtime latency draw while keeping profile lookup reproducible."""
        if self.latency_mode != "stochastic":
            return self.latency_ms(provider, batch_size)
        if rng is None:
            raise ValueError("stochastic latency requires an explicit numpy RNG")
        p50 = max(self._profile_value(provider, batch_size, "p50_ms"), 1e-9)
        p95 = max(self._profile_value(provider, batch_size, "p95_ms"), p50)
        p99 = max(self._profile_value(provider, batch_size, "p99_ms"), p95)
        sigma95 = max(np.log(p95 / p50) / 1.6448536269514722, 0.0)
        sigma99 = max(np.log(p99 / p50) / 2.3263478740408408, 0.0)
        sigma = 0.5 * (sigma95 + sigma99)
        if sigma <= 1e-12:
            return float(p50)
        return float(rng.lognormal(mean=np.log(p50), sigma=sigma))

    def service_ms(self, provider, count, forced_batch_size=None):
        batches = split_largest_fit(count, self.max_batch_size, forced_batch_size)
        return float(sum(self.latency_ms(provider, batch) for batch in batches))

    def energy_j(self, provider, batch_size):
        batch_size = int(batch_size)
        if batch_size <= 0:
            return 0.0
        if str(provider) not in self.energy_curves:
            raise ValueError("energy profile is unavailable for provider=%s" % provider)
        x, y = self.energy_curves[str(provider)]
        if batch_size > int(x.max()):
            return sum(self.energy_j(provider, b) for b in split_largest_fit(batch_size, int(x.max())))
        if self.lookup_mode == "measured_bucket" or batch_size < int(x.min()):
            batch_size = self.effective_batch_size(provider, batch_size)
            return float(y[np.where(x == batch_size)[0][0]])
        return float(np.interp(batch_size, x, y))

    def service_energy_j(self, provider, count, forced_batch_size=None):
        batches = split_largest_fit(count, self.max_batch_size, forced_batch_size)
        return float(sum(self.energy_j(provider, batch) for batch in batches))

    def best_throughput(self, provider):
        x, y = self.curves[str(provider)]
        return float(np.max(1000.0 * x / y))


class MicroBatchEdgeEnv:
    """Fixed-window dispatcher with request arrivals and time-valued queues."""

    def __init__(
        self,
        profile_csv,
        arrival_lambda=35.0,
        window_s=1.0,
        deadline_s=3.0,
        horizon=100,
        seed=0,
        server_profiles=None,
        profile_lookup_mode="linear",
        profile_latency_mode="mean",
        record_request_latencies=False,
    ):
        self.profile_csv = str(Path(profile_csv))
        self.profile_lookup_mode = str(profile_lookup_mode)
        self.profile_latency_mode = str(profile_latency_mode).lower()
        self.profile = LatencyProfileTable(
            profile_csv,
            lookup_mode=self.profile_lookup_mode,
            latency_mode=self.profile_latency_mode,
        )
        self.providers = list(server_profiles or self.profile.providers)
        unknown = sorted(set(self.providers) - set(self.profile.providers))
        if unknown:
            raise ValueError("unknown server profiles: %s" % unknown)
        self.N = len(self.providers)
        self.arrival_lambda = float(arrival_lambda)
        self.window_s = float(window_s)
        self.deadline_s = float(deadline_s)
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.record_request_latencies = bool(record_request_latencies)
        self.w = 0.5
        self.available = np.ones(self.N, dtype=bool)
        self.rng = np.random.default_rng(self.seed)
        self.latency_rng = np.random.default_rng(self.seed + 1_000_003)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.latency_rng = np.random.default_rng(self.seed + 1_000_003)
        self.time_s = 0.0
        self.step_index = 0
        self.server_available_s = np.zeros(self.N, dtype=np.float64)
        self.episode_infos = []
        self._draw_arrivals()
        return self.observation()

    def _draw_arrivals(self):
        self.M_t = int(self.rng.poisson(self.arrival_lambda))
        offsets = self.rng.uniform(0.0, self.window_s, size=self.M_t)
        self.arrival_times_s = self.time_s + np.sort(offsets)

    def observation(self):
        dispatch_s = self.time_s + self.window_s
        backlog_s = np.maximum(self.server_available_s - dispatch_s, 0.0)
        rows = []
        max_rate = max(self.profile.best_throughput(p) for p in self.providers)
        energy_available = all(provider in self.profile.energy_curves for provider in self.providers)
        max_b1_energy = (
            max(self.profile.energy_j(provider, 1) for provider in self.providers)
            if energy_available
            else 1.0
        )
        for i, provider in enumerate(self.providers):
            rows.append(
                [
                    self.profile.best_throughput(provider) / max_rate,
                    (
                        self.profile.energy_j(provider, 1) / max(max_b1_energy, 1e-9)
                        if energy_available
                        else 0.0
                    ),
                    backlog_s[i] / max(self.deadline_s, 1e-9),
                    1.0,
                    1.0,
                    self.profile.latency_ms(provider, 1) / (1000.0 * self.deadline_s),
                    self.profile.latency_ms(provider, 32) / (1000.0 * self.deadline_s),
                    (
                        self.profile.energy_j(provider, 1) / max(max_b1_energy, 1e-9)
                        if energy_available
                        else 0.0
                    ),
                    self.profile.latency_ms(provider, 4) / (1000.0 * self.deadline_s),
                    self.profile.latency_ms(provider, 8) / (1000.0 * self.deadline_s),
                    self.profile.latency_ms(provider, 16) / (1000.0 * self.deadline_s),
                    (
                        self.profile.energy_j(provider, 4) / max(4.0 * max_b1_energy, 1e-9)
                        if energy_available
                        else 0.0
                    ),
                    (
                        self.profile.energy_j(provider, 8) / max(8.0 * max_b1_energy, 1e-9)
                        if energy_available
                        else 0.0
                    ),
                    (
                        self.profile.energy_j(provider, 16) / max(16.0 * max_b1_energy, 1e-9)
                        if energy_available
                        else 0.0
                    ),
                    float(self.M_t) / max(
                        float(getattr(self, "request_scale", 2.0 * self.arrival_lambda * self.window_s)),
                        1.0,
                    ),
                    float(self.arrival_lambda * self.window_s) / max(
                        float(getattr(self, "request_scale", 2.0 * self.arrival_lambda * self.window_s)),
                        1.0,
                    ),
                    float(self.available[i]),
                ]
            )
        return {
            "servers": np.asarray(rows, dtype=np.float32),
            "omega": float(self.w),
            "M_t": int(self.M_t),
            "arrival_pressure": float(self.arrival_lambda * self.window_s),
            "deadline_s": float(self.deadline_s),
            "window_s": float(self.window_s),
        }

    def step(self, action, forced_batch_size=None):
        dispatch_s = self.time_s + self.window_s
        a_policy = normalize_available(action, self.available)
        server_batch_caps = resolve_server_batch_caps(forced_batch_size, self.N)
        counts, a_exec, rounding_l1 = round_to_chunks_largest_remainder(a_policy, self.M_t)
        if counts is None:
            counts = np.zeros(self.N, dtype=np.int64)
            a_exec = a_policy.copy()
        counts = np.asarray(counts, dtype=np.int64)
        counts[~self.available] = 0
        permutation = self.rng.permutation(self.M_t)
        cursor = 0
        latencies = np.zeros(self.M_t, dtype=np.float64)
        server_batches = []
        server_profile_batches = []
        padded_requests = np.zeros(self.N, dtype=np.int64)
        server_energy_j = np.zeros(self.N, dtype=np.float64)
        queue_before_s = np.maximum(self.server_available_s - dispatch_s, 0.0)
        for i, (provider, count) in enumerate(zip(self.providers, counts)):
            assigned = permutation[cursor : cursor + int(count)]
            cursor += int(count)
            assigned = assigned[np.argsort(self.arrival_times_s[assigned])] if len(assigned) else assigned
            batches, profile_batches = self.profile.execution_batches(
                provider, count, server_batch_caps[i]
            )
            server_batches.append(batches)
            server_profile_batches.append(profile_batches)
            padded_requests[i] = int(sum(profile_batches) - sum(batches))
            if provider in self.profile.energy_curves:
                server_energy_j[i] = sum(self.profile.energy_j(provider, batch) for batch in batches)
            start_s = max(float(self.server_available_s[i]), dispatch_s)
            local_cursor = 0
            for batch_size, profile_batch_size in zip(batches, profile_batches):
                runtime_ms = self.profile.runtime_latency_ms(
                    provider, profile_batch_size, rng=self.latency_rng
                )
                finish_s = start_s + runtime_ms / 1000.0
                ids = assigned[local_cursor : local_cursor + batch_size]
                latencies[ids] = finish_s - self.arrival_times_s[ids]
                local_cursor += batch_size
                start_s = finish_s
            if count > 0:
                self.server_available_s[i] = start_s
        if cursor != self.M_t:
            raise AssertionError("not every request was assigned exactly once")

        self.time_s = dispatch_s
        self.step_index += 1
        queue_after_s = np.maximum(self.server_available_s - self.time_s, 0.0)
        violations = latencies > self.deadline_s if self.M_t else np.zeros(0, dtype=bool)
        info = {
            "M_t": int(self.M_t),
            "counts": counts.copy(),
            "server_batches": server_batches,
            "server_profile_batches": server_profile_batches,
            "padded_requests": padded_requests.copy(),
            "total_padded_requests": int(padded_requests.sum()),
            "assigned_requests": int(counts.sum()),
            "queue_before_s": queue_before_s.copy(),
            "queue_after_s": queue_after_s.copy(),
            "mean_latency_s": float(latencies.mean()) if self.M_t else 0.0,
            "p95_latency_s": float(np.percentile(latencies, 95)) if self.M_t else 0.0,
            "p99_latency_s": float(np.percentile(latencies, 99)) if self.M_t else 0.0,
            "makespan_s": float(latencies.max()) if self.M_t else 0.0,
            "violation_rate": float(violations.mean()) if self.M_t else 0.0,
            "delay": float(latencies.max()) if self.M_t else 0.0,
            "energy": float(server_energy_j.sum()),
            "deadline": float(self.deadline_s),
            "a_policy": np.asarray(a_policy, dtype=np.float32),
            "a_exec": np.asarray(a_exec, dtype=np.float32),
            "n_chunks": counts.copy(),
            "chunk_counts": counts.copy(),
            "rounding_l1": float(rounding_l1),
            "K_active": int((counts > 0).sum()),
            "K_exec": int((counts > 0).sum()),
            "max_frac_exec": float(np.max(a_exec)) if len(a_exec) else 0.0,
            "server_energy_j": server_energy_j.copy(),
            "completed_requests": int(self.M_t),
            "forced_batch_size": (
                int(forced_batch_size)
                if forced_batch_size is not None and np.isscalar(forced_batch_size)
                else 0
            ),
            "server_batch_caps": np.asarray(
                [0 if cap is None else cap for cap in server_batch_caps], dtype=np.int64
            ),
            "profile_latency_mode": self.profile_latency_mode,
        }
        if self.record_request_latencies:
            info["request_latencies_s"] = latencies.copy()
        self.episode_infos.append(info)
        done = self.step_index >= self.horizon
        if not done:
            self._draw_arrivals()
        return self.observation(), done, info

    def episode_sla_summary(self):
        if not self.episode_infos:
            return {
                "delay": 0.0,
                "energy": 0.0,
                "violation_rate": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max_delay": 0.0,
                "mean_K": 0.0,
                "mean_M_t": 0.0,
            }
        weights = np.asarray([max(1, row["M_t"]) for row in self.episode_infos], dtype=np.float64)
        weighted_viol = np.average(
            [row["violation_rate"] for row in self.episode_infos], weights=weights
        )
        mean_delay = float(np.mean([row["delay"] for row in self.episode_infos]))
        mean_energy = float(np.mean([row["energy"] for row in self.episode_infos]))
        p95_delay = float(np.mean([row["p95_latency_s"] for row in self.episode_infos]))
        p99_delay = float(np.mean([row["p99_latency_s"] for row in self.episode_infos]))
        max_delay = float(np.max([row["delay"] for row in self.episode_infos]))
        return {
            "delay": mean_delay,
            "energy": mean_energy,
            "mean_delay": mean_delay,
            "mean_energy": mean_energy,
            "violation_rate": float(weighted_viol),
            "p95": p95_delay,
            "p99": p99_delay,
            "p95_delay": p95_delay,
            "p99_delay": p99_delay,
            "max_delay": max_delay,
            "mean_q": float(np.mean([np.mean(row["queue_after_s"]) for row in self.episode_infos])),
            "max_q": float(np.max([np.max(row["queue_after_s"]) for row in self.episode_infos])),
            "post_burst_recovery": 0.0,
            "mean_K": float(np.mean([row["K_active"] for row in self.episode_infos])),
            "mean_M_t": float(np.mean([row["M_t"] for row in self.episode_infos])),
        }


class MicroBatchRLEnv(MicroBatchEdgeEnv):
    """Compatibility wrapper for the existing grid-best training/eval loops."""

    def __init__(
        self,
        profile_csv,
        server_profiles,
        arrival_loads=(15.0, 35.0, 60.0),
        window_s=1.0,
        deadline_s=3.0,
        horizon=30,
        seed=0,
        scenario="microbatch_pilot",
        profile_lookup_mode="linear",
        profile_latency_mode="mean",
        record_request_latencies=False,
    ):
        self.arrival_loads = tuple(float(value) for value in arrival_loads)
        if not self.arrival_loads:
            raise ValueError("arrival_loads cannot be empty")
        self.base_seed = int(seed)
        self.episode_counter = 0
        self.scenario = str(scenario)
        self.request_scale = max(2.0 * max(self.arrival_loads) * float(window_s), 1.0)
        super().__init__(
            profile_csv=profile_csv,
            arrival_lambda=self.arrival_loads[0],
            window_s=window_s,
            deadline_s=deadline_s,
            horizon=horizon,
            seed=seed,
            server_profiles=server_profiles,
            profile_lookup_mode=profile_lookup_mode,
            profile_latency_mode=profile_latency_mode,
            record_request_latencies=record_request_latencies,
        )
        self.deadline = self.deadline_s
        self.delay_ref = self.deadline_s
        max_b1 = max(self.profile.energy_j(provider, 1) for provider in self.providers)
        self.energy_ref = max(self.arrival_loads) * self.window_s * max_b1
        self.workload_mode = "microbatch_profile"

    def reset(self, seed=None):
        if seed is None:
            episode_seed = self.base_seed + self.episode_counter
            load_index = self.episode_counter % len(self.arrival_loads)
            self.episode_counter += 1
        else:
            episode_seed = int(seed)
            load_index = int(seed) % len(self.arrival_loads)
        self.arrival_lambda = self.arrival_loads[load_index]
        return super().reset(seed=episode_seed)

    def step(self, action, forced_batch_size=None):
        obs, done, info = super().step(action, forced_batch_size=forced_batch_size)
        info["scenario"] = self.scenario
        info["in_burst"] = 0.0
        return obs, 0.0, done, info


def uniform_policy(env):
    return env.available.astype(np.float64) / max(int(env.available.sum()), 1)


def fastest_policy(env):
    projected = []
    dispatch_s = env.time_s + env.window_s
    for i, provider in enumerate(env.providers):
        backlog = max(float(env.server_available_s[i]) - dispatch_s, 0.0)
        projected.append(backlog + env.profile.service_ms(provider, env.M_t) / 1000.0)
    action = np.zeros(env.N, dtype=np.float64)
    action[int(np.argmin(projected))] = 1.0
    return action


def min_queue_policy(env):
    dispatch_s = env.time_s + env.window_s
    backlog = np.maximum(env.server_available_s - dispatch_s, 0.0)
    backlog[~env.available] = np.inf
    action = np.zeros(env.N, dtype=np.float64)
    action[int(np.argmin(backlog))] = 1.0
    return action


def no_batch_jsq_policy(env):
    dispatch_s = env.time_s + env.window_s
    projected = np.maximum(env.server_available_s - dispatch_s, 0.0).copy()
    counts = np.zeros(env.N, dtype=np.int64)
    for _ in range(env.M_t):
        scores = projected.copy()
        scores[~env.available] = np.inf
        idx = int(np.argmin(scores))
        counts[idx] += 1
        projected[idx] += env.profile.latency_ms(env.providers[idx], 1) / 1000.0
    return counts.astype(np.float64) / max(env.M_t, 1)
