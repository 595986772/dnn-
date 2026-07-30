"""Event-driven request routing and dynamic batching for collaborative inference."""

from collections import deque
import heapq
import itertools

import numpy as np
import pandas as pd

from microbatch_env import LatencyProfileTable


class MultiModelProfileCatalog:
    """One measured latency-energy table per DNN model."""

    def __init__(self, csv_path, lookup_mode="measured_bucket", latency_mode="mean"):
        frame = pd.read_csv(csv_path)
        valid = frame[(frame["status"] == "ok") & frame["mean_ms"].notna()].copy()
        if valid.empty:
            raise ValueError("profile contains no valid rows")
        if "model" in valid.columns:
            self.models = sorted(valid["model"].dropna().astype(str).unique().tolist())
        else:
            self.models = ["default"]
        self.tables = {}
        for model in self.models:
            selected = model if "model" in valid.columns else None
            self.tables[model] = LatencyProfileTable(
                csv_path,
                lookup_mode=lookup_mode,
                latency_mode=latency_mode,
                model_name=selected,
            )
        provider_sets = [set(table.providers) for table in self.tables.values()]
        self.providers = sorted(set.intersection(*provider_sets))
        if not self.providers:
            raise ValueError("models do not share any server provider")
        self.max_batch_size = min(table.max_batch_size for table in self.tables.values())

    def table(self, model):
        return self.tables[str(model)]

    def mixed_provider_peak_rps(self, provider, model_mix=None):
        """Approximate one provider's capacity under a fixed DNN request mix."""
        provider = str(provider)
        if provider not in self.providers:
            raise ValueError("unknown provider: %s" % provider)
        mix = _normalize_model_mix(self.models, model_mix)
        service_seconds = 0.0
        for probability, model in zip(mix, self.models):
            rate = self.table(model).best_throughput(provider)
            service_seconds += float(probability) / max(float(rate), 1e-12)
        return float(1.0 / max(service_seconds, 1e-12))

    def mixed_provider_rates(self, providers=None, model_mix=None):
        providers = list(providers or self.providers)
        return np.asarray(
            [self.mixed_provider_peak_rps(provider, model_mix) for provider in providers],
            dtype=np.float64,
        )

    def aggregate_peak_rps(self, providers=None, model=None, model_mix=None):
        providers = list(providers or self.providers)
        if model is not None and model_mix is not None:
            raise ValueError("select either model or model_mix, not both")
        if model is not None:
            model = str(model)
            return float(
                sum(
                    self.table(model).best_throughput(provider)
                    for provider in providers
                )
            )
        return float(self.mixed_provider_rates(providers, model_mix).sum())


def _normalize_model_mix(models, model_mix):
    if model_mix is None:
        values = np.ones(len(models), dtype=np.float64)
    elif isinstance(model_mix, dict):
        values = np.asarray([float(model_mix.get(model, 0.0)) for model in models])
    else:
        values = np.asarray(model_mix, dtype=np.float64).reshape(-1)
    if len(values) != len(models) or np.any(values < 0.0) or float(values.sum()) <= 0.0:
        raise ValueError("model_mix must provide non-negative mass for every model")
    return values / values.sum()


def _expand_matrix(value, n_models, n_servers, name, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = np.full((n_models, n_servers), array.item(), dtype=dtype)
    elif array.ndim == 1 and len(array) == n_servers:
        array = np.tile(array.reshape(1, -1), (n_models, 1))
    elif array.shape != (n_models, n_servers):
        raise ValueError("%s must be scalar, per-server, or model-by-server" % name)
    return array


def normalize_routing(routing, n_models, n_servers):
    routing = _expand_matrix(routing, n_models, n_servers, "routing")
    if np.any(routing < 0.0):
        raise ValueError("routing weights must be non-negative")
    totals = routing.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("each model needs at least one positive routing weight")
    return routing / totals


class WeightedDeficitRouter:
    """Deterministic weighted routing without future-arrival information."""

    def __init__(self, routing):
        self.routing = np.asarray(routing, dtype=np.float64)
        self.assigned = np.zeros_like(self.routing, dtype=np.int64)
        self.totals = np.zeros(self.routing.shape[0], dtype=np.int64)

    def choose(self, model_index):
        model_index = int(model_index)
        weights = self.routing[model_index]
        target = weights * float(self.totals[model_index] + 1)
        deficit = target - self.assigned[model_index]
        deficit[weights <= 0.0] = -np.inf
        server = int(np.argmax(deficit))
        self.assigned[model_index, server] += 1
        self.totals[model_index] += 1
        return server


class OnlineBatchEdgeEnv:
    """Continuous-time simulator with batch-full, timeout, and SLA-urgent dispatch."""

    MODES = {"timeout", "deadline_aware"}

    def __init__(
        self,
        profile_csv,
        server_profiles,
        arrival_rate_rps,
        horizon_s=20.0,
        deadline_s=0.10,
        model_mix=None,
        seed=0,
        profile_lookup_mode="measured_bucket",
        profile_latency_mode="mean",
        catalog=None,
    ):
        self.catalog = catalog or MultiModelProfileCatalog(
            profile_csv,
            lookup_mode=profile_lookup_mode,
            latency_mode=profile_latency_mode,
        )
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
        if self.arrival_rate_rps < 0.0 or self.horizon_s <= 0.0 or self.deadline_s <= 0.0:
            raise ValueError("arrival rate, horizon, and deadline must be valid")

    def _arrival_trace(self, rng):
        count = int(rng.poisson(self.arrival_rate_rps * self.horizon_s))
        times = np.sort(rng.uniform(0.0, self.horizon_s, size=count))
        model_indices = rng.choice(self.n_models, size=count, p=self.model_mix)
        return times, np.asarray(model_indices, dtype=np.int64)

    def run(
        self,
        routing,
        batch_caps,
        max_wait_s,
        dispatch_mode="deadline_aware",
        record_requests=False,
    ):
        if dispatch_mode not in self.MODES:
            raise ValueError("dispatch_mode must be timeout or deadline_aware")
        routing = normalize_routing(routing, self.n_models, self.N)
        batch_caps = _expand_matrix(
            batch_caps, self.n_models, self.N, "batch_caps", dtype=np.int64
        )
        max_wait_s = _expand_matrix(max_wait_s, self.n_models, self.N, "max_wait_s")
        if np.any(batch_caps <= 0) or np.any(batch_caps > self.catalog.max_batch_size):
            raise ValueError("batch caps must be within the measured profile range")
        if np.any(max_wait_s < 0.0):
            raise ValueError("max_wait_s must be non-negative")

        rng = np.random.default_rng(self.seed)
        latency_rng = np.random.default_rng(self.seed + 1_000_003)
        arrival_times, request_models = self._arrival_trace(rng)
        request_count = len(arrival_times)
        request_servers = np.full(request_count, -1, dtype=np.int64)
        completion_times = np.full(request_count, np.nan, dtype=np.float64)
        dispatch_times = np.full(request_count, np.nan, dtype=np.float64)
        router = WeightedDeficitRouter(routing)
        pending = [
            [deque() for _ in range(self.n_models)]
            for _ in range(self.N)
        ]
        busy = np.zeros(self.N, dtype=bool)
        wake_versions = np.zeros(self.N, dtype=np.int64)
        event_counter = itertools.count()
        events = []
        energy_by_server = np.zeros(self.N, dtype=np.float64)
        dispatch_reasons = {"full": 0, "timeout": 0, "deadline": 0}
        batches = []
        total_padding = 0

        def push(time_s, event_type, payload):
            heapq.heappush(
                events,
                (float(time_s), next(event_counter), str(event_type), payload),
            )

        for request_id, arrival_s in enumerate(arrival_times):
            push(arrival_s, "arrival", int(request_id))

        def queue_trigger(server, model_index, now_s):
            queue = pending[server][model_index]
            if not queue:
                return None
            cap = int(batch_caps[model_index, server])
            if len(queue) >= cap:
                return float(now_s), "full"
            oldest = int(queue[0])
            timeout_s = float(arrival_times[oldest] + max_wait_s[model_index, server])
            candidates = [(timeout_s, "timeout")]
            if dispatch_mode == "deadline_aware":
                size = min(len(queue), cap)
                table = self.catalog.table(self.models[model_index])
                service_s = table.latency_ms(self.providers[server], size) / 1000.0
                urgent_s = float(arrival_times[oldest] + self.deadline_s - service_s)
                candidates.append((urgent_s, "deadline"))
            trigger_s, reason = min(candidates, key=lambda item: (item[0], item[1]))
            return max(float(now_s), trigger_s), reason

        def dispatch(server, model_index, now_s, reason):
            nonlocal total_padding
            queue = pending[server][model_index]
            cap = int(batch_caps[model_index, server])
            size = min(len(queue), cap)
            request_ids = [int(queue.popleft()) for _ in range(size)]
            model = self.models[model_index]
            provider = self.providers[server]
            table = self.catalog.table(model)
            profile_batch = table.effective_batch_size(provider, size)
            runtime_s = table.runtime_latency_ms(
                provider, profile_batch, rng=latency_rng
            ) / 1000.0
            finish_s = float(now_s + runtime_s)
            dispatch_times[request_ids] = now_s
            energy_j = table.energy_j(provider, size)
            energy_by_server[server] += energy_j
            padding = int(profile_batch - size)
            total_padding += padding
            dispatch_reasons[reason] += 1
            batches.append(
                {
                    "server": int(server),
                    "model": model,
                    "batch_size": int(size),
                    "profile_batch_size": int(profile_batch),
                    "dispatch_s": float(now_s),
                    "finish_s": finish_s,
                    "reason": reason,
                    "energy_j": float(energy_j),
                }
            )
            busy[server] = True
            wake_versions[server] += 1
            push(finish_s, "completion", (int(server), request_ids))

        def plan_server(server, now_s):
            if busy[server]:
                return
            candidates = []
            for model_index in range(self.n_models):
                trigger = queue_trigger(server, model_index, now_s)
                if trigger is not None:
                    candidates.append((trigger[0], trigger[1], model_index))
            if not candidates:
                wake_versions[server] += 1
                return
            trigger_s, reason, model_index = min(
                candidates, key=lambda item: (item[0], item[1], item[2])
            )
            if trigger_s <= now_s + 1e-12:
                dispatch(server, model_index, now_s, reason)
                return
            wake_versions[server] += 1
            version = int(wake_versions[server])
            push(trigger_s, "wake", (int(server), version))

        while events:
            now_s, _, event_type, payload = heapq.heappop(events)
            if event_type == "arrival":
                request_id = int(payload)
                model_index = int(request_models[request_id])
                server = router.choose(model_index)
                request_servers[request_id] = server
                pending[server][model_index].append(request_id)
                plan_server(server, now_s)
            elif event_type == "completion":
                server, request_ids = payload
                completion_times[request_ids] = now_s
                busy[server] = False
                plan_server(server, now_s)
            elif event_type == "wake":
                server, version = payload
                if version == wake_versions[server] and not busy[server]:
                    plan_server(server, now_s)
            else:
                raise AssertionError("unknown event type")

        if request_count and (
            np.any(request_servers < 0)
            or np.any(~np.isfinite(completion_times))
            or np.any(~np.isfinite(dispatch_times))
        ):
            raise AssertionError("every request must be routed, dispatched, and completed")
        latencies = completion_times - arrival_times
        waits = dispatch_times - arrival_times
        violations = latencies > self.deadline_s
        batch_sizes = np.asarray([row["batch_size"] for row in batches], dtype=np.float64)
        result = {
            "requests": int(request_count),
            "completed_requests": int(np.isfinite(completion_times).sum()),
            "mean_latency_s": float(latencies.mean()) if request_count else 0.0,
            "p50_latency_s": float(np.percentile(latencies, 50)) if request_count else 0.0,
            "p95_latency_s": float(np.percentile(latencies, 95)) if request_count else 0.0,
            "p99_latency_s": float(np.percentile(latencies, 99)) if request_count else 0.0,
            "max_latency_s": float(latencies.max()) if request_count else 0.0,
            "mean_wait_s": float(waits.mean()) if request_count else 0.0,
            "p95_wait_s": float(np.percentile(waits, 95)) if request_count else 0.0,
            "violation_count": int(violations.sum()),
            "violation_rate": float(violations.mean()) if request_count else 0.0,
            "energy_total_j": float(energy_by_server.sum()),
            "energy_per_request_j": float(energy_by_server.sum() / max(request_count, 1)),
            "throughput_rps": float(request_count / self.horizon_s),
            "mean_batch_size": float(batch_sizes.mean()) if len(batch_sizes) else 0.0,
            "p95_batch_size": float(np.percentile(batch_sizes, 95)) if len(batch_sizes) else 0.0,
            "batch_count": int(len(batches)),
            "total_padded_requests": int(total_padding),
            "full_dispatches": int(dispatch_reasons["full"]),
            "timeout_dispatches": int(dispatch_reasons["timeout"]),
            "deadline_dispatches": int(dispatch_reasons["deadline"]),
            "routing_counts": router.assigned.copy(),
            "energy_by_server_j": energy_by_server.copy(),
        }
        if record_requests:
            result.update(
                {
                    "arrival_times_s": arrival_times.copy(),
                    "request_model_indices": request_models.copy(),
                    "request_servers": request_servers.copy(),
                    "dispatch_times_s": dispatch_times.copy(),
                    "completion_times_s": completion_times.copy(),
                    "request_latencies_s": latencies.copy(),
                    "batches": list(batches),
                }
            )
        return result
