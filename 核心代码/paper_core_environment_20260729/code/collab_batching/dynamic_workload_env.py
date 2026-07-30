"""Hidden-interference environment with reproducible dynamic arrivals.

The workload schedule is used only by the arrival generator.  Scheduler
observations are still rebuilt from causally revealed arrivals, so policies do
not receive the current or future schedule.
"""

from dataclasses import asdict, dataclass

import numpy as np

from collab_batching.hidden_interference_env import (
    HiddenInterferenceRouteBatchEnv,
)


@dataclass(frozen=True)
class WorkloadSegment:
    start_s: float
    end_s: float
    rate_factor: float
    model_mix: tuple

    def to_dict(self):
        return asdict(self)


class DynamicWorkloadHiddenInterferenceRouteBatchEnv(
    HiddenInterferenceRouteBatchEnv
):
    """Concurrent inference with an unobserved piecewise arrival schedule."""

    VALID_SCENARIOS = {
        "constant",
        "step",
        "random_switch",
        "burst",
        "model_mix_drift",
    }

    def __init__(
        self,
        *args,
        workload_scenario="constant",
        workload_segment_s=0.25,
        workload_burst_bin_s=0.05,
        workload_seed_offset=8_104_729,
        **kwargs,
    ):
        self.workload_scenario = str(workload_scenario).strip().lower()
        if self.workload_scenario not in self.VALID_SCENARIOS:
            raise ValueError(
                "unknown dynamic workload scenario: %s" % self.workload_scenario
            )
        self.workload_segment_s = float(workload_segment_s)
        self.workload_burst_bin_s = float(workload_burst_bin_s)
        self.workload_seed_offset = int(workload_seed_offset)
        if self.workload_segment_s <= 0.0 or self.workload_burst_bin_s <= 0.0:
            raise ValueError("dynamic workload intervals must be positive")
        self._workload_segments = ()
        super().__init__(*args, **kwargs)

    @staticmethod
    def _normalize_factors(rows, horizon_s):
        weighted = sum(
            (float(end) - float(start)) * float(factor)
            for start, end, factor, _ in rows
        )
        mean = weighted / max(float(horizon_s), 1e-12)
        if mean <= 0.0:
            raise ValueError("dynamic workload has a nonpositive mean rate")
        return [
            (start, end, float(factor) / mean, mix)
            for start, end, factor, mix in rows
        ]

    def _fixed_width_rows(self, width_s, factors, mixes=None):
        rows = []
        start = 0.0
        index = 0
        while start < self.horizon_s - 1e-12:
            end = min(start + float(width_s), self.horizon_s)
            mix = self.model_mix if mixes is None else mixes[index]
            rows.append((start, end, float(factors[index]), mix))
            start = end
            index += 1
        return rows

    def _build_workload_schedule(self):
        rng = np.random.default_rng(int(self.seed) + self.workload_seed_offset)
        horizon = float(self.horizon_s)
        base_mix = np.asarray(self.model_mix, dtype=np.float64)
        scenario = self.workload_scenario

        if scenario == "constant":
            rows = [(0.0, horizon, 1.0, base_mix)]
        elif scenario == "step":
            # At target rho=0.7 this is 0.5 -> 0.9 -> 0.5.  The middle
            # interval occupies half of the episode, preserving mean rho.
            rows = [
                (0.0, 0.25 * horizon, 5.0 / 7.0, base_mix),
                (0.25 * horizon, 0.75 * horizon, 9.0 / 7.0, base_mix),
                (0.75 * horizon, horizon, 5.0 / 7.0, base_mix),
            ]
        elif scenario == "random_switch":
            count = max(int(np.ceil(horizon / self.workload_segment_s)), 1)
            palette = np.asarray([4.0, 6.0, 8.0, 10.0]) / 7.0
            factors = []
            while len(factors) < count:
                factors.extend(rng.permutation(palette).tolist())
            rows = self._fixed_width_rows(
                self.workload_segment_s, factors[:count]
            )
        elif scenario == "burst":
            count = max(int(np.ceil(horizon / self.workload_burst_bin_s)), 1)
            durations = np.full(count, self.workload_burst_bin_s, dtype=np.float64)
            durations[-1] = horizon - self.workload_burst_bin_s * (count - 1)
            high_count = max(int(round(0.25 * count)), 1)
            # Smooth random scores create contiguous 50-150 ms burst regions.
            scores = rng.normal(size=count)
            kernel_width = max(
                int(round(0.10 / self.workload_burst_bin_s)), 1
            )
            smooth = np.convolve(
                scores, np.ones(kernel_width, dtype=np.float64), mode="same"
            )
            high = np.zeros(count, dtype=bool)
            high[np.argsort(smooth)[-high_count:]] = True
            factors = np.where(high, 10.0 / 7.0, 6.0 / 7.0)
            rows = []
            start = 0.0
            for index in range(count):
                end = min(start + float(durations[index]), horizon)
                rows.append((start, end, factors[index], base_mix))
                start = end
        else:
            # Rotate the dominant DNN while keeping the average composition
            # uniform.  Arrival rate follows the local aggregate capacity so
            # this remains a composition shift rather than a hidden overload.
            focus_order = rng.permutation(self.n_models)
            rows = []
            for index, focus in enumerate(focus_order):
                start = horizon * index / self.n_models
                end = horizon * (index + 1) / self.n_models
                mix = np.full(self.n_models, 0.4 / max(self.n_models - 1, 1))
                mix[int(focus)] = 0.6
                local_peak = self.catalog.aggregate_peak_rps(
                    self.providers, model_mix=mix
                )
                factor = local_peak / max(self._aggregate_peak_rps, 1e-12)
                rows.append((start, end, factor, mix))

        if scenario in {"random_switch", "burst"}:
            rows = self._normalize_factors(rows, horizon)
        segments = []
        for start, end, factor, mix in rows:
            values = np.asarray(mix, dtype=np.float64)
            values = values / values.sum()
            segments.append(
                WorkloadSegment(
                    start_s=float(start),
                    end_s=float(end),
                    rate_factor=float(factor),
                    model_mix=tuple(float(value) for value in values),
                )
            )
        self._workload_segments = tuple(segments)

    def _segment_at(self, time_s):
        if not self._workload_segments:
            self._build_workload_schedule()
        time_s = float(time_s)
        for segment in self._workload_segments:
            if time_s < segment.end_s - 1e-12:
                return segment
        return self._workload_segments[-1]

    def scheduled_arrival_rate_rps(self, time_s):
        segment = self._segment_at(time_s)
        return float(self.arrival_rate_rps * segment.rate_factor)

    def scheduled_model_mix(self, time_s):
        segment = self._segment_at(time_s)
        return np.asarray(segment.model_mix, dtype=np.float64)

    def workload_schedule_metadata(self):
        if not self._workload_segments:
            self._build_workload_schedule()
        return [segment.to_dict() for segment in self._workload_segments]

    def _arrival_trace(self, rng):
        if self.workload_scenario == "constant":
            return super()._arrival_trace(rng)
        self._build_workload_schedule()
        times = []
        models = []
        for segment in self._workload_segments:
            duration = segment.end_s - segment.start_s
            rate = self.arrival_rate_rps * segment.rate_factor
            count = int(rng.poisson(max(rate * duration, 0.0)))
            if count <= 0:
                continue
            local_times = segment.start_s + rng.uniform(0.0, duration, size=count)
            local_models = rng.choice(
                self.n_models,
                size=count,
                p=np.asarray(segment.model_mix, dtype=np.float64),
            )
            times.extend(local_times.tolist())
            models.extend(local_models.tolist())
        if not times:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
        order = np.argsort(np.asarray(times, dtype=np.float64), kind="stable")
        return (
            np.asarray(times, dtype=np.float64)[order],
            np.asarray(models, dtype=np.int64)[order],
        )

    def reset(self):
        self._workload_segments = ()
        return super().reset()

    def result(self, record_requests=False):
        result = super().result(record_requests=record_requests)
        result.update(
            {
                "workload_scenario": self.workload_scenario,
                "workload_schedule": self.workload_schedule_metadata(),
            }
        )
        return result
