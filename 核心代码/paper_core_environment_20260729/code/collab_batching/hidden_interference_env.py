"""Partially observable concurrent multi-DNN interference environment."""

import numpy as np

from collab_batching.interference_route_batch_env import (
    ConcurrentInterferenceRouteBatchEnv,
)


class HiddenInterferenceRouteBatchEnv(ConcurrentInterferenceRouteBatchEnv):
    """Concurrent inference with hidden episode-level contention severity."""

    def __init__(
        self,
        *args,
        hidden_scale_low=0.75,
        hidden_scale_high=1.25,
        pairwise_spread=0.25,
        expose_interference=False,
        model_mix_schedule=None,
        **kwargs,
    ):
        self.hidden_scale_low = float(hidden_scale_low)
        self.hidden_scale_high = float(hidden_scale_high)
        self.pairwise_spread = float(pairwise_spread)
        self.expose_interference = bool(expose_interference)
        self.model_mix_schedule = None
        if not 0.0 < self.hidden_scale_low <= self.hidden_scale_high:
            raise ValueError("invalid hidden interference scale range")
        if self.pairwise_spread < 0.0:
            raise ValueError("pairwise_spread must be nonnegative")
        self.hidden_interference_scale = 1.0
        super().__init__(*args, **kwargs)
        if model_mix_schedule is not None:
            self.set_model_mix_schedule(model_mix_schedule)
        complexity = self._complexity.mean(axis=0)
        pair = np.ones((self.n_models, self.n_models), dtype=np.float64)
        for left in range(self.n_models):
            for right in range(self.n_models):
                if left == right:
                    continue
                pair[left, right] = self.cross_model_factor * (
                    1.0
                    + self.pairwise_spread
                    * abs(float(complexity[left] - complexity[right]))
                )
        self._pairwise_interference = pair

    def set_model_mix_schedule(self, schedule):
        normalized = []
        previous_end = 0.0
        for end_fraction, mix in schedule:
            end_fraction = float(end_fraction)
            values = np.asarray(mix, dtype=np.float64)
            if (
                end_fraction <= previous_end
                or end_fraction > 1.0 + 1e-12
                or values.shape != (self.n_models,)
                or np.any(values < 0.0)
                or values.sum() <= 0.0
            ):
                raise ValueError("invalid model-mix drift schedule")
            normalized.append((end_fraction, values / values.sum()))
            previous_end = end_fraction
        if not normalized or previous_end < 1.0 - 1e-12:
            raise ValueError("model-mix schedule must cover the full episode")
        self.model_mix_schedule = tuple(normalized)

    def scheduled_model_mix(self, time_s):
        if self.model_mix_schedule is None:
            return np.asarray(self.model_mix, dtype=np.float64).copy()
        fraction = float(time_s) / max(float(self.horizon_s), 1e-12)
        for end_fraction, mix in self.model_mix_schedule:
            if fraction <= end_fraction + 1e-12:
                return mix.copy()
        return self.model_mix_schedule[-1][1].copy()

    def _arrival_trace(self, rng):
        times, models = super()._arrival_trace(rng)
        if self.model_mix_schedule is None or not len(times):
            return times, models
        resampled = np.empty(len(times), dtype=np.int64)
        start_fraction = 0.0
        for end_fraction, mix in self.model_mix_schedule:
            selected = (
                (times >= start_fraction * self.horizon_s - 1e-12)
                & (times <= end_fraction * self.horizon_s + 1e-12)
            )
            count = int(np.count_nonzero(selected))
            if count:
                resampled[selected] = rng.choice(
                    self.n_models, size=count, p=mix
                )
            start_fraction = end_fraction
        return times, resampled

    def reset(self):
        rng = np.random.default_rng(int(self.seed) + 7_901_311)
        self.hidden_interference_scale = float(
            rng.uniform(self.hidden_scale_low, self.hidden_scale_high)
        )
        return super().reset()

    def _slowdown(self, row, active_rows=None):
        active_rows = (
            self._active_for_server(row["server"])
            if active_rows is None
            else list(active_rows)
        )
        own_demand = self._record_demand(row)
        penalty = 0.0
        for other in active_rows:
            if int(other["batch_id"]) == int(row["batch_id"]):
                continue
            pair = self._pairwise_interference[
                int(row["model_index"]), int(other["model_index"])
            ]
            penalty += pair * np.sqrt(own_demand * self._record_demand(other))
        return float(
            np.clip(
                1.0
                + self.interference_strength
                * self.hidden_interference_scale
                * penalty,
                1.0,
                self.max_slowdown,
            )
        )

    def _concurrency_observation(self):
        values = super()._concurrency_observation()
        if not self.expose_interference:
            values[self.N : 2 * self.N] = 0.0
        return values

    def result(self, record_requests=False):
        result = super().result(record_requests=record_requests)
        result.update(
            {
                "hidden_interference_scale": float(
                    self.hidden_interference_scale
                ),
                "model_deadline_s": np.asarray(
                    [self.deadline_s_for_model(index) for index in range(self.n_models)]
                ),
            }
        )
        return result
