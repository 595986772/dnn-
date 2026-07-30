"""Leakage-free OpenB-guided request arrival generation.

OpenB records cluster-level pod creation times, not edge-inference requests.
The generator therefore replays the trace's normalized, chronological arrival
intensity while preserving the simulator's requested mean arrival rate.  DNN
labels are sampled independently from the environment's configured model mix.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OpenBSplitStats:
    split: str
    start_s: float
    end_s: float
    event_count: int
    bin_count: int
    mean_events_per_raw_bin: float
    fano_factor: float
    lag1_correlation: float

    def to_dict(self):
        return {
            "split": self.split,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "event_count": self.event_count,
            "bin_count": self.bin_count,
            "mean_events_per_raw_bin": self.mean_events_per_raw_bin,
            "fano_factor": self.fano_factor,
            "lag1_correlation": self.lag1_correlation,
        }


def _lag1(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or values[:-1].std() <= 1e-12 or values[1:].std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


@lru_cache(maxsize=8)
def _load_creation_times(path):
    frame = pd.read_csv(
        path,
        usecols=["creation_time"],
        dtype={"creation_time": "float64"},
    )
    values = frame["creation_time"].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("OpenB trace contains no finite creation_time values")
    return np.sort(values, kind="stable")


class OpenBArrivalSource:
    """Chronologically split empirical intensity source for OpenB arrivals."""

    VALID_SPLITS = ("train", "validation", "test")

    def __init__(
        self,
        csv_path,
        lookback_days=30.0,
        raw_bin_s=3600.0,
        sim_bin_s=0.25,
        split_fractions=(2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
    ):
        self.csv_path = str(Path(csv_path).resolve())
        self.lookback_days = float(lookback_days)
        self.raw_bin_s = float(raw_bin_s)
        self.sim_bin_s = float(sim_bin_s)
        fractions = np.asarray(split_fractions, dtype=np.float64)
        if self.lookback_days <= 0.0:
            raise ValueError("lookback_days must be positive")
        if self.raw_bin_s <= 0.0 or self.sim_bin_s <= 0.0:
            raise ValueError("OpenB raw and simulation bins must be positive")
        if fractions.shape != (3,) or np.any(fractions <= 0.0):
            raise ValueError("OpenB split fractions must contain three positive values")
        self.split_fractions = fractions / fractions.sum()
        self._counts = self._build_counts()

    def _build_counts(self):
        times = _load_creation_times(self.csv_path)
        end_s = float(times[-1]) + 1e-9
        start_s = max(float(times[0]), end_s - self.lookback_days * 86400.0)
        if end_s <= start_s:
            raise ValueError("OpenB lookback interval is empty")
        boundaries = np.concatenate(
            ([start_s], start_s + np.cumsum(self.split_fractions) * (end_s - start_s))
        )
        boundaries[-1] = end_s
        result = {}
        for index, split in enumerate(self.VALID_SPLITS):
            left = float(boundaries[index])
            right = float(boundaries[index + 1])
            bin_count = max(int(np.ceil((right - left) / self.raw_bin_s)), 2)
            edges = left + np.arange(bin_count + 1, dtype=np.float64) * self.raw_bin_s
            edges[-1] = max(edges[-1], right)
            selected = times[(times >= left) & (times < right)]
            counts, _ = np.histogram(selected, bins=edges)
            counts = counts.astype(np.float64)
            mean = float(counts.mean())
            if mean <= 0.0:
                raise ValueError("OpenB %s split contains no arrivals" % split)
            result[split] = {
                "intensity": counts / mean,
                "stats": OpenBSplitStats(
                    split=split,
                    start_s=left,
                    end_s=right,
                    event_count=int(len(selected)),
                    bin_count=int(len(counts)),
                    mean_events_per_raw_bin=mean,
                    fano_factor=float(counts.var() / mean),
                    lag1_correlation=_lag1(counts),
                ),
            }
        return result

    def stats(self, split):
        split = str(split).strip().lower()
        if split not in self._counts:
            raise ValueError("OpenB split must be train, validation, or test")
        return self._counts[split]["stats"]

    def _hard_start_pool(self, split, bins_needed, quantile):
        """Return train-only starts covering high-load or rising segments."""

        split = str(split).strip().lower()
        if split != "train":
            raise ValueError("hard OpenB sampling is restricted to the train split")
        intensity = self._counts[split]["intensity"]
        offsets = np.arange(int(bins_needed), dtype=np.int64)
        segments = np.stack(
            [
                intensity[(start + offsets) % len(intensity)]
                for start in range(len(intensity))
            ],
            axis=0,
        )
        mean_load = segments.mean(axis=1)
        if segments.shape[1] > 1:
            positive_ramp = np.maximum(
                np.diff(segments, axis=1), 0.0
            ).max(axis=1)
        else:
            positive_ramp = np.zeros(len(segments), dtype=np.float64)
        load_cut = float(np.quantile(mean_load, float(quantile)))
        ramp_cut = float(np.quantile(positive_ramp, float(quantile)))
        hard = (mean_load >= load_cut) | (
            (positive_ramp > 0.0) & (positive_ramp >= ramp_cut)
        )
        pool = np.flatnonzero(hard)
        if len(pool) == 0:
            pool = np.arange(len(intensity), dtype=np.int64)
        return pool, mean_load, positive_ramp

    def sample(
        self,
        split,
        target_rate_rps,
        horizon_s,
        model_mix,
        rng,
        sampling_mode="uniform",
        hard_fraction=0.5,
        hard_quantile=0.8,
    ):
        """Sample one episode from a contiguous chronological intensity segment."""

        split = str(split).strip().lower()
        if split not in self._counts:
            raise ValueError("OpenB split must be train, validation, or test")
        target_rate_rps = float(target_rate_rps)
        horizon_s = float(horizon_s)
        model_mix = np.asarray(model_mix, dtype=np.float64)
        if target_rate_rps < 0.0 or horizon_s <= 0.0:
            raise ValueError("target rate and horizon must be valid")
        if model_mix.ndim != 1 or len(model_mix) == 0:
            raise ValueError("model_mix must be a nonempty vector")
        sampling_mode = str(sampling_mode).strip().lower()
        if sampling_mode not in {"uniform", "burst_balanced"}:
            raise ValueError("OpenB sampling mode must be uniform or burst_balanced")
        hard_fraction = float(hard_fraction)
        hard_quantile = float(hard_quantile)
        if not 0.0 <= hard_fraction <= 1.0:
            raise ValueError("OpenB hard fraction must be in [0, 1]")
        if not 0.0 < hard_quantile < 1.0:
            raise ValueError("OpenB hard quantile must be in (0, 1)")
        if sampling_mode != "uniform" and split != "train":
            raise ValueError(
                "non-uniform OpenB sampling is restricted to the train split"
            )
        intensity = self._counts[split]["intensity"]
        bins_needed = int(np.ceil(horizon_s / self.sim_bin_s))
        # Circular indexing keeps a segment chronological when a long episode
        # spans the end of a held-out split; it never crosses split boundaries.
        hard_sample = False
        if sampling_mode == "burst_balanced" and rng.random() < hard_fraction:
            pool, mean_load, positive_ramp = self._hard_start_pool(
                split, bins_needed, hard_quantile
            )
            start = int(rng.choice(pool))
            hard_sample = True
        else:
            start = int(rng.integers(0, len(intensity)))
            offsets = np.arange(bins_needed, dtype=np.int64)
            segment = intensity[(start + offsets) % len(intensity)]
            mean_load = np.asarray([segment.mean()], dtype=np.float64)
            positive_ramp = np.asarray(
                [
                    np.maximum(np.diff(segment), 0.0).max()
                    if len(segment) > 1
                    else 0.0
                ],
                dtype=np.float64,
            )
        indices = (start + np.arange(bins_needed, dtype=np.int64)) % len(intensity)
        local_intensity = intensity[indices]
        self.last_sample_info = {
            "split": split,
            "sampling_mode": sampling_mode,
            "hard_sample": bool(hard_sample),
            "start_index": int(start),
            "mean_intensity": float(local_intensity.mean()),
            "max_intensity": float(local_intensity.max()),
            "max_positive_ramp": float(
                np.maximum(np.diff(local_intensity), 0.0).max()
                if len(local_intensity) > 1
                else 0.0
            ),
        }
        times = []
        for index, multiplier in enumerate(local_intensity):
            begin = index * self.sim_bin_s
            duration = min(self.sim_bin_s, horizon_s - begin)
            if duration <= 0.0:
                break
            count = int(rng.poisson(max(target_rate_rps * multiplier * duration, 0.0)))
            if count:
                times.extend((begin + rng.uniform(0.0, duration, size=count)).tolist())
        if not times:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
        times = np.sort(np.asarray(times, dtype=np.float64), kind="stable")
        models = rng.choice(len(model_mix), size=len(times), p=model_mix)
        return times, np.asarray(models, dtype=np.int64)

    def metadata(self, split):
        return {
            "mode": "openb_guided",
            "trace_csv": self.csv_path,
            "lookback_days": self.lookback_days,
            "raw_bin_s": self.raw_bin_s,
            "sim_bin_s": self.sim_bin_s,
            "split_fractions": self.split_fractions.tolist(),
            "split_stats": self.stats(split).to_dict(),
        }
