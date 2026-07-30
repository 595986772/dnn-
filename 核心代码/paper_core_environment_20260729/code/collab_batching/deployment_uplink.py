"""Action-independent uplink latency for deployment-aware evaluation.

The final scheduling environment models queueing, batching, and execution after
requests reach the edge ingress. This module adds an optional client-to-edge
upload delay without changing arrivals seen by the scheduler, policy actions,
rewards, or training. It is disabled by default.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


UPLINK_RATE_RANGES_MBPS = {
    "slow": (2.5, 5.0),
    "default": (5.0, 10.0),
    "fast": (10.0, 20.0),
}


@dataclass(frozen=True)
class DeploymentUplinkConfig:
    enabled: bool = False
    profile: str = "default"
    fallback_size_kb_low: float = 50.0
    fallback_size_kb_high: float = 100.0
    jpeg_root: str = ""
    base_seed: int = 20260724

    def validate(self):
        if self.profile not in UPLINK_RATE_RANGES_MBPS:
            raise ValueError("unknown uplink profile: %s" % self.profile)
        if self.fallback_size_kb_low <= 0.0:
            raise ValueError("fallback_size_kb_low must be positive")
        if self.fallback_size_kb_high < self.fallback_size_kb_low:
            raise ValueError(
                "fallback_size_kb_high must be at least fallback_size_kb_low"
            )
        return self


def stable_sample_seed(base_seed, *parts):
    """Return a process-independent seed for common random numbers."""
    payload = "|".join([str(int(base_seed))] + [str(part) for part in parts])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def find_jpeg_sizes_bytes(root):
    """Return file sizes for JPEG inputs, excluding PNG and other formats."""
    if root is None or not str(root).strip():
        return np.zeros(0, dtype=np.float64)
    path = Path(root)
    if not path.exists():
        return np.zeros(0, dtype=np.float64)
    sizes = [
        float(candidate.stat().st_size)
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg"}
    ]
    return np.asarray(sorted(sizes), dtype=np.float64)


def common_uplink_quantiles(request_count, config, *sample_key):
    """Generate method-independent size/rate quantiles for one request trace."""
    config.validate()
    count = int(request_count)
    if count < 0:
        raise ValueError("request_count must be nonnegative")
    rng = np.random.default_rng(
        stable_sample_seed(config.base_seed, *sample_key)
    )
    return rng.random(count), rng.random(count)


def payload_sizes_bytes(size_quantiles, config, jpeg_sizes_bytes=None):
    """Map common quantiles to measured JPEG sizes or the configured fallback."""
    config.validate()
    quantiles = np.asarray(size_quantiles, dtype=np.float64)
    measured = np.asarray(
        [] if jpeg_sizes_bytes is None else jpeg_sizes_bytes,
        dtype=np.float64,
    )
    measured = measured[measured > 0.0]
    if len(measured):
        ordered = np.sort(measured)
        indices = np.minimum(
            (np.clip(quantiles, 0.0, 1.0) * len(ordered)).astype(np.int64),
            len(ordered) - 1,
        )
        return ordered[indices], "jpeg_file_size"
    low = 1000.0 * float(config.fallback_size_kb_low)
    high = 1000.0 * float(config.fallback_size_kb_high)
    return low + (high - low) * quantiles, "uniform_50_100_kb_fallback"


def uplink_rates_mbytes_per_s(rate_quantiles, profile):
    if profile not in UPLINK_RATE_RANGES_MBPS:
        raise ValueError("unknown uplink profile: %s" % profile)
    low, high = UPLINK_RATE_RANGES_MBPS[profile]
    quantiles = np.asarray(rate_quantiles, dtype=np.float64)
    return float(low) + (float(high) - float(low)) * quantiles


def deployment_latency_metrics(
    result,
    config=None,
    sample_key=(),
    jpeg_sizes_bytes=None,
):
    """Compute edge-side and deployment-aware request metrics.

    Upload is an additive client-side latency. It intentionally does not alter
    edge arrival times, scheduling actions, queue evolution, energy, or service
    throughput.
    """
    config = (config or DeploymentUplinkConfig()).validate()
    edge = np.asarray(result["request_latencies_s"], dtype=np.float64)
    deadlines = np.asarray(result["request_deadline_s"], dtype=np.float64)
    if edge.shape != deadlines.shape:
        raise ValueError("request latencies and deadlines must have equal shape")

    size_q, rate_q = common_uplink_quantiles(
        len(edge), config, *tuple(sample_key)
    )
    sizes, size_source = payload_sizes_bytes(
        size_q, config, jpeg_sizes_bytes=jpeg_sizes_bytes
    )
    rates = uplink_rates_mbytes_per_s(rate_q, config.profile)
    upload = sizes / np.maximum(rates * 1_000_000.0, 1e-12)
    if not config.enabled:
        upload = np.zeros_like(edge)
    total = edge + upload

    def percentile(values, q):
        return float(np.percentile(values, q)) if len(values) else 0.0

    edge_violations = edge > deadlines + 1e-12
    total_violations = total > deadlines + 1e-12
    return {
        "uplink_enabled": bool(config.enabled),
        "uplink_profile": str(config.profile),
        "input_size_source": str(size_source),
        "requests": int(len(edge)),
        "edge_violation_count": int(edge_violations.sum()),
        "edge_violation_rate": float(edge_violations.mean()) if len(edge) else 0.0,
        "deployment_violation_count": int(total_violations.sum()),
        "deployment_violation_rate": (
            float(total_violations.mean()) if len(total) else 0.0
        ),
        "additional_violation_count": int(
            total_violations.sum() - edge_violations.sum()
        ),
        "edge_p95_latency_ms": 1000.0 * percentile(edge, 95),
        "edge_p99_latency_ms": 1000.0 * percentile(edge, 99),
        "deployment_p95_latency_ms": 1000.0 * percentile(total, 95),
        "deployment_p99_latency_ms": 1000.0 * percentile(total, 99),
        "mean_upload_latency_ms": 1000.0 * float(upload.mean())
        if len(upload)
        else 0.0,
        "p95_upload_latency_ms": 1000.0 * percentile(upload, 95),
        "p99_upload_latency_ms": 1000.0 * percentile(upload, 99),
        "mean_input_size_kb": float(sizes.mean() / 1000.0)
        if len(sizes)
        else 0.0,
        "mean_uplink_rate_mbytes_s": float(rates.mean()) if len(rates) else 0.0,
        "edge_latencies_s": edge,
        "deployment_latencies_s": total,
        "request_deadline_s": deadlines,
        "upload_latencies_s": upload,
        "input_sizes_bytes": sizes,
        "uplink_rates_mbytes_s": rates,
    }
