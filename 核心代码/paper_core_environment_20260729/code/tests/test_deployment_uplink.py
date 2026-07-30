from pathlib import Path

import numpy as np

from collab_batching.deployment_uplink import (
    DeploymentUplinkConfig,
    common_uplink_quantiles,
    deployment_latency_metrics,
    find_jpeg_sizes_bytes,
)


def sample_result():
    return {
        "request_latencies_s": np.asarray([0.020, 0.030, 0.040]),
        "request_deadline_s": np.asarray([0.035, 0.035, 0.050]),
    }


def test_uplink_is_disabled_by_default():
    metrics = deployment_latency_metrics(
        sample_result(), sample_key=("static", "rho_0.7", 1)
    )
    assert not metrics["uplink_enabled"]
    np.testing.assert_allclose(
        metrics["edge_latencies_s"], metrics["deployment_latencies_s"]
    )
    assert metrics["mean_upload_latency_ms"] == 0.0


def test_default_uplink_delay_is_in_requested_range():
    config = DeploymentUplinkConfig(enabled=True, profile="default")
    result = {
        "request_latencies_s": np.zeros(10000),
        "request_deadline_s": np.ones(10000),
    }
    metrics = deployment_latency_metrics(
        result, config=config, sample_key=("range", 7)
    )
    delays_ms = 1000.0 * metrics["upload_latencies_s"]
    assert delays_ms.min() >= 5.0
    assert delays_ms.max() <= 20.0


def test_common_random_numbers_do_not_depend_on_method():
    config = DeploymentUplinkConfig(enabled=True, profile="default")
    first = common_uplink_quantiles(32, config, "static", "rho_0.7", 9200)
    second = common_uplink_quantiles(32, config, "static", "rho_0.7", 9200)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_profiles_have_ordered_upload_delays():
    result = {
        "request_latencies_s": np.zeros(256),
        "request_deadline_s": np.ones(256),
    }
    values = {}
    for profile in ("slow", "default", "fast"):
        metrics = deployment_latency_metrics(
            result,
            config=DeploymentUplinkConfig(enabled=True, profile=profile),
            sample_key=("ordered", 3),
        )
        values[profile] = metrics["upload_latencies_s"]
    assert np.all(values["slow"] > values["default"])
    assert np.all(values["default"] > values["fast"])


def test_jpeg_sizes_are_used_when_available(tmp_path):
    jpeg = Path(tmp_path) / "sample.jpg"
    jpeg.write_bytes(b"x" * 12345)
    (Path(tmp_path) / "ignored.png").write_bytes(b"x" * 99999)
    sizes = find_jpeg_sizes_bytes(tmp_path)
    np.testing.assert_array_equal(sizes, np.asarray([12345.0]))
    metrics = deployment_latency_metrics(
        sample_result(),
        config=DeploymentUplinkConfig(
            enabled=True, profile="default", jpeg_root=str(tmp_path)
        ),
        sample_key=("jpeg", 1),
        jpeg_sizes_bytes=sizes,
    )
    assert metrics["input_size_source"] == "jpeg_file_size"
    np.testing.assert_allclose(metrics["input_sizes_bytes"], 12345.0)


def test_total_latency_uses_original_deadline():
    result = {
        "request_latencies_s": np.asarray([0.030]),
        "request_deadline_s": np.asarray([0.035]),
    }
    metrics = deployment_latency_metrics(
        result,
        config=DeploymentUplinkConfig(enabled=True, profile="default"),
        sample_key=("deadline", 1),
    )
    assert metrics["edge_violation_count"] == 0
    assert metrics["deployment_violation_count"] == 1
    assert metrics["additional_violation_count"] == 1
