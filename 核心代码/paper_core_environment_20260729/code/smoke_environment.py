"""Minimal executable check for the packaged paper environment."""

import json
import os
from pathlib import Path
import sys

# The original Windows/Anaconda runtime loads OpenMP through both NumPy and
# PyTorch. This affects only the standalone smoke process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from collab_batching.dynamic_workload_env import (  # noqa: E402
    DynamicWorkloadHiddenInterferenceRouteBatchEnv,
)
from collab_batching.online_batch_env import MultiModelProfileCatalog  # noqa: E402


def _csv_ints(value):
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def main():
    config_path = ROOT / "configs" / "final_ours_config_portable.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    profile_path = ROOT / config["profile_csv"]
    catalog = MultiModelProfileCatalog(str(profile_path))
    mix = np.asarray(
        [config["model_mix"][model] for model in catalog.models],
        dtype=np.float64,
    )
    peak_rps = catalog.aggregate_peak_rps(config["providers"], model_mix=mix)

    env = DynamicWorkloadHiddenInterferenceRouteBatchEnv(
        str(profile_path),
        config["providers"],
        arrival_rate_rps=0.7 * peak_rps,
        horizon_s=0.10,
        deadline_s=config["deadlines"]["default"],
        deadline_multiplier_by_model=config[
            "deadline_multiplier_by_model_resolved"
        ],
        model_mix=mix,
        seed=20260729,
        batch_sizes=_csv_ints(config["batch_sizes"]),
        energy_ref_per_request=config["energy_ref_per_request_resolved"],
        sla_penalty_timing=config["sla_penalty_timing"],
        predicted_safety_mask=False,
        overload_observation=True,
        queue_distribution_observation=True,
        catalog=catalog,
        server_concurrency=_csv_ints(config["server_concurrency"]),
        interference_strength=config["interference_strength"],
        cross_model_factor=config["cross_model_factor"],
        max_slowdown=config["max_slowdown"],
        energy_interference_ratio=config["energy_interference_ratio"],
        hidden_scale_low=config["hidden_interference_low"],
        hidden_scale_high=config["hidden_interference_high"],
        pairwise_spread=config["pairwise_interference_spread"],
        expose_interference=False,
        workload_scenario="random_switch",
        workload_segment_s=config["workload_segment_s"],
        workload_burst_bin_s=config["workload_burst_bin_s"],
    )

    observation, mask = env.reset()
    steps = 0
    while not env._done and steps < 10000:
        dispatch = np.flatnonzero(mask[1:]) + 1
        action = int(dispatch[0]) if len(dispatch) else 0
        observation, mask, _, _, _ = env.step(action)
        steps += 1

    if not env._done:
        raise RuntimeError("environment did not terminate within the smoke limit")
    result = env.result(record_requests=True)
    if result["requests"] != result["completed_requests"]:
        raise RuntimeError("request conservation check failed")
    print(
        "OK: actions=%d requests=%d completed=%d steps=%d"
        % (
            env.action_dim,
            result["requests"],
            result["completed_requests"],
            steps,
        )
    )


if __name__ == "__main__":
    main()
