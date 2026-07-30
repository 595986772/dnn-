# Paper Core Environment Package

This package contains only the environment side of the final paper project.
It intentionally excludes policies, checkpoints, baselines, result tables, and
exploratory environment variants.

## Environment chain

1. `MultiModelProfileCatalog` loads measured RTX 3060 and M3 latency/energy
   profiles for MobileNetV3-Small, EfficientNet-B0, and ResNet50.
2. `SequentialRouteBatchEnv` implements event-driven arrivals, per-DNN queues,
   request deadlines, WAIT, device routing, and batch-size decisions.
3. `ConcurrentInterferenceRouteBatchEnv` adds concurrent execution and
   configurable co-running slowdown.
4. `HiddenInterferenceRouteBatchEnv` makes the realized interference latent to
   the scheduler while retaining causal execution feedback.
5. `DynamicWorkloadHiddenInterferenceRouteBatchEnv` adds constant, step,
   random-switch, burst, and model-mix-drift arrival scenarios.
6. `deployment_uplink.py` optionally adds a common uplink delay at evaluation
   time. It is disabled in the original edge-side environment.

With the frozen action library, the scheduler chooses one WAIT action or one
of `3 DNNs x 2 devices x 6 batch sizes`, for 37 actions in total.

## Integrity boundary

- The latency and energy profile table is measured on RTX 3060 and M3.
- The final training environment uses a configurable stochastic interference
  abstraction. Hardware co-running measurements validate the existence of
  interference but are not embedded as a directional lookup table here.
- The scheduler cannot observe the true workload rho, scenario label, future
  arrivals, or hidden interference scale.
- The optional uplink module is evaluation-only and disabled by default.

## Quick check

From this package root:

```powershell
$env:PYTHONPATH = "$PWD\code"
python code\smoke_environment.py
python -m pytest -q code\tests\test_deployment_uplink.py
```

The source project used Python 3.9.12 and PyTorch 2.5.1.
