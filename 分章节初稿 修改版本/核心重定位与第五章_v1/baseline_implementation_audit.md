# Baseline Implementation Audit

| Baseline | Current implementation | Future/hidden information |
|---|---|---|
| Coinf-Adapted | Coinf-style prediction/urgency + A2C, extended with device choice | No future arrivals or hidden interference scale |
| TF-Serving-style | Fixed cap 32 and bounded wait | Current queue/device only |
| Triton-Dynamic-Batcher | Preferred batches 16/32 and queue delay | Current queue/device only |
| Ebird-style | Adapted multi-batch portfolio to 2+1 slots | Current queue/device only |
| EDF-StaticBatch | Earliest deadline, fixed cap, offline-profile routing | Current queue/device + fixed profile |
| FCFS | Globally oldest request, fixed cap, first available device | Current queue/device only |

All methods share the same arrival traces, deadlines, profile table, device
capacity, physical action mask, runtime perturbations, uplink samples, and
evaluation seeds. Coinf-Adapted is an environment adaptation, not the original
paper's executable artifact.
