# Terminology Audit

| Preferred term | Usage rule | Avoided alternatives |
|---|---|---|
| RiskBudget-SAC | Single canonical method name | RiskBudget, Ours-SamePolicy, Ours-LoadGated |
| edge ingress / 边缘入口 | Start of the core scheduling model | Treating wireless transmission as a learned action |
| execution device / 执行设备 | All compute-resource references, indexed by `d` | Mixing EC, server, provider, node |
| dispatch / 派发 | Queue-to-device batch action | “卸载” when no end-edge-cloud placement is decided |
| physical action set | Queue, batch, capacity, and verified WAIT feasibility | Claiming WAIT is always available |
| selective Risk Gate | Constructs an admissible nonempty subset only | Safety guarantee, oracle gate, causal-inference gate |
| revealed arrivals / completed-batch feedback | Only information used for online adaptation | Future arrivals, true load factor, scenario label |
| baseline service profile | Known DNN–device–batch execution attributes | Claiming exact online performance |
| profile-estimated energy | Execution-profile accounting in J/request | Actual whole-system energy |
| edge-side latency | Measured from edge ingress | Mixing with client uplink |
| deployment-aware latency | Action-independent uplink plus edge-side latency | Claiming wireless-resource optimization |
| model-level deadline / request-level SLA violation | Target and binary outcome | Deterministic SLA guarantee |
| tail-risk surrogate | Training signal from queue/slack/physical time | Directly optimizing empirical p99 |
| paired evaluation seeds | Environment/evaluation repeats for one checkpoint | Independent training seeds |

## Claim boundary

- The method is not described as the first use of SAC, GRU, batching, Twin Critics, Lagrangian constraints, or prioritized replay.
- Reported improvements are macro averages relative to the strongest baseline for each metric.
- The text explicitly states that not every load point and metric is best.
- No training-stability claim is made from the five paired evaluation seeds.
