# Terminology Audit v2

| Canonical term | Required meaning |
|---|---|
| RiskBudget-SAC | The only method name |
| Historical Feedback GRU | Core module encoding completed-batch feedback |
| Feedback-Calibrated Selective Risk Gate | Core module constructing a nonempty admissible set |
| SLA-constrained and p99-oriented SAC | Core training objective |
| action-semantic Actor / action-aligned Twin Critics | Policy backbone, not the first contribution |
| edge ingress | Start of the controlled system |
| deployment-aware latency | Action-independent uplink plus edge-side latency |
| profile-estimated energy | Execution-profile accounting, not whole-system energy |
| paired evaluation seeds | Environment repeats for one checkpoint, not training seeds |

Prohibited overclaims:

- no strict oracle-free claim for the entire simulator observation/mask;
- no deterministic SLA/safety guarantee;
- no direct empirical-p99 optimization claim;
- no minimum-energy claim;
- no five-independent-training-runs claim.
