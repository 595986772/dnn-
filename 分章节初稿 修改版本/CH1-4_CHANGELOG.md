# CH1-4 Change Log

## Outputs

- `Introduction_中文修订稿_v5_四章对齐版.docx`
- `Related_Work_中文修订稿_v4_四章对齐版.docx`
- `System_Model_中文初稿_v8_四章对齐版.docx`
- `Method_中文初稿_v9_四章对齐版.docx`

All source DOCX files remain unchanged.

## Chapter 1

- Reorganized the body into eight progressive paragraphs plus four contributions.
- Unified the method name as **RiskBudget-SAC**.
- Added the exact edge-ingress scope and separated edge-side latency from action-independent deployment-aware uplink evaluation.
- Replaced a paper-by-paper gap list with the three coupled gaps: structured joint action, revealed-feedback adaptation, and differentiated SLA/tail/estimated-energy control.
- Narrowed claims to one frozen 400-episode checkpoint and five paired evaluation seeds per condition.

## Chapter 2

- Reorganized prior work into heterogeneous multi-DNN serving, learning-based scheduling/risk control, adjacent MEC/collaborative inference, and a final gap summary.
- Preserved the original 22 references and their bibliographic text.
- Did not introduce unsupported EMERALD/BlastNet references; only works present in the source bibliography are discussed.
- Clarified that SAC, DRL, batching, deadline awareness, and interference awareness are prior art rather than independent contributions.

## Chapter 3

- Rebuilt the model from edge ingress; removed Shannon-rate and wireless-resource decisions from the core system.
- Added the action-independent deployment-aware uplink term only as an evaluation overlay.
- Collapsed redundant binary dispatch variables into one event-level physical action set.
- Unified all compute resources as execution devices indexed by `d`.
- Distinguished baseline service profiles, hidden runtime slowdown, and profile-estimated energy.
- Replaced lexicographic direct-p99 optimization with a tail-surrogate objective under soft SLA and energy constraints.

## Chapter 4

- Matched the final implementation: causal EWMA from revealed arrivals, completed-batch EWMA runtime correction, GRU history, selective Risk Gate, base-plus-action-residual Actor, and action-aligned Twin Critics.
- Corrected the Critic MSE to the sum of two squared errors.
- Defined the exact trapezoidal queue/slack tail-risk surrogate used by the source code.
- Added physical-time n-step targets, model-level SLA dual updates, balanced tail replay, and two complete algorithms.
- Added a deployment path and linear-in-action-library complexity statement.
