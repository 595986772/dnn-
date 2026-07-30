# Formula Cross-Reference Audit v2

## Chapter 3

- (1)-(6): request, deadline, arrivals, FIFO queue, physical actions, queue evolution.
- (6a): hidden state, observable history, policy information set, variable action duration.
- (7)-(9): baseline service properties, hidden runtime slowdown, profile-estimated energy.
- (10)-(16): waiting, edge/deployment latency, SLA, p99, energy/request.
- (17): tail-risk objective with per-model SLA constraints and a fixed energy-budget hinge.

## Chapter 4

- (18)-(20): causal arrival estimate, completed-feedback runtime correction, explicit 10-D history record and GRU context.
- (21a)-(21c): pressure/slack, predicted finish margin, nonempty Gate set.
- (22)-(25): action semantics, Actor score, masked policy, action-aligned Twin Critics.
- (26): physical-time tail-risk surrogate.
- (27a)-(27b): raw SLA/energy components and reward reconstruction.
- (28): per-model SLA dual update.
- (29a)-(29b): component-wise physical-time aggregation and soft target.
- (30): sum of two Critic MSE terms.
- (31): discrete SAC Actor objective.

Checks:

- Gate does not consume the GRU hidden vector directly.
- High-pressure/urgent Gate branches preselect a DNN before Actor device-batch selection.
- Replay aggregates raw components before current-dual reward reconstruction.
- Empirical p99 is an evaluation metric, not a differentiable training target.
