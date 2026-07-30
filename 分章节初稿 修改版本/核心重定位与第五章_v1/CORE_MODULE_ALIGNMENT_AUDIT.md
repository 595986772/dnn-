# Core Module Alignment Audit

| Chapter | Historical Feedback GRU | Selective Risk Gate | SLA/p99-oriented SAC | Action semantics |
|---|---|---|---|---|
| Introduction | Core contribution | Core contribution | Core contribution | Secondary backbone |
| Related Work | Partial-observability gap | Risk-boundary gap | SLA/tail gap | Not a standalone gap |
| System Model | Formalized by (6a) | Physical set separated from Gate | Objective (17) | Joint action defined |
| Method | Section 4.2 | Section 4.3 + Algorithm 1 | Section 4.4 + Algorithm 2 | Section 4.4.1 |
| Experiments | w/o GRU and joint ablation | behavior + ablation | overall metrics; partial sensitivity | Plain/architecture evidence only |
