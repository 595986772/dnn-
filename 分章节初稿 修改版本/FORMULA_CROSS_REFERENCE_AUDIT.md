# Formula Cross-Reference Audit

## Chapter 3: (1)–(17)

| No. | Definition | Main downstream references |
|---|---|---|
| (1) | Edge-ingress request tuple | (2), (10)–(15) |
| (2) | Absolute deadline and slack | Sec. 3.2, Sec. 4.2–4.5 |
| (3) | Piecewise Poisson arrivals | (6), (18) |
| (4) | Per-DNN FIFO queue | (5), (6), Risk Gate |
| (5) | Physical action set | (6), (17), (21), Algorithms 1–2 |
| (6) | Queue evolution | Event transition and tail risk |
| (7) | Baseline service profile | (8), (9), (19), (22) |
| (8) | Hidden slowdown and completion time | (10), (11), feedback ratios |
| (9) | Profile-estimated batch energy | (16), (17), (27) |
| (10) | Queue/batching wait | (11) |
| (11) | Edge-side latency | (12)–(15) |
| (12) | Deployment-aware latency | (13)–(15), evaluation only |
| (13) | Request violation indicator | (14), (27), (28) |
| (14) | Cohort model-level violation rate | (17), (28) |
| (15) | Empirical p99 | Evaluation only; not a training reward |
| (16) | Estimated energy/request | (17), (27) |
| (17) | Soft-constrained control objective | Chapter 4 approximation |

## Chapter 4: (18)–(31)

| No. | Definition | Main downstream references |
|---|---|---|
| (18) | Revealed-arrival EWMA | (21), (26), Algorithm 1 |
| (19) | Completed-feedback runtime correction | (21), Algorithm 1 |
| (20) | GRU history and Actor/Critic context | (23)–(25), (29)–(31) |
| (21) | Nonempty selective Risk Gate | (24), (29), (31) |
| (22) | Action semantic vector | (23), (25) |
| (23) | Base plus action-residual Actor score | (24) |
| (24) | Masked policy | (29), (31), deployment |
| (25) | Action-aligned Twin Critics | (29)–(31) |
| (26) | Physical-time tail-risk surrogate | (27), replay |
| (27) | Reconstructed risk-budget reward | (29) |
| (28) | Per-model SLA dual update | (27), Algorithm 2 |
| (29) | Physical-time n-step soft target | (30) |
| (30) | Twin Critic MSE | Algorithm 2 |
| (31) | Discrete SAC Actor objective | Algorithm 2 |

## Checks

- Formula ranges are continuous and non-overlapping.
- Chapter 4 does not redefine the physical action set; it references (5).
- `p99` is defined only as an empirical evaluation metric in (15).
- Deployment-aware uplink `U_i` appears only in (12) and evaluation explanations.
- Runtime slowdown and hidden interference are never written as policy inputs.
- Every new symbol introduced by a formula is explained immediately below it or in Table 3.1.
