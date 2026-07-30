# Paper Evidence Manifest

## Main figures

| Evidence | Source |
|---|---|
| Static 60-s main figure | `D:\python_project\实验版9\figures\publication_main_long60s_fcfs_3metric_v2_ieee\figure4_static_load_three_metrics.png` |
| Random-switch 60-s main figure | `D:\python_project\实验版9\figures\publication_main_long60s_fcfs_3metric_v2_ieee\figure5_random_switch_three_metrics.png` |
| Per-DNN 60-s figure | `D:\python_project\实验版9\figures\publication_per_dnn_long60s_canonical_v2_ieee\figure_per_dnn_60s_canonical.png` |
| Gate 60-s mechanism figure | `D:\python_project\实验版9\figures\publication_gate_mechanism_long60s_v1_ieee\figure_gate_trigger_random_switch_60s_3panel_5seed.png` |
| Joint History-GRU/Gate ablation | `D:\python_project\实验版9\results\paper_evidence\ablation_joint_no_history_no_gate_default_uplink_long60s_20260730\figure_joint_ablation_publication_safe.png` |

## Main source data

- `D:\python_project\实验版9\figures\publication_main_long60s_fcfs_3metric_v2_ieee\figure4_source_data.csv`
- `D:\python_project\实验版9\figures\publication_main_long60s_fcfs_3metric_v2_ieee\figure5_source_data.csv`
- `D:\python_project\实验版9\figures\publication_per_dnn_long60s_canonical_v2_ieee\figure_per_dnn_source_data.csv`
- `D:\python_project\实验版9\figures\publication_gate_mechanism_long60s_v1_ieee\gate_trigger_random_switch_60s_5seed_statistics.csv`
- `D:\python_project\实验版9\results\paper_evidence\ablation_core_mainaligned_default_uplink_long60s_20260729\ablation_long60s_macro.csv`
- `D:\python_project\实验版9\results\paper_evidence\ablation_joint_no_history_no_gate_default_uplink_long60s_20260730\joint_ablation_macro.csv`
- `D:\python_project\实验版9\results\paper_evidence\supplements\overhead\scheduler_overhead.csv`

## Protocol boundary

- One trained checkpoint (`UnifiedRiskBudget.pt`, train seed 0).
- Five paired environment/evaluation seeds per main condition.
- 60-s main horizon with 1-s warm-up.
- Default action-independent uplink is included only in deployment-aware metrics.
- Energy is profile-estimated execution energy, not whole-system metered energy.
