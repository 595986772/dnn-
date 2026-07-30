# 5 实验评估

本章 Word 版本包含：

- 5.1 实验设置：硬件/profile、环境、训练协议、baseline 公平性；
- 5.2 静态八档负载与 random-switch 主结果；
- 5.3 分 DNN 结果；
- 5.4 Gate 机制诊断；
- 5.5 History-GRU / Gate / 联合去除 / Plain SAC 消融；
- 5.6 决策开销、deadline/干扰敏感性与局限。

## 可复算主结论

- 静态宏平均：SLA 违约率 33.75%，相对最强违约率基线降低 18.8%；p99 92.68 ms，相对最强 p99 基线降低 33.4%。
- Random-switch：SLA 违约率 31.32%，相对最强违约率基线降低 10.0%；p99 95.84 ms，相对最强 p99 基线降低 8.7%。
- 两类场景的 profile-estimated energy/request 均低于 0.109587 J 预算，但不是最低能耗。
- 上述结果来自一个训练 checkpoint 和每条件 5 个 paired evaluation seeds，不是五次独立训练。
