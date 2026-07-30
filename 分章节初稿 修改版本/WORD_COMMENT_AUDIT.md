# Word Comment Audit

The four clean revised documents were preserved. Separate commented copies were created from those clean files.

## Introduction_中文修订稿_v5_四章对齐版_批注版.docx

- Comments: 14
- Source: `D:\小论文\分章节初稿 修改版本\Introduction_中文修订稿_v5_四章对齐版.docx`
- Output: `D:\小论文\分章节初稿 修改版本\Introduction_中文修订稿_v5_四章对齐版_批注版.docx`

- `paragraph` `1 引言`: 【整体重组】本章由原稿重组为 8 个递进正文段落和 4 条贡献，并统一方法名、研究边界和实验声明口径。
- `paragraph` `面向智能交通、移动机器人和现场视觉分析等应用，DNN 推理结果只有在规定时间内返回才具有服务价值。将推理服务部署在靠近数据源的网络边缘，能够减少对远程云中心的依赖，并为实时感知与决策提供邻近算力。然而，边缘设备的资源规模有限，请求又会持续到`: 【压缩重写】合并原“云—边缘背景”和“实时 DNN 推理价值”两段，删除泛化云边科普及无线优化暗示，直接从请求级时效风险切入。
- `paragraph` `实际边缘推理平台通常同时承载多种 DNN 服务。不同模型具有不同的计算开销、到达过程和服务级目标（Service-Level Objective, SLO）；本文用模型级请求 deadline 表示该目标，并将超时完成记为一次服务级协议（S`: 【内容统一】重写多 DNN、异构执行设备、模型级 deadline/SLO/SLA 与 profile-estimated execution energy 的定义，使术语与第三章一致。
- `paragraph` `动态组批使上述问题进一步转化为紧密耦合的在线联合决策。等待更多同类请求可以形成较大 batch，提升硬件并行度并分摊执行开销，但等待本身会消耗 deadline 裕量；立即派发小 batch 可缩短排队时间，却可能降低吞吐并增加并发竞争。调`: 【段落合并】合并原动态组批段和重复的联合决策段，集中说明 queue–device–batch–WAIT 的耦合关系及当前动作对后续队列和设备槽位的影响。
- `paragraph` `这一联合决策必须在持续变化的运行条件下在线完成。短时间内的请求到达强度和模型组成可能发生波动，共置 batch 之间的资源竞争也会使实际执行时间偏离孤立运行时的基准水平 [2], [3], [10], [11]。调度器无法预先获知未来请求，`: 【信息边界】补充时变到达、模型组成变化与隐藏运行时偏移，并明确策略只能使用当前状态、已揭示到达和已完成 batch 反馈。
- `paragraph` `本文聚焦请求到达边缘入口后的在线服务调度。每个请求的 DNN 类型和模型级 deadline 由应用预先给定，调度器不改变模型、不执行网络切分、输入压缩或精度调整，也不分配无线资源。系统按照 DNN 类型维护 FIFO 队列，并由一个集中式`: 【新增研究边界】明确核心系统从 edge ingress 开始，不涉及模型切分、输入压缩、精度调整、无线资源分配或端边云卸载；客户端上行仅作为动作无关评估附加项。
- `paragraph` `现有工作已经从动态 batching、并发控制、异构执行、deadline-aware queue ordering 和学习式调度等角度改善边缘 DNN serving，因此本文不把采用 DRL 或 batching 本身视为创新。仍待解决`: 【研究空缺重写】将原逐篇论文罗列改为三个耦合缺口：结构化联合动作、已揭示反馈适应，以及模型级 SLA、尾部风险与估计能耗预算的统一控制；同时明确 batching 和 DRL 本身不是创新。
- `paragraph` `为解决上述问题，本文提出 RiskBudget-SAC，一种面向异构多 DNN 边缘推理服务的事件驱动在线调度框架。其一，调度器不把离散动作仅视为互不相关的编号，而是显式编码候选动作对应的 DNN 队列、执行设备、batch size、WA`: 【方法概览收敛】统一方法名为 RiskBudget-SAC，只保留“动作语义评分”和“基于已揭示反馈的风险控制”两个核心设计；GRU、EWMA、replay、n-step 等降为配套实现。
- `paragraph` `总体而言，本文研究 edge-ingress 多 DNN 服务中的队列选择、设备路由、动态组批与等待控制，目标是在预定义的 profile-estimated 单位请求能耗预算内降低请求级 SLA 违约率和 p99 时延。本文不将经验 p9`: 【目标收紧】将目标统一为在 profile-estimated 单位请求能耗预算内降低请求级 SLA 违约率和 p99；明确经验 p99 通过连续 tail-risk surrogate 间接改善，而非直接优化。
- `paragraph` `问题与系统建模。 本文构建一个面向 edge-ingress 多 DNN 服务的事件驱动联合调度模型，在每个事件上统一选择 DNN 队列、异构执行设备、batch size 或 WAIT，并显式描述模型级 deadline、设备并发槽位、时`: 【贡献 1 重写】贡献改为 edge-ingress 事件驱动的 queue–device–batch–WAIT 联合调度模型，并与第三章式 (5)–(17) 对齐。
- `paragraph` `动作语义对齐的 SAC 调度器。 本文设计 action-semantic Actor 与 action-aligned twin Critics，将候选动作的模型、设备、batch 和基准服务属性与其局部队列、设备状态对齐后逐动作评分，以`: 【贡献 2 重写】仅强调 action-semantic Actor 和 action-aligned Twin Critics 对动作共享结构的利用，不把 SAC 或 Twin Critic 本身表述为新理论。
- `paragraph` `基于已揭示反馈的风险控制。 本文利用在线到达估计和已完成 batch 反馈构造选择性 Risk Gate，并通过模型级 SLA 对偶、尾风险代理和估计能耗预算协同训练策略。历史反馈编码、tail-focused replay 与物理时间 n`: 【贡献 3 重写】将 Risk Gate、模型级 SLA 对偶、尾风险代理和估计能耗预算组织为一项整体机制；历史编码和 replay 等不再单列为创新。
- `paragraph` `严格边界下的实验验证。 在一个冻结的 400-episode 训练检查点上，本文使用每条件 5 个 paired evaluation seeds，并在默认动作无关上行配置下对测试条件等权宏平均。相对各指标对应的最强 baseline，Ri`: 【贡献 4 校准】补充冻结检查点、paired evaluation seeds、动作无关上行、宏平均和“各指标对应最强 baseline”等边界，避免把环境评估重复误写成多训练种子或全场景全面占优。
- `paragraph` `参考文献`: 【引用核对】参考文献条目沿用原稿，未新增或改写文献事实；正文引用位置按重组后的论证顺序重新对齐。

## Related_Work_中文修订稿_v4_四章对齐版_批注版.docx

- Comments: 16
- Source: `D:\小论文\分章节初稿 修改版本\Related_Work_中文修订稿_v4_四章对齐版.docx`
- Output: `D:\小论文\分章节初稿 修改版本\Related_Work_中文修订稿_v4_四章对齐版_批注版.docx`

- `paragraph` `2 相关工作`: 【整体重组】相关工作由原三节重组为四节，并按研究对象从多 DNN serving、学习式风险调度过渡到相邻 MEC/offloading，最后独立收束研究空缺。
- `paragraph` `本节围绕异构多 DNN 服务与动态组批、学习式联合调度与风险控制，以及相邻的 MEC 卸载与协同推理研究展开，并从决策范围、可观测反馈和风险目标三个方面说明本文的研究边界。`: 【组织逻辑重写】章节比较维度统一为决策范围、可观测反馈和风险目标，避免按论文或算法类型平铺罗列。
- `paragraph` `2.1 异构多 DNN 服务与动态组批`: 【小节重组】将 dynamic batching、异构执行、并发控制和共置干扰文献集中到本节。
- `paragraph` `HarmonyBatch 面向具有不同 SLO 的 DNN 请求，在异构 CPU/GPU serverless functions 上联合完成请求分组与资源配置 [19]。ELTO 使用实测性能数据和回归预测在线调整动态组批参数，以权衡推理`: 【文献前移与压缩】将 HarmonyBatch、ELTO 和 BCEdge 从原分散位置移入动态组批主线；保留原参考文献编号和贡献事实，删除无关实现细节。
- `paragraph` `在执行机制方面，Miriam 通过弹性 GPU kernel 协调实时多 DNN 推理 [2]。OctopInf 面向动态边缘视频分析，将工作负载感知的跨设备分配、动态 batching 和共置调度结合起来 [3]。Ling 等研究 CPU`: 【代表工作筛选】只保留与异构执行、跨设备分配及共置运行直接相关的代表性研究，避免把多个弱相关工作强行糅合为一句结论。
- `paragraph` `上述研究已经覆盖动态 batching、异构执行和共置干扰中的多个关键环节，但这些能力并不自动等价于本文关注的事件级结构化联合动作。本文要求调度器在同一决策中选择 DNN 队列、执行设备、batch size 或 WAIT，并在缺少未来负载`: 【边界补充】新增小节收束，说明已有能力并不自动等价于单事件的 queue–device–batch–WAIT 联合选择和 revealed-feedback 风险控制。
- `paragraph` `2.2 学习式联合调度、SLA 与尾部风险`: 【小节重组】集中讨论学习式 batching、deadline/SLA 与尾部性能；明确 DRL、SAC、性能预测和紧迫度感知均属于已有技术。
- `paragraph` `强化学习可通过环境交互学习当前调度动作对后续队列、吞吐和截止期限风险的长期影响。BCEdge 采用最大熵深度强化学习联合控制 batch size 与并发实例数 [5]。Coinf 结合性能预测识别紧急任务，并利用深度强化学习选择 QoS `: 【论证重写】将 BCEdge、Coinf 和离散 SAC batching 放入同一学习式调度脉络，并删除把“使用强化学习”直接当作创新的表述。
- `paragraph` `面向 deadline 和尾部性能，EdgeServing 协调多个 DNN 队列的服务顺序，以降低 deadline 违约风险 [9]。TailGuard 针对用户服务的尾部时延 SLO 设计任务级调度机制 [22]。SHEPHERD 对`: 【风险目标补充】将 EdgeServing、TailGuard 和生产负载观察用于说明 deadline awareness、尾部目标与动态适应的既有基础，而非宣称这些问题无人研究。
- `paragraph` `本文关注的是上述能力尚未完全覆盖的组合边界：在单个事件上联合选择 queue、device、batch 与 WAIT；显式表示候选动作语义，并使价值网络读取与该动作对应的局部状态；仅依据已揭示到达和完成反馈适应隐藏运行变化；同时区分模型级 `: 【精确差异】把本文差异限定为结构化联合动作、动作语义与局部状态对齐、已揭示反馈适应，以及模型级 SLA/尾风险/估计能耗预算的组合。
- `paragraph` `2.3 相邻的 MEC 卸载与协同推理`: 【研究边界调整】将 MEC 卸载、DAG、切分、压缩、DVFS 和无线资源工作压缩为相邻研究，避免与 edge-ingress serving 混为同一问题。
- `paragraph` `传统 MEC 研究主要决定任务在终端、边缘节点和云端之间的执行位置，并常与无线资源、计算资源或能耗控制联合优化 [12]。强化学习进一步被用于适应动态信道、任务依赖和多工作负载变化 [13]。例如，Sharma 等将具有依赖关系的任务表示为`: 【内容压缩】保留代表性强化学习卸载工作及其决策变量，只用于说明其动作通常是执行位置或资源分配，而非持续服务队列中的 route–batch–WAIT。
- `paragraph` `协同推理研究则进一步考虑 DNN 模型结构与硬件特性。DVFO 联合选择 DNN 切分位置及 CPU、GPU 和内存的动态电压频率，以降低边云协同推理的时延与能耗 [17]。Qian 等采用循环深度强化学习联合配置无线资源、卸载比例与边缘侧`: 【边界明确】说明这些工作同时优化通信、切分、压缩或精度；本文从请求到达 edge ingress 后开始，不控制这些变量。
- `paragraph` `2.4 研究空缺总结`: 【新增独立小节】用一段精确 gap summary 将第一章贡献、第三章定义和第四章方法逐项对应。
- `paragraph` `综上，现有研究已分别验证动态 batching、并发控制、deadline-aware queue ordering、性能预测和学习式调度的有效性，因而本文不把使用 SAC、GRU 或 batching 本身视为创新。本文关注的尚未充分解决`: 【结论替换】替换原泛化总结，明确本文不把 SAC、GRU 或 batching 本身作为创新，并限定信息边界与软风险目标。
- `paragraph` `参考文献`: 【引用保持】22 条参考文献及其编号、作者、年份和出版信息沿用原稿；未补入无法核实的文献或更改原文献贡献。

## System_Model_中文初稿_v8_四章对齐版_批注版.docx

- Comments: 30
- Source: `D:\小论文\分章节初稿 修改版本\System_Model_中文初稿_v8_四章对齐版.docx`
- Output: `D:\小论文\分章节初稿 修改版本\System_Model_中文初稿_v8_四章对齐版_批注版.docx`

- `paragraph` `3 系统模型与问题表述`: 【整体重建】核心系统改为从 edge ingress 开始，公式统一为 (1)–(17)；无线资源控制从核心模型移除，动作无关上行仅作为 deployment-aware 评价叠加项。
- `paragraph` `本文研究请求到达边缘入口后的集中式在线多 DNN 推理服务。请求按 DNN 类型进入独立 FIFO 队列，调度器在请求到达、batch 完成或等待边界等事件上作出决策，并统一管理多个执行设备。本章依次定义服务架构与请求、已揭示到达和事件级动`: 【章节主线重写】按请求/架构、已揭示到达与动作、基准服务与隐藏变化、双时延口径和软约束目标重新组织。
- `paragraph` `设 为系统承载的 DNN 服务集合， 为边缘入口统一管理的执行设备集合。系统为每个 DNN 维护一条 FIFO 队列。设备 可并发运行至多 个 batch，DNN 在设备 上允许采用的 batch size 集合记为 。每个请求只对应一种 `: 【对象与约束统一】执行资源统一称为“执行设备”并用 d 索引；明确请求不可拆分、同 DNN 组 batch、派发后不迁移且不抢占。
- `paragraph` `表 3.1 汇总了后续建模使用的集合、状态和性能参数。核心系统从请求到达 edge ingress 开始；与客户端网络有关的上行只在 deployment-aware 评价中以动作无关附加时延表示，不作为调度器的控制变量。基准服务特性用于区`: 【参数表重建】删除 Shannon 速率、发射功率和信道控制变量；加入 edge arrival、决策事件、基准服务特性、已揭示反馈、双时延口径和 profile-estimated energy。
- `paragraph` `为统一描述平稳到达、负载切换和模型组成漂移，本文采用分段时变泊松过程生成不同 DNN 的请求。模型 m 在区间 [t_1,t_2) 内的到达数满足`: 【到达模型校准】保留分段时变泊松族以统一静态、负载切换和模型比例漂移，并明确真实瞬时到达强度不作为策略输入。
- `paragraph` `系统在事件时刻 t_k 作出第 k 次决策。事件由请求到达、batch 完成或等待边界触发。联合动作 a_k 要么从 DNN m 的队首提取 b 个请求形成 batch 并派发到设备 d，要么在满足事件推进和紧迫性条件时执行 WAIT。物理`: 【动作过程重写】将原多个二元派发/WAIT 变量及约束收敛为事件级联合动作；WAIT 只有在系统可推进且不违反最晚安全等待边界时才可选。
- `paragraph` `为刻画不同 DNN、执行设备和 batch size 组合的固有服务差异，定义组合 (m,d,b) 的基准服务特性为`: 【新增基础参数层】显式定义 DNN–device–batch 的基准执行时延、单位请求基准能耗和基准服务率，使第三章参数与第四章动作语义一致。
- `paragraph` `运行阶段，共置 batch 和后台资源竞争可能改变设备提供给 batch 的瞬时服务速度。对已派发 batch j，记其模型、设备、batch size 和开始时刻为 m_j、d_j、b_j 和 σ_j，并以 Γ_j(t)≥1 表示不可被调`: 【运行变化抽象化】用不可观测、有界的减速因子和累计服务进度描述在线偏移；隐藏变量仅用于环境演化，不进入策略状态。
- `paragraph` `batch j 的执行能耗采用 profile-estimated 口径建模为`: 【能耗口径修正】将能耗统一为 profile-estimated batch execution energy，并明确不等同于整机实测总能耗，也不含通信、空闲和基础设施能耗。
- `paragraph` `3.4 Edge-Side 与 Deployment-Aware 指标及软约束目标`: 【评价口径拆分】将核心 edge-side latency 与附加客户端上行的 deployment-aware latency 分开定义和报告。
- `paragraph` `在评价口径 x∈{edge,dep} 下，请求级 SLA 违约指示量定义为`: 【统计口径修正】SLA 违约按 edge/dep 两种口径定义，并使用固定 arrival cohort；episode 结束后 drain，避免只统计提前完成请求。
- `paragraph` `式 (15) 的 P99^x 是实验评价指标，不被直接写入强化学习的即时奖励。第四章采用由当前队列和 deadline 裕量构造的连续尾部风险代理，以提供更及时的训练信号。`: 【目标边界澄清】经验 p99 仅作为评价指标，训练使用连续 tail-risk surrogate，不再声称直接优化经验分位数。
- `paragraph` `设 为满足信息边界的在线策略集合。以 表示策略 下的长期平均尾部风险代理，以 和 分别表示长期模型级违约率与 profile-estimated 单位请求能耗。本文采用如下软约束控制目标：`: 【问题重新定式化】删除原两阶段词典序 direct-p99 优化，改为在物理动作范围内，以模型级 SLA 和 profile-estimated energy 为软约束，最小化长期尾部风险代理。
- `formula` `(1)`: 【公式替换】请求元组改为 edge arrival、DNN 类型和相对 deadline；删除客户端生成时刻、输入大小等无线侧变量。
- `formula` `(2)`: 【公式改写】绝对截止时刻与 deadline slack 统一从 edge arrival 起算。
- `formula` `(3)`: 【公式保留小修】保留分段时变泊松到达，但补充“真实 λ 不进入策略状态”的信息边界。
- `formula` `(4)`: 【公式统一】FIFO 队列符号与新的请求、deadline 和 slack 记号对齐。
- `formula` `(5)`: 【公式合并】原多个 0/1 派发变量、WAIT 变量和可行性约束合并为一个事件级物理动作集合；WAIT 改为条件可行。
- `formula` `(6)`: 【公式改写】队列演化改用事件到达量和动作指示函数，不再依赖已删除的二元变量。
- `formula` `(7)`: 【新增公式】定义基准服务特性 P(m,d,b)，统一基础时延、单位请求基准能耗和基准服务率。
- `formula` `(8)`: 【公式抽象化】将原多条干扰/执行公式合并为有界隐藏减速下的累计服务进度与完成时间。
- `formula` `(9)`: 【公式重写】执行能耗改为 profile-estimated batch energy 加有界修正，不再表述为实际整机能耗。
- `formula` `(10)`: 【公式拆分】单独定义排队与组批等待时延。
- `formula` `(11)`: 【公式拆分】核心服务时延统一从 edge ingress 计时，仅包含等待和执行。
- `formula` `(12)`: 【新增评估叠加】deployment-aware 时延仅在 edge-side latency 上叠加动作无关上行 U_i；该项不进入训练或动作。
- `formula` `(13)`: 【公式扩展】请求级 SLA 违约增加 edge/dep 两种评价口径。
- `formula` `(14)`: 【统计口径改写】模型级违约率使用固定到达 cohort 作为分母，并配合 episode 后 drain。
- `formula` `(15)`: 【公式简化】删除辅助指示变量，只保留固定 cohort 上的经验 0.99 分位数，并限定为评价指标。
- `formula` `(16)`: 【公式口径统一】单位请求能耗改为 profile-estimated J/request，不含客户端通信、空闲和基础设施能耗。
- `formula` `(17)`: 【优化问题替换】由词典序 direct-p99 优化改为 tail-risk surrogate 最小化，并施加模型级 SLA、估计能耗和物理动作软约束；不构成确定性保证。

## Method_中文初稿_v9_四章对齐版_批注版.docx

- Comments: 33
- Source: `D:\小论文\分章节初稿 修改版本\Method_中文初稿_v9_四章对齐版.docx`
- Output: `D:\小论文\分章节初稿 修改版本\Method_中文初稿_v9_四章对齐版_批注版.docx`

- `paragraph` `4 RiskBudget-SAC 方法`: 【整体对齐】方法名统一为 RiskBudget-SAC，公式连续编号为 (18)–(31)，并按最终实现重新组织在线决策、训练和部署路径。
- `paragraph` `RiskBudget-SAC 按“已揭示信息建模、选择性风险筛选、动作语义评分和尾部风险预算训练”四个环节运行。在线阶段，调度器只使用当前队列与设备状态、已经发生的到达和已完成 batch 反馈；Risk Gate 先从第三章定义的物理动作`: 【总览重写】原多模块并列叙事收敛为两条主线：在线的 context→Risk Gate→action-semantic Actor，以及训练期的 tail-aware risk-budget SAC。
- `paragraph` `方法的两个核心设计分别对应第三章的两类困难：动作语义评分利用 DNN、设备、batch 和基准服务属性之间的共享结构，避免把有限动作库仅视为互不相关的编号；基于已揭示反馈的风险控制则利用在线到达估计和完成反馈，在负载压力或 deadline`: 【创新点收敛】只保留动作语义评分和基于已揭示反馈的风险控制两个核心设计；EWMA、GRU、Twin Critic、replay 和 n-step 作为配套实现。
- `paragraph` `调度器在每个事件上刷新即时状态 s_k，其中包含各 DNN 的队列与 deadline 裕量、设备占用和并发执行组成。为反映短时到达变化，对最近长度为 Δ 的观测窗口计数，并按模型使用 EWMA 更新在线到达率：`: 【状态表征校准】即时状态只包含队列、deadline 裕量、设备占用和并发组成；真实负载、未来请求和隐藏减速不进入输入。
- `paragraph` `执行时间估计以第三章的基准 profile 为起点，并由相同 DNN–设备组合的已完成 batch 反馈进行保守校正。设 μ^ρ_{m,d,k} 和 σ^ρ_{m,d,k} 分别为截至 t_k 观测到的执行时延比值的 EWMA 均值和标准差`: 【预测器源码对齐】完成时间校正改为已完成 batch 的 latency-ratio EWMA 均值、方差和不确定性裕量，而不是虚构神经预测器或 oracle。
- `paragraph` `令 表示 前最近 条已完成 batch 的反馈窗口。每条记录包含完成时间的新近程度、DNN 类型、执行设备、batch size，以及相对于基准 profile 的执行时延和能耗比值。GRU 按完成顺序编码该窗口，并与即时状态组成 Acto`: 【历史窗口校准】历史反馈窗口只列已完成 batch 可观测字段，GRU 与即时状态共同形成 Actor/Critic 上下文。
- `paragraph` `Risk Gate 的输入始终受第三章式 (5) 限定。它读取物理动作集合、当前可观测状态、在线到达率估计和以完成反馈校正的动作时延估计，并返回非空风险动作集合：`: 【职责边界重写】Risk Gate 直接引用第三章物理动作集合，只读取当前可观测状态、已揭示到达率和完成反馈校正的时延估计，并保证返回非空子集。
- `paragraph` `若紧迫队列不存在预测可及时完成的动作，Gate 不返回空集合，而是保留预计按时吞吐率较高的动作；若所有候选均预计超时，则退回到服务率较高的动作以继续排空积压。Gate 只构造允许集合，最终的 DNN、设备和 batch 决策仍由 Actor`: 【声明收紧】补充非空 fallback，并明确 Gate 只产生 selective mask，不替代 Actor、不负责对保留动作排序，也不提供 SLA 或安全保证。
- `paragraph` `对每个候选动作，RiskBudget-SAC 不仅保留动作编号，还编码其可共享语义。动作特征由 WAIT 标识、DNN one-hot、执行设备 one-hot、batch size 的双尺度表示和第三章式 (7) 的基准服务属性组成：`: 【动作表示重写】动作特征与第三章基准服务特性对齐，统一编码 WAIT、DNN、设备、batch 双尺度表示和基准服务属性。
- `paragraph` `Actor 分别编码上下文与动作语义。基础评分头根据当前上下文为有限动作库给出初始分数，动作条件残差头再利用共享动作表示进行校正：`: 【Actor 结构对齐】按最终源码写为基础评分头加动作条件残差头；省略内部逐元素交互细节，但保留真实的 base+residual 结构。
- `paragraph` `两个独立 Critic 在全局上下文与静态动作语义之外，还读取与候选动作对应的局部状态。对派发动作，该局部状态包含目标 DNN 的队列、队首 slack 与分布特征，以及目标设备的占用、并发组成和可观测干扰水平；WAIT 则使用相应的全局聚`: 【Critic 结构对齐】Critic 除全局上下文和动作语义外，显式读取与候选动作对应的队列、slack 和设备局部状态。
- `paragraph` `经验 p99 只有在收集完整请求 cohort 后才能计算，难以为每个事件提供及时训练信号。为此，本文根据当前队列积压与队首 slack 构造连续尾部风险水平，并对一个动作持续期间的风险进行梯形积分近似：`: 【训练目标校准】将经验 p99 从即时奖励中移除，改为按队列、slack 和真实动作持续时间构造连续 tail-risk surrogate。
- `paragraph` `每个物理时间 n-step 转移同时保存模型级违约计数、纳入 SLA 统计的请求数和 profile-estimated 能耗。Critic 更新时使用当前模型级对偶变量重构即时奖励：`: 【奖励重构】replay 保存违约计数、请求数、尾风险和估计能耗分量，并在更新时使用当前 SLA 对偶变量重构 risk-budget reward。
- `paragraph` `事件动作具有不同持续时间。令 为从事件 开始，最先达到预设物理时间窗、最大决策数或 episode 边界的聚合步数，并令 。物理时间 n-step soft target 为`: 【时间建模补充】由一步固定折扣改为按实际经过时间聚合的 physical-time n-step target，以区分 WAIT 和不同 batch 动作的持续时间。
- `paragraph` `Twin Critics 通过同一 target 进行均方误差回归：`: 【公式错误修正】Twin Critic MSE 改为两个平方误差相加；原稿中的相减会使第二个 Critic 的误差被错误奖励。
- `paragraph` `部署时移除 Twin Critics、经验回放、梯度更新和 SLA 对偶更新。系统仅保留已揭示到达率 EWMA、完成时间校正器、GRU 历史编码器、Risk Gate 和一次确定性 Actor 前向计算。每个事件先构造物理与风险动作集合，再`: 【新增部署说明】明确部署只保留到达率估计、完成时间校正、GRU、Risk Gate 和一次确定性 Actor 前向；移除 Critics、replay、梯度和 dual update，不使用候选搜索或 oracle 信息。
- `paragraph` `设当前物理动作数为 ，历史窗口长度为 ，且隐层宽度固定。到达统计和历史编码的开销分别随 DNN 数量与 线性增长；完成时间估计、Risk Gate 和动作评分均至多对有限动作库扫描一次，因此单事件决策复杂度关于 为线性量级。动作语义编码和局`: 【新增复杂度说明】给出关于有限动作库线性扫描的单事件复杂度，并说明动作语义和局部状态可向量化。
- `formula` `(18)`: 【公式改写】到达率 EWMA 只统计截至当前已经揭示的请求，并统一事件索引；不读取真实 ρ、model mix 或未来计数。
- `formula` `(19)`: 【新增源码一致公式】用完成 batch 的时延比 EWMA 和不确定性裕量校正基准执行时间。
- `formula` `(20)`: 【公式合并】GRU 历史编码与上下文拼接合并定义，避免把历史编码器单独包装为创新模块。
- `formula` `(21)`: 【Gate 公式重写】统一为非空 selective Risk Gate；输入受信息边界限制，且不重复定义第三章物理动作集合。
- `formula` `(22)`: 【动作语义统一】动作向量与第三章式 (7) 的基准服务属性对齐，并为 WAIT 使用独立标识。
- `formula` `(23)`: 【Actor 公式对齐】采用最终实现的 context encoder、action encoder、base logits 与 action-conditioned residual logits。
- `formula` `(24)`: 【策略约束统一】MaskedSoftmax 仅在 Risk Gate 保留的动作集合上归一化；训练采样、评估/部署单次 argmax。
- `formula` `(25)`: 【Critic 公式完善】Twin Critics 同时读取全局上下文、动作语义和动作相关局部状态，并在 target/Actor 更新中取较小值。
- `formula` `(26)`: 【新增公式】按最终代码定义队列/slack 风险水平及动作持续期间的梯形积分近似，不把经验 p99 直接作为 loss。
- `formula` `(27)`: 【奖励公式重写】统一组合 tail-risk、模型级 SLA 偏差和 profile-estimated energy budget hinge；历史转移可按当前对偶变量重构奖励。
- `formula` `(28)`: 【对偶更新限定】模型级 SLA 权重按 episode 违约率软更新；用于目标适应，不构成硬 SLA 保证。
- `formula` `(29)`: 【Bellman target 替换】由一步 target 改为按物理时间折扣的 N-step soft target，N 同时受时间窗、最大步数和 episode 边界限制。
- `formula` `(30)`: 【关键修正】Twin Critic 回归损失使用两个 MSE 之和；目标网络不反向传播，回放按场景、压力和尾风险平衡采样。
- `formula` `(31)`: 【Actor 目标完善】补全门控动作集合上的离散 SAC 最大熵目标，并使用两个在线 Critic 的较小预测值。
- `table` `Algorithm 1: 反馈校准的选择性 Risk Gate`: 【算法占位补全】用完整 Risk Gate 伪代码替换原占位，明确物理 mask、触发条件、压力/紧迫筛选、非空 fallback 和 WAIT 处理；算法只返回动作集合。
- `table` `Algorithm 2: 尾部风险预算 SAC 训练`: 【算法占位补全】用完整训练伪代码替换原占位，包含分量化 replay、风险奖励重构、physical-time n-step、Twin Critic/Actor/熵温度更新及模型级 SLA 对偶更新。
