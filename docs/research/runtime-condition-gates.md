# 运行时条件门控：实现、证据与边界

记录日期：2026-11-15。本说明对应新增的条件门控实现，不修改既有实验的源码身份或历史结论。CPU 反例验证及本次 A100 云端闭环已经完成：[实测报告](2026-11-15-runtime-condition-results.md)记录两项生成、一项排队取消、下载与最终释放，历史检查 13 pass，新增条件检查 7 pass / 2 unknown；未在 GPU 上触发的 unknown/conflict 分支仍保持未知。

目标是让“当前可以做什么”成为有责任主体、身份和证据的判断。判断允许发出执行邀约、判断容器接受任务、判断取得 GPU、判断阶段成功，是不同事件。任一门控通过只授权对应的下一步，不保证后续计算、文件系统或外部平台必然成功。

## 1. 四值判定，避免把缺证据当成真

[condition_guard.py](../../runtime/minimax_h3/condition_guard.py) 的 evaluate_gate 是无 I/O、无锁的判定函数。每个 gate 声明非空且不重复的必要条件集合，最多 64 项；仅当每个声明条件的状态都是 true，allowed 才为 true。缺少条件不能通过空集合取得“全真”。

| 状态 | 含义 | 原始观察例子 | 允许该条件通过 |
| --- | --- | --- | --- |
| true | 有肯定证据，当前输入未提供相反证据 | 原生布尔 True | 是 |
| false | 有否定证据 | 原生布尔 False | 否 |
| unknown | 缺少可用证据或数据类型不符 | 缺字段、None、数字 1、字符串 “true” | 否 |
| conflict | 同一命题存在互相矛盾的证据 | 同一命题的见证列表 [True, False] | 否 |

见证列表用于同一个命题：True 与缺失观察并列仍有肯定见证，True 与 False 并列为 conflict。不同必要义务必须使用不同条件名；不能把“模型已加载”和“前阶段已完成”装入同一个见证列表，使其中一个 True 替另一个缺失条件通过。管理器对当前/缓存能力观察还要求两个输入都是布尔值，防止缺失的新观察借用旧 True。

否定操作也保留不确定性：unknown 的否定仍为 unknown，conflict 的否定仍为 conflict。没有证据说明“请求取消”，不等于已经证明“未请求取消”。

每份回执包含 schema_version、gate_version、gate_id、decision_id、observed_at、monotonic_ns、context、conditions、allowed、blocked。时间戳是这次判定的时间，不把旧远程观察自动刷新为当前事实。context 是有大小与嵌套上限的有限 JSON 副本。

## 2. 三个原子域，接收端负责最终复验

| 决策所有者 | 原子范围和真实作用 | 跨域边界 |
| --- | --- | --- |
| 主事件管理器 | _control_lock 仲裁派发与取消；状态锁和数据库事务内检查队列、尝试、实例、观察新鲜度、进程及能力；记录 manager.dispatch 回执后提交执行邀约身份。 | 远程状态只允许发出邀约。网络传输期间模型可能变化，容器必须重新判断。当前有界私有 RPC 仍位于控制仲裁范围内，不能声称所有网络等待都已移出控制锁。 |
| 模型容器 | execute 在模型锁内复验新接纳条件；每次 _pipeline_stage 在同一锁内复验当前实例/尝试、能力、前序成功、取消与租约，并先持久化回执和阶段开始事件。 | 等待预算和实际预处理、GPU 推理、编码校验在模型锁外。准入判定不是覆盖整个计算时段的全局锁；已开始的 GPU 工作在安全边界处理取消。 |
| 阶段预算 | PipelineBudget 的 Condition 锁内同时检查预算事实并授予/回收 reservation 和 lease；取消账本更新与授权属于同一预算域。 | 不在预算锁内调用可能取得模型锁的任意回调。旧兼容回调在锁外采样；生产接入使用 Event，随后模型锁内再次复验。 |

来源分别为 [global_manager.py](../../runtime/minimax_h3/global_manager.py) 的 _dispatch_gate / _dispatch_next / cancel、[model_container.py](../../runtime/minimax_h3/model_container.py) 的 execute / _pipeline_stage，以及 [pipeline_scheduler.py](../../runtime/minimax_h3/pipeline_scheduler.py) 的 reserve / stage / finish。

模型锁内允许短暂查询预算见证，锁序是模型 → 预算；预算不得反向调用模型。等待资源时先释放模型锁，预算条件等待也释放预算锁，因此不会因为 A 等 GPU 而禁止 B 释放 CPU 或 GPU。

跨进程不存在共同内存锁，也没有把远程状态探测和远端 GPU 授权包装成一个事务。管理器发布意图、容器接纳、预算授予、模型实际阶段授权，分别承担自己的责任。邀约响应缺失时，管理器按原 job / attempt / instance 对账；只有明确 guarantees_not_accepted 才允许返回排队，不凭超时重复生成。

## 3. 具体门控与依赖

| 门控 | 必要证据范围 | 通过后发生的事 |
| --- | --- | --- |
| manager.dispatch | 管理器未关闭、任务仍 queued、尚未分配 attempt、当前模型身份一致、观察新鲜、模型进程存活、ready / worker_alive / accepting 明确、监督线程有效。 | 持久化本次派发意图，向对应模型实例发送一次执行邀约。 |
| pipeline.reserve | 请求估算在上限内、当前 RAM/磁盘工作集可容纳、宿主剩余内存与磁盘检查可确认、结果身份未占用。 | 建立该 attempt 的工作集预留。 |
| model.execute | 当前实例、有效请求、模型已加载、worker 存活、能力阶段有效、未 drain/stop、attempt/job 未冲突、产物路径未占用、生成选项有效、预留属于本预算和本 attempt。 | 持久化容器接纳事件并进入内部执行队列。 |
| pipeline.stage.cpu / gpu | 本 attempt 预留有效、预算取消状态与采样取消状态明确、该 attempt 无其他活动阶段、CPU 通道足够；GPU 还要求单 GPU 租约空闲。 | 原子授予带身份的阶段租约。 |
| model.stage.prepare | 本实例/attempt 所有权、接纳成功、仍为活动尝试、模型能力与 worker 有效、阶段未开始、取消明确未请求、预留与租约有效。 | 开始预处理。 |
| model.stage.gpu | 同上，并要求 prepare 的成功完成见证。 | 开始 GPU 推理，租约覆盖同步和 CPU 交接。 |
| model.stage.encode_validate | 同上，并要求 gpu 的成功完成见证。 | 开始 CPU 编码和完整产物校验。 |

每个模型阶段只有实际工作成功且 stage.finished 成功持久化后，才将对应 stage_completions 设为 True。仅有 stage.started、资源已下载、拿到租约或 Python 函数进入，都不能代替该完成证据。持久化失败不解锁下一阶段；实际设备工作已发生而回执缺失时，也不反推其已成功提交。

同一任务保留 prepare → gpu → encode_validate 的依赖。不同任务无全队列批次屏障：A 推理期间 B 可以预处理，A 编码期间 B 可以推理，只要共享预算允许。GPU 仍是单活动租约，GPU 的宿主 CPU 通道也计入同一 CPU 池。drain 阻止容器新接纳，已接纳任务正常排空；stop 仍等这些执行阶段结束再卸载模型。

上述条件集合是当前 H3 实现对所负责动作的有限契约，不是全部物理因素的穷尽。带宽尚未成为运行时准入维度；磁盘容量预留不是磁盘吞吐控制，RAM 工作集预留也不是进程 RSS 的精确测量。已有预算检查不证明避免所有 OOM，更不证明达到了理想并行物理下界或全局最优。

## 4. 观察、允许、完成、取消与失败分别表达

API 可以在模型加载完成前接受持久化排队；排队接纳不需要假装 GPU 已经可用。公网 API 就绪仍依赖当前 API、隧道和同实例端到端探测，不由“下载完成”替代。本次新增门控没有改变这些启动依赖。

| 观察或事件 | 可以得出的结论 | 不能替代的结论 |
| --- | --- | --- |
| allowed=True | 该责任域、该次决策允许下一步 | 后续必然成功、其他实例也可执行 |
| allowed=False，含 unknown/conflict | 本次证据不足或矛盾，不能开始对应动作 | 任务已取消、模型永远不可用 |
| cancel_requested | 已记录取消意图 | GPU 已立即停止、清理已经完成 |
| condition_gate_blocked 终态 | 执行尝试因条件证据被阻断，status=failed | 取消成功、生成完成 |
| 产物校验及终态提交成功 | 服务端拥有与身份、长度、哈希绑定的可用产物 | 客户端已经下载、内容质量已满足主观需求 |
| 阶段租约释放 | 当前阶段不再占用该预算租约 | 模型进程已退出、持久显存已归还 |
| worker 退出及清理回执 | 模型内部卸载/阶段排空的观察 | 所有者已确认进程退出、云 VM 已释放 |

阻断决策不永久禁止后续新的允许决策。分析应绑定 decision_id，不能把较早的一次拒绝套到后来已重新检查并获准的执行上。

模型结果保留 condition_gates 和 stage_completions；管理器持久化为 execution_condition_gates 和 stage_completions 后才确认接收结果。管理器记录 cancellation_requested，但不会把已报告的 failed 强行改成 cancelled。取消与清理也分别处理，未确认清理的任务仍保留所有权。

## 5. 查询与审计

| 查询/事件位置 | 关键字段和用途 |
| --- | --- |
| 管理器 snapshot | condition_gates 展示最近判断；具体 job 的 condition_gate 绑定最近派发判断。 |
| 管理器事件 | condition.checked 携带 receipt；job.dispatch_intent 携带 condition_decision_id。相同阻断轮询会合并，不能把缺少重复行解释为未检查。 |
| 容器 status | last_condition_gate；活动 attempts 的 condition_gates、stage_completions；终态 results 保留同类证据。 |
| 模型 condition.gate | 完整 receipt；context 绑定 instance_id、job_id、attempt_id、source_file、source_sha256，阶段判断还关联预算与租约。 |
| 模型 reservation.acquired / reservation.blocked | reservation_witness 与 reservation_receipt，保存当次预留证据，避免后续查询覆盖前一个任务的判断。 |
| 模型 execution.accepted / stage.started / stage.finished | condition_decision_id 连接授权与实际动作；阶段事件还有 lease_id、budget_instance_id、完整 lease_witness。 |
| 模型 lease.acquired / lease.released | 保留预算 grant_receipt、授权身份与单调时钟起止；释放见证 owned=False。 |
| 预算 snapshot | budget_instance_id、capacity_version、last_gate_receipts、active_leases、当前占用以及 lease_history。 |
| model.worker_exited | 全部阶段线程结束后记录 budget_snapshot，使最后一次释放不依赖定时采样恰好命中。 |

公共单任务查询现在也由 [server.py](../../runtime/minimax_h3/server.py) 的 public_condition_fields / public_gate_receipt 投射四类字段：condition_gate、execution_condition_gates、stage_completions、cancellation_requested。它们来自管理器持久任务记录，因此模型确认并清除执行结果后仍可查询，不依赖该模型结果继续留在内存。字段尚不存在时不补造对应历史；未知或冲突条件保留其原状态。

该接口采用固定允许字段及类型校验，不公开任意 receipt/context JSON。门控限定已知 gate_id 和必要条件名；context 仅保留明确允许的实例/任务/尝试/预算/租约 ID、源码文件名与 SHA，以及指定枚举和布尔值。输入参数、本机路径、任意额外上下文及密钥不进入这一公开投射，随后仍执行既有秘密值过滤。阶段完成仅发布布尔或 null，取消事实也不接受字符串、数字等伪布尔。若上游声称 allowed=True，却缺少全部必要真条件，公开 allowed 为 null，不能据此显示已获许可。公开字段是安全投射，原始完整审计回执仍由所有者保存，不能要求二者逐字节相同。

对应 [公共投射测试](../../runtime/minimax_h3/test_server_condition_projection.py) 使用真实管理器的结果确认路径，验证模型结果被清除后的 HTTP 查询、unknown/conflict 保留、私有字段过滤和实例边界。本轮还通过受管 API 停止/启动替换接口进程，保持管理器、模型及已有任务身份，随后实际公共任务查询已观察到 condition_gate；本次接口源码变化单独记账，不声称整个生命周期源码始终不变。真实生成终态、客户端下载与最终释放是否完成仍以本轮独立验收记录为准。

source_sha256 定位源码字节，capacity_version 定位预算配置；二者都不是签名，也不能认证“这些字节一定真实执行”。核验需把回执、实际部署清单、实例身份、完整事件流与终结所有者观察一起比较。单调时钟时长应在可比的同一时钟域内计算，不能任意跨主机相减。

模型关键门控与阶段事件通过 flush / fsync 后推进。预算的内部历史本身是内存记录，模型将获得/释放见证投射为持久事件；不能只因为预算提供了查询接口就宣称日志耐久。进程或磁盘在投射前失效时，缺失部分必须保持未知，不能补造成功回执。

## 6. 有界历史不等于完整历史

预算 lease_history 最多保留 512 个端点事件，每次获取与释放各占一个事件。count 是实例生命周期累计事件数，dropped 表示本次快照未包含的端点数；普通status使用included=False而不复制历史，不能把该省略视为磁盘数据丢失。事件sequence贯穿同一预算实例连续递增。0.5.0将每个端点先持久化至独立SQLite日志，内存尾只用于诊断。

释放见证查询最多保留最近 256 份；按 attempt / kind 查询的条件回执最多保留最近 128 项。最近值不能代替完整历史。0.4最终worker事件仅有当时有界尾；0.5.0另有lease_journal的最终count/tail见证，收集器通过固定截止分页恢复完整端点，磁盘读不占主状态锁，重启只恢复历史读取而不恢复执行权。

模型持久lease页须验证实例绑定、请求游标/上限、链摘要、连续性、终结尾序号和所有者退出观察。同内容重叠页可去重，冲突页失败，缺页unknown。仅有history_complete=False的尾无法证明完整；但已验证分页与最终count/tail共同覆盖时可恢复完整CPU/GPU账本。当前GPU无占用仍不足以证明全程无超借。写盘失败锁存，释放证据缺失不得报告cleanup成功；全RAM/磁盘账本仍未完整。

独立分析入口为 [analyze_condition_gates.py](../../research/analyze_condition_gates.py)，它补充条件与预算审计，不改变旧 [analyze_trace.py](../../research/analyze_trace.py) 的历史检查定义。

## 7. 已验证与待验证

CPU 验证使用注入的轻量后端与线程屏障，不下载权重、不执行 CUDA、不生成真实媒体。[test_model_condition_gates.py](../../runtime/minimax_h3/test_model_condition_gates.py) 的 29 项测试覆盖：缺失/伪布尔/冲突加载状态、估算期间 drain、前序缺失/失败/冲突、未知取消、实例变化、缺失或错属租约、审计失败、drain 排空、幂等重查以及终结历史完整性。

其中终结测试运行两个任务，确认六个阶段租约的十二个获取/释放端点连续、活动租约为空、最后释放早于 worker 终结记录。与既有模型容器和流水线测试联合运行时为 104 passed、1 skipped；跳过项是本地 OS 账户不能创建符号链接。最后仅补充错误分类后，新 29 项已重新通过。这些数字限定于该轮本地测试，不代表未来任意改动或真实 GPU 情况。

本次运行时改动已在 A100 原生 BF16 与上游 FP32 参数例外下完成公网接纳、参考上传、多任务排队、一项排队取消、两项新内容生成与下载、最终清理及平台释放。部署源码、已加载参数、任务门控和租约分别有证据；精度审计只覆盖 loaded parameters。期间曾通过管理器受管重启一次 API 补齐公共字段投射，模型和管理器未重启，接口前后源码身份单独记录。完整结果见[本轮实测报告](2026-11-15-runtime-condition-results.md)。

新增条件分析有 7 pass / 0 fail / 2 unknown：unknown/conflict 阻断路径在真实 GPU 运行中未触发，CPU 反例通过不将其改写为 GPU 已证。阶段重叠与正确租约账本也不构成物理最优或全部故障组合的证明。

当前没有证明全部故障组合、任意插件/模型、跨主机调度、真实带宽最优分配或对设备失效的事务回滚。该实现提供的是可执行、可查询、可反驳的局部条件契约，以及继续用真实实验检验它们的证据结构。

## 8. 八个有限研究模型与运行时的对应范围

[condition-models.json](../../research/condition-models.json) 的八个模型用于有限条件搜索。它们的聚合原子不是生产字段的一比一解释器；生产代码按各所有者的具体身份、输入和副作用边界展开检查。本次将三个执行准入域接入统一 evaluate_gate，并未实现“读取八个研究模型即可完整驱动生产服务”的运行时解释器。

| 有限模型 ID | 实际所有者与对应实现 | 当前落实方式与边界 |
| --- | --- | --- |
| queue_admission | [server.py](../../runtime/minimax_h3/server.py) 的鉴权、RequestGuard、submit；[global_manager.py](../../runtime/minimax_h3/global_manager.py) 的 submit。 | 沿用各所有者已有校验：身份与参数、上传数量/字节/文件提交、输入归属、接纳开关、持久队列存储和幂等冲突。没有把模型下载或模型 ready 加为排队前提；本次未改成统一 gate 回执。 |
| public_api_transport | global_manager.py 的 probe_api、_transport、refresh、snapshot。 | 沿用现有严格实例与能力校验：探测 api_ready is True、隧道 connected is True、当前进程/意图 epoch、观察新鲜度与监督有效性共同构成 public_api_ready。该命题不要求模型 ready；不能与还包含模型能力的 public_ready 混用。本次未改成统一 gate 回执。 |
| manager_execution_offer | global_manager.py 的 _dispatch_gate / _dispatch_next。 | 新增统一 manager.dispatch 门控，本地仲裁和持久事务负责授权发出邀约。模型可以在传输期间改变能力，接收端必须复验；它不要求 GPU 此刻空闲，也不证明邀约已接收。 |
| container_new_attempt_admission | model_container.py 的 execute；pipeline_scheduler.py 的 reserve。 | 新增统一 model.execute 和 pipeline.reserve 门控，绑定新 attempt、模型能力、输入/产物冲突与工作集预留。既有 attempt 的幂等查询是独立路径，不能用重放授权再次开始计算。 |
| gpu_stage_eligibility | model_container.py 的 _pipeline_stage；pipeline_scheduler.py 的 stage。 | 新增统一 model.stage.gpu 与 pipeline.stage.gpu 门控，要求当前 attempt、prepare 成功、明确未取消及实际 GPU/宿主 CPU 授权。prepare 和 encode_validate 也采用自己的阶段门控，保持真实依赖与独立并行。研究原子 model_ready 在这里展开为已接纳任务的有效执行能力，不能直接套用关闭新接纳后的 ready/accepting 显示值来中断正常 drain。 |
| artifact_commit_eligibility | model_container.py 的 verify_artifact；global_manager.py 的 _consume_result / _verified_artifact。 | 沿用现有严格校验：结果 job/attempt/model 归属、cleanup_complete is True、产物路径/存在性/长度/SHA、full_decode_passed is True、取消策略和事务提交。服务端成功提交与客户端随后下载分别验收；本次未改成统一 gate 回执，也不宣称文件检查与未来访问是不可分割的事务。 |
| closed_lifetime_review | [analyze_trace.py](../../research/analyze_trace.py)、process_owner.py 及本轮独立平台观察。 | 属于事后闭环检查：终态回执、任务清理、服务退出、平台释放分别需要证据。不能前置到 can_start，否则要求尚未开始的生命周期先已经结束；进程退出也不能替代 VM 释放。 |
| illustrative_alternative_evidence | [objective-condition-sets.md](objective-condition-sets.md) 与有限集合搜索。 | 仅为研究反例，说明不同包含极小集合不一定等长，较长的极小集合不能因“非最短”而丢弃。它不是 H3 的生产证据替代规则，不参与任何实际执行授权。 |

表中的“沿用校验”只说明相应所有者已经执行具体检查，不表示它们也已获得统一四值回执或完整未知/冲突投射。源码、局部反例、有限搜索与真实云端轨迹仍是不同证据层；不能从八个模型的搜索结果推导八个目标已在生产运行中全部得到验证。
