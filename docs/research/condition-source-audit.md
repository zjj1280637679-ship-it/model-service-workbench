# 客观条件集合的源码来源核对

核对日期：2026-11-15。基线提交：44bf769d787b0c4870aaf376eb2e41de1b448665。本次只读检查拥有决策权的运行代码，编写本文件；没有修改运行时、申请 GPU 或追加实测。四个逻辑模块仍映射到当前 H3 专用实现，不能从本表推导任意模型或平台已经兼容。

这里的“必要”必须附目标、时刻、身份和策略：允许上传、持久化排队、发出执行邀约、接受执行尝试、取得 GPU、提交产物、客户端下载、归还进程、平台释放，是不同目标。某函数的检查列表只定义它当前负责的准入判断，不是自然界全部充分条件；即使检查全真，后续 I/O、设备、进程或外部平台仍可失败。

## 一、来源索引与可核对证据

| 编号 | 拥有决策的源文件与函数 | 能核对的命题及证据 |
| --- | --- | --- |
| S1 | [preflight.py](../../runtime/minimax_h3/preflight.py)：check_resources、cached_weight_budget | Linux/NVIDIA/A100、显存、MemAvailable、已验证可复用权重与剩余磁盘预算；报告中的阈值是项目策略，函数明确不保证无 OOM。 |
| S2 | [model_backend.py](../../runtime/minimax_h3/model_backend.py)：MiniMaxBackend.load、_require_loaded_components、_audit_parameter_precision | ref2va 必需组件实际存在；参数 materialized、类型、dtype、量化标记；审计范围为 loaded_parameters_only。 |
| S3 | [model_container.py](../../runtime/minimax_h3/model_container.py)：status、_request、execute、_pipeline_stage、_run_pipeline_attempt、verify_artifact、ack_results、_work | 当前实例的加载/worker/accepting、请求身份与媒体、原子接纳、阶段进入、同步交接、产物完整解码、结果保留与清理。对应 status、execution/stage 事件、结果回执。 |
| S4 | [pipeline_scheduler.py](../../runtime/minimax_h3/pipeline_scheduler.py)：detect_cpu_capacity、PipelineBudget.reserve/stage/finish/acknowledge/snapshot | affinity/cgroup CPU 容量、RAM/磁盘估算账本、宿主 CPU 通道、GPU 单租约；snapshot 明示 disk_capacity_only、bandwidth_calibrated=false。 |
| S5 | [pipeline_backend.py](../../runtime/minimax_h3/pipeline_backend.py)：estimate_resources、configure_cpu_budget、cpu_budget_snapshot、prepare、infer、generation_context、_apply_generation_transformer、cleanup | 媒体工作集估算、原生线程池读回、预处理子任务、实际选项应用与恢复、CUDA 同步后 CPU 交接；metrics 与 generation_options.effective。 |
| S6 | [global_manager.py](../../runtime/minimax_h3/global_manager.py)：start、submit、_dispatch_next、cancel、_consume_result、_verified_artifact、_checkpoint、close | API/模型/隧道受管启动、durable queue、尝试所有权、取消仲裁、身份匹配结果提交、先持久化再 ack、最终审计回执。 |
| S7 | [global_manager.py](../../runtime/minimax_h3/global_manager.py)：probe_api、_transport、refresh、snapshot | 同实例活性探测、候选隧道 URL/连接、API 意图 epoch、模型实例更换屏障、观测新鲜度与分层 ready。 |
| S8 | [server.py](../../runtime/minimax_h3/server.py)：create_app、RequestGuard、submit、public_job、system、capabilities、download | API 不加载模型；鉴权先于 multipart 解析；参数、实际字节与上传并发门槛；未知提交按原 ID 对账；下载核对归属/长度/SHA。 |
| S9 | [generation_options.py](../../runtime/minimax_h3/generation_options.py)：validate_generation_options、effective_options、generation_capabilities；[adapter_config.py](../../runtime/minimax_h3/adapter_config.py) | supported/configured/ready 分开；允许的注意力、LoRA、步数；预加载资产与逐任务启用不同；能力声明不是加速证明。 |
| S10 | [startup.py](../../runtime/minimax_h3/startup.py)：StartupSession.start/snapshot/stop、_remember、_unavailable_snapshot；[process_owner.py](../../runtime/minimax_h3/process_owner.py) | 宿主所拥有进程世代、最终状态与退出观察；陈旧 ready 不覆盖当前退出；PID 与创建时间共同识别所有者。 |
| S11 | [notebook_console.py](../../runtime/minimax_h3/notebook_console.py)：submit、_bootstrap、_bind_bootstrap_host、wait_startup_delivery、_connection_context、snapshot、_operate；[console.html](../../runtime/minimax_h3/console.html) | 操作身份/指纹/通道、启动请求绑定、API 前台交付与后台模型准备分开、候选 URL/可用 URL、隐藏密钥、进度未知、停止与释放意图。 |
| S12 | [control_protocol.py](../../runtime/minimax_h3/control_protocol.py)：rpc_call、make_server；[tunnel.py](../../runtime/minimax_h3/tunnel.py)：run_tunnel、watch_owner | 私有 loopback RPC、独立密钥、有限请求体/超时；隧道分配与连接注册及父进程观察。公网可达最终仍由 S7 探测。 |
| S13 | [collect_runtime.py](../../research/collect_runtime.py)、[analyze_trace.py](../../research/analyze_trace.py)、[run_experiment.py](../../research/run_experiment.py) | 同实例轨迹、独立客户端产物核对、有限采样、最终 owner 与平台观察的验证；这些是实验观察器，不替生产决策。 |

源码说明“实现了何种判断”；测试说明“指定反例被覆盖”；真实轨迹说明“某次运行发生了什么”。三种证据不能互相冒充。文件指纹只定位版本，不认证事实真实性。

## 二、四模块 × 四视角覆盖矩阵

表中“策略”均为可替换选择；选定后可成为当前实例约束，但不提升为普遍物理事实。未知项必须保留，不可用配置或静态常量补成观测。

### 通用模型容器：当前 H3 实例

| 视角与需求目标 | 真实必要输入/观察 | 当前可选策略 | 状态、来源与待补证据 |
| --- | --- | --- | --- |
| 核心：形成可执行能力与有效产物 | 当前 GPU/可用 RAM/磁盘；必需组件与参数审计；容器 loaded、worker alive、未 failed/stopped/draining；具体输入的解码/时长/资源估算；最终媒体验证。下载百分比不是这些条件的替代。 | resident/staged、组件/分组 CPU offload、原生 BF16 主干与官方 FP32 参数例外、选定资产。 | S1—S3 已实现 H3 门槛；预检阈值非 OOM 保证，任意模型适配器与通用产物协议待建。参数审计不覆盖激活、buffer、所有算子和累加路径。 |
| 交互：接纳、查询、取消与可靠交接 | 当前 model instance、job/attempt ID、输入指纹、同身份重放一致、无其他活跃同 job 尝试/既有产物冲突；原子 reserve 通过；结果在管理器提交后才 ack。 | 接纳前明确拒绝；执行中取消等待安全边界；有限超时、终态保留、幂等确认。 | S3、S4、S6、S12 已实现；accepting 是粗容量信号，不能替某个请求的 reserve。任意传输、第三方适配器协商待建。 |
| 递归：子阶段按依赖与共享预算推进 | prepare 完成后才进入 GPU；取得单 GPU 与宿主 CPU 通道、取消未触发、容器未失败；infer 完成同步 CPU 交接后才能编码校验；失败清理不确认则不归还相应所有权。 | 固定三阶段流水；预算内参考子任务并行；全局原生线程池在加载前设定，运行中只读回。 | S3—S5 已实现固定图；当前资源估算给每任务 CPU 阶段 1 通道，不代表已用尽可能并行。任意递归 DAG、多 GPU、强制 OS 内存配额待建。 |
| 表达：请求、实际执行、成功、清理各自可追踪 | requested 在排队时存在、effective 初始 null；只有在 GPU 租约内成功应用注意力/适配器后才记实际选择；事件带模型与尝试身份；产物 SHA 与 cleanup 回执分列。 | 结构化阶段/配置/产物回执；未知与错误显式保留；不以日志行数衡量充分性。 | S3、S5、S9 已实现主要字段；generation_options 是选项应用回执，不是所有 GPU 内核的独立测量。阶段事件不完全等同于 acquire/release 全边界。 |

### 交互面板

| 视角与需求目标 | 真实必要输入/观察 | 当前可选策略 | 状态、来源与待补证据 |
| --- | --- | --- | --- |
| 核心：一键开始并持续保有控制 | 操作 request_id/指纹、面板未 closed、相应通道可用；重复启动绑定同宿主与配置；新实例需旧实例清理已确认；preflight/install 需提供明确通过凭据。 | setup/generate/control 三操作通道；启动工作后台执行；前台有界观察 API 交付，正常不要求第二次点击。 | S11 已实现。浏览器、Colab 回调链与外部授权状态仍是环境条件；“前台已返回”不能推出模型失败或成功。 |
| 交互：用户意图能被确认、取消、对账 | 同 request_id 内容一致；bootstrap 与 host instance 绑定；停止先取消并等待启动步骤、客户端进程收敛；组件命令 accepted 与 succeeded 分开。 | 重复请求返回原记录；冲突拒绝；必要确认放日志；高级控制折叠。 | S11、S10 已实现；操作 accepted 非执行完成。完整用户理解率、跨浏览器可用性与压力下交互时延未充分实测。 |
| 递归：面板操作的子生命周期可收敛 | 安装/生成客户端子进程拥有者、bootstrap 完成事件、稳定宿主身份、最终服务停止回执；release 回调发出后仍需平台证据。 | 子进程注册/终止/等待；停止与释放按真实依赖；不重建第二份生成主队列。 | S10、S11 已实现有限子操作；任意面板插件递归、平台通用资源释放证明待建。平台断线仅是观察之一，不能独立证明所有资源释放。 |
| 表达：降低认知负荷而不丢判断语义 | 同实例 fresh 状态、候选 URL 与可达 URL、密钥可用性、下载已知分母、模型 readiness、日志来源。没有分母时百分比未知；历史进度不变成新实例能力。 | 左上状态、左下控制、右侧日志；密钥默认隐藏；必要通知位于日志；静态布局与动态值分离。 | S11、S7 已实现布局和状态映射。候选 URL 可显示但必须标注未验证；所有 UI/API/日志输入状态的等效性未做穷尽证明。 |

### 主事件调度

| 视角与需求目标 | 真实必要输入/观察 | 当前可选策略 | 状态、来源与待补证据 |
| --- | --- | --- | --- |
| 核心：可靠入队与尝试派发 | 入队需当前 manager、合法参数/输入归属、幂等指纹、未关闭准入、真实磁盘余量；不需 model ready。发出执行邀约需当前模型 ready/accepting/worker_alive，且不在 shutdown/model failure；容器另做原子预算。 | durable insertion FIFO，无固定任务条数上限；队首资源忙时不绕过；永久不合格请求终态化；命令控制通道另有 4 槽上限。 | S6 已实现。队列条数无上限不等于无限存储、无限吞吐；FIFO 是公平/顺序策略，不是全局最优结论。 |
| 交互：控制、取消和结果具有同一因果身份 | manager/model/job/attempt 各身份匹配；先记录 dispatch intent；未知 execute 回应按原尝试查证，不重发；queued cancel 与 dispatch 共用仲裁；运行中取消仍持有执行责任。 | 数据库事务、控制锁、有界 RPC/探测；失联有界撤销；新实例不自动重放未知昂贵执行。 | S6、S7、S12 已实现。当前部分有界网络等待仍在控制锁内，不能宣称所有慢操作均已移出主仲裁域。 |
| 递归：独立模块并行、共享资源不重复出借 | API、模型、已选隧道是独立启动分支；前置状态/owner/实例更换屏障；CPU affinity/cgroup、阶段 RAM/磁盘估算、GPU 宿主 CPU 与 CPU 阶段共用通道。 | 子进程并发准备；固定流水线阶段预算；模型失败保留 API；并行发停止请求后分别等待退出。 | S4、S6、S7、S10 已实现。不是任意 DAG 求解器；I/O 带宽、网络竞争、多机故障域、强制资源隔离未进入完整主调度模型。 |
| 表达：主状态只陈述所拥有事实 | running 代表尝试归属，不代表占 GPU；最终结果须同实例并核对媒体/哈希；持久化提交在 ack 前；job cleanup、service shutdown、platform release 分开。 | 分层状态、可查 command_id、可重放事件游标、迟到回执拒绝、最终审计后才发布 shutdown_complete。 | S6、S7、S10 已实现主要闭环。完整租约/每个子资源的全边界枚举仍缺；有限轨迹通过不能替无限时序验证。 |

### API 创建与服务入口

| 视角与需求目标 | 真实必要输入/观察 | 当前可选策略 | 状态、来源与待补证据 |
| --- | --- | --- | --- |
| 核心：尽早提供有身份的 API | 合法公开密钥、manager 连接/私有密钥、API 实例、可写输出目录；API 存活与查询就绪不要求 CUDA/模型加载。生产宿主生成并检查公私密钥不同。 | FastAPI、loopback 私有 RPC、可选临时隧道；CORS 与 Bearer 分开。 | S8、S10、S12 已实现。create_app 单独接受测试 controller，不能据此假设所有宿主都自动执行 StartupSession 的密钥边界。通用多租户 API 工厂待建。 |
| 交互：上传、队列、取消、下载可闭环 | multipart 前鉴权；实际收包字节、文件数量/扩展名/非空、参数范围、至少一图或视频、上传磁盘与并发限制；有效幂等指纹；任务与实例身份。媒体是否真正可解码随后由容器确认。 | 202 异步任务、原 ID 对账、指定取消、鉴权下载；上传并发 3，单文件 32 MiB/合计 64 MiB，与队列条数不同。 | S8、S6 已实现。上传被接受不保证媒体处理或生成成功；下载路由返回 FileResponse 也不证明客户端完整接收，需独立客户端长度/SHA/解码回执。 |
| 递归：公网子服务和父服务独立收敛 | 当前 API 和隧道的受管进程身份；候选 URL 属于同 manager、连接已注册；公网活性探测返回当前 instance；停止/重启意图 epoch 未变化；父进程观察。 | 隧道准备与模型准备独立；API stop/start 不复制模型状态；稳定命令 ID 处理关闭自身回应链的问题。 | S7、S10、S12 已实现当前 Cloudflare 路径；任意 TLS/DNS/代理、多地域、持久公网、集群服务工厂未验证。 |
| 表达：活性、可达、能力、完成分开 | /health/live 只报告 API；/health 另看同 manager；capabilities 的 supported/configured/ready 分开；public_api_ready 与 public_ready 不同；/v1/system 投射 manager 快照，消费端仍须绑定预期实例。 | 字段显式分层、候选地址标注、未知提交保留身份、错误码不冒充业务结果。 | S7—S9 已实现各自处理，不存在“所有接口已替消费者统一校验预期实例”的万能保证。公开能力 ready 也不等于特定 shape/kernel 一定成功。 |

## 三、有限条件搜索应保留的决策边界

以下是源码判定责任的分解，不是新增可执行规则，也不是声称列出的合取已覆盖所有外部故障。

1. **允许持久化排队**：API 鉴权/上传验证 → manager 参数与归属/幂等/准入/磁盘检查 → SQLite 提交。不要额外加入模型加载完成、GPU 空闲、隧道就绪；本机请求可以不经过公网。也不要删去真实写入失败这一可能结果。
2. **允许发送执行邀约**：同一轮模型观察 ready、accepting、worker_alive，管理器无 shutdown/model failure，FIFO 队首仍 queued；持久化新的 attempt 所有权后发送。此时 running 不表示模型已接受，更不表示 CUDA 在执行。
3. **允许容器接受尝试**：模型实例匹配、同 attempt 内容一致、输入有效、没有冲突、当前 accepting、选项可用、具体请求 RAM/磁盘估算可原子 reserve。未接纳的 409/429 与回应未知必须分支；粗 has_capacity 不是 reserve 成功的充分证据。
4. **允许进入 GPU 阶段**：该尝试已接纳并预处理完成，取消未触发、容器未失败，原子取得 GPU 单租约及宿主 CPU 通道。该瞬间仍不保证推理最终成功；交接必须取得 CUDA 同步与 CPU 所有权回执。
5. **允许声明公网 API 可达**：本地 API 可用 + 隧道受管进程存活 + 同实例候选 URL/连接注册 + 匹配实例公网探测 + 当前观察新鲜/监督存活/无进程观察错误/非关闭中，并排除旧 API epoch/旧模型代的探测。目标字段是 **public_api_ready**。代码里的 **public_ready** 还要求模型可用且不排空；不能用后者替代纯公网传输条件。
6. **允许提交成功产物**：模型阶段完成、媒体完整验证、局部清理确认、结果身份匹配；管理器复核产物归属/存在/长度/SHA 后持久化；随后才 ack 模型结果。有效产物提交、下载成功、语义质量满意分别取证。
7. **允许声明资源释放**：job/attempt 清理、服务进程树退出、管理器最终审计回执、CUDA context 随拥有进程退出、云实例平台确认分别验证。API 不可达、empty_cache 返回、无单个 PID 都不是全部释放的充分条件。

## 四、不能误归类为客观物理条件的策略

| 容易误归类的量 | 当前准确归属 | 有效验证范围 |
| --- | --- | --- |
| resident 140 GiB / staged 70 GiB 可用 RAM 门槛 | S1 的项目预检策略，配合当前 profile | 通过只说明门槛成立；不证明全部工作集峰值和未来资源压力。 |
| 8 GiB 阶段 RAM 池、模型峰值估算与 OS 预留 | S4、S5 的分层估算与准入策略 | 阶段预留账本不等于模型总 RSS；实时 available/RSS 是观察，预算估算是策略参数。 |
| CPU 通道数、native pools 设为 1、单 GPU 租约 | S4、S5 的并发控制策略 | 当前读回能查已知原生线程池；通道占用不是 CPU 利用率，单租约不是硬件只能串行。 |
| Turbo 预加载、5 个调度点、4 次评估 | S9 的模型配置与受支持 profile | configured 只允许请求此选项；effective 才记录成功应用，不证明等质量提速。 |
| BF16 + 官方 FP32 例外 | S2 的精度承诺与参数元数据审计 | 拒绝已知量化/非原生参数和错误 dtype；load 另设置 allow_tf32=True，不能推出全部 FP32 算术均采用完整 FP32 尾数。 |
| 校准得到的 GB/s、TFLOP/s | 特定环境与工作负载的经验观察 | 不是全局容量硬上限，不可直接给所有模型/shape 使用；校准结果没有自动接入主调度。 |
| FIFO、公网超时、重试/撤销阈值 | S6、S7 的控制与公平策略 | 可检查遵循策略，不因此证明全局最优吞吐或最小延迟。 |

源码中 [resource_scheduler.py](../../runtime/minimax_h3/resource_scheduler.py) 的 MediaController.plan/optimal_allocation 有可选局部媒体校准 profile 与带宽参数；缺 profile 时明确串行。**当前主流水线的 pipeline_backend.prepare 使用显式 cpu_threads 限制的 lanes，不由该局部优化器给主流水线做带宽准入。** 不应因为仓库存在一个带 bandwidth 参数的函数，就把跨任务 I/O 调度标为已校准或最优。

## 五、已实现、已测和待补的边界

- **已实现**：上述 H3 固定路径的身份、准入、分阶段预算、同实例探测、请求/实际选项分离、产物校验、结果提交和停止回执。对应局部反例入口为 [主调度测试](../../runtime/minimax_h3/test_global_manager.py)、[阶段调度测试](../../runtime/minimax_h3/test_pipeline_scheduler.py)、[生成选项测试](../../runtime/minimax_h3/test_generation_options.py)、[API 实例测试](../../runtime/minimax_h3/test_server_manager_identity.py)、[面板连接测试](../../runtime/minimax_h3/test_console_connection.py)、[宿主测试](../../runtime/minimax_h3/test_startup.py)。本次只核对来源，不把旧测试通过数写成新执行结果。
- **真实有限观察**：[本轮研究结果](2026-11-15-results.md)和[测量审稿](measurement-review.md)支持指定 A100/代码/输入下的排队、取消、两任务完成、阶段重叠和独立释放观察。[轨迹检查](../../evidence/research-2026-11-15/analysis.json)的 13 项通过不是通用假设全集通过；原模块候选指标的分母、私有 RPC 时延、逐资源完整租约等仍有 unknown。
- **待建或未充分验证**：任意模型/profile 协商、任意递归 DAG、真实带宽驱动主调度、多 GPU/多机、强制资源隔离、全阶段内存/流量/FLOPs 硬上界、采样扰动、同内容或匹配分组串并行因果对照、全部异常与跨平台表达等效性。

“充分搜索”只能在声明的有限条件宇宙、约束、目标谓词、证据时效与反例集合内成立；“未找到反例”与“物理因素已穷尽”不是同一命题。代码中没有采集的事实应保留未知；策略常量必须带策略来源；旧实例、旧时刻、仅配置或仅缺失错误的记录，都不能填充新实例的客观成功条件。
