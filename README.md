# Model Service Workbench

可查询、可控制、可审计的模型服务工程与模块知识库。仓库封装版本 **0.4.1**；当前参考实现为 MiniMax-H3 ref2va API **0.4.0**，配套网页为独立项目 **0.9.0**。

从同一份契约看四件事：模块的核心能力、对外交互、内部递归组合、对外表达。每个视角分别写清 **主观抽象的需求 → 客观具象的条件 → 主观具象的可选策略 → 客观抽象的可选验证指标**。这四类是不同责任，不是可信度等级，也不能互相替代。

## 从哪里开始

| 要做的事 | 入口 |
| --- | --- |
| 理解四类知识与四个视角 | [知识模型](docs/knowledge-model.md) |
| 理解模块关系、状态与并行边界 | [架构与生命周期](docs/architecture.md) |
| 运行现有 A100 BF16 实例 | [运行说明](docs/operations.md) |
| 查看真实生成和资源释放证据 | [验证范围](docs/validation.md)、[历史验收摘要](evidence/acceptance-2026-11-15.json) |
| 复核架构假设与新研究 | [实验方法](docs/research/experimental-method.md)、[物理与经验参照](docs/research/physical-and-empirical-bounds.md)、[研究结果](docs/research/2026-11-15-results.md) |
| 搜索与验证客观条件集合 | [条件设计](docs/research/objective-condition-sets.md)、[源码条件审查](docs/research/condition-source-audit.md)、[执行结果](docs/research/condition-search-results.md) |
| 复核执行前条件验证与本轮 A100 闭环 | [运行时条件契约](docs/research/runtime-condition-gates.md)、[0.4.0 实测结果](docs/research/2026-11-15-runtime-condition-results.md) |
| 检查条件变化后的策略与指标债 | [技术债审计](docs/research/condition-strategy-metric-debt.md)、[机器账本](knowledge/strategy-metric-debt.json) |
| 修改代码或增加新模型 | [维护入口](AGENTS.md)、[知识索引](knowledge/index.json) |

## 四个模块

| 模块 | 当前职责 | 详细契约 |
| --- | --- | --- |
| 通用模型容器 | 持有模型与执行尝试，完成预处理、GPU 推理、编码校验和清理 | [模型容器](docs/modules/model-container.md) |
| 交互面板 | 看板呈现事实，控制提交意图，日志呈现事件与必要确认 | [交互面板](docs/modules/interaction-panel.md) |
| 主事件调度 | 持有主任务状态、进程所有权、依赖图、资源准入与终态对账 | [主事件调度](docs/modules/event-orchestrator.md) |
| API 创建 | 配置并维护服务入口、鉴权、传输、可达性与任务接口 | [API 创建](docs/modules/api-factory.md) |

这些是逻辑责任。当前运行时仍是总管理、API 后台、大模型容器三个受管进程，加可选公网隧道；面板使用 Colab 宿主。**通用契约已整理，任意模型插件注册、任意平台部署、多机调度尚未实现。** 现有代码作为第一个专用实例保留，文档不会让它自动成为通用框架。

## v0.1.0 的历史基线

2026-11-15 在 Colab A100 80 GB、原生 BF16 下跑通：一键启动、公网 API 先上线、模型下载期间排队、指定取消、模型就绪后自动处理、三个全新视频的网页预览和下载校验、服务进程清理、平台会话释放。共 3 成功、2 取消、1 个早期依赖失败；短暂 503 按原任务 ID 对账恢复。没有用不同内容的运行时间推导加速倍数。

v0.1.0 建仓阶段只做本地和 CI 的 CPU 验证，没有重新申请 GPU；上述真实 GPU 观察作为历史来源保留。参考实现保持 BF16 主干、官方 FP32 例外，以及逐任务注意力后端、加速 LoRA 和步数控制。

## 从假设到实验

[研究协议](research/protocol.json)定义八条可证伪假设；[研究结果](docs/research/2026-11-15-results.md)另行记录实际条件、偏差、证据与未解决问题。协议中的 `planned_not_a_result` 表示它是事前定义，不是运行结果。

0.2.0 已完成一台 A100 80 GB 原生 BF16 的真实实验：两个全新单图视频成功生成与下载、一条排队取消，服务与平台资源均已释放。事件分析 13 项通过，观察到 3.502 秒跨任务 CPU/GPU 阶段重叠。有限模型给出“必要下界 3、最优可行值 4”的反例；实际全模型物理效率与策略加速倍数仍未知。原始计时、微基准样本、成片和公开证据可由结果页复核。

研究分开使用有限模型穷举、设备微基准和真实云端服务轨迹。模型中的最优值只适用于枚举空间；微基准反映特定内核和设备条件；真实任务轨迹检验本轮依赖、预算、取消和清理闭环。它们不能相互代替，也不由一次运行推出全局最优或普遍可靠性。物理必要下界、经验能力和阶段耗时各自标注来源；未测量项保留未知。

## 使用

0.3.0 的离线研究工具在声明的有限条件空间内搜索所有包含极小充分集合、给出不充分集合的反例，再以来源哈希、实例身份和有效窗口核对历史证据。该版本未改动生产调度。复核命令为 `python research/run_condition_study.py --output condition-report.json`。

0.4.0 将已审查的条件契约接入实际派发、容器接纳、资源授予和阶段启动。每个状态所有者在自己的仲裁锁内复验；跨进程由接收方再次确认。未知、冲突和假都不放行，条件判断与执行完成各有回执；独立阶段仍按共享预算并行。实现范围和审计字段见[运行时条件验证](docs/research/runtime-condition-gates.md)。

0.4.0 已在 Colab A100 80 GB、原生 BF16 主干与官方 FP32 例外下完成实际闭环：公网 API 在模型未就绪时接纳三个全新内容任务，指定取消一条，另两条自动生成并通过下载哈希及完整解码检查；最后确认服务进程清理与平台会话释放。当前面板布局未改动。运行中为补齐公共 API 的条件回执投射，管理器单独重启过一次 API；这项修正和源码身份保留在[实测报告](docs/research/2026-11-15-runtime-condition-results.md)。正常 GPU 运行没有触发未知/冲突故障，相关拒绝行为由 CPU 反例测试覆盖，不冒充已做 GPU 故障注入。

0.4.1 在不改变需求、运行时和布局的前提下复审条件变化后的策略与指标：登记12项债并修正4项，修复容量版本审计误放行、拆开CPU/GPU与全资源指标、补齐模块契约和独立复审检查。其余8项保留验收要求和依赖顺序；本版没有重新申请GPU。详见[审计结果](docs/research/condition-strategy-metric-debt.md)。

在 Colab 打开 [自包含笔记本](https://colab.research.google.com/github/zjj1280637679-ship-it/model-service-workbench/blob/main/notebooks/minimax-h3-ref2va.ipynb)，选择 A100 / 高 RAM 后运行唯一启动单元。源码与构建入口均在本仓库；模型权重由运行时按配置下载，不存入 Git。

已有的[极简网页](https://zjj1280637679-ship-it.github.io/simple-llm-api-web/)保留文本、图片和视频功能；在视频设置中导入本次 API 连接文件。其[独立源码](https://github.com/zjj1280637679-ship-it/simple-llm-api-web/tree/v0.9.0)继续维护，不在此复制第二套业务状态。

本地仅验证工程：

```bash
python -m pip install -r requirements-test.txt
python -m pytest
python tools/check_knowledge.py
python tools/check_condition_reviews.py
python tools/check_publication.py
python tools/build_notebook.py --check-only --preload-acceleration lightx2v-ref2va-4step
```

详细资源、公开 API 和释放步骤见[运行说明](docs/operations.md)。模型、第三方库与外部客户端的来源边界见 [NOTICE](NOTICE.md)。
