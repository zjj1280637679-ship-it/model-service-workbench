# 0.5.0：历史源码绑定与八项技术债修复

需求基线仍为 v0.4.0 的四模块、十六视角，requirements 文本保持。先完成 TD-010，随后并行处理相互独立的投射、审计和指标分支，再对共同接口做联测。布局保留左上看板、左下控制及折叠高级项、右侧日志；A100 原生 BF16 主干与官方 FP32 例外不变。

本版实施与验证范围是 CPU 控制面、实际本地进程/API、有限队列模型、历史证据重放和 notebook 构建。没有申请新的 GPU，不把 v0.4.0 的生成与平台释放记录改称新版实测。具体测试与 CI 结果见 [repository-validation.json](../../evidence/repository-validation.json)。

## 修复与关闭依据

| 债项 | 本版修复 | 可复核的验证 |
| --- | --- | --- |
| TD-010 历史来源 | 绑定旧提交、实验/实例和四份执行源文件SHA；缺失不回落当前源码 | `test_historical_source_binding.py`；只读历史重放仍7 pass/2 unknown，旧证据未改 |
| TD-004 单任务当前证据 | 按manager/job/attempt/model与观测时效投射阶段条件及完成见证；ack后保留历史记录 | 准备、GPU等待、编码、迟到、错身份、缺/未来时间及终态重启反例 |
| TD-005 默认诊断 | 共用有界脱敏摘要，分开当前等待、历史决定、已完成阶段 | Python与实际console JavaScript同源fixture；CSS/DOM布局不变；未声称认知负荷有测量改善 |
| TD-006 长会话审计 | 每个预算实例独立SQLite日志，固定截止序号和哈希链分页；512内存尾不扩大 | 86次完整三阶段、516端点；重启历史只读、重复页、缺段、冲突、最终释放、写盘失败反例 |
| TD-007 查询成本 | `has_capacity`直接读容量，`status`只读一次无历史快照；分页磁盘读在主状态锁外 | 空/12/516端点，每组300次的请求、锁时间、字节、错误与超2秒计数；范围仅预算查询片段 |
| TD-008 队头权衡 | 默认有界资源回填，可选严格FIFO；只绕过明确资源不足且保证未接纳的任务 | 有限对照同时报告后继阻塞、等待、越过次数与长后继的代价；真实管理器覆盖拒绝、未知、丢ack、取消与年龄边界 |
| TD-011 实验来源复用 | 导出强制显式协议、声明基线、执行源码清单、实例与干预；缺失或矛盾拒绝 | 导出反例覆盖旧默认、错协议/源/身份、干预遗漏、变更中输入和隐私投射；只在全部验证后发布新目录 |
| TD-012 指标重算 | 五项适配器按身份、窗口、资源范围、覆盖和原始回执重算，并直接核对知识观察 | 产物2/2、主事件3/3、GPU租约违规0、CPU/GPU超借0、全资源unknown；错分母、错实例、缺覆盖与造假数值拒绝 |

关闭的是这些已定义缺口，不是全系统证明。全部 RAM/磁盘租约端点、端到端控制 p95、语义质量和完整物理效率仍缺足够证据，继续 unknown。

## 并行与公平的范围

当前同任务依赖仍是 prepare → GPU → encode_validate；不同任务的独立阶段按共享预算交叠。授权提交在所属仲裁域有序，推理与独立读盘不占用主状态锁；跨进程接收端再次复验。未知接纳先按原身份对账，不能借“回填”重复发起生成。

默认回填只接受明确的 `429/resource_budget_exhausted/guarantees_not_accepted=true`。候选窗口默认8，每个被越过的等待任务受最多2次成功越过与30秒年龄窗口约束；已接纳的长任务可能延迟队头，因此没有通用完成时间上界。409、未知所有权、未知/冲突条件和宿主压力观察均不能用作绕过理由。候选数限制不限制队列总任务数。

[有限对照报告](../../evidence/debt-resolution-2026-11-15/queue-policy-study.json)使用 model_tick，包含长后继导致队头等待增加的反例。这是选定策略的有限评价，不是 GPU 测速、任意到达序列的防饥饿证明或数学全局最优。

## 审计和指标的证据边界

授予/释放先持久化再改变内存所有权，写盘失败锁存，无法确认的释放继续占账且 cleanup 不能成功。日志保存在每个预算实例独立文件中；磁盘空间仍有限，耗尽必须失败，不能伪造完成或扩大无界内存。重启读取历史不恢复执行权。序号、链摘要、最终 count/tail 和实例身份共同验证覆盖，哈希不是第三方签名证明。

[预算查询对照](../../evidence/debt-resolution-2026-11-15/budget-query-benchmark.json)比较当前轻量路径与按当前结构复现的旧重复复制方式。它测量本机CPU、已预热文件系统、预算片段与JSON编码；没有网络、推理竞争或完整模型状态锁，不填入云端私有status p95，也不声称A100提速。

[指标重算结果](../../evidence/debt-resolution-2026-11-15/metric-replay.json)只覆盖显式选择的0.4实验五项观察。原有数值保持，仅新增机器可核对的窗口、作用域和证据绑定；旧0.2观察不被顺带声称验证。记录的媒体回执不等于本次重新解码，全资源未知匹配表示诚实保留未知，不表示全资源通过。

历史绑定来自 [runtime-conditions-v040.json](../../research/source-bindings/runtime-conditions-v040.json)。执行源码通过记录的文件字节核对；导出描述符的基线提交是声明，不据此伪造Git执行血缘。历史SHA从不回写新源码。

## 复核命令

```bash
python research/replay_historical_conditions.py research/source-bindings/runtime-conditions-v040.json
python research/replay_metrics.py --trace evidence/runtime-conditions-2026-11-15/trace.json --client evidence/runtime-conditions-2026-11-15/client-receipt.json --contract research/metric-replays/runtime-conditions-v040.json --source-binding research/source-bindings/runtime-conditions-v040.json --contract-id h3-runtime-conditions-v040-closed-batch --knowledge-module knowledge/modules/model-container.json --knowledge-module knowledge/modules/event-orchestrator.json
python -m pytest runtime/minimax_h3 tests
node --test tests/test_console_delivery.cjs
python tools/check_condition_reviews.py
python tools/check_knowledge.py
python tools/check_publication.py
python tools/build_notebook.py --check-only --preload-acceleration lightx2v-ref2va-4step
```

复审快照与普通源码清单仍分别维护；改实现、条件或指标后，必须先评估差异，再记录已审输入。CI同时验证当前实现和固定历史，不让重算指纹替代指标或策略复审。
