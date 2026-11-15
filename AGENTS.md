# 工程维护入口

先读 README、docs/knowledge-model.md、docs/architecture.md，再读对应 docs/modules 与 knowledge/modules 契约。用户本轮指令优先；本文件不额外引入审批。

- 保持需求、条件、策略、指标四类独立，第三类准确称为“可选策略”。策略被场景选中后可产生该场景的约束，但不是硬件或世界的固有事实。
- 当前 runtime/minimax_h3 是专用参考实现。新增通用适配器、宿主或多机支持必须实际实现并验证，不能只修改名字或契约状态。
- 主任务、执行尝试、组件实例、GPU 租约、产物、清理、平台会话各有身份和责任。can_start、ready、completed、cleanup_complete、runtime_released 不合并。
- 从真实依赖及共享预算决定并行。状态提交有序不等于整个主流程串行。缺少带宽/FLOPs/关键路径测量时不声称达到数学最优。
- 默认 CPU 测试不下载权重、不启动公网、不申请 GPU。真实实验是单独工作；如执行生成，每次使用全新内容，提交结果未知先按原 ID 对账。
- 修改路径、文档或 notebook 时记录迁移差异；已运行源码的历史 SHA 不回写为新源码。evidence 的摘要不是签名证明，结构校验不等于语义完备。
- 秘钥、私有连接、模型缓存、用户参考、运行数据库、原始运行日志不进 Git。发布使用显式源文件范围；不要复制整个工作目录。
- 修改完成后运行对应 CPU 测试、知识校验和 notebook 检查。更新 evidence/source-manifest.json 的明确方法见 docs/validation.md。不要把过期指纹包装成当前 MATCH。
- 条件、策略、指标或已审输入改变时，检查 knowledge/condition-reviews.json 与技术债账本；先读差异、复审影响，再记录新指纹。check_condition_reviews 的 current 不是债务清零，freeze_sources 不替代此复审。运行时变更前先处理账本 TD-010 的历史源码绑定。
