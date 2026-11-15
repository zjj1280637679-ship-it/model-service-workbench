# MiniMax-H3 ref2va 参考运行时 · API 0.5.0

这里保存已验收的专用实现：总管理、API 后台、大模型容器三个受管进程，独立的可选公网隧道与 Colab 控制台。模型容器是模型运行时进程，不把 GPU 硬件本身称为容器。

- 当前模型：MiniMax-H3，仅 ref2va；至少图片或视频参考，可附加音频。
- A100 原生 BF16 主干与 LoRA，保留官方 FP32 例外，不启用权重量化。
- 任务分为 CPU 预处理、独占 GPU 推理、CPU 编码校验；按真实依赖及资源预算调度。
- 队列无固定条数上限，可指定 ID 取消；资源不足、未知提交和运行中取消各自有真实状态。
- API 可先于模型就绪接收排队。生成、产物完整校验、任务清理、服务进程退出和云平台释放分别确认。
- 每任务控制 attention_backend、acceleration_lora、steps；Turbo 要显式 steps=5，对应4次模型评估。预加载不等于当前任务启用。

仓库内的[运行说明](../../docs/operations.md)、[架构](../../docs/architecture.md)、[验证边界](../../docs/validation.md)提供当前入口。此 README 进入自包含 notebook 时，仓库相对链接仅用于源码导航；云端实际操作使用面板与日志。

```bash
python run.py --help
python smoke_test.py --help
```

历史 A100 实测不是当前机器的一次新运行。源码来源与精度/配置/任务/释放证据见仓库 evidence。真实生成验收每次必须是全新内容；未知结果按原 ID 对账，不自动重新生成。
