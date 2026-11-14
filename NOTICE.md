# 来源与范围

本仓库整理本项目已实现的控制与推理服务代码，不包含模型权重或第三方依赖的分发包。迁移来源和字节身份见 [import-provenance](evidence/import-provenance.json)。

- 推理参考：MiniMax-H3；通过 Diffusers 0.43.0 与 Transformers 5.7.0 等运行库调用。准确安装约束见 runtime/minimax_h3/requirements*.txt。
- 加速 LoRA 参考：[ModelTC / Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo)。资产标识、固定修订及校验规则在 [adapter_config.py](runtime/minimax_h3/adapter_config.py)。
- 注意力入口来自 [Diffusers attention_dispatch.py](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/attention_dispatch.py) 与 PyTorch SDPA；没有把量化内核称为原生 BF16。
- 外部网页：[simple-llm-api-web v0.9.0](https://github.com/zjj1280637679-ship-it/simple-llm-api-web/tree/v0.9.0)，保留独立仓库、版本和证据。
- 临时公网由 Cloudflare cloudflared 提供；Colab 是本次历史 GPU 验收宿主，不能由此推定任意云环境等价。

第三方代码与模型的使用条件分别以各自上游资料为准。本仓库的通用契约是项目设计，不代表上述厂商提供了相同的通用模块接口。
