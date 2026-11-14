# 验证与证据边界

## 当前知识的证据层

| 层 | 本仓库实际能检查什么 | 不能自动推出 |
| --- | --- | --- |
| 结构 | 四模块/四视角/四类齐全，指标字段、引用、身份有效 | 目标适当、策略最优、指标完备 |
| 源码身份 | manifest SHA 与现有文件一致 | 源码正确、历史日志真实 |
| CPU 回归 | 在合成或本地进程条件下检查接纳、幂等、取消、预算、清理等 | CUDA、画质、任意平台兼容 |
| Notebook 构建 | 自包含代码可组装、Python AST 有效、启动配置被嵌入 | 当前 Colab 可分配 GPU、依赖永远不漂移 |
| 历史 GPU 实验 | 指定版本、配置、输入、设备上的实际观察 | 全称正确性、通用模型兼容、数学最优 |

## 2026-11-15 A100 实验

[结构化摘要](../evidence/acceptance-2026-11-15.json)保留六个真实任务：3 成功、2 取消、1 个早期依赖失败。成功场景为雪地蜗牛天文台、折纸狐狸叶舟、玻璃茶壶星球；覆盖图片+视频、图片+音频和单图，均为本轮全新内容。成功成片为 512×288、124 帧、24 fps，含音频；浏览器预览、下载哈希与完整解码通过。

最终 06:03 一键启动、06:05 API 与已选公网先交付、模型下载期排队，指定取消一条，保留任务自动生成；06:20:29 完成进程清理，06:22 平台会话列表无活动会话。A100 原生 BF16 + 官方 FP32 例外；一次 Turbo 后的普通任务确认 `none` 生效。没有可比较的同输入 A/B 性能实验。

早期可选 torchao 版本阻断 PEFT 的问题修复后，最终新实例通过自动依赖早检；原 PyTorch 栈保持。未知提交含任务 ID 时按原 ID 对账；短暂 503 没有变成重复生成。原始完整记录见[固定提交的真实验收报告](https://github.com/zjj1280637679-ship-it/simple-llm-api-web/blob/3a693f121d265849783eb1588dbcae380dd5ce2c/docs/reviews/2026-11-15-bf16-browser.md)。

公开仓库只保存选定的脱敏事实、任务和媒体身份、源文件哈希。未发布原始 runtime 数据库、私有连接、签名下载链接、整份运行日志或本机路径。摘要是项目记录，不是第三方签名证明；源文件指纹和事实真实性分别评估。

## 本次建仓验证

本次迁移不启动 GPU。受管运行时代码与历史来源做 LF 归一后的身份核对；构建器和测试调整仓库路径，runtime README 改为仓库内入口。另补齐两项旧测试私有调用夹具的请求/实例身份，并断言绑定关系；修复一项测试的同步点，等待实际等待阶段日志写入后才检查日志，避免把开始观察误当作副作用已完成。运行时代码未改变。已验收原始 notebook 的 SHA 保留在 import-provenance，仓库 notebook 只因 README 文档嵌入而有独立 SHA；不得将两者说成逐字节相同。

运行：

```bash
python -m pip install -r requirements-test.txt
python -m pytest
python tools/check_knowledge.py
python tools/check_publication.py
python tools/build_notebook.py --check-only --preload-acceleration lightx2v-ref2va-4step
```

本地完整 Python 回归 **644 通过、8 跳过**；8 项均为 Windows 账号缺少符号链接权限，Linux CI 可在权限允许时执行。控制台实际 JavaScript 的 Node 回归 **2 通过**。结果与测试范围记录在 [repository-validation.json](../evidence/repository-validation.json)；CI 独立执行同一类检查。GPU 依赖不安装到 CPU 测试环境，跳过项必须在报告中明确列出，不计入通过。

`tools/check_publication.py` 只覆盖它声明的发布检查模式，不是对所有可能敏感信息的完备检测。发布前仍应审查新增文件来源与实际 diff。

## 更新身份记录

完成源码和知识文档改动后执行 `python tools/freeze_sources.py`，再执行知识、发布和对应回归检查。冻结工具只重算明确范围内的文件身份，不重写历史验收、不声称测试通过，也不更新记录者的观测时间。`evidence/source-manifest.json` 不包含自身或易变运行结果，避免递归自哈希。

当模型、依赖、调度、宿主、API 语义或内容条件改变，先说明变更影响的命题，再选择需要重新执行的实验。没有真实测量时保留 unknown；理论下界、CPU 模型和真实耗时的单位及覆盖范围不一致时，不进行直接比较。
