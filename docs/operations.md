# 运行与控制

本仓库保存一个已实测的 MiniMax-H3 ref2va 实例，以及面向通用模块的设计。以下操作会占用实际 GPU；日常知识整理和 CPU 测试不执行这些步骤。

## Colab 一键入口

打开 [Colab 笔记本](https://colab.research.google.com/github/zjj1280637679-ship-it/model-service-workbench/blob/main/notebooks/minimax-h3-ref2va.ipynb)，选择 A100 / 高 RAM，运行唯一启动单元。是否分配 80 GB 由平台决定，以预检为准。该路径保留平台原有 PyTorch 栈，主干和 LoRA 为原生 BF16，官方指定的 FP32 例外保留。

正常启动自动创建本次密钥，安装/检查依赖，建立管理器、API、模型与可选公网隧道。API 和公网可达即可接收符合条件的上传排队；模型就绪、资源可取得后自动推理。模型下载、LoRA 下载、加载、可用和生成完成分别显示。启动前台观察有界到 180 秒；观察到期只交回面板控制，不等于后台失败。正常使用无需再按第二个启动按钮。

左上是静态布局的只读看板，左下是常用控制与折叠高级项，右侧是动态日志；“静态”指位置与责任稳定，不表示状态值永不刷新。额外提示与必要确认放在日志面板。

1. 从看板导出本次 API 连接文件，导入[视频网页](https://zjj1280637679-ship-it.github.io/simple-llm-api-web/)。URL、密钥和实例是本轮的，不复用历史临时地址。
2. 输入新提示词与至少一张图片或一个视频；音频可以一起参考。当前不是纯文本或纯音频生成入口。
3. 选择注意力、加速 LoRA 与步数。默认关闭加速 LoRA；预加载 Turbo 后仍可逐任务关闭。Turbo 明确使用 5 个调度点，对应 4 次模型评估，不悄悄覆盖步数。
4. 可以继续提交、查询或按 ID 取消排队。队列无固定条数上限，磁盘和上传等真实资源限制仍有效。收到含 ID 的未知响应，按原 ID 对账，不重新提交生成。
5. 成功后下载并核对产物。完成后先“停止服务”，等最终清理回执，再“释放云实例”。平台会话确认与项目进程清理是两种证据。

## 独立 Linux GPU 主机

下面是现有 CLI 入口；依赖安装与硬件预检仍需在目标环境验证。本轮历史成功来自 Colab，不据此宣称独立云镜像已经验收。这里的“模型容器”首先指受管运行时进程，仓库没有声称提供已验证的 Docker/OCI 镜像。

```bash
# 在仓库根目录执行。以下系统安装命令适用于 Debian / Ubuntu。
# 其他发行版使用其包管理器安装同时提供 ffmpeg 和 ffprobe 的 FFmpeg 包。
sudo apt-get update
sudo apt-get install -y ffmpeg
command -v ffmpeg && command -v ffprobe
ffmpeg -version
ffprobe -version
(
cd runtime/minimax_h3 || exit 1
python -m pip install -r requirements.txt
# 只在需要 Turbo/风格 LoRA 时安装可选依赖并配置对应资产。
python -m pip install -r requirements-lora.txt
# --help 只显示用法，不代表依赖预检通过。
python lora_preflight.py --help
python run.py --help
python run.py --public --install-cloudflared
)
```

独立依赖清单与 Colab 清单不同。单 A100 80 GB resident 为当前常规配置；不要因出现显存压力默认改为量化权重。40 GB staged 路径保留在实现中，具体可行性与本轮 80 GB 证据分开。

已配置加速资产通过 `H3_ACCELERATION_LORA=lightx2v-ref2va-4step` 预加载；每个 job 的 `acceleration_lora` 决定是否应用。公共 API Bearer 与私有控制密钥分离，私有 RPC 仅 loopback。公网创建由显式运行选项控制。

## API 能力与命令

| 接口 | 责任 |
| --- | --- |
| `GET /health/live` | API 自身活性；不证明模型可生成 |
| `GET /v1/system` | 组件状态、实例、排队和资源观察 |
| `GET /v1/audit/budget` | 按模型/预算实例及固定截止序号读取持久租约页，每页最多256端点 |
| `GET /v1/capabilities` | supported/configured/ready 及可接受参数 |
| `POST /v1/jobs` | multipart 上传并接收异步任务 |
| `GET /v1/jobs/{id}` | 按原 ID 查询阶段、请求/实际配置、终态与清理 |
| `POST /v1/jobs/{id}/cancel` | 定点取消；已运行任务要等安全边界 |
| `GET /v1/jobs/{id}/file` | 已校验成片的鉴权下载 |
| `POST /v1/control` | 显式生命周期命令；会中断连接的命令使用稳定 command_id |

全部 `/v1/` 使用 Bearer。具体字段和允许动作以 [server.py](../runtime/minimax_h3/server.py)、[generation_options.py](../runtime/minimax_h3/generation_options.py) 为准；字段说明不能替代运行实例的能力查询。

当前单任务查询的 `execution_observation` 区分当前、历史、过期和未知；`condition_summary.last_decision` 是过去决策，不能当作当前许可。普通状态查询不复制整段租约历史；长期采集需使用审计分页并核对最终 count/tail，不以最后一页或当前零占用声称全程完整。审计写盘失败会阻断新的资源授予，释放证据缺失时清理不能显示成功。

默认 `H3_QUEUE_POLICY=bounded_backfill`。资源不足且确定未接纳的任务允许有界考察后继；身份未知、409、未确认接纳、当前宿主压力均不能绕过。`H3_QUEUE_MAX_BYPASSES=2`、`H3_QUEUE_LOOKAHEAD=8`、`H3_QUEUE_HEAD_AGE_SECONDS=30` 约束成功越过次数、候选扫描和新越过的年龄窗口；这些不是队列条数限制，也不是完成时限。设 `H3_QUEUE_POLICY=fifo` 可选择严格 FIFO。长任务已经接纳后仍可能延迟前序任务；控制取消也可能等待正在进行的有界 RPC 与对账，不承诺0.25秒完成。

## 重建笔记本

从仓库根目录执行：

```bash
python tools/build_notebook.py --check-only --preload-acceleration lightx2v-ref2va-4step
python tools/build_notebook.py --preload-acceleration lightx2v-ref2va-4step
```

构建器从仓库源码生成自包含单单元笔记本，保持面板布局。构建不是执行，不下载模型、不申请 GPU。变更后更新源码指纹和迁移记录，再做对应 CPU 验证；新源码不自动继承旧 GPU 实验的全部结论。
