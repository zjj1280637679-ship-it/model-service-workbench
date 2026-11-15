"""Build the self-contained Colab notebook from the reviewed project files."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import sys
import textwrap

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check-only', action='store_true', help='Validate assembled cells without writing the final notebook')
parser.add_argument('--output', type=Path, help='Write the notebook to this path')
parser.add_argument('--preload-acceleration', choices=['none', 'lightx2v-ref2va-4step'], default='none',
                    help='Preload a reviewed acceleration adapter at one-click startup; public job default remains none')
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "runtime" / "minimax_h3"
DESTINATION = ROOT / "notebooks" / "minimax-h3-ref2va.ipynb"
NAMES = ["preflight.py", "server.py", "model_backend.py", "staged_backend.py", "staged_server.py", "smoke_test.py", "runtime_options.py", "adapter_config.py", "resource_scheduler.py", "performance_model.py", "性能参照模板.json", "requirements.txt", "requirements-colab.txt", "requirements-lora.txt", "run.py", "startup.py", "control_protocol.py", "process_owner.py", "global_manager.py", "model_container.py", "tunnel.py", "client.py", "README.md", "notebook_ui.py"]
NAMES += ["pipeline_backend.py", "pipeline_scheduler.py", "pipeline_smoke_test.py"]
NAMES += ["notebook_console.py", "console.html", "download_progress.py"]
NAMES += ["generation_options.py"]
NAMES += ["lora_preflight.py"]
NAMES += ["condition_guard.py"]
FILES = {name: (PROJECT / name).read_text(encoding="utf-8") for name in NAMES}


def embedded_literal(value):
    # Colab scans form directives even inside Python string literals.
    # Escape only the marker; Python restores the exact audited source.
    return repr(value).replace("#@", "\\x23@")


FORM_LABELS = {
    "PREVIEW_LAST_VIDEO": "预览最近成片", "STAGED_BF16": "使用40GB分阶段方案",
    "MODEL_ALREADY_CACHED": "权重已完整缓存", "START_GPU_API": "启动服务", "OPEN_PUBLIC_API": "开放临时公网",
    "RUN_SMOKE_TEST": "执行本次验收", "TEST_API_URL": "验收接口", "RESUME_JOB_ID": "已有任务ID",
    "TEST_REFERENCE_FILE": "新参考图路径", "TEST_PROMPT": "新视频描述", "STOP_SERVER": "停止服务",
    "RELEASE_COLAB_RUNTIME": "释放云实例", "ACCELERATION_LORA": "加速LoRA", "STYLE_LORA": "风格LoRA",
    "STYLE_LORA_SCALE": "风格强度", "ATTENTION_BACKEND": "注意力后端", "DENOISER_OFFLOAD": "去噪卸载",
    "CONTROL_ACTION": "组件操作", "FORCE_MODEL_STOP": "强制停止模型", "COMMAND_ID": "查询命令ID",
    "EXPORT_CONNECTION": "导出私密连接文件", "LOG_SOURCE": "日志来源", "LOG_LINES": "显示行数",
}


def md(source):
    source = textwrap.dedent(source).strip()
    return {"id": "h3-" + hashlib.sha256(source.encode()).hexdigest()[:12], "cell_type": "markdown",
            "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    source = textwrap.dedent(source).strip() + "\n"
    # Human labels are visible in Colab; internal aliases keep the service interface stable.
    for name, label in FORM_LABELS.items():
        source = re.sub(r"(?m)^" + name + r"( = .*#@param.*)$",
                        lambda match: label + match[1] + "\n" + name + " = " + label, source)
    ast.parse(source)
    return {"id": "h3-" + hashlib.sha256(source.encode()).hexdigest()[:12], "cell_type": "code", "execution_count": None,
            "metadata": {"cellView": "form", "jupyter": {"source_hidden": True}},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [md("""
    # MiniMax-H3 · 参考生视频
    图片 / 视频 / 音频参考 · A100 原生 BF16 · 仅 ref2va

    **看板看状态，控制面板做操作，日志用于排错。**
    首次运行看板，再到控制面板检查、安装和启动；操作结果回到看板。
    完成后在控制面板停止服务并释放实例。

    默认 A100 80GB 常驻：启动前至少 140 GiB 可用 RAM、首次至少 140 GiB 空余磁盘。
    连接 Colab 实例会使用计算资源；看板刷新不启动模型、下载权重或公开 API。
    """)]

cells.append(md("## 1 · 检查资源"))
preflight = '''#@title 资源与驻留方案
#@markdown 只读取硬件与空间；不下载权重，不开启公网。默认A100 80GB常驻；40GB需显式选择分阶段。
if not globals().get("dashboard"):
    raise RuntimeError("先运行顶部的刷新看板，再执行资源检查。")
import os, sys
import contextlib, io
from pathlib import Path
PROJECT_DIR = Path("/content/minimax-h3-ref-api")
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
STAGED_BF16 = False #@param {type:"boolean"}
MODEL_ALREADY_CACHED = False #@param {type:"boolean"}
RESIDENCY = "staged" if STAGED_BF16 else "resident"
os.environ["H3_MODE"] = "bf16"
os.environ["H3_SKIP_DISK_CHECK"] = "1" if MODEL_ALREADY_CACHED else "0"
PREFLIGHT_OK = False
'''
preflight += '(PROJECT_DIR / "preflight.py").write_text(' + repr(FILES["preflight.py"]) + ', encoding="utf-8")\n'
preflight += '(PROJECT_DIR / "notebook_ui.py").write_text(' + repr(FILES["notebook_ui.py"]) + ', encoding="utf-8")\n'
preflight += '''if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from preflight import check_resources
from notebook_ui import StatusCard, elapsed_text, latest_stage, log_tail, run_logged, json_probe
resource_card = StatusCard("检查资源", badge="检查中", tone="running")
captured = io.StringIO()
try:
    with contextlib.redirect_stdout(captured):
        RESOURCE_REPORT = check_resources(residency=RESIDENCY)
except Exception as error:
    resource_card.update("资源检查未通过", badge="需要处理", tone="error", note=str(error),
                         details=captured.getvalue(), details_label="展开资源详情")
    raise
finally:
    (PROJECT_DIR / "preflight.log").write_text(captured.getvalue(), encoding="utf-8")
PREFLIGHT_OK = True
resource_card.update("资源检查通过", badge="可继续", tone="success", note="本格未下载模型；已有缓存会在加载时检查。",
    fields=[("GPU", RESOURCE_REPORT["gpu"]), ("可用 RAM", str(RESOURCE_REPORT["available_ram_gib"]) + " GiB"),
            ("空余磁盘", str(RESOURCE_REPORT["cache_free_disk_gib"]) + " GiB"), ("驻留方案", RESIDENCY)],
    details=captured.getvalue(), details_label="展开完整检查")
'''
cells.append(code(preflight))

cells.append(md("""
    ## 高级设置 · 可选
    可直接跳过并使用默认值。需要LoRA时先配置本格，再执行安装；只修改配置，不启动GPU。
    """))
advanced_section_id = cells[-1]["id"]
cells.append(code('''
    #@title 性能与 LoRA
    #@markdown 两类LoRA独立选择，可组合；新选项尚未做GPU验收。默认none保持原有模型。
    ACCELERATION_LORA = "none" #@param ["none", "lightx2v-ref2va-4step"]
    STYLE_LORA = "none" #@param ["none", "fal-realism-people"]
    STYLE_LORA_SCALE = 1.0 #@param {type:"slider", min:0.1, max:1.0, step:0.1}
    #@markdown Turbo使用5个sigma格点=4次去噪评估，验收会自动匹配；它不保证当前硬件的实际速度。
    ATTENTION_BACKEND = "auto" #@param ["auto", "native", "_native_flash"]
    DENOISER_OFFLOAD = "stream" #@param ["stream", "sync"]
    #@markdown auto+stream保持默认。_native_flash和sync属于实验；三阶段执行由资源预算调度。
    import os
    os.environ["H3_ACCELERATION_LORA"] = ACCELERATION_LORA
    os.environ["H3_STYLE_LORA"] = STYLE_LORA
    os.environ["H3_STYLE_LORA_SCALE"] = str(STYLE_LORA_SCALE)
    os.environ["H3_ATTENTION_BACKEND"] = ATTENTION_BACKEND
    os.environ["H3_DENOISER_OFFLOAD"] = DENOISER_OFFLOAD
    os.environ["H3_MEDIA_WORKERS"] = "auto"
    if "StatusCard" in globals():
        StatusCard("高级配置已记录", badge="尚未启动", tone="idle",
                   fields=[("加速 LoRA", ACCELERATION_LORA), ("风格 LoRA", STYLE_LORA),
                           ("注意力", ATTENTION_BACKEND), ("去噪卸载", DENOISER_OFFLOAD)],
                   note="下一次启动时生效；选择LoRA后请执行安装依赖安装依赖。已运行服务不受本次修改影响。")
    else:
        print("高级配置已记录；未启动模型。先运行看板和资源检查，再继续安装。")
    '''))
cells.append(md("## 2 · 安装依赖"))
install = '''#@title 安装与检查
#@markdown 保留 Colab 预装 PyTorch 栈。完整安装记录写入 install.log，界面只显示当前状态。
import subprocess, shutil, json
INSTALL_OK = False
if not globals().get("PREFLIGHT_OK"):
    raise RuntimeError("先运行资源检查；资源检查不通过时不安装或下载。")
existing_host = globals().get("startup_host")
if existing_host:
    existing_state = existing_host.snapshot()
    if (existing_state["server_alive"] or existing_state["tunnel_alive"] or
            existing_state.get("model_alive") or existing_state.get("manager_alive") or
            (existing_state["phase"] != "new" and
             (existing_state["phase"] not in {"failed", "stopped"} or
              not existing_state.get("cleanup_complete", False)))):
        raise RuntimeError("旧生命周期仍在运行或尚未完成清理，不能安装依赖；先在停止与释放 停止并确认子进程已退出。")
'''
stack_check = '''import importlib.metadata as metadata, json, platform, torch, psutil
versions = {}
for name in ("torch", "torchvision", "torchaudio"):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
report = {"python": platform.python_version(), "pytorch_stack": versions,
          "host_runtime": {"psutil": metadata.version("psutil")},
          "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
          "native_bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
print(json.dumps(report, ensure_ascii=False))
if psutil.__version__ != report["host_runtime"]["psutil"]:
    raise RuntimeError("psutil imported module and installed distribution versions do not match")
if not report["native_bf16"]:
    raise RuntimeError("The preinstalled PyTorch stack has no native-BF16 CUDA support")
'''
dependency_check = '''import json, os, platform, torch, diffusers, transformers, fastapi, uvicorn, av, PIL, psutil
from diffusers import ModularPipeline, MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference, MiniMaxH3AudioReference
from transformers import Qwen3VLForConditionalGeneration
from diffusers.utils.export_utils import encode_video
from adapter_config import load_adapter_settings
settings = load_adapter_settings()
if settings.acceleration != "none" or settings.style != "none":
    try:
        import peft
    except ImportError as exc:
        raise RuntimeError("LoRA enabled: rerun notebook step 2 to install requirements-lora.txt") from exc
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError("Native-BF16 CUDA support is required")
print(json.dumps({"fresh_process_imports": "passed", "python": platform.python_version(),
                  "torch": torch.__version__, "cuda": torch.version.cuda,
                  "diffusers": diffusers.__version__, "transformers": transformers.__version__,
                  "host_runtime": {"psutil": psutil.__version__},
                  "native_bf16": True}, ensure_ascii=False))
'''
install += 'PYTORCH_STACK_CHECK = ' + repr(stack_check) + '\n'
install += 'DEPENDENCY_CHECK = ' + repr(dependency_check) + '\n'
install += 'PACKAGE_FILES = ' + embedded_literal(FILES) + '\n'
install += '''for name, content in PACKAGE_FILES.items():
    (PROJECT_DIR / name).write_text(content, encoding="utf-8")
install_log = PROJECT_DIR / "install.log"
install_card = StatusCard("检查预装依赖", badge="检查中", tone="running")
stack_before = json_probe([sys.executable, "-c", PYTORCH_STACK_CHECK], install_log, card=install_card)
constraints = PROJECT_DIR / "colab-runtime-constraints.txt"
protected_versions = {**stack_before["pytorch_stack"], **stack_before["host_runtime"]}
constraints.write_text("\\n".join(name + "==" + version for name, version in protected_versions.items()) + "\\n", encoding="utf-8")
if not shutil.which("ffprobe"):
    run_logged(["apt-get", "update", "-qq"], install_log, title="准备系统依赖", card=install_card)
    run_logged(["apt-get", "install", "-y", "-qq", "ffmpeg"], install_log, title="安装 FFmpeg", card=install_card)
run_logged([sys.executable, "-m", "pip", "install", "-r", str(PROJECT_DIR / "requirements-colab.txt"), "-c", str(constraints)],
           install_log, title="安装 Python 依赖", card=install_card)
from adapter_config import load_adapter_settings
adapter_settings = load_adapter_settings()
LORA_PREFLIGHT_REPORT = None
if adapter_settings.acceleration != "none" or adapter_settings.style != "none":
    run_logged([sys.executable, "-m", "pip", "install", "-r", str(PROJECT_DIR / "requirements-lora.txt"), "-c", str(constraints)],
               install_log, title="安装 LoRA 依赖", card=install_card)
    LORA_PREFLIGHT_REPORT = json_probe([sys.executable, str(PROJECT_DIR / "lora_preflight.py"), "--repair-optional-torchao"],
                                      install_log, cwd=PROJECT_DIR, card=install_card)
    if LORA_PREFLIGHT_REPORT.get("status") != "passed" or LORA_PREFLIGHT_REPORT.get("stack_unchanged") is not True:
        raise RuntimeError("LoRA 实际 BF16 注入检查未通过；停止模型下载和加载。")
stack_after = json_probe([sys.executable, "-c", PYTORCH_STACK_CHECK], install_log, card=install_card)
if stack_after["pytorch_stack"] != stack_before["pytorch_stack"]:
    install_card.update("预装 PyTorch 栈已改变", badge="停止启动", tone="error", note="版本比对不一致；请检查 install.log。")
    raise RuntimeError("预装 PyTorch 栈发生变化，停止启动；检查安装日志。")
if stack_after["host_runtime"] != stack_before["host_runtime"]:
    install_card.update("宿主进程依赖已改变", badge="停止启动", tone="error", note="psutil 版本比对不一致；请检查 install.log。")
    raise RuntimeError("宿主进程依赖 psutil 发生变化，停止启动；检查安装日志。")
DEPENDENCY_REPORT = json_probe([sys.executable, "-c", DEPENDENCY_CHECK], install_log, cwd=PROJECT_DIR, card=install_card)
DEPENDENCY_REPORT["lora_preflight"] = LORA_PREFLIGHT_REPORT
INSTALL_OK = True
install_card.update("依赖已就绪", badge="检查通过", tone="success", note="预装 PyTorch 栈与宿主 psutil 保持不变；新进程导入检查通过。",
    fields=[("Python", DEPENDENCY_REPORT["python"]), ("PyTorch", DEPENDENCY_REPORT["torch"]),
            ("Diffusers", DEPENDENCY_REPORT["diffusers"]), ("原生 BF16", "支持")],
    details="完整安装日志：" + str(install_log) + "\\n" + json.dumps(DEPENDENCY_REPORT, ensure_ascii=False, indent=2),
    details_label="展开版本与日志位置")
'''
cells.append(code(install))

cells.append(md("## 3 · 启动三模块"))
cells.append(code('''
    #@title 启动总管理、API与模型容器
    #@markdown 总管理独立运行并持久化事件。API、模型与已启用公网并行准备；此格返回不代表生成已就绪。
    START_GPU_API = False #@param {type:"boolean"}
    OPEN_PUBLIC_API = False #@param {type:"boolean"}
    if START_GPU_API:
        import json, socket
        from startup import StartupSession
        if not globals().get("INSTALL_OK"):
            raise RuntimeError("先完成资源检查和安装依赖。")
        previous_host = globals().get("startup_host")
        if previous_host:
            previous_state = previous_host.snapshot()
            if (previous_state["server_alive"] or previous_state["tunnel_alive"] or
                    previous_state.get("model_alive") or previous_state.get("manager_alive") or
                    (previous_state["phase"] != "new" and
                     (previous_state["phase"] not in {"failed", "stopped"} or
                      not previous_state.get("cleanup_complete", False)))):
                raise RuntimeError("旧生命周期仍在运行或尚未完成清理；就绪检查查询或停止与释放停止并确认子进程已退出，不要重复启动。")
        startup_card = StatusCard("检查启动条件", badge="检查中", tone="running")
        DEPENDENCY_REPORT = json_probe([sys.executable, "-c", DEPENDENCY_CHECK], PROJECT_DIR / "install.log", cwd=PROJECT_DIR, card=startup_card)
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8000)) == 0:
                raise RuntimeError("8000 端口已占用；先停止原服务。")
        server_environment = os.environ.copy()
        if globals().get("BOOTSTRAP_API_KEY"):
            server_environment["H3_API_KEY"] = BOOTSTRAP_API_KEY
        server_environment.setdefault("HF_XET_FIXED_DOWNLOAD_CONCURRENCY", "4")
        ACTIVE_SERVER_OPTIONS = {name: server_environment.get(name, default) for name, default in {
            "H3_ACCELERATION_LORA": "none", "H3_STYLE_LORA": "none", "H3_STYLE_LORA_SCALE": "1.0",
            "H3_ATTENTION_BACKEND": "auto", "H3_DENOISER_OFFLOAD": "stream", "H3_MEDIA_WORKERS": "auto"}.items()}
        PUBLIC_API_URL = None
        API_KEY = None
        startup_host = StartupSession(PROJECT_DIR, staged=RESIDENCY == "staged", public=OPEN_PUBLIC_API,
                                      install_cloudflared=OPEN_PUBLIC_API, env=server_environment)
        startup_host.start()
        API_KEY = startup_host.api_key
        server_log = startup_host.server_log
        startup_card.update("准备分支已启动", badge="准备中", tone="running",
            note="API与模型容器独立推进；刷新顶部看板观察当前能力。后台可在模型未就绪时查询与控制。",
            fields=[("实例", startup_host.instance_id), ("驻留方案", RESIDENCY),
                    ("公网准备", "已并行启动" if OPEN_PUBLIC_API else "未启用")],
            details="状态：" + str(startup_host.state_path)
                    + "\\n模型日志：" + str(startup_host.server_log) + "\\n隧道日志：" + str(startup_host.tunnel_log), details_label="诊断记录")
    else:
        StatusCard("本次未启动", badge="等待开启", tone="idle",
                   note="勾选启动服务；需要公网时同时勾选开放临时公网。")
    '''))

cells.append(md("## 4 · 汇合检查与当前状态"))
cells.append(code('''
    #@title 等待就绪或重新查询
    #@markdown 仅检查已启动分支。API可用、模型就绪、公网可达分别显示；模型失败时保留后台查询和控制。
    if not globals().get("startup_host"):
        StatusCard("尚无运行实例", badge="未启动", tone="idle", note="先在启动服务启动本次生命周期。")
    else:
        join_card = StatusCard("检查汇合条件", badge="检查中", tone="running")
        def show_readiness(state):
            join_card.update("等待当前能力就绪", badge=state["phase"], tone="running",
                note="候选URL或某一步完成，不等于公网API可用。每次读取当前状态。",
                fields=[("已用时间", elapsed_text(state["elapsed_seconds"])),
                        ("API后台", "可用" if state.get("api_ready") else "准备中"),
                        ("模型容器", "就绪" if state["model_ready"] else "未就绪"),
                        ("隧道连接", "已连接" if state["tunnel_connected"] else ("等待中" if state["public_requested"] else "未启用")),
                        ("公网探测", "通过" if state.get("public_api_ready") else ("未通过" if state["public_requested"] else "不需要"))],
                details="状态：" + str(startup_host.state_path) + "\\n模型日志：" + str(startup_host.server_log)
                        + "\\n隧道日志：" + str(startup_host.tunnel_log), details_label="诊断记录")
        try:
            live_state = startup_host.wait_ready(on_update=show_readiness)
        except BaseException as error:
            join_card.update("生成能力尚未就绪", badge="可查询控制", tone="error", note=str(error)[:400],
                             details="状态：" + str(startup_host.state_path), details_label="查看失败记录")
            raise
        PUBLIC_API_URL = live_state["public_url"] if live_state["public_ready"] else None
        API_KEY = startup_host.api_key
        join_card.update("公网 API 已就绪" if live_state["public_ready"] else "本地 API 已就绪",
            badge="ready", tone="success", note="当前实例探测通过；生成验收另行验证真实视频生成。宿主持续监测就绪失效。",
            fields=[("API 地址", PUBLIC_API_URL or "http://127.0.0.1:8000"),
                    ("实例", live_state["instance_id"]), ("驻留方案", RESIDENCY)],
            details="状态：" + str(startup_host.state_path),
            details_label="宿主状态位置")
    '''))

cells.append(code('''
    #@title 查询、事件与模型控制
    #@markdown 仅在需要单独控制组件时使用。命令接收不代表完成；原始状态和事件在日志区查看。
    CONTROL_ACTION = "不发送命令" #@param ["不发送命令", "暂停接收", "恢复接收", "停止模型", "启动模型", "停止API", "启动API"]
    FORCE_MODEL_STOP = False #@param {type:"boolean"}
    COMMAND_ID = "" #@param {type:"string"}
    if not globals().get("startup_host"):
        StatusCard("尚无运行实例", badge="未发送", tone="idle", note="先启动服务。")
    else:
        action_names = {"暂停接收": "drain", "恢复接收": "resume", "停止模型": "stop_model",
                        "启动模型": "start_model", "停止API": "stop_api", "启动API": "start_api"}
        if CONTROL_ACTION in action_names:
            control_result = startup_host.control(action_names[CONTROL_ACTION], force=FORCE_MODEL_STOP)
        elif COMMAND_ID.strip():
            control_result = startup_host.command(COMMAND_ID.strip())
        else:
            control_result = {"status": "未发送", "command_id": "—"}
        control_card = StatusCard("组件控制", badge=control_result.get("status", "已接收"), tone="idle",
            note="填写命令 ID 并选择不发送命令，可查询执行结果。",
            fields=[("命令 ID", control_result.get("command_id", "—"))],
            details=json.dumps(control_result, ensure_ascii=False, indent=2), details_label="展开真实状态或命令回执")
    '''))

cells.append(md("## 5 · 验收与下载"))
cells.append(code('''
    #@title 生成验收与恢复查询
    #@markdown 单图512×288、8格点；每次新生成必须提供新的参考图和提示词。填写已有任务ID只查询下载。多任务验收使用pipeline_smoke_test.py。
    RUN_SMOKE_TEST = False #@param {type:"boolean"}
    TEST_API_URL = "临时公网" #@param ["临时公网", "本机API"]
    TEST_API_URL = {"临时公网": "PUBLIC_API_URL", "本机API": "http://127.0.0.1:8000"}[TEST_API_URL]
    RESUME_JOB_ID = "" #@param {type:"string"}
    TEST_REFERENCE_FILE = "" #@param {type:"string"}
    TEST_PROMPT = "" #@param {type:"string"}
    if RUN_SMOKE_TEST:
        import json, re, subprocess, time, uuid, zipfile
        from pathlib import Path
        from IPython.display import display, Video
        from google.colab import files
        live_host = globals().get("startup_host")
        live_state = live_host.snapshot() if live_host else None
        if not RESUME_JOB_ID.strip() and (not live_state or live_state.get("observation_stale")
                                          or live_state.get("api_ready") is not True):
            raise RuntimeError("API 当前尚不可达；等待后台就绪后即可提交排队，无需等待模型加载。")
        if TEST_API_URL == "PUBLIC_API_URL" and not RESUME_JOB_ID.strip():
            if live_state.get("public_api_ready") is not True:
                raise RuntimeError("公网 API 尚未通过实际探测；可继续等待或选择本机 API。")
            PUBLIC_API_URL = live_state["public_url"]
        test_base_url = globals().get("PUBLIC_API_URL") if TEST_API_URL == "PUBLIC_API_URL" else TEST_API_URL
        if not test_base_url:
            raise RuntimeError("尚无可达的公网地址；等待公网 API 就绪，或选择本机 API。")
        if not globals().get("API_KEY"):
            raise RuntimeError("缺少启动服务的API密钥；先启动本笔记本的模型服务。")
        result_dir = PROJECT_DIR / ("smoke_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        smoke_environment = os.environ.copy()
        smoke_environment["H3_API_KEY"] = API_KEY
        result_dir.mkdir(parents=True, exist_ok=True)
        smoke_log = result_dir / "smoke.log"
        report_path = result_dir / "test_report.json"
        resume_id = RESUME_JOB_ID.strip()
        command = [sys.executable, "-u", str(PROJECT_DIR / "smoke_test.py"), "--url", test_base_url,
                   "--outputdir", str(result_dir), "--generation-timeout", "5400", "--poll-seconds", "30"]
        if resume_id:
            if not re.fullmatch(r"[a-f0-9]{32}", resume_id):
                raise ValueError("已有任务ID必须是32位小写十六进制字符。")
            command += ["--resume-job", resume_id]
            action_title = "继续查询与下载"
        else:
            if not TEST_REFERENCE_FILE.strip() or not Path(TEST_REFERENCE_FILE).is_file() or not TEST_PROMPT.strip():
                raise ValueError("新生成必须提供本次全新参考图的路径和提示词；不自动复用历史素材。")
            active_options = globals().get("ACTIVE_SERVER_OPTIONS", {})
            turbo = active_options.get("H3_ACCELERATION_LORA", "none") == "lightx2v-ref2va-4step"
            command += ["--steps", "5" if turbo else "8"]
            command += ["--reference", TEST_REFERENCE_FILE, "--prompt", TEST_PROMPT,
                        "--content-history", str(PROJECT_DIR / "content-history.json")]
            action_title = "真实 API 验收"
        def current_job_status():
            if not report_path.is_file():
                return []
            checkpoint = json.loads(report_path.read_text(encoding="utf-8"))
            history = checkpoint.get("status_history") or []
            latest = history[-1] if history else {}
            phase = latest.get("active_phase")
            if isinstance(phase, dict):
                phase_name = phase.get("phase")
                phase_seconds = phase.get("elapsed_seconds")
                phase = str(phase_name) if phase_name is not None else None
                if phase is not None and isinstance(phase_seconds, (int, float)):
                    phase += " · " + str(round(phase_seconds, 1)) + " 秒"
            return [(label, value) for label, value in [
                ("任务 ID", checkpoint.get("job_id")), ("真实状态", latest.get("status")),
                ("当前阶段", phase)] if value is not None]
        result_card = StatusCard(action_title, badge="执行中", tone="running",
                                 note="读取真实任务状态；无百分比估算。填写旧任务ID时不创建新生成任务。")
        returncode = run_logged(command, smoke_log, cwd=PROJECT_DIR, env=smoke_environment,
                               title=action_title, card=result_card, status_reader=current_job_status)
        if not report_path.is_file():
            raise RuntimeError("验收脚本未生成报告；展开底部日志查看错误。")
        test_report = json.loads(report_path.read_text(encoding="utf-8"))
        if returncode != 0 or test_report.get("passed") is not True:
            raise RuntimeError("真实验收未通过。超时/中断后服务端任务可能仍在运行，请先查询任务再决定是否重试。")
        video_path = Path(test_report["video_path"]).resolve()
        if video_path.parent != result_dir.resolve() or not video_path.is_file():
            raise RuntimeError("报告中的成片路径无效。")
        video_info = test_report["video"]
        result_card.update("视频已生成并通过解码检查", badge="验收通过", tone="success", note="真实链路验收通过；画质仍以预览为准。",
            fields=[("分辨率", str(video_info["width"]) + " × " + str(video_info["height"])),
                    ("时长", str(round(video_info["duration_seconds"], 3)) + " 秒"),
                    ("帧数 / 帧率", str(video_info["decoded_frames"]) + " / " + str(video_info["fps"]) + " fps"),
                    ("本次验收耗时", elapsed_text(test_report["elapsed_seconds"]) if "elapsed_seconds" in test_report else "未记录")],
            details=json.dumps({key: test_report[key] for key in ("job_id", "terminal_status", "video", "parameters", "test_scope", "metrics", "lifecycle") if key in test_report}, ensure_ascii=False, indent=2)
                    + "\\n完整报告：" + str(report_path) + "\\n日志：" + str(smoke_log),
            details_label="展开参数、校验与报告位置")
        # Media preview belongs to the dashboard, never to the command output.
        bundle = result_dir.with_suffix(".zip")
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for artifact in (report_path, video_path):
                if artifact.is_file():
                    archive.write(artifact, arcname=artifact.name)
            if not resume_id:
                archive.write(TEST_REFERENCE_FILE, arcname="reference" + Path(TEST_REFERENCE_FILE).suffix)
        print("结果包已准备：报告、成片、参考图；不包含 API 密钥。")
        files.download(str(bundle))
    else:
        StatusCard("本次未提交验收", badge="等待开启", tone="idle", note="勾选执行本次验收；通过后下载结果包，看板可预览成片。")
    '''))

cells.append(md("## 6 · 停止与释放"))
cells.append(code('''
    #@title 关闭选项
    #@markdown 停止服务保留实例；释放实例会丢失临时文件。先保存结果，再执行释放。
    STOP_SERVER = False #@param {type:"boolean"}
    RELEASE_COLAB_RUNTIME = False #@param {type:"boolean"}
    if STOP_SERVER or RELEASE_COLAB_RUNTIME:
        if globals().get("startup_host"):
            stopped = startup_host.stop()
            from notebook_ui import redact_text
            STOP_RECEIPT = {key: stopped.get(key) for key in (
                "phase", "error", "cleanup_complete", "manager_alive", "server_alive",
                "model_alive", "tunnel_alive", "process_observation_errors", "cleanup_errors")}
            stop_detail = redact_text(json.dumps(STOP_RECEIPT, ensure_ascii=False), secrets=[
                getattr(startup_host, "api_key", ""), getattr(startup_host, "_control_key", "")])
            print("停止回执：" + stop_detail)
            process_names = {"manager_alive": "总管理", "server_alive": "API",
                             "model_alive": "模型", "tunnel_alive": "公网隧道"}
            live_processes = [label for key, label in process_names.items() if stopped.get(key) is True]
            unknown_processes = [label for key, label in process_names.items()
                                 if stopped.get(key) is not True and stopped.get(key) is not False]
            stop_issue = None
            if live_processes:
                stop_issue = "仍有进程存活：" + "、".join(live_processes)
            elif unknown_processes or stopped.get("process_observation_errors"):
                stop_issue = "进程退出状态尚未确认" + ("：" + "、".join(unknown_processes) if unknown_processes else "")
            elif stopped.get("cleanup_complete") is not True or stopped.get("phase") not in {"stopped", "failed"}:
                stop_issue = "进程已退出，但最终清理审计尚未确认"
            if stop_issue:
                raise RuntimeError(stop_issue + "；不声明关闭完成或释放实例。停止回执：" + stop_detail)
        PUBLIC_API_URL = None
        StatusCard("本笔记本的进程已停止", badge="已停止", tone="success", note="停止进程不等于释放云 GPU；实例仍可能计费。")
    if RELEASE_COLAB_RUNTIME:
        from google.colab import runtime
        StatusCard("已发起实例释放", badge="释放中", tone="running", note="等待 Colab 断开；临时文件将丢失。此消息不代表平台已确认释放。")
        runtime.unassign()
    elif not STOP_SERVER:
        StatusCard("本次未执行关闭操作", badge="等待选择", tone="idle", note="完成后勾选释放云实例。")
    '''))

# Presentation groups are independent of the backend's dependency graph.
# Refreshing the dashboard never waits for readiness or dispatches a command.
dashboard_source = '''#@title 刷新看板
#@markdown 只查看当前状态。操作中的结果会更新这里；没有后台自动轮询，空闲时可再次运行刷新。
PREVIEW_LAST_VIDEO = False #@param {type:"boolean"}
import sys, json
from pathlib import Path
PROJECT_DIR = Path("/content/minimax-h3-ref-api")
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
'''
dashboard_source += '(PROJECT_DIR / "notebook_ui.py").write_text(' + repr(FILES["notebook_ui.py"]) + ', encoding="utf-8")\n'
dashboard_source += '''if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from notebook_ui import mount_dashboard, StatusCard, elapsed_text, log_tail, run_logged, json_probe
def dashboard_context():
    return {"host": globals().get("startup_host"), "resource_report": globals().get("RESOURCE_REPORT"),
            "dependency_report": globals().get("DEPENDENCY_REPORT"),
            "preflight_ok": globals().get("PREFLIGHT_OK", False),
            "install_ok": globals().get("INSTALL_OK", False)}
dashboard = mount_dashboard(dashboard_context)
if PREVIEW_LAST_VIDEO and globals().get("video_path") and Path(video_path).is_file():
    from IPython.display import display, Video
    display(Video(str(video_path), embed=True))
'''

connection_cell = code('''
    #@title 导出 API 连接信息
    #@markdown 需要外部客户端时再导出。文件包含密钥，请单独保存；密钥不显示在看板或日志里。
    EXPORT_CONNECTION = False #@param {type:"boolean"}
    if EXPORT_CONNECTION:
        from google.colab import files
        host = globals().get("startup_host")
        state = host.snapshot() if host else {}
        if not state.get("api_ready"):
            raise RuntimeError("API 尚不可用；刷新看板查看当前状态。")
        base_url = state.get("public_url") if state.get("public_api_ready") else "http://127.0.0.1:8000"
        connection_path = PROJECT_DIR / "api-connection.private.json"
        connection_path.write_text(json.dumps({"url": base_url, "api_key": host.api_key,
                                              "instance_id": host.instance_id}), encoding="utf-8")
        connection_path.chmod(0o600)
        files.download(str(connection_path))
        StatusCard("连接文件已准备", badge="已导出", tone="success", note="文件包含密钥，请单独保存。")
    ''' )

logs_cell = code('''
    #@title 查看日志
    #@markdown 只读取选择的来源；不自动滚动、不发送控制命令。显示有限尾部并隐藏密钥。
    LOG_SOURCE = "最近诊断" #@param ["最近诊断", "资源检查", "安装依赖", "API后台", "模型容器", "全局管理", "公网隧道", "调度事件", "完整状态", "最近验收"]
    LOG_LINES = 100 #@param {type:"slider", min:20, max:200, step:20}
    from IPython.display import HTML, display
    from notebook_ui import diagnostics_html, last_diagnostics
    host = globals().get("startup_host")
    log_files = {"资源检查": PROJECT_DIR / "preflight.log", "安装依赖": PROJECT_DIR / "install.log"}
    if host:
        log_files.update({"API后台": host.api_log, "模型容器": host.server_log,
                          "全局管理": host.manager_log, "公网隧道": host.tunnel_log})
    if globals().get("smoke_log"):
        log_files["最近验收"] = smoke_log
    if LOG_SOURCE == "最近诊断":
        log_text = last_diagnostics() or "暂无诊断记录。"
    elif LOG_SOURCE in {"调度事件", "完整状态"}:
        value = (host.events(limit=50) if LOG_SOURCE == "调度事件" else host.snapshot()) if host else {"状态": "尚无运行实例"}
        log_text = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        selected_path = log_files.get(LOG_SOURCE)
        log_text = log_tail(selected_path, limit=32768) if selected_path else ""
        log_text = log_text or "该来源暂无日志。"
    log_text = "\\n".join(log_text.splitlines()[-max(20, min(200, LOG_LINES)):])
    display(HTML(diagnostics_html(LOG_SOURCE, log_text, secrets=[getattr(host, "api_key", ""), globals().get("API_KEY", "") or ""])))
    ''' )

def action_cell(title):
    return next(cell for cell in cells if cell["cell_type"] == "code"
                and ("#@title " + title + "\n") in "".join(cell["source"]))

advanced_heading = md("""
    ### 高级设置与组件控制
    LoRA 和性能配置在安装前设置；默认均不启用。需要排错或单独控制组件时再展开。
    """)
logs_heading = md("""
    ## 日志
    排错时展开。选择一个来源，运行查看；详细记录只显示在这里。
    """)
cells = [cells[0], md("## 看板\n当前能力、队列和下一步。状态来自本次观察；页面打开不代表服务正在运行。"), code(dashboard_source),
         md("""
    ## 控制面板
    首次使用：**资源检查 → 安装 → 启动**。之后按需生成，完成后停止与释放。
    API 与公网在模型加载期间也可就绪；无需等模型就绪才查询或提交排队。
    """),
         action_cell("资源与驻留方案"), action_cell("安装与检查"),
         action_cell("启动总管理、API与模型容器"), action_cell("生成验收与恢复查询"),
         action_cell("关闭选项"), advanced_heading, action_cell("性能与 LoRA"),
         action_cell("等待就绪或重新查询"), action_cell("查询、事件与模型控制"), connection_cell,
         logs_heading, logs_cell]

program_titles = {'preflight': '资源与驻留方案', 'install': '安装与检查',
                  'start': '启动总管理、API与模型容器', 'generate': '生成验收与恢复查询',
                  'stop': '关闭选项', 'advanced': '性能与 LoRA',
                  'wait': '等待就绪或重新查询', 'control': '查询、事件与模型控制',
                  'export': '导出 API 连接信息', 'logs': '查看日志'}
programs = {name: ''.join(action_cell(title)['source']) for name, title in program_titles.items()}
bootstrap_options = {'public': True, 'acceleration': args.preload_acceleration}
console_bundle_hash = hashlib.sha256(json.dumps({'files': FILES, 'programs': programs, 'bootstrap_options': bootstrap_options},
    ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
console_source = '''#@title 一键启动
import sys, json, importlib
from pathlib import Path
PROJECT_DIR = Path("/content/minimax-h3-ref-api")
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
'''
console_source += 'PACKAGE_FILES = ' + embedded_literal(FILES) + '\n'
console_source += 'CONSOLE_PROGRAMS = ' + embedded_literal(programs) + '\n'
console_source += 'H3_CONSOLE_BUNDLE_HASH = ' + repr(console_bundle_hash) + '\n'
console_source += 'H3_BOOTSTRAP_OPTIONS = ' + repr(bootstrap_options) + '\n'
console_source += '''_h3_previous = globals().get("h3_console")
_h3_reuse = (_h3_previous is not None and not getattr(_h3_previous, "closed", False)
             and globals().get("H3_LOADED_BUNDLE_HASH") == H3_CONSOLE_BUNDLE_HASH)
if not _h3_reuse:
    # Never replace code or discard ownership while old work may still be active.
    if _h3_previous is not None and getattr(_h3_previous, "active", None) != {}:
        raise RuntimeError("旧控制台仍有操作或无法确认空闲；先在旧控制台停止并等待完成，再运行本格升级。")
    _h3_old_host = globals().get("startup_host")
    if _h3_old_host is None and _h3_previous is not None:
        _h3_old_host = getattr(_h3_previous, "scope", {}).get("startup_host")
    if _h3_old_host is not None:
        _h3_old_state = _h3_old_host.snapshot()
        _h3_no_processes = all(_h3_old_state.get(name) is False
                               for name in ("manager_alive", "server_alive", "model_alive", "tunnel_alive"))
        _h3_clean_terminal = (_h3_old_state.get("phase") in {"failed", "stopped"}
                              and _h3_old_state.get("cleanup_complete") is True)
        _h3_never_started = (_h3_old_state.get("phase") == "new"
                             and not _h3_old_state.get("observation_stale"))
        if (not _h3_no_processes or _h3_old_state.get("process_observation_errors")
                or not (_h3_clean_terminal or _h3_never_started)):
            raise RuntimeError("旧生命周期仍在运行或清理未确认；保留旧控制台和文件。先停止服务并确认资源释放，再运行本格升级。")
    if _h3_previous is not None and not getattr(_h3_previous, "closed", False):
        _h3_previous.close()
    globals().pop("h3_console", None)
    for name, content in PACKAGE_FILES.items():
        (PROJECT_DIR / name).write_text(content, encoding="utf-8")
    # Python's import cache must not silently retain the old controller signature.
    _h3_project_root = PROJECT_DIR.resolve()
    for _h3_name, _h3_module in list(sys.modules.items()):
        _h3_module_file = getattr(_h3_module, "__file__", None)
        if _h3_name != "__main__" and _h3_module_file:
            try:
                _h3_is_project_module = Path(_h3_module_file).resolve().is_relative_to(_h3_project_root)
            except (OSError, TypeError, ValueError):
                _h3_is_project_module = False
            if _h3_is_project_module:
                sys.modules.pop(_h3_name, None)
    importlib.invalidate_caches()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from notebook_console import mount_console
h3_console = mount_console(PROJECT_DIR, CONSOLE_PROGRAMS, globals(), auto_start=True,
                           bootstrap_payload=H3_BOOTSTRAP_OPTIONS, foreground_bootstrap_timeout=180)
H3_LOADED_BUNDLE_HASH = H3_CONSOLE_BUNDLE_HASH
'''
cells = [code(console_source)]
sys.path.insert(0, str(PROJECT))
from notebook_console import render_console
cells[0]['outputs'] = [{'output_type': 'display_data', 'metadata': {}, 'data': {
    'text/html': [render_console(FILES['console.html'])],
    'text/plain': ['控制台尚未挂载。首次运行本单元即自动开始检查、安装、API 和模型准备。']}}]

notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "colab": {"name": DESTINATION.name, "provenance": [], "gpuType": "A100", "machine_shape": "hm"},
    "accelerator": "GPU",
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"}}, "cells": cells}
args.output = args.output or DESTINATION
if args.check_only:
    print(f"Validated {len(cells)} cells with AST parsing; final notebook was not written.")
else:
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"Built {args.output} with {len(cells)} cells; all Python cells passed AST parsing.")
