"""Read-only dashboard and separate diagnostics for the notebook control panel."""
from __future__ import annotations

import html
import json
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import subprocess
import time

STYLE = """
<style>
.h3-card{--bg:#f8fafc;--ink:#172033;--muted:#627086;--line:#dbe3ec;--accent:#187c72;box-sizing:border-box;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:12px;padding:17px 20px;margin:8px 0;font:14px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:980px}
.h3-card *{box-sizing:border-box}.h3-card[data-tone="error"]{--accent:#ba4242}.h3-card[data-tone="idle"]{--accent:#7b8797}.h3-card[data-tone="running"]{--accent:#bd7c16}
.h3-head{display:flex;gap:12px;justify-content:space-between;align-items:center;flex-wrap:wrap}.h3-title{font-size:17px;font-weight:650;letter-spacing:-.2px}.h3-badge{font-size:11px;font-weight:650;letter-spacing:.3px;color:var(--accent);border:1px solid var(--line);padding:3px 9px;border-radius:99px;white-space:nowrap}.h3-note{color:var(--muted);margin:5px 0 0}
.h3-fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px 18px;margin-top:13px}.h3-label{display:block;color:var(--muted);font-size:11px;letter-spacing:.25px;margin-bottom:2px}.h3-value{font-size:14px;font-weight:550;overflow-wrap:anywhere}.h3-card details{margin-top:12px;border-top:1px solid var(--line);padding-top:10px}.h3-card summary{cursor:pointer;color:var(--muted);font-size:12px}.h3-card pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;margin:9px 0 0;max-height:280px;overflow:auto}.h3-card a{color:var(--accent);text-decoration:none}.h3-card a:hover{text-decoration:underline}
.h3-board{border-left-width:1px;padding:22px 24px}.h3-board h2{font-size:23px;line-height:1.3;margin:4px 0 8px}.h3-kicker{font-size:12px;color:var(--muted)}.h3-next{margin:12px 0 21px;padding:11px 14px;border-radius:8px;background:color-mix(in srgb,var(--accent) 8%,var(--bg));font-size:15px}.h3-modules{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:12px 0 19px}.h3-module{padding:12px 14px;border:1px solid var(--line);border-radius:8px}.h3-module strong{font-weight:650;display:block;font-size:16px;margin-top:4px}.h3-subtle{color:var(--muted);font-size:12px;margin-top:4px}.h3-summary{display:flex;flex-wrap:wrap;gap:8px 22px;margin:12px 0}.h3-activity{border-top:1px solid var(--line);margin-top:18px;padding-top:13px}.h3-activity .h3-title{font-size:14px}.h3-activity .h3-fields{margin-top:8px;gap:8px 18px}.h3-stamp{font-size:11px;color:var(--muted);margin-top:17px}.h3-logs{border-left-width:1px}.h3-logs pre{max-height:360px;margin:14px 0 0}.h3-logs pre:focus{outline:2px solid var(--accent);outline-offset:3px}
@media(max-width:620px){.h3-board{padding:17px 15px}.h3-modules{grid-template-columns:1fr;gap:8px}.h3-module{padding:9px 12px;display:grid;grid-template-columns:95px 1fr;align-items:center}.h3-module strong{margin:0}.h3-module .h3-subtle{grid-column:2}.h3-board h2{font-size:21px}.h3-fields{grid-template-columns:1fr}}
@media(prefers-color-scheme:dark){.h3-card{--bg:#17202d;--ink:#e4ebf4;--muted:#a5b3c6;--line:#344358;--accent:#67cdbb}.h3-card[data-tone="error"]{--accent:#ff9696}.h3-card[data-tone="running"]{--accent:#e4ba6d}.h3-card[data-tone="idle"]{--accent:#b5bfcb}}
</style>
"""

_dashboard = None
_diagnostics = deque(maxlen=20)
_known_secrets = ()
_process_observer = None
_DIAGNOSTIC_LIMIT = 12000


def redact_text(text, secrets=()):
    """Redact known credentials and common authorization/key assignments before display."""
    text = str(text)
    for secret in sorted({str(s) for s in (*_known_secrets, *secrets) if s}, key=len, reverse=True):
        text = text.replace(secret, '[已隐藏]')
    text = re.sub(r'(?i)(\bBearer\s+)[^\s\"\'<>;,]+', r'\1[已隐藏]', text)
    text = re.sub(r'''(?ix)(["']?(?:H3_API_KEY|API_KEY|api[-_]?key|access_token|authorization)["']?\s*[:=]\s*)(["'])(.*?)(\2)''',
                  lambda match: match.group(1) + match.group(2) + '[已隐藏]' + match.group(2), text)
    text = re.sub(r'(?i)(\b(?:H3_API_KEY|API_KEY|api[-_]?key|access_token)\s*[:=]\s*)[^\s\"\'<>;,]+', r'\1[已隐藏]', text)
    return text


def _remember(title, text):
    if text:
        _diagnostics.append(redact_text(str(title) + '\n' + str(text))[-_DIAGNOSTIC_LIMIT:])


def last_diagnostics():
    return '\n\n'.join(_diagnostics)[-_DIAGNOSTIC_LIMIT:]


def diagnostics_html(title, text, *, secrets=()):
    """A bounded, keyboard-scrollable log view; never embedded in dashboard cards."""
    value = redact_text(text, secrets=secrets)
    truncated = len(value) > _DIAGNOSTIC_LIMIT
    value = value[-_DIAGNOSTIC_LIMIT:]
    note = '仅显示最后 12,000 个字符；完整记录保留在日志文件。' if truncated else '只读日志 · 敏感凭据已隐藏'
    return (STYLE + '<section class="h3-card h3-logs" aria-label="日志"><div class="h3-title">'
            + html.escape(redact_text(title), quote=True) + '</div><p class="h3-note">' + note
            + '</p><pre tabindex="0" aria-label="日志内容">' + html.escape(value or '暂无记录。', quote=True)
            + '</pre></section>')


def _short(value, maximum=160):
    text = redact_text(value).replace('\n', ' ').replace('\r', ' ')
    return text if len(text) <= maximum else text[:maximum - 1] + '…'


def _activity(title, kwargs):
    _remember(title, kwargs.get('details'))
    fields = [(label, value) for label, value in kwargs.get('fields', ())
              if not any(word in str(label) for word in ('日志', '路径', '密钥', '当前记录'))]
    return {'title': _short(title, 70), 'badge': _short(kwargs.get('badge', ''), 35),
            'note': _short(kwargs.get('note', '')), 'tone': kwargs.get('tone', 'idle'),
            'fields': [(str(k), _short(v, 95)) for k, v in fields[:4]]}


def card_html(title, *, badge="", note="", fields=(), tone="idle", details=None, details_label="查看详情"):
    escape = lambda value: html.escape(_short(value), quote=True)
    if tone not in {"idle", "running", "success", "error"}:
        tone = "idle"
    values = "".join(
        '<div><span class="h3-label">' + escape(label) + '</span><span class="h3-value">'
        + escape(value) + '</span></div>' for label, value in fields
    )
    return (STYLE + '<section class="h3-card" data-tone="' + tone + '"><div class="h3-head">'
            + '<div class="h3-title">' + escape(title) + '</div><span class="h3-badge">'
            + escape(badge) + '</span></div>' + ('<p class="h3-note">' + escape(note) + '</p>' if note else "")
            + ('<div class="h3-fields">' + values + '</div>' if fields else "") + "</section>")


def render_dashboard(context, activity=None):
    """Render supplied observations only. This pure function performs no host I/O."""
    escape = lambda value: html.escape(_short(value), quote=True)
    state = context.get('state') or {}
    stale = bool(state.get('observation_stale'))
    failed_read = bool(context.get('observation_error'))
    phase = state.get('phase')
    if context.get('placeholder'):
        headline, next_step, tone = '等待首次读取', '运行「刷新看板」查看本次状态；此处是界面占位，没有实时观察。', 'idle'
    elif failed_read:
        headline, next_step, tone = '状态读取失败', '重新刷新看板；如仍失败，在日志区查看原因。', 'error'
    elif stale:
        headline, next_step, tone = '状态需要刷新', '当前观察已过期；刷新后再判断可用能力。', 'running'
    elif phase == 'stopped' and state.get('cleanup_complete'):
        headline, next_step, tone = '本次服务已停止', '确认结果已保存后，可在控制面板释放 Colab 实例。', 'idle'
    elif phase == 'stopping':
        headline, next_step, tone = '正在停止与回收', '等待受管进程退出并确认清理完成；此时不要重复启动。', 'running'
    elif state.get('error') or phase in {'failed', 'degraded'}:
        headline, next_step, tone = '有模块需要处理', '先看下方可用能力；具体原因在日志区，控制面板仍可查询或停止。', 'error'
    elif state.get('draining'):
        headline, next_step, tone = '已暂停接收新任务', '已接收的队列继续执行；可恢复接收，也可停止服务。', 'idle'
    elif state.get('ready'):
        headline, next_step, tone = '可以生成', '在控制面板提交新内容，或按任务 ID 查询已有任务。', 'success'
    elif state.get('api_ready'):
        accepting = (state.get('queue') or {}).get('admission_open') is True
        headline, next_step, tone = ('API 可用 · 生成尚未就绪',
            ('可先查询或排队；' if accepting else '可先查询；') + '模型、公网的当前能力分别见下方。', 'running')
    elif state:
        headline, next_step, tone = '服务正在准备', '刷新看板观察进展；不要重复启动。', 'running'
    elif not context.get('preflight_ok'):
        headline, next_step, tone = '尚未检查资源', '在控制面板先运行「资源与驻留方案」。', 'idle'
    elif not context.get('install_ok'):
        headline, next_step, tone = '资源检查通过', '在控制面板运行「安装与检查」。', 'idle'
    else:
        headline, next_step, tone = '准备工作已完成', '在控制面板选择是否开放公网，再启动服务。', 'idle'

    def module(name, alive_key, ready_key=None):
        alive, ready = state.get(alive_key), state.get(ready_key) if ready_key else None
        if not state:
            status, detail = ('尚未读取' if context.get('placeholder') else '未启动'), '尚无运行观察'
        elif ready is True:
            status, detail = '可用', '进程运行中' if alive is True else '存活状态未确认'
        elif alive is True:
            status, detail = ('运行中', '调度与控制') if ready_key is None else ('准备中', '进程运行，能力未就绪')
        elif alive is False:
            status, detail = '已停止', '进程未运行'
        else:
            status, detail = '未确认', '缺少当前观察'
        if stale or failed_read:
            status, detail = '待刷新', '不据旧状态判断就绪'
        return ('<div class="h3-module"><span class="h3-label">' + escape(name) + '</span><strong>'
                + escape(status) + '</strong><div class="h3-subtle">' + escape(detail) + '</div></div>')

    if stale or failed_read:
        public = '待刷新'
    elif not state:
        public = '尚未观察'
    elif not state.get('public_requested'):
        public = '未启用'
    elif state.get('public_api_ready'):
        public = 'API 探测通过'
    elif state.get('tunnel_connected'):
        public = '隧道已连接 · API 待验证'
    else:
        public = '准备中'
    queue = state.get('queue') or {}
    summary = [('公网', public)]
    if queue and not stale and not failed_read:
        summary += [('任务', f"排队 {queue.get('queued', '—')} · 执行 {queue.get('running', '—')}")]
        if queue.get('admission_open') is False:
            summary.append(('接收任务', '已暂停'))
    resources = context.get('resource_report') or {}
    if resources:
        summary += [('GPU', resources.get('gpu', '未记录')),
                    ('可用内存', str(resources.get('available_ram_gib', '—')) + ' GiB（预检）')]
    resource_text = ''.join('<span><span class="h3-label">' + escape(k) + '</span>' + escape(v) + '</span>' for k, v in summary)
    stamp = state.get('observed_at')
    if isinstance(stamp, (int, float)):
        try:
            stamp = datetime.fromtimestamp(stamp, timezone(timedelta(hours=8))).strftime('%m-%d %H:%M:%S 北京时间')
        except (OverflowError, OSError, ValueError):
            stamp = '未记录'
    else:
        stamp = '尚无服务快照'
    body = (STYLE + '<section class="h3-card h3-board" data-tone="' + tone + '" aria-label="运行看板">'
            '<div class="h3-kicker">当前状态</div><h2>' + escape(headline) + '</h2>'
            '<div class="h3-next"><b>下一步</b>　' + escape(next_step) + '</div><div class="h3-modules">'
            + module('总管理', 'manager_alive') + module('API 后台', 'server_alive', 'api_ready')
            + module('大模型容器', 'model_alive', 'model_ready') + '</div>'
            '<div class="h3-summary">' + resource_text + '</div>')
    if activity:
        activity_tone = activity.get('tone', 'idle')
        if activity_tone not in {'idle', 'running', 'success', 'error'}:
            activity_tone = 'idle'
        body += ('<div class="h3-activity" data-tone="' + activity_tone + '"><div class="h3-label">最近操作</div>'
                 '<div class="h3-head"><span class="h3-title">' + escape(activity.get('title', ''))
                 + '</span><span class="h3-badge">' + escape(activity.get('badge', '')) + '</span></div>')
        if activity.get('note'):
            body += '<div class="h3-note">' + escape(activity['note']) + '</div>'
        fields = activity.get('fields') or ()
        if fields:
            body += '<div class="h3-fields">' + ''.join('<div><span class="h3-label">' + escape(k)
                + '</span><span class="h3-value">' + escape(v) + '</span></div>' for k, v in fields[:4]) + '</div>'
        body += '</div>'
    return body + '<div class="h3-stamp">服务观察：' + escape(stamp) + ' · 按需刷新，不执行控制命令</div></section>'


class Dashboard:
    def __init__(self, context_reader):
        self.context_reader = context_reader
        self.activity = None
        self._handle = None

    def refresh(self):
        from IPython.display import HTML, display
        global _known_secrets
        context = dict(self.context_reader())
        host = context.pop('host', None)
        if host is not None:
            key = getattr(host, 'api_key', None)
            if key:
                _known_secrets = tuple(set((*_known_secrets[-3:], str(key))))
            try:
                context['state'] = host.snapshot()
            except Exception as error:
                context['state'] = {}
                context['observation_error'] = True
                _remember('看板状态读取失败', str(error))
        rendered = HTML(render_dashboard(context, self.activity))
        if self._handle is None:
            self._handle = display(rendered, display_id='h3-dashboard')
        else:
            self._handle.update(rendered)
        return context


def mount_dashboard(context_reader):
    global _dashboard
    _dashboard = Dashboard(context_reader)
    _dashboard.refresh()
    return _dashboard


class StatusCard:
    def __init__(self, title, **kwargs):
        self._handle = None
        self.update(title, **kwargs)

    def update(self, title, **kwargs):
        from IPython.display import HTML, display
        activity = _activity(title, kwargs)
        if _dashboard is not None:
            _dashboard.activity = activity
            _dashboard.refresh()
        elif self._handle is None:
            self._handle = display(HTML(card_html(**activity)), display_id=True)
        else:
            self._handle.update(HTML(card_html(**activity)))


def elapsed_text(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60}分{seconds % 60:02d}秒" if seconds >= 60 else f"{seconds}秒"


def log_tail(path, limit=6000):
    path = Path(path)
    if not path.is_file():
        return ""
    with path.open("rb") as source:
        source.seek(max(0, path.stat().st_size - limit))
        text = source.read().decode("utf-8", errors="replace")
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).replace("\r", "\n")


def latest_stage(path):
    lines = [line.strip() for line in log_tail(path).splitlines() if line.strip()]
    for line in reversed(lines):
        if "BF16 staged:" in line:
            return line.split("BF16 staged:", 1)[1].strip()[:145]
        if "status=" in line or "API_SMOKE_TEST_PASSED" in line:
            return line[:145]
    return lines[-1][:145] if lines else "进程已启动，等待首条日志"


def run_logged(command, log_path, *, title, cwd=None, env=None, card=None, status_reader=None):
    """Run a real child process; persist logs and update a bounded status card."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    status = card or StatusCard(title, badge="执行中", tone="running", note="完整输出写入日志。")
    started = time.monotonic()
    with path.open("a", encoding="utf-8") as output:
        output.write("\n--- " + title + " ---\n")
        output.flush()
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
        observer = _process_observer
        if observer is not None:
            observer.process_started(process)
        try:
            while process.poll() is None:
                if observer is not None:
                    observer.process_progress(process)
                current = []
                if status_reader is not None:
                    try:
                        current = status_reader()
                    except (OSError, ValueError):
                        pass
                status.update(title, badge="执行中", tone="running", fields=
                    [("已用时间", elapsed_text(time.monotonic() - started))]
                    + current)
                time.sleep(2)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            status.update(title, badge="已中断", tone="error", note="操作进程已停止；已提交的远程任务可能仍在执行。", fields=[("日志", str(path))])
            raise
        finally:
            if observer is not None:
                observer.process_finished(process)
    if process.returncode:
        status.update(title, badge="未完成", tone="error", note="查看日志中的实际错误后再重试。",
                      fields=[("日志", str(path))], details=log_tail(path), details_label="展开最后的错误日志")
        raise RuntimeError(f"{title}退出码 {process.returncode}；完整日志：{path}")
    status.update(title, badge="完成", tone="success", fields=[
        ("已用时间", elapsed_text(time.monotonic() - started)), ("完整日志", str(path)),
    ])
    return process.returncode


def json_probe(command, log_path, *, cwd=None, env=None, card=None):
    """Capture a short JSON-producing check without dumping full notebook output."""
    process = subprocess.Popen(command, cwd=cwd, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    observer = _process_observer
    try:
        if observer is not None:
            observer.process_started(process)
        stdout, stderr = process.communicate(timeout=180)
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
        raise
    finally:
        if observer is not None:
            observer.process_finished(process)
    with Path(log_path).open("a", encoding="utf-8") as log:
        log.write(stdout + "\n" + stderr + "\n")
    if process.returncode:
        if card is not None:
            card.update("依赖检查未通过", badge="需要处理", tone="error", note="检查日志中的版本或导入错误后再继续。",
                        fields=[("日志", str(log_path))], details=stderr[-4000:], details_label="展开检查错误")
        raise RuntimeError(f"依赖检查失败；请在日志区查看：{log_path}")
    return json.loads(stdout.strip().splitlines()[-1])
