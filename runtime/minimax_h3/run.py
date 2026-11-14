"""Start/query/control the API, model container and global event manager."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

from startup import StartupSession
from control_protocol import rpc_call


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--staged", action="store_true", help="Explicit sequential BF16 loading for an A100 with >=70 GiB available host RAM; not the default resident profile")
    parser.add_argument("--public", action="store_true", help="Prepare a temporary tunnel alongside the model; readiness requires their live join")
    parser.add_argument("--install-cloudflared", action="store_true", help="With --public, fetch an official SHA-256 checked cloudflared release if needed")
    parser.add_argument("--attention-backend", choices=["auto", "native", "_native_flash"])
    parser.add_argument("--denoiser-offload", choices=["stream", "sync"])
    parser.add_argument("--acceleration-lora", choices=["none", "lightx2v-ref2va-4step"])
    parser.add_argument("--style-lora", choices=["none", "fal-realism-people"])
    parser.add_argument("--style-lora-scale", type=float)
    parser.add_argument("--media-workers", choices=["auto", "1", "2", "3", "4"], help="A ceiling; parallel preparation also requires a local measured profile")
    parser.add_argument("--media-profile", type=Path, help="Local calibration JSON for bounded CPU child scheduling")
    parser.add_argument("--validate-config", action="store_true", help="Validate configuration without initializing CUDA, downloading, or starting a server")
    parser.add_argument("--manager-url", help="Existing private loopback controller; use H3_CONTROL_KEY")
    parser.add_argument("--action", choices=["status", "events", "drain", "resume", "start_model", "stop_model", "start_api", "stop_api", "shutdown"], default="status")
    parser.add_argument("--force", action="store_true", help="Explicitly permit stopping an occupied model container")
    args = parser.parse_args()
    if args.manager_url:
        token = os.environ.get("H3_CONTROL_KEY", "")
        if len(token) < 32:
            raise ValueError("Set the existing private H3_CONTROL_KEY for controller access")
        operation = args.action if args.action in {"status", "events"} else "control"
        payload = {} if operation != "control" else {"action": args.action, "force": args.force}
        print(json.dumps(rpc_call(args.manager_url, token, operation, payload), ensure_ascii=False, indent=2))
        return
    for argument, name in {
        "attention_backend": "H3_ATTENTION_BACKEND", "denoiser_offload": "H3_DENOISER_OFFLOAD",
        "acceleration_lora": "H3_ACCELERATION_LORA", "style_lora": "H3_STYLE_LORA",
        "style_lora_scale": "H3_STYLE_LORA_SCALE", "media_workers": "H3_MEDIA_WORKERS",
        "media_profile": "H3_MEDIA_PROFILE",
    }.items():
        value = getattr(args, argument)
        if value is not None:
            os.environ[name] = str(value)
    from adapter_config import load_adapter_settings
    from runtime_options import RuntimeOptions
    runtime_options, adapters = RuntimeOptions.from_env(), load_adapter_settings()
    if os.environ.get("H3_MEDIA_WORKERS", "auto") not in {"auto", "1", "2", "3", "4"}:
        raise ValueError("H3_MEDIA_WORKERS must be auto or 1..4")
    if args.media_profile and not args.media_profile.is_file():
        raise FileNotFoundError("The media calibration profile does not exist")
    if args.validate_config:
        print(json.dumps({"configuration": "valid", "residency": "staged" if args.staged else "resident",
                          "runtime": runtime_options.__dict__, "adapters": adapters.as_dict(),
                          "media_workers": os.environ.get("H3_MEDIA_WORKERS", "auto"),
                          "gpu_started": False}, ensure_ascii=False, indent=2))
        return
    residency = "staged" if args.staged else "resident"
    new_key = not os.environ.get("H3_API_KEY")
    session = StartupSession(Path(__file__).resolve().parent, port=args.port, staged=args.staged,
                             public=args.public, install_cloudflared=args.install_cloudflared)
    if new_key:
        print("Temporary API key (shown once; keep private): " + session.api_key, flush=True)
    print(f"BF16 residency profile: {residency}; lifecycle status: {session.state_path}", flush=True)
    last_state = None

    def show_state(state):
        nonlocal last_state
        summary = {name: state[name] for name in (
            "instance_id", "phase", "ready", "api_ready", "model_ready", "tunnel_connected", "public_ready", "public_url", "error")}
        if summary != last_state:
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            last_state = summary

    try:
        session.start()
        print(f"Manager control: {session.manager_url}; API log: {session.api_log}; model log: {session.server_log}", flush=True)
        try:
            state = session.wait_ready(on_update=show_state)
            print(f"Generation ready: http://127.0.0.1:{args.port}/docs", flush=True)
            if args.public:
                print(f"Public generation verified for this instance: {state['public_url']}", flush=True)
        except (RuntimeError, TimeoutError) as error:
            print(f"Generation unavailable: {error}. Query/control remain available while the manager is alive.", flush=True)
        while True:
            state = session.snapshot()
            show_state(state)
            if state.get("manager_alive") is False and state["phase"] != "stopped":
                raise RuntimeError(state["error"])
            if state["phase"] == "stopped":
                break
            time.sleep(1)
    finally:
        session.stop()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        main()
    except KeyboardInterrupt:
        print("Server stopped. Release the cloud GPU instance separately to stop provider billing.")
