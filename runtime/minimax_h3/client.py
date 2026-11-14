"""Submit ordered references to the ref2va API, poll, and download the MP4."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import mimetypes
import os
from pathlib import Path
import time
import requests


class ReferenceAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        items = list(getattr(namespace, self.dest, None) or [])
        items.append((option_string.removeprefix("--"), Path(value)))
        setattr(namespace, self.dest, items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="API base URL, e.g. https://example.trycloudflare.com")
    parser.add_argument("--prompt", required=True)
    for kind in ("image", "video", "audio"):
        parser.add_argument("--" + kind, dest="references", action=ReferenceAction, metavar="FILE")
    parser.add_argument("--output", type=Path, default=Path("result.mp4"))
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200, help="Maximum poll time in seconds")
    args = parser.parse_args()
    key = os.environ.get("H3_API_KEY")
    if not key:
        parser.error("Set H3_API_KEY in the caller's environment first")
    if not args.references:
        parser.error("At least one --image or --video reference is required")
    if args.output.exists():
        parser.error("Output file already exists; choose another --output path")
    base = args.url.rstrip("/")
    if not (base.startswith("https://") or base.startswith("http://127.0.0.1:")):
        parser.error("Use HTTPS for public endpoints, or HTTP on 127.0.0.1 for local testing")
    headers = {"Authorization": "Bearer " + key}
    with ExitStack() as stack:
        files = [("files", (path.name, stack.enter_context(path.open("rb")), mimetypes.guess_type(path.name)[0] or "application/octet-stream"))
                 for _, path in args.references]
        response = requests.post(
            base + "/v1/jobs", headers=headers, files=files,
            data={"kinds": json.dumps([kind for kind, _ in args.references]), "prompt": args.prompt,
                  "seconds": args.seconds, "width": args.width, "height": args.height,
                  "steps": args.steps, "seed": args.seed}, timeout=120,
        )
    response.raise_for_status()
    job_id = response.json()["id"]
    print("Job:", job_id, flush=True)
    deadline = time.monotonic() + args.timeout
    while True:
        response = requests.get(base + f"/v1/jobs/{job_id}", headers=headers, timeout=30)
        response.raise_for_status()
        job = response.json()
        print(job["status"], flush=True)
        if job["status"] == "failed":
            raise RuntimeError(json.dumps(job, ensure_ascii=False))
        if job["status"] == "succeeded":
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Polling timed out. Job {job_id} may still be running; query it later.")
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    created_partial = False
    try:
        with requests.get(base + f"/v1/jobs/{job_id}/file", headers=headers, timeout=120, stream=True) as response:
            response.raise_for_status()
            with partial.open("xb") as output:
                created_partial = True
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        partial.replace(args.output)
    finally:
        if created_partial:
            partial.unlink(missing_ok=True)
    print("Saved:", args.output.resolve())


if __name__ == "__main__":
    main()
