"""Check an explicit set of publication hazards without printing secret values.

No scanner can certify absence of every secret. This complements the repository's
allowlisted import and human review. Test fixture paths are exempt from the local
home-path heuristic, but never from token/known-secret or file-name checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

IGNORED_DIRS = {".git", ".pytest_cache", "__pycache__", ".venv", "venv", "node_modules"}
WEIGHT_SUFFIXES = {".safetensors", ".gguf", ".ggml", ".ckpt", ".pt", ".pth", ".onnx"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".txt", ".yaml", ".yml", ".html", ".js", ".mjs", ".css", ".ipynb", ".ini", ".toml"}
TOKEN = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
LOCAL_HOME = re.compile(r"(?:[A-Za-z]:[\\/]+(?:Users|OneDrive)[\\/]+[^\s\"'<>]+|/(?:Users|home)/[^/\s]+/)")


def fixture(path):
    return "tests" in path.parts or path.name.startswith("test_")


def check(root: Path, known_secrets=()):
    root = root.resolve()
    issues, checked = [], 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        if path.is_symlink():
            issues.append({"path": relative.as_posix(), "rule": "publication_symlink"})
            continue
        if not path.is_file():
            continue
        checked += 1
        name = path.name.lower()
        reason = None
        if name.endswith(".private.json") or name.startswith("control.sqlite3") or name in {".env", "api-connection.json", "pipeline_connection.json"}:
            reason = "private_runtime_or_credentials"
        elif path.suffix.lower() in WEIGHT_SUFFIXES:
            reason = "model_weights"
        elif path.stat().st_size > 32 * 1024 * 1024:
            reason = "unreviewed_large_file_over_32mib"
        if reason:
            issues.append({"path": relative.as_posix(), "rule": reason})
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"AGENTS.md", ".gitignore", ".gitattributes"}:
            try:
                content = path.read_text(encoding="utf-8-sig")
            except UnicodeError:
                issues.append({"path": relative.as_posix(), "rule": "non_utf8_text"})
                continue
            if TOKEN.search(content):
                issues.append({"path": relative.as_posix(), "rule": "credential_signature"})
            if not fixture(relative) and LOCAL_HOME.search(content):
                issues.append({"path": relative.as_posix(), "rule": "personal_absolute_path"})
            if any(secret and secret in content for secret in known_secrets):
                issues.append({"path": relative.as_posix(), "rule": "known_secret"})
    return {"status": "PASS" if not issues else "FAIL", "checked_files": checked, "issues": issues,
            "scope": "Selected private/runtime filenames, model-weight extensions, symlinks, 32 MiB size limit, token signatures, supplied known secrets and personal home paths outside test fixtures; not an exhaustive secret detector."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--known-secrets-file", type=Path,
                        help="Optional external JSON list of actual secret strings; never stored in report")
    args = parser.parse_args()
    secrets = json.loads(args.known_secrets_file.read_text(encoding="utf-8-sig")) if args.known_secrets_file else []
    if not isinstance(secrets, list) or not all(isinstance(value, str) and len(value) >= 12 for value in secrets):
        parser.error("known secrets must be a JSON list of strings of at least 12 characters")
    result = check(args.root, secrets)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
