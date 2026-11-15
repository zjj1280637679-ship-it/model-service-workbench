"""Replay a historical trace against its explicitly frozen source bytes."""
import argparse
import json

from analyze_condition_gates import analyze
from source_bindings import bound_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binding")
    args = parser.parse_args()
    trace, source_root = bound_trace(args.binding)
    report = analyze(trace, source_root=source_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    unexercised = {"unknown_blocked_without_effect", "conflict_blocked_without_effect"}
    return int(report["counts"]["fail"] != 0 or any(
        row["status"] != "pass" for name, row in report["checks"].items() if name not in unexercised))


if __name__ == "__main__":
    raise SystemExit(main())
