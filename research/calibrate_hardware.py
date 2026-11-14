"""Bounded empirical calibration, never a physical peak or inference benchmark.

Run before loading a model, in a separate process owned by the experiment host:
  python research/calibrate_hardware.py --gpu --transfers --disk-dir /content \
      --output /content/calibration.json
The default omits GPU work and does not import torch. No dependencies, weights,
network, clock changes, global cache drop, or inference services are started.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time

MIB = 1024 ** 2
GIB = 1024 ** 3
SOURCES = [
    "https://docs.pytorch.org/docs/stable/notes/cuda.html",
    "https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html",
    "https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html",
]


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def summary(values):
    """Descriptive statistics only: samples from this run are not independent trials."""
    values = [float(positive(value, "sample")) for value in values]
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None, "mean": None, "sample_stdev": None}
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values), "mean": statistics.mean(values),
            "sample_stdev": statistics.stdev(values) if len(values) > 1 else None}


def measured_rates(samples, work_per_iteration, unit, duration_key="wall_seconds"):
    positive(work_per_iteration, "work_per_iteration")
    if unit not in {"FLOP", "byte"}:
        raise ValueError("Explicit unit must be FLOP or byte")
    rates = []
    for sample in samples:
        count = sample["iterations"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("iterations must be a positive integer")
        rates.append(work_per_iteration * count / positive(sample[duration_key], duration_key))
    return {"kind": "empirical_microbenchmark", "unit": unit + "/s", "duration_key": duration_key,
            "work_per_iteration": work_per_iteration, "samples": rates, "summary": summary(rates),
            "physical_peak": None, "model_capacity": None}


def cuda_samples(torch, operation, samples, iterations, warmup=2):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    rows = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        for _ in range(iterations):
            operation()
        end.record()
        end.synchronize()
        wall = time.perf_counter() - wall_start
        seconds = start.elapsed_time(end) / 1000.0
        positive(seconds, "CUDA event seconds")
        rows.append({"iterations": iterations, "cuda_event_seconds": seconds, "wall_seconds": wall})
    return rows


def wall_samples(operation, samples, iterations, warmup=2):
    for _ in range(warmup):
        operation()
    rows = []
    for _ in range(samples):
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        for _ in range(iterations):
            operation()
        rows.append({"iterations": iterations, "wall_seconds": time.perf_counter() - wall_start,
                     "process_cpu_seconds": time.process_time() - cpu_start})
    return rows


def cpu_cases(size, samples, iterations):
    source = bytearray(os.urandom(size))
    target = bytearray(size)
    source_view, target_view = memoryview(source), memoryview(target)

    def copy():
        target_view[:] = source_view

    rows = wall_samples(copy, samples, iterations)
    if target != source:
        raise RuntimeError("host copy verification failed")
    copy_result = {"status": "measured", "samples": rows, "warmup_iterations": 2,
                   "payload_bytes_per_iteration": size, "logical_read_plus_write_bytes_per_iteration": 2 * size,
                   "rate": measured_rates(rows, 2 * size, "byte"), "verified": True,
                   "interpretation": "Single memoryview copy stream; logical read+write, not measured memory-bus traffic. CPU caches, write allocation and NUMA remain unmeasured."}
    # Fixed-size SHA256 provides a named CPU operation; bytes are not generic CPU FLOPs.
    block = memoryview(source)[:min(size, 16 * MIB)]
    rows = wall_samples(lambda: hashlib.sha256(block).digest(), samples, iterations)
    return {"host_copy": copy_result,
            "cpu_sha256": {"status": "measured", "samples": rows, "warmup_iterations": 2,
                           "rate": measured_rates(rows, len(block), "byte"),
                           "interpretation": "SHA256 input-byte rate including Python calls; neither generic CPU work capacity nor CPU-memory bandwidth."}}


def disk_cases(directory, size, samples):
    """Use only one uniquely owned file; no filesystem-wide cache operations."""
    data = os.urandom(size)
    expected = hashlib.sha256(data).hexdigest()
    write_rows, read_rows = [], []
    fd, filename = tempfile.mkstemp(prefix="model-calibration-", suffix=".bin", dir=directory)
    path = Path(filename)
    try:
        os.close(fd)
        for _ in range(samples):
            start = time.perf_counter()
            with path.open("wb", buffering=0) as output:
                view = memoryview(data)
                while view:
                    written = output.write(view)
                    if not written:
                        raise RuntimeError("disk write made no progress")
                    view = view[written:]
                os.fsync(output.fileno())
            write_rows.append({"iterations": 1, "wall_seconds": time.perf_counter() - start})
            start = time.perf_counter()
            with path.open("rb", buffering=0) as source:
                observed = source.read(size + 1)
            read_rows.append({"iterations": 1, "wall_seconds": time.perf_counter() - start})
            if len(observed) != size or hashlib.sha256(observed).hexdigest() != expected:
                raise RuntimeError("disk content verification failed")
    finally:
        path.unlink(missing_ok=True)
    return {"status": "measured", "file_bytes": size, "total_bytes_written": size * samples,
            "write_fsync": {"samples": write_rows, "rate": measured_rates(write_rows, size, "byte")},
            "read_after_write": {"samples": read_rows, "rate": measured_rates(read_rows, size, "byte")},
            "verified": True, "owned_file_removed": not path.exists(), "cache_state": "uncontrolled_likely_warm",
            "interpretation": "Application-visible file I/O. Write includes fsync; read immediately follows write and may be served from page cache. No cold physical-disk bandwidth claim; backing storage/durability are unknown."}


def other_gpu_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=10)
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if any(not row.isdecimal() for row in rows):
        raise RuntimeError("compute-process observation is unavailable")
    return [int(row) for row in rows if int(row) != os.getpid()]


def gpu_preflight(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    name = properties.name
    if ("A100" not in name.upper() or "MIG" in name.upper() or properties.multi_processor_count != 108
            or (properties.major, properties.minor) != (8, 0)):
        raise RuntimeError("A full A100 with native BF16 is required; other GPUs and MIG are not this experiment")
    try:
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        # Older PyTorch lacks this parameter; the explicit full-A100 SM80 check
        # above supplies the native-instruction requirement independently.
        bf16 = torch.cuda.is_bf16_supported()
    if not bf16:
        raise RuntimeError("native BF16 unavailable")
    competitors = other_gpu_processes()
    if competitors:
        raise RuntimeError("other GPU compute processes exist; run calibration before model startup")
    free, total = torch.cuda.mem_get_info()
    if free < 2 * GIB or torch.cuda.memory_allocated() > 0:
        raise RuntimeError("calibration requires a fresh process and at least 2 GiB free GPU memory")
    return {"name": name, "compute_capability": [properties.major, properties.minor],
            "multiprocessors": properties.multi_processor_count, "total_memory_bytes": properties.total_memory,
            "free_memory_bytes_before": free, "runtime_total_memory_bytes": total,
            "native_bf16_supported": True, "other_compute_processes_at_preflight": 0,
            "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
            "isolation_limit": "A point-in-time process observation; no external GPU lock or guarantee against later contenders."}


def gpu_cases(torch, size, matrix_size, samples, iterations, transfers):
    results = {}
    n = matrix_size
    with torch.inference_mode():
        a = torch.randn((n, n), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((n, n), device="cuda", dtype=torch.bfloat16)
        c = torch.empty_like(a)
        rows = cuda_samples(torch, lambda: torch.mm(a, b, out=c), samples, iterations)
        if c.dtype != torch.bfloat16 or not bool(torch.isfinite(c[:8, :8]).all().item()):
            raise RuntimeError("BF16 GEMM sample verification failed")
        results["bf16_gemm"] = {"status": "measured", "shape": [n, n, n], "input_output_dtype": "bfloat16",
            "samples": rows, "warmup_iterations": 2, "flops_per_iteration": 2 * n ** 3,
            "rate": measured_rates(rows, 2 * n ** 3, "FLOP", "cuda_event_seconds"),
            "explicit_device_tensor_bytes": 3 * n * n * 2, "finite_8x8_output_sample": True,
            "interpretation": "Dense square GEMM, 2 FLOPs per multiply-add. cuBLAS chooses the implementation; accumulation path is not hardware-profiled. Does not measure the model's mixed BF16/FP32 workload."}
        del a, b, c
        torch.cuda.empty_cache()
        source = torch.empty(size, dtype=torch.uint8, device="cuda").random_(0, 256)
        target = torch.empty_like(source)
        rows = cuda_samples(torch, lambda: target.copy_(source, non_blocking=True), samples, iterations)
        if not torch.equal(source[:1024], target[:1024]):
            raise RuntimeError("device copy sample verification failed")
        results["device_copy"] = {"status": "measured", "samples": rows, "warmup_iterations": 2,
            "payload_bytes_per_iteration": size, "logical_read_plus_write_bytes_per_iteration": 2 * size,
            "payload_rate": measured_rates(rows, size, "byte", "cuda_event_seconds"),
            "logical_traffic_rate": measured_rates(rows, 2 * size, "byte", "cuda_event_seconds"),
            "explicit_device_tensor_bytes": 2 * size, "verified_sample_bytes": 1024,
            "interpretation": "Repeated device copy; logical read+write is not an HBM traffic counter. Buffer exceeds A100 L2 but caching and actual bus traffic remain unmeasured."}
        del source, target
        torch.cuda.empty_cache()
        if transfers:
            host = torch.empty(size, dtype=torch.uint8, pin_memory=True).random_(0, 256)
            device = torch.empty(size, dtype=torch.uint8, device="cuda")
            expected = host[:1024].clone()
            rows = cuda_samples(torch, lambda: device.copy_(host, non_blocking=True), samples, iterations)
            if not torch.equal(device[:1024].cpu(), expected):
                raise RuntimeError("H2D sample verification failed")
            h2d = {"status": "measured", "samples": rows, "rate": measured_rates(rows, size, "byte", "cuda_event_seconds")}
            host[:1024].zero_()
            rows = cuda_samples(torch, lambda: host.copy_(device, non_blocking=True), samples, iterations)
            if not torch.equal(host[:1024], expected):
                raise RuntimeError("D2H sample verification failed")
            d2h = {"status": "measured", "samples": rows, "rate": measured_rates(rows, size, "byte", "cuda_event_seconds")}
            results["pinned_transfers"] = {"status": "measured", "host_to_device": h2d, "device_to_host": d2h,
                "payload_bytes_per_iteration": size, "explicit_pinned_host_bytes": size,
                "explicit_device_tensor_bytes": size, "warmup_iterations": 2, "verified_sample_bytes": 1024,
                "interpretation": "Pinned one-way transfers measured sequentially, with completion synchronization. Directions are never added and presented as a one-way rate; topology is not inferred."}
    return results


def run(args, torch_loader=None):
    for name, allowed in {"buffer_mib": (64, 128, 256), "matrix_size": (4096, 8192),
                          "samples": range(3, 16), "iterations": range(1, 21), "disk_mib": (16, 32, 64, 128)}.items():
        if type(getattr(args, name)) is not int or getattr(args, name) not in allowed:
            raise ValueError(f"{name} exceeds the declared experiment bounds")
    if type(args.gpu) is not bool or type(args.transfers) is not bool or args.transfers and not args.gpu:
        raise ValueError("GPU work must be explicitly selected; transfers require GPU")
    report = {"schema_version": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": platform.python_version(), "os": platform.system(), "machine": platform.machine(),
                              "logical_cpu_count": os.cpu_count()},
              "configuration": {key: getattr(args, key) for key in ("gpu", "transfers", "buffer_mib", "matrix_size", "samples", "iterations", "disk_mib")},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "gpu": {"status": "unknown", "reason": "not_requested"}, "benchmarks": {},
              "scope": "Empirical microbenchmarks under this run's conditions, not physical maxima, model latency predictions, architecture acceptance or global scheduling optimality.",
              "physical_peak_reference": None, "theoretical_model_seconds": None, "sources": SOURCES,
              "limits": {"explicit_device_tensors_bytes": max(3 * args.matrix_size ** 2 * 2, 2 * args.buffer_mib * MIB) if args.gpu else 0,
                         "explicit_cpu_buffers_bytes": max(3 * args.buffer_mib * MIB, 2 * args.disk_mib * MIB),
                         "disk_file_bytes": args.disk_mib * MIB if args.disk_dir else 0,
                         "cuda_context_library_allocator_overhead": "Not included in tensor budget; allocated/reserved peaks are observed, not a hard process cap."},
              "notes": ["Repeated samples describe within-run variation, not independent machines or statistical confidence intervals.",
                        "Short warmup does not guarantee steady clocks; clocks and power are not changed.",
                        "A caller should bound process wall time and own its lifecycle; CUDA/OS calls can block beyond nominal work limits."]}
    torch = None
    try:
        report["benchmarks"].update(cpu_cases(args.buffer_mib * MIB, args.samples, args.iterations))
        if args.disk_dir:
            report["benchmarks"]["disk"] = disk_cases(args.disk_dir, args.disk_mib * MIB, args.samples)
        else:
            report["benchmarks"]["disk"] = {"status": "unknown", "reason": "not_requested"}
        if args.gpu:
            if torch_loader is None:
                import torch as torch_module
                torch = torch_module
            else:
                torch = torch_loader()
            report["gpu"] = {"status": "preflight_passed", **gpu_preflight(torch)}
            torch.cuda.reset_peak_memory_stats()
            report["benchmarks"].update(gpu_cases(torch, args.buffer_mib * MIB, args.matrix_size,
                                                  args.samples, args.iterations, args.transfers))
            report["gpu"].update(status="measured", peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                 peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report["status"] = "completed"
    except Exception as error:
        report.update(status="refused_or_failed", error_type=type(error).__name__, error=str(error)[:400])
        if args.gpu and report["gpu"]["status"] != "measured":
            report["gpu"].update(status="unknown", reason="requested_measurement_did_not_complete")
    finally:
        if torch is not None and torch.cuda.is_initialized():
            try:
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                report["gpu"]["final_allocated_bytes"] = torch.cuda.memory_allocated()
                report["gpu"]["explicit_allocations_released"] = torch.cuda.memory_allocated() == 0
                report["gpu"]["context_release"] = "requires_process_exit_not_cuda_empty_cache"
            except Exception as error:
                report["gpu"]["cleanup_error_type"] = type(error).__name__
                report["status"] = "refused_or_failed"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--gpu", action="store_true", help="Explicitly run full-A100 native BF16 calibration before model startup")
    result.add_argument("--transfers", action="store_true", help="Also measure sequential pinned H2D and D2H; requires --gpu")
    result.add_argument("--buffer-mib", type=int, choices=(64, 128, 256), default=128)
    result.add_argument("--matrix-size", type=int, choices=(4096, 8192), default=8192)
    result.add_argument("--samples", type=int, choices=range(3, 16), default=7)
    result.add_argument("--iterations", type=int, choices=range(1, 21), default=5)
    result.add_argument("--disk-dir", type=Path, help="Existing directory for one owned temporary file; only that file is removed")
    result.add_argument("--disk-mib", type=int, choices=(16, 32, 64, 128), default=32)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.transfers and not args.gpu:
        cli.error("--transfers requires --gpu")
    if args.disk_dir and not args.disk_dir.is_dir():
        cli.error("--disk-dir must already exist")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "gpu": report["gpu"]["status"],
                      "measurements": list(report["benchmarks"])}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
