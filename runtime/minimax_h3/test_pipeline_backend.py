"""CPU stage/ownership regressions. Tiny codec fixtures are not model generation."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import weakref

import numpy as np
from PIL import Image
import pytest

import pipeline_backend as pb
from model_backend import MiniMaxBackend, aligned_frames
from runtime_options import RuntimeOptions
from staged_backend import StagedMiniMaxBackend


class Backend(pb.PipelineBackendSupport):
    _aligned_frames = staticmethod(aligned_frames)
    def __init__(self):
        self._init_runtime(RuntimeOptions())
        self.pipe = object()

    def validate_request(self, *, steps):
        if steps < 2:
            raise ValueError("steps")

    @staticmethod
    def _reference_probe(path, kind):
        return 0 if kind == "image" else 5

    @staticmethod
    def _normalize_reference(path, target=None):
        target = target or path.with_name(path.stem + "_24fps.mp4")
        target.write_bytes(b"normalized codec fixture")
        return target


class Tensor:
    def __init__(self, value, device="cuda"):
        self.value, self.device = np.array(value), device

    def detach(self):
        return self

    def to(self, device, copy=False):
        return Tensor(self.value.copy() if copy else self.value, device)

    def cpu(self):
        return self.to("cpu", copy=True)

    def is_floating_point(self):
        return self.value.dtype.kind == "f"

    def float(self):
        return Tensor(self.value.astype(np.float32), self.device)

    def numpy(self):
        assert self.device == "cpu"
        return self.value


class Generator:
    def __init__(self, device):
        assert device == "cpu"

    def manual_seed(self, seed):
        self.seed = seed
        return self


@pytest.fixture
def torch(monkeypatch):
    calls = []
    value = SimpleNamespace(Tensor=Tensor, Generator=Generator, inference_mode=nullcontext,
                            from_numpy=lambda values: Tensor(values, "cpu"),
                            cuda=SimpleNamespace(synchronize=lambda: calls.append("sync"),
                                                 empty_cache=lambda: calls.append("empty_cache"),
                                                 is_initialized=lambda: False), calls=calls)
    monkeypatch.setitem(sys.modules, "torch", value)
    return value


@pytest.fixture
def ref_classes(monkeypatch):
    class Reference:
        def __init__(self, **values):
            self.__dict__.update(values)

        @classmethod
        def from_file(cls, path):
            with Image.open(path) as source:
                return cls(image=source.convert("RGB"))
    module = SimpleNamespace(**{name: Reference for name in (
        "MiniMaxH3ImageReference", "MiniMaxH3VideoReference", "MiniMaxH3AudioReference")})
    monkeypatch.setitem(sys.modules, "diffusers.modular_pipelines.minimax_h3", module)
    return Reference


def parameters(tmp_path, count=1):
    refs = []
    for i in range(count):
        path = tmp_path / f"image{i}.png"
        Image.new("RGB", (64, 32), (i * 30, 40, 50)).save(path)
        refs.append({"kind": "image", "path": str(path)})
    return dict(prompt="fixture", references=refs, seconds=5, width=512,
                height=288, steps=8, seed=42)


def result(backend, frames=5, cpu_threads=1):
    return pb.CPUResult([np.zeros((32, 64, 3), dtype=np.uint8) for _ in range(frames)],
                        np.zeros((2, round(frames / 24 * 32000)), dtype=np.float32),
                        32000, backend._new_metrics("generation"), cpu_threads, True)


def test_cpu_budget_accounts_for_output_and_decode_copies(tmp_path):
    backend = Backend()
    p = parameters(tmp_path)
    demand = backend.estimate_resources(p)
    assert demand["reference_retained_bytes"] == 64 * 32 * 4
    assert demand["output_retained_bytes"] >= 124 * 512 * 288 * 3
    assert demand["pipeline_reference_workspace_bytes"] == 2048 * 4096 * 20
    assert demand["ram_bytes"] == 3 * demand["reference_retained_bytes"] + 4 * demand["output_retained_bytes"] + demand["pipeline_reference_workspace_bytes"] + 256 * pb.MIB
    assert demand["cpu_threads"] == 1 and demand["disk_bytes"] > demand["output_retained_bytes"]
    assert "not calibrated" in demand["estimate_kind"]
    p["references"] = [{"kind": "video", "path": p["references"][0]["path"]}] * 4
    with pytest.raises(ValueError, match="Total reference"):
        backend.estimate_resources(p)


def test_prepare_budget_parallelizes_independent_cpu_refs_and_preserves_order(tmp_path, monkeypatch, ref_classes):
    backend, barrier = Backend(), threading.Barrier(2)
    original = pb.decode_reference
    def decode(kind, path, classes):
        barrier.wait(2)
        return original(kind, path, classes)
    monkeypatch.setattr(pb, "decode_reference", decode)
    prepared = backend.prepare(cpu_threads=2, **parameters(tmp_path, 2))
    assert [ref.image.getpixel((0, 0))[0] for ref in prepared.references] == [0, 30]
    assert prepared.metrics.data["scheduling"][0]["workers"] == 2
    backend.cleanup(prepared=prepared)
    backend.cleanup(prepared=prepared)
    assert prepared.references == [] and prepared.cleaned


def test_failed_prepare_joins_children_and_cleans_normalized_files(tmp_path, monkeypatch, ref_classes):
    backend = Backend()
    p = parameters(tmp_path, 2)
    for ref in p["references"]:
        ref["kind"] = "video"
    barrier = threading.Barrier(2)
    finished = []
    def decode(kind, path, classes):
        barrier.wait(2)
        finished.append(path.name)
        raise ValueError("decode rejected")
    monkeypatch.setattr(pb, "decode_reference", decode)
    with pytest.raises(ValueError, match="decode rejected"):
        backend.prepare(cpu_threads=2, **p)
    assert len(finished) == 2
    assert not list(tmp_path.glob('*_24fps.mp4'))


def test_infer_detaches_audio_and_synchronizes_before_handoff(torch, monkeypatch):
    backend, raw_refs = Backend(), []
    def infer(prepared):
        audio = Tensor(np.ones((2, 40), np.float32))
        raw_refs.append(weakref.ref(audio))
        return {"videos": [[Image.new("RGB", (32, 32))]], "audio": [audio], "sampling_rate": 32000}
    backend._infer_raw = infer
    prepared = pb.Prepared({}, [object()], [], backend._new_metrics("generation"))
    output = backend.infer(prepared)
    assert output.cpu_handoff_complete
    assert isinstance(output.audio, np.ndarray) and output.audio.dtype == np.float32
    assert torch.calls == ["sync", "empty_cache"]
    assert raw_refs[0]() is None and prepared.references == []
    assert output.video[0].dtype == np.uint8


def test_infer_failure_also_finishes_gpu_cleanup_before_propagating(torch):
    backend = Backend()
    backend._infer_raw = lambda prepared: (_ for _ in ()).throw(RuntimeError("GPU failed"))
    prepared = pb.Prepared({}, [object()], [], backend._new_metrics("generation"))
    with pytest.raises(RuntimeError, match="GPU failed"):
        backend.infer(prepared)
    assert torch.calls == ["sync", "empty_cache"]
    assert prepared.references == [] and prepared.metrics.data["status"] == "failed"


def test_context_metrics_do_not_cross_jobs_during_overlap():
    backend, barrier = Backend(), threading.Barrier(2)
    a, b = backend._new_metrics("generation"), backend._new_metrics("generation")
    def run(metrics, name):
        with backend.metrics_context(metrics), backend._stage(name):
            barrier.wait(2)
            assert backend.current_phase == name
        return backend._active_metrics
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, a, "prepare.a"), pool.submit(run, b, "encode.b")]
        assert [f.result() for f in futures] == [None, None]
    assert [s["phase"] for s in a.data["stages"]] == ["prepare.a"]
    assert [s["phase"] for s in b.data["stages"]] == ["encode.b"]
    assert backend._active_metrics is None


def test_encode_and_cleanup_are_cpu_only_and_produce_decodable_media(tmp_path, torch):
    import av
    backend = Backend()
    output = result(backend, cpu_threads=2)
    path = tmp_path / "result.mp4"
    metrics = backend.encode(output, path)
    with av.open(str(path)) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        assert sum(1 for _ in container.decode(video=0)) == 5
    with av.open(str(path)) as container:
        assert container.streams.audio[0].codec_context.name == "aac"
        assert sum(f.samples for f in container.decode(audio=0)) > 0
    assert metrics["status"] == "succeeded" and torch.calls == []
    assert json.loads(path.with_suffix('.metrics.json').read_text())["status"] == "succeeded"
    backend.cleanup(result=output)
    backend.cleanup(result=output)
    assert output.cleaned and output.audio is None and output.video == [] and torch.calls == []


def test_encode_rejects_gpu_ownership_not_yet_transferred(tmp_path):
    output = result(Backend())
    output.cpu_handoff_complete = False
    with pytest.raises(RuntimeError, match="ownership handoff"):
        pb.encode_cpu_media(output, tmp_path / "no.mp4")


def test_real_cpu_video_and_audio_reference_decode_keep_rates(tmp_path, torch, ref_classes):
    backend = Backend()
    output = result(backend, frames=48)
    path = tmp_path / "reference.mp4"
    pb.encode_cpu_media(output, path)
    classes = {kind: ref_classes for kind in ("image", "video", "audio")}
    video = pb.decode_reference("video", path, classes)
    audio = pb.decode_reference("audio", path, classes)
    assert video.frames.shape == (48, 32, 64, 3)
    assert video.frames.dtype == np.uint8 and video.fps == 24
    assert video.sample_rate == audio.sample_rate == 32000
    assert video.audio.device == audio.audio.device == "cpu"
    assert video.audio.value.shape[0] == audio.audio.value.shape[0] == 2
    assert torch.calls == []


@pytest.mark.parametrize("backend_type", [MiniMaxBackend, StagedMiniMaxBackend])
def test_both_backends_expose_real_stages_without_fallback(backend_type):
    backend = backend_type(RuntimeOptions())
    assert all(callable(getattr(backend, name)) for name in ("prepare", "infer", "encode", "cleanup", "estimate_resources", "generate"))
    assert backend_type._infer_raw is not getattr(pb.PipelineBackendSupport, '_infer_raw', None)


def test_resident_infer_keeps_original_workflow_arguments(torch):
    backend, received = MiniMaxBackend(RuntimeOptions()), {}
    def pipe(**values):
        received.update(values)
        return {"videos": [], "audio": [], "sampling_rate": 32000}
    backend.pipe = pipe
    p = dict(prompt="fixture", seconds=5, width=512, height=288, steps=8, seed=42)
    refs = [object()]
    backend._infer_raw(pb.Prepared(p, refs, [], backend._new_metrics("generation")))
    assert received["references"] is refs
    assert received["num_frames"] == 124 and received["num_inference_steps"] == 8
    assert received["generator"].seed == 42
    assert received["output"] == ["videos", "audio", "sampling_rate"]
    assert received["output_type"] == "pil"


def test_staged_infer_keeps_text_vae_denoise_decode_under_one_stage(torch, monkeypatch):
    class State:
        def __init__(self, values): self.values = values
        def get(self, key): return self.values[key]
        def set(self, key, value): self.values[key] = value
    monkeypatch.setitem(sys.modules, "diffusers.modular_pipelines.modular_pipeline", SimpleNamespace(PipelineState=State))
    backend, calls = StagedMiniMaxBackend(RuntimeOptions()), []
    backend.pipe = SimpleNamespace(unload_components=lambda name: calls.append("unload." + name),
                                   blocks=SimpleNamespace(sub_blocks={"before_encode": object()}))
    backend._switch = lambda phase: calls.append("switch." + phase)
    def run(state, phase):
        calls.append("run." + phase)
        if phase == "encode": state.set("prompt_embeds", Tensor([1]))
        if phase == "decode":
            state.values.update(videos=[[Image.new('RGB',(32,32))]],audio=[Tensor(np.zeros((2,5)))],sampling_rate=32000)
        return state
    backend._run = run
    p = dict(prompt="fixture", seconds=5, width=512, height=288, steps=8, seed=42)
    prepared = pb.Prepared(p, [], [], backend._new_metrics("generation"))
    output = backend.infer(prepared)
    assert calls == ['switch.encode','run.encode','switch.denoise','run.denoise','unload.transformer_ref','run.decode']
    assert output.cpu_handoff_complete and torch.calls == ['sync','empty_cache']


def test_standalone_generate_preserves_sequential_stage_lifetime(tmp_path, torch, ref_classes):
    backend = Backend()
    backend._infer_raw = lambda p: {"videos": [[Image.new('RGB',(32,32)) for _ in range(5)]],
                                   "audio": [Tensor(np.zeros((2, 6667)))],"sampling_rate":32000}
    path = tmp_path / 'standalone.mp4'
    backend.generate(output_path=path, **parameters(tmp_path))
    assert path.is_file() and backend.last_metrics['status']=='succeeded'
    phases=[x['phase'] for x in backend.last_metrics['stages']]
    assert phases.index('references.prepare') < phases.index('pipeline') < phases.index('encode_mp4')
    assert phases[-1]=='generate.total'


@pytest.fixture
def native_pools(monkeypatch, torch):
    import preflight
    import threadpoolctl
    state = {"intra": 12, "interop": 4, "pool": 12, "mutations": [], "interop_locked": False,
             "preflight": []}
    monkeypatch.setattr(preflight, "check_resources", lambda **kw: state["preflight"].append(kw))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_MAX_ACTIVE_LEVELS",
                "OMP_DYNAMIC", "MKL_DYNAMIC", "TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    def set_intra(value):
        state["mutations"].append(("intra", value))
        state["intra"] = value
    def set_interop(value):
        state["mutations"].append(("interop", value))
        if state["interop_locked"]:
            raise RuntimeError("already initialized")
        state["interop"] = value
    def limit(*, limits):
        state["mutations"].append(("pool", limits))
        state["pool"] = limits
        return object()
    torch.set_num_threads = set_intra
    torch.set_num_interop_threads = set_interop
    torch.get_num_threads = lambda: state["intra"]
    torch.get_num_interop_threads = lambda: state["interop"]
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", limit)
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [
        {"internal_api": "openblas", "user_api": "blas", "num_threads": state["pool"]}])
    return state


@pytest.mark.parametrize("backend_type,residency", [(MiniMaxBackend, "resident"), (StagedMiniMaxBackend, "staged")])
def test_native_budget_applied_once_before_load_and_actual_pools_reported(native_pools, backend_type, residency):
    backend = backend_type(RuntimeOptions())
    report = backend.configure_cpu_budget(2)
    assert report["configured"] and report["torch_num_threads"] == 2
    assert report["torch_num_interop_threads"] == 1
    assert report["native_pools"][0]["num_threads"] == 2
    assert report["environment_defaults"]["TOKENIZERS_PARALLELISM"] == "false"
    assert native_pools["preflight"] == [{"residency": residency}]
    assert native_pools["mutations"] == [("intra", 2), ("interop", 1), ("pool", 2)]
    backend._loaded = True
    assert backend.configure_cpu_budget(2) == report  # Same request is read-only.
    assert len(native_pools["mutations"]) == 3
    with pytest.raises(RuntimeError, match="cannot be reconfigured"):
        backend.configure_cpu_budget(3)


def test_late_first_configuration_rejected_without_native_mutation(native_pools):
    backend = MiniMaxBackend(RuntimeOptions())
    backend._loaded = True
    with pytest.raises(RuntimeError, match="before model loading"):
        backend.configure_cpu_budget(1)
    assert native_pools["mutations"] == []


def test_uncontrollable_interop_threads_fail_readiness(native_pools):
    backend = MiniMaxBackend(RuntimeOptions())
    native_pools.update(interop=2, interop_locked=True)
    with pytest.raises(RuntimeError, match="Cannot enforce one torch interop"):
        backend.configure_cpu_budget(1)
    assert not hasattr(backend, "_native_cpu_requested")


def test_already_initialized_interop_one_is_reported_honestly(native_pools):
    native_pools.update(interop=1, interop_locked=True)
    report = MiniMaxBackend(RuntimeOptions()).configure_cpu_budget(1)
    assert report["interop_configuration"] == "already_initialized_at_one"
    assert report["torch_num_interop_threads"] == 1


def test_after_load_audit_rejects_new_overbudget_pool_without_mutation(native_pools):
    backend = MiniMaxBackend(RuntimeOptions())
    backend.configure_cpu_budget(1)
    original_mutations = list(native_pools["mutations"])
    native_pools["pool"] = 8
    with pytest.raises(RuntimeError, match="Native pool CPU limit is not confirmed"):
        backend.cpu_budget_snapshot()
    assert native_pools["mutations"] == original_mutations


def test_preflight_failure_remains_before_torch_configuration(native_pools, monkeypatch):
    import preflight
    def fail(**kwargs):
        raise RuntimeError("Deploy this project on a Linux NVIDIA GPU machine.")
    monkeypatch.setattr(preflight, "check_resources", fail)
    with pytest.raises(RuntimeError, match="Linux NVIDIA"):
        MiniMaxBackend(RuntimeOptions()).configure_cpu_budget(1)
    assert native_pools["mutations"] == []
