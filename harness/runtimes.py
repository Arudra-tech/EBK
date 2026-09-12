"""Runtime abstraction: how a DeployConfig actually executes on this device.

The HTTP ``runtime`` string ("pytorch" / "tensorrt") is the *deployment choice*
the agent reasons about. ``backend_actual`` is what really ran, which normally
matches but can be substituted via HARNESS_BACKEND_MAP for the failure ladder:

    HARNESS_BACKEND_MAP="pytorch=onnx"                # torch CUDA broken, GPU+TRT fine
    HARNESS_BACKEND_MAP="pytorch=cpu,tensorrt=cpu"    # no GPU at all

Substitution is always surfaced in /device and in benchmark extras — never hidden.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

import numpy as np

from . import artifacts
from .config import SETTINGS, DeployConfig
from .timing import InferStats

log = logging.getLogger("harness.runtimes")

BACKENDS = {"pytorch", "tensorrt", "onnx", "cpu"}


class RuntimeUnavailable(RuntimeError):
    """Raised when a config cannot be loaded on this device (missing engine, no CUDA, ...)."""


_DEFAULT_CPU_THREADS: int | None = None


def _default_cpu_threads() -> int:
    """torch's thread count before any runtime touched it (set_num_threads is global)."""
    global _DEFAULT_CPU_THREADS
    if _DEFAULT_CPU_THREADS is None:
        import torch

        _DEFAULT_CPU_THREADS = torch.get_num_threads()
    return _DEFAULT_CPU_THREADS


class Runtime(ABC):
    backend: str
    backend_actual: str
    uses_cuda: bool = True

    def __init__(self, cfg: DeployConfig):
        self.cfg = cfg
        self.loaded = False

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def infer(self, frames: list[np.ndarray]) -> InferStats: ...

    def warm(self, frames: list[np.ndarray], n: int) -> None:
        for _ in range(n):
            self.infer(frames)

    def close(self) -> None:
        pass

    def describe(self) -> dict:
        return {"config": self.cfg.model_dump(), "backend_actual": self.backend_actual, "loaded": self.loaded}


class UltralyticsRuntime(Runtime):
    """YOLO via ultralytics for all backends. The timed section is ``predict()`` on
    pre-decoded frames: letterbox + H2D + forward + NMS + results to host."""

    def __init__(self, cfg: DeployConfig, backend: str):
        super().__init__(cfg)
        self.backend = backend
        self.model = None
        self.half = False
        self.device: str | int = 0
        self.cpu_threads = 0
        self.uses_cuda = backend != "cpu"
        self.backend_actual = {
            "pytorch": "pytorch-cuda",
            "tensorrt": "tensorrt",
            "onnx": "onnxruntime-cuda",
            "cpu": "pytorch-cpu",
        }[backend]
        self.source_path = None

    def load(self) -> None:
        from ultralytics import YOLO

        cfg = self.cfg
        if self.backend == "tensorrt":
            p = artifacts.engine_path_for(cfg)
            if not p.exists():
                raise RuntimeUnavailable(f"missing TensorRT engine {p} — run tools/build_engines.py")
            self.model = YOLO(str(p), task="detect")
            self.half = False  # precision is baked into the engine
        elif self.backend == "onnx":
            p = artifacts.onnx_path(cfg.resolution)
            if not p.exists():
                raise RuntimeUnavailable(f"missing ONNX export {p} — run tools/build_engines.py --onnx")
            try:
                import onnxruntime as ort

                provs = ort.get_available_providers()
                if "CUDAExecutionProvider" not in provs and "TensorrtExecutionProvider" not in provs:
                    log.warning("onnxruntime has no CUDA provider (%s); ONNX backend will run on CPU", provs)
                    self.backend_actual = "onnxruntime-cpu"
                    self.uses_cuda = False
            except ImportError as e:
                raise RuntimeUnavailable(f"onnxruntime not installed: {e}")
            self.model = YOLO(str(p), task="detect")
            self.half = cfg.precision == "fp16"
        else:
            p = artifacts.pt_path()
            if not p.exists():
                raise RuntimeUnavailable(f"missing weights {p}")
            self.model = YOLO(str(p))
            self.half = cfg.precision == "fp16"
            if self.backend == "cpu":
                self.device = "cpu"
                import torch

                # Precision → thread count so the no-GPU ladder still yields distinct real numbers.
                # (set_num_threads is process-global; it is re-applied on every infer() below.)
                default_threads = _default_cpu_threads()
                self.cpu_threads = SETTINGS.cpu_threads or (
                    max(1, default_threads // 2) if cfg.precision == "fp32" else default_threads
                )
                n = self.cpu_threads
                torch.set_num_threads(n)
                self.half = False  # CPU fp16 is slower and unsupported for many ops
                self.backend_actual = f"pytorch-cpu-{n}t"
        self.source_path = str(p)
        self.loaded = True

    def infer(self, frames: list[np.ndarray]) -> InferStats:
        if self.backend == "cpu":
            import torch

            if torch.get_num_threads() != self.cpu_threads:
                torch.set_num_threads(self.cpu_threads)
        results = self.model.predict(
            frames,
            imgsz=self.cfg.resolution,
            half=self.half,
            device=self.device,
            conf=0.25,
            verbose=False,
        )
        r0 = results[0]
        sp = getattr(r0, "speed", None) or {}
        return InferStats(
            pre_ms=sp.get("preprocess"),
            inf_ms=sp.get("inference"),
            post_ms=sp.get("postprocess"),
            n_det=int(len(r0.boxes)) if getattr(r0, "boxes", None) is not None else None,
        )

    def close(self) -> None:
        self.model = None
        self.loaded = False


def resolve_backend(cfg: DeployConfig) -> str:
    overrides = SETTINGS.backend_overrides()
    backend = overrides.get(cfg.runtime, cfg.runtime)
    if backend not in BACKENDS:
        raise RuntimeUnavailable(f"unknown backend {backend!r} in HARNESS_BACKEND_MAP")
    return backend


def make_runtime(cfg: DeployConfig) -> Runtime:
    backend = resolve_backend(cfg)
    try:
        import torch
    except ImportError as e:
        raise RuntimeUnavailable(f"torch not importable: {e}")
    if backend in ("pytorch", "tensorrt") and not torch.cuda.is_available():
        raise RuntimeUnavailable(
            "torch.cuda.is_available() is False — fix the wheel or set HARNESS_BACKEND_MAP "
            "(pytorch=onnx | pytorch=cpu,tensorrt=cpu)"
        )
    return UltralyticsRuntime(cfg, backend)


class RuntimeCache:
    """One runtime object per config key. Loading happens under the GPU lock so it
    never races the live loop."""

    def __init__(self, gpu_lock: threading.Lock):
        self._rt: dict[tuple, Runtime] = {}
        self._errors: dict[str, str] = {}
        self._gpu_lock = gpu_lock
        self._meta_lock = threading.Lock()

    def peek(self, cfg: DeployConfig) -> Runtime | None:
        return self._rt.get(cfg.key())

    def get(self, cfg: DeployConfig) -> Runtime:
        rt = self._rt.get(cfg.key())
        if rt is not None and rt.loaded:
            return rt
        with self._gpu_lock:
            rt = self._rt.get(cfg.key())
            if rt is not None and rt.loaded:
                return rt
            rt = make_runtime(cfg)
            log.info("loading %s via %s", cfg.label(), rt.backend_actual)
            rt.load()
            with self._meta_lock:
                self._rt[cfg.key()] = rt
                self._errors.pop(cfg.slug(), None)
            return rt

    def preload(self, cfgs: list[DeployConfig]) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for c in cfgs:
            try:
                self.get(c)
                out[c.slug()] = None
            except Exception as e:  # keep booting; the candidate fails loudly at /benchmark time
                msg = f"{type(e).__name__}: {e}"
                log.warning("preload %s failed: %s", c.label(), msg)
                with self._meta_lock:
                    self._errors[c.slug()] = msg
                out[c.slug()] = msg
        return out

    def status(self) -> dict:
        return {
            "loaded": [rt.describe() for rt in self._rt.values() if rt.loaded],
            "errors": dict(self._errors),
        }
