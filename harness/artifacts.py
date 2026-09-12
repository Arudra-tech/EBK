"""Artifact path resolution — model weights, ONNX exports, TensorRT engines, accuracy tables.

Engines are keyed by device: TensorRT engines are not portable across GPUs or
TensorRT versions, so ``artifacts/<device_key>/`` holds the ones built on this box.
"""

import re
from pathlib import Path

from .config import SETTINGS, DeployConfig


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "unknown"


def gpu_name() -> str:
    """Best-effort GPU name without importing torch (cheap; used for device_key)."""
    # Jetson: device-tree model string
    dt = Path("/proc/device-tree/model")
    if dt.exists():
        try:
            return dt.read_bytes().decode("utf-8", "ignore").strip("\x00 \n")
        except OSError:
            pass
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
            return name.decode() if isinstance(name, bytes) else str(name)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def trt_version() -> str | None:
    try:
        import tensorrt

        return tensorrt.__version__
    except Exception:
        return None


_DEVICE_KEY: str | None = None


def device_key() -> str:
    """e.g. ``nvidia-gb10_trt10.13`` / ``nvidia-jetson-orin-nano-developer-kit_trt10.3`` / ``cpu_notrt``."""
    global _DEVICE_KEY
    if _DEVICE_KEY is None:
        name = _slug(gpu_name())
        v = trt_version()
        vs = "trt" + ".".join(v.split(".")[:2]) if v else "notrt"
        _DEVICE_KEY = f"{name}_{vs}"
    return _DEVICE_KEY


def model_stem() -> str:
    return Path(SETTINGS.model).stem


def artifacts_dir() -> Path:
    return SETTINGS.artifacts_dir


def device_dir() -> Path:
    return artifacts_dir() / device_key()


def pt_path() -> Path:
    p = Path(SETTINGS.model)
    if p.is_absolute() or p.exists():
        return p
    return artifacts_dir() / p.name


def onnx_path(resolution: int) -> Path:
    return artifacts_dir() / f"{model_stem()}_{resolution}.onnx"


def engine_path(precision: str, resolution: int, batch_size: int = 1) -> Path:
    return device_dir() / f"{model_stem()}_{precision}_{resolution}_b{batch_size}.engine"


def engine_path_for(cfg: DeployConfig) -> Path:
    return engine_path(cfg.precision, cfg.resolution, cfg.batch_size)


def accuracy_path() -> Path:
    if SETTINGS.accuracy_table:
        return Path(SETTINGS.accuracy_table)
    return artifacts_dir() / f"accuracy_{device_key()}_{model_stem()}.json"


def missing_engines(configs: list[DeployConfig]) -> list[Path]:
    """Engines required by these configs that aren't on disk. Respects HARNESS_BACKEND_MAP
    (a config mapped to onnx/cpu needs no engine)."""
    overrides = SETTINGS.backend_overrides()
    out = []
    for c in configs:
        if overrides.get(c.runtime, c.runtime) == "tensorrt":
            p = engine_path_for(c)
            if not p.exists():
                out.append(p)
    return out
