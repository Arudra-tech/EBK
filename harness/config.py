"""Deployment config model, accepted value sets, and env-driven settings.

DeployConfig and validate_supported() mirror simulator/main.py exactly — the
server's contract is the same four fields with the same value sets.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

RUNTIMES = {"pytorch", "tensorrt"}
PRECISIONS = {"fp32", "fp16", "int8"}
RESOLUTIONS = {640, 512, 416}
BATCH_SIZES = {1, 2, 4}

ConfigKey = tuple[str, str, int, int]


class DeployConfig(BaseModel):
    runtime: str = "pytorch"
    precision: str = "fp32"
    resolution: int = 640
    batch_size: int = 1

    def validate_supported(self) -> None:
        if self.runtime not in RUNTIMES:
            raise HTTPException(422, f"unsupported runtime {self.runtime}")
        if self.precision not in PRECISIONS:
            raise HTTPException(422, f"unsupported precision {self.precision}")
        if self.resolution not in RESOLUTIONS:
            raise HTTPException(422, f"unsupported resolution {self.resolution}")
        if self.batch_size not in BATCH_SIZES:
            raise HTTPException(422, f"unsupported batch_size {self.batch_size}")
        if self.runtime == "pytorch" and self.precision == "int8":
            # No eager-mode INT8 path for YOLO; the server never proposes this.
            raise HTTPException(422, "pytorch/int8 is not supported on this harness (use tensorrt/int8)")

    def key(self) -> ConfigKey:
        return (self.runtime, self.precision, self.resolution, self.batch_size)

    def label(self) -> str:
        rt = "TensorRT" if self.runtime == "tensorrt" else "PyTorch"
        return f"{self.precision.upper()}+{rt} @{self.resolution} b{self.batch_size}"

    def slug(self) -> str:
        return f"{self.runtime}_{self.precision}_{self.resolution}_b{self.batch_size}"


BASELINE = DeployConfig(runtime="pytorch", precision="fp32", resolution=640, batch_size=1)

# The configs the server's proposer ladder actually asks for (server/proposer.py) + baseline.
# Engines and accuracy entries must exist for all of these.
PROPOSED: list[DeployConfig] = [
    BASELINE,
    DeployConfig(runtime="tensorrt", precision="fp16", resolution=640, batch_size=1),
    DeployConfig(runtime="tensorrt", precision="int8", resolution=640, batch_size=1),
    DeployConfig(runtime="tensorrt", precision="fp16", resolution=512, batch_size=1),
    DeployConfig(runtime="tensorrt", precision="int8", resolution=512, batch_size=1),
    DeployConfig(runtime="tensorrt", precision="fp16", resolution=416, batch_size=1),
    DeployConfig(runtime="pytorch", precision="fp16", resolution=640, batch_size=1),
]


def all_configs(batches: tuple[int, ...] = (1,)) -> list[DeployConfig]:
    out = []
    for rt in sorted(RUNTIMES):
        for p in ("fp32", "fp16", "int8"):
            if rt == "pytorch" and p == "int8":
                continue
            for res in sorted(RESOLUTIONS, reverse=True):
                for b in batches:
                    out.append(DeployConfig(runtime=rt, precision=p, resolution=res, batch_size=b))
    return out


def _env(name: str, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


HARNESS_DIR = Path(__file__).resolve().parent
REPO_DIR = HARNESS_DIR.parent


@dataclass
class Settings:
    model: str = field(default_factory=lambda: _env("HARNESS_MODEL", "yolov8n.pt"))
    port: int = field(default_factory=lambda: _env("HARNESS_PORT", 8100))
    live_fps: float = field(default_factory=lambda: _env("LIVE_FPS", 30.0))
    live_source: str = field(default_factory=lambda: _env("LIVE_SOURCE", ""))  # "" = frames dir, "video:<path>"
    window: int = field(default_factory=lambda: _env("WINDOW", 90))
    warmup: int = field(default_factory=lambda: _env("WARMUP", 10))
    iters_min: int = field(default_factory=lambda: _env("ITERS_MIN", 20))
    iters_max: int = field(default_factory=lambda: _env("ITERS_MAX", 50))
    bench_budget_s: float = field(default_factory=lambda: _env("BENCH_BUDGET_S", 18.0))
    thermal_max_c: float = field(default_factory=lambda: _env("THERMAL_MAX_C", 0.0))  # 0 = per-device default
    thermal_wait_s: float = field(default_factory=lambda: _env("THERMAL_WAIT_S", 8.0))
    preload: str = field(default_factory=lambda: _env("PRELOAD", "all"))  # all | baseline | none
    backend_map: str = field(default_factory=lambda: _env("HARNESS_BACKEND_MAP", ""))  # "pytorch=onnx,tensorrt=cpu"
    accuracy_table: str = field(default_factory=lambda: _env("ACCURACY_TABLE", ""))
    device_profile: str = field(default_factory=lambda: _env("DEVICE_PROFILE", ""))  # edge-hi | edge-lo, "" = auto
    artifacts_dir: Path = field(default_factory=lambda: Path(_env("HARNESS_ARTIFACTS", str(HARNESS_DIR / "artifacts"))))
    data_dir: Path = field(default_factory=lambda: Path(_env("HARNESS_DATA", str(HARNESS_DIR / "data"))))
    cpu_threads: int = field(default_factory=lambda: _env("CPU_THREADS", 0))  # 0 = torch default

    def backend_overrides(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for part in self.backend_map.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip()
        return out


SETTINGS = Settings()
