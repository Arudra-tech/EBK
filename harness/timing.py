"""Timing helpers. One definition of "latency" for every runtime:

    letterbox preprocess + H2D + forward + NMS + results to host

measured with a wall clock around predict() and a CUDA sync so async kernels are
included. Disk I/O and JPEG decode are excluded (frames are pre-decoded).
"""

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class InferStats:
    latency_ms: float = 0.0
    pre_ms: float | None = None
    inf_ms: float | None = None
    post_ms: float | None = None
    n_det: int | None = None
    extra: dict = field(default_factory=dict)


def cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def timed_infer(rt, frames: list[np.ndarray]) -> InferStats:
    t0 = time.perf_counter()
    stats = rt.infer(frames)
    if rt.uses_cuda:
        cuda_sync()
    stats.latency_ms = (time.perf_counter() - t0) * 1000.0
    return stats


def percentiles(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    p50, p95 = np.percentile(np.asarray(xs, dtype=np.float64), [50, 95])
    return float(p50), float(p95)
