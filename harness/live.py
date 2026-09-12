"""The live workload: continuous inference on the current config in a background
thread, feeding a rolling window that /telemetry reads.

Paced at LIVE_FPS to mimic a camera stream (and keep Orin thermals stable).
The GPU lock is held per frame; a benchmark holds it for its whole run, which
pauses this loop — the ring keeps serving its last stats so the chart stays flat.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .config import SETTINGS, DeployConfig
from .runtimes import RuntimeCache
from .timing import percentiles, timed_infer

log = logging.getLogger("harness.live")


class FrameSource:
    """Pre-decoded BGR frames. Cycles a directory of JPEGs, or a looping video via
    LIVE_SOURCE=video:<path>. Falls back to synthetic frames if nothing is found so
    the harness (and the CPU-mode contract test) always boots."""

    def __init__(self, frames_dir: Path, source: str = "", max_frames: int = 64):
        self.frames: list[np.ndarray] = []
        self.origin = "synthetic"
        self._i = 0
        import cv2

        if source.startswith("video:"):
            path = source.split(":", 1)[1]
            cap = cv2.VideoCapture(path)
            while len(self.frames) < max_frames:
                ok, f = cap.read()
                if not ok:
                    break
                self.frames.append(f)
            cap.release()
            if self.frames:
                self.origin = f"video:{path} ({len(self.frames)} frames)"
        if not self.frames and frames_dir.exists():
            for p in sorted(frames_dir.glob("*.jpg"))[:max_frames]:
                f = cv2.imread(str(p))
                if f is not None:
                    self.frames.append(f)
            if self.frames:
                self.origin = f"{frames_dir} ({len(self.frames)} jpgs)"
        if not self.frames:
            rng = np.random.default_rng(0)
            self.frames = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(8)]
            log.warning("no frames found in %s — using synthetic noise frames", frames_dir)

    def next(self, n: int = 1) -> list[np.ndarray]:
        out = []
        for _ in range(n):
            out.append(self.frames[self._i % len(self.frames)])
            self._i += 1
        return out

    def fixed(self, n: int = 1) -> list[np.ndarray]:
        """Same frames every call — used by the benchmark so preprocessing cost is constant."""
        return [self.frames[i % len(self.frames)] for i in range(n)]


class RingBuffer:
    def __init__(self, maxlen: int):
        self._d: deque[tuple[float, float]] = deque(maxlen=maxlen)  # (ts, latency_ms)
        self._lock = threading.Lock()
        self.last_p50: float | None = None
        self.last_p95: float | None = None

    def push(self, ms: float) -> None:
        with self._lock:
            self._d.append((time.time(), ms))

    def reset(self) -> None:
        with self._lock:
            self._d.clear()

    def stats(self) -> dict:
        with self._lock:
            items = list(self._d)
        if not items:
            return {"p50": self.last_p50, "p95": self.last_p95, "n": 0, "fps_actual": 0.0}
        xs = [ms for _, ms in items]
        p50, p95 = percentiles(xs)
        self.last_p50, self.last_p95 = p50, p95
        span = items[-1][0] - items[0][0]
        fps = (len(items) - 1) / span if span > 0 and len(items) > 1 else 0.0
        return {"p50": p50, "p95": p95, "n": len(items), "fps_actual": fps}


class LiveLoop(threading.Thread):
    def __init__(self, cache: RuntimeCache, gpu_lock: threading.Lock, frames: FrameSource):
        super().__init__(name="live-loop", daemon=True)
        self.cache = cache
        self.gpu_lock = gpu_lock
        self.frames = frames
        self.ring = RingBuffer(SETTINGS.window)
        self.current: DeployConfig | None = None
        self._rt = None
        self._skip = 0
        self._stop = threading.Event()
        self._cfg_lock = threading.Lock()
        self.benchmarking = False
        self.started_at: float | None = None
        self.frames_done = 0
        self.last_error: str | None = None
        self.last_stats = None

    def set_config(self, cfg: DeployConfig) -> None:
        rt = self.cache.get(cfg)  # loads under gpu_lock if needed; raises RuntimeUnavailable
        with self._cfg_lock:
            self.current = cfg
            self._rt = rt
            self._skip = 3
            self.ring.reset()
        log.info("live config → %s (%s)", cfg.label(), rt.backend_actual)

    def pause_for_benchmark(self):
        """Context manager: hold the GPU lock for the caller and mark telemetry."""
        loop = self

        class _Ctx:
            def __enter__(self_inner):
                loop.gpu_lock.acquire()
                loop.benchmarking = True

            def __exit__(self_inner, *exc):
                loop.benchmarking = False
                loop._skip = max(loop._skip, 2)  # clock ramp after idle
                loop.gpu_lock.release()

        return _Ctx()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        self.started_at = time.time()
        period = 1.0 / SETTINGS.live_fps if SETTINGS.live_fps > 0 else 0.0
        next_t = time.perf_counter()
        while not self._stop.is_set():
            with self._cfg_lock:
                rt, cfg = self._rt, self.current
            if rt is None:
                time.sleep(0.05)
                continue
            frames = self.frames.next(cfg.batch_size)
            try:
                with self.gpu_lock:
                    stats = timed_infer(rt, frames)
                self.last_stats = stats
                self.last_error = None
                if self._skip > 0:
                    self._skip -= 1
                else:
                    self.ring.push(stats.latency_ms)
                self.frames_done += 1
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("live inference failed")
                time.sleep(0.5)
                continue
            if period:
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()  # can't keep up; don't accumulate debt
