"""EBK hardware harness — the real workload service (GB10 / Jetson Orin Nano).

Same HTTP contract as simulator/main.py, answered with measured numbers:

    GET  /telemetry   -> rolling p50/p95 of the live loop + probe sample
    GET  /config      -> current live deployment config
    POST /config      -> switch the live workload (also the regression-injection hook)
    POST /benchmark   -> warmup + measured iterations on the candidate; live config untouched
    GET  /accuracy    -> precomputed held-out accuracy for a config
    GET  /device      -> device fingerprint (not needed by the server; for the dashboard/pitch)
    GET  /health      -> readiness

Run with ONE worker — this process owns the GPU:
    uvicorn harness.main:app --host 0.0.0.0 --port 8100
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from . import artifacts
from .accuracy import AccuracyTable
from .bench import run_benchmark
from .config import BASELINE, PROPOSED, SETTINGS, DeployConfig
from .device import DeviceProbe, NullProbe, default_profile, detect, device_info
from .live import FrameSource, LiveLoop
from .runtimes import RuntimeCache, RuntimeUnavailable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("harness")


class State:
    probe: DeviceProbe
    cache: RuntimeCache
    live: LiveLoop
    accuracy: AccuracyTable
    gpu_lock: threading.Lock
    preload_errors: dict[str, str | None]
    boot_ts: float
    bench_lock: threading.Lock
    last_bench: dict | None = None
    boot_phase: str = "starting"
    boot_error: str | None = None
    device_info: dict | None = None


S = State()


def _phase(p: str) -> None:
    S.boot_phase = p
    log.info("boot: %s", p)


def _boot() -> None:
    """Everything slow happens here, off the event loop, so the port is bound and
    /health answers from the first second. Watch `boot_phase` to see where it is."""
    try:
        _phase("probe")
        S.probe = detect()
        S.probe.start()
        _phase("accuracy table")
        S.accuracy = AccuracyTable(artifacts.accuracy_path())
        _phase("frames")
        frames = FrameSource(SETTINGS.data_dir / "frames", SETTINGS.live_source)
        log.info("frames: %s", frames.origin)
        S.live = LiveLoop(S.cache, S.gpu_lock, frames)
        S.live.start()

        _phase(f"loading baseline {BASELINE.label()} (first CUDA init: 5-30 s)")
        S.preload_errors = S.cache.preload([BASELINE])
        if S.preload_errors.get(BASELINE.slug()) is None:
            S.live.set_config(BASELINE)
        else:
            log.error("baseline failed to load: %s — /telemetry will report no latency", S.preload_errors)
        log.info("harness up: device_key=%s profile=%s model=%s", artifacts.device_key(), default_profile(S.probe), SETTINGS.model)

        if SETTINGS.preload == "all":
            rest = [c for c in PROPOSED if c.key() != BASELINE.key()]
            for i, c in enumerate(rest, 1):
                # a benchmark request must never queue behind an engine deserialize
                while S.bench_lock.locked():
                    time.sleep(0.1)
                _phase(f"preloading {i}/{len(rest)} {c.label()}")
                S.preload_errors.update(S.cache.preload([c]))
            log.info("preload done: %s", S.cache.status()["errors"] or "all ok")
        _phase("device info")
        S.device_info = device_info(S.probe)  # shells out to nvidia-smi/nvpmodel once, not per request
        _phase("ready")
    except Exception as e:  # never die silently — surface it in /health
        S.boot_error = f"{type(e).__name__}: {e}"
        _phase(f"FAILED: {S.boot_error}")
        log.exception("boot failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    S.boot_ts = time.time()
    S.gpu_lock = threading.Lock()
    S.bench_lock = threading.Lock()
    S.probe = NullProbe()  # placeholders until _boot swaps the real ones in
    S.accuracy = AccuracyTable(artifacts.accuracy_path())
    S.cache = RuntimeCache(S.gpu_lock)
    S.live = LiveLoop(S.cache, S.gpu_lock, FrameSource.empty())
    S.preload_errors = {}
    threading.Thread(target=_boot, name="boot", daemon=True).start()
    yield
    S.live.stop()
    S.probe.stop()


app = FastAPI(title="EBK Hardware Harness", lifespan=lifespan)


@app.get("/telemetry")
def telemetry():
    st = S.live.ring.stats()
    cfg = S.live.current
    d = S.probe.last
    p50 = st["p50"]
    if p50 is None:
        # Nothing measured yet (boot / baseline failed). The watcher needs a float.
        p50 = S.last_bench["median_latency_ms"] if S.last_bench else 0.0
    p95 = st["p95"] if st["p95"] is not None else p50
    bs = cfg.batch_size if cfg else 1
    return {
        "latency_ms": round(float(p50), 2),
        "p95_ms": round(float(p95), 2),
        "throughput_fps": round(bs / (p50 / 1000.0), 1) if p50 else 0.0,
        "gpu_util": round(float(d.gpu_util), 2),
        "temp_c": round(float(d.temp_c), 1),
        "config": cfg.model_dump() if cfg else BASELINE.model_dump(),
        # extras (ignored by the server, useful for the dashboard)
        "live_fps": round(st["fps_actual"], 1),
        "window_n": st["n"],
        "benchmarking": S.live.benchmarking,
        "power_w": d.power_w,
        "sm_clock_mhz": d.sm_clock_mhz,
        "cpu_util": d.cpu_util,
        "device": artifacts.device_key(),
        "profile": default_profile(S.probe),
        "live_error": S.live.last_error,
        "boot_phase": S.boot_phase,
    }


def _require_booted() -> None:
    if S.live.current is None:
        raise HTTPException(503, f"harness not ready: boot_phase={S.boot_phase!r}"
                                 + (f" error={S.boot_error}" if S.boot_error else ""))


@app.get("/config")
def get_config():
    return (S.live.current or BASELINE).model_dump()


@app.post("/config")
async def set_config(cfg: DeployConfig):
    cfg.validate_supported()
    _require_booted()
    try:
        await run_in_threadpool(S.live.set_config, cfg)
    except RuntimeUnavailable as e:
        raise HTTPException(503, str(e))
    return {"applied": True, "config": cfg.model_dump()}


@app.post("/benchmark")
async def benchmark(cfg: DeployConfig):
    cfg.validate_supported()
    _require_booted()
    if not S.bench_lock.acquire(blocking=False):
        raise HTTPException(409, "a benchmark is already running")
    try:
        res = await run_in_threadpool(run_benchmark, cfg, S.cache, S.live, S.probe)
    except RuntimeUnavailable as e:
        raise HTTPException(503, str(e))
    finally:
        S.bench_lock.release()
    out = res.as_dict()
    S.last_bench = out
    return out


@app.get("/accuracy")
def accuracy(runtime: str = "pytorch", precision: str = "fp32", resolution: int = 640, batch_size: int = 1):
    cfg = DeployConfig(runtime=runtime, precision=precision, resolution=resolution, batch_size=batch_size)
    cfg.validate_supported()
    if not S.accuracy.available:
        raise HTTPException(503, f"accuracy table missing: {S.accuracy.path} — run tools/precompute_accuracy.py")
    try:
        e = S.accuracy.lookup(cfg)
    except KeyError as k:
        raise HTTPException(503, f"no accuracy entry for {k} in {S.accuracy.path}")
    base = S.accuracy.baseline()
    acc = float(e["map50"])
    return {
        "accuracy": round(acc, 4),
        "delta_from_baseline": round(acc - base, 4) if base is not None else None,
        "metric": S.accuracy.metric,
        "map50_95": e.get("map50_95"),
        "n_images": (S.accuracy.data or {}).get("n_images"),
        "precomputed": True,
    }


@app.get("/device")
def device():
    return {
        **(S.device_info or {"probe": S.probe.kind, "note": f"booting: {S.boot_phase}"}),
        "boot_phase": S.boot_phase,
        "sample": S.probe.last.as_dict(),
        "runtimes": S.cache.status(),
        "accuracy_table": S.accuracy.summary(),
        "frames": S.live.frames.origin,
        "settings": {
            "live_fps": SETTINGS.live_fps, "window": SETTINGS.window, "warmup": SETTINGS.warmup,
            "iters": [SETTINGS.iters_min, SETTINGS.iters_max], "bench_budget_s": SETTINGS.bench_budget_s,
            "preload": SETTINGS.preload,
        },
        "uptime_s": round(time.time() - S.boot_ts, 1),
        "last_benchmark": S.last_bench,
    }


@app.get("/health")
def health():
    missing = artifacts.missing_engines(PROPOSED)
    ready = S.live.current is not None and S.live.ring.stats()["n"] > 0
    return {
        "ok": S.boot_error is None,
        "ready": ready,
        "boot_phase": S.boot_phase,
        "boot_error": S.boot_error,
        "uptime_s": round(time.time() - S.boot_ts, 1),
        "live_config": S.live.current.model_dump() if S.live.current else None,
        "frames_done": S.live.frames_done,
        "engines_missing": [str(p) for p in missing],
        "accuracy_missing": S.accuracy.missing(PROPOSED) if S.accuracy.available else "table missing",
        "preload_errors": {k: v for k, v in S.preload_errors.items() if v},
    }
