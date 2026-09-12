"""Benchmark a candidate config on this device.

Protocol: thermal guard → WARMUP iterations → N measured iterations (N adaptive
to fit BENCH_BUDGET_S, clamped to [ITERS_MIN, ITERS_MAX]) → median + p95.
The live loop is paused for the whole run (GPU lock) so the measurement is clean.
Device state before/after is recorded in ``extras`` so the pitch can show it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from . import artifacts
from .config import SETTINGS, DeployConfig
from .device import DeviceProbe
from .live import LiveLoop
from .runtimes import RuntimeCache
from .timing import percentiles, timed_infer

log = logging.getLogger("harness.bench")


@dataclass
class BenchResult:
    median_latency_ms: float
    p95_latency_ms: float
    throughput_fps: float
    temp_c: float
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "median_latency_ms": round(self.median_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "throughput_fps": round(self.throughput_fps, 2),
            "temp_c": round(self.temp_c, 1),
            "extras": self.extras,
        }


def thermal_guard(probe: DeviceProbe) -> dict:
    limit = probe.thermal_max_c()
    waited = 0.0
    start_temp = probe.last.temp_c
    while probe.last.temp_c > limit and waited < SETTINGS.thermal_wait_s:
        log.warning("thermal guard: %.1f°C > %.0f°C, waiting", probe.last.temp_c, limit)
        time.sleep(1.0)
        waited += 1.0
    return {
        "thermal_limit_c": limit,
        "thermal_wait_s": waited,
        "thermal_throttle_risk": probe.last.temp_c > limit,
        "temp_at_start_c": start_temp,
    }


def run_benchmark(cfg: DeployConfig, cache: RuntimeCache, live: LiveLoop, probe: DeviceProbe) -> BenchResult:
    t_wall = time.perf_counter()
    rt = cache.get(cfg)  # may deserialize an engine; still well inside the 30 s budget
    t_load = time.perf_counter() - t_wall
    frames = live.frames.fixed(cfg.batch_size)

    with live.pause_for_benchmark():
        guard = thermal_guard(probe)
        before = probe.last

        warm_lat = [timed_infer(rt, frames).latency_ms for _ in range(SETTINGS.warmup)]
        est_ms = max(0.1, percentiles(warm_lat)[0]) if warm_lat else 10.0
        remaining_s = SETTINGS.bench_budget_s - (time.perf_counter() - t_wall)
        iters = int(remaining_s * 1000.0 / est_ms)
        iters = max(SETTINGS.iters_min, min(SETTINGS.iters_max, iters))

        xs: list[float] = []
        last_stats = None
        t_meas = time.perf_counter()
        for _ in range(iters):
            last_stats = timed_infer(rt, frames)
            xs.append(last_stats.latency_ms)
        meas_s = time.perf_counter() - t_meas
        after = probe.last

    p50, p95 = percentiles(xs)
    fps = cfg.batch_size / (p50 / 1000.0) if p50 > 0 else 0.0
    extras = {
        "device_key": artifacts.device_key(),
        "backend_actual": rt.backend_actual,
        "warmup": len(warm_lat),
        "iters": iters,
        "min_ms": round(min(xs), 3),
        "max_ms": round(max(xs), 3),
        "mean_ms": round(sum(xs) / len(xs), 3),
        "load_s": round(t_load, 3),
        "measure_s": round(meas_s, 3),
        "wall_s": round(time.perf_counter() - t_wall, 3),
        "temp_before_c": before.temp_c,
        "temp_after_c": after.temp_c,
        "sm_clock_mhz_before": before.sm_clock_mhz,
        "sm_clock_mhz_after": after.sm_clock_mhz,
        "power_w_after": after.power_w,
        "gpu_util_after": after.gpu_util,
        **guard,
    }
    if last_stats is not None:
        extras["speed_breakdown_ms"] = {
            "preprocess": last_stats.pre_ms,
            "inference": last_stats.inf_ms,
            "postprocess": last_stats.post_ms,
        }
        extras["n_det_last"] = last_stats.n_det
    log.info(
        "bench %s: p50=%.2f ms p95=%.2f ms (%d iters, %s, %.1f°C)",
        cfg.label(), p50, p95, iters, rt.backend_actual, after.temp_c,
    )
    return BenchResult(p50, p95, fps, after.temp_c, extras)
