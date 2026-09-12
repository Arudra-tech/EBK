"""GB10 workload simulator.

Stands in for the real Dell Pro Max GB10 harness. Exposes the exact HTTP
contract the hardware-side service must implement on hackathon day:

    GET  /telemetry            -> live latency/throughput/thermals for current config
    GET  /config               -> current deployment config
    POST /config               -> apply a deployment config (also the regression-injection hook)
    POST /benchmark            -> benchmark a candidate config (warmup + measure, ~2.5s)
    GET  /accuracy             -> accuracy of a config on the held-out eval set

Latency model is a lookup table keyed on (runtime, precision), scaled by
resolution and batch size, plus Gaussian noise and slow thermal drift.
Numbers are anchored to the project doc: fp32/pytorch/640 ~58ms,
fp16/tensorrt/640 ~31ms, int8/tensorrt/640 ~22ms, fp16/tensorrt/512 ~18ms.
"""

import asyncio
import math
import random
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="EBK GB10 Simulator")

RUNTIMES = {"pytorch", "tensorrt"}
PRECISIONS = {"fp32", "fp16", "int8"}
RESOLUTIONS = {640, 512, 416}
BATCH_SIZES = {1, 2, 4}


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


# Median per-batch latency (ms) at 640 / batch 1.
BASE_LATENCY_MS = {
    ("pytorch", "fp32"): 58.0,
    ("pytorch", "fp16"): 44.0,
    ("pytorch", "int8"): 50.0,  # poorly optimized path, deliberately unattractive
    ("tensorrt", "fp32"): 40.0,
    ("tensorrt", "fp16"): 31.0,
    ("tensorrt", "int8"): 22.0,
}
RESOLUTION_EXPONENT = 2.4  # calibrated so fp16/trt drops 31ms -> ~18ms at 512
BATCH_EXPONENT = 0.85      # sublinear batch scaling

BASE_ACCURACY = 0.914
INT8_PENALTY = 0.004
RESOLUTION_PENALTY = {640: 0.0, 512: 0.017, 416: 0.036}


def median_latency_ms(cfg: DeployConfig) -> float:
    base = BASE_LATENCY_MS[(cfg.runtime, cfg.precision)]
    res_scale = (cfg.resolution / 640) ** RESOLUTION_EXPONENT
    batch_scale = cfg.batch_size ** BATCH_EXPONENT
    return base * res_scale * batch_scale


def accuracy_for(cfg: DeployConfig) -> float:
    acc = BASE_ACCURACY - RESOLUTION_PENALTY[cfg.resolution]
    if cfg.precision == "int8":
        acc -= INT8_PENALTY
    return round(acc, 4)


def throughput_fps(cfg: DeployConfig, latency_ms: float) -> float:
    return round(cfg.batch_size / (latency_ms / 1000.0), 1)


state = {"config": DeployConfig(), "started": time.time()}


def thermal_temp_c() -> float:
    elapsed = time.time() - state["started"]
    return round(55.0 + 8.0 * math.sin(elapsed / 300.0) + random.gauss(0, 0.6), 1)


@app.get("/telemetry")
def telemetry():
    cfg = state["config"]
    median = median_latency_ms(cfg)
    sample = random.gauss(median, median * 0.035)
    if random.random() < 0.02:  # occasional scheduler hiccup for realism
        sample *= 1.15
    sample = max(sample, median * 0.85)
    return {
        "latency_ms": round(sample, 2),
        "p95_ms": round(median * 1.09, 2),
        "throughput_fps": throughput_fps(cfg, sample),
        "gpu_util": round(min(0.98, 0.35 + 12.0 / sample + random.gauss(0, 0.03)), 2),
        "temp_c": thermal_temp_c(),
        "config": cfg.model_dump(),
    }


@app.get("/config")
def get_config():
    return state["config"].model_dump()


@app.post("/config")
def set_config(cfg: DeployConfig):
    cfg.validate_supported()
    state["config"] = cfg
    return {"applied": True, "config": cfg.model_dump()}


@app.post("/benchmark")
async def benchmark(cfg: DeployConfig):
    cfg.validate_supported()
    # Simulated warmup + measured iterations; real harness does 5-10 warmup,
    # 20-50 measured, reports the median.
    await asyncio.sleep(2.5)
    median = median_latency_ms(cfg) * random.uniform(0.99, 1.01)
    return {
        "median_latency_ms": round(median, 2),
        "p95_latency_ms": round(median * 1.09, 2),
        "throughput_fps": throughput_fps(cfg, median),
        "temp_c": thermal_temp_c(),
    }


@app.get("/accuracy")
async def accuracy(runtime: str, precision: str, resolution: int, batch_size: int = 1):
    cfg = DeployConfig(
        runtime=runtime, precision=precision, resolution=resolution, batch_size=batch_size
    )
    cfg.validate_supported()
    await asyncio.sleep(0.8)  # held-out eval set pass
    acc = accuracy_for(cfg)
    return {"accuracy": acc, "delta_from_baseline": round(acc - BASE_ACCURACY, 4)}
