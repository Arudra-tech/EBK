"""HTTP client for the workload harness (simulator now, real GB10 later).

The four calls below are the entire hardware contract. On hackathon day,
point WORKLOAD_URL at the real harness on the GB10 and nothing else changes.
"""

import os

import httpx

from .models import DeployConfig

WORKLOAD_URL = os.environ.get("WORKLOAD_URL", "http://localhost:8100")

_client = httpx.AsyncClient(base_url=WORKLOAD_URL, timeout=30.0)


async def get_telemetry() -> dict:
    r = await _client.get("/telemetry")
    r.raise_for_status()
    return r.json()


async def get_config() -> DeployConfig:
    r = await _client.get("/config")
    r.raise_for_status()
    return DeployConfig(**r.json())


async def apply_config(cfg: DeployConfig) -> dict:
    r = await _client.post("/config", json=cfg.model_dump())
    r.raise_for_status()
    return r.json()


async def benchmark(cfg: DeployConfig) -> dict:
    r = await _client.post("/benchmark", json=cfg.model_dump())
    r.raise_for_status()
    return r.json()


async def evaluate_accuracy(cfg: DeployConfig) -> dict:
    r = await _client.get("/accuracy", params=cfg.model_dump())
    r.raise_for_status()
    return r.json()
