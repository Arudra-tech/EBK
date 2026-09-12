"""A candidate the harness cannot run (503) must be rejected, not abort the run.

Reproduces GB10 run b580803be834: NemoClaw proposed tensorrt/fp32/512 (legal in the
search space) but no FP32 TensorRT engine exists on the device, so /benchmark returned
503 and the whole run died with run_failed. The controller must treat that as a
rejected candidate and carry on to the winner selection.
"""

import asyncio

import httpx
import pytest

from server import agent, bus
from server.models import DeployConfig

# --- minimal in-memory stand-ins for the Mongo collections agent.py touches ---


class FakeCollection:
    def __init__(self):
        self.docs: list[dict] = []

    async def insert_one(self, doc):
        self.docs.append(dict(doc))

    async def update_one(self, flt, update, upsert=False):
        for d in self.docs:
            if all(d.get(k) == v for k, v in flt.items()):
                d.update(update.get("$set", {}))
                return
        if upsert:
            self.docs.append({**flt, **update.get("$set", {})})

    async def find_one(self, flt=None, sort=None):
        return self.docs[-1] if self.docs else None


class FakeDB:
    def __init__(self):
        for name in ("telemetry", "runs", "experiments", "events", "config_history",
                     "agent_traces", "settings"):
            setattr(self, name, FakeCollection())

    async def get_settings(self):
        return {"slo": {"target_latency_ms": 25.0, "max_accuracy_loss_pp": 0.5},
                "device_profile": "edge-hi"}


def http_503(path: str, detail: str) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", f"http://harness:8100{path}")
    resp = httpx.Response(503, json={"detail": detail}, request=req)
    return httpx.HTTPStatusError("503", request=req, response=resp)


ENGINES = {  # what "the device" can run: (runtime, precision, resolution) -> latency ms
    ("pytorch", "fp32", 640): 3.0,
    ("tensorrt", "fp16", 640): 1.5,
    ("tensorrt", "fp16", 512): 1.2,
}
ACCURACY = {("pytorch", "fp32", 640): 0.538, ("tensorrt", "fp16", 640): 0.541,
            ("tensorrt", "fp16", 512): 0.521}


class FakeWorkload:
    def __init__(self):
        self.applied: list[DeployConfig] = []

    async def get_config(self):
        return DeployConfig()

    async def benchmark(self, cfg):
        key = (cfg.runtime, cfg.precision, cfg.resolution)
        if key not in ENGINES:
            raise http_503("/benchmark", f"missing TensorRT engine yolov8n_{cfg.precision}_{cfg.resolution}_b1.engine")
        lat = ENGINES[key]
        return {"median_latency_ms": lat, "p95_latency_ms": lat * 1.1, "throughput_fps": 1000 / lat}

    async def evaluate_accuracy(self, cfg):
        key = (cfg.runtime, cfg.precision, cfg.resolution)
        if key not in ACCURACY:
            raise http_503("/accuracy", "no accuracy entry")
        return {"accuracy": ACCURACY[key]}

    async def apply_config(self, cfg):
        self.applied.append(cfg)
        return {"applied": True}


@pytest.fixture
def wired(monkeypatch):
    fdb, fwl = FakeDB(), FakeWorkload()
    monkeypatch.setattr(agent, "db", fdb)
    monkeypatch.setattr(bus, "db", fdb)
    monkeypatch.setattr(agent, "workload", fwl)
    monkeypatch.setattr(agent, "MAX_ROUNDS", 1)
    agent.state["active_run_id"] = None
    agent.state["await_recovery"] = None

    async def fake_propose(baseline, history, slo, round_num, telemetry=None):
        return "model reasoning", [
            (DeployConfig(runtime="tensorrt", precision="fp16", resolution=640), "fp16 first"),
            (DeployConfig(runtime="tensorrt", precision="fp32", resolution=512), "no engine for this one"),
            (DeployConfig(runtime="tensorrt", precision="fp16", resolution=512), "spend accuracy"),
        ]

    monkeypatch.setattr(agent.proposer, "propose", fake_propose)
    return fdb, fwl


def test_unrunnable_candidate_is_rejected_and_run_completes(wired):
    fdb, fwl = wired
    asyncio.run(agent._run_inner("run1", "manual"))

    types = [e["type"] for e in fdb.events.docs]
    assert "run_failed" not in types
    assert types.count("candidate_rejected") == 2  # 503 one + accuracy-gate one (512 @ -1.7 pp)
    assert types[-1] == "slo_met" and "config_applied" in types

    rejected_503 = next(e for e in fdb.events.docs
                        if e["type"] == "candidate_rejected" and e["payload"].get("unrunnable"))
    assert rejected_503["payload"]["config"]["precision"] == "fp32"
    assert "503" in rejected_503["payload"]["rejection_reason"]
    assert "missing TensorRT engine" in rejected_503["message"]

    # Unrunnable candidate is NOT stored as an experiment (dashboard formats latency).
    assert all(e["config"]["precision"] != "fp32" for e in fdb.experiments.docs)
    assert len(fdb.experiments.docs) == 2

    # The measured winner is applied: fp16@640 (fp16@512 fails the accuracy gate).
    assert [c.key() for c in fwl.applied] == [("tensorrt", "fp16", 640, 1)]
    assert fdb.runs.docs[-1]["status"] == "done" and fdb.runs.docs[-1]["speedup"] == 2.0


def test_baseline_harness_error_still_fails_the_run(wired, monkeypatch):
    fdb, fwl = wired

    async def broken_benchmark(cfg):
        raise http_503("/benchmark", "not ready")

    monkeypatch.setattr(fwl, "benchmark", broken_benchmark)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(agent._run_inner("run2", "manual"))
