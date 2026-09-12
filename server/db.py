"""MongoDB access layer (PyMongo async API)."""

import os

from pymongo import AsyncMongoClient, DESCENDING

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("MONGO_DB", "ebk")

client = AsyncMongoClient(MONGO_URL)
db = client[DB_NAME]

telemetry = db.telemetry
runs = db.runs
experiments = db.experiments
events = db.events
config_history = db.config_history
agent_traces = db.agent_traces
settings = db.settings
device_profiles = db.device_profiles

DEFAULT_SETTINGS = {
    "_id": "current",
    "slo": {"target_latency_ms": 25.0, "max_accuracy_loss_pp": 0.5},
    "device_profile": "edge-hi",
    "watcher_enabled": True,
}


async def ensure_indexes() -> None:
    await telemetry.create_index("ts", expireAfterSeconds=24 * 3600)
    await experiments.create_index([("run_id", DESCENDING), ("ts", DESCENDING)])
    await events.create_index([("ts", DESCENDING)])
    await events.create_index("run_id")
    await runs.create_index([("started_at", DESCENDING)])
    await settings.update_one(
        {"_id": "current"}, {"$setOnInsert": DEFAULT_SETTINGS}, upsert=True
    )
    # reference_accuracy is a within-process ratchet (see agent._accuracy_floor):
    # reset it on every boot, since a workload-harness swap (WORKLOAD_URL) can
    # only happen between processes, never during one.
    await settings.update_one({"_id": "current"}, {"$unset": {"reference_accuracy": ""}})
    for profile in (
        {
            "_id": "edge-hi",
            "name": "edge-hi",
            "description": "GB10 at full clocks",
            "constraints": {},
        },
        {
            "_id": "edge-lo",
            "name": "edge-lo",
            "description": "GB10 clock-capped to 40%, 4 CPU cores (heterogeneity demo)",
            "constraints": {"gpu_clock_pct": 40, "cpu_cores": 4},
        },
    ):
        await device_profiles.update_one(
            {"_id": profile["_id"]}, {"$setOnInsert": profile}, upsert=True
        )


async def get_settings() -> dict:
    doc = await settings.find_one({"_id": "current"})
    return doc or DEFAULT_SETTINGS
