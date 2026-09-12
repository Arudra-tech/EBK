"""Always-on telemetry loop (Graft 1).

Samples the workload at ~3 Hz, persists + broadcasts every sample, and
auto-activates the agent after a sustained SLO breach. After a run applies a
fix it confirms recovery on live telemetry and emits spike_resolved.
"""

import asyncio
import logging

from . import agent, db, workload
from .bus import bus, emit_event, now

SAMPLE_INTERVAL_S = 0.33
BREACH_SAMPLES = 9    # ~3 s sustained breach before triggering
RECOVERY_SAMPLES = 6  # ~2 s below target to declare recovery

log = logging.getLogger("ebk.watcher")


async def watcher_loop() -> None:
    breach_streak = 0
    recovery_streak = 0
    spike_open = False

    while True:
        try:
            t = await workload.get_telemetry()
            settings = await db.get_settings()
            target = settings["slo"]["target_latency_ms"]

            doc = {
                "ts": now(),
                "latency_ms": t["latency_ms"],
                "p95_ms": t.get("p95_ms"),
                "throughput_fps": t.get("throughput_fps"),
                "gpu_util": t.get("gpu_util"),
                "temp_c": t.get("temp_c"),
                "config": t.get("config"),
                "device": settings.get("device_profile", "edge-hi"),
            }
            await db.telemetry.insert_one({**doc})
            await bus.broadcast(
                {
                    "type": "telemetry",
                    "sample": {**{k: v for k, v in doc.items() if k != "_id"},
                               "ts": doc["ts"].isoformat()},
                }
            )

            breaching = t["latency_ms"] > target
            breach_streak = breach_streak + 1 if breaching else 0
            recovery_streak = recovery_streak + 1 if not breaching else 0

            if (
                settings.get("watcher_enabled", True)
                and breach_streak >= BREACH_SAMPLES
                and not agent.state["active_run_id"]
                and not spike_open
            ):
                spike_open = True
                await emit_event(
                    "spike_detected",
                    f"Sustained latency spike: p50 {t['latency_ms']:.1f} ms over the "
                    f"{target:.0f} ms SLO for {BREACH_SAMPLES} consecutive samples. "
                    f"Activating agent.",
                    payload={"latency_ms": t["latency_ms"], "target_ms": target},
                )
                await agent.start_run(trigger="watcher")

            pending = agent.state["await_recovery"]
            if pending and recovery_streak >= RECOVERY_SAMPLES:
                agent.state["await_recovery"] = None
                if spike_open:
                    await emit_event(
                        "spike_resolved",
                        f"Latency spike resolved: {pending['latency_before_ms']:.1f} ms → "
                        f"{pending['latency_after_ms']:.1f} ms, back under the "
                        f"{target:.0f} ms SLO.",
                        pending["run_id"],
                        pending,
                    )
                spike_open = False
            if spike_open and not agent.state["active_run_id"] and not pending and breach_streak == 0:
                spike_open = False  # resolved without a fix (e.g. transient)

        except Exception as exc:  # noqa: BLE001 — keep the loop alive
            log.warning("watcher tick failed: %s", exc)
            await asyncio.sleep(1.0)

        await asyncio.sleep(SAMPLE_INTERVAL_S)
