"""REST API."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pymongo import ASCENDING, DESCENDING

from . import agent, db, workload
from .models import DeployConfig, Slo

router = APIRouter(prefix="/api")


def clean(doc: dict | None) -> dict | None:
    if doc is None:
        return None
    out = {}
    for k, v in doc.items():
        if k == "_id":
            continue
        if isinstance(v, datetime):
            # PyMongo returns naive UTC datetimes; tag them so browsers don't
            # parse them as local time (which desyncs REST backfill from the
            # tz-aware WebSocket stream).
            v = v.replace(tzinfo=timezone.utc).isoformat()
        out[k] = v
    return out


@router.get("/state")
async def get_state():
    settings = await db.get_settings()
    current = await workload.get_config()
    latest_run = await db.runs.find_one({}, sort=[("started_at", DESCENDING)])
    return {
        "slo": settings["slo"],
        "device_profile": settings.get("device_profile"),
        "watcher_enabled": settings.get("watcher_enabled", True),
        "current_config": current.model_dump(),
        "active_run_id": agent.state["active_run_id"],
        "latest_run": clean(latest_run),
    }


@router.post("/slo")
async def set_slo(slo: Slo):
    await db.settings.update_one(
        {"_id": "current"}, {"$set": {"slo": slo.model_dump()}}, upsert=True
    )
    return {"ok": True, "slo": slo.model_dump()}


@router.post("/agent/activate")
async def activate_agent():
    try:
        run_id = await agent.start_run(trigger="manual")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True, "run_id": run_id}


@router.get("/telemetry")
async def get_telemetry(window: int = 120):
    cursor = db.telemetry.find({}, sort=[("ts", DESCENDING)]).limit(window * 3)
    docs = [clean(d) async for d in cursor]
    return list(reversed(docs))


@router.get("/runs")
async def list_runs(limit: int = 20):
    cursor = db.runs.find({}, sort=[("started_at", DESCENDING)]).limit(limit)
    return [clean(d) async for d in cursor]


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    doc = await db.runs.find_one({"run_id": run_id})
    if not doc:
        raise HTTPException(404, "run not found")
    return clean(doc)


@router.get("/experiments")
async def list_experiments(run_id: str | None = None, limit: int = 50):
    query = {"run_id": run_id} if run_id else {}
    cursor = db.experiments.find(query, sort=[("ts", DESCENDING)]).limit(limit)
    return [clean(d) async for d in cursor]


@router.get("/events")
async def list_events(run_id: str | None = None, limit: int = 100):
    query = {"run_id": run_id} if run_id else {}
    cursor = db.events.find(query, sort=[("ts", DESCENDING)]).limit(limit)
    return [clean(d) async for d in cursor]


@router.post("/inject-regression")
async def inject_regression():
    """Demo helper: push a bad config to the workload (simulates a careless deploy)."""
    bad = DeployConfig(runtime="pytorch", precision="fp32", resolution=640, batch_size=1)
    await workload.apply_config(bad)
    return {"ok": True, "applied": bad.model_dump()}


@router.get("/report/{run_id}", response_class=PlainTextResponse)
async def run_report(run_id: str):
    run = await db.runs.find_one({"run_id": run_id})
    if not run:
        raise HTTPException(404, "run not found")
    experiments = [
        clean(d)
        async for d in db.experiments.find({"run_id": run_id}, sort=[("ts", ASCENDING)])
    ]
    fixes = [
        clean(d)
        async for d in db.config_history.find({"run_id": run_id}, sort=[("ts", ASCENDING)])
    ]
    traces = [
        clean(d)
        async for d in db.agent_traces.find({"run_id": run_id}, sort=[("ts", ASCENDING)])
    ]
    run = clean(run)

    def cfg_label(c: dict) -> str:
        return DeployConfig(**c).label()

    slo = run["slo"]
    base = run["baseline"]
    lines = [
        f"# EBK Optimization Report — run `{run_id}`",
        "",
        f"- Trigger: **{run['trigger']}**  ·  Device: **{run.get('device_profile', 'edge-hi')}**",
        f"- Started: {run['started_at']}  ·  Finished: {run.get('finished_at') or '—'}",
        f"- Objective: latency ≤ **{slo['target_latency_ms']:.0f} ms**, "
        f"accuracy loss ≤ **{slo['max_accuracy_loss_pp']} pp**",
        "",
        "## Baseline",
        f"- {cfg_label(base['config'])}: **{base['median_latency_ms']:.1f} ms**, "
        f"accuracy **{base['accuracy'] * 100:.1f}%**",
        "",
        "## Agent reasoning",
    ]
    lines += [f"- Round {t['round']}: {t['reasoning']}" for t in traces] or ["- —"]
    lines += [
        "",
        "## Experiments",
        "| Configuration | Latency (ms) | Accuracy | Δ acc (pp) | Verdict |",
        "|---|---|---|---|---|",
    ]
    for e in experiments:
        verdict = "✅ valid" if e["valid"] else f"❌ rejected — {e['rejection_reason']}"
        lines.append(
            f"| {cfg_label(e['config'])} | {e['median_latency_ms']:.1f} | "
            f"{e['accuracy'] * 100:.1f}% | {e['accuracy_delta_pp']:+.1f} | {verdict} |"
        )
    lines += ["", "## Parameter fixes applied"]
    if fixes:
        for f in fixes:
            lines += [
                f"- {f['ts']}: {cfg_label(f['before_config'])} → "
                f"**{cfg_label(f['after_config'])}**",
                f"  - Latency {f['latency_before_ms']:.1f} ms → {f['latency_after_ms']:.1f} ms "
                f"({f['latency_before_ms'] / f['latency_after_ms']:.2f}×)",
                f"  - Accuracy {f['accuracy_before'] * 100:.1f}% → {f['accuracy_after'] * 100:.1f}%",
                f"  - Reason: {f['reason']}",
            ]
    else:
        lines.append("- None (no valid candidate improved on the baseline).")
    if run.get("winner"):
        w = run["winner"]
        lines += [
            "",
            "## Outcome",
            f"- Recommended: **{cfg_label(w['config'])}**",
            f"- {base['median_latency_ms']:.1f} ms → **{w['median_latency_ms']:.1f} ms** "
            f"(**{run['speedup']:.2f}×**)",
            f"- Accuracy {base['accuracy'] * 100:.1f}% → {w['accuracy'] * 100:.1f}% "
            f"({w['accuracy_delta_pp']:+.1f} pp)",
            f"- SLO {'**satisfied**' if run.get('slo_met') else '**not met** (fastest valid config applied)'}",
        ]
    lines += ["", "_All numbers are hardware measurements. The LLM proposes; the hardware decides._", ""]
    return "\n".join(lines)
