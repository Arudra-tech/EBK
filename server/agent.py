"""Optimization controller.

baseline -> propose -> benchmark -> accuracy gate -> recommend -> apply.
All performance numbers come from the workload harness; the proposer only
chooses which experiments to run.
"""

import asyncio
import uuid

from . import db, proposer, workload
from .bus import emit_event, now
from .models import DeployConfig, Slo

MAX_ROUNDS = 2

# Cross-module runtime state (watcher reads this).
state: dict = {"active_run_id": None, "await_recovery": None}


def _min_accuracy(baseline_acc: float, slo: Slo) -> float:
    return baseline_acc - slo.max_accuracy_loss_pp / 100.0


async def start_run(trigger: str) -> str:
    if state["active_run_id"]:
        raise RuntimeError("a run is already active")
    run_id = uuid.uuid4().hex[:12]
    state["active_run_id"] = run_id
    asyncio.create_task(_run(run_id, trigger))
    return run_id


async def _run(run_id: str, trigger: str) -> None:
    try:
        await _run_inner(run_id, trigger)
    except Exception as exc:  # noqa: BLE001 — a failed run must never wedge the server
        await db.runs.update_one(
            {"run_id": run_id},
            {"$set": {"status": "failed", "error": str(exc), "finished_at": now()}},
        )
        await emit_event("run_failed", f"Optimization run failed: {exc}", run_id)
    finally:
        state["active_run_id"] = None


async def _run_inner(run_id: str, trigger: str) -> None:
    settings = await db.get_settings()
    slo = Slo(**settings["slo"])

    await emit_event(
        "agent_activated",
        f"Agent activated ({'watcher auto-trigger' if trigger == 'watcher' else 'manual'}). "
        f"Objective: latency ≤ {slo.target_latency_ms:.0f} ms, "
        f"accuracy loss ≤ {slo.max_accuracy_loss_pp} pp.",
        run_id,
        {"trigger": trigger, "slo": slo.model_dump()},
    )

    # --- Baseline ---
    baseline_cfg = await workload.get_config()
    bench = await workload.benchmark(baseline_cfg)
    acc = await workload.evaluate_accuracy(baseline_cfg)
    baseline = {
        "config": baseline_cfg.model_dump(),
        "median_latency_ms": bench["median_latency_ms"],
        "accuracy": acc["accuracy"],
    }
    await db.runs.insert_one(
        {
            "run_id": run_id,
            "trigger": trigger,
            "slo": slo.model_dump(),
            "baseline": baseline,
            "status": "running",
            "winner": None,
            "speedup": None,
            "started_at": now(),
            "finished_at": None,
            "device_profile": settings.get("device_profile", "edge-hi"),
        }
    )
    await emit_event(
        "baseline_measured",
        f"Baseline {baseline_cfg.label()}: {bench['median_latency_ms']:.1f} ms, "
        f"accuracy {acc['accuracy'] * 100:.1f}%.",
        run_id,
        baseline,
    )

    min_acc = _min_accuracy(acc["accuracy"], slo)
    history: list[dict] = []

    for round_num in range(1, MAX_ROUNDS + 1):
        candidates = proposer.propose(baseline_cfg, history, slo, round_num)
        if not candidates:
            break
        rationale = proposer.round_rationale(
            baseline["median_latency_ms"], slo, round_num
        )
        await db.agent_traces.insert_one(
            {"ts": now(), "run_id": run_id, "round": round_num, "reasoning": rationale}
        )
        await emit_event("agent_reasoning", rationale, run_id, {"round": round_num})

        for cfg, why in candidates:
            await emit_event(
                "candidate_proposed", f"Proposing {cfg.label()} — {why}", run_id,
                {"config": cfg.model_dump()},
            )
            await emit_event(
                "benchmark_started", f"Benchmarking {cfg.label()} on hardware…", run_id,
                {"config": cfg.model_dump()},
            )
            bench_c = await workload.benchmark(cfg)
            acc_c = await workload.evaluate_accuracy(cfg)
            delta_pp = (acc_c["accuracy"] - acc["accuracy"]) * 100
            valid = acc_c["accuracy"] >= min_acc
            rejection = (
                None
                if valid
                else f"accuracy {acc_c['accuracy'] * 100:.1f}% is below the floor "
                f"{min_acc * 100:.1f}% (budget {slo.max_accuracy_loss_pp} pp)"
            )
            experiment = {
                "run_id": run_id,
                "config": cfg.model_dump(),
                "median_latency_ms": bench_c["median_latency_ms"],
                "p95_latency_ms": bench_c["p95_latency_ms"],
                "throughput_fps": bench_c["throughput_fps"],
                "accuracy": acc_c["accuracy"],
                "accuracy_delta_pp": round(delta_pp, 2),
                "valid": valid,
                "rejection_reason": rejection,
                "ts": now(),
            }
            await db.experiments.insert_one({**experiment})
            if valid:
                await emit_event(
                    "benchmark_done",
                    f"{cfg.label()}: {bench_c['median_latency_ms']:.1f} ms, "
                    f"accuracy {acc_c['accuracy'] * 100:.1f}% ({delta_pp:+.1f} pp) — valid.",
                    run_id,
                    {k: v for k, v in experiment.items() if k != "ts"},
                )
            else:
                await emit_event(
                    "candidate_rejected",
                    f"{cfg.label()} rejected: {rejection}. "
                    f"({bench_c['median_latency_ms']:.1f} ms would have met the target.)",
                    run_id,
                    {k: v for k, v in experiment.items() if k != "ts"},
                )
            history.append(experiment)

        slo_achievable = baseline["median_latency_ms"] <= slo.target_latency_ms or any(
            e["valid"] and e["median_latency_ms"] <= slo.target_latency_ms
            for e in history
        )
        if slo_achievable:
            break  # no need for more aggressive rounds

    # --- Selection ---
    # The baseline competes as an implicit candidate: if nothing measured
    # beats it, the right answer is "change nothing", never a downgrade.
    baseline_entry = {
        "config": baseline["config"],
        "median_latency_ms": baseline["median_latency_ms"],
        "accuracy": baseline["accuracy"],
        "accuracy_delta_pp": 0.0,
        "valid": True,
        "is_baseline": True,
    }
    valid = [e for e in history if e["valid"]] + [baseline_entry]
    meets_slo = [e for e in valid if e["median_latency_ms"] <= slo.target_latency_ms]
    pool = meets_slo or valid
    winner = min(pool, key=lambda e: e["median_latency_ms"]) if pool else None

    if winner is not None and winner.get("is_baseline"):
        await db.runs.update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "status": "done",
                    "slo_met": winner["median_latency_ms"] <= slo.target_latency_ms,
                    "finished_at": now(),
                }
            },
        )
        await emit_event(
            "already_optimal",
            f"No measured candidate beats the current deployment "
            f"({baseline['median_latency_ms']:.1f} ms). Keeping "
            f"{baseline_cfg.label()} unchanged.",
            run_id,
        )
        return

    if winner is None:
        await db.runs.update_one(
            {"run_id": run_id}, {"$set": {"status": "done", "finished_at": now()}}
        )
        await emit_event(
            "slo_not_met",
            "No candidate satisfied the accuracy budget; keeping the baseline configuration.",
            run_id,
        )
        return

    winner_cfg = DeployConfig(**winner["config"])
    speedup = baseline["median_latency_ms"] / winner["median_latency_ms"]

    if not meets_slo:
        await emit_event(
            "slo_not_met",
            f"No candidate met the {slo.target_latency_ms:.0f} ms target inside the "
            f"accuracy budget. Applying the fastest valid configuration instead: "
            f"{winner_cfg.label()} at {winner['median_latency_ms']:.1f} ms.",
            run_id,
        )

    await workload.apply_config(winner_cfg)
    await db.config_history.insert_one(
        {
            "ts": now(),
            "run_id": run_id,
            "before_config": baseline["config"],
            "after_config": winner["config"],
            "reason": f"Best measured configuration for SLO ≤{slo.target_latency_ms:.0f} ms "
            f"within accuracy budget {slo.max_accuracy_loss_pp} pp.",
            "latency_before_ms": baseline["median_latency_ms"],
            "latency_after_ms": winner["median_latency_ms"],
            "accuracy_before": baseline["accuracy"],
            "accuracy_after": winner["accuracy"],
        }
    )
    await emit_event(
        "config_applied",
        f"Applied {winner_cfg.label()}: {baseline['median_latency_ms']:.1f} ms → "
        f"{winner['median_latency_ms']:.1f} ms ({speedup:.2f}×), accuracy "
        f"{baseline['accuracy'] * 100:.1f}% → {winner['accuracy'] * 100:.1f}% "
        f"({winner['accuracy_delta_pp']:+.1f} pp).",
        run_id,
        {"config": winner["config"], "speedup": round(speedup, 2)},
    )
    await db.runs.update_one(
        {"run_id": run_id},
        {
            "$set": {
                "status": "done",
                "winner": {k: v for k, v in winner.items() if k != "ts"},
                "speedup": round(speedup, 2),
                "slo_met": bool(meets_slo),
                "finished_at": now(),
            }
        },
    )
    if meets_slo:
        await emit_event(
            "slo_met",
            f"SLO satisfied: {winner['median_latency_ms']:.1f} ms ≤ "
            f"{slo.target_latency_ms:.0f} ms with {winner['accuracy_delta_pp']:+.1f} pp accuracy.",
            run_id,
        )
    # Let the watcher confirm recovery on live telemetry and emit spike_resolved.
    state["await_recovery"] = {
        "run_id": run_id,
        "latency_before_ms": baseline["median_latency_ms"],
        "latency_after_ms": winner["median_latency_ms"],
    }
