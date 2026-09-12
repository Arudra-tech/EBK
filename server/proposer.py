"""Candidate proposal.

Rule-based ladder for now. On hackathon day, swap the body of propose() for a
Nemotron/OpenClaw call that returns the same (DeployConfig, reasoning) pairs —
the controller, storage, and dashboard don't change. The LLM proposes; the
hardware decides.
"""

from .models import DeployConfig, Slo
import json

ALLOWED_SEARCH_SPACE = {
    "runtime": ["pytorch", "onnx", "tensorrt"],
    "precision": ["fp32", "fp16"],
    "resolution": [640, 512, 416],
    "batch_size": [1],
}

# Ordered by expected value: safe speedups first, aggressive tradeoffs later.
LADDER: list[tuple[DeployConfig, str]] = [
    (
        DeployConfig(
            runtime="tensorrt", precision="fp16", resolution=640, batch_size=1
        ),
        "FP16+TensorRT is the highest-value first experiment: large expected "
        "speedup with usually zero accuracy cost.",
    ),
    (
        DeployConfig(
            runtime="tensorrt", precision="fp16", resolution=512, batch_size=1
        ),
        "Lower input resolution cuts compute quadratically, but is not "
        "semantics-preserving — the accuracy gate must validate it.",
    ),
    (
        DeployConfig(
            runtime="tensorrt", precision="fp16", resolution=416, batch_size=1
        ),
        "Aggressive resolution reduction as a last resort; expected to stress "
        "the accuracy budget.",
    ),
    (
        DeployConfig(runtime="pytorch", precision="fp16", resolution=640, batch_size=1),
        "FP16 without TensorRT, in case the TensorRT path is unavailable on this device.",
    ),
]

ROUND_SIZE = 3


async def propose(
    baseline: DeployConfig,
    history: list[dict],
    slo: Slo,
    round_num: int,
    telemetry: dict | None = None,
) -> list[tuple[DeployConfig, str]]:
    context = {
        "slo": {
            "target_latency_ms": slo.target_latency_ms,
            "max_accuracy_loss_pp": slo.max_accuracy_loss_pp,
        },
        "baseline_config": baseline.model_dump(),
        "telemetry": {
            k: (
                v.isoformat()
                if hasattr(v, "isoformat")
                else str(v) if k == "_id" else v
            )
            for k, v in (telemetry or {}).items()
        },
        "previous_experiments": history,
        "round_num": round_num,
        "allowed_search_space": ALLOWED_SEARCH_SPACE,
    }
    prompt = build_agent_prompt(context)
    
    print(prompt)
    """Return up to ROUND_SIZE untested candidates for this round."""
    tested = {DeployConfig(**e["config"]).key() for e in history}
    tested.add(baseline.key())
    remaining = [(cfg, why) for cfg, why in LADDER if cfg.key() not in tested]
    return remaining[:ROUND_SIZE]


def round_rationale(baseline_latency: float, slo: Slo, round_num: int) -> str:
    ratio = baseline_latency / slo.target_latency_ms if slo.target_latency_ms else 0
    if round_num == 1:
        if ratio > 1:
            return (
                f"Current deployment is {ratio:.1f}x over the {slo.target_latency_ms:.0f} ms "
                f"target. Proposing a first round of candidates: safe precision/runtime "
                f"changes first, then tradeoffs that spend accuracy budget."
            )
        return (
            f"Current deployment already meets the {slo.target_latency_ms:.0f} ms target "
            f"({baseline_latency:.1f} ms). Testing whether any configuration is faster "
            f"within the accuracy budget."
        )
    return (
        "No round-1 candidate satisfied the SLO inside the accuracy budget. "
        "Proposing more aggressive configurations."
    )


def build_agent_prompt(context: dict) -> str:
    return f"""
You are a local AI deployment performance engineer.

Your job is to choose up to 3 deployment configurations to benchmark next.

Important rules:
- Do NOT predict or invent latency or accuracy values.
- Only choose configurations from allowed_search_space.
- Do NOT repeat configurations already present in previous_experiments.
- Prefer changes likely to reduce latency while preserving accuracy.
- The hardware benchmark, not you, decides which configuration wins.

Current context:
{json.dumps(context, indent=2, default=str)}

Return JSON only in this format:

{{
  "reasoning": "brief explanation",
  "candidates": [
    {{
      "runtime": "tensorrt",
      "precision": "fp16",
      "resolution": 640,
      "batch_size": 1,
      "reason": "why this experiment is worth testing"
    }}
  ]
}}
"""
async def call_agent_model(prompt: str) -> str:
    """
    Temporary placeholder for the local OpenClaw/Nemotron call.
    """
    raise NotImplementedError("Agent model not connected yet")