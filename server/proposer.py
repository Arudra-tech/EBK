"""Candidate proposal.

Rule-based ladder for now. On hackathon day, swap the body of propose() for a
Nemotron/OpenClaw call that returns the same (DeployConfig, reasoning) pairs —
the controller, storage, and dashboard don't change. The LLM proposes; the
hardware decides.
"""

from .models import DeployConfig, Slo

# Ordered by expected value: safe speedups first, aggressive tradeoffs later.
LADDER: list[tuple[DeployConfig, str]] = [
    (
        DeployConfig(runtime="tensorrt", precision="fp16", resolution=640, batch_size=1),
        "FP16+TensorRT is the highest-value first experiment: large expected "
        "speedup with usually zero accuracy cost.",
    ),
    (
        DeployConfig(runtime="tensorrt", precision="int8", resolution=640, batch_size=1),
        "The SLO likely needs more than FP16 alone; INT8 trades a small, "
        "measurable amount of accuracy for further latency reduction.",
    ),
    (
        DeployConfig(runtime="tensorrt", precision="fp16", resolution=512, batch_size=1),
        "Lower input resolution cuts compute quadratically, but is not "
        "semantics-preserving — the accuracy gate must validate it.",
    ),
    (
        DeployConfig(runtime="tensorrt", precision="int8", resolution=512, batch_size=1),
        "Combining INT8 with reduced resolution, in case neither alone meets the SLO.",
    ),
    (
        DeployConfig(runtime="tensorrt", precision="fp16", resolution=416, batch_size=1),
        "Aggressive resolution reduction as a last resort; expected to stress "
        "the accuracy budget.",
    ),
    (
        DeployConfig(runtime="pytorch", precision="fp16", resolution=640, batch_size=1),
        "FP16 without TensorRT, in case the TensorRT path is unavailable on this device.",
    ),
]

ROUND_SIZE = 3


def propose(
    baseline: DeployConfig, history: list[dict], slo: Slo, round_num: int
) -> list[tuple[DeployConfig, str]]:
    """Return up to ROUND_SIZE untested candidates for this round."""
    tested = {DeployConfig(**e["config"]).key() for e in history}
    tested.add(baseline.key())
    remaining = [(cfg, why) for cfg, why in LADDER if cfg.key() not in tested]
    return remaining[:ROUND_SIZE]


def round_rationale(baseline_latency: float, slo: Slo, round_num: int) -> str:
    ratio = baseline_latency / slo.target_latency_ms if slo.target_latency_ms else 0
    if round_num == 1:
        return (
            f"Current deployment is {ratio:.1f}x over the {slo.target_latency_ms:.0f} ms "
            f"target. Proposing a first round of candidates: safe precision/runtime "
            f"changes first, then tradeoffs that spend accuracy budget."
        )
    return (
        "No round-1 candidate satisfied the SLO inside the accuracy budget. "
        "Proposing more aggressive configurations."
    )
