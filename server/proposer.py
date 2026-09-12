"""Candidate proposal.

Uses a local LLM to propose deployment configurations to benchmark.
If the model is unavailable, malformed, or produces unusable candidates,
the deterministic fallback ladder is used instead.

The LLM proposes; the hardware decides.
"""

import json

import httpx

from .models import DeployConfig, Slo

import asyncio
import os


ALLOWED_SEARCH_SPACE = {
    "runtime": ["pytorch", "tensorrt"],
    "precision": ["fp32", "fp16"],
    "resolution": [640, 512, 416],
    "batch_size": [1],
}


# Deterministic fallback if the local agent is unavailable.
LADDER: list[tuple[DeployConfig, str]] = [
    (
        DeployConfig(
            runtime="tensorrt",
            precision="fp16",
            resolution=640,
            batch_size=1,
        ),
        "FP16+TensorRT is the highest-value first experiment: "
        "large expected speedup with usually zero accuracy cost.",
    ),
    (
        DeployConfig(
            runtime="tensorrt",
            precision="fp16",
            resolution=512,
            batch_size=1,
        ),
        "Lower input resolution cuts compute substantially, but is not "
        "semantics-preserving — the accuracy gate must validate it.",
    ),
    (
        DeployConfig(
            runtime="tensorrt",
            precision="fp16",
            resolution=416,
            batch_size=1,
        ),
        "Aggressive resolution reduction as a last resort; expected to "
        "stress the accuracy budget.",
    ),
    (
        DeployConfig(
            runtime="pytorch",
            precision="fp16",
            resolution=640,
            batch_size=1,
        ),
        "FP16 without TensorRT, in case the TensorRT path is unavailable "
        "on this device.",
    ),
]


ROUND_SIZE = 3


async def propose(
    baseline: DeployConfig,
    history: list[dict],
    slo: Slo,
    round_num: int,
    telemetry: dict | None = None,
) -> tuple[str, list[tuple[DeployConfig, str]]]:
    """Return agent reasoning and up to three untested candidates."""

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
                else str(v)
                if k == "_id"
                else v
            )
            for k, v in (telemetry or {}).items()
        },
        "previous_experiments": history,
        "round_num": round_num,
        "allowed_search_space": ALLOWED_SEARCH_SPACE,
    }

    prompt = build_agent_prompt(context)

    try:
        raw_response = await call_agent_model(prompt)

        reasoning, candidates = parse_agent_response(raw_response)

        # Never repeat the baseline or an experiment already run.
        tested = {
            DeployConfig(**experiment["config"]).key()
            for experiment in history
        }
        tested.add(baseline.key())

        candidates = [
            (cfg, why)
            for cfg, why in candidates
            if cfg.key() not in tested
        ]

        # Bad or useless model output should fall back safely.
        if not candidates:
            raise ValueError("Agent returned no usable candidates")

        return reasoning, candidates

    except Exception:
        # Deterministic fallback path.
        tested = {
            DeployConfig(**experiment["config"]).key()
            for experiment in history
        }
        tested.add(baseline.key())

        remaining = [
            (cfg, why)
            for cfg, why in LADDER
            if cfg.key() not in tested
        ]

        fallback_reasoning = round_rationale(
            baseline_latency=(
                telemetry.get("latency_ms", 0)
                if telemetry
                else 0
            ),
            slo=slo,
            round_num=round_num,
        )

        return fallback_reasoning, remaining[:ROUND_SIZE]


def round_rationale(
    baseline_latency: float,
    slo: Slo,
    round_num: int,
) -> str:
    """Reasoning used by the deterministic fallback."""

    ratio = (
        baseline_latency / slo.target_latency_ms
        if slo.target_latency_ms
        else 0
    )

    if round_num == 1:
        if ratio > 1:
            return (
                f"Current deployment is {ratio:.1f}x over the "
                f"{slo.target_latency_ms:.0f} ms target. "
                "Proposing a first round of candidates: safe "
                "precision/runtime changes first, then tradeoffs "
                "that spend accuracy budget."
            )

        return (
            f"Current deployment already meets the "
            f"{slo.target_latency_ms:.0f} ms target "
            f"({baseline_latency:.1f} ms). Testing whether any "
            "configuration is faster within the accuracy budget."
        )

    return (
        "No round-1 candidate satisfied the SLO inside the accuracy "
        "budget. Proposing more aggressive configurations."
    )


def build_agent_prompt(context: dict) -> str:
    """Create the prompt sent to the local deployment agent."""

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
    backend = os.getenv("AGENT_BACKEND", "ollama").lower()

    if backend == "nemoclaw":
        sandbox = os.getenv("NEMOCLAW_SANDBOX", "ebk-agent")
        session_id = os.getenv("NEMOCLAW_SESSION_ID", "ebk-optimizer")

        proc = await asyncio.create_subprocess_exec(
            "nemoclaw",
            sandbox,
            "agent",
            "--session-id",
            session_id,
            "-m",
            prompt,
            "--json",
            "--timeout",
            "60",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"NemoClaw failed: {stderr.decode().strip()}"
            )

        raw = stdout.decode().strip()

        # OpenClaw/NemoClaw JSON wrapper may contain the assistant response.
        data = json.loads(raw)

        # Keep this flexible because exact wrapper shape can vary.
        if isinstance(data, str):
            return data

        for key in ("response", "message", "content", "text"):
            if key in data and isinstance(data[key], str):
                return data[key]

        raise ValueError(
            f"Could not find agent response in NemoClaw output: {raw}"
        )

    # Development fallback: direct local Ollama.
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
            },
        )

        response.raise_for_status()

        data = response.json()
        return data["response"]


def parse_agent_response(
    raw_response: str,
) -> tuple[str, list[tuple[DeployConfig, str]]]:
    """Validate model output and convert it into DeployConfig objects."""

    data = json.loads(raw_response)

    candidates: list[tuple[DeployConfig, str]] = []

    for item in data["candidates"]:
        if item["runtime"] not in ALLOWED_SEARCH_SPACE["runtime"]:
            continue

        if item["precision"] not in ALLOWED_SEARCH_SPACE["precision"]:
            continue

        if item["resolution"] not in ALLOWED_SEARCH_SPACE["resolution"]:
            continue

        if item["batch_size"] not in ALLOWED_SEARCH_SPACE["batch_size"]:
            continue

        cfg = DeployConfig(
            runtime=item["runtime"],
            precision=item["precision"],
            resolution=item["resolution"],
            batch_size=item["batch_size"],
        )

        candidates.append(
            (
                cfg,
                item.get(
                    "reason",
                    "Proposed by local agent",
                ),
            )
        )

    reasoning = data.get(
        "reasoning",
        "No reasoning provided.",
    )

    return reasoning, candidates[:3]