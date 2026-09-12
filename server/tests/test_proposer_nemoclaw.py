"""Pin the NemoClaw/OpenClaw stdout shape that call_agent_model() must unwrap.

The fixture below is the real output captured on the Dell GB10 (NemoClaw v0.0.123,
sandbox `my-assistant`, OpenClaw + local vLLM Qwen3.6 via inference.local), trimmed
to the keys the parser touches.

Run:  python -m pytest server/tests -q
"""

import asyncio
import json

import pytest

from server import proposer
from server.models import DeployConfig

GB10_STDOUT = """\
{
  "runId": "1c04851d-c9b0-4bf9-9bdc-c3bfa61fa891",
  "status": "ok",
  "summary": "completed",
  "result": {
    "payloads": [
      {
        "text": "```json\\n{\\n  \\"reasoning\\": \\"hi\\",\\n  \\"candidates\\": []\\n}\\n```",
        "mediaUrl": null
      }
    ],
    "meta": {
      "durationMs": 1427,
      "agentMeta": {"sessionId": "ebk-optimizer", "provider": "inference",
                    "model": "nvidia/Qwen3.6-35B-A3B-NVFP4"},
      "finalPromptText": "Reply with JSON only: {\\"reasoning\\":\\"hi\\",\\"candidates\\":[]}",
      "finalAssistantVisibleText": "```json\\n{\\n  \\"reasoning\\": \\"hi\\",\\n  \\"candidates\\": []\\n}\\n```",
      "finalAssistantRawText": "```json\\n{\\n  \\"reasoning\\": \\"hi\\",\\n  \\"candidates\\": []\\n}\\n```",
      "stopReason": "stop"
    }
  }
}
"""

THREE_CANDIDATES = {
    "reasoning": "FP16 first, then spend resolution budget.",
    "candidates": [
        {"runtime": "tensorrt", "precision": "fp16", "resolution": 640, "batch_size": 1,
         "reason": "free speedup"},
        {"runtime": "tensorrt", "precision": "fp16", "resolution": 512, "batch_size": 1,
         "reason": "trade accuracy"},
        {"runtime": "pytorch", "precision": "fp16", "resolution": 640, "batch_size": 1,
         "reason": "no-TRT path"},
        # outside the allowed search space -> must be dropped by parse_agent_response
        {"runtime": "tensorrt", "precision": "int8", "resolution": 640, "batch_size": 1,
         "reason": "not allowed"},
    ],
}


def nemoclaw_wrap(text: str, preamble: str = "") -> str:
    body = json.dumps({"runId": "x", "status": "ok",
                       "result": {"payloads": [{"text": text, "mediaUrl": None}], "meta": {}}})
    return preamble + body


class FakeProc:
    def __init__(self, stdout: str, stderr: str = "", rc: int = 0):
        self._out, self._err, self.returncode = stdout.encode(), stderr.encode(), rc

    async def communicate(self):
        return self._out, self._err

    def kill(self):
        pass


@pytest.fixture
def nemoclaw_env(monkeypatch):
    monkeypatch.setenv("AGENT_BACKEND", "nemoclaw")
    monkeypatch.setenv("NEMOCLAW_SANDBOX", "my-assistant")

    calls: list[list[str]] = []

    def install(stdout: str, stderr: str = "", rc: int = 0):
        async def fake_exec(*cmd, **_kw):
            calls.append(list(cmd))
            return FakeProc(stdout, stderr, rc)

        monkeypatch.setattr(proposer.asyncio, "create_subprocess_exec", fake_exec)
        return calls

    return install


def test_real_gb10_output_unwraps_to_inner_json():
    text = proposer.extract_agent_text(GB10_STDOUT)
    assert json.loads(text) == {"reasoning": "hi", "candidates": []}
    assert proposer.parse_agent_response(text) == ("hi", [])


def test_preamble_line_before_json_is_skipped():
    raw = "✓ Active gateway set to 'nemoclaw'\n" + GB10_STDOUT
    assert json.loads(proposer.extract_agent_text(raw))["reasoning"] == "hi"


def test_bare_openclaw_final_shape():
    raw = json.dumps({"ok": True, "final": "```json\n{\"reasoning\": \"r\", \"candidates\": []}\n```",
                      "payloads": [{"text": "ignored if final present"}]})
    assert json.loads(proposer.extract_agent_text(raw)) == {"reasoning": "r", "candidates": []}


def test_legacy_top_level_keys_and_plain_text():
    assert proposer.extract_agent_text(json.dumps({"response": "{\"a\": 1}"})) == '{"a": 1}'
    assert proposer.extract_agent_text('{"reasoning": "direct", "candidates": []}') != ""
    assert proposer.extract_agent_text("```json\n{\"x\": 2}\n```") == '{"x": 2}'


def test_prose_around_json_is_tolerated():
    chatty = "Sure! Here is my plan:\n```json\n{\"reasoning\": \"p\", \"candidates\": []}\n```\nLet me know."
    assert json.loads(proposer.extract_agent_text(nemoclaw_wrap(chatty))) == {"reasoning": "p", "candidates": []}
    bare = "Plan: {\"reasoning\": \"q\", \"candidates\": []} done"
    assert json.loads(proposer.extract_agent_text(nemoclaw_wrap(bare)))["reasoning"] == "q"


def test_unknown_wrapper_raises():
    with pytest.raises(ValueError):
        proposer.extract_agent_text(json.dumps({"runId": "x", "status": "ok"}))


def test_call_agent_model_runs_nemoclaw_and_returns_candidates(nemoclaw_env):
    fenced = "```json\n" + json.dumps(THREE_CANDIDATES, indent=2) + "\n```"
    calls = nemoclaw_env(nemoclaw_wrap(fenced, preamble="✓ Active gateway set to 'nemoclaw'\n"))

    text = asyncio.run(proposer.call_agent_model("PROMPT"))
    reasoning, candidates = proposer.parse_agent_response(text)

    assert reasoning == THREE_CANDIDATES["reasoning"]
    assert [c.key() for c, _ in candidates] == [
        ("tensorrt", "fp16", 640, 1),
        ("tensorrt", "fp16", 512, 1),
        ("pytorch", "fp16", 640, 1),
    ]
    assert candidates[0][0] == DeployConfig(runtime="tensorrt", precision="fp16", resolution=640, batch_size=1)

    cmd = calls[0]
    assert cmd[:3] == ["nemoclaw", "my-assistant", "agent"]
    assert "--json" in cmd and "-m" in cmd and cmd[cmd.index("-m") + 1] == "PROMPT"
    assert cmd[cmd.index("--session-id") + 1] == "ebk-optimizer"


def test_call_agent_model_nonzero_exit_raises(nemoclaw_env):
    nemoclaw_env("", stderr="Sandbox 'ebk-agent' does not exist.", rc=1)
    with pytest.raises(RuntimeError, match="does not exist"):
        asyncio.run(proposer.call_agent_model("PROMPT"))


def test_propose_falls_back_and_logs_when_agent_fails(nemoclaw_env, caplog):
    nemoclaw_env("", stderr="boom", rc=1)
    from server.models import Slo

    with caplog.at_level("WARNING", logger="ebk.proposer"):
        reasoning, candidates = asyncio.run(
            proposer.propose(DeployConfig(), [], Slo(), 1, telemetry={"latency_ms": 58.0})
        )
    assert "fallback ladder" in caplog.text
    assert candidates and candidates[0][0].key() == ("tensorrt", "fp16", 640, 1)
    assert reasoning.startswith("Current deployment is")
