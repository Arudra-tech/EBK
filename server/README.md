# Agent Architecture

The optimization system is intentionally split into two pieces:

- `server/agent.py` — controls the optimization workflow
- `server/proposer.py` — decides which deployment configurations should be tested

The key design rule is:

> **The LLM proposes. The hardware decides.**

The language model is never trusted to invent benchmark numbers or choose the final winner. It only selects experiments. All latency and accuracy values come from the hardware harness.

---

## `server/agent.py`

`agent.py` is the **optimization controller**.

It owns the complete lifecycle of an optimization run:

```text
SLO violation
    ↓
measure baseline
    ↓
ask proposer for candidates
    ↓
benchmark candidates on hardware
    ↓
measure accuracy
    ↓
reject candidates outside quality budget
    ↓
select fastest valid configuration
    ↓
apply configuration
    ↓
store results in MongoDB