# EBK — Local AI Deployment Optimization Agent

**The LLM proposes. The hardware decides.**

Dell x NVIDIA Hackathon project. An always-on agent that watches a local AI
workload on a Dell Pro Max (GB10), and when latency breaches the user's SLO —
or when manually activated — proposes candidate deployment configurations
(precision / runtime / resolution / batch size), benchmarks each one on the
actual hardware, rejects candidates that blow the accuracy budget, and applies
the best measured configuration.

```
User SLO ("≤25 ms, ≤0.5 pp accuracy loss")
        │
        ▼
┌───────────────┐   candidates    ┌──────────────────┐
│  Agent        │ ──────────────▶ │ Workload harness │  GET  /telemetry
│  (proposer +  │                 │ (simulator now,  │  GET/POST /config
│  controller + │ ◀────────────── │  GB10 later)     │  POST /benchmark
│  watcher)     │   measurements  └──────────────────┘  GET  /accuracy
        │
        ▼
   MongoDB (telemetry · runs · experiments · events · config_history · traces)
        │
        ▼
   Dashboard (live EKG latency chart · agent panel · experiments · change log)
```

## Run it

Prereqs: Python 3.12+, [uv](https://docs.astral.sh/uv/), Node 20+, MongoDB.

```bash
# MongoDB — pick one:
make mongo         # macOS, Homebrew (brew tap mongodb/brew; brew install mongodb-community)
make mongo-docker  # anyone with Docker, and the GB10

uv sync
cd dashboard && npm install && cd ..

make dev           # simulator :8100 + server :8000 + dashboard :5173
```

Open http://localhost:5173. The watcher is on by default: with a fresh
simulator (FP32/PyTorch/640 ≈ 58 ms) and the default 25 ms SLO, it detects the
breach within seconds and runs the agent unprompted. To demo it live:

```bash
# simulate a careless config push to production
curl -X POST localhost:8000/api/inject-regression
```

Watch: chart climbs past the SLO line → red "spike detected" toast → agent
benchmarks candidates (one gets rejected on the accuracy gate) → best valid
config applied → chart drops → green "spike resolved" toast. Download the
Markdown run report from the change-log panel.

## Layout

| Path | What |
|---|---|
| `simulator/main.py` | Mock GB10 workload. Implements the 4-endpoint harness contract with a config-keyed latency model + accuracy table. |
| `server/` | FastAPI: REST + WebSocket (`/ws`), always-on watcher, optimization controller (`agent.py`), candidate proposer (`proposer.py`), Mongo layer (`db.py`). |
| `dashboard/` | Vite + React + Tailwind + Recharts ops console. |

## The hardware contract (GB10 swap)

The **only** thing the hardware side must provide is a service with these four
endpoints (see `simulator/main.py` for exact request/response shapes):

- `GET /telemetry` — live `{latency_ms, p95_ms, throughput_fps, gpu_util, temp_c, config}`
- `GET /config` / `POST /config` — read / apply `{runtime, precision, resolution, batch_size}`
- `POST /benchmark` — warmup + measured iterations for a candidate config → `{median_latency_ms, p95_latency_ms, throughput_fps}`
- `GET /accuracy?runtime=..&precision=..&resolution=..&batch_size=..` — held-out eval → `{accuracy, delta_from_baseline}`

Then point the server at it: `WORKLOAD_URL=http://<gb10>:8100`. Server,
watcher, dashboard, and Mongo schema are unchanged.

Swapping the LLM in: replace the rule ladder in `server/proposer.py::propose()`
with a Nemotron/OpenClaw call returning the same `(config, reasoning)` pairs.

## MongoDB collections

`telemetry` (24h TTL) · `runs` (one per optimization session) · `experiments`
(every hardware measurement) · `events` (audit feed → toasts + log) ·
`config_history` (parameter fixes: before/after config + metrics) ·
`agent_traces` (proposer reasoning) · `settings` (SLO, device profile) ·
`device_profiles` (edge-hi / edge-lo).

## Env vars

| Var | Default | |
|---|---|---|
| `MONGO_URL` | `mongodb://localhost:27017` | |
| `MONGO_DB` | `ebk` | |
| `WORKLOAD_URL` | `http://localhost:8100` | point at the real GB10 harness on swap day |
