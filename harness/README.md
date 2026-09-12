# Hardware harness — GB10 (`edge-hi`) / Jetson Orin Nano (`edge-lo`)

The real workload service. Same four endpoints as `simulator/main.py`, answered
with **measured** numbers from the device it runs on. The server swaps in with
`WORKLOAD_URL=http://<device>:8100`; nothing else changes.

```
uvicorn (1 worker — this process owns the GPU)
 ├─ LiveLoop thread   continuous YOLO inference on the current config @ LIVE_FPS → rolling p50/p95
 ├─ Probe thread      NVML (GB10)  |  tegrastats (Jetson)                        → gpu_util/temp/power/clock
 └─ POST /benchmark   pauses the live loop, warmup + measured iters, median/p95 (+ device state in `extras`)
```

| Endpoint | Returns |
|---|---|
| `GET /telemetry` | `latency_ms` (rolling p50), `p95_ms`, `throughput_fps`, `gpu_util` (0..1), `temp_c`, `config` + extras (`benchmarking`, `power_w`, `sm_clock_mhz`, `live_fps`, `device`, `profile`) |
| `GET/POST /config` | current / switch the live config (`{"applied": true, "config": …}`) |
| `POST /benchmark` | `median_latency_ms`, `p95_latency_ms`, `throughput_fps`, `temp_c`, `extras` (iters, temps before/after, clocks, `backend_actual`, ultralytics pre/inf/post breakdown) |
| `GET /accuracy?…` | `accuracy` (mAP50, 0..1), `delta_from_baseline` — **precomputed** by `tools/precompute_accuracy.py` on this device |
| `GET /device` | fingerprint: GPU, arch, CUDA/TRT/torch versions, power mode, loaded runtimes, accuracy table, last benchmark |
| `GET /health` | readiness, missing engines / accuracy entries, preload errors |

Latency definition (identical for every runtime): letterbox preprocess + H2D + forward + NMS + results to host,
on a frame already decoded in RAM. `perf_counter` around `predict()` + `torch.cuda.synchronize()`.

## Setup on a device

```bash
python3 -m venv ~/hv --system-site-packages && . ~/hv/bin/activate     # Jetson: system-site-packages sees python3-libnvinfer
pip install -r harness/requirements-gb10.txt      # or requirements-jetson.txt
pip install -r harness/requirements.txt
tar xzf coco_subset.tgz -C harness/data           # from the night-before USB: coco/ + frames/
cp yolov8n.pt harness/artifacts/

python tools/preflight.py                          # 10:05 — every check + the ladder step on FAIL
python tools/build_engines.py                      # tmux; Orin: 30-45 min. --precisions fp16 to skip INT8
python tools/precompute_accuracy.py                # ~3 min GB10 / ~10 min Orin
sudo bash tools/lock_clocks.sh
make harness                                       # → :8100
```

Then on the server laptop: `WORKLOAD_URL=http://<device-ip>:8100 make server`.

## Env knobs

| Var | Default | |
|---|---|---|
| `HARNESS_MODEL` | `yolov8n.pt` | pick `yolov8s.pt`/`m` on GB10 if `n` is too fast to show FP16 gains — engines + table are per-model |
| `LIVE_FPS` | 30 | live loop pacing; 0 = unpaced |
| `WINDOW` | 90 | rolling window (frames) for telemetry p50/p95 |
| `WARMUP` / `ITERS_MIN` / `ITERS_MAX` / `BENCH_BUDGET_S` | 10 / 20 / 50 / 18 | benchmark protocol |
| `PRELOAD` | `all` | `all` loads the 7 proposed configs at boot (background), `baseline`, `none` |
| `HARNESS_BACKEND_MAP` | — | failure ladder: `pytorch=onnx` (torch CUDA broken), `pytorch=cpu,tensorrt=cpu` (no GPU) |
| `ACCURACY_TABLE` | — | use a table computed elsewhere (labelled in `/device`) |
| `DEVICE_PROFILE` | auto | `edge-hi` / `edge-lo` (auto: Jetson → edge-lo) |
| `LIVE_SOURCE` | frames dir | `video:/path/clip.mp4` to loop a video instead |
| `THERMAL_MAX_C` | 80 GB10 / 85 Jetson | benchmark waits up to `THERMAL_WAIT_S` if hotter, then flags it |

## If something looks stuck

Nothing in the harness should ever block silently. Where to look:

| Symptom | What's happening | Check |
|---|---|---|
| `make harness` prints "Uvicorn running" but numbers are 0 | Boot runs in the background: CUDA init + baseline load (5–30 s), then preloading engines one by one | `curl -s localhost:8100/health` → `boot_phase` says exactly what it's doing; `boot_error` if it died |
| `build_engines.py` sits there | TensorRT engine builds take 1–10 min each (INT8 longer); TRT log lines scroll while it works | It prints `>> building …` with a timestamp before each; if you see `pip install` output, the network is the problem (see below) |
| Anything hangs on first use of ultralytics | ultralytics tries to `pip install` missing packages and retries forever on a bad network | We set `YOLO_AUTOINSTALL=false` in the Makefile and tools so it fails fast instead; make sure `pip install -r harness/requirements.txt` finished (it includes `onnx`, `onnxslim`) |
| `/benchmark` slow while preloading | Preload yields to a pending benchmark between models, but one in-flight engine deserialize (a few seconds) can't be interrupted | Wait for `boot_phase: ready`, or start with `PRELOAD=baseline` |
| `/config` or `/benchmark` return 503 "not ready" | Baseline hasn't loaded yet | `/health` → `boot_phase` |
| On Windows, `localhost` requests take ~2 s | IPv6-first resolution falling back to IPv4 | Use `127.0.0.1` |

## Laptop contract test (no GPU)

```bash
pip install -r harness/requirements-cpu.txt -r harness/requirements.txt
cp yolov8n.pt harness/artifacts/
HARNESS_BACKEND_MAP=pytorch=cpu,tensorrt=cpu ACCURACY_TABLE=harness/artifacts/accuracy_example.json make harness
make server && make dash      # in other shells; watcher fires, agent runs against real CPU numbers
```

## Verify

```bash
H=http://<device>:8100
curl -s $H/health | jq .; curl -s $H/device | jq .
curl -s $H/telemetry | jq .
curl -s -X POST $H/config -H 'content-type: application/json' -d '{"runtime":"tensorrt","precision":"fp16","resolution":640,"batch_size":1}'
time curl -s -X POST $H/benchmark -H 'content-type: application/json' -d '{"runtime":"tensorrt","precision":"int8","resolution":512,"batch_size":1}'   # < 30 s
curl -s $H/config                                                                       # unchanged by benchmark
curl -s "$H/accuracy?runtime=tensorrt&precision=fp16&resolution=512&batch_size=1"
curl -s -o /dev/null -w '%{http_code}\n' "$H/accuracy?runtime=onnx&precision=fp16&resolution=640"   # 422
```
