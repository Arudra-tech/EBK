"""Assert a running harness satisfies the server's HTTP contract.

    python tools/contract_check.py http://<device>:8100 [--bench]

Checks shapes/types the server actually reads (server/workload.py, watcher.py,
agent.py), that /telemetry is cheap, that /benchmark leaves the live config alone
and finishes inside the 30 s httpx timeout, and that config switches show up in
telemetry within ~1 s. Exit code = number of failures.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

FAILS = 0


def ok(cond: bool, msg: str, detail: str = ""):
    global FAILS
    print(("[PASS] " if cond else "[FAIL] ") + msg + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS += 1


def req(url: str, method="GET", body=None, timeout=30.0):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method, headers={"content-type": "application/json"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}"), time.perf_counter() - t
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.perf_counter() - t


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:8100")
    ap.add_argument("--bench", action="store_true", help="also run a /benchmark (takes a few seconds)")
    ap.add_argument("--candidate", default='{"runtime":"tensorrt","precision":"fp16","resolution":640,"batch_size":1}')
    a = ap.parse_args()
    H = a.url.rstrip("/")
    cand = json.loads(a.candidate)

    st, cfg0, _ = req(f"{H}/config")
    ok(st == 200 and set(cfg0) >= {"runtime", "precision", "resolution", "batch_size"}, "GET /config shape", str(cfg0))

    st, t, dt = req(f"{H}/telemetry")
    ok(st == 200, "GET /telemetry 200")
    ok(is_num(t.get("latency_ms")), "telemetry.latency_ms is a number (REQUIRED by watcher)", str(t.get("latency_ms")))
    ok(is_num(t.get("p95_ms")), "telemetry.p95_ms number", str(t.get("p95_ms")))
    ok(is_num(t.get("throughput_fps")), "telemetry.throughput_fps number")
    ok(is_num(t.get("gpu_util")) and 0 <= t["gpu_util"] <= 1, "telemetry.gpu_util in 0..1", str(t.get("gpu_util")))
    ok(is_num(t.get("temp_c")), "telemetry.temp_c number", str(t.get("temp_c")))
    ok(t.get("config") == cfg0, "telemetry.config == GET /config")
    worst = max(req(f"{H}/telemetry")[2] for _ in range(20))
    ok(worst < 0.05, "telemetry cost < 50 ms (3 Hz poll)", f"worst {worst * 1000:.1f} ms")

    st, acc, _ = req(f"{H}/accuracy?runtime={cfg0['runtime']}&precision={cfg0['precision']}&resolution={cfg0['resolution']}&batch_size={cfg0['batch_size']}")
    ok(st == 200 and is_num(acc.get("accuracy")) and 0 <= acc["accuracy"] <= 1, "GET /accuracy baseline in 0..1", str(acc))
    st, acc, _ = req(f"{H}/accuracy?runtime={cand['runtime']}&precision={cand['precision']}&resolution={cand['resolution']}&batch_size={cand['batch_size']}")
    ok(st == 200 and is_num(acc.get("accuracy")), "GET /accuracy candidate", str(acc))
    st, _, _ = req(f"{H}/accuracy?runtime=onnx&precision=fp16&resolution=640&batch_size=1")
    ok(st == 422, "GET /accuracy rejects unknown runtime with 422", str(st))
    st, _, _ = req(f"{H}/config", "POST", {"runtime": "pytorch", "precision": "fp32", "resolution": 999, "batch_size": 1})
    ok(st == 422, "POST /config rejects bad resolution with 422", str(st))

    if a.bench:
        st, b, dt = req(f"{H}/benchmark", "POST", cand, timeout=35)
        ok(st == 200, "POST /benchmark 200", str(b)[:200] if st != 200 else "")
        for k in ("median_latency_ms", "p95_latency_ms", "throughput_fps"):
            ok(is_num(b.get(k)), f"benchmark.{k} number (REQUIRED by agent)", str(b.get(k)))
        ok(dt < 30, "benchmark under the 30 s httpx timeout", f"{dt:.1f} s")
        st, cfg1, _ = req(f"{H}/config")
        ok(cfg1 == cfg0, "live config unchanged by /benchmark", f"{cfg0} -> {cfg1}")

    # switch live config and confirm telemetry follows, then restore
    st, r, dt = req(f"{H}/config", "POST", cand)
    ok(st == 200 and r.get("applied") is True and r.get("config") == cand, "POST /config applies", str(r))
    time.sleep(1.2)
    st, t2, _ = req(f"{H}/telemetry")
    ok(t2.get("config") == cand, "telemetry.config follows switch within ~1 s", str(t2.get("config")))
    ok(is_num(t2.get("latency_ms")) and t2.get("window_n", 1) > 0, "telemetry has fresh samples after switch",
       f"latency {t2.get('latency_ms')} n={t2.get('window_n')}")
    req(f"{H}/config", "POST", cfg0)
    time.sleep(0.5)
    st, cfg2, _ = req(f"{H}/config")
    ok(cfg2 == cfg0, "restored original config")

    st, h, _ = req(f"{H}/health")
    if st == 200:
        print(f"       health: ready={h.get('ready')} engines_missing={len(h.get('engines_missing', []))} "
              f"accuracy_missing={h.get('accuracy_missing')} preload_errors={h.get('preload_errors')}")

    print(f"\n{FAILS} failure(s)")
    sys.exit(FAILS)


if __name__ == "__main__":
    main()
