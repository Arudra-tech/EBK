"""The 10:05 script. Runs every check the harness depends on, prints PASS/FAIL
with the failure-ladder step for each FAIL, and exits with the number of failures.

    python tools/preflight.py            # everything
    python tools/preflight.py --quick    # skip the inference timing checks

Don't debug a failing step for more than 25 minutes — take the ladder step.
"""

import argparse
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness import artifacts  # noqa: E402
from harness.config import BASELINE, PROPOSED, SETTINGS, DeployConfig  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn, ladder: str = ""):
    t = time.time()
    try:
        detail = fn()
        ok = True
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        ok = False
    dt = time.time() - t
    RESULTS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name:34s} {detail if detail else ''}  ({dt:.1f}s)")
    if not ok and ladder:
        print(f"       → {ladder}")
    return ok


def sh(cmd: list[str], timeout=10) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:200] or f"exit {r.returncode}")
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    from harness.device import is_jetson

    jetson = is_jetson()
    print(f"== preflight  device_key={artifacts.device_key()}  jetson={jetson}  model={SETTINGS.model}  backend_map={SETTINGS.backend_map!r}")

    # 1. driver / platform
    if jetson:
        check("nv_tegra_release", lambda: Path("/etc/nv_tegra_release").read_text().splitlines()[0][:80],
              "not a Jetson? then NVML path applies; if JetPack 7.x the wheel index differs (jp7)")
        check("nvpmodel -q", lambda: " ".join(sh(["nvpmodel", "-q"]).split())[:80],
              "sudo nvpmodel -m 0 && sudo jetson_clocks  (tools/lock_clocks.sh)")
        def tegrastats_check():
            exe = shutil.which("tegrastats") or ("/usr/bin/tegrastats" if Path("/usr/bin/tegrastats").exists() else None)
            if not exe:
                raise FileNotFoundError("tegrastats not on PATH")
            return exe

        check("tegrastats present", tegrastats_check, "sysfs fallback is automatic; telemetry will lack power/clock")
    else:
        check("nvidia-smi", lambda: sh(["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu", "--format=csv,noheader"]),
              "driver problem → escalate to organizers NOW; meanwhile HARNESS_BACKEND_MAP=pytorch=cpu,tensorrt=cpu")

    # 2. torch + CUDA
    def torch_check():
        import torch

        v = torch.__version__
        if "+cpu" in v:
            raise RuntimeError(f"torch {v} is a CPU-only wheel")
        if not torch.cuda.is_available():
            raise RuntimeError(f"torch {v} cuda={torch.version.cuda} but is_available()=False")
        torch.zeros(1).cuda()
        return f"torch {v} cuda {torch.version.cuda} on {torch.cuda.get_device_name(0)}"

    torch_ok = check("torch CUDA", torch_check,
                     "GB10: pip install torch --index-url https://download.pytorch.org/whl/cu130 | Jetson: --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/ | "
                     "or system python3 / --system-site-packages | or the ultralytics container. If GPU is fine but torch isn't: HARNESS_BACKEND_MAP=pytorch=onnx")

    # 3. tensorrt
    def trt_check():
        import tensorrt

        exe = shutil.which("trtexec") or ("/usr/src/tensorrt/bin/trtexec" if Path("/usr/src/tensorrt/bin/trtexec").exists() else None)
        return f"tensorrt {tensorrt.__version__}, trtexec={'yes' if exe else 'no'}"

    check("tensorrt import", trt_check,
          "GB10: pip install tensorrt (→ tensorrt-cu13) | Jetson: sudo apt install python3-libnvinfer python3-libnvinfer-dev | "
          "if hopeless: HARNESS_BACKEND_MAP=tensorrt=onnx (needs onnxruntime-gpu) or tensorrt=cpu")

    # 4. ultralytics + weights
    def ultra_check():
        import ultralytics
        from ultralytics import YOLO

        p = artifacts.pt_path()
        if not p.exists():
            raise FileNotFoundError(f"{p} — copy the .pt into harness/artifacts/ or run tools/build_engines.py --onnx-only")
        YOLO(str(p))
        return f"ultralytics {ultralytics.__version__}, {p.name}"

    check("ultralytics + weights", ultra_check, "pip install ultralytics; put yolov8n.pt in harness/artifacts/")

    # 5. probe
    def probe_check():
        from harness.device import detect

        p = detect()
        p.start()
        t0 = time.time()
        while time.time() - t0 < 3.0:
            s = p.last
            if s.ts > t0 or p.kind == "null":
                break
            time.sleep(0.1)
        p.stop()
        s = p.last
        if p.kind == "null":
            raise RuntimeError("null probe — no NVML and not Jetson")
        return f"{p.kind}: util={s.gpu_util:.2f} temp={s.temp_c:.1f}C clk={s.sm_clock_mhz} pwr={s.power_w}"

    check("device probe", probe_check, "pip install pynvml (GB10) / check tegrastats (Jetson); harness still runs with zeros")

    # 6. engines
    def engines_check():
        m = artifacts.missing_engines(PROPOSED)
        if m:
            raise FileNotFoundError(f"{len(m)} missing: " + ", ".join(p.name for p in m))
        return f"all {sum(1 for c in PROPOSED if c.runtime == 'tensorrt')} present in {artifacts.device_dir().name}"

    check("tensorrt engines", engines_check, "python tools/build_engines.py  (tmux; Orin ~30-45 min). Harness boots without them; those candidates 503.")

    # 7. accuracy table
    def acc_check():
        from harness.accuracy import AccuracyTable

        t = AccuracyTable(artifacts.accuracy_path())
        if not t.available:
            raise FileNotFoundError(str(t.path))
        m = t.missing(PROPOSED)
        if m:
            raise KeyError(f"missing entries: {m}")
        return f"{t.path.name}: {len(t.entries)} entries, baseline {t.baseline():.4f}"

    check("accuracy table", acc_check, "python tools/precompute_accuracy.py  (or ACCURACY_TABLE=<path from another device>)")

    # 8. inference timing
    if not a.quick and torch_ok:
        def timing(cfg: DeployConfig):
            def _f():
                import numpy as np

                from harness.runtimes import make_runtime
                from harness.timing import percentiles, timed_infer

                rt = make_runtime(cfg)
                rt.load()
                frames_dir = SETTINGS.data_dir / "frames"
                import cv2

                fs = sorted(frames_dir.glob("*.jpg"))
                frame = cv2.imread(str(fs[0])) if fs else np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
                rt.warm([frame], 5)
                xs = [timed_infer(rt, [frame]).latency_ms for _ in range(20)]
                p50, p95 = percentiles(xs)
                return f"{rt.backend_actual}: p50 {p50:.2f} ms, p95 {p95:.2f} ms" + ("  (synthetic frame)" if not fs else "")

            return _f

        check(f"infer {BASELINE.label()}", timing(BASELINE), "baseline must run; see torch step")
        check("infer FP16+TensorRT @640 b1", timing(DeployConfig(runtime="tensorrt", precision="fp16", resolution=640, batch_size=1)),
              "build engines first; or HARNESS_BACKEND_MAP=tensorrt=onnx")

    # 9. port + IP
    def port_check():
        s = socket.socket()
        try:
            s.bind(("0.0.0.0", SETTINGS.port))
        finally:
            s.close()
        ip = "?"
        try:
            u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            u.connect(("10.255.255.255", 1))
            ip = u.getsockname()[0]
            u.close()
        except OSError:
            pass
        return f"port {SETTINGS.port} free; hand the server team WORKLOAD_URL=http://{ip}:{SETTINGS.port}"

    check("port / LAN IP", port_check, "make stop, or HARNESS_PORT=8101")

    fails = [r for r in RESULTS if not r[1]]
    print(f"\n== {len(RESULTS) - len(fails)}/{len(RESULTS)} passed")
    sys.exit(len(fails))


if __name__ == "__main__":
    main()
