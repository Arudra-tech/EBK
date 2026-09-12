"""Build the TensorRT engines (and ONNX exports) this device needs, into
harness/artifacts/<device_key>/. Run once per device; engines are not portable.

    python tools/build_engines.py                        # all proposed configs, skip existing
    python tools/build_engines.py --precisions fp16      # skip INT8 (calibration is the slow part)
    python tools/build_engines.py --configs all          # every precision × resolution
    python tools/build_engines.py --onnx-only            # CPU-side ONNX exports (laptop, night before)
    python tools/build_engines.py --via trtexec          # fallback if ultralytics export breaks (FP16/FP32 only)

Orin Nano: expect ~3-6 min per engine, INT8 longer. Run in tmux the moment preflight passes.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness import artifacts  # noqa: E402
from harness.config import PROPOSED, SETTINGS, DeployConfig, all_configs  # noqa: E402
from harness.data import resolve_dataset_yaml  # noqa: E402


def wants(cfgs, precisions, resolutions, batches):
    seen = set()
    for c in cfgs:
        if c.runtime != "tensorrt":
            continue
        if precisions and c.precision not in precisions:
            continue
        if resolutions and c.resolution not in resolutions:
            continue
        if batches and c.batch_size not in batches:
            continue
        if c.key() in seen:
            continue
        seen.add(c.key())
        yield c


def quantize_kwargs(precision: str) -> dict:
    """ultralytics >= 8.4.80 uses quantize=; older versions use half=/int8=."""
    import ultralytics

    ver = tuple(int(x) for x in ultralytics.__version__.split(".")[:3])
    if ver >= (8, 4, 80):
        return {"quantize": {"fp32": 32, "fp16": 16, "int8": 8}[precision]}
    return {"half": precision == "fp16", "int8": precision == "int8"}


def export_onnx(model, res: int, force: bool) -> Path:
    dst = artifacts.onnx_path(res)
    if dst.exists() and not force:
        print(f"  onnx {res}: exists")
        return dst
    t = time.time()
    out = model.export(format="onnx", imgsz=res, opset=17, dynamic=False, simplify=True, verbose=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out), dst)
    print(f"  onnx {res}: {dst} ({time.time() - t:.0f}s)")
    return dst


def export_engine_ultralytics(model, cfg: DeployConfig, data_yaml: Path, workspace: int, force: bool) -> tuple[Path, float]:
    dst = artifacts.engine_path_for(cfg)
    if dst.exists() and not force:
        return dst, 0.0
    kw = dict(format="engine", imgsz=cfg.resolution, batch=cfg.batch_size, device=0,
              workspace=workspace, dynamic=False, simplify=True, verbose=False)
    kw.update(quantize_kwargs(cfg.precision))
    if cfg.precision == "int8":
        # calibrate on the *train* split (calib300), never on the eval images
        kw.update(data=str(data_yaml), fraction=1.0)
    t = time.time()
    out = model.export(**kw)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out), dst)
    return dst, time.time() - t


def export_engine_trtexec(cfg: DeployConfig, onnx: Path, force: bool) -> tuple[Path, float]:
    dst = artifacts.engine_path_for(cfg)
    if dst.exists() and not force:
        return dst, 0.0
    if cfg.precision == "int8":
        raise RuntimeError("trtexec path does not do INT8 calibration here — build INT8 via ultralytics or skip")
    exe = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    cmd = [exe, f"--onnx={onnx}", f"--saveEngine={dst}", "--builderOptimizationLevel=3"]
    if cfg.precision == "fp16":
        cmd.append("--fp16")
    dst.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dst, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SETTINGS.model)
    ap.add_argument("--configs", choices=["proposed", "all"], default="proposed")
    ap.add_argument("--precisions", default="", help="comma list, e.g. fp16,int8 (default: all in the config set)")
    ap.add_argument("--resolutions", default="", help="comma list, e.g. 640,512")
    ap.add_argument("--batches", default="", help="comma list, e.g. 1,2,4")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "harness" / "data" / "coco_val200.yaml"))
    ap.add_argument("--workspace", type=int, default=0, help="GB (default: 2 on Jetson, 4 otherwise)")
    ap.add_argument("--via", choices=["ultralytics", "trtexec"], default="ultralytics")
    ap.add_argument("--onnx-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    SETTINGS.model = a.model
    from ultralytics import YOLO

    pt = artifacts.pt_path()
    if not pt.exists():
        # ultralytics downloads well-known names; then we park the file in artifacts/
        print(f"{pt} missing — letting ultralytics fetch {a.model}")
        m = YOLO(a.model)
        src = Path(m.ckpt_path) if getattr(m, "ckpt_path", None) else Path(a.model)
        pt.parent.mkdir(parents=True, exist_ok=True)
        if src.exists() and src.resolve() != pt.resolve():
            shutil.copy(src, pt)
    model = YOLO(str(pt))
    print(f"model: {pt}")

    precisions = set(a.precisions.split(",")) - {""}
    resolutions = {int(x) for x in a.resolutions.split(",") if x}
    batches = {int(x) for x in a.batches.split(",") if x}
    base = PROPOSED if a.configs == "proposed" else all_configs((1, 2, 4))
    cfgs = list(wants(base, precisions, resolutions, batches))
    res_needed = sorted({c.resolution for c in cfgs} or resolutions or {640, 512, 416}, reverse=True)

    print("== ONNX exports")
    onnx = {r: export_onnx(model, r, a.force) for r in res_needed}
    if a.onnx_only:
        return

    from harness.device import is_jetson

    workspace = a.workspace or (2 if is_jetson() else 4)
    data_yaml = resolve_dataset_yaml(a.data)
    print(f"== TensorRT engines → {artifacts.device_dir()}  (workspace {workspace} GB, via {a.via})")
    rows = []
    for c in cfgs:
        try:
            if a.via == "trtexec":
                dst, secs = export_engine_trtexec(c, onnx[c.resolution], a.force)
            else:
                dst, secs = export_engine_ultralytics(model, c, data_yaml, workspace, a.force)
            status = "exists" if secs == 0.0 else f"built {secs:.0f}s"
            print(f"  {c.label():28s} {status}  {dst.name}")
            rows.append((c.label(), status))
        except Exception as e:
            print(f"  {c.label():28s} FAILED: {type(e).__name__}: {e}")
            rows.append((c.label(), f"FAILED {e}"))
    print("== summary")
    for lbl, st in rows:
        print(f"  {lbl:28s} {st}")
    missing = artifacts.missing_engines(PROPOSED)
    if missing:
        print(f"still missing {len(missing)} proposed engines:")
        for m in missing:
            print("   ", m)
        sys.exit(1)


if __name__ == "__main__":
    main()
