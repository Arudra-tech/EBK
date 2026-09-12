"""Precompute the accuracy table for this device: run the held-out eval for every
config and write harness/artifacts/accuracy_<device_key>_<model>.json.

    python tools/precompute_accuracy.py                  # proposed configs (7), ~3 min GB10 / ~10 min Orin
    python tools/precompute_accuracy.py --configs all
    # bigger eval set if 200 images look noisy: rebuild with make_subset.py --n-val 500

Each config goes through the same make_runtime() path the harness uses, so the
accuracy is measured on exactly the artifact (engine / half weights) that gets
benchmarked. rect=False is forced so PyTorch and TensorRT see identical square
letterboxed inputs — otherwise their scores differ for a non-precision reason.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness import artifacts  # noqa: E402
from harness.accuracy import entry_key  # noqa: E402
from harness.config import BASELINE, PROPOSED, SETTINGS, all_configs  # noqa: E402
from harness.data import resolve_dataset_yaml  # noqa: E402
from harness.runtimes import make_runtime, precision_kwargs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SETTINGS.model)
    ap.add_argument("--configs", choices=["proposed", "all"], default="proposed")
    ap.add_argument("--data", default=str(REPO / "harness" / "data" / "coco_val200.yaml"))
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-existing", action="store_true", help="keep entries already in the table")
    a = ap.parse_args()
    SETTINGS.model = a.model

    data_yaml = resolve_dataset_yaml(a.data)
    out = Path(a.out) if a.out else artifacts.accuracy_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    table = {}
    if a.skip_existing and out.exists():
        table = json.loads(out.read_text()).get("entries", {})

    cfgs = PROPOSED if a.configs == "proposed" else all_configs((1,))
    cfgs = [c for c in cfgs if c.batch_size == 1]  # accuracy is batch-invariant
    dev = os.environ.get("HARNESS_BACKEND_MAP", "")
    print(f"device_key={artifacts.device_key()} model={a.model} data={data_yaml} backend_map={dev!r}")
    print(f"→ {out}")

    n_images = _count_lines(data_yaml)
    for c in cfgs:
        k = entry_key(c)
        if k in table:
            print(f"  {c.label():28s} kept  map50={table[k]['map50']:.4f}")
            continue
        try:
            rt = make_runtime(c)
            rt.load()
            kw = dict(data=str(data_yaml), imgsz=c.resolution, batch=1, rect=False, conf=0.001, iou=0.7,
                      device=rt.device, plots=False, verbose=False, save_json=False)
            if rt.backend in ("pytorch", "onnx"):
                kw.update(precision_kwargs(rt.half))
            t = time.time()
            m = rt.model.val(**kw)
            secs = time.time() - t
            entry = {
                "map50": float(m.box.map50),
                "map50_95": float(m.box.map),
                "precision": float(m.box.mp),
                "recall": float(m.box.mr),
                "eval_s": round(secs, 1),
                "backend_actual": rt.backend_actual,
                "source": rt.source_path,
            }
            table[k] = entry
            print(f"  {c.label():28s} map50={entry['map50']:.4f} map50-95={entry['map50_95']:.4f} ({secs:.0f}s, {rt.backend_actual})")
            rt.close()
        except Exception as e:
            print(f"  {c.label():28s} FAILED: {type(e).__name__}: {e}")
        # write after every config so a crash keeps partial progress
        payload = {
            "model": a.model,
            "device_key": artifacts.device_key(),
            "dataset": Path(a.data).name,
            "metric": "mAP50",
            "n_images": n_images,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "baseline_key": entry_key(BASELINE),
            "backend_map": dev,
            "entries": table,
        }
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, out)

    base = table.get(entry_key(BASELINE), {}).get("map50")
    print("\n== delta vs baseline (pp)")
    for k, e in table.items():
        d = (e["map50"] - base) * 100 if base is not None else float("nan")
        flag = "  <- would be REJECTED at 0.5 pp budget" if base is not None and d < -0.5 else ""
        print(f"  {k:28s} {e['map50']:.4f}  {d:+.2f} pp{flag}")
    missing = [entry_key(c) for c in PROPOSED if entry_key(c) not in table]
    if missing:
        print("missing entries:", missing)
        sys.exit(1)


def _count_lines(data_yaml: Path) -> int | None:
    try:
        import yaml

        d = yaml.safe_load(Path(data_yaml).read_text())
        lst = Path(d["path"]) / d["val"]
        return sum(1 for line in lst.read_text().splitlines() if line.strip())
    except Exception:
        return None


if __name__ == "__main__":
    main()
