"""Accuracy table: precomputed offline by tools/precompute_accuracy.py on this
device, served as a lookup by /accuracy.

Why precomputed: a full held-out eval takes minutes on Orin and the server times
out every call at 30 s. Latency is always measured live; accuracy is measured
once per (model, device, artifact set) and stated openly as such.

File shape (accuracy_<device_key>_<model>.json):
{
  "model": "yolov8n.pt", "device_key": "...", "dataset": "coco_val200.yaml",
  "metric": "mAP50", "n_images": 200, "created": "...",
  "baseline_key": "pytorch|fp32|640|1",
  "entries": {"tensorrt|fp16|640|1": {"map50": 0.52, "map50_95": 0.37, "eval_s": 12.3, "backend_actual": "tensorrt"}, ...}
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import DeployConfig

log = logging.getLogger("harness.accuracy")


def entry_key(cfg: DeployConfig) -> str:
    return f"{cfg.runtime}|{cfg.precision}|{cfg.resolution}|{cfg.batch_size}"


class AccuracyTable:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict | None = None
        self.load()

    def load(self) -> bool:
        if not self.path.exists():
            log.warning("accuracy table missing: %s (run tools/precompute_accuracy.py)", self.path)
            self.data = None
            return False
        self.data = json.loads(self.path.read_text())
        log.info("accuracy table: %s (%d entries, metric=%s)", self.path, len(self.entries), self.metric)
        return True

    @property
    def available(self) -> bool:
        return self.data is not None

    @property
    def entries(self) -> dict:
        return (self.data or {}).get("entries", {})

    @property
    def metric(self) -> str:
        return (self.data or {}).get("metric", "mAP50")

    def _get(self, cfg: DeployConfig) -> dict | None:
        e = self.entries.get(entry_key(cfg))
        if e is None and cfg.batch_size != 1:
            # accuracy is batch-invariant
            e = self.entries.get(entry_key(cfg.model_copy(update={"batch_size": 1})))
        return e

    def lookup(self, cfg: DeployConfig) -> dict:
        e = self._get(cfg)
        if e is None:
            raise KeyError(entry_key(cfg))
        return e

    def baseline(self) -> float | None:
        if not self.data:
            return None
        bk = self.data.get("baseline_key")
        e = self.entries.get(bk) if bk else None
        return e["map50"] if e else None

    def missing(self, cfgs: list[DeployConfig]) -> list[str]:
        return [entry_key(c) for c in cfgs if self._get(c) is None]

    def summary(self) -> dict:
        if not self.data:
            return {"available": False, "path": str(self.path)}
        d = {k: v for k, v in self.data.items() if k != "entries"}
        d["available"] = True
        d["path"] = str(self.path)
        d["n_entries"] = len(self.entries)
        return d
