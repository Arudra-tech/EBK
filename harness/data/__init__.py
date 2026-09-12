"""Dataset helpers shared by the tools."""

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def resolve_dataset_yaml(src: Path | str = HERE / "coco_val200.yaml") -> Path:
    """ultralytics resolves a relative ``path:`` against its own datasets_dir, not
    the yaml's location. Write a sibling ``*.resolved.yaml`` with an absolute path
    so val/export work no matter the cwd. (Gitignored.)"""
    src = Path(src)
    if not src.exists():
        # a bare ultralytics dataset name like "coco8.yaml" — let ultralytics resolve/download it
        return src
    d = yaml.safe_load(src.read_text())
    p = Path(d.get("path", "."))
    if not p.is_absolute():
        p = (src.parent / p).resolve()
    d["path"] = str(p)
    out = src.with_suffix(".resolved.yaml")
    out.write_text(yaml.safe_dump(d, sort_keys=False))
    return out
