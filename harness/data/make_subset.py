"""Build the held-out COCO subset used for the accuracy gate (run once, on a laptop).

Inputs (downloaded if missing, ~780 MB + ~50 MB — do this the night before):
    http://images.cocodataset.org/zips/val2017.zip
    https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip

Output: harness/data/coco/
    images/val2017/*.jpg     500 images (200 eval + 300 INT8-calibration, disjoint)
    labels/val2017/*.txt     matching YOLO-format labels
    val200.txt, calib300.txt lists with ./images/val2017/... paths (relative to this dir)
plus harness/data/frames/*.jpg (32 eval images, for the live loop).

Selection is seeded so every device gets the identical subset. Tar the coco/ dir
and carry it to the devices:  tar czf coco_subset.tgz -C harness/data coco frames

Usage:
    python harness/data/make_subset.py [--n-val 200] [--n-calib 300] [--n-frames 32] [--seed 0]
                                       [--val-zip PATH] [--labels-zip PATH] [--keep-downloads]
"""

import argparse
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"


def download(url: str, dst: Path) -> Path:
    if dst.exists():
        print(f"using cached {dst}")
        return dst
    print(f"downloading {url} → {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    def hook(n, bs, total):
        if total > 0:
            done = n * bs * 100 // total
            sys.stdout.write(f"\r  {min(done, 100)}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, hook)
    print()
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--n-frames", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-zip", type=Path, default=HERE / "_downloads" / "val2017.zip")
    ap.add_argument("--labels-zip", type=Path, default=HERE / "_downloads" / "coco2017labels.zip")
    ap.add_argument("--out", type=Path, default=HERE / "coco")
    ap.add_argument("--frames-out", type=Path, default=HERE / "frames")
    ap.add_argument("--keep-downloads", action="store_true")
    a = ap.parse_args()

    download(VAL_URL, a.val_zip)
    download(LABELS_URL, a.labels_zip)

    with zipfile.ZipFile(a.labels_zip) as zl:
        label_members = {Path(m).stem: m for m in zl.namelist() if "labels/val2017/" in m and m.endswith(".txt")}
        print(f"labels in zip: {len(label_members)}")
        with zipfile.ZipFile(a.val_zip) as zv:
            img_members = {Path(m).stem: m for m in zv.namelist() if m.endswith(".jpg")}
            print(f"images in zip: {len(img_members)}")
            # only images that have a label file (ultralytics treats missing labels as background,
            # which is fine, but keeping labelled images makes mAP less noisy on a small subset)
            stems = sorted(s for s in img_members if s in label_members)
            rng = random.Random(a.seed)
            rng.shuffle(stems)
            val = sorted(stems[: a.n_val])
            calib = sorted(stems[a.n_val : a.n_val + a.n_calib])
            print(f"selected {len(val)} eval + {len(calib)} calibration images")

            img_dir = a.out / "images" / "val2017"
            lbl_dir = a.out / "labels" / "val2017"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for s in val + calib:
                with zv.open(img_members[s]) as src, open(img_dir / f"{s}.jpg", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                with zl.open(label_members[s]) as src, open(lbl_dir / f"{s}.txt", "wb") as dst:
                    shutil.copyfileobj(src, dst)

    (a.out / "val200.txt").write_text("".join(f"./images/val2017/{s}.jpg\n" for s in val))
    (a.out / "calib300.txt").write_text("".join(f"./images/val2017/{s}.jpg\n" for s in calib))

    a.frames_out.mkdir(parents=True, exist_ok=True)
    for s in val[: a.n_frames]:
        shutil.copy(a.out / "images" / "val2017" / f"{s}.jpg", a.frames_out / f"{s}.jpg")
    print(f"wrote {a.out} and {a.n_frames} live frames to {a.frames_out}")

    if not a.keep_downloads:
        print("(downloads kept in", a.val_zip.parent, "— delete manually or pass --keep-downloads to silence)")


if __name__ == "__main__":
    main()
