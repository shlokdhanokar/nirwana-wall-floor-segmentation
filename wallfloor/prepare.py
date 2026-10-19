"""ADE20K preparation: keep interior scenes, remap labels to 3 classes, cache
resized copies. Deliberately torch-free so worker processes stay light.

    python -m wallfloor.prepare --ade-root data/ADEChallengeData2016 --out data/cache
"""

import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

import cv2

from .labels import is_interior, remap_ade

ADE_SPLITS = {"train": "training", "val": "validation"}


def _prepare_one(name, ade_root, split_dir, out_dir, short_side):
    if os.path.exists(os.path.join(out_dir, name + ".png")):  # resumable
        return name
    lab = cv2.imread(os.path.join(ade_root, "annotations", split_dir, name + ".png"), cv2.IMREAD_GRAYSCALE)
    if lab is None or not is_interior(lab):
        return None
    img = cv2.imread(os.path.join(ade_root, "images", split_dir, name + ".jpg"), cv2.IMREAD_COLOR)
    h, w = lab.shape
    s = short_side / min(h, w)
    if s < 1:
        size = (round(w * s), round(h * s))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        lab = cv2.resize(lab, size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(out_dir, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir, name + ".png"), remap_ade(lab))
    return name


def prepare_ade20k(ade_root, out_root, short_side=448, workers=None):
    """Filter ADE20K to interior scenes, remap labels to 3 classes and cache resized copies."""
    manifest = {}
    for split, split_dir in ADE_SPLITS.items():
        out_dir = os.path.join(out_root, split)
        os.makedirs(out_dir, exist_ok=True)
        names = sorted(f[:-4] for f in os.listdir(os.path.join(ade_root, "annotations", split_dir)) if f.endswith(".png"))
        fn = partial(_prepare_one, ade_root=ade_root, split_dir=split_dir, out_dir=out_dir, short_side=short_side)
        with Pool(workers or os.cpu_count()) as pool:
            kept = [n for n in pool.imap(fn, names, chunksize=32) if n]
        manifest[split] = kept
        print(f"{split}: kept {len(kept)}/{len(names)} interior images")
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ade-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--short-side", type=int, default=448)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    prepare_ade20k(a.ade_root, a.out, a.short_side, a.workers)
