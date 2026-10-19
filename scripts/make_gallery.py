"""Qualitative results: input | wall | floor | overlay panels for the reference
image and held-out ADE20K interior validation images (best, median and worst
by per-image mIoU, so failure modes are shown, not just successes).

    python scripts/make_gallery.py --weights weights/wallfloor_mbv3.pt --n 300
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wallfloor.infer import WallFloorSegmenter, load_rgb  # noqa: E402
from wallfloor.labels import FLOOR, IGNORE, WALL  # noqa: E402
from wallfloor.metrics import binary_iou  # noqa: E402
from wallfloor.viz import panel, save_outputs  # noqa: E402


def gt_panel(img, gt, width=1600):
    return panel(img, {"wall": gt == WALL, "floor": gt == FLOOR}, width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/wallfloor_mbv3.pt")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="results/gallery")
    ap.add_argument("--samples-out", default="results/samples")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seg = WallFloorSegmenter(a.weights)

    # reference image: full-resolution masks + panel
    for p in sorted(os.listdir("samples")):
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            img = load_rgb(os.path.join("samples", p))
            out = seg(img)
            stem = os.path.splitext(p)[0]
            save_outputs(img, out, a.samples_out, stem)
            cv2.imwrite(os.path.join(a.out, f"sample_{stem}.jpg"), cv2.cvtColor(panel(img, out), cv2.COLOR_RGB2BGR))

    names = json.load(open(os.path.join(a.cache, "manifest.json")))["val"]
    names = list(np.random.default_rng(1).choice(names, min(a.n, len(names)), replace=False))
    scored = []
    for n in names:
        img = cv2.cvtColor(cv2.imread(os.path.join(a.cache, "val", n + ".jpg")), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(os.path.join(a.cache, "val", n + ".png"), cv2.IMREAD_GRAYSCALE)
        if not ((gt == WALL).mean() > 0.05 and (gt == FLOOR).mean() > 0.05):
            continue  # gallery: images where both classes are present
        out = seg(img)
        v = gt != IGNORE
        s = np.mean([binary_iou(out["wall"] & v, gt == WALL), binary_iou(out["floor"] & v, gt == FLOOR)])
        scored.append((s, n))
    scored.sort()
    k = len(scored)
    picks = {"worst": scored[:3], "median": scored[k // 2 - 1: k // 2 + 2], "best": scored[-3:]}
    for tag, items in picks.items():
        rows = []
        for s, n in items:
            img = cv2.cvtColor(cv2.imread(os.path.join(a.cache, "val", n + ".jpg")), cv2.COLOR_BGR2RGB)
            gt = cv2.imread(os.path.join(a.cache, "val", n + ".png"), cv2.IMREAD_GRAYSCALE)
            pred = panel(img, seg(img))
            ref = gt_panel(img, gt)
            cv2.putText(pred, f"pred  mIoU={s:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(ref, "ground truth", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            rows += [pred, ref, np.full((6, pred.shape[1], 3), 255, np.uint8)]
        cv2.imwrite(os.path.join(a.out, f"val_{tag}.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{k} val images with both classes; per-image mIoU median={scored[k // 2][0]:.3f}, "
          f"p10={scored[k // 10][0]:.3f}, p90={scored[9 * k // 10][0]:.3f}")


if __name__ == "__main__":
    main()
