"""Quality + speed evaluation of every pipeline variant.

Quality : IoU (wall, floor) and boundary-F on the held-out ADE20K interior
          validation split (never used for training or model selection beyond
          checkpoint choice) and on the provided reference image with its GT.
Speed   : end-to-end latency on the 3558x2160 reference image, always on CPU
          (--threads threads), median of N runs, including resize/refine/cleanup.

    python scripts/evaluate.py --weights weights/wallfloor_mbv3.pt --n-val 500
"""

import argparse
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wallfloor.classical import segment_classical  # noqa: E402
from wallfloor.infer import WallFloorSegmenter, load_rgb  # noqa: E402
from wallfloor.labels import FLOOR, IGNORE, WALL  # noqa: E402
from wallfloor.metrics import ConfusionMatrix, binary_iou, boundary_f1  # noqa: E402

SAMPLE = "samples/Contemporary_Stone_Accent_Lounge.webp"
SAMPLE_GT = {"wall": "samples/gt/Contemporary_Stone_Accent_Lounge_wall.png",
             "floor": "samples/gt/Contemporary_Stone_Accent_Lounge_floor.png"}


def to_labels(out, shape):
    lab = np.zeros(shape, np.uint8)
    lab[out["wall"]] = WALL
    lab[out["floor"]] = FLOOR
    return lab


def eval_val(fn, cache, names):
    cm, bf = ConfusionMatrix(), {"wall": [], "floor": []}
    for n in names:
        img = cv2.cvtColor(cv2.imread(os.path.join(cache, "val", n + ".jpg")), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(os.path.join(cache, "val", n + ".png"), cv2.IMREAD_GRAYSCALE)
        out = fn(img)
        cm.update(to_labels(out, gt.shape).ravel(), gt.ravel())
        valid = gt != IGNORE
        for k, c in (("wall", WALL), ("floor", FLOOR)):
            if (gt == c).any():
                bf[k].append(boundary_f1(out[k] & valid, (gt == c)))
    iou = cm.iou()
    return {"val_iou_wall": float(iou[WALL]), "val_iou_floor": float(iou[FLOOR]),
            "val_miou": float((iou[WALL] + iou[FLOOR]) / 2),
            "val_bf_wall": float(np.mean(bf["wall"])), "val_bf_floor": float(np.mean(bf["floor"]))}


def eval_sample(fn, img, gts):
    out = fn(img)
    r = {}
    for k in ("wall", "floor"):
        r[f"sample_iou_{k}"] = binary_iou(out[k], gts[k])
        r[f"sample_bf_{k}"] = boundary_f1(out[k], gts[k])
    return r, out


def latency(fn, img, runs):
    fn(img)  # warm-up
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(img)
        ts.append(time.perf_counter() - t0)
    return 1000 * statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/wallfloor_mbv3.pt")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--n-val", type=int, default=0, help="0 = whole interior val split")
    ap.add_argument("--n-val-classical", type=int, default=150)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for the latency benchmark")
    ap.add_argument("--device", default=None, help="device for the quality pass of PyTorch variants (default: cuda if available)")
    ap.add_argument("--out", default="results/metrics.json")
    ap.add_argument("--only", nargs="*", default=None,
                    help="re-run only variants whose name contains one of these substrings; "
                         "their rows replace the matching rows of an existing --out file")
    a = ap.parse_args()

    names = json.load(open(os.path.join(a.cache, "manifest.json")))["val"]
    if a.n_val:
        names = list(np.random.default_rng(0).choice(names, a.n_val, replace=False))
    img = load_rgb(SAMPLE)
    gts = {k: cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127 for k, p in SAMPLE_GT.items()}

    onnx32 = a.weights.replace(".pt", ".onnx")
    onnx8 = a.weights.replace(".pt", "_int8.onnx")
    pt = a.weights
    # (name, weights, short_side, tta, refine); quality runs on --device, latency always on CPU
    variants = [
        ("classical (LSD horizon + GrabCut)", None, None, None, None),
        ("net 384, bilinear upsample", pt, 384, False, False),
        ("net 384 + guided refine", pt, 384, False, True),
        ("net 384 + refine + flip-TTA", pt, 384, True, True),
        ("ONNX fp32 384 + refine + TTA", onnx32, 384, True, True),
        ("ONNX fp32 384 + refine", onnx32, 384, False, True),
        ("ONNX int8 384 + refine", onnx8, 384, False, True),
        ("ONNX fp32 256 + refine", onnx32, 256, False, True),
        ("ONNX fp32 320 + refine", onnx32, 320, False, True),
        ("ONNX fp32 448 + refine", onnx32, 448, False, True),
    ]
    canonical = [v[0] for v in variants]  # table order
    results = []
    if a.only is not None:
        variants = [v for v in variants if any(o in v[0] for o in a.only)]
        if os.path.exists(a.out):
            results = json.load(open(a.out))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for name, w, short, tta, refine in variants:
        if w is not None and not os.path.exists(w):
            print("skip (missing model):", name)
            continue
        if w is None:
            fn = fn_cpu = segment_classical
            val_names = names[:a.n_val_classical]
        else:
            fn = WallFloorSegmenter(w, short, tta=tta, refine=refine, device=a.device)
            fn_cpu = WallFloorSegmenter(w, short, tta=tta, refine=refine, device="cpu", threads=a.threads)
            val_names = names
        r = {"variant": name}
        r.update(eval_val(fn, a.cache, val_names))
        r["val_images"] = len(val_names)
        s, _ = eval_sample(fn, img, gts)
        r.update(s)
        r["latency_ms_3558x2160_cpu"] = latency(fn_cpu, img, 3 if w is None else a.runs)
        results = [x for x in results if x["variant"] != name] + [r]
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        json.dump(results, open(a.out, "w"), indent=1)

    results.sort(key=lambda r: canonical.index(r["variant"]) if r["variant"] in canonical else len(canonical))
    json.dump(results, open(a.out, "w"), indent=1)

    # markdown table for the README
    hdr = "| Variant | val IoU wall | val IoU floor | val mIoU | val BF wall | val BF floor | sample IoU wall | sample IoU floor | CPU latency (ms) |"
    lines = [hdr, "|" + "---|" * 9]
    for r in results:
        lines.append(f"| {r['variant']} | {r['val_iou_wall']:.3f} | {r['val_iou_floor']:.3f} | {r['val_miou']:.3f} | "
                     f"{r['val_bf_wall']:.3f} | {r['val_bf_floor']:.3f} | {r['sample_iou_wall']:.3f} | "
                     f"{r['sample_iou_floor']:.3f} | {r['latency_ms_3558x2160_cpu']:.0f} |")
    open(a.out.replace(".json", ".md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
