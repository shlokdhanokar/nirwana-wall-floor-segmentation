"""End-to-end inference: image -> wall mask, floor mask.

    from wallfloor.infer import WallFloorSegmenter
    seg = WallFloorSegmenter("weights/wallfloor_mbv3.pt")          # PyTorch
    seg = WallFloorSegmenter("weights/wallfloor_mbv3.onnx")        # ONNX Runtime (CPU-fast)
    out = seg(image_rgb)   # out["wall"], out["floor"]: HxW bool at the input resolution

    python -m wallfloor.infer --weights weights/wallfloor_mbv3.pt --out results path/to/img.jpg ...
"""

import argparse
import os
import time

import cv2
import numpy as np

from .labels import MEAN, STD
from .labels import FLOOR, WALL
from .refine import clean_mask, refine_wall_floor, upsample_wall_floor


def load_rgb(path):
    """Read any image (incl. .webp) as RGB uint8."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:  # fall back to PIL for formats the local OpenCV build lacks
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class WallFloorSegmenter:
    def __init__(self, weights, short_side=384, tta=True, refine=True, device=None, threads=None, out_long=2048):
        """out_long: masks are decided at <= out_long px (long side) and nearest-upsampled
        beyond that; the edge error is <= W/out_long px (~1.7 px for a 12 MP photo)."""
        self.short_side, self.tta, self.refine, self.out_long = short_side, tta, refine, out_long
        self.backend = "onnx" if weights.endswith(".onnx") else "torch"
        if self.backend == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = ort.InferenceSession(weights, so, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
        else:
            import torch
            from .model import load_model
            if threads:
                torch.set_num_threads(threads)
            self.torch = torch
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model, _ = load_model(weights, self.device)

    # ------------------------------------------------------------------ network
    def _preprocess(self, img):
        H, W = img.shape[:2]
        s = self.short_side / min(H, W)
        h, w = max(32, round(H * s / 32) * 32), max(32, round(W * s / 32) * 32)  # /32 for the encoder
        if s < 0.25:  # two-step downscale: cheap integer-ish area pass first, then the exact size
            img = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        x = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        x = ((x.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        if self.tta:  # horizontal flip is the one augmentation that preserves room layout
            x = np.concatenate([x, x[..., ::-1]], 0)
        return np.ascontiguousarray(x)

    def _forward(self, x):
        if self.backend == "onnx":
            logits = self.sess.run(None, {self.inp: x})[0]
        else:
            torch = self.torch
            with torch.inference_mode(), torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
                logits = self.model(torch.from_numpy(x).to(self.device)).float().cpu().numpy()
        e = np.exp(logits - logits.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        if self.tta:
            p = 0.5 * (p[0] + p[1][..., ::-1])
        else:
            p = p[0]
        return p.transpose(1, 2, 0)  # hxwx3

    # ------------------------------------------------------------------ public
    def __call__(self, img_rgb):
        t = {}
        t0 = time.perf_counter()
        x = self._preprocess(img_rgb)
        t["preprocess"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        probs = self._forward(x)
        t["network"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        H, W = img_rgb.shape[:2]
        so = min(1.0, self.out_long / max(H, W))
        Ho, Wo = round(H * so), round(W * so)
        if self.refine:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            if so < 1:
                gray = cv2.resize(gray, (Wo, Ho), interpolation=cv2.INTER_AREA)
            wall, floor = refine_wall_floor(gray, probs)
        else:
            wall, floor = upsample_wall_floor((Ho, Wo), probs)
        wall = clean_mask(wall)
        floor = cv2.bitwise_and(clean_mask(floor), cv2.bitwise_not(wall))
        labels = cv2.add(cv2.bitwise_and(wall, WALL), cv2.bitwise_and(floor, FLOOR))  # 0 other / 1 wall / 2 floor
        if so < 1:
            labels = cv2.resize(labels, (W, H), interpolation=cv2.INTER_NEAREST)
        wall, floor = labels == WALL, labels == FLOOR
        t["refine"] = time.perf_counter() - t0
        t["total"] = sum(t.values())
        return {"wall": wall, "floor": floor, "labels": labels, "probs": probs, "timings": t}


def main():
    from .viz import save_outputs
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights", default="weights/wallfloor_mbv3.pt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--short-side", type=int, default=384)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-refine", action="store_true")
    a = ap.parse_args()
    seg = WallFloorSegmenter(a.weights, a.short_side, tta=not a.no_tta, refine=not a.no_refine)
    os.makedirs(a.out, exist_ok=True)
    for p in a.images:
        img = load_rgb(p)
        out = seg(img)
        save_outputs(img, out, a.out, os.path.splitext(os.path.basename(p))[0])
        print(f"{p}: {', '.join(f'{k}={v * 1000:.0f}ms' for k, v in out['timings'].items())}")


if __name__ == "__main__":
    main()
