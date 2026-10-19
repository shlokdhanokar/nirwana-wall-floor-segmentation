"""Export WallFloorNet to ONNX (FP32) and an INT8 statically-quantised ONNX model.

    python -m wallfloor.export --weights weights/wallfloor_mbv3.pt --calib data/cache
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch

from .labels import MEAN, STD
from .model import load_model


def export_onnx(weights, out_path, opset=17):
    model, _ = load_model(weights, "cpu")
    x = torch.randn(1, 3, 384, 512)
    torch.onnx.export(model, x, out_path, input_names=["image"], output_names=["logits"], opset_version=opset,
                      dynamic_axes={"image": {0: "n", 2: "h", 3: "w"}, "logits": {0: "n", 2: "h", 3: "w"}},
                      dynamo=False)
    return out_path


def quantize_int8(fp32_path, out_path, calib_images, short_side=384):
    """Static INT8 (QDQ) quantisation calibrated on real interior images.
    Static rather than dynamic: dynamic quantisation only covers MatMul/Gemm,
    while this network is almost entirely Conv.
    Calibration: percentile (99.999 %) rather than min/max. MobileNetV3's
    hard-swish activations have long-tailed ranges, and min/max calibration wastes
    most of the 8-bit range on rare outliers (measured: -6 to -8 IoU points).
    Histogram calibration needs a few GB of RAM; min/max is the fallback."""
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib_images)

        def get_next(self):
            p = next(self.it, None)
            if p is None:
                return None
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            # fixed calibration shape keeps the activation ranges comparable
            img = cv2.resize(img, (512, short_side), interpolation=cv2.INTER_AREA)
            x = ((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
            return {"image": np.ascontiguousarray(x)}

    pre = fp32_path.replace(".onnx", "_pre.onnx")
    quant_pre_process(fp32_path, pre)
    common = dict(quant_format=QuantFormat.QDQ, per_channel=True,
                  activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    try:
        quantize_static(pre, out_path, Reader(), calibrate_method=CalibrationMethod.Percentile,
                        extra_options={"CalibPercentile": 99.999, "CalibMaxIntermediateOutputs": 8}, **common)
        method = "percentile"
    except MemoryError:
        quantize_static(pre, out_path, Reader(), calibrate_method=CalibrationMethod.MinMax, **common)
        method = "minmax"
    os.remove(pre)
    print("INT8 calibration:", method)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/wallfloor_mbv3.pt")
    ap.add_argument("--calib", default="data/cache", help="prepared cache dir (uses train images)")
    ap.add_argument("--n-calib", type=int, default=64)
    a = ap.parse_args()
    fp32 = export_onnx(a.weights, a.weights.replace(".pt", ".onnx"))
    print("exported", fp32, f"{os.path.getsize(fp32) / 1e6:.1f} MB")
    if os.path.exists(os.path.join(a.calib, "manifest.json")):
        names = json.load(open(os.path.join(a.calib, "manifest.json")))["train"]
        rng = np.random.default_rng(0)
        calib = [os.path.join(a.calib, "train", n + ".jpg") for n in rng.choice(names, a.n_calib, replace=False)]
        int8 = quantize_int8(fp32, a.weights.replace(".pt", "_int8.onnx"), calib)
        print("quantised", int8, f"{os.path.getsize(int8) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
