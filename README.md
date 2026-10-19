# Wall & Floor Segmentation for Interior Photographs

Nirwana AI · AI Engineer take-home. Given an interior photo, produce pixel-level **wall** and **floor** masks.

| | |
|---|---|
| **Submission notebook** | [`notebooks/Wall_Floor_Segmentation.ipynb`](notebooks/Wall_Floor_Segmentation.ipynb) · [Open in Colab](https://colab.research.google.com/github/shlokdhanokar/nirwana-wall-floor-segmentation/blob/main/notebooks/Wall_Floor_Segmentation.ipynb). Runs end-to-end on a fresh runtime (CPU is enough). Upload your own image in §3. |
| **Training notebook** | [`notebooks/Train_on_Colab.ipynb`](notebooks/Train_on_Colab.ipynb) · [Open in Colab](https://colab.research.google.com/github/shlokdhanokar/nirwana-wall-floor-segmentation/blob/main/notebooks/Train_on_Colab.ipynb). Reproduces `weights/` and `results/` on a T4 in about 1.5 h. |
| **Technical discussion** | [`docs/discussion.md`](docs/discussion.md): why this approach, hard cases, alternatives, evaluation, robustness, limitations |

## Approach in one paragraph
Wall vs. floor is a **layout** decision, not an appearance one. The same beige texture is floor at the bottom of the frame and wall behind the sofa, and a white wall, white ceiling and white window can be colour-identical. So the problem is set up as 3-class dense labelling (*other / wall / floor*) and solved with **WallFloorNet**, a 3.3 M-parameter encoder-decoder. It combines a MobileNetV3-Large encoder with my own global-context FPN decoder, **trained for this task** on ~9.2k interior scenes filtered out of ADE20K (wall → wall, floor + rug → floor). No pre-trained or fine-tuned segmentation model is used. The network runs at ~384 px. Its probabilities are upsampled and decided at up to 2048 px, then connected-component cleanup removes specks and pinholes. A training-free **classical baseline** (LSD vanishing-point horizon + GrabCut) was built first. Its failure on colour-ambiguous walls is why a learned model was needed. Guided-filter edge refinement, flip-TTA and INT8 quantisation were also built and measured, then **switched off because the numbers did not justify them** (see below).

```
image ─► resize (short side 384) ─► WallFloorNet ─► probs ─► upsample (≤2048 px) ─► argmax ─► CC cleanup ─► wall / floor masks
```

## Results
**Held-out ADE20K interior validation split** (922 images, never trained on) and the **reference image** shipped with the task (3558×2160, with its GT masks).
IoU is per class over the split. BF is the boundary F-score at a tolerance of 0.75 % of the image diagonal.
Latency is end-to-end on the reference photo (AMD Ryzen 5 4600H laptop CPU, 2 threads, median of 10 runs), including resize and post-processing. The table from the original Colab run, with refinement on, is kept in [`results/metrics_colab_refine_on.md`](results/metrics_colab_refine_on.md).

| Variant | val IoU wall | val IoU floor | val mIoU | val BF wall | val BF floor | sample IoU wall | sample IoU floor | latency (ms) |
|---|---|---|---|---|---|---|---|---|
| classical (LSD horizon + GrabCut) | 0.384 | 0.324 | 0.354 | 0.433 | 0.407 | 0.321 | 0.793 | 4976 |
| net 384 (default) | 0.751 | 0.839 | 0.795 | 0.696 | 0.764 | 0.930 | 0.946 | 317 |
| net 384 + guided-filter refine | 0.749 | 0.837 | 0.793 | 0.684 | 0.750 | 0.934 | 0.950 | 478 |
| net 384 + flip-TTA | 0.753 | 0.842 | 0.797 | 0.699 | 0.769 | 0.906 | 0.945 | 537 |
| ONNX fp32 384 (default) | 0.751 | 0.839 | 0.795 | 0.696 | 0.764 | 0.930 | 0.946 | 256 |
| ONNX int8 384 | 0.686 | 0.797 | 0.741 | 0.640 | 0.726 | 0.914 | 0.928 | 247 |
| ONNX fp32 256 | 0.729 | 0.809 | 0.769 | 0.658 | 0.727 | 0.919 | 0.878 | 173 |
| ONNX fp32 320 | 0.744 | 0.832 | 0.788 | 0.684 | 0.755 | 0.909 | 0.926 | 223 |
| ONNX fp32 448 | 0.748 | 0.835 | 0.792 | 0.696 | 0.764 | 0.913 | 0.947 | 260 |

* **Default pipeline** = "ONNX fp32 384 (default)", which gives the same masks as the PyTorch model, 60 ms faster. On the reference image it reaches **wall IoU 0.93, floor IoU 0.95**, against 0.32 / 0.79 for the classical baseline.
* **Learning was necessary:** val mIoU 0.354 (classical) → 0.795.
* **Guided-filter refinement was rejected:** an 11-setting sweep ([`results/refine_sweep.md`](results/refine_sweep.md)) found no setting that beats plain bilinear upsampling on boundary F.
* **Flip-TTA isn't worth it:** +0.002 val mIoU, but it drops the reference wall IoU from 0.930 to 0.906 and costs 1.7× latency.
* **384 px is the resolution knee:** 320 px costs 0.007 mIoU (−13 % latency), 256 px costs 0.026, and 448 px gains nothing.
* **INT8 was rejected:** −5.4 mIoU for a 4 % speed-up.

Training curve: best val mIoU 0.775 (raw network output at 384 px) at epoch 28 of 30, about 42 min on a T4 ([`weights/wallfloor_mbv3_history.json`](weights/wallfloor_mbv3_history.json)).
Per-image mIoU over 234 validation images with both classes present: median 0.824, 10th percentile 0.588, 90th percentile 0.929.
Qualitative worst / median / best validation panels are in [`results/gallery/`](results/gallery/), and full-resolution masks for the reference image are in [`results/samples/`](results/samples/).

## Performance optimisation
| Change | Effect |
|---|---|
| Network at short side 384 instead of full resolution | ~50× fewer network FLOPs on a 12 MP photo, and *more* accurate (448 px already gains nothing) |
| Argmax evaluated as three **linear decision functions** (q_w−q_f, 2q_w+q_f−1, 2q_f+q_w−1) with fused single-channel OpenCV ops instead of numpy passes over float probability maps | 12 MP post-processing ~0.9 s → ~0.1 s |
| Decisions capped at 2048 px long side, then the uint8 label map is nearest-upsampled (edge error ≤ 1.7 px on 12 MP) | ~40 % less post-processing on very large photos |
| Connected-component cleanup on a max-pooled ≤1024 px copy (conservative by construction) | ~150 ms → ~30 ms per mask |
| ONNX Runtime (graph fusion) | 317 → 256 ms end-to-end vs. eager PyTorch on CPU |
| **Overall** | ~1.3 s → **~0.25 s** end-to-end on a 3558×2160 photo, 2 CPU threads |
| INT8 static quantisation (QDQ, per-channel, percentile calibration), *rejected* | 13.3 MB → 3.8 MB, but −5.4 mIoU for only 4 % lower latency. QAT would be needed |
| Training side: thread-pool data loader, augmentation as one affine warp plus uint8 LUT colour jitter, normalisation on the GPU | ~2× data-pipeline throughput (46 → ~100 img/s) |

## Repository layout
```
wallfloor/            the package
  labels.py           3-class label space, ADE20K remap, interior-scene filter
  prepare.py          ADE20K → interior subset cache (torch-free, multiprocess)
  data.py             training dataset + augmentations, thread-pool loader
  model.py            WallFloorNet (encoder + global-context FPN decoder)
  train.py            training loop (CE + soft Dice, AMP, cosine LR)
  refine.py           linear-decision upsampling, CC cleanup, optional guided filter
  infer.py            WallFloorSegmenter: image → masks (PyTorch or ONNX backends)
  classical.py        training-free baseline (LSD horizon + GrabCut)
  metrics.py          IoU, boundary F-score
  export.py           ONNX export + INT8 static quantisation
  viz.py              overlays and panels
scripts/
  evaluate.py         every pipeline variant: quality (val + reference) and CPU latency
  make_gallery.py     worst/median/best qualitative panels + sample masks
  build_notebook.py   generates both notebooks (numbers are embedded from results/)
samples/              reference image + its ground-truth masks
weights/              trained model (.pt), ONNX FP32 and INT8 models, training history
results/              metrics, refinement sweep, gallery, sample output masks
docs/discussion.md    technical discussion
```

## Run locally
```bash
pip install -r requirements.txt
python -m wallfloor.infer --weights weights/wallfloor_mbv3.onnx --out results/mine path/to/room.jpg
```
```python
from wallfloor.infer import WallFloorSegmenter, load_rgb
seg = WallFloorSegmenter("weights/wallfloor_mbv3.onnx")   # or the .pt file
out = seg(load_rgb("room.jpg"))                          # out["wall"], out["floor"]: HxW bool
```

## Reproduce training
```bash
# ADE20K SceneParsing: http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip → data/
python -m wallfloor.prepare --ade-root data/ADEChallengeData2016 --out data/cache
python -m wallfloor.train   --cache data/cache --out weights/wallfloor_mbv3.pt --epochs 30   # ~42 min on a T4
python -m wallfloor.export  --weights weights/wallfloor_mbv3.pt
python scripts/evaluate.py && python scripts/make_gallery.py && python scripts/build_notebook.py
```

## Constraint compliance & disclosures
* **No pre-trained or fine-tuned segmentation model.** No segmentation weights of any kind are used. The segmentation network, losses and training were developed and trained here.
* **Encoder initialisation:** torchvision MobileNetV3-Large **ImageNet-1k classification** weights (an image classifier). `--no-pretrained-encoder` trains it from random initialisation instead.
* **Dataset:** ADE20K SceneParsing (MIT CSAIL; Zhou et al., CVPR 2017), used only for its wall/floor/rug labels on interior scenes.
* **Libraries:** PyTorch, torchvision, OpenCV, NumPy, ONNX, ONNX Runtime, Matplotlib, pandas.
* **No external APIs or cloud services** are used for segmentation. Everything runs locally.
* **AI tools:** Claude Code (Anthropic) was used as a coding assistant during development.
