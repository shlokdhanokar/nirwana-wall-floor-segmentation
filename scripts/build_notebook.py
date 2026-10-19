"""Generate notebooks/Wall_Floor_Segmentation.ipynb (the Colab submission).

Narrative + numbers live here so the notebook, README and results stay in sync:
results/metrics.md and weights/*_history.json are embedded at build time.

    python scripts/build_notebook.py --repo-url https://github.com/<user>/<repo>
"""

import argparse
import json
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(p, default=""):
    p = os.path.join(ROOT, p)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else default


def build(repo_url):
    metrics_md = read("results/metrics.md", "_(run scripts/evaluate.py)_")
    discussion = read("docs/discussion.md")
    hist = json.loads(read("weights/wallfloor_mbv3_history.json", "[]") or "[]")
    best = max(hist, key=lambda r: r["miou_wall_floor"]) if hist else None
    best_line = (f"Best checkpoint: epoch {best['epoch']}, ADE20K-interior val IoU wall **{best['iou_wall']:.3f}**, "
                 f"floor **{best['iou_floor']:.3f}**." if best else "")
    colab_badge = repo_url.replace("https://github.com/", "https://colab.research.google.com/github/").rstrip("/") \
        + "/blob/main/notebooks/Wall_Floor_Segmentation.ipynb"

    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md(f"""# Wall & Floor Segmentation for Interior Photographs
**Nirwana AI · AI Engineer take-home**  ·  [Open in Colab]({colab_badge})

Given an interior photo, produce pixel-level **wall** and **floor** masks.

**Runtime → Run all** works on a fresh Colab runtime (CPU is enough; GPU is used if present).
Jump to **§3** to upload your own image.

**TL;DR**
* **Formulation:** 3-class dense labelling *(other / wall / floor)* rather than two independent binary problems. Wall and floor are mutually exclusive, and the hard negatives (ceiling, windows, doors, furniture) need a class of their own.
* **Model:** *WallFloorNet*, a 3.3 M-parameter encoder-decoder (MobileNetV3 encoder plus my own global-context FPN decoder), **trained by me** on ~9.2k interior scenes filtered from ADE20K. No pre-trained or fine-tuned *segmentation* model is used anywhere (see §0).
* **Classical post-processing:** a fast guided filter snaps the low-resolution predictions to real image edges at full resolution, and connected-component cleanup removes specks.
* **Baseline / first experiment:** a training-free geometric pipeline (LSD vanishing-point horizon plus GrabCut), kept for comparison. It handles floors reasonably and walls poorly. That result is the reason a learned model was needed.
* **Speed:** ONNX Runtime plus INT8 static quantisation gives a fast CPU path for a full-resolution 3558×2160 photo (numbers in §6).

{best_line}
"""),
        md("""## 0 · Constraint compliance & disclosures
| Constraint | How it is met |
|---|---|
| No pre-trained / fine-tuned **segmentation** model | No segmentation weights of any kind are loaded. The full segmentation network (decoder, head, loss, training loop) is defined in `wallfloor/model.py` and `wallfloor/train.py` and was trained from scratch for this task. |
| Segmentation logic developed in this solution | The label space and ADE20K remapping, the interior-scene filter, the architecture, augmentations, losses, guided-filter refinement, cleanup and the classical baseline are all written for this project. |
| No segmentation API / cloud service | Everything runs locally in this notebook (PyTorch or ONNX Runtime on CPU/GPU). |
| Disclose external models / data / libraries / AI tools | **Encoder initialisation:** torchvision's MobileNetV3-Large **ImageNet-1k classification** weights (an image classifier, not a segmenter). `--no-pretrained-encoder` trains it from random init instead. **Dataset:** ADE20K SceneParsing (MIT CSAIL), used for its *wall*, *floor* and *rug* labels only. **Libraries:** PyTorch, torchvision, OpenCV, NumPy, ONNX / ONNX Runtime, Matplotlib. **AI tools:** Claude Code (Anthropic) was used as a coding assistant. |
"""),
        code("""# --- setup: get the code + trained weights (works on a fresh Colab runtime and locally) ---
import os, sys, subprocess, importlib
REPO_URL = "%s"
if os.path.isdir("wallfloor"):
    ROOT = os.getcwd()
elif os.path.isdir("../wallfloor"):
    ROOT = os.path.abspath("..")
else:
    if not os.path.isdir("repo"):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "repo"], check=True)
    ROOT = os.path.abspath("repo")
os.chdir(ROOT); sys.path.insert(0, ROOT)
if importlib.util.find_spec("onnxruntime") is None:   # the only dependency Colab lacks
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], check=True)

import cv2, numpy as np, torch, matplotlib.pyplot as plt, json, time
from wallfloor.infer import WallFloorSegmenter, load_rgb
from wallfloor.classical import segment_classical
from wallfloor.metrics import binary_iou, boundary_f1
from wallfloor.viz import show, overlay, save_outputs
print("root:", ROOT, "| torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
print(sorted(os.listdir("weights")))""" % repo_url),
        md("""## 1 · Problem formulation
* **Input:** one RGB interior photo of any size and aspect ratio. **Output:** two binary masks at the *input* resolution.
* **Label definitions** (the main ambiguity in the task):
  * *Wall*: vertical room surfaces, including paint, wallpaper, tile and stone cladding. Windows, doors, curtains, pictures and furniture in front of the wall are **not** wall.
  * *Floor*: walkable ground surface. A **rug or carpet counts as floor**, matching the reference GT supplied with the task. Furniture legs and objects on the floor are **not** floor.
  * Everything else (ceiling, windows, furniture, people, plants, ...) is *other*.
* **Why 3 classes and not 2 binary problems:** the masks must not overlap, and the softmax forces the network to explain the confusers (ceiling vs. wall, glossy floor vs. reflection, window vs. bright wall) explicitly.
* **Why learning:** wall and floor are defined by **3-D layout and semantics**, not appearance. A white wall, white ceiling and white floor can share the same colour, and a striped rug can look like wooden planks. Appearance-only rules break on that (see §5, experiment 1), so the system has to learn layout priors from data.
"""),
        md("""## 2 · Pipeline
```
image (any size) ─► resize (short side 384, /32) ─► WallFloorNet ─► softmax probs (≈384 px)
                                                      ▲  flip-TTA (optional)
                                                      │
 full-res guide (gray) ─────────────────────────────► fast guided filter ─► argmax ─► CC cleanup ─► wall, floor masks
```
**WallFloorNet** (`wallfloor/model.py`)
* *Encoder*: MobileNetV3-Large feature extractor, taps at strides 4/8/16/32. Chosen for its accuracy-per-FLOP and because it quantises well to INT8 for the CPU path.
* *Context block* (stride 32): wall vs. floor is mostly a **global layout** decision. A pixel of beige texture is floor at the bottom of the room and wall behind the sofa. The block gates a local projection with a global-average-pooled descriptor (LR-ASPP style) and concatenates an explicit global context vector.
* *Top-down FPN*: fuses stride 16 → 8 → 4 with lateral 1×1 convs and depthwise-separable 3×3 smoothing, which keeps boundaries (skirting lines, wall corners) sharp at low cost.
* *Loss*: cross-entropy (`ignore_index` for unannotated pixels, mild class weights) **plus soft Dice** on wall and floor, so large easy regions don't dominate.
* *Training data*: ADE20K filtered to **interior scenes** with a label-statistics heuristic (`wallfloor/labels.py::is_interior`: no sky, some wall/floor, ceiling or floor present). That leaves 9.2k train and 922 val images. Labels are remapped `wall→1, floor|rug→2, other→0, unlabelled→ignore`.
* *Augmentation*: scale 0.6–1.6, crops, ±8° rotation (handheld photos), **horizontal flip only** (a vertical flip would break the floor-is-down prior the model should learn), strong colour, exposure, blur and noise jitter (walls come in every colour, and interiors range from tungsten-dim to blown-out daylight).
"""),
        md("## 3 · Run on an image\nThe reference image shipped with the assignment, scored against its ground-truth masks:"),
        code("""seg = WallFloorSegmenter("weights/wallfloor_mbv3.pt", short_side=384, tta=True, refine=True)

img = load_rgb("samples/Contemporary_Stone_Accent_Lounge.webp")
out = seg(img)
show(img, out, title=f"{img.shape[1]}x{img.shape[0]}  ·  " +
     "  ".join(f"{k} {v*1000:.0f} ms" for k, v in out["timings"].items()))

gt = {k: cv2.imread(f"samples/gt/Contemporary_Stone_Accent_Lounge_{k}.png", 0) > 127 for k in ("wall", "floor")}
for k in ("wall", "floor"):
    print(f"{k:5s}  IoU = {binary_iou(out[k], gt[k]):.3f}   boundary-F = {boundary_f1(out[k], gt[k]):.3f}")
save_outputs(img, out, "results/samples", "Contemporary_Stone_Accent_Lounge")   # full-res masks as PNG"""),
        md("""**Reading the score on this image:** the reference GT labels only the **stone accent wall** as wall. The **dark vertical slat panelling** at both sides is left out of both masks. Under the ADE20K convention the model learned, wall cladding is wall, so the model also marks the slat panels. Most of the remaining wall "error" is that **annotation-convention difference**, not a localisation error. The floor (wood planks + rug, excluding furniture legs) is where the definitions agree.

### Try your own image
On Colab, run the next cell and pick a file (or set `IMAGE_URL`). Any interior photo works, at any resolution."""),
        code("""IMAGE_URL = ""   # optionally paste a direct image URL here
paths = []
try:
    from google.colab import files            # Colab upload widget
    up = files.upload()
    paths = list(up.keys())
except Exception:
    pass
if IMAGE_URL:
    import urllib.request
    urllib.request.urlretrieve(IMAGE_URL, "url_image.jpg"); paths.append("url_image.jpg")
if not paths:
    print("No upload - using the bundled sample image.")
    paths = ["samples/Contemporary_Stone_Accent_Lounge.webp"]
for p in paths:
    im = load_rgb(p)
    o = seg(im)
    show(im, o, title=os.path.basename(p))
    save_outputs(im, o, "results/uploads", os.path.splitext(os.path.basename(p))[0])
print("masks saved to results/uploads/")"""),
        md("## 4 · What each stage contributes\nNetwork probabilities at ~384 px vs. the full-resolution result after guided-filter refinement (zoomed crops):"),
        code("""raw = WallFloorSegmenter("weights/wallfloor_mbv3.pt", tta=False, refine=False)(img)
ref = WallFloorSegmenter("weights/wallfloor_mbv3.pt", tta=False, refine=True)(img)
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
ax[0].imshow(out["probs"][..., 1], cmap="magma"); ax[0].set_title("P(wall) - network output")
ax[1].imshow(out["probs"][..., 2], cmap="magma"); ax[1].set_title("P(floor) - network output")
ax[2].imshow(out["probs"].argmax(2), cmap="viridis"); ax[2].set_title("argmax (0 other / 1 wall / 2 floor)")
[a.axis("off") for a in ax]; plt.show()

H, W = img.shape[:2]
y0, y1, x0, x1 = int(.50*H), int(.80*H), int(.30*W), int(.62*W)    # sofa / coffee-table / rug region
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
for a, (o, t) in zip(ax, [(raw, "bilinear upsampling only"), (ref, "guided-filter refinement")]):
    a.imshow(overlay(img, o["wall"], o["floor"])[y0:y1, x0:x1]); a.set_title(t); a.axis("off")
plt.show()
for name, o in [("raw", raw), ("refined", ref)]:
    print(f"{name:8s} floor IoU {binary_iou(o['floor'], gt['floor']):.3f}  BF {boundary_f1(o['floor'], gt['floor']):.3f} | "
          f"wall IoU {binary_iou(o['wall'], gt['wall']):.3f}  BF {boundary_f1(o['wall'], gt['wall']):.3f}")"""),
        md("""## 5 · Experiments & iteration
**Experiment 1: training-free geometric baseline** (`wallfloor/classical.py`). This came first, to see how far image-formation priors alone go:
1. LSD line segments → pairwise intersections of oblique segments → weighted-median **horizon** (Manhattan-world vanishing geometry).
2. **GrabCut** seeded with *floor below the horizon* / *definite floor in a bottom-centre band*, then a second GrabCut for *wall between ceiling band and horizon*.

It gets the floor of the reference image largely right but leaks wall into the bright window and curtains. That is expected: GrabCut's colour GMMs cannot tell a white wall from a white window or ceiling. Walls are defined by geometry and semantics, not by colour. It is also slow (~seconds per image) because of the iterative graph cuts. This failure is what motivated a learned model."""),
        code("""t0 = time.perf_counter(); cl = segment_classical(img); t_cl = time.perf_counter() - t0
show(img, cl, title=f"Classical baseline (horizon at {cl['horizon']:.2f}·H) - {t_cl:.1f}s")
for k in ("wall", "floor"):
    print(f"classical {k:5s} IoU {binary_iou(cl[k], gt[k]):.3f}   vs  WallFloorNet {binary_iou(out[k], gt[k]):.3f}")"""),
        md(f"""**Experiments 2–6: learned model, refinement, TTA, resolution, quantisation.** Quantitative comparison on the
held-out **ADE20K interior validation split** (922 images, never trained on) and on the reference image.
IoU is per-class over the whole split. BF is the boundary F-score at a tolerance of 0.75 % of the image diagonal, so it measures edge quality, which IoU barely reflects.
Latency is end-to-end on the 3558×2160 reference photo (CPU, 4 threads, median of 10 runs), *including* resize, refinement and cleanup.

{metrics_md}
"""),
        code("""import pandas as pd
hist = pd.DataFrame(json.load(open("weights/wallfloor_mbv3_history.json")))
fig, ax = plt.subplots(1, 2, figsize=(14, 3.5))
ax[0].plot(hist.epoch, hist.loss); ax[0].set_title("train loss (CE + 0.5·Dice)"); ax[0].set_xlabel("epoch")
for c in ("iou_wall", "iou_floor", "miou_wall_floor"):
    ax[1].plot(hist.epoch, hist[c], label=c)
ax[1].legend(); ax[1].set_title("ADE20K-interior val IoU"); ax[1].set_xlabel("epoch"); plt.show()
pd.DataFrame(json.load(open("results/metrics.json"))).round(3)"""),
        md("Held-out validation images, **worst / median / best** by per-image mIoU (prediction row, then ground-truth row). The failure cases matter as much as the successes:"),
        code("""for tag in ("worst", "median", "best"):
    p = f"results/gallery/val_{tag}.jpg"
    if os.path.exists(p):
        plt.figure(figsize=(18, 14)); plt.imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
        plt.title(tag); plt.axis("off"); plt.show()"""),
        md("""## 6 · Performance optimisation (bonus)
Levers, in order of impact:
1. **Architecture budget:** MobileNetV3 plus a 0.34 M-parameter decoder: 3.3 M params in total, 13 MB FP32.
2. **Decouple network resolution from output resolution:** the network sees ~384 px. Full resolution comes from the guided filter, where only the per-pixel linear coefficients are upsampled, so cost is O(pixels) with a tiny constant. Running the network at full 3558×2160 would be roughly 50× more compute and *less* accurate, because the receptive field would cover a smaller share of the room.
3. **ONNX Runtime** graph optimisations (operator fusion) for the CPU path.
4. **INT8 static quantisation** (QDQ, per-channel weights, calibrated on 64 interior images): 4× smaller model, faster convolutions on CPU, with the IoU change measured in the table above.
5. Flip-TTA is optional: about +2× network time for a small accuracy gain. It is a switch for quality-vs-latency, off in the fast path.

Live benchmark on *this* runtime:"""),
        code("""def bench(s, im, n=5):
    s(im); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); s(im); ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))

threads = os.cpu_count()
cfgs = [("PyTorch FP32 CPU, TTA", WallFloorSegmenter("weights/wallfloor_mbv3.pt", tta=True, device="cpu", threads=threads)),
        ("PyTorch FP32 CPU",      WallFloorSegmenter("weights/wallfloor_mbv3.pt", tta=False, device="cpu", threads=threads)),
        ("ONNX FP32 CPU",         WallFloorSegmenter("weights/wallfloor_mbv3.onnx", tta=False, threads=threads)),
        ("ONNX INT8 CPU",         WallFloorSegmenter("weights/wallfloor_mbv3_int8.onnx", tta=False, threads=threads))]
if torch.cuda.is_available():
    cfgs.append(("PyTorch GPU (AMP), TTA", WallFloorSegmenter("weights/wallfloor_mbv3.pt", tta=True, device="cuda")))
rows = []
for name, s in cfgs:
    o = s(img)
    rows.append({"config": name, "latency_ms (3558x2160)": round(bench(s, img)),
                 "IoU wall": round(binary_iou(o["wall"], gt["wall"]), 3), "IoU floor": round(binary_iou(o["floor"], gt["floor"]), 3)})
for f in ("wallfloor_mbv3.pt", "wallfloor_mbv3.onnx", "wallfloor_mbv3_int8.onnx"):
    print(f"{f:28s} {os.path.getsize('weights/' + f)/1e6:5.1f} MB")
pd.DataFrame(rows)"""),
        md("""## 7 · Reproduce training (optional, ~1 h on a Colab T4)
Set `RUN_TRAINING = True` to download ADE20K (~1 GB), rebuild the interior subset and retrain from scratch. The trained weights in `weights/` were produced by exactly this code."""),
        code("""RUN_TRAINING = False
if RUN_TRAINING:
    if not os.path.isdir("data/ADEChallengeData2016"):
        os.makedirs("data", exist_ok=True)
        subprocess.run(["wget", "-q", "-O", "data/ade.zip",
                        "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"], check=True)
        subprocess.run(["unzip", "-q", "data/ade.zip", "-d", "data"], check=True)
    subprocess.run([sys.executable, "-m", "wallfloor.prepare", "--ade-root", "data/ADEChallengeData2016",
                    "--out", "data/cache", "--workers", "2"], check=True)
    subprocess.run([sys.executable, "-m", "wallfloor.train", "--cache", "data/cache", "--out", "weights/retrained.pt",
                    "--epochs", "40", "--batch", "16", "--workers", "2"], check=True)
    subprocess.run([sys.executable, "-m", "wallfloor.export", "--weights", "weights/retrained.pt"], check=True)"""),
        md(discussion or "## 8 · Technical discussion\n_(see docs/discussion.md)_"),
    ]
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                      "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}}
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    out = os.path.join(ROOT, "notebooks", "Wall_Floor_Segmentation.ipynb")
    nbf.write(nb, out)
    print("wrote", out)


def build_training(repo_url):
    """Colab notebook that reproduces weights/, results/metrics.* and results/gallery/ on a GPU runtime."""
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("""# Train WallFloorNet on Colab (reproduces `weights/` and `results/`)
**Runtime → Change runtime type → T4 GPU**, then **Run all**. Takes roughly 1.5 h end to end:
dataset download and prep ~10 min, training ~60–75 min, evaluation ~10 min.
At the end, `trained_artifacts.zip` (weights, ONNX/INT8 models, metrics, gallery) downloads automatically.
Unzip it into the repository root and commit."""),
        code("""import os, sys, subprocess, torch
assert torch.cuda.is_available(), "Switch the runtime to a GPU (Runtime > Change runtime type > T4 GPU)"
print(torch.cuda.get_device_name(0), "| CPUs:", os.cpu_count())
REPO_URL = "%s"
if not os.path.isdir("repo"):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "repo"], check=True)
os.chdir("repo")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime", "onnx"], check=True)""" % repo_url),
        code("""%%time
# ADE20K SceneParsing (MIT CSAIL): ~1 GB
if not os.path.isdir("data/ADEChallengeData2016"):
    os.makedirs("data", exist_ok=True)
    !wget -q -O data/ade.zip http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
    !unzip -q data/ade.zip -d data && rm data/ade.zip
!python -m wallfloor.prepare --ade-root data/ADEChallengeData2016 --out data/cache --workers {os.cpu_count()}"""),
        code("""EPOCHS = 30
# encoder from ImageNet classification weights (disclosed); add --no-pretrained-encoder for the from-scratch ablation
!python -m wallfloor.train --cache data/cache --out weights/wallfloor_mbv3.pt --epochs {EPOCHS} --batch 16 --workers {os.cpu_count()} --loader process"""),
        code("""!python -m wallfloor.export --weights weights/wallfloor_mbv3.pt --calib data/cache --n-calib 64
!ls -la weights"""),
        code("""%%time
# quality on the GPU; latency on this runtime's CPU (threads = all vCPUs)
!python scripts/evaluate.py --weights weights/wallfloor_mbv3.pt --threads {os.cpu_count()}
!python scripts/make_gallery.py --weights weights/wallfloor_mbv3.pt --n 300"""),
        code("""!zip -q -r /content/trained_artifacts.zip weights results
try:
    from google.colab import files
    files.download("/content/trained_artifacts.zip")
except Exception as e:
    print("Download /content/trained_artifacts.zip from the file browser:", e)"""),
    ]
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                      "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}}
    out = os.path.join(ROOT, "notebooks", "Train_on_Colab.ipynb")
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="https://github.com/shlokdhanokar/nirwana-wall-floor-segmentation")
    a = ap.parse_args()
    build(a.repo_url)
    build_training(a.repo_url)
