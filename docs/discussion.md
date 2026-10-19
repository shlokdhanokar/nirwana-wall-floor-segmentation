## 8 · Technical discussion

### Why this approach, and what it assumes about interior images
* **Wall and floor are layout concepts, not appearance concepts.** The same beige texture is floor at the bottom of the frame and wall behind the sofa. A white wall, white ceiling and white window can be colour-identical. Any method that decides pixel-by-pixel from colour and texture is capped well below what's needed (experiment 1). The model has to reason about **where a surface sits in the room**. That is why the decoder has an explicit global-context path, and why the network runs at a moderate resolution where its receptive field covers the whole room.
* **A learned model, trained by me, under the "no pre-trained segmentation model" rule.** The useful knowledge (rooms have a floor at the bottom, walls meet it at a skirting line, rugs lie on floors, windows sit in walls) is easiest to acquire from labelled interiors. ADE20K provides wall/floor/rug labels for ~9k interiors. The segmentation network was designed and trained from scratch for this task. Only the encoder starts from generic ImageNet-classification features, and that option is disclosed and switchable.
* **Classical CV where it is strongest:** edge-aware upsampling (guided filter) and topology cleanup are cheap, deterministic and explainable, and they fix the network's main weakness: coarse boundaries.
* **Assumptions:** (1) the photo is an interior taken roughly upright (±~10° roll; the model sees ±8° rotations in training); (2) "floor" includes rugs and carpets and excludes objects standing on it; (3) "wall" includes any wall finish (paint, tile, stone, panelling) and excludes openings (windows, doors) and objects hung on it; (4) typical perspective photography. Fisheye or 360° panoramas are out of distribution.

### Hardest cases and how they were handled
| Case | Why it is hard | Handling |
|---|---|---|
| White wall vs. white ceiling vs. bright window | Same colour, no texture | Global-context block plus training data where ceilings and windows are explicit *other*; GrabCut failed exactly here |
| Floor vs. rug vs. wood-look panelling | Rugs have strong patterns; plank textures appear on walls too | Rug merged into floor in the label map (matches the provided GT); position/layout context separates floor planks from wall planks |
| Thin structures (chair and table legs, pendant cords) | Thinner than one network pixel at 384 px | Guided filter recovers edges from the full-res image; tiny holes < 0.05 % of the image are filled, larger holes (furniture) are preserved |
| Reflections on glossy floors / mirrors | Reflection of a wall *looks like* a wall | Partially handled by context; remains a failure mode (see gallery) |
| Annotation-convention ambiguity (e.g. slat panelling, built-in cabinets, wardrobes) | Graders and datasets disagree on what "wall" is | Made explicit in §1. In a product, the convention would be fixed with the design team and encoded in the training labels |
| Arbitrary resolutions / aspect ratios | 3558×2160 input would be expensive and shift the receptive field | Network runs on short side 384 (aspect preserved, /32), output upsampled with guided filter |

### Alternatives considered
1. **Purely classical** (vanishing points + GrabCut / superpixel region growing). Implemented as the baseline: no data needed, explainable, but fails on colour-ambiguous scenes and is slower. Kept as a fallback and as evidence for the decision.
2. **Room-layout estimation** (fit a cuboid room, e.g. LayoutNet/HorizonNet-style corner and edge regression), then subtract objects. It gives very clean planar walls, but it needs layout-annotated data, assumes a cuboid and still needs a separate "clutter" mask. A good second stage, though, see below.
3. **Monocular depth + surface normals → plane fitting** (floor = large horizontal plane at the bottom, walls = vertical planes). It would need a pre-trained depth network. Possibly allowed, since it isn't a segmentation model, but it is a grey area under the rules and a heavy dependency, so it was not used.
4. **Self-supervised features (DINOv2/CLIP) + a small trained head.** Probably the most accurate per hour of work, but a large backbone, slower on CPU, and closer to the spirit of "using a foundation segmenter" than I wanted.
5. **From-scratch encoder** (no ImageNet init). Supported by `--no-pretrained-encoder`. It converges more slowly and generalises somewhat worse from 9k images. ImageNet initialisation is the standard, disclosed trade-off.

### How mask quality was evaluated
* **Held-out quantitative:** per-class IoU and mIoU over the 922-image ADE20K interior *validation* split. It is never used for training, only for picking the best epoch.
* **Boundary F-score** (tolerance 0.75 % of the diagonal), because IoU is dominated by large interior regions and barely moves when edges are off by 10 px. Edge quality is what a re-texturing product actually shows the user.
* **Reference image** supplied with the task: IoU/BF against its GT masks, with the slat-panel convention mismatch made explicit.
* **Qualitative stratified review:** worst / median / best validation images by per-image mIoU, so failure modes are inspected rather than cherry-picked successes.
* **Every pipeline change is an ablation row** (refinement on/off, TTA, input resolution, FP32 vs INT8) with quality *and* latency, so trade-offs are explicit.

### What I would do next for real-world robustness
1. **Data:** collect and label a few thousand in-domain photos (real-estate listings, phone photos, renders like the reference image) under the *product's* wall/floor convention. Add hard-negative mining of mirrors, glass partitions, glossy floors, open-plan spaces and empty rooms. Label noise and convention mismatch currently cost more than model capacity.
2. **Geometry-aware second stage:** fit planes or a room layout to the predicted wall/floor regions, regularise each wall to a planar polygon, and enforce wall–floor adjacency at a single skirting line. That gives crisp, product-grade boundaries and lets one wall be separated from another (useful for per-wall re-texturing).
3. **Instance-level walls** (each wall plane separately), plus an explicit *ceiling* class.
4. **Larger capacity where latency allows**, e.g. a higher-resolution second pass on the boundary band only, or a heavier encoder on GPU servers with the current model as the on-device path.
5. **Uncertainty output:** expose per-pixel confidence (the softmax margin is already computed) so the product can ask the user to confirm low-confidence regions instead of failing silently.
6. **Calibration of the INT8 model** on in-domain photos, quantisation-aware training if accuracy drops, and a TensorRT / CoreML build for edge deployment.

### Limitations (honest summary)
* Learned priors come from ADE20K's annotation conventions, which differ from the reference GT on panelling and cladding.
* No explicit 3-D reasoning, so mirrors and reflections can be labelled as the surface they reflect.
* Very unusual viewpoints (top-down, strong roll, fisheye) and non-residential interiors with little visible floor are weaker.
* A single "wall" mask merges all wall planes. Separate wall instances would need the next step above.
