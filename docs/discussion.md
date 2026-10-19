## 8 · Technical discussion

### Why this approach, and what it assumes about interior images
* **Wall and floor are layout concepts, not appearance concepts.** The same beige texture is floor at the bottom of the frame and wall behind the sofa, and a white wall, white ceiling and white window can be colour-identical. A method that decides pixel-by-pixel from colour and texture is capped well below what's needed. The classical baseline reaches 0.38 wall IoU on held-out interiors, against 0.75 for the learned model. The model has to reason about **where a surface sits in the room**. That is why the decoder has an explicit global-context path, and why the network runs at a moderate resolution (384 px) where its receptive field covers the whole room. Going to 448 px did not help.
* **A learned model, trained by me, under the "no pre-trained segmentation model" rule.** The useful knowledge (rooms have a floor at the bottom, walls meet it at a skirting line, rugs lie on floors, windows sit in walls) is easiest to acquire from labelled interiors. ADE20K provides wall/floor/rug labels for ~9k interiors. The segmentation network was designed and trained from scratch for this task. Only the encoder starts from generic ImageNet-classification features, and that is disclosed and switchable.
* **Classical CV where it measurably helps:** connected-component cleanup (cheap, deterministic, removes specks and pinholes) and the classical baseline as a yardstick. Guided-filter edge snapping was built and **rejected on evidence** (below).
* **Assumptions:**
  1. The photo is an interior taken roughly upright (±~10° roll; the model sees ±8° rotations in training).
  2. "Floor" includes rugs and carpets and excludes objects standing on it.
  3. "Wall" means wall surfaces and excludes openings (windows, doors) and objects hung on it.
  4. Typical perspective photography. Fisheye or 360° panoramas are out of distribution.

### Hardest cases and how they were handled
| Case | Why it is hard | Handling / status |
|---|---|---|
| White wall vs. white ceiling vs. bright window | Same colour, no texture | Global-context block plus training data where ceilings and windows are explicit *other*. GrabCut failed exactly here (wall leaked into the window: 0.32 IoU on the reference image vs. 0.93 learned) |
| Floor vs. rug vs. wood-look panelling | Rugs have strong patterns; plank textures appear on walls too | Rug merged into floor in the label map, matching the provided GT. Layout context separates floor planks from wall planks |
| Accent wall vs. side panelling | Is dark slat panelling "wall"? The reference GT says no | The model agrees with the GT on the reference image (wall IoU 0.93). In general this is a *convention* question to fix with the product team and encode in the labels |
| Thin structures (chair and table legs) | A few pixels wide even at full resolution, sub-pixel at 384 px | Cleanup fills only holes < 0.05 % of the image, so furniture holes are preserved. Guided filtering did *not* improve these boundaries (sweep below) |
| Out-of-scope "interiors" | The label-statistics interior filter lets through vehicle cabins and multi-storey atria | Visible in the worst-case gallery. A scene classifier or scene-category filter would clean the training/val data |
| Noisy / inconsistent ground truth | Some ADE20K annotations disagree with the task's convention | Visible in the worst-case gallery. It caps measured IoU; the fix is in-domain relabelling |
| Arbitrary resolutions / aspect ratios | A 3558×2160 input would be expensive and shift the receptive field | Network runs on short side 384 (aspect preserved, /32). Decisions are made at ≤ 2048 px and nearest-upsampled |

### Experiments that changed the design (all measured, see §5)
* **Classical → learned:** val mIoU 0.354 → 0.795, and ~19× faster.
* **Guided-filter refinement: hypothesis rejected.** An 11-setting sweep (`results/refine_sweep.md`) found no radius/eps that beats plain bilinear upsampling on boundary F. The grayscale guide snaps to texture edges (plank seams, stone veins) as readily as to surface boundaries. It is off by default.
* **Flip-TTA: not worth it.** +0.002 val mIoU, a worse reference wall (0.930 → 0.906), 1.7× latency. Off by default.
* **Input resolution:** 384 px is the knee of the curve (320: −0.007 mIoU, 256: −0.026, 448: no gain).
* **INT8 post-training quantisation: rejected.** −5.4 mIoU (percentile calibration; min/max is worse) for only a 4 % latency reduction.

### Alternatives considered
1. **Purely classical** (vanishing points + GrabCut / superpixel region growing). Implemented as the baseline: no data needed and explainable, but it fails on colour-ambiguous scenes and is slower. Kept as evidence for the decision.
2. **Room-layout estimation** (fit a cuboid room, e.g. LayoutNet/HorizonNet-style corner and edge regression), then subtract objects. It gives very clean planar walls, but it needs layout-annotated data, assumes a cuboid and still needs a separate clutter mask. A good second stage, though.
3. **Monocular depth + surface normals → plane fitting** (floor = large horizontal plane at the bottom, walls = vertical planes). It would need a pre-trained depth network. That isn't a segmentation model, but it is a grey area under the rules and a heavy dependency, so it was not used.
4. **Self-supervised features (DINOv2/CLIP) + a small trained head.** Probably the most accurate per hour of work, but it means a large backbone, it's slower on CPU, and it's closer to "using a foundation segmenter" than I wanted.
5. **From-scratch encoder** (no ImageNet init). Supported by `--no-pretrained-encoder` but **not run** within the time budget. With 9k training images I'd expect slower convergence and lower accuracy. It is the obvious next ablation for a strict reading of the rules.

### How mask quality was evaluated
* **Held-out quantitative:** per-class IoU and mIoU over the 922-image ADE20K interior *validation* split. It is never trained on and is used only to pick the best epoch.
* **Boundary F-score** (tolerance 0.75 % of the diagonal), because IoU is dominated by large interior regions and barely moves when edges are off by 10 px. Edge quality is what a re-texturing product actually shows the user. This metric is what exposed that guided refinement was not helping, even though it nudged IoU up on the reference image.
* **Reference image** supplied with the task: IoU/BF against its GT masks.
* **Qualitative stratified review:** worst / median / best validation images by per-image mIoU, so failure modes are inspected rather than cherry-picked successes.
* **Every pipeline change is an ablation row** (refinement, TTA, input resolution, FP32 vs INT8) with quality *and* latency, so trade-offs are explicit.

### What I would do next for real-world robustness
1. **Data:** collect and label a few thousand in-domain photos (real-estate listings, phone photos, renders like the reference image) under the *product's* wall/floor convention. Add hard-negative mining of mirrors, glass partitions, glossy floors, open-plan spaces and empty rooms. Label noise and convention mismatch currently cost more than model capacity.
2. **Geometry-aware second stage:** fit planes or a room layout to the predicted wall/floor regions, regularise each wall to a planar polygon, and enforce wall–floor adjacency at a single skirting line. That gives crisp, product-grade boundaries (where the guided filter failed) and lets one wall be separated from another (useful for per-wall re-texturing).
3. **Better edge refinement:** a guide that encodes *surface identity* rather than intensity, i.e. the network's own stride-4 features, or a dense CRF / learned refinement head trained on boundary bands.
4. **Instance-level walls** (each wall plane separately), plus an explicit *ceiling* class.
5. **Uncertainty output:** expose per-pixel confidence (the softmax margin is already computed) so the product can ask the user to confirm low-confidence regions instead of failing silently.
6. **Compression done properly:** quantisation-aware training if the model must ship on-device, and a TensorRT / CoreML build for edge deployment.

### Limitations (honest summary)
* Learned priors come from ADE20K's annotation conventions, which won't always match a product's definition of "wall".
* No explicit 3-D reasoning, so mirrors and reflections can be labelled as the surface they reflect.
* Very unusual viewpoints (top-down atria, strong roll, fisheye) and scenes with little visible floor are weaker.
* A single "wall" mask merges all wall planes. Separate wall instances would need the next steps above.
* The from-scratch-encoder ablation was not run, so how much the ImageNet initialisation contributes is not quantified.
