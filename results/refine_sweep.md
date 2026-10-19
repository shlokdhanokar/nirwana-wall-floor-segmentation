# Guided-filter refinement sweep

Hypothesis: the network runs at ~384 px, so snapping its probabilities to full-resolution image edges with a
guided filter (grayscale guide) should sharpen mask boundaries.

Setup: trained WallFloorNet (ONNX FP32, short side 384), 200 random ADE20K interior *validation* images plus
the 3558×2160 reference image. Every row includes the same connected-component cleanup.
`None` = bilinear upsampling of the probabilities (no guided filter).
BF = boundary F-score at a tolerance of 0.75 % of the image diagonal. Produced with `scripts/` tooling on 2026-10-19.

| refinement (radius, eps, work res) | val mIoU | val BF wall | val BF floor | sample IoU wall | sample IoU floor | sample BF wall | sample BF floor |
|---|---|---|---|---|---|---|---|
| **None (bilinear)** | 0.8095 | **0.7155** | **0.7793** | 0.9304 | 0.9456 | 0.5773 | **0.7670** |
| r=2, eps=1e-3, 768 | **0.8098** | 0.7140 | 0.7774 | 0.9325 | 0.9459 | 0.5798 | 0.7489 |
| r=2, eps=1e-2, 768 | 0.8095 | 0.7135 | 0.7767 | 0.9310 | 0.9461 | 0.5729 | 0.7443 |
| r=2, eps=1e-1, 768 | 0.8091 | 0.7127 | 0.7759 | 0.9293 | 0.9452 | 0.5694 | 0.7459 |
| r=4, eps=1e-3, 768 | 0.8092 | 0.7105 | 0.7735 | 0.9329 | 0.9472 | 0.5749 | 0.7346 |
| r=4, eps=1e-2, 768 | 0.8087 | 0.7092 | 0.7713 | 0.9316 | 0.9471 | 0.5679 | 0.7264 |
| r=4, eps=1e-1, 768 | 0.8079 | 0.7060 | 0.7686 | 0.9280 | 0.9447 | 0.5595 | 0.7245 |
| r=8, eps=1e-3, 768 (initial default) | 0.8069 | 0.6988 | 0.7634 | **0.9342** | **0.9500** | **0.5909** | 0.6991 |
| r=8, eps=1e-2, 768 | 0.8063 | 0.6945 | 0.7589 | 0.9331 | 0.9494 | 0.5861 | 0.6939 |
| r=8, eps=1e-1, 768 | 0.8043 | 0.6839 | 0.7489 | 0.9275 | 0.9435 | 0.5661 | 0.6704 |
| r=4, eps=1e-2, 1024 | 0.8088 | 0.7092 | 0.7712 | 0.9313 | 0.9464 | 0.5697 | 0.7377 |
| r=8, eps=1e-2, 1536 | 0.8063 | 0.6944 | 0.7588 | 0.9314 | 0.9469 | 0.5666 | 0.7279 |

**Conclusion:** the hypothesis is not supported. No setting beats plain bilinear upsampling on validation boundary
quality, and the larger radii that add ~0.004 IoU on the reference image cost 0.07 floor boundary-F on it. The
network's boundaries are already well placed at 384 px. A grayscale guide also pulls them toward strong *texture*
edges (plank seams, stone veins, rug patterns) rather than surface boundaries. Refinement is therefore **off by
default** (`WallFloorSegmenter(..., refine=False)`) and kept as an option. A guide that encodes surface identity rather than
intensity (e.g. the network's own features, or a dense CRF with learned pairwise terms) would be the next thing to try.
