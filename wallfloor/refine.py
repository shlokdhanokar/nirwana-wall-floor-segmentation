"""Classical post-processing that turns coarse network probabilities into
full-resolution, edge-aligned masks.

1. Fast guided filter (He & Sun 2015): the network runs at ~384 px, so its
   probabilities are blurry at full resolution. We fit a local linear model
   q = A*I + B between the guide image I and each probability map at a working
   resolution, then upsample only A and B and apply them to the full-res guide.
   Boundaries snap to real image edges (wall/ceiling corners, skirting boards,
   furniture outlines) at the cost of a handful of box filters.
   The filter is linear and the class probabilities sum to 1, so filtering
   wall and floor is enough: q_other = 1 - q_wall - q_floor.
2. Connected-component cleanup: drop tiny islands and fill pin-holes.
"""

import cv2
import numpy as np


def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_coefficients(guide: np.ndarray, probs: np.ndarray, radius=8, eps=1e-3):
    """Per-pixel linear coefficients A, B (hxwxC) such that q = A*guide + B."""
    I3 = guide[..., None]
    mI = _box(guide, radius)[..., None]
    mP = _box(probs, radius).reshape(probs.shape)
    varI = _box(guide * guide, radius)[..., None] - mI * mI
    cov = _box(I3 * probs, radius).reshape(probs.shape) - mI * mP
    A = cov / (varI + eps)
    B = mP - A * mI
    return _box(A, radius).reshape(probs.shape), _box(B, radius).reshape(probs.shape)


def refine_wall_floor(gray: np.ndarray, probs: np.ndarray, work_long=768, radius=8, eps=1e-3):
    """gray: uint8 guide at the *output* resolution; probs: hxwx3 (other, wall, floor)
    at network resolution. Returns (wall, floor) uint8 masks (0/255) at the guide's
    resolution: the argmax of the guided-filtered probabilities.

    Speed trick: argmax over {other, wall, floor} with q_other = 1 - q_w - q_f is
        wall  <=> q_w - q_f >= 0  and  2 q_w + q_f - 1 > 0
        floor <=> q_w - q_f <  0  and  2 q_f + q_w - 1 > 0
    and every q is linear in the guide (q = A*I + B), so the three decision
    functions are linear in I too. Their coefficients are combined at low
    resolution; at output resolution only single-channel resize / multiply-add /
    compare run, and no probability maps are materialised."""
    H, W = gray.shape
    s = min(1.0, work_long / max(H, W))
    w, h = max(1, round(W * s)), max(1, round(H * s))
    I = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) * (1 / 255)
    P = cv2.resize(np.ascontiguousarray(probs[..., 1:3], dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    A, B = guided_coefficients(I, P, radius, eps)
    Aw, Af, Bw, Bf = A[..., 0], A[..., 1], B[..., 0], B[..., 1]
    DA = [(Aw - Af) / 255, (2 * Aw + Af) / 255, (2 * Af + Aw) / 255]  # applied to the 0..255 guide
    DB = [Bw - Bf, 2 * Bw + Bf - 1, 2 * Bf + Bw - 1]
    g = gray.astype(np.float32)
    up = lambda x: cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR)  # noqa: E731
    return _decide(*[cv2.add(cv2.multiply(up(da), g), up(db)) for da, db in zip(DA, DB)])


def upsample_wall_floor(shape, probs):
    """Baseline without refinement: bilinear upsampling of the probabilities to `shape`."""
    H, W = shape[:2]
    pw, pf = probs[..., 1].astype(np.float32), probs[..., 2].astype(np.float32)
    up = lambda x: cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR)  # noqa: E731
    return _decide(up(pw - pf), up(2 * pw + pf - 1), up(2 * pf + pw - 1))


def _decide(d_wf, d_w, d_f):
    wall_side = cv2.compare(d_wf, 0, cv2.CMP_GE)
    wall = cv2.bitwise_and(wall_side, cv2.compare(d_w, 0, cv2.CMP_GT))
    floor = cv2.bitwise_and(cv2.bitwise_not(wall_side), cv2.compare(d_f, 0, cv2.CMP_GT))
    return wall, floor


def _small_components(m: np.ndarray, max_frac: float, connectivity: int) -> np.ndarray:
    """uint8 0/255 map of pixels belonging to components of m smaller than max_frac of the image."""
    _, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=connectivity)
    small = stats[:, cv2.CC_STAT_AREA] < max_frac * m.size
    small[0] = False
    return (small.astype(np.uint8) * 255)[cc]  # lookup table instead of np.isin: one gather


def clean_mask(mask: np.ndarray, min_island=0.002, max_hole=0.0005, work_long=1024) -> np.ndarray:
    """Remove connected components smaller than min_island * area and fill holes
    smaller than max_hole * area. Holes are only *small* ones: big holes in a
    floor mask are furniture and must stay holes. mask: uint8 0/255 -> uint8 0/255.

    Component labelling runs on a max-pooled copy (long side <= work_long), so a
    12 MP photo costs about as much as a 1 MP one. Max-pooling keeps it conservative: a
    full-res speck can only be dropped if it is also disconnected at low res,
    and a dropped low-res cell contains nothing but that speck."""
    H, W = mask.shape
    s = min(1.0, work_long / max(H, W))
    size = (max(1, round(W * s)), max(1, round(H * s)))
    up = lambda x: x if s == 1.0 else cv2.resize(x, (W, H), interpolation=cv2.INTER_NEAREST)  # noqa: E731
    pool = lambda x: x if s == 1.0 else cv2.compare(cv2.resize(x, size, interpolation=cv2.INTER_AREA), 0, cv2.CMP_GT)  # noqa: E731
    drop = up(_small_components(pool(mask), min_island, 8))
    out = cv2.bitwise_and(mask, cv2.bitwise_not(drop))
    fill = up(_small_components(pool(cv2.bitwise_not(out)), max_hole, 4))
    return cv2.bitwise_or(out, fill)
