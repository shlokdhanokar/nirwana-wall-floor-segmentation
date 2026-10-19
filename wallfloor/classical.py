"""Training-free baseline: perspective geometry + GrabCut.

This was the first approach tried; it uses no learned weights of any kind.

1. Horizon estimation. Line segments (OpenCV LSD) that are neither vertical nor
   horizontal in the image are, in a Manhattan-world room, mostly receding
   edges (floor planks, skirting, wall/ceiling corners) that converge at
   vanishing points on the horizon. The median height of their pairwise
   intersections estimates the horizon (camera eye level).
2. Floor: GrabCut (colour GMMs + graph cut) seeded with "probably floor" below
   the horizon, "definitely floor" in a bottom-centre band and "background"
   above the horizon.
3. Wall: GrabCut seeded with "probably wall" between the ceiling band and the
   horizon, excluding the floor result; ceiling band and floor are background.

Assumptions (and therefore failure modes): level-ish camera, the floor is
visible at the bottom-centre, walls are visible above eye level, and
wall/floor colours differ from the furniture in front of them.
"""

import cv2
import numpy as np


def estimate_horizon(gray: np.ndarray, rng_seed=0) -> float:
    """Return horizon height as a fraction of image height (0=top)."""
    H, W = gray.shape
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return 0.5
    l = lines.reshape(-1, 4)
    d = l[:, 2:] - l[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    ang = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))  # 0=horizontal, 90=vertical
    keep = (length > 0.04 * np.hypot(H, W)) & (ang > 4) & (ang < 75)
    l, length = l[keep], length[keep]
    if len(l) < 4:
        return 0.5
    # homogeneous line coordinates, pairwise intersections of a random subset
    p1 = np.c_[l[:, :2], np.ones(len(l))]
    p2 = np.c_[l[:, 2:], np.ones(len(l))]
    L = np.cross(p1, p2)
    rng = np.random.default_rng(rng_seed)
    n = min(3000, len(l) * (len(l) - 1) // 2)
    i, j = rng.integers(0, len(l), n), rng.integers(0, len(l), n)
    m = i != j
    X = np.cross(L[i[m]], L[j[m]])
    ok = np.abs(X[:, 2]) > 1e-6
    y = X[ok, 1] / X[ok, 2]
    x = X[ok, 0] / X[ok, 2]
    w = (length[i[m]] * length[j[m]])[ok]
    sel = (y > -0.25 * H) & (y < 1.25 * H) & (x > -W) & (x < 2 * W)
    if sel.sum() < 5:
        return 0.5
    order = np.argsort(y[sel])
    cw = np.cumsum(w[sel][order])
    y_med = y[sel][order][np.searchsorted(cw, cw[-1] / 2)]  # weighted median
    return float(np.clip(y_med / H, 0.25, 0.75))


def _grabcut(img_bgr, init, iters=5):
    mask = init.copy()
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


def segment_classical(img_rgb: np.ndarray, work_width=480):
    H0, W0 = img_rgb.shape[:2]
    s = work_width / W0
    img = cv2.resize(img_rgb, (work_width, round(H0 * s)), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bgr = cv2.bilateralFilter(bgr, 7, 40, 7)  # flatten texture (wood grain, stone veins) before colour modelling
    hz = int(estimate_horizon(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)) * H)
    ys = np.arange(H)[:, None] * np.ones((1, W), int)

    # floor
    init = np.full((H, W), cv2.GC_PR_BGD, np.uint8)
    init[ys > hz + 0.05 * H] = cv2.GC_PR_FGD
    init[ys < hz] = cv2.GC_BGD
    init[int(0.93 * H):, int(0.2 * W):int(0.8 * W)] = cv2.GC_FGD
    floor = _grabcut(bgr, init)

    # wall
    init = np.full((H, W), cv2.GC_PR_BGD, np.uint8)
    init[(ys > 0.1 * H) & (ys < hz)] = cv2.GC_PR_FGD
    init[: max(1, int(0.03 * H))] = cv2.GC_BGD
    init[floor] = cv2.GC_BGD
    wall = _grabcut(bgr, init) & ~floor

    up = lambda m: cv2.resize(m.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)  # noqa: E731
    return {"wall": up(wall), "floor": up(floor), "horizon": hz / H}
