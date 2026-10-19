"""Mask quality metrics: confusion-matrix IoU and boundary F-score."""

import cv2
import numpy as np

from .labels import IGNORE, NUM_CLASSES


class ConfusionMatrix:
    def __init__(self, n=NUM_CLASSES):
        self.n = n
        self.m = np.zeros((n, n), dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray):
        valid = gt != IGNORE
        idx = gt[valid].astype(np.int64) * self.n + pred[valid].astype(np.int64)
        self.m += np.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def iou(self):
        tp = np.diag(self.m)
        return tp / np.maximum(self.m.sum(0) + self.m.sum(1) - tp, 1)

    def pixel_acc(self):
        return np.diag(self.m).sum() / max(self.m.sum(), 1)


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = (pred | gt).sum()
    return float((pred & gt).sum() / union) if union else 1.0


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol_frac=0.0075) -> float:
    """Boundary F-score (Perazzi et al. 2016): precision/recall of mask contours
    matched within a tolerance of tol_frac * image diagonal. IoU is dominated by
    large interior areas; this measures how well the *edges* line up, which is
    what matters for re-texturing a wall or floor."""
    pred, gt = pred.astype(np.uint8), gt.astype(np.uint8)
    tol = max(1, int(round(tol_frac * np.hypot(*gt.shape))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))

    def edges(m):
        return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0

    ep, eg = edges(pred), edges(gt)
    if not ep.any() and not eg.any():
        return 1.0
    prec = (ep & cv2.dilate(eg.astype(np.uint8), k).astype(bool)).sum() / max(ep.sum(), 1)
    rec = (eg & cv2.dilate(ep.astype(np.uint8), k).astype(bool)).sum() / max(eg.sum(), 1)
    return float(2 * prec * rec / max(prec + rec, 1e-9))
