"""Visualisation helpers: overlays, side-by-side panels, saving masks."""

import os

import cv2
import numpy as np

WALL_COLOR = np.array([66, 133, 244], np.float32)   # blue
FLOOR_COLOR = np.array([255, 160, 0], np.float32)   # amber


def overlay(img, wall, floor, alpha=0.5):
    out = img.astype(np.float32).copy()
    out[wall] = (1 - alpha) * out[wall] + alpha * WALL_COLOR
    out[floor] = (1 - alpha) * out[floor] + alpha * FLOOR_COLOR
    out = out.astype(np.uint8)
    for m, c in ((wall, WALL_COLOR), (floor, FLOOR_COLOR)):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, tuple(int(v) for v in c), max(1, img.shape[1] // 600))
    return out


def show(img, out, title=None, gt=None, figsize=(20, 5)):
    """Original | wall mask | floor mask | overlay (matplotlib)."""
    import matplotlib.pyplot as plt
    panels = [(img, "Input"), (out["wall"], "Wall mask"), (out["floor"], "Floor mask"),
              (overlay(img, out["wall"], out["floor"]), "Overlay (blue=wall, amber=floor)")]
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1 if im.ndim == 2 else None)
        ax.set_title(t)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_outputs(img, out, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_wall.png"), out["wall"].astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_floor.png"), out["floor"].astype(np.uint8) * 255)
    ov = overlay(img, out["wall"], out["floor"])
    s = min(1.0, 1600 / img.shape[1])
    ov = cv2.resize(ov, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_overlay.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def panel(img, out, width=1600):
    """Single RGB image: input | wall | floor | overlay, for README/reports."""
    s = width / 4 / img.shape[1]
    r = lambda a: cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)  # noqa: E731
    m2rgb = lambda m: np.repeat((m.astype(np.uint8) * 255)[..., None], 3, 2)  # noqa: E731
    tiles = [img, m2rgb(out["wall"]), m2rgb(out["floor"]), overlay(img, out["wall"], out["floor"])]
    return np.concatenate([r(t) for t in tiles], 1)
