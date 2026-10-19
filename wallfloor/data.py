"""Training / evaluation dataset over the cache built by wallfloor.prepare."""

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import IGNORE, MEAN, STD


def normalize(img_rgb: np.ndarray) -> torch.Tensor:
    x = (img_rgb.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


_MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
_STD_T = torch.tensor(STD).view(1, 3, 1, 1)


def normalize_batch(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 NCHW batch (already on the training device) -> normalised float.
    Done on the GPU so the loader threads only move bytes."""
    return (x_uint8.float() - _MEAN_T.to(x_uint8.device)) / _STD_T.to(x_uint8.device)


class WallFloorDataset(Dataset):
    """Returns (uint8 CHW image tensor, uint8 HW label tensor); normalise with normalize_batch."""

    def __init__(self, cache_root, split, crop=384, train=True, base_short=384):
        with open(os.path.join(cache_root, "manifest.json")) as f:
            self.names = json.load(f)[split]
        self.dir = os.path.join(cache_root, split)
        self.crop, self.train, self.base_short = crop, train, base_short

    def __len__(self):
        return len(self.names)

    def _load(self, i):
        p = os.path.join(self.dir, self.names[i])
        img = cv2.cvtColor(cv2.imread(p + ".jpg"), cv2.COLOR_BGR2RGB)
        lab = cv2.imread(p + ".png", cv2.IMREAD_GRAYSCALE)
        return img, lab

    def __getitem__(self, i):
        img, lab = self._load(i)
        if self.train:
            img, lab = self._augment(img, lab)
        else:  # evaluate on the whole image, short side = base_short, padded to /32
            s = self.base_short / min(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), torch.from_numpy(lab)

    def _augment(self, img, lab):
        """Scale, crop, rotation, flip and padding composed into ONE affine warp,
        so only the c x c output is ever resampled."""
        rng = np.random
        h, w = lab.shape
        c = self.crop
        s = self.base_short / min(h, w) * rng.uniform(0.6, 1.6)  # random scale around the base size
        cs = c / s  # crop side in source pixels
        x0 = rng.uniform(0, max(w - cs, 0))
        y0 = rng.uniform(0, max(h - cs, 0))
        M = np.array([[s, 0, -s * x0], [0, s, -s * y0], [0, 0, 1]])
        if rng.rand() < 0.3:  # small roll: handheld photos are rarely perfectly level
            R = cv2.getRotationMatrix2D((c / 2, c / 2), rng.uniform(-8, 8), 1.0)
            M = np.vstack([R, [0, 0, 1]]) @ M
        if rng.rand() < 0.5:  # horizontal flip is label-preserving; vertical flip is NOT (floor is down)
            M = np.array([[-1, 0, c - 1], [0, 1, 0], [0, 0, 1]]) @ M
        M = M[:2]
        img = cv2.warpAffine(img, M, (c, c), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        lab = cv2.warpAffine(lab, M, (c, c), flags=cv2.INTER_NEAREST, borderValue=IGNORE)
        return self._photometric(img, rng), lab

    @staticmethod
    def _photometric(img, rng):
        """Colour / exposure jitter. Walls and floors come in every colour, and
        interior photos range from dim tungsten to blown-out daylight.
        Implemented with uint8 lookup tables (OpenCV releases the GIL)."""
        ramp = np.arange(256, dtype=np.float32)
        hh, ss, vv = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        hh = cv2.LUT(hh, ((ramp + rng.uniform(-12, 12)) % 180).astype(np.uint8))
        ss = cv2.LUT(ss, np.clip(ramp * rng.uniform(0.6, 1.4), 0, 255).astype(np.uint8))
        vv = cv2.LUT(vv, np.clip(ramp * rng.uniform(0.65, 1.35), 0, 255).astype(np.uint8))
        img = cv2.cvtColor(cv2.merge([hh, ss, vv]), cv2.COLOR_HSV2RGB)
        alpha = rng.uniform(0.75, 1.25)  # contrast around the mean intensity
        img = cv2.addWeighted(img, alpha, img, 0, (1 - alpha) * float(np.mean(cv2.mean(img)[:3])))
        if rng.rand() < 0.1:
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        if rng.rand() < 0.2:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
        if rng.rand() < 0.2:
            noise = np.empty(img.shape, np.int16)
            cv2.randn(noise, 0, rng.uniform(2, 8))
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img


class ThreadedLoader:
    """Minimal DataLoader replacement backed by a thread pool with bounded prefetch.

    On Windows every DataLoader worker is a spawned process that re-imports torch
    and maps the CUDA DLLs (~1 GB of commit each), which exhausts the page file on
    memory-constrained machines. The augmentation here is OpenCV-heavy and OpenCV
    releases the GIL, so threads parallelise it well at no extra memory cost."""

    def __init__(self, ds, batch_size, shuffle=False, drop_last=False, threads=6, collate_fn=None, prefetch=4):
        from concurrent.futures import ThreadPoolExecutor
        self.ds, self.bs, self.shuffle, self.drop_last = ds, batch_size, shuffle, drop_last
        self.collate = collate_fn or (lambda b: (torch.stack([x for x, _ in b]), torch.stack([y for _, y in b])))
        self.pool = ThreadPoolExecutor(threads)
        self.prefetch = prefetch

    def __len__(self):
        n = len(self.ds) // self.bs
        return n if self.drop_last or len(self.ds) % self.bs == 0 else n + 1

    def __iter__(self):
        from collections import deque
        idx = np.random.permutation(len(self.ds)) if self.shuffle else np.arange(len(self.ds))
        batches = [idx[i:i + self.bs] for i in range(0, len(idx), self.bs)]
        if self.drop_last and len(batches) and len(batches[-1]) < self.bs:
            batches.pop()
        pending = deque()
        it = iter(batches)
        for b in it:
            pending.append([self.pool.submit(self.ds.__getitem__, int(i)) for i in b])
            if len(pending) >= self.prefetch:
                break
        while pending:
            futs = pending.popleft()
            nxt = next(it, None)
            if nxt is not None:
                pending.append([self.pool.submit(self.ds.__getitem__, int(i)) for i in nxt])
            yield self.collate([f.result() for f in futs])


def worker_init(_):
    # workers parallelise across images; nested OpenCV thread pools only cost memory
    cv2.setNumThreads(0)


def pad_collate(batch):
    """Pads variable-sized eval images to a common size divisible by 32."""
    H = max(x.shape[1] for x, _ in batch)
    W = max(x.shape[2] for x, _ in batch)
    H, W = -(-H // 32) * 32, -(-W // 32) * 32
    xs = torch.zeros(len(batch), 3, H, W, dtype=torch.uint8)
    ys = torch.full((len(batch), H, W), IGNORE, dtype=torch.uint8)
    for k, (x, y) in enumerate(batch):
        xs[k, :, : x.shape[1], : x.shape[2]] = x
        ys[k, : y.shape[0], : y.shape[1]] = y
    return xs, ys
