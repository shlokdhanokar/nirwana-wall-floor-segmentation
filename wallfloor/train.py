"""Train WallFloorNet on the prepared ADE20K interior subset.

    python -m wallfloor.train --cache data/cache --out weights/wallfloor_mbv3.pt --epochs 30
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ThreadedLoader, WallFloorDataset, normalize_batch, pad_collate, worker_init
from .labels import CLASS_NAMES, IGNORE, NUM_CLASSES
from .metrics import ConfusionMatrix
from .model import WallFloorNet


def dice_loss(logits, target, eps=1.0):
    """Soft Dice on wall & floor; complements CE by optimising region overlap
    directly and is less dominated by the many easy interior pixels."""
    valid = (target != IGNORE).unsqueeze(1)
    prob = logits.softmax(1) * valid
    onehot = F.one_hot(target.clamp(max=NUM_CLASSES - 1), NUM_CLASSES).permute(0, 3, 1, 2) * valid
    prob, onehot = prob[:, 1:], onehot[:, 1:].float()
    inter = (prob * onehot).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    cm = ConfusionMatrix()
    for b, (x, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(normalize_batch(x.to(device, non_blocking=True)))
        cm.update(logits.argmax(1).cpu().numpy().ravel(), y.numpy().ravel())
    model.train()
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="weights/wallfloor_mbv3.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count()))
    ap.add_argument("--loader", choices=["process", "thread"], default="thread" if os.name == "nt" else "process",
                    help="thread = thread-pool loader (no per-worker torch/CUDA import; default on Windows)")
    ap.add_argument("--no-pretrained-encoder", action="store_true",
                    help="train the encoder from random init (no ImageNet weights at all)")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after N steps")
    a = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = WallFloorDataset(a.cache, "train", crop=a.crop, train=True)
    val_ds = WallFloorDataset(a.cache, "val", train=False)
    if a.loader == "thread":
        train_dl = ThreadedLoader(train_ds, a.batch, shuffle=True, drop_last=True, threads=a.workers)
        val_dl = ThreadedLoader(val_ds, 8, threads=a.workers, collate_fn=pad_collate)
    else:
        train_dl = DataLoader(train_ds, a.batch, shuffle=True, num_workers=a.workers, pin_memory=True,
                              drop_last=True, persistent_workers=a.workers > 0, worker_init_fn=worker_init)
        val_dl = DataLoader(val_ds, 8, num_workers=a.workers, collate_fn=pad_collate, worker_init_fn=worker_init)
    print(f"device={device} train={len(train_ds)} val={len(val_ds)}")

    model = WallFloorNet(pretrained_encoder=not a.no_pretrained_encoder).to(device)
    # encoder starts from generic features -> smaller LR than the fresh decoder
    enc = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    dec = [p for p in model.parameters() if id(p) not in enc_ids]
    enc_lr = a.lr * (0.3 if not a.no_pretrained_encoder else 1.0)
    opt = torch.optim.AdamW([{"params": enc, "lr": enc_lr}, {"params": dec, "lr": a.lr}], weight_decay=a.wd)
    total = a.epochs * len(train_dl)
    warm = min(500, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    # class weights: 'other' is the majority class; nudge the loss toward wall/floor
    ce_w = torch.tensor([0.8, 1.0, 1.2], device=device)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    history, best, step = [], -1.0, 0
    for epoch in range(a.epochs):
        t0, run = time.time(), []
        for x, y in train_dl:
            x = normalize_batch(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True).long()
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y, weight=ce_w, ignore_index=IGNORE)
            loss = loss + 0.5 * dice_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run.append(loss.item())
            step += 1
            if a.max_steps and step >= a.max_steps:
                break
        cm = evaluate(model, val_dl, device, max_batches=4 if a.max_steps else None)
        iou = cm.iou()
        score = float(iou[1:].mean())
        rec = {"epoch": epoch + 1, "loss": float(np.mean(run)), "time_s": round(time.time() - t0, 1),
               **{f"iou_{n}": round(float(v), 4) for n, v in zip(CLASS_NAMES, iou)},
               "miou_wall_floor": round(score, 4), "pixel_acc": round(float(cm.pixel_acc()), 4)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "model_kwargs": {}, "epoch": epoch + 1,
                        "val": rec, "args": vars(a)}, a.out)
        if a.max_steps and step >= a.max_steps:
            break
    with open(os.path.splitext(a.out)[0] + "_history.json", "w") as f:
        json.dump(history, f, indent=1)
    print(f"best val mIoU(wall,floor) = {best:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
