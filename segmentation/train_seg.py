"""
Train the segmentation CNN (MobileNetV3-Small U-Net, or --model resnet18_unet) on the synthetic crops.

    python segmentation/train_seg.py --data data/segmentation_data --seed 0 \
        --epochs 35 --patience 35 --out runs/seg/seg_mnv3_s0
    python segmentation/train_seg.py ... --overfit 8      # sanity check: memorise 8 training crops

Loss: class-weighted cross-entropy + soft Dice on the two shadow classes. AdamW, linear warm-up then cosine
decay. Each epoch is scored on val; the checkpoint
with the best val inner-shadow IoU is kept as best.pt, and training stops after `patience` epochs without a
gain (--patience equal to --epochs disables early stopping). With --save-every N it also keeps
checkpoints/epoch_XXX.pt every N epochs. Writes history.csv, best.pt, last.pt, config.json.
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from seg_common import CLASS_NAMES, MODEL, SegDataset, SegLoss, build_model, confusion, iou_from_confusion


def worker_init(_):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)   # different per worker and epoch


def run_epoch(model, loader, crit, device, opt=None):
    train = opt is not None
    model.train(train)
    cm = np.zeros((3, 3), np.int64)
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = crit(logits, y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            tot += loss.item() * len(x)
            n += len(x)
            cm += confusion(logits.argmax(1).cpu().numpy().ravel(), y.cpu().numpy().ravel())
    return tot / n, iou_from_confusion(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100, help="the paper uses 35 (with --patience 35)")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overfit", type=int, default=0, help="train and score on this many training crops, no aug")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also keep checkpoints/epoch_XXX.pt every N epochs (0: only best.pt and last.pt)")
    ap.add_argument("--model", default=MODEL, choices=["mobilenet_unet", "resnet18_unet"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)

    if a.overfit:
        names = [l.strip() for l in open(os.path.join(a.data, "train.txt")) if l.strip()][:a.overfit]
        tr = SegDataset(a.data, "train", augment=False, names=names, seed=a.seed)
        va = SegDataset(a.data, "train", augment=False, names=names, seed=a.seed)
        a.patience = a.epochs
    else:
        tr = SegDataset(a.data, "train", augment=True, seed=a.seed)
        va = SegDataset(a.data, "val", augment=False, seed=a.seed)
    g = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(tr, batch_size=min(a.batch, len(tr)), shuffle=True, drop_last=not a.overfit,
                    num_workers=a.workers, worker_init_fn=worker_init, generator=g, pin_memory=True)
    vl = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True)

    model = build_model(a.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / a.warmup if e < a.warmup
        else 0.5 * (1 + math.cos(math.pi * (e - a.warmup) / max(1, a.epochs - a.warmup))))
    crit = SegLoss()
    cfg = dict(vars(a), model=a.model, n_params=n_params, device=str(device),
               gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
               n_train=len(tr), n_val=len(va))
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    print(json.dumps(cfg))

    fields = ["epoch", "lr", "train_loss", "val_loss"] + \
             [f"val_iou_{c}" for c in CLASS_NAMES] + ["val_miou", "train_iou_inner_shadow", "seconds"]
    hist = open(os.path.join(a.out, "history.csv"), "w", newline="")
    wr = csv.DictWriter(hist, fieldnames=fields)
    wr.writeheader()
    best, best_ep = -1.0, -1
    for ep in range(a.epochs):
        t0 = time.time()
        lr = opt.param_groups[0]["lr"]
        tl_loss, tr_iou = run_epoch(model, tl, crit, device, opt)
        sched.step()
        vl_loss, iou = run_epoch(model, vl, crit, device)
        row = {"epoch": ep + 1, "lr": round(lr, 7), "train_loss": round(tl_loss, 5), "val_loss": round(vl_loss, 5),
               **{f"val_iou_{c}": round(float(v), 5) for c, v in zip(CLASS_NAMES, iou)},
               "val_miou": round(float(np.nanmean(iou)), 5), "train_iou_inner_shadow": round(float(tr_iou[2]), 5),
               "seconds": round(time.time() - t0, 2)}
        wr.writerow(row)
        hist.flush()
        score = float(np.nan_to_num(iou[2]))
        state = {"model": model.state_dict(), "epoch": ep + 1, "val_iou_inner_shadow": score, "config": cfg}
        torch.save(state, os.path.join(a.out, "last.pt"))
        if a.save_every and (ep + 1) % a.save_every == 0:                    # kept for later comparison
            os.makedirs(os.path.join(a.out, "checkpoints"), exist_ok=True)
            torch.save(state, os.path.join(a.out, "checkpoints", f"epoch_{ep + 1:03d}.pt"))
        flag = ""
        if score > best:
            best, best_ep = score, ep + 1
            torch.save(state, os.path.join(a.out, "best.pt"))
            flag = " *"
        print(f"ep {ep + 1:3d} lr {lr:.2e} loss {tl_loss:.4f}/{vl_loss:.4f} "
              f"val IoU bg {iou[0]:.3f} ground {iou[1]:.3f} inner {iou[2]:.3f} ({row['seconds']:.1f}s){flag}",
              flush=True)
        if ep + 1 - best_ep >= a.patience:
            print(f"early stop: no gain in val inner-shadow IoU for {a.patience} epochs")
            break
    hist.close()
    json.dump({"best_epoch": best_ep, "best_val_iou_inner_shadow": best, "epochs_run": ep + 1},
              open(os.path.join(a.out, "train_summary.json"), "w"), indent=1)
    print(f"best val inner-shadow IoU {best:.4f} at epoch {best_ep}")


if __name__ == "__main__":
    main()
