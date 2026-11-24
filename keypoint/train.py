"""Train the keypoint model. Reads only the train and val splits; test is never touched.

    python keypoint/train.py --config keypoint/config.yaml [key=value ...]
    python keypoint/train.py --resume runs/keypoint/kp_r18_s0/weights/last.pt

Per epoch: train and val L, L_hm, L_vis, val mean keypoint error (native px) and val phi_R MAE go to
<run>/results.csv, <run>/losses.png and one log line. best.pt = lowest val loss.
"""
import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry_kp as G  # noqa: E402
import physics_kp as P  # noqa: E402
from dataset import KeypointDataset  # noqa: E402
from model import KeypointNet, decode_heatmaps, keypoint_loss, loss_cfg  # noqa: E402
from plot_losses import plot_run  # noqa: E402

PARTS = ["loss", "loss_hm", "loss_vis", "loss_coord", "loss_disp"]  # coord/disp are 0 unless enabled
FIELDS = ["epoch", *[f"train_{p}" for p in PARTS], *[f"val_{p}" for p in PARTS],
          "val_px_err", "val_phi_mae", "lr", "epoch_time_s"]
SELECT = {"val_loss": lambda loss, px, phi: loss,     # original rule
          "val_px_err": lambda loss, px, phi: px,      # lowest val keypoint error (recommended)
          "val_phi_mae": lambda loss, px, phi: phi}


def parse_overrides(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def aux_ramp(cfg, epoch, aux_on=None):
    """Weight of L_coord / L_disp in this epoch: 0 before the start epoch, then a linear ramp to 1 over
    aux_ramp_epochs. Location losses only make sense once the heatmaps have peaks: switched on while the
    heatmaps are still blank they dominate the loss and destabilise training (seen in a CPU check).
    With aux_hm_gate set, the start epoch is `aux_on`: the epoch after train L_hm first fell below the gate."""
    c = loss_cfg(cfg)
    start, ramp = c["aux_start_epoch"], c["aux_ramp_epochs"]
    if c["aux_hm_gate"] is not None:
        if aux_on is None:
            return 0.0
        start = max(start, aux_on)
    if epoch < start:
        return 0.0
    return 1.0 if ramp <= 0 else min(1.0, (epoch - start + 1) / ramp)


def run_loss(model_out, b, device, cfg, aux_scale=1.0):
    hm, val = model_out
    stride = cfg["input_size"] / cfg["heatmap_size"]
    return keypoint_loss(hm, val, b["target"].to(device), b["v"].to(device), cfg, kp=b["kp"].to(device),
                         rho=b["rho"].to(device), stride=stride, aux_scale=aux_scale)


def part_values(parts):
    return np.array([float(parts[p]) if p in parts else 0.0 for p in PARTS])


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (0.01 + 0.99 * step / max(warmup, 1))
    p = (step - warmup) / max(total - warmup, 1)
    return base * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def predict(model, loader, device, cfg, aux_scale=1.0):
    """Run the model; return losses and per-image native-pixel keypoints and validity probabilities."""
    model.eval()
    tot = np.zeros(len(PARTS))
    n = 0
    kp_native, prob, vis, idx = [], [], [], []
    stride = cfg["input_size"] / cfg["heatmap_size"]
    for b in loader:
        img = b["img"].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=cfg["amp"] and device.type == "cuda"):
            hm, val = model(img)
        _, parts = run_loss((hm, val), b, device, cfg, aux_scale)
        tot += part_values(parts) * len(img)
        n += len(img)
        kp_out = decode_heatmaps(hm.float().cpu().numpy(), stride)
        for j in range(len(img)):
            Minv = G.invert_affine(b["M"][j].numpy())
            kp_native.append(G.apply_affine(Minv, kp_out[j]))
        prob.append(torch.sigmoid(val.float()).cpu().numpy())
        vis.append(b["v"].numpy())
        idx.append(b["idx"].numpy())
    return {"loss": tot / max(n, 1), "kp": np.array(kp_native), "prob": np.concatenate(prob),
            "v": np.concatenate(vis), "idx": np.concatenate(idx)}


def quick_metrics(res, recs):
    """Mean keypoint error over visible keypoints (native px) and phi_R MAE (median fusion)."""
    errs, phi_err = [], []
    for kp, prob, v, i in zip(res["kp"], res["prob"], res["v"], res["idx"]):
        r = recs[i]
        gt = np.array(r["kp_native"])
        m = v == 2
        errs.append(np.linalg.norm(kp[m] - gt[m], axis=1))
        use = prob > 0.5
        est = P.estimate(kp, use[G.GROUP_SLICE["inner_shadow"]], use[G.GROUP_SLICE["ground_shadow"]], r["physics"])
        if not np.isnan(est["phi_R"]):
            phi_err.append(abs(est["phi_R"] - P.phi_true(r["physics"])))
    return float(np.concatenate(errs).mean()), float(np.mean(phi_err)) if phi_err else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="keypoint/config.yaml")
    ap.add_argument("--resume", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        cfg.update(parse_overrides(args.overrides))
    else:
        cfg = yaml.safe_load(open(args.config))
        cfg.update(parse_overrides(args.overrides))
    seed_all(cfg["seed"])
    run = Path(cfg["project"]) / cfg["name"]
    (run / "weights").mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run / "config.yaml", "w"), sort_keys=False)
    log = open(run / "train.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = KeypointDataset(cfg["data"], "train", cfg, train=True)
    val_ds = KeypointDataset(cfg["data"], "val", cfg, train=False)
    train_set, val_set = train_ds, val_ds
    if cfg.get("limit"):
        train_set = Subset(train_ds, range(min(cfg["limit"], len(train_ds))))
        val_set = Subset(val_ds, range(min(cfg["limit"], len(val_ds))))
    say(f"device {device} | train {len(train_set)} | val {len(val_set)} | test not loaded")
    kw = dict(num_workers=cfg["workers"], pin_memory=device.type == "cuda", persistent_workers=cfg["workers"] > 0)
    train_dl = DataLoader(train_set, batch_size=cfg["batch"], shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_set, batch_size=cfg["batch"], shuffle=False, **kw)

    model = KeypointNet(cfg["num_kp"], cfg["deconv_layers"], pretrained=cfg["pretrained"] and ckpt is None,
                        vis_detach=cfg.get("vis_detach", False)).to(device)
    select = cfg.get("select_metric", "val_loss")
    say(f"loss: { {k: v for k, v in loss_cfg(cfg).items()} } | vis_detach {cfg.get('vis_detach', False)} | "
        f"best.pt chosen by {select}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=cfg["amp"] and device.type == "cuda")
    start, best, best_px = 1, float("inf"), float("inf")
    aux_on = None  # first epoch with L_coord / L_disp enabled (aux_hm_gate)
    csv_path = run / "results.csv"
    rewarm_from = None  # (first step, lr at that step) when a resumed run's schedule jumps up
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start, best = ckpt["epoch"] + 1, ckpt["best"]
        best_px = ckpt.get("best_px", float("inf"))
        aux_on = ckpt.get("aux_on")
        # Keep results.csv rows of the epochs already done (drop rows of any later, interrupted attempt).
        with open(csv_path) as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < start]
        with open(csv_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(rows)
        if "best_px" not in ckpt and rows:  # checkpoint from before best_kp.pt existed
            i = min(range(len(rows)), key=lambda j: float(rows[j]["val_px_err"]))
            best_px = float(rows[i]["val_px_err"])
            if int(rows[i]["epoch"]) == ckpt["epoch"]:
                torch.save({**ckpt, "best_px": best_px}, run / "weights" / "best_kp.pt")
        prev_lr = ckpt["opt"]["param_groups"][0]["lr"]
        rewarm_from = ((start - 1) * len(train_dl), prev_lr)
        say(f"resumed from {args.resume} at epoch {start} (best val L {best:.5f}, best val px {best_px:.2f})")
    n_params = sum(p.numel() for p in model.parameters())
    say(f"params {n_params / 1e6:.2f}M | epochs {cfg['epochs']} | batch {cfg['batch']} | steps/epoch {len(train_dl)}")

    total = cfg["epochs"] * len(train_dl)
    warm = cfg["warmup_epochs"] * len(train_dl)
    if start == 1:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)
    for epoch in range(start, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        tot, n = np.zeros(len(PARTS)), 0
        aux = aux_ramp(cfg, epoch, aux_on)
        for it, b in enumerate(train_dl):
            step = (epoch - 1) * len(train_dl) + it
            lr = lr_at(step, total, warm, cfg["lr"])
            if rewarm_from is not None:
                # Extending a finished schedule (e.g. 10 -> 30 epochs): the cosine for the new length is
                # far above the old final lr, so ramp linearly to it over one epoch instead of jumping.
                s0, lr0 = rewarm_from
                f = (step - s0 + 1) / len(train_dl)
                if f < 1 and lr > lr0:
                    lr = lr0 + f * (lr - lr0)
            for g in opt.param_groups:
                g["lr"] = lr
            img = b["img"].to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=cfg["amp"] and device.type == "cuda"):
                out = model(img)
            l, parts = run_loss(out, b, device, cfg, aux)
            opt.zero_grad(set_to_none=True)
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
            tot += part_values(parts) * len(img)
            n += len(img)
        tr = tot / max(n, 1)
        gate = loss_cfg(cfg)["aux_hm_gate"]
        if gate is not None and aux_on is None and tr[1] < gate:
            aux_on = epoch + 1
            say(f"train L_hm {tr[1]:.5f} < aux_hm_gate {gate}: L_coord / L_disp start in epoch {aux_on}")
        res = predict(model, val_dl, device, cfg, aux)
        px, phi = quick_metrics(res, val_ds.recs)
        dt = time.time() - t0
        row = [epoch, *tr, *res["loss"], px, phi, lr, dt]
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{x:.6g}" if isinstance(x, float) else x for x in row])
        score = SELECT[select](res["loss"][0], px, phi)
        is_best = score < best
        best = min(best, score)
        is_best_px = px < best_px
        best_px = min(best_px, px)
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "best": best, "best_px": best_px, "aux_on": aux_on, "cfg": cfg}
        torch.save(state, run / "weights" / "last.pt")
        if is_best:
            torch.save(state, run / "weights" / "best.pt")
        if is_best_px:  # lowest val keypoint error (L is dominated by the validity term)
            torch.save(state, run / "weights" / "best_kp.pt")
        if cfg.get("save_period") and epoch % int(cfg["save_period"]) == 0:   # kept for later comparison:
            torch.save({k: state[k] for k in ("model", "epoch", "best", "best_px", "aux_on", "cfg")},  # weights only
                       run / "weights" / f"epoch_{epoch:03d}.pt")
        plot_run(run)
        def fmt(x):
            s = f"L {x[0]:.5f} hm {x[1]:.5f} vis {x[2]:.4f}"
            return s + (f" coord {x[3]:.3f} disp {x[4]:.3f}" if x[3] or x[4] else "")
        say(f"epoch {epoch}/{cfg['epochs']} | train {fmt(tr)} | val {fmt(res['loss'])} | "
            f"val px {px:.2f} phi_mae {phi:.4f} | aux {aux:.2f} | lr {lr:.2e} | {dt:.0f}s{' *best' if is_best else ''}{' *best_kp' if is_best_px else ''}")
    say(f"done. best {select} {best:.5f} -> {run / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
