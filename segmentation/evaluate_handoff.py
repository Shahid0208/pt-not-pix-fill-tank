"""
Handoff §2.2 fill methods from segmentation, to put next to M3 (evaluate_seg.py), as calc_handoff.py does for
keypoints:
  M1  phi_R = 1 - d_int / d_ext        boxed formula, no sun/camera metadata
  M2  z_C - z_R = d_int GSD tan(theta) single shadow, exact only for a nadir view
  M3  forward model Gamma(z)            from per_image_<split>.csv (evaluate_seg.py)

d_ext needs the whole ground shadow, which often leaves the 1.5 x diameter training crop. The model is
therefore run on a 5 x 5 mosaic of tiles at the training scale (same crop size and scale as in training)
centred on the tank, which contains the shadow tip for 99.4 % of the 500 tanks; M1 abstains if the fitted
shadow tip is outside the mosaic. d_int comes from the centre tile (the training crop itself).

    python segmentation/evaluate_handoff.py --run runs/seg/mobilenet_unet_none_s0 \
        --data data/segmentation_data --src data/SynTank-25k
"""
import argparse
import csv
import json
import multiprocessing as mp
import os

import cv2
import numpy as np
import torch

from fill_from_mask import fill_m1_m2, fit_ground_displacement, fit_inner_displacement
from prepare_synthetic_seg import GROUND_SHADOW_PART, INNER_SHADOW_PART, crop
from seg_common import build_model, encode_input
from evaluate_seg import boot_ci

TILES = 5


def class_map(msk_bgr):
    part = msk_bgr[..., 1]
    c = np.zeros(part.shape, np.uint8)
    c[part == GROUND_SHADOW_PART] = 1
    c[part == INNER_SHADOW_PART] = 2
    return c


def mosaic(meta, img, cls, model, device):
    """Predicted class / centre-tile inner-shadow probability, true class map, validity and native coords."""
    c = meta["crop"]
    side, n = c["side_native_px"], c["size"]
    cx, cy = meta["rim_center_native"]
    h = TILES // 2
    P = np.zeros((TILES * n, TILES * n), np.uint8)
    G = np.zeros_like(P)
    V = np.zeros(P.shape, bool)
    xs, centre_prob = [], None
    tiles = [(i, j) for j in range(-h, h + 1) for i in range(-h, h + 1)]
    for i, j in tiles:
        tx, ty = cx + i * side, cy + j * side
        interp = cv2.INTER_AREA if side > n else cv2.INTER_LINEAR
        ti, _, _, valid = crop(img, tx, ty, side, n, interp, (0, 0, 0))
        tg, _, _, _ = crop(cls, tx, ty, side, n, cv2.INTER_NEAREST, 0)
        sl = (slice((j + h) * n, (j + h + 1) * n), slice((i + h) * n, (i + h + 1) * n))
        G[sl], V[sl] = np.where(valid, tg, 0), valid
        xs.append(encode_input(cv2.cvtColor(ti, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0))
    with torch.no_grad():
        logits = torch.cat([model(torch.stack(xs[k:k + 5]).to(device)) for k in range(0, len(xs), 5)])
    pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
    prob = logits.softmax(1)[:, 2].cpu().numpy()
    for t, (i, j) in enumerate(tiles):
        sl = (slice((j + h) * n, (j + h + 1) * n), slice((i + h) * n, (i + h + 1) * n))
        P[sl] = np.where(V[sl], pred[t], 0)
        if i == 0 and j == 0:
            centre_prob = np.where(V[sl], prob[t], 0.0)
    u = (np.arange(TILES * n) + 0.5) / c["k"]
    q = np.stack(np.meshgrid(u + cx - TILES * side / 2, u + cy - TILES * side / 2), -1)
    centre = (slice(h * n, (h + 1) * n), slice(h * n, (h + 1) * n))
    return P, centre_prob, G, G[centre], V, q


NPROC = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def _init_worker():
    cv2.setNumThreads(1)


def _pool():
    """Worker pool for the CPU-side fits. 'spawn' (fresh processes): forking a process that already ran
    PyTorch can deadlock the children on locks held by its thread pools (seen with 4 workers stuck at 0 %
    CPU). One BLAS/OpenMP thread per worker, so NPROC workers use NPROC cores."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"                       # inherited by the spawned workers before they import numpy
    return mp.get_context("spawn").Pool(NPROC, initializer=_init_worker)


def mosaic_coords(meta):
    """Native-pixel coordinates (H, W, 2) of the mosaic pixels (recomputed in workers instead of stored)."""
    c = meta["crop"]
    n, side = c["size"], c["side_native_px"]
    cx, cy = meta["rim_center_native"]
    u = (np.arange(TILES * n) + 0.5) / c["k"]
    return np.stack(np.meshgrid(u + cx - TILES * side / 2, u + cy - TILES * side / 2), -1)


def true_mosaic(meta, cls):
    """True class mosaic, its centre tile and the validity mask."""
    h, n = TILES // 2, meta["crop"]["size"]
    side, (cx, cy) = meta["crop"]["side_native_px"], meta["rim_center_native"]
    G = np.zeros((TILES * n, TILES * n), np.uint8)
    V = np.zeros(G.shape, bool)
    for j in range(-h, h + 1):
        for i in range(-h, h + 1):
            tg, _, _, valid = crop(cls, cx + i * side, cy + j * side, side, n, cv2.INTER_NEAREST, 0)
            sl = (slice((j + h) * n, (j + h + 1) * n), slice((i + h) * n, (i + h + 1) * n))
            G[sl], V[sl] = np.where(valid, tg, 0), valid
    return G, G[h * n:(h + 1) * n, h * n:(h + 1) * n], V


def _tank_row(job, dev=None):
    """M1/M2 from the predicted and the true mosaic of one tank (in a CPU worker, or on the GPU `dev`)."""
    name, split, meta, P, prob, G, Gc, V = job
    q = mosaic_coords(meta)
    pred = methods(meta, prob, P, V, q, dev) if P is not None else None
    orc = methods(meta, Gc, G, V, q, dev)
    r = {"image": name, "split": split, "true_phi_R": meta["phi_R"],
         "inner_shadow_native_px": meta["inner_shadow_native_px"],
         "sun_elevation_deg": meta["sun_elevation_deg"], "camera_elevation_deg": meta["camera_elevation_deg"]}
    for tag, res in (("seg", pred), ("oracle", orc)):
        if res is None:
            continue
        for mth in ("M1", "M2"):
            v = res.get(f"{mth}_phi_R")
            r[f"{tag}_{mth}_phi_R"] = v if v is not None else ""
            r[f"{tag}_{mth}_err"] = v - meta["phi_R"] if v is not None else ""
        r[f"{tag}_d_int_px"] = res.get("d_int_px", "")
        r[f"{tag}_d_ext_px"] = res.get("d_ext_px", "")
        r[f"{tag}_M1_reason"] = res["M1_reason"]
    return r


def methods(meta, inner_target, cls_mosaic, valid, q, dev=None):
    if dev is not None and dev.type == "cuda":
        from fill_gpu import fit_ground_displacement_gpu, fit_inner_displacement_gpu
        d3, why = fit_ground_displacement_gpu(cls_mosaic, q, valid, meta, dev)
        t1 = fit_inner_displacement_gpu(inner_target, meta, dev, direction_px=d3)
    else:
        d3, why = fit_ground_displacement(cls_mosaic, q, valid, meta)
        t1 = fit_inner_displacement(inner_target, meta, direction_px=d3)     # along d3 when it exists
    out = fill_m1_m2(t1, d3, meta)
    out["M1_reason"] = "ok" if "M1_phi_R" in out else (why if t1 is not None else "inner shadow not measurable")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--src", required=True, help="synthetic renders (images/, masks/)")
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--oracle-only", action="store_true", help="true masks only (no model): method floor")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if not a.oracle_only:
        ck = torch.load(os.path.join(a.run, a.weights), map_location=device, weights_only=False)
        model = build_model(ck["config"].get("model", "mobilenet_unet")).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
    os.makedirs(os.path.join(a.run, "pred_mosaic"), exist_ok=True)
    summary = {}
    for split in a.splits.split(","):
        names = [l.strip() for l in open(os.path.join(a.data, f"{split}.txt")) if l.strip()]
        m3 = {}
        p3 = os.path.join(a.run, f"per_image_{split}.csv")
        if os.path.exists(p3):
            m3 = {r["image"]: r for r in csv.DictReader(open(p3))}
        jobs = []                                   # stage 1: mosaics (model on the GPU), one tank at a time
        for name in names:
            meta = json.load(open(os.path.join(a.data, "meta", name + ".json")))
            cls = class_map(cv2.imread(os.path.join(a.src, "masks", meta["folder"], name + ".png")))
            if a.oracle_only:
                G, Gc, V = true_mosaic(meta, cls)
                P = prob = None
            else:
                img = cv2.imread(os.path.join(a.src, "images", meta["folder"], name + ".png"))
                P, prob, G, Gc, V, _ = mosaic(meta, img, cls, model, device)
                cv2.imwrite(os.path.join(a.run, "pred_mosaic", name + ".png"), P)
            jobs.append((name, split, meta, P, prob, G, Gc, V))
        # stage 2: the displacement fits of every tank: on the run's GPU when there is one (fill_gpu.py, verified
        # identical to the CPU fits by check_fill_gpu.py, ~57x faster), else in parallel on the run's CPUs
        if device.type == "cuda":
            rows = [_tank_row(job, device) for job in jobs]
        else:
            with _pool() as pool:
                rows = pool.map(_tank_row, jobs, chunksize=2)
        for r in rows:
            k3 = m3.get(r["image"])
            if k3:
                ab = k3["seg_M3_abstain"] == "True"
                r["seg_M3_err"] = "" if ab else float(k3["seg_M3_err"])
                r["oracle_M3_err"] = "" if k3["oracle_abstain"] == "True" else float(k3["oracle_err"])
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("image", "split"), k))
        out_csv = os.path.join(a.run, f"per_image_handoff_{split}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        s = {"n": len(rows)}
        for tag in ("seg", "oracle"):
            for mth in ("M1", "M2", "M3"):
                e = [abs(r[f"{tag}_{mth}_err"]) for r in rows if r.get(f"{tag}_{mth}_err", "") != ""]
                if not e and f"{tag}_{mth}_err" not in rows[0]:
                    continue
                s[f"{tag}_{mth}"] = {"n_scored": len(e), "coverage": len(e) / len(rows), "MAE": boot_ci(e),
                                     "median": float(np.median(e)) if e else None,
                                     "fail_rate_0.02": float(np.mean(np.array(e) > 0.02)) if e else None}
            reasons = {}
            for r in rows:
                if f"{tag}_M1_reason" in r:
                    reasons[r[f"{tag}_M1_reason"]] = reasons.get(r[f"{tag}_M1_reason"], 0) + 1
            if reasons:
                s[f"{tag}_M1_reasons"] = reasons
        summary[split] = s
        print(split, json.dumps(s, default=float))
    json.dump(summary, open(os.path.join(a.run, "summary_handoff.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
