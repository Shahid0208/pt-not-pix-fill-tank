"""
Score a trained segmentation run on val and test: segmentation quality and fill from the predicted mask.

    python segmentation/evaluate_seg.py --run runs/seg/seg_mnv3_s0 --data data/segmentation_data

Per image (per_image_<split>.csv):
  - IoU per class (nan when the class is in neither mask) and boundary F1 (1 px tolerance) of both shadows
  - fill phi_R (ground-referenced, as the keypoint M3) with fill_from_mask.py from the predicted inner-shadow
    probability ("seg_M3", sub-pixel), from the true mask ("oracle": the method's own floor), and the original
    area-ratio heuristic ("area_ratio"). seg_M3 / oracle abstain below 10 px of inner shadow; MAEs are over the
    tanks that were not abstained (the count is reported, and the MAE including them as well)
Summary (summary.json): dataset-level IoU, per-image means with bootstrap 95 % CIs, fill MAE per method and per
inner-shadow length band.
"""
import argparse
import csv
import json
import multiprocessing as mp
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from fill_from_mask import area_ratio_fill, fill_from_mask
from seg_common import CLASS_NAMES, IGNORE, SegDataset, boundary_f1, build_model, confusion, iou_from_confusion

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


def _image_row(job):
    """Metrics and fills of one image (run in a worker process). Returns (row, 3x3 confusion)."""
    meta_path, name, split, pred_dir, p, pi, g = job
    meta = json.load(open(meta_path))
    g = g.astype(np.int64)
    valid = g != IGNORE
    p = np.where(valid, p, 0).astype(np.uint8)
    cv2.imwrite(os.path.join(pred_dir, name + ".png"), p)
    cm = confusion(p.ravel().astype(np.int64), g.ravel())
    iou = iou_from_confusion(cm)
    g0 = np.where(valid, g, 0)
    fs = fill_from_mask(np.where(valid, pi, 0.0).astype(np.float32), meta)   # soft: sub-pixel
    fo = fill_from_mask(g0, meta)
    ar = area_ratio_fill(p, meta["sun_elevation_deg"])
    row = {
        "image": name, "split": split, "folder": meta["folder"], "true_phi_R": meta["phi_R"],
        "seg_M3_phi_R": fs["phi_R"], "oracle_phi_R": fo["phi_R"], "area_ratio_phi_R": ar,
        "seg_M3_err": fs["phi_R"] - meta["phi_R"], "oracle_err": fo["phi_R"] - meta["phi_R"],
        "area_ratio_err": ar - meta["phi_R"], "seg_M3_mismatch": fs["mismatch"],
        "seg_M3_abstain": fs["abstain"], "oracle_abstain": fo["abstain"],
        "seg_M3_evidence_px": fs["evidence_px"],
        "pred_inner_px": int((p == 2).sum()), "true_inner_px": int((g == 2).sum()),
        **{f"iou_{c}": float(v) for c, v in zip(CLASS_NAMES, iou)},
        "bf1_ground_shadow": boundary_f1(p == 1, g0 == 1, valid),
        "bf1_inner_shadow": boundary_f1(p == 2, g0 == 2, valid),
        "inner_shadow_native_px": meta["inner_shadow_native_px"],
        "inner_shadow_crop_px": meta["inner_shadow_crop_px"],
        "sun_elevation_deg": meta["sun_elevation_deg"],
        "camera_elevation_deg": meta["camera_elevation_deg"],
        "naip_scale": meta["naip_scale"], "atmosphere": meta["atmosphere"],
        "rim_diam_native_px": meta["rim_diam_native_px"], "roof_tilt_deg": meta["roof_tilt_deg"],
        "holdout": "|".join(meta["holdout"]) or "none"}
    return row, cm
BANDS = [(0, 6, "<6 px"), (6, 12, "6-12 px"), (12, 1e9, ">=12 px")]   # inner-shadow width, native px


def boot_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) == 0:
        return [float("nan")] * 3
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--splits", default="val,test")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(a.run, a.weights), map_location=device, weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg.get("model", "mobilenet_unet")).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    summary = {"run": a.run, "model": cfg["model"], "seed": cfg["seed"],
               "weights": a.weights, "best_epoch": ck["epoch"], "n_params": cfg["n_params"]}

    for split in a.splits.split(","):
        ds = SegDataset(a.data, split, augment=False)
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=min(NPROC, 8))
        pred_dir = os.path.join(a.run, "pred", split)
        os.makedirs(pred_dir, exist_ok=True)
        jobs = []                                   # stage 1: all predictions on the GPU
        with torch.no_grad():
            for x, y, idx in dl:
                logits = model(x.to(device))
                pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
                prob = logits.softmax(1)[:, 2].cpu().numpy().astype(np.float32)
                for p, pi, g, i in zip(pred, prob, y.numpy().astype(np.uint8), idx.numpy()):
                    jobs.append((os.path.join(a.data, "meta", ds.names[int(i)] + ".json"), ds.names[int(i)], split,
                                 pred_dir, p, pi, g))
        # stage 2: metrics and fill fits for every image in parallel on the run's CPUs (same order as before)
        with _pool() as pool:
            out = pool.map(_image_row, jobs, chunksize=8)
        rows = [r for r, _ in out]
        cm_tot = np.sum([c for _, c in out], axis=0)
        with open(os.path.join(a.run, f"per_image_{split}.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)

        s = {"n": len(rows), "dataset_iou": dict(zip(CLASS_NAMES, iou_from_confusion(cm_tot).round(5).tolist()))}
        for k in [f"iou_{c}" for c in CLASS_NAMES] + ["bf1_ground_shadow", "bf1_inner_shadow"]:
            s[k] = boot_ci([r[k] for r in rows])
        for m in ("seg_M3", "oracle", "area_ratio"):
            ab = [r.get(f"{m}_abstain", False) for r in rows]          # area_ratio never abstains
            e = [abs(r[f"{m}_err"]) for r, b in zip(rows, ab) if not b]
            s[f"{m}_n_abstain"] = int(sum(ab))
            s[f"{m}_fill_MAE"] = boot_ci(e)
            s[f"{m}_fill_median_abs"] = float(np.median(e)) if e else float("nan")
            s[f"{m}_fail_rate_0.02"] = float(np.mean(np.array(e) > 0.02)) if e else float("nan")
            s[f"{m}_fill_MAE_incl_abstain"] = float(np.mean([abs(r[f"{m}_err"]) for r in rows]))
        s["by_shadow_band"] = {lab: {"n": len(sub), **{f"{m}_MAE": float(np.mean([abs(r[f'{m}_err']) for r in sub
                                                                                  if not r[f'{m}_abstain']] or [np.nan]))
                                                          for m in ("seg_M3", "oracle")},
                                     "iou_inner_shadow": float(np.nanmean([r["iou_inner_shadow"] for r in sub]))}
                               for lo, hi, lab in BANDS
                               for sub in [[r for r in rows if lo <= r["inner_shadow_native_px"] < hi]] if sub}
        summary[split] = s
        print(f"{split}: n={len(rows)} dataset IoU {s['dataset_iou']} | inner-shadow BF1 {s['bf1_inner_shadow'][0]:.3f}"
              f" | fill MAE seg_M3 {s['seg_M3_fill_MAE'][0]:.4f} oracle {s['oracle_fill_MAE'][0]:.4f} "
              f"area_ratio {s['area_ratio_fill_MAE'][0]:.4f}")

    json.dump(summary, open(os.path.join(a.run, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
