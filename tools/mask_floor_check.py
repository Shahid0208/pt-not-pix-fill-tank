"""
True-mask M3 floor at a different crop resolution (no model involved).

The paper's true-mask reference (0.0135 test fill MAE) uses 128 px crops spanning 1.5 rim diameters (~85 px per
diameter). The keypoint model sees 256 px over four radii (~128 px per diameter). This script re-runs the same
mask-to-fill conversion (segmentation/fill_from_mask.py) on true masks re-rasterised at another crop, e.g. 256 px
over 2 diameters, to see where the floor lands. No model is involved.

Step 1 (crops + meta, test split only), using the existing prep script with a split dir holding only test.txt:
    python segmentation/prepare_synthetic_seg.py --src data/SynTank-25k --split-dir $SPLIT_ONLY_TEST \
        --out data/seg_floor_256 --ctx 2.0 --size 256
Step 2 (this script):
    python tools/mask_floor_check.py --data data/seg_floor_256 --split test --out runs/mask_floor_256

Writes <out>/per_tank.csv (image, true_phi_R, oracle_phi_R, oracle_err, oracle_abstain, evidence_px,
inner_shadow_native_px) and <out>/summary.json (MAE over non-abstained tanks, by shadow-width band).
Compare with learned keypoints on matched tanks locally (tools/review_matched_cohorts.py --floor <out>).
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "segmentation"))
from fill_from_mask import fill_from_mask  # noqa: E402

BANDS = [(0, 6, "<6 px"), (6, 12, "6-12 px"), (12, 1e9, ">=12 px")]
IGNORE = 255


def one(job):
    data, name = job
    meta = json.load(open(os.path.join(data, "meta", name + ".json")))
    g = cv2.imread(os.path.join(data, "masks", name + ".png"), cv2.IMREAD_GRAYSCALE)
    g = np.where(g == IGNORE, 0, g).astype(np.uint8)
    fo = fill_from_mask(g, meta)
    return {"image": name, "true_phi_R": meta["phi_R"], "oracle_phi_R": fo["phi_R"],
            "oracle_err": fo["phi_R"] - meta["phi_R"], "oracle_abstain": fo["abstain"],
            "evidence_px": fo["evidence_px"], "inner_shadow_native_px": meta["inner_shadow_native_px"],
            "crop_size": meta["crop"]["size"], "crop_ctx": meta["crop"]["ctx"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="output of prepare_synthetic_seg.py")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count()))
    a = ap.parse_args()
    names = [l.strip() for l in open(os.path.join(a.data, f"{a.split}.txt")) if l.strip()]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(one, [(a.data, n) for n in names], chunksize=4)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "per_tank.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if not r["oracle_abstain"]]
    s = {"split": a.split, "n": len(rows), "n_scored": len(ok), "crop_size": rows[0]["crop_size"],
         "crop_ctx": rows[0]["crop_ctx"], "oracle_M3_MAE": float(np.mean([abs(r["oracle_err"]) for r in ok])),
         "bands": {}}
    for lo, hi, lab in BANDS:
        b = [abs(r["oracle_err"]) for r in ok if lo <= r["inner_shadow_native_px"] < hi]
        s["bands"][lab] = {"n": len(b), "MAE": float(np.mean(b)) if b else None}
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
