"""Independent, read-only leakage audit of a YOLO dataset root (real/ + synthetic/ split lists).

    python yolo_obb/audit_leakage.py --root data/yolo > audit.json

Checks: list overlaps between splits; train_os.txt only holds train chips; which lists the data YAMLs use;
byte-identical images across splits (md5); visual near-duplicates (16x16 grey thumbnails, normalised
correlation > 0.98) of every val/test image against all train images, with the train-train baseline. Each
candidate is then confirmed at 128x128 on the pixels that are not NAIP no-data (black) in either image: mean
absolute grey difference < 2 = the same picture. (Tile-edge chips that are mostly black correlate highly by
their shared black area alone, so the thumbnail test by itself gives false alarms; the same holds for synthetic
tiles that are mostly the grey padding canvas, which is excluded in the same way.)
Exits 1 if any val/test image is a byte copy or a confirmed near-duplicate of a train image.
"""
import argparse, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2, yaml
ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=Path("data/yolo"))
R = ap.parse_args().root; SPL = ("train", "val", "test")
def lst(d, s): return [l.strip() for l in open(R/d/f"{s}.txt") if l.strip()]
out = {}
# 1. lists: overlap, oversampled list, yaml wiring
for d in ("real", "synthetic"):
    L = {s: set(lst(d, s)) for s in SPL}
    out[f"{d}_list_overlap"] = {f"{a}-{b}": len(L[a] & L[b]) for a, b in (("train","val"),("train","test"),("val","test"))}
os_ = set(lst("real", "train_os"))
out["train_os_subset_of_train"] = os_ <= set(lst("real", "train"))
out["train_os_in_val_or_test"] = len(os_ & (set(lst("real","val")) | set(lst("real","test"))))
for y in ("data_real.yaml", "data_mixed.yaml", "data_mixed_os.yaml"):
    c = yaml.safe_load(open(R/y)); out[y] = {"train": c["train"], "val": c["val"], "test": c["test"]}
# 2. identical files across splits (md5 of the real bytes behind the links)
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
for d in ("real", "synthetic"):
    h = defaultdict(set)
    for s in SPL:
        for e in lst(d, s): h[md5((R/d/e[2:]).resolve())].add(s)
    out[f"{d}_identical_files_across_splits"] = sum(len(v) > 1 for v in h.values())
# 3. visual near-duplicates: 16x16 grey, normalised correlation, test/val vs train
# Synthetic tiles of zoomed renders are mostly the flat grey canvas prepare_synthetic.py pads them onto
# (PAD_VALUE 114); two different small tanks on that canvas correlate > 0.98 and differ by < 2 grey levels on
# the shared canvas alone (25k run: 10 false "duplicates", checked by eye). Pixels that are canvas in both images
# are therefore not compared, as the black NAIP no-data of real chips is not.
SYN_PAD = 114
def feats(d, s):
    F, names = [], []
    for e in lst(d, s):
        im = cv2.imread(str(R/d/e[2:]), cv2.IMREAD_GRAYSCALE)
        v = cv2.resize(im, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        v -= v.mean(); n = np.linalg.norm(v)
        F.append(v / n if n > 1e-3 else v); names.append(Path(e).stem)
    return np.stack(F), names
for d in ("real", "synthetic"):
    Ft, nt = feats(d, "train")
    tr_tr = Ft[:2000] @ Ft.T
    for i in range(min(2000, len(Ft))): tr_tr[i, i] = -1
    res = {"train_train_nn_median": float(np.median(tr_tr.max(1)))}
    def g128(name, split):
        e = next(x for x in lst(d, split) if Path(x).stem == name)
        im = cv2.imread(str(R/d/e[2:]), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    for s in ("val", "test"):
        Fs, ns = feats(d, s); S = Fs @ Ft.T
        confirmed, cand = [], 0
        for i, jj in zip(*np.nonzero(S > 0.98)):
            cand += 1
            A, B = g128(ns[i], s), g128(nt[jj], "train")
            valid = (A > 8) & (B > 8)                   # skip NAIP no-data (black)
            if d == "synthetic":                        # and the grey canvas around small rescaled renders
                valid &= ~((np.abs(A - SYN_PAD) < 0.5) & (np.abs(B - SYN_PAD) < 0.5))
            if valid.mean() > 0.02 and np.abs(A - B)[valid].mean() < 2:
                confirmed.append((ns[i], nt[jj], round(float(S[i, jj]), 4)))
        nn = S.max(1); j = S.argmax(1); top = np.argsort(-nn)[:5]
        res[s] = {"nn_median": float(np.median(nn)), "n_over_0.98": cand, "n_confirmed": len(confirmed),
                  "confirmed": confirmed[:20], "top": [(ns[i], nt[j[i]], round(float(nn[i]), 4)) for i in top]}
    out[f"{d}_near_duplicates"] = res
print(json.dumps(out, indent=1))
bad = out["real_identical_files_across_splits"] + out["synthetic_identical_files_across_splits"] + sum(
    out[f"{d}_near_duplicates"][s]["n_confirmed"] for d in ("real", "synthetic") for s in ("val", "test"))
sys.exit(1 if bad else 0)
