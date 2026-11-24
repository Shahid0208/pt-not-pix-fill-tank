"""
Leakage audit of a synthetic render set and the keypoint + segmentation data built from it (read-only).

    python synthetic_data/audit_syn_split.py --merged <render set> --split-file <shared_split.json> \
        --kp <keypoint data> --seg <segmentation data> --out <audit.json>

Exits 1 on any failure:
  1. split file: every render of the set in exactly one split, nothing else; no scene (location + date + local
     solar time) and no keypoint scene group (location | sun elevation | sun azimuth) in two splits. Both are
     re-derived from the renders' metadata (make_shared_split.load), not read from the split file's report;
  2. keypoint data: exactly the split file's renders, each record and each train/val/test.txt with its split;
  3. segmentation data: train/val/test.txt identical to the keypoint lists, and every crop's meta split matches;
  4. no byte-identical render in two splits (MD5 of every image);
  5. no near-duplicate crops across splits: 16x16 grey thumbnails (keypoint crops) with correlation > 0.99,
     confirmed at 128x128 by mean absolute grey difference < 2 .
     Only cross-split pairs are compared (val-train, test-train, test-val), in blocks.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_shared_split import SPLITS, load  # noqa: E402

CORR, MAX_DIFF, BLOCK = 0.99, 2.0, 4096


def stem(n):
    return n[:-4] if n.endswith(".png") else n        # metadata / split file keys carry .png, data lists do not


def lists(folder):
    return {s: [l.strip() for l in open(os.path.join(folder, f"{s}.txt")) if l.strip()] for s in SPLITS}


def thumbs(crops, names, size):
    out = np.zeros((len(names), size * size), np.float32)
    for i, n in enumerate(names):
        im = cv2.imread(os.path.join(crops, n + ".png"), cv2.IMREAD_GRAYSCALE)
        v = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        if size == 16:
            v -= v.mean()
            v /= max(float(np.linalg.norm(v)), 1e-6)
        out[i] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--kp", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    issues, rep = [], {}

    # 1. split file vs the renders' own metadata
    renders = {stem(n): v for n, v in load(a.merged).items()}
    split_of = {stem(n): s for n, s in json.load(open(a.split_file))["split"].items()}
    if set(split_of) != set(renders):
        issues.append(f"split file covers {len(set(split_of) & set(renders))} of {len(renders)} renders, "
                      f"{len(set(split_of) - set(renders))} unknown names")
    for key in ("scene", "group"):
        m = defaultdict(set)
        for n, s in split_of.items():
            if n in renders:
                m[renders[n][key]].add(s)
        bad = [k for k, v in m.items() if len(v) > 1]
        rep[f"{key}s"] = len(m)
        if bad:
            issues.append(f"{len(bad)} {key}s span splits, e.g. {bad[0]}")
    rep["split_sizes"] = dict(Counter(split_of.values()))

    # 2. keypoint data
    recs = json.load(open(os.path.join(a.kp, "keypoints.json")))
    kp_split = {r["image"]: r["split"] for r in recs}
    if set(kp_split) != set(split_of):
        issues.append(f"keypoint data has {len(kp_split)} records vs {len(split_of)} in the split file")
    wrong = [n for n, s in kp_split.items() if split_of.get(n) != s]
    if wrong:
        issues.append(f"{len(wrong)} keypoint records carry another split than the file, e.g. {wrong[0]}")
    kp_lists = lists(a.kp)
    for s in SPLITS:
        if set(kp_lists[s]) != {n for n, x in split_of.items() if x == s}:
            issues.append(f"keypoint {s}.txt differs from the split file")
    if sum(len(v) for v in kp_lists.values()) != len(set().union(*map(set, kp_lists.values()))):
        issues.append("a render is in two keypoint split lists")

    # 3. segmentation data
    seg_lists = lists(a.seg)
    for s in SPLITS:
        if seg_lists[s] != kp_lists[s] and set(seg_lists[s]) != set(kp_lists[s]):
            issues.append(f"segmentation {s}.txt differs from the keypoint {s}.txt")
    bad_meta = 0
    for s in SPLITS:
        for n in seg_lists[s]:
            if json.load(open(os.path.join(a.seg, "meta", n + ".json")))["split"] != s:
                bad_meta += 1
    if bad_meta:
        issues.append(f"{bad_meta} segmentation crops carry another split in their meta")

    # 4. byte-identical renders across splits
    by_hash = defaultdict(set)
    for n, v in renders.items():
        p = os.path.join(a.merged, "images", v["folder"], n + ".png")
        by_hash[hashlib.md5(open(p, "rb").read()).hexdigest()].add(split_of.get(n))
    dup = sum(1 for v in by_hash.values() if len(v) > 1)
    rep["identical_renders_across_splits"] = dup
    if dup:
        issues.append(f"{dup} byte-identical renders in two splits")

    # 5. near-duplicate crops across splits
    crops = os.path.join(a.kp, "crops")
    names = {s: kp_lists[s] for s in SPLITS}
    T = {s: thumbs(crops, names[s], 16) for s in SPLITS}
    found, nn = [], {}
    for sa, sb in (("val", "train"), ("test", "train"), ("test", "val")):
        best = np.full(len(names[sa]), -1.0, np.float32)
        for i0 in range(0, len(names[sa]), BLOCK):
            for j0 in range(0, len(names[sb]), BLOCK):
                S = T[sa][i0:i0 + BLOCK] @ T[sb][j0:j0 + BLOCK].T
                best[i0:i0 + BLOCK] = np.maximum(best[i0:i0 + BLOCK], S.max(1))
                for i, j in zip(*np.nonzero(S > CORR)):
                    na, nb = names[sa][i0 + i], names[sb][j0 + j]
                    ga, gb = thumbs(crops, [na], 128)[0], thumbs(crops, [nb], 128)[0]
                    if np.abs(ga - gb).mean() < MAX_DIFF:
                        found.append((sa, na, sb, nb))
        nn[f"{sa}->{sb}"] = {"median_best_corr": float(np.median(best)), "max_best_corr": float(best.max())}
    rep["near_duplicate_pairs_across_splits"] = len(found)
    rep["nearest_cross_split_similarity"] = nn
    if found:
        issues.append(f"{len(found)} near-duplicate crop pairs across splits, e.g. {found[0]}")
        rep["near_duplicates"] = found[:50]

    rep["issues"] = issues
    rep["result"] = "PASS: no leakage" if not issues else "FAIL"
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "near_duplicates"}, indent=1))
    if issues:
        sys.exit("LEAKAGE AUDIT FAILED: " + "; ".join(issues))
    print("no leakage found")


if __name__ == "__main__":
    main()
