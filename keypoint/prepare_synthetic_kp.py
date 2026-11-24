"""Synthetic labels -> sun-aligned context crops + 65 keypoints + grouped 90/5/5 split.

    python keypoint/prepare_synthetic_kp.py --src synthetic_data --out keypoint_data/synthetic
    python keypoint/prepare_synthetic_kp.py --src synthetic_data --out /tmp/kp_smoke --limit 60   # smoke test

Writes crops/<image>.png (320x320, half-size 2.5 R), keypoints.json, train/val/test.txt, manifest.csv,
prep_report.json and overlays/ (keypoints drawn on 24 crops for a visual check).
Stops with an error if a split shares a scene group with another split, or if any exported boundary
point is further than CHECK_TOL_PX from the analytic curve the keypoints are built from.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import geometry_kp as G

CTX_FACTOR = 2.5
CTX_SIZE = 320
PAD_VALUE = 114
SPLITS = ("train", "val", "test")
FILL_BIN = 0.1   # stratum width for continuous (v2) fill levels


def load_entries(src: Path):
    out = []
    for meta_path in sorted(src.glob("images/*/metadata.json")):
        for fn, e in sorted(json.load(open(meta_path)).items()):
            loc = (e.get("location") or {}).get("preset") or ""
            sc = e["solar_conditions"]
            out.append({
                "image": Path(fn).stem, "folder": meta_path.parent.name,
                "img_path": src / e["image_metadata"]["relative_path"],
                "label_path": src / e["image_metadata"]["label_relative_path"],
                "fill": round(e["tank_geometry"]["fill_level_percentage"] / 100.0, 3),
                "group": f"{loc}|{sc['sun_elevation_degrees']}|{sc['sun_azimuth_degrees']}",
                "camera_elevation": e["satellite_sensor"]["camera_elevation_degrees"],
                "atmosphere": e["satellite_sensor"]["atmospheric_condition"],
                # v2 renders: held-out design tags (never train/val/test) and continuous fill; v1 has neither
                "holdout": e.get("holdout") or [],
                "continuous_fill": "holdout" in e,
            })
    return out


def split_groups(entries, fracs, seed):
    """Grouped (scene) and stratified (size folder, fill) split with exact global fractions:
    groups are visited stratum by stratum in shuffled order and each goes to the split furthest
    below its target share so far. v2 renders are stratified on fill bins of FILL_BIN (their fill is
    continuous). A group with a held-out image ("holdout" in its metadata) goes to "ood:<tags>" and is
    written to test_ood_<tag>.txt instead."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for e in entries:
        groups[e["group"]].append(e)
    split_of = {}
    for g, es in groups.items():
        tags = sorted({t for e in es for t in e["holdout"]})
        if tags:
            split_of[g] = "ood:" + ",".join(tags)
    strata = defaultdict(list)
    for g, es in groups.items():
        if g in split_of:
            continue
        fill = es[0]["fill"]
        key = min(int(fill / FILL_BIN + 1e-9), int(round(1 / FILL_BIN)) - 1) if es[0]["continuous_fill"] else fill
        strata[(es[0]["folder"], key)].append(g)
    counts = Counter()
    seen = 0
    for st in sorted(strata):
        gs = sorted(strata[st])
        rng.shuffle(gs)
        for g in gs:
            n = len(groups[g])
            deficit = {s: f * (seen + n) - counts[s] for s, f in zip(SPLITS, fracs)}
            s = max(SPLITS, key=lambda k: deficit[k])
            split_of[g] = s
            counts[s] += n
            seen += n
    return split_of


def make_crop(img, M):
    s = abs(np.linalg.det(M[:, :2])) ** 0.5
    if s < 1.0:  # anti-alias before shrinking
        sigma = 0.5 * (1.0 / s - 1.0) + 1e-3
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.warpAffine(img, M, (CTX_SIZE, CTX_SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(PAD_VALUE,) * 3)


def draw(crop, kp, exists):
    vis = crop.copy()
    colours = {"center": (255, 255, 255), "rim": (0, 255, 255), "inner_shadow": (0, 0, 255),
               "ground_shadow": (255, 0, 0), "roof_edge": (0, 255, 0)}
    for i, (p, ok) in enumerate(zip(kp, exists)):
        if ok:
            cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 2, colours[G.group_of(i)], -1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("synthetic_data"))
    ap.add_argument("--out", type=Path, default=Path("keypoint_data/synthetic"))
    ap.add_argument("--split", type=float, nargs=3, default=[0.9, 0.05, 0.05])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="use only N images (smoke test)")
    ap.add_argument("--split-file", type=Path,
                    help="shared split JSON from synthetic_data/make_shared_split.py (the YOLO data uses the same "
                         "file): every render goes to the split it names; held-out tags are not used")
    args = ap.parse_args()
    assert abs(sum(args.split) - 1) < 1e-6

    entries = load_entries(args.src)
    if args.limit:
        entries = random.Random(0).sample(entries, args.limit)
    if args.split_file:
        named = json.loads(args.split_file.read_text())["split"]
        have = {e["img_path"].name for e in entries}
        if set(named) != have:
            raise SystemExit(f"split file and {args.src} differ: {len(set(named) - have)} renders only in the file, "
                             f"{len(have - set(named))} only in the data")
        split_of = {}
        for e in entries:
            s = named[e["img_path"].name]
            if split_of.setdefault(e["group"], s) != s:
                raise SystemExit(f"scene group {e['group']} spans splits in the split file")
    else:
        split_of = split_groups(entries, args.split, args.seed)

    (args.out / "crops").mkdir(parents=True, exist_ok=True)
    (args.out / "overlays").mkdir(exist_ok=True)
    records, worst = [], Counter()
    exist_count = Counter()
    for n, e in enumerate(entries):
        lab = json.load(open(e["label_path"]))
        pts, exists, phys, check = G.keypoints_from_label(lab)
        for g, v in check.items():
            worst[g] = max(worst[g], v)
            if v > G.CHECK_TOL_PX:
                raise SystemExit(f"{e['image']}: exported {g} boundary is {v:.3f} px from the analytic curve")
        img = cv2.imread(str(e["img_path"]), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        M = G.crop_matrix(pts[0], phys["R_px"], phys["shadow_dir_img"], CTX_FACTOR, CTX_SIZE)
        crop = make_crop(img, M)
        cv2.imwrite(str(args.out / "crops" / f"{e['image']}.png"), crop)
        kp_ctx = G.apply_affine(M, pts)
        in_image = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
        for g, s in G.GROUP_SLICE.items():
            exist_count[g] += int(exists[s].sum())
        if n < 24:
            cv2.imwrite(str(args.out / "overlays" / f"{e['image']}.png"), draw(crop, kp_ctx, exists & in_image))
        records.append({
            "image": e["image"], "split": split_of[e["group"]], "group": e["group"], "folder": e["folder"],
            "fill": e["fill"], "camera_elevation": e["camera_elevation"], "atmosphere": e["atmosphere"],
            "image_size": [W, H], "M_ctx": M.round(8).tolist(),
            "kp_native": pts.round(4).tolist(), "kp_ctx": kp_ctx.round(4).tolist(),
            "exists": exists.astype(int).tolist(), "in_image": in_image.astype(int).tolist(),
            "physics": phys,
        })
        if (n + 1) % 100 == 0:
            print(f"{n + 1}/{len(entries)}", flush=True)

    # Splits must not share a scene group (val and test stay separate from each other and from train).
    gsets = {s: {r["group"] for r in records if r["split"] == s} for s in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b and gsets[a] & gsets[b]:
                raise SystemExit(f"split leak: {a} and {b} share groups {sorted(gsets[a] & gsets[b])[:3]}")

    json.dump(records, open(args.out / "keypoints.json", "w"))
    for s in SPLITS:
        (args.out / f"{s}.txt").write_text("".join(r["image"] + "\n" for r in records if r["split"] == s))
    ood_tags = sorted({t for r in records if r["split"].startswith("ood:") for t in r["split"][4:].split(",")})
    for t in ood_tags:
        (args.out / f"test_ood_{t}.txt").write_text("".join(
            r["image"] + "\n" for r in records if r["split"].startswith("ood:") and t in r["split"][4:].split(",")))
    with open(args.out / "manifest.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "split", "group", "folder", "fill", "sun_elevation", "camera_elevation",
                     "atmosphere", "R_px", "delta_1_px", "phi_R"])
        for r in records:
            p = r["physics"]
            wr.writerow([r["image"], r["split"], r["group"], r["folder"], r["fill"], p["sun_elevation_deg"],
                         r["camera_elevation"], r["atmosphere"], round(p["R_px"], 2),
                         round(p["delta_1_mag_px"], 3), round(p["phi_R"], 4)])
    report = {
        "n_images": len(records),
        "split_images": {s: sum(r["split"] == s for r in records) for s in SPLITS},
        "split_groups": {s: len(gsets[s]) for s in SPLITS},
        "test_ood_images": {t: sum(r["split"].startswith("ood:") and t in r["split"][4:].split(",") for r in records)
                            for t in ood_tags},
        "boundary_check_max_px": dict(worst),
        "existing_keypoints": dict(exist_count),
        "ctx_factor": CTX_FACTOR, "ctx_size": CTX_SIZE,
    }
    json.dump(report, open(args.out / "prep_report.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
