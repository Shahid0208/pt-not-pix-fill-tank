"""Convert the Blender synthetic set to YOLO-OBB format.

Picks a stratified fraction of the synthetic images (default 20%, equal share per size folder),
rescales each to the real data's ground resolution (NAIP, 0.6 m/px), and pads or tiles it to
512x512 so tanks appear at the same pixel size as in the real chips. OBBs come from the
instance masks (red channel = tank instance id, green 1-5 = tank parts), not from the JSON.

    python yolo_obb/prepare_synthetic.py --src synthetic_data --out yolo_data/synthetic

Held-out synthetic test set: every render NOT used for training (the training tiles' manifest.csv
lists the renders they came from), converted the same way, with a test.txt list:

    python yolo_obb/prepare_synthetic.py --src synthetic_data --out yolo_data/synthetic_test \
        --exclude-manifest yolo_data/synthetic/manifest.csv --list-name test.txt

Train/val/test split of ALL renders (equal share per size folder; every tile of a render stays in
one split), writing train.txt, val.txt and test.txt:

    python yolo_obb/prepare_synthetic.py --src synthetic_data --out yolo_data/synthetic --split 0.9 0.05 0.05

Split given by a shared split file (synthetic_data/make_shared_split.py; the keypoint data uses the same
file, so every render is in the same split for both models). Converts every render listed there:

    python yolo_obb/prepare_synthetic.py --src data/SynTank-25k \
        --out data/yolo_synthetic --split-file data/SynTank-25k/shared_split.json
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from common import CLASS_ID, SYNTHETIC_KIND_TO_CLASS, format_obb_line, write_label, write_split_list

PAD_VALUE = 114          # same grey Ultralytics uses for letterbox / mosaic padding
TANK_PARTS = (1, 5)      # mask green: 1 shell outside .. 5 ladder; 6/7 are shadows


def holdout_tags(folder: Path):
    """filename -> held-out design tags ("shape", "sun", "site") from the folder's metadata.json.
    v2 renders only; v1 renders have no tags, so their selection and splits are unchanged."""
    meta = folder / "metadata.json"
    if not meta.exists():
        return {}
    return {name: e["holdout"] for name, e in json.loads(meta.read_text()).items() if e.get("holdout")}


def select_images(src: Path, fraction: float, seed: int):
    """Equal fraction from every size folder, reproducible. Held-out renders are never selected."""
    rng = np.random.default_rng(seed)
    chosen = []
    for folder in sorted(p for p in (src / "images").iterdir() if p.is_dir()):
        held = holdout_tags(folder)
        files = sorted(f for f in folder.glob("*.png") if f.name not in held)
        n = round(len(files) * fraction)
        idx = rng.choice(len(files), size=n, replace=False)
        chosen += [files[i] for i in sorted(idx)]
    return chosen


def split_images(src: Path, fractions, seed: int):
    """Render -> split ("train"/"val"/"test"), stratified by size folder, reproducible. Held-out renders
    (metadata "holdout") map to "ood:<tags>" and are written to test_ood_<tag>.txt, never to train/val/test."""
    rng = np.random.default_rng(seed)
    mapping = {}
    for folder in sorted(p for p in (src / "images").iterdir() if p.is_dir()):
        held = holdout_tags(folder)
        for f in sorted(folder.glob("*.png")):
            if f.name in held:
                mapping[f] = "ood:" + ",".join(held[f.name])
        files = sorted(f for f in folder.glob("*.png") if f.name not in held)
        order = rng.permutation(len(files))
        n_val, n_test = round(len(files) * fractions[1]), round(len(files) * fractions[2])
        for rank, i in enumerate(order):
            mapping[files[i]] = "val" if rank < n_val else "test" if rank < n_val + n_test else "train"
    return mapping


def unused_images(src: Path, manifest: Path):
    """All renders not listed in a previous conversion's manifest.csv (column `source`)."""
    with open(manifest, newline="") as f:
        used = {row["source"] for row in csv.DictReader(f)}
    images = sorted((src / "images").glob("*/*.png"))
    return [p for p in images if str(p.relative_to(src / "images")) not in used], used


def instance_classes(label: dict):
    classes = {1: SYNTHETIC_KIND_TO_CLASS["floating"]}
    for t in label["scene"]["neighbour_tanks"]:
        classes[int(t["instance_id"])] = SYNTHETIC_KIND_TO_CLASS.get(t["kind"])
    return classes


def instance_obb(inst_mask: np.ndarray, circle_tol: float):
    """Oriented box of one instance. Near-circular footprints get an axis-aligned box: the
    angle of a circle's box is arbitrary, and the real labels are all axis-aligned."""
    contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = np.concatenate(contours).reshape(-1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    w, h = w + 1, h + 1  # contour runs through pixel centres; add one pixel of extent
    if max(w, h) / max(min(w, h), 1e-6) < circle_tol:
        x, y, bw, bh = cv2.boundingRect(pts)
        return np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.float64)
    return cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.float64)


def tile_origins(length: int, tile: int, stride: int):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def process_image(img_path: Path, src: Path, out: Path, args, rng, stats: Counter):
    rel = img_path.relative_to(src / "images")
    label = json.loads((src / "labels" / rel.with_suffix(".json")).read_text())
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(src / "masks" / rel), cv2.IMREAD_COLOR)
    inst, part = mask[..., 2], mask[..., 1]           # OpenCV loads BGR: red = index 2
    tank = (part >= TANK_PARTS[0]) & (part <= TANK_PARTS[1])
    inst = np.where(tank, inst, 0).astype(np.uint8)
    classes = instance_classes(label)

    if args.target_gsd > 0:
        scale = label["image_geometry"]["meters_per_pixel"] / args.target_gsd
        size = (max(1, round(img.shape[1] * scale)), max(1, round(img.shape[0] * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, size, interpolation=interp)
        inst = cv2.resize(inst, size, interpolation=cv2.INTER_NEAREST)

    full_area = Counter(dict(zip(*np.unique(inst, return_counts=True))))
    H, W = inst.shape
    T = args.tile
    rows = []
    for k, (y0, x0) in enumerate((y, x) for y in tile_origins(H, T, args.stride)
                                 for x in tile_origins(W, T, args.stride)):
        crop_img = img[y0:y0 + T, x0:x0 + T]
        crop_inst = inst[y0:y0 + T, x0:x0 + T]
        h, w = crop_inst.shape
        # Smaller than a tile: place on a padded canvas at a random offset.
        oy = int(rng.integers(0, T - h + 1)) if h < T else 0
        ox = int(rng.integers(0, T - w + 1)) if w < T else 0
        canvas = np.full((T, T, 3), PAD_VALUE, dtype=np.uint8)
        canvas[oy:oy + h, ox:ox + w] = crop_img

        lines = []
        for iid, area in zip(*np.unique(crop_inst, return_counts=True)):
            if iid == 0:
                continue
            if area < args.min_area or area < args.min_visible * full_area[iid]:
                stats["dropped_small_or_cut"] += 1
                continue
            cls = classes.get(int(iid))
            if cls is None:
                stats["dropped_unknown_instance"] += 1
                continue
            corners = instance_obb((crop_inst == iid).astype(np.uint8), args.circle_tol)
            corners += (ox, oy)
            lines.append(format_obb_line(CLASS_ID[cls], corners, T, T))
            stats[cls] += 1
        if not lines:
            stats["empty_tiles_skipped"] += 1
            continue

        name = f"{img_path.stem}_t{k}"
        out_img = out / "images" / f"{name}.jpg"
        out_img.parent.mkdir(parents=True, exist_ok=True)
        # JPEG like the real NAIP chips, so compression artefacts are not a domain cue.
        cv2.imwrite(str(out_img), canvas, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        write_label(out / "labels" / f"{name}.txt", lines)
        rows.append({"image": out_img.name, "source": str(rel), "folder": rel.parent.name,
                     "tile": k, "objects": len(lines),
                     "meters_per_pixel": label["image_geometry"]["meters_per_pixel"],
                     "fill_level": label["tank_geometry"]["fill_level_percentage"]})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("synthetic_data"))
    ap.add_argument("--out", type=Path, default=Path("yolo_data/synthetic"))
    ap.add_argument("--fraction", type=float, default=0.2, help="share of synthetic images to use")
    ap.add_argument("--target-gsd", type=float, default=0.6,
                    help="rescale to this m/px (real NAIP = 0.6); 0 keeps native resolution")
    ap.add_argument("--tile", type=int, default=512, help="output size, matches the real chips")
    ap.add_argument("--stride", type=int, default=384, help="tiling stride for images larger than --tile")
    ap.add_argument("--min-area", type=int, default=16, help="drop instances below this many pixels")
    ap.add_argument("--min-visible", type=float, default=0.25,
                    help="drop instances whose visible part in a tile is below this share of the whole")
    ap.add_argument("--circle-tol", type=float, default=1.15,
                    help="aspect ratio below which a footprint gets an axis-aligned box")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-manifest", type=Path,
                    help="use every render NOT in this manifest.csv (e.g. the training tiles'); ignores --fraction")
    ap.add_argument("--list-name", default="train.txt", help="image list file to write (test.txt for a test set)")
    ap.add_argument("--split", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
                    help="convert ALL renders and split them, e.g. 0.9 0.05 0.05; ignores --fraction")
    ap.add_argument("--split-file", type=Path,
                    help="shared split JSON from synthetic_data/make_shared_split.py: converts every render in it "
                         "into the split it names (held-out tags are not used); ignores --fraction and --split")
    args = ap.parse_args()
    if args.split and abs(sum(args.split) - 1) > 1e-6:
        raise SystemExit("--split fractions must add up to 1")

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; remove it first so old tiles do not mix in")

    split_of = None
    if args.split_file:
        named = json.loads(args.split_file.read_text())["split"]
        files = {p.name: p for p in (args.src / "images").glob("*/*.png")}
        missing = sorted(set(named) - set(files))
        if missing:
            raise SystemExit(f"{len(missing)} renders of the split file are not in {args.src}, e.g. {missing[0]}")
        extra = sorted(set(files) - set(named))
        if extra:
            raise SystemExit(f"{len(extra)} renders in {args.src} are not in the split file, e.g. {extra[0]}")
        split_of = {files[n]: s for n, s in named.items()}
        chosen = sorted(split_of)
    elif args.split:
        split_of = split_images(args.src, args.split, args.seed)
        chosen = sorted(split_of)
    elif args.exclude_manifest:
        chosen, used = unused_images(args.src, args.exclude_manifest)
        print(f"{len(used)} renders excluded (in {args.exclude_manifest}); {len(chosen)} unused renders selected")
    else:
        chosen = select_images(args.src, args.fraction, args.seed)
    rng = np.random.default_rng(args.seed)
    stats, rows = Counter(), []
    for i, p in enumerate(chosen, 1):
        new = process_image(p, args.src, args.out, args, rng, stats)
        for r in new:
            r["split"] = split_of[p] if split_of else Path(args.list_name).stem
        rows += new
        if i % 50 == 0 or i == len(chosen):
            print(f"[{i}/{len(chosen)}] tiles written: {len(rows)}")

    if split_of:
        tags = sorted({t for r in rows if r["split"].startswith("ood:") for t in r["split"][4:].split(",")})
        for s in ("train", "val", "test") + tuple(f"test_ood_{t}" for t in tags):
            if s.startswith("test_ood_"):
                part = [r for r in rows if r["split"].startswith("ood:") and s[9:] in r["split"][4:].split(",")]
            else:
                part = [r for r in rows if r["split"] == s]
            write_split_list(args.out / f"{s}.txt", [args.out / "images" / r["image"] for r in part], args.out)
            renders = len({r["source"] for r in part})
            print(f"{s:5s}: {renders:5d} renders, {len(part):5d} tiles, {sum(r['objects'] for r in part):6d} objects")
    else:
        write_split_list(args.out / args.list_name, [args.out / "images" / r["image"] for r in rows], args.out)
    with open(args.out / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(chosen)} source images -> {len(rows)} tiles in {args.out}")
    for k, v in sorted(stats.items()):
        print(f"  {k:32s} {v}")


if __name__ == "__main__":
    main()
