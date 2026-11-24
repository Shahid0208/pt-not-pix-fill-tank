"""Convert the real NAIP tank dataset (512x512 JPG + PASCAL VOC XML) to YOLO-OBB format.

VOC boxes are axis-aligned, so each becomes a 4-corner box with angle 0. The train/val/test
split is grouped by source NAIP tile: chips of one tile are neighbours and often share
tanks across their borders, so a random per-chip split would leak them into val/test.
Whole tiles are assigned (shuffled with --seed) until val and test reach their share of
images, so the split hits 90/5/5 closely even though tiles have 1-156 chips.

    python yolo_obb/prepare_real.py --images data/images --xmls data/xmls --out data/yolo_src/real

(make_split_v2.py then builds the final leakage-free split; the paper's lists are in splits/yolo_real/.)
"""
import argparse
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from common import CLASS_ID, SPLITS, box_to_corners, format_obb_line, write_label, write_split_list

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import parse_image_name  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif")


def assign_splits(stems, val: float, test: float, seed: int):
    """Image stem -> split. Whole NAIP tiles go to one split; tiles are taken in a seeded random
    order and fill test, then val, up to their share of images. A tile that would overshoot a
    target by more than half its own size goes to train, so the targets are met closely."""
    tiles = defaultdict(list)
    for stem in stems:
        tiles[parse_image_name(stem).tile].append(stem)
    order = sorted(tiles)
    random.Random(seed).shuffle(order)
    targets = {"test": round(len(stems) * test), "val": round(len(stems) * val)}
    counts = {"test": 0, "val": 0}
    mapping = {}
    for tile in order:
        size, split = len(tiles[tile]), "train"
        for s in ("test", "val"):
            if counts[s] < targets[s] and counts[s] + size - targets[s] <= size / 2:
                split = s
                counts[s] += size
                break
        mapping.update({stem: split for stem in tiles[tile]})
    return mapping


def find_image(images: Path, stem: str):
    for ext in IMG_EXTS:
        p = images / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_xml(path: Path, stats: Counter):
    if path.stat().st_size == 0:  # the dataset's 47 background chips have 0-byte XMLs: no tanks
        stats["empty_xml_background_image"] += 1
        return None, None, []
    root = ET.parse(path).getroot()
    size = root.find("size")
    W, H = int(size.findtext("width")), int(size.findtext("height"))
    objects = []
    for obj in root.iter("object"):
        name = obj.findtext("name").strip()
        if name not in CLASS_ID:
            stats[f"skipped_unknown_class:{name}"] += 1
            continue
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, min(x1, x2)), min(float(W), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(H), max(y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            stats["skipped_degenerate_box"] += 1
            continue
        objects.append((CLASS_ID[name], box_to_corners(x1, y1, x2, y2)))
    return W, H, objects


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="folder of real image chips")
    ap.add_argument("--xmls", type=Path, required=True, help="folder of PASCAL VOC XMLs")
    ap.add_argument("--out", type=Path, default=Path("data/yolo_src/real"))
    ap.add_argument("--val", type=float, default=0.05, help="share of images for validation")
    ap.add_argument("--test", type=float, default=0.05, help="share of images for test")
    ap.add_argument("--seed", type=int, default=158,
                    help="tile order for the split; 158 = most even class shares (all classes 3.9-6.0%% in val and test)")
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking (use if the source may move)")
    args = ap.parse_args()

    out_images, out_labels = args.out / "images", args.out / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    stats, split_files, split_objects = Counter(), {s: [] for s in SPLITS}, Counter()

    xmls = sorted(args.xmls.glob("*.xml"))
    tile_split = assign_splits([x.stem for x in xmls if find_image(args.images, x.stem)],
                               args.val, args.test, args.seed)
    for i, xml in enumerate(xmls, 1):
        img = find_image(args.images, xml.stem)
        if img is None:
            stats["xml_without_image"] += 1
            continue
        W, H, objects = parse_xml(xml, stats)
        lines = [format_obb_line(c, corners, W, H) for c, corners in objects]
        for c, _ in objects:
            stats[f"class:{c}"] += 1

        dst = out_images / img.name
        if not dst.exists() and not dst.is_symlink():
            if args.copy:
                shutil.copy2(img, dst)
            else:
                os.symlink(img.resolve(), dst)
        # Empty label file = background image (a useful negative), not an error.
        write_label(out_labels / f"{img.stem}.txt", lines)

        split = tile_split[xml.stem]
        split_files[split].append(dst)
        split_objects[split] += len(lines)
        if i % 2000 == 0:
            print(f"[{i}/{len(xmls)}]")

    for s in SPLITS:
        write_split_list(args.out / f"{s}.txt", split_files[s], args.out)
        n = sum(len(v) for v in split_files.values())
        print(f"{s:5s}: {len(split_files[s]):6d} images ({100 * len(split_files[s]) / n:4.1f}%), "
              f"{split_objects[s]:7d} objects")
    for k, v in sorted(stats.items()):
        print(f"  {k:40s} {v}")


if __name__ == "__main__":
    main()
