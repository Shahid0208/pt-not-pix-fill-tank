"""Repeat real training images that contain chosen classes, and write a dataset YAML that uses them.

An image is listed `max(repeat of its classes)` times (classes not named count as 1), so an image
with a floating-roof tank and a water tower still appears only twice with the defaults. Each repeat
gets different random augmentation (mosaic, flips, colour, ...), so this works like class-balanced
augmentation. Validation and test lists are not touched; synthetic tiles are listed once.

    python yolo_obb/oversample.py --root yolo_data          # -> real/train_os.txt + data_mixed_os.yaml
"""
import argparse
from collections import Counter
from pathlib import Path

import yaml

from common import CLASS_ID, CLASS_NAMES

DEFAULT_REPEAT = ["external_floating_roof_tank=2", "water_tower=2", "spherical_tank=2"]


def parse_repeat(items):
    out = {}
    for item in items:
        name, _, n = item.partition("=")
        if name not in CLASS_ID or not n.isdigit() or int(n) < 1:
            raise SystemExit(f"bad --repeat {item!r}: expected <class name>=<int >= 1>")
        out[CLASS_ID[name]] = int(n)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("yolo_data"))
    ap.add_argument("--repeat", nargs="+", default=DEFAULT_REPEAT, help="<class>=<times>, e.g. water_tower=2")
    ap.add_argument("--list-out", default="real/train_os.txt")
    ap.add_argument("--yaml-out", default="data_mixed_os.yaml")
    args = ap.parse_args()
    repeat = parse_repeat(args.repeat)

    lst = args.root / "real" / "train.txt"
    entries = [line.strip() for line in lst.read_text().splitlines() if line.strip()]
    out, boxes_before, boxes_after = [], Counter(), Counter()
    for e in entries:
        label = lst.parent / "labels" / (Path(e).stem + ".txt")
        classes = [int(r.split()[0]) for r in label.read_text().splitlines() if r.strip()] if label.exists() else []
        times = max((repeat.get(c, 1) for c in classes), default=1)
        out += [e] * times
        for c in classes:
            boxes_before[c] += 1
            boxes_after[c] += times

    (args.root / args.list_out).write_text("\n".join(out) + "\n")
    data = yaml.safe_load((args.root / "data_mixed.yaml").read_text())
    data["train"] = [args.list_out, "synthetic/train.txt"]
    (args.root / args.yaml_out).write_text(yaml.safe_dump(data, sort_keys=False))

    print(f"real train images: {len(entries)} -> {len(out)} (+{len(out) - len(entries)}, "
          f"{100 * (len(out) / len(entries) - 1):.1f}%)")
    print(f"{'class':30s} {'boxes':>8s} {'seen/epoch':>11s} {'x':>5s}")
    for c, name in enumerate(CLASS_NAMES):
        b, a = boxes_before[c], boxes_after[c]
        print(f"{name:30s} {b:8d} {a:11d} {a / b if b else 0:5.2f}")
    print(f"wrote {args.root / args.list_out} and {args.root / args.yaml_out}")


if __name__ == "__main__":
    main()
