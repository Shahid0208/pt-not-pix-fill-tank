"""Write the Ultralytics dataset YAMLs next to the prepared real/ and synthetic/ folders.

    data_real.yaml   train = real                 (baseline)
    data_mixed.yaml  train = real + synthetic     (only if synthetic/train.txt exists)
Val and test are real-only in both, so the two runs are directly comparable.

    python yolo_obb/make_data_yaml.py --root yolo_data
"""
import argparse
from pathlib import Path

import yaml

from common import CLASS_NAMES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("yolo_data"))
    args = ap.parse_args()

    for f in ("train.txt", "val.txt", "test.txt"):
        if not (args.root / "real" / f).exists():
            raise SystemExit(f"missing {args.root / 'real' / f}; run prepare_real.py first")

    variants = {"data_real.yaml": ["real/train.txt"]}
    if (args.root / "synthetic" / "train.txt").exists():
        variants["data_mixed.yaml"] = ["real/train.txt", "synthetic/train.txt"]

    for fname, train in variants.items():
        data = {
            # `path` is rewritten to this file's folder by train.py, so the dataset can be
            # copied anywhere (e.g. a local disk) without editing this file.
            "path": str(args.root.resolve()),
            "train": train,
            "val": "real/val.txt",
            "test": "real/test.txt",
            "names": dict(enumerate(CLASS_NAMES)),
        }
        (args.root / fname).write_text(yaml.safe_dump(data, sort_keys=False))
        print(f"wrote {args.root / fname}")


if __name__ == "__main__":
    main()
