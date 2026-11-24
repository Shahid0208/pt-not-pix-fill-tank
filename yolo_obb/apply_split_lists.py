"""Use the paper's exact real-image detection split (splits/yolo_real/) in a dataset root built by make_split_v2.py.

    python yolo_obb/apply_split_lists.py --root data/yolo --lists splits/yolo_real

make_split_v2.py picks its real-image split by a class-balance search, so a rebuild can give a different real split.
This replaces real/{train,val,test}.txt with the paper's lists (every listed image must exist in the root), then
rewrites the data YAMLs and the oversampled train list. Synthetic tiles follow the shared render split and need no
fixing.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--lists", type=Path, default=HERE.parent / "splits" / "yolo_real")
    a = ap.parse_args()
    for f in ("train.txt", "val.txt", "test.txt"):
        names = [line.strip() for line in open(a.lists / f) if line.strip()]
        missing = [n for n in names if not (a.root / "real" / n).exists()]
        if missing:
            raise SystemExit(f"real/{f}: {len(missing)} listed images missing in {a.root}/real, e.g. {missing[0]}")
        old = (a.root / "real" / f).read_text().split() if (a.root / "real" / f).exists() else []
        shutil.copy(a.lists / f, a.root / "real" / f)
        print(f"real/{f}: {len(names)} images (rebuilt list had {len(old)}, same set: {set(old) == set(names)})")
    for script in ("make_data_yaml.py", "oversample.py"):
        subprocess.run([sys.executable, str(HERE / script), "--root", str(a.root)], check=True)


if __name__ == "__main__":
    main()
