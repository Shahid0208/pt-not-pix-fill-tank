"""Write the real-only and synthetic-only data YAMLs next to data_mixed_os.yaml (same val/test lists).

    python yolo_obb/make_mode_yamls.py --root data/yolo

data_mixed_os.yaml  oversampled real + synthetic train, real val           (made by oversample.py)
data_real_os.yaml   oversampled real train only, real val
data_synth.yaml     synthetic train only, synthetic val (never sees a real image)
All three share the same real and synthetic test lists.
"""
import argparse
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args()
    base = yaml.safe_load((a.root / "data_mixed_os.yaml").read_text())
    real = dict(base, train=["real/train_os.txt"])
    syn = dict(base, train=["synthetic/train.txt"], val="synthetic/val.txt")
    (a.root / "data_real_os.yaml").write_text(yaml.safe_dump(real, sort_keys=False))
    (a.root / "data_synth.yaml").write_text(yaml.safe_dump(syn, sort_keys=False))
    for f in ("data_mixed_os.yaml", "data_real_os.yaml", "data_synth.yaml"):
        d = yaml.safe_load((a.root / f).read_text())
        n = sum(len((a.root / t).read_text().split()) for t in d["train"])
        print(f"{f}: train {d['train']} ({n} entries), val {d['val']}, test {d['test']}")


if __name__ == "__main__":
    main()
