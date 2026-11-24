"""Remove renders that failed validation from a render set you generated yourself (not needed for the Zenodo data,
which already excludes them).

    python synthetic_data/exclude_renders.py --data merged --list synthetic_data/excluded_renders.txt

Deletes each listed render's image, mask and label and drops it from images/<size>/metadata.json. The paper's set
excludes the 13 renders in excluded_renders.txt (from src/validate_renders.py), leaving 24,987 of 25,000.
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="render set with images/<size>/metadata.json")
    ap.add_argument("--list", required=True, help="file names to remove, one per line")
    a = ap.parse_args()
    names = {line.strip() for line in open(a.list) if line.strip()}
    gone = 0
    for meta_path in glob.glob(os.path.join(a.data, "images", "*", "metadata.json")):
        folder = os.path.basename(os.path.dirname(meta_path))
        meta = json.load(open(meta_path))
        for n in [n for n in meta if n in names]:
            meta.pop(n)
            for p in (os.path.join(a.data, "images", folder, n), os.path.join(a.data, "masks", folder, n),
                      os.path.join(a.data, "labels", folder, n[:-4] + ".json")):
                if os.path.exists(p):
                    os.remove(p)
            gone += 1
        json.dump(meta, open(meta_path, "w"), indent=1)
    print(f"excluded {gone} of {len(names)} listed renders from {a.data}")
    if gone != len(names):
        raise SystemExit("some listed renders were not found")


if __name__ == "__main__":
    main()
