"""
One train/val/test split of a synthetic render set, shared by YOLO-OBB and the keypoint model (no leakage).

    python synthetic_data/make_shared_split.py --data data/SynTank-25k \
        --split 0.8 0.15 0.05 --out data/SynTank-25k/shared_split.json

Every render is assigned to exactly one split, and both pipelines read this file
(yolo_obb/prepare_synthetic.py --split-file, keypoint/prepare_synthetic_kp.py --split-file), so no render is
train for one model and val/test for the other.

Leakage units (union-find, a whole unit goes to one split):
  * scene = location + date + local solar time (the same sun; the YOLO split's unit), from the file name;
  * the keypoint scene group = location | sun elevation | sun azimuth, from the metadata.
Units are stratified by size folder x fill bin (width 0.1) and visited stratum by stratum in a seeded random
order; each goes to the split furthest below its target share so far, so the global fractions are met
exactly (e.g. 500 -> 400/75/25) and every stratum is spread over the splits.
Held-out design tags in the metadata ("holdout") are ignored: every render is split like the others.

Checks at the end (exits 1 on any failure): every render in exactly one split, no scene or keypoint group in
two splits, split sizes equal the targets.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

SPLITS = ("train", "val", "test")
FILL_BIN = 0.1


def scene_of(name):
    # v1 names: ..._lst1230_fill_..., v2 names: ..._lst1230_r24.8_h10.1_fill_...
    m = re.search(r"b\d+_(.+)_(\d{8})_lst(\d{4})(?:_r[\d.]+_h[\d.]+)?_fill", name)
    if not m:
        raise ValueError(f"cannot read the scene from {name}")
    return "|".join(m.groups())


class DSU:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def load(data):
    renders = {}
    for meta_path in sorted(glob.glob(os.path.join(data, "images", "*", "metadata.json"))):
        folder = os.path.basename(os.path.dirname(meta_path))
        for name, e in json.load(open(meta_path)).items():
            sc = e["solar_conditions"]
            renders[name] = {
                "folder": folder,
                "fill": e["tank_geometry"]["fill_level_percentage"] / 100.0,
                "scene": scene_of(name),
                "group": f"{(e.get('location') or {}).get('preset') or ''}|{sc['sun_elevation_degrees']}|"
                         f"{sc['sun_azimuth_degrees']}",
            }
    return renders


def split(renders, fracs, seed):
    names = sorted(renders)
    dsu = DSU(names)
    for key in ("scene", "group"):
        first = {}
        for n in names:
            k = renders[n][key]
            if k in first:
                dsu.union(first[k], n)
            else:
                first[k] = n
    units = defaultdict(list)
    for n in names:
        units[dsu.find(n)].append(n)
    strata = defaultdict(list)
    for u, members in units.items():
        r = renders[members[0]]
        strata[(r["folder"], min(int(r["fill"] / FILL_BIN + 1e-9), int(round(1 / FILL_BIN)) - 1))].append(u)
    rng = random.Random(seed)
    counts, seen, split_of = Counter(), 0, {}
    for st in sorted(strata):
        us = sorted(strata[st])
        rng.shuffle(us)
        for u in us:
            k = len(units[u])
            deficit = {s: f * (seen + k) - counts[s] for s, f in zip(SPLITS, fracs)}
            s = max(SPLITS, key=lambda x: deficit[x])
            for n in units[u]:
                split_of[n] = s
            counts[s] += k
            seen += k
    return split_of, units


def check(renders, split_of, fracs):
    issues = []
    if set(split_of) != set(renders):
        issues.append("not every render has exactly one split")
    for key in ("scene", "group"):
        m = defaultdict(set)
        for n, s in split_of.items():
            m[renders[n][key]].add(s)
        bad = [k for k, v in m.items() if len(v) > 1]
        if bad:
            issues.append(f"{len(bad)} {key}s span splits, e.g. {bad[0]}")
    n = len(renders)
    got = Counter(split_of.values())
    for s, f in zip(SPLITS, fracs):
        if abs(got[s] - f * n) > 1:
            issues.append(f"{s} has {got[s]} renders, target {f * n:.0f}")
    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="render set with images/<folder>/metadata.json")
    ap.add_argument("--split", type=float, nargs=3, default=[0.8, 0.15, 0.05])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="split file to write (JSON)")
    a = ap.parse_args()
    if abs(sum(a.split) - 1) > 1e-6:
        raise SystemExit("--split fractions must add up to 1")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; remove it first")
    renders = load(a.data)
    split_of, units = split(renders, a.split, a.seed)
    issues = check(renders, split_of, a.split)
    per = {s: Counter(renders[n]["folder"] for n, x in split_of.items() if x == s) for s in SPLITS}
    report = {
        "data": a.data, "fractions": a.split, "seed": a.seed, "renders": len(renders),
        "leakage_units": len(units), "largest_unit": max(len(u) for u in units.values()),
        "scenes": len({r["scene"] for r in renders.values()}),
        "split_renders": dict(Counter(split_of.values())),
        "split_per_folder": {s: dict(sorted(c.items())) for s, c in per.items()},
        "split_mean_fill": {s: round(sum(renders[n]["fill"] for n, x in split_of.items() if x == s)
                                     / max(1, sum(1 for x in split_of.values() if x == s)), 4) for s in SPLITS},
        "issues": issues,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"split": dict(sorted(split_of.items())), "report": report}, open(a.out, "w"), indent=1)
    print(json.dumps(report, indent=2))
    if issues:
        sys.exit("LEAKAGE OR SIZE PROBLEM: " + "; ".join(issues))
    print(f"\nno leakage found; wrote {a.out}")


if __name__ == "__main__":
    main()
