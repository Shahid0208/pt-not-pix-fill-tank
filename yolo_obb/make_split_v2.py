"""Leakage-free 90/5/5 split of ALL real chips + 20% of the synthetic renders, as a new dataset root.

    python yolo_obb/make_split_v2.py --src data/yolo_src --synthetic-src data/yolo_synthetic \
        --split 0.8 0.15 0.05 --out data/yolo
    python yolo_obb/make_data_yaml.py --root data/yolo      # data_real.yaml, data_mixed.yaml
    python yolo_obb/oversample.py --root data/yolo          # real/train_os.txt, data_mixed_os.yaml

Only new list files and symlinks are written; the images and labels of --src are reused and not changed.
The layout matches --src (real/, synthetic/ with train/val/test.txt), so train.py works as is.

Real (all chips). The old split grouped by NAIP tile, but one place is imaged in several years (a different tile
per year), and neighbouring quarter-quads overlap at their edges, so the same tanks could be in train and test.
Here chips are joined into groups (union-find) when they
  * come from the same NAIP quarter-quad (m_QQQQQQQ_qq), whatever the year, or
  * have overlapping ground footprints (lat/lon corners from image_characteristics.csv), any tile, any year, or
Duplicates are removed first: the source dataset stores some chips several times under different names (other
places / years; 94 byte-identical groups, plus re-encoded copies), with the same boxes. Pairs whose 16x16
thumbnails correlate > 0.99 are compared at 128x128 and count as the same picture if the mean absolute grey
difference < 2 (plus byte-identical files). Of each group of copies, the one the tank inventory references most
(tank_inventory.json lists the chip names each tank is in) is kept as the genuine chip, ties -> first name; the
other copies are dropped from every split (real/dropped_duplicates.csv). The inventory lists every copy, so it
rarely decides; the kept chip may therefore carry the wrong place name. It is treated as possibly lying at EVERY
copy's footprint: it is grouped with any chip overlapping any of those footprints (footprint level only). Joining
whole places of all copies instead would chain unrelated places into one huge group that could only go to train.
whole groups go to one split. Groups are assigned in a seeded random order: test, then val, up to 5% of the
images each; a group that would overshoot by more than half its size goes to train. Of --seeds candidate seeds,
the one whose val and test hold the share of every class's boxes closest to 5% is kept (label counts only).

Synthetic (20% of renders). A scene = location + date + local solar time (the same sun). Whole scenes are chosen
until every size folder has 20% of its renders, then split 90/5/5 by scene (per size folder), and every tile of
a render stays with it.

Synthetic from a shared split (--synthetic-src DIR): DIR is a converted render set from
prepare_synthetic.py --split-file (images/, labels/, manifest.csv with a split per tile). Every tile is taken
with the split of its render, unchanged, so YOLO and the keypoint data (same split file) agree render by
render. --synthetic-fraction is then not used (the paper: all real + SynTank-25k, 80/15/5, first command above).

A separate check at the end recomputes all overlaps from scratch and stops with an error on any leak.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import CLASS_NAMES as CLASSES

SPLITS = ("train", "val", "test")


# ------------------------------------------------------------------------------------------------
class DSU:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def read_list(path):
    return [l.strip() for l in open(path) if l.strip()]


def place_of(stem):
    return "_".join(stem.split("_")[:3])               # m_QQQQQQQ_qq: NAIP quarter-quad, any year


def footprints(chars_csv, stems, shrink_deg=2e-5):
    """stem -> (lon0, lat0, lon1, lat1), shrunk by ~2 m so chips that only touch do not count as overlapping."""
    want, out = set(stems), {}
    with open(chars_csv) as f:
        for r in csv.DictReader(f):
            s = r["image_name"]
            if s not in want:
                continue
            la = sorted([float(r["nw_lat_image_coord"]), float(r["se_lat_image_coord"])])
            lo = sorted([float(r["nw_lon_image_coord"]), float(r["se_lon_image_coord"])])
            out[s] = (lo[0] + shrink_deg, la[0] + shrink_deg, lo[1] - shrink_deg, la[1] - shrink_deg)
    return out


def overlapping_pairs(fp, cell=0.01):
    """All pairs of stems whose footprints overlap (grid hashing)."""
    grid = defaultdict(list)
    for s, (x0, y0, x1, y1) in fp.items():
        for i in range(math.floor(x0 / cell), math.floor(x1 / cell) + 1):
            for j in range(math.floor(y0 / cell), math.floor(y1 / cell) + 1):
                grid[(i, j)].append(s)
    pairs = set()
    for members in grid.values():
        for a in range(len(members)):
            ax0, ay0, ax1, ay1 = fp[members[a]]
            for b in range(a + 1, len(members)):
                bx0, by0, bx1, by1 = fp[members[b]]
                if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                    pairs.add(tuple(sorted((members[a], members[b]))))
    return pairs


def duplicate_image_pairs(paths, corr=0.99, max_diff=2.0, chunk=2000):
    """Pairs of stems whose images show the same picture (exact or re-encoded copies)."""
    stems = sorted(paths)
    thumbs, cache = [], {}

    def small(st):                                     # 128x128 grey, loaded only for candidate pairs
        if st not in cache:
            im = cv2.imread(str(paths[st]), cv2.IMREAD_GRAYSCALE)
            cache[st] = cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
        return cache[st]

    for st in stems:
        im = cv2.imread(str(paths[st]), cv2.IMREAD_GRAYSCALE)
        v = cv2.resize(im, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        v -= v.mean()
        n = np.linalg.norm(v)
        thumbs.append(v / n if n > 1e-3 else v)
    F = np.stack(thumbs)
    pairs = set()
    for i0 in range(0, len(F), chunk):
        S = F[i0:i0 + chunk] @ F.T
        for a, b in zip(*np.nonzero(S > corr)):
            i, j = i0 + int(a), int(b)
            if i < j and np.abs(small(stems[i]) - small(stems[j])).mean() < max_diff:
                pairs.add((stems[i], stems[j]))
    # flat (no texture) chips have zero thumbnails and never match above; catch exact copies of them by bytes
    by_md5 = defaultdict(list)
    for st in stems:
        by_md5[hashlib.md5(open(paths[st], "rb").read()).hexdigest()].append(st)
    for members in by_md5.values():
        for m in members[1:]:
            pairs.add(tuple(sorted((members[0], m))))
    return pairs


def label_classes(label_path):
    if not label_path.exists():
        return []
    return [int(l.split()[0]) for l in label_path.read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------------------------------------
def assign_groups(groups, fr, seed):
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    n = sum(len(g) for g in groups.values())
    targets = {"test": round(n * fr[2]), "val": round(n * fr[1])}
    counts = {"test": 0, "val": 0}
    split_of = {}
    for gid in order:
        size, split = len(groups[gid]), "train"
        for s in ("test", "val"):
            if counts[s] < targets[s] and counts[s] + size - targets[s] <= size / 2:
                split, counts[s] = s, counts[s] + size
                break
        for stem in groups[gid]:
            split_of[stem] = split
    return split_of


def class_balance_cost(split_of, boxes, fr):
    """Sum over classes of |share of the class's boxes in val - fr_val| + |... test - fr_test| (relative)."""
    tot, part = Counter(), {s: Counter() for s in SPLITS}
    for stem, cl in boxes.items():
        for c in cl:
            tot[c] += 1
            part[split_of[stem]][c] += 1
    cost = 0.0
    for c in tot:
        for s, f in (("val", fr[1]), ("test", fr[2])):
            cost += abs(part[s][c] / tot[c] - f) / f
    return cost


def inventory_refs(path):
    """chip name -> number of inventory tanks that list it."""
    refs = Counter()
    for f in json.load(open(path))["features"]:
        names = f["properties"].get("image_name") or []
        refs.update([names] if isinstance(names, str) else names)
    return refs


def deduplicate(stems, paths, refs):
    """Returns (kept stems, rows describing dropped copies, info)."""
    pairs = duplicate_image_pairs({s: paths[s] for s in stems})
    dsu = DSU(stems)
    for a, b in pairs:
        dsu.union(a, b)
    clusters = defaultdict(list)
    for s in stems:
        clusters[dsu.find(s)].append(s)
    dropped, ties = [], 0
    for members in clusters.values():
        if len(members) == 1:
            continue
        members.sort()
        best = max(members, key=lambda m: refs[m])        # max keeps the first name on ties
        if sum(refs[m] == refs[best] for m in members) > 1:
            ties += 1
        dropped += [{"dropped": m, "kept": best, "dropped_inventory_refs": refs[m], "kept_inventory_refs": refs[best]}
                    for m in members if m != best]
    gone = {d["dropped"] for d in dropped}
    info = {"duplicate_pairs": len(pairs), "duplicate_groups": sum(len(m) > 1 for m in clusters.values()),
            "chips_in_duplicate_groups": sum(len(m) for m in clusters.values() if len(m) > 1),
            "dropped_copies": len(gone), "groups_where_inventory_did_not_decide": ties}
    return [s for s in stems if s not in gone], dropped, info


def split_real(src, chars, fr, seeds, inventory):
    entries = {}
    for s in SPLITS:
        for e in read_list(src / "real" / f"{s}.txt"):
            entries[Path(e).stem] = e
    all_stems = sorted(entries)
    paths = {s: (src / "real" / entries[s][2:]).resolve() for s in all_stems}
    stems, dropped, dinfo = deduplicate(all_stems, paths, inventory_refs(inventory))
    fp_all = footprints(chars, all_stems)
    fp = {s: fp_all[s] for s in stems if s in fp_all}
    dsu = DSU(stems)
    by_place = defaultdict(list)
    for s in stems:
        by_place[place_of(s)].append(s)
    for members in by_place.values():
        for m in members[1:]:
            dsu.union(members[0], m)
    pairs = overlapping_pairs(fp)
    for a, b in pairs:
        dsu.union(a, b)
    # a kept duplicate may really lie at any of its copies' footprints
    virtual = {f"{d['kept']}#copy{i}": fp_all[d["dropped"]] for i, d in enumerate(dropped) if d["dropped"] in fp_all}
    vpairs = 0
    for a, b in overlapping_pairs({**fp, **virtual}):
        a, b = a.split("#")[0], b.split("#")[0]
        if a != b:
            dsu.union(a, b)
            vpairs += 1
    groups = defaultdict(list)
    for s in stems:
        groups[dsu.find(s)].append(s)
    boxes = {s: label_classes(src / "real" / "labels" / f"{s}.txt") for s in stems}
    best = min(range(seeds), key=lambda sd: class_balance_cost(assign_groups(groups, fr, sd), boxes, fr))
    split_of = assign_groups(groups, fr, best)
    info = {"images_in_source": len(all_stems), **dinfo, "images": len(stems),
            "missing_footprint": len(stems) - len(fp), "places": len(by_place),
            "overlapping_chip_pairs": len(pairs), "pairs_incl_copy_footprints": vpairs, "groups": len(groups),
            "largest_group": max(len(g) for g in groups.values()), "seed": best,
            "class_balance_cost": class_balance_cost(split_of, boxes, fr)}
    return entries, split_of, boxes, fp_all, info, {s: paths[s] for s in stems}, dropped


def scene_of(source):
    # v1 names: ..._lst1230_fill_..., v2 names: ..._lst1230_r24.8_h10.1_fill_...
    m = re.search(r"b\d+_(.+)_(\d{8})_lst(\d{4})(?:_r[\d.]+_h[\d.]+)?_fill", Path(source).name)
    return "|".join(m.groups())


def split_synthetic_given(syn_root):
    """All tiles of a converted render set, each with the split of its render (from its manifest)."""
    rows = list(csv.DictReader(open(syn_root / "manifest.csv")))
    renders, render_split = defaultdict(list), {}
    for r in rows:
        renders[r["source"]].append(r)
        if render_split.setdefault(r["source"], r["split"]) != r["split"]:
            raise SystemExit(f"tiles of {r['source']} are in different splits")
    bad = sorted({s for s in render_split.values() if s not in SPLITS})
    if bad:
        raise SystemExit(f"{syn_root}/manifest.csv has splits other than train/val/test: {bad}")
    info = {"renders_total": len(renders), "scenes_total": len({scene_of(r) for r in renders}),
            "renders_chosen": len(render_split), "source": str(syn_root), "split_from": "manifest (shared split file)"}
    return renders, render_split, info


def split_synthetic(src, fraction, fr, seed):
    rows = list(csv.DictReader(open(src / "synthetic" / "manifest.csv")))
    renders = defaultdict(list)                        # render -> its tile rows
    for r in rows:
        renders[r["source"]].append(r)
    folder_of = {src_: rs[0]["folder"] for src_, rs in renders.items()}
    scenes = defaultdict(list)
    for src_ in renders:
        scenes[scene_of(src_)].append(src_)
    per_folder = Counter(folder_of.values())
    target = {f: round(n * fraction) for f, n in per_folder.items()}

    rng = random.Random(seed)
    order = sorted(scenes)
    rng.shuffle(order)
    chosen, have = [], Counter()
    for sc in order:
        need = Counter(folder_of[r] for r in scenes[sc])
        if all(have[f] + k <= target[f] for f, k in need.items()):
            chosen.append(sc)
            have.update(need)
        if all(have[f] >= target[f] for f in target):
            break

    # 90/5/5 by scene, filling each split's per-folder share in turn (largest deficit first)
    order = sorted(chosen)
    rng.shuffle(order)
    order.sort(key=lambda sc: folder_of[scenes[sc][0]])
    counts = {s: Counter() for s in SPLITS}
    seen = Counter()
    split_of_scene = {}
    for sc in order:
        f = folder_of[scenes[sc][0]]
        k = len(scenes[sc])
        deficit = {s: fr[i] * (seen[f] + k) - counts[s][f] for i, s in enumerate(SPLITS)}
        s = max(SPLITS, key=lambda x: deficit[x])
        split_of_scene[sc] = s
        counts[s][f] += k
        seen[f] += k
    render_split = {r: split_of_scene[sc] for sc in chosen for r in scenes[sc]}
    info = {"renders_total": len(renders), "scenes_total": len(scenes), "renders_chosen": len(render_split),
            "scenes_chosen": len(chosen), "target_per_folder": target, "chosen_per_folder": dict(have)}
    return renders, render_split, info


# ------------------------------------------------------------------------------------------------
def link(target, link_path):
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    os.symlink(os.path.abspath(target), link_path)


def write_list(path, entries):
    path.write_text("\n".join(sorted(entries)) + "\n")


def verify(out, fp, renders_split_rows, paths, dropped):
    """Independent leakage check on the written lists."""
    issues = []
    real = {s: [Path(e).stem for e in read_list(out / "real" / f"{s}.txt")] for s in SPLITS}
    split_of = {st: s for s in SPLITS for st in real[s]}
    if len(split_of) != sum(len(v) for v in real.values()):
        issues.append("a real chip is listed in more than one split")
    for name, key in (("NAIP tile", lambda st: st.rsplit("_", 2)[0]), ("place (quarter-quad, any year)", place_of)):
        m = defaultdict(set)
        for st, s in split_of.items():
            m[key(st)].add(s)
        n = sum(len(v) > 1 for v in m.values())
        if n:
            issues.append(f"{n} real {name} groups span splits")
    live = {k: v for k, v in fp.items() if k in split_of}
    live.update({f"{d['kept']}#copy{i}": fp[d["dropped"]] for i, d in enumerate(dropped) if d["dropped"] in fp})
    cross = [(a, b) for a, b in overlapping_pairs(live)
             if a.split("#")[0] != b.split("#")[0] and split_of[a.split("#")[0]] != split_of[b.split("#")[0]]]
    if cross:
        issues.append(f"{len(cross)} overlapping real chip pairs span splits, e.g. {cross[0]}")
    dup_left = list(duplicate_image_pairs({k: v for k, v in paths.items() if k in split_of}))
    if dup_left:                                     # after de-duplication none may remain, in any split
        issues.append(f"{len(dup_left)} duplicate real image pairs remain, e.g. {dup_left[0]}")
    syn = {s: {Path(e).stem for e in read_list(out / "synthetic" / f"{s}.txt")} for s in SPLITS}
    tile_split = {t: s for s in SPLITS for t in syn[s]}
    by_render, by_scene = defaultdict(set), defaultdict(set)
    for r in renders_split_rows:
        t = Path(r["image"]).stem
        if t in tile_split:
            by_render[r["source"]].add(tile_split[t])
            by_scene[scene_of(r["source"])].add(tile_split[t])
    for name, m in (("render", by_render), ("scene (location+date+time)", by_scene)):
        n = sum(len(v) > 1 for v in m.values())
        if n:
            issues.append(f"{n} synthetic {name}s span splits")
    return issues, cross


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("yolo_data"))
    ap.add_argument("--out", type=Path, default=Path("data/yolo"))
    ap.add_argument("--chars", type=Path, default=Path("data/image_characteristics.csv"))
    ap.add_argument("--inventory", type=Path, default=Path("data/tank_inventory.json"))
    ap.add_argument("--synthetic-fraction", type=float, default=0.2)
    ap.add_argument("--synthetic-src", type=Path,
                    help="converted render set with its own split (prepare_synthetic.py --split-file); "
                         "default: <src>/synthetic with --synthetic-fraction")
    ap.add_argument("--split", type=float, nargs=3, default=[0.9, 0.05, 0.05])
    ap.add_argument("--seeds", type=int, default=50, help="candidate seeds for the real split (class balance)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the synthetic selection and split")
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"{a.out} is not empty; remove it first")
    fr = a.split

    # real
    entries, split_of, boxes, fp, rinfo, paths, dropped = split_real(a.src, a.chars, fr, a.seeds, a.inventory)
    (a.out / "real").mkdir(parents=True)
    link(a.src / "real" / "images", a.out / "real" / "images")
    link(a.src / "real" / "labels", a.out / "real" / "labels")
    for s in SPLITS:
        write_list(a.out / "real" / f"{s}.txt", [entries[st] for st, x in split_of.items() if x == s])
    with open(a.out / "real" / "dropped_duplicates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dropped", "kept", "dropped_inventory_refs", "kept_inventory_refs"])
        w.writeheader()
        w.writerows(dropped)

    # synthetic
    if a.synthetic_src:
        syn_root = a.synthetic_src
        renders, render_split, sinfo = split_synthetic_given(syn_root)
    else:
        syn_root = a.src / "synthetic"
        renders, render_split, sinfo = split_synthetic(a.src, a.synthetic_fraction, fr, a.seed)
    (a.out / "synthetic").mkdir(parents=True)
    rows_out, lists = [], {s: [] for s in SPLITS}
    for src_, s in sorted(render_split.items()):
        for r in renders[src_]:
            t = Path(r["image"])
            link(syn_root / "images" / t.name, a.out / "synthetic" / "images" / t.name)
            lab = syn_root / "labels" / f"{t.stem}.txt"
            if lab.exists():
                link(lab, a.out / "synthetic" / "labels" / f"{t.stem}.txt")
            lists[s].append(f"./images/{t.name}")
            rows_out.append({**r, "split": s, "scene": scene_of(src_)})
    for s in SPLITS:
        write_list(a.out / "synthetic" / f"{s}.txt", lists[s])
    with open(a.out / "synthetic" / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)

    # report + independent leakage check
    issues, _ = verify(a.out, fp, rows_out, paths, dropped)
    rep = {"real": rinfo, "synthetic": sinfo, "leakage_issues": issues, "splits": {}}
    for s in SPLITS:
        st = [x for x, y in split_of.items() if y == s]
        cls = Counter(c for x in st for c in boxes[x])
        rs = [r for r in rows_out if r["split"] == s]
        rep["splits"][s] = {
            "real_images": len(st), "real_boxes": sum(cls.values()),
            "real_boxes_per_class": {CLASSES[c] if c < len(CLASSES) else str(c): n for c, n in sorted(cls.items())},
            "synthetic_renders": len({r["source"] for r in rs}), "synthetic_tiles": len(rs),
            "synthetic_objects": sum(int(r["objects"]) for r in rs),
            "synthetic_renders_per_folder": dict(Counter(r["folder"] for r in {r["source"]: r for r in rs}.values())),
        }
    json.dump(rep, open(a.out / "split_report.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))
    if issues:
        raise SystemExit("LEAKAGE FOUND: " + "; ".join(issues))
    print(f"\nno leakage found; wrote {a.out}")


if __name__ == "__main__":
    main()
