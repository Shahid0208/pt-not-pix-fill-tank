"""
Merge several generated datasets (the original set and the shards of a follow-up batch) into one dataset
folder with the same layout: images/<size>/ (+ metadata.json), masks/<size>/, labels/<size>/,
images/sensor_sim_done.json and images/balanced_plan.json.

    python src/merge_batches.py --out merged --sources shard00 shard01 shard02 shard03

Only finished images are merged: the PNG, its metadata entry, its label file and its mask must exist, and
(if the source used the sensor simulation) the image must be in that source's sensor log. Files are
hard-linked (no extra disk space; falls back to copying across file systems). A file name that appears in
two sources stops the merge. Planned images that are not finished are listed in merge_report.json.
"""
import argparse
import json
import os
import shutil
import sys


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="new dataset folder (must not contain images/ yet)")
    ap.add_argument("--sources", nargs="+", required=True, help="dataset roots, each with images/ labels/ masks/")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    if os.path.exists(os.path.join(out, "images")):
        sys.exit(f"{out}/images already exists; choose a new --out")

    meta_out, sensor_out, plan_out, seen = {}, {}, [], {}
    report = {"sources": {}, "unfinished": []}
    for src in a.sources:
        src = os.path.abspath(src)
        img_root = os.path.join(src, "images")
        sensor = load(os.path.join(img_root, "sensor_sim_done.json"), {})
        plan = load(os.path.join(img_root, "balanced_plan.json"), [])
        n_ok = n_skip = 0
        finished = set()
        for folder in sorted(os.listdir(img_root)):
            meta = load(os.path.join(img_root, folder, "metadata.json"), None)
            if meta is None:
                continue
            for name, entry in meta.items():
                stem = os.path.splitext(name)[0]
                files = {"images": os.path.join(img_root, folder, name),
                         "masks": os.path.join(src, "masks", folder, name),
                         "labels": os.path.join(src, "labels", folder, stem + ".json")}
                processed = not sensor or f"{folder}/{name}" in sensor
                if not all(os.path.exists(p) for p in files.values()) or not processed:
                    n_skip += 1
                    continue
                if name in seen:
                    sys.exit(f"{name} is in both {seen[name]} and {src}")
                seen[name] = src
                for kind, p in files.items():
                    link(p, os.path.join(out, kind, folder, os.path.basename(p)))
                meta_out.setdefault(folder, {})[name] = entry
                if f"{folder}/{name}" in sensor:
                    sensor_out[f"{folder}/{name}"] = sensor[f"{folder}/{name}"]
                finished.add(name)
                n_ok += 1
        # the shards all carry the full plan: keep the entries of images this source finished
        plan_out += [p for p in plan if p["filename"] in finished]
        report["sources"][src] = {"merged": n_ok, "skipped_incomplete": n_skip}
        print(f"{src}: {n_ok} merged, {n_skip} incomplete skipped")

    # planned images that no source finished (a shard plan lists all 20,000; each shard renders a part)
    all_planned = {}
    for src in a.sources:
        for p in load(os.path.join(os.path.abspath(src), "images", "balanced_plan.json"), []):
            all_planned[p["filename"]] = p["folder"]
    report["unfinished"] = sorted(n for n in all_planned if n not in seen)

    for folder, meta in meta_out.items():
        with open(os.path.join(out, "images", folder, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)
    with open(os.path.join(out, "images", "sensor_sim_done.json"), "w") as f:
        json.dump(sensor_out, f, indent=1)
    plan_out.sort(key=lambda p: p["index"])
    with open(os.path.join(out, "images", "balanced_plan.json"), "w") as f:
        json.dump(plan_out, f, indent=1)
    report["total_merged"] = len(seen)
    report["per_folder"] = {k: len(v) for k, v in sorted(meta_out.items())}
    with open(os.path.join(out, "merge_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"merged {len(seen)} images into {out} | per folder {report['per_folder']} | "
          f"unfinished planned images: {len(report['unfinished'])}")


if __name__ == "__main__":
    main()
