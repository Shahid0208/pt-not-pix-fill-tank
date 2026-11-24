"""Prepare the Zenodo download of SynTank-25k for this code (run once, CPU, a few minutes).

    python synthetic_data/from_zenodo.py --data data/SynTank-25k

The preparation scripts read two files that the Zenodo archives do not contain:
  - images/<size>/metadata.json   one entry per render (the label without the dense point lists, plus the paths
                                  and the sensor-model settings), as written by the generator;
  - shared_split.json             {"split": {file name: train|val|test}}, the paper's scene-level split,
                                  rebuilt from metadata/splits.csv.
Both are written into --data. Nothing else is changed.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

# Label blocks / point lists that the generator keeps out of metadata.json (they stay in the per-image label).
DROP_BLOCKS = ("acquisition", "anchor_annotations")
DROP_PROJECTED = ("rim_arc_points", "inner_shadow_boundary", "ground_shadow_boundary", "roof_edge_boundary",
                  "tank_base_boundary", "caster_shadow_pairs", "visibility_mask_base",
                  "visibility_mask_inner_shadow", "visibility_mask_roof_edge", "visibility_mask_ground_shadow")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the unzipped SynTank-25k folder (images/, labels/, metadata/)")
    a = ap.parse_args()
    root = a.data
    splits = list(csv.DictReader(open(os.path.join(root, "metadata", "splits.csv"))))
    sensor_path = os.path.join(root, "metadata", "sensor_settings.json")
    sensor = json.load(open(sensor_path)) if os.path.exists(sensor_path) else {}

    by_folder = defaultdict(dict)
    for r in splits:
        name, folder = r["filename"], r["size_class_folder"]
        stem = name[:-4]
        label_rel = f"labels/{folder}/{stem}.json"
        if not os.path.exists(os.path.join(root, "images", folder, name)):
            raise SystemExit(f"missing image: images/{folder}/{name} (are all archives unzipped?)")
        e = json.load(open(os.path.join(root, label_rel)))
        for k in DROP_BLOCKS:
            e.pop(k, None)
        for k in DROP_PROJECTED:
            e.get("projected_truth", {}).pop(k, None)
        e["image_metadata"]["label_relative_path"] = label_rel
        pp = sensor.get(f"{folder}/{name}")
        if pp is not None:
            e.setdefault("satellite_sensor", {})["post_processing"] = pp
        by_folder[folder][name] = e

    for folder, entries in sorted(by_folder.items()):
        json.dump(entries, open(os.path.join(root, "images", folder, "metadata.json"), "w"), indent=1)
        print(f"images/{folder}/metadata.json: {len(entries)} renders")
    split = {r["filename"]: r["split"] for r in splits}
    counts = {s: sum(v == s for v in split.values()) for s in ("train", "val", "test")}
    json.dump({"split": dict(sorted(split.items())), "report": {"source": "metadata/splits.csv",
                                                                "split_renders": counts}},
              open(os.path.join(root, "shared_split.json"), "w"), indent=1)
    print(f"shared_split.json: {counts}")


if __name__ == "__main__":
    main()
