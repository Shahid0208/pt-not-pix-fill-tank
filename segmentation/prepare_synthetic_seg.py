"""
Segmentation crops and 3-class masks from the synthetic v2 renders, in the keypoint run's shared split.

    python segmentation/prepare_synthetic_seg.py --src data/SynTank-25k --split-dir data/keypoint_data \
        --out data/segmentation_data

Crop: square, centred on the target tank's rim centre, side = CTX x rim diameter (2 a_px), resized to SIZE px
(image: area/linear interpolation, mask: nearest). Pixels outside the render are padded with zeros and
marked in the mask as IGNORE (255).

Classes (from the render mask: red = tank instance, 1 = labelled tank; green = part id, see
blender_render_script.PART_IDS):
  0 background   everything else
  1 ground shadow   part 7: ground lying in a tank's shadow
  2 inner shadow    part 6: a floating roof lying in a shadow (the target's, and any neighbour's in view)

Output: <out>/{images,masks}/<name>.png, <out>/meta/<name>.json (crop transform, calibration and physics
for fill from a mask), <out>/{train,val,test}.txt, manifest.csv, prep_report.json, overlays/.
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

CTX = 1.5        # crop side / rim diameter
SIZE = 128
IGNORE = 255
INNER_SHADOW_PART, GROUND_SHADOW_PART = 6, 7


def find_label(src, name):
    for folder in sorted(os.listdir(os.path.join(src, "labels"))):
        p = os.path.join(src, "labels", folder, name + ".json")
        if os.path.exists(p):
            return folder, p
    raise FileNotFoundError(name)


def crop(arr, cx, cy, side, size, interp, fill):
    """Square crop of `side` native px centred on (cx, cy), resized to size x size. Returns crop, scale k,
    origin (x0, y0) with crop_px = k * (native_px - origin), and the valid-pixel mask."""
    k = size / side
    x0, y0 = cx - side / 2, cy - side / 2
    M = np.array([[k, 0, -k * x0], [0, k, -k * y0]], np.float32)
    out = cv2.warpAffine(arr, M, (size, size), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    valid = cv2.warpAffine(np.ones(arr.shape[:2], np.uint8), M, (size, size), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, k, (x0, y0), valid.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--split-dir", required=True, help="folder with train.txt / val.txt / test.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=float, default=CTX)
    ap.add_argument("--size", type=int, default=SIZE)
    a = ap.parse_args()
    for d in ("images", "masks", "meta", "overlays"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    split_of = {}
    for s in ("train", "val", "test"):
        names = [l.strip() for l in open(os.path.join(a.split_dir, f"{s}.txt")) if l.strip()]
        names = [os.path.basename(n).rsplit(".", 1)[0] if n.endswith((".png", ".jpg")) else os.path.basename(n)
                 for n in names]
        for n in names:
            assert n not in split_of, f"{n} in two splits"
            split_of[n] = s
        with open(os.path.join(a.out, f"{s}.txt"), "w") as f:
            f.write("\n".join(names) + "\n")

    rows, counts = [], {s: np.zeros(3, np.int64) for s in ("train", "val", "test")}
    n_overlay = 0
    for name, split in sorted(split_of.items()):
        folder, lp = find_label(a.src, name)
        lab = json.load(open(lp))
        img = cv2.imread(os.path.join(a.src, lab["image_metadata"]["relative_path"]), cv2.IMREAD_COLOR)
        msk = cv2.imread(os.path.join(a.src, lab["image_metadata"]["mask_relative_path"]), cv2.IMREAD_COLOR)
        assert img is not None and msk is not None, name
        inst, part = msk[..., 2], msk[..., 1]          # OpenCV loads BGR: red = channel 2
        cls = np.zeros(part.shape, np.uint8)
        cls[part == GROUND_SHADOW_PART] = 1
        cls[part == INNER_SHADOW_PART] = 2

        pt = lab["projected_truth"]
        e = pt["rim_ellipse"]
        cx, cy = e["cx_px"], e["cy_px"]
        side = a.ctx * 2 * e["a_px"]
        interp = cv2.INTER_AREA if side > a.size else cv2.INTER_LINEAR
        ci, k, (x0, y0), valid = crop(img, cx, cy, side, a.size, interp, (0, 0, 0))
        cm, _, _, _ = crop(cls, cx, cy, side, a.size, cv2.INTER_NEAREST, 0)
        cm[~valid] = IGNORE
        target_inner_native = int(((part == INNER_SHADOW_PART) & (inst == 1)).sum())
        cv2.imwrite(os.path.join(a.out, "images", name + ".png"), ci)
        cv2.imwrite(os.path.join(a.out, "masks", name + ".png"), cm)

        w, acq, ig = lab["world_truth"], lab["acquisition"], lab["image_geometry"]
        meta = {
            "name": name, "split": split, "folder": folder,
            "crop": {"k": k, "x0": x0, "y0": y0, "side_native_px": side, "size": a.size, "ctx": a.ctx},
            "rim_center_native": [cx, cy],
            "A": acq["A_jacobian"], "b": acq["b_vector"], "a_s": acq["a_s_vector"],
            "gsd_m_per_px": acq["gsd_m_per_px"], "sun_elevation_deg": acq["sun_elevation_deg"],
            "camera_elevation_deg": lab["satellite_sensor"]["camera_elevation_degrees"],
            "R_m": w["radius_m"], "r_inner_m": w["radius_inner_m"], "roof_radius_m": w["roof_radius_m"],
            "z_C_m": w["z_C_m"], "z_F_m": w["z_F_m"], "z_G_m": w["z_G_m"], "z_R_m": w["z_R_m"],
            "phi_F": w["phi_R"], "phi_R": (w["z_R_m"] - w["z_G_m"]) / (w["z_C_m"] - w["z_G_m"]),
            "roof_tilt_deg": w.get("roof_tilt_deg"), "seal_gap_m": w.get("seal_gap_m"),
            "inner_shadow_native_px": ig["internal_shadow_width_pixels"],
            "inner_shadow_crop_px": ig["internal_shadow_width_pixels"] * k,
            "rim_diam_native_px": 2 * e["a_px"], "naip_scale": lab["satellite_sensor"]["zoom"] is None,
            "atmosphere": lab["satellite_sensor"]["atmospheric_condition"],
            "location": lab["location"]["preset"], "holdout": lab.get("holdout", []),
            "target_inner_shadow_native_px": target_inner_native,
        }
        json.dump(meta, open(os.path.join(a.out, "meta", name + ".json"), "w"), indent=1)

        cnt = np.bincount(cm[cm != IGNORE].ravel(), minlength=3)[:3]
        counts[split] += cnt
        rows.append({"image": name, "split": split, "folder": folder, "phi_R": round(meta["phi_R"], 4),
                     "inner_shadow_crop_px": round(meta["inner_shadow_crop_px"], 2),
                     "px_bg": int(cnt[0]), "px_ground_shadow": int(cnt[1]), "px_inner_shadow": int(cnt[2]),
                     "ignore_px": int((cm == IGNORE).sum())})

        if n_overlay < 24 and (split != "train" or n_overlay < 12):
            ov = ci.copy()
            ov[cm == 1] = (0.5 * ov[cm == 1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            ov[cm == 2] = (0.5 * ov[cm == 2] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            both = np.concatenate([ci, ov], 1)
            cv2.imwrite(os.path.join(a.out, "overlays", f"{split}_{name}.png"),
                        cv2.resize(both, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
            n_overlay += 1

    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    rep = {"n_images": len(rows), "ctx": a.ctx, "size": a.size,
           "split_images": {s: sum(r["split"] == s for r in rows) for s in counts},
           "class_pixel_share": {s: (c / max(c.sum(), 1)).round(4).tolist() for s, c in counts.items()},
           "images_without_inner_shadow": sum(r["px_inner_shadow"] == 0 for r in rows),
           "images_with_ignore_px": sum(r["ignore_px"] > 0 for r in rows)}
    json.dump(rep, open(os.path.join(a.out, "prep_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
