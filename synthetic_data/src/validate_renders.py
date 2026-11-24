"""
Checks finished renders against their labels (run on a dataset folder after rendering; no Blender).

    python src/validate_renders.py batch_v2_1000/merged            # exits 1 if any image fails

Per image:
  1. image, mask and label exist; image and mask are 800 x 600
  2. NAIP-scale images: recorded metres per pixel >= requested (equal unless the frame had to grow)
  3. roof geometry in tank_geometry equals world_truth (seal gap, offset, tilt, edge type and height)
  4. the sunlit roof is not dark: median brightness of the labelled tank's lit roof (mask part 3) is at
     least ROOF_GROUND_MIN x the lit ground's (the old seal bug made roofs black: ratio ~0.1)
  5. keypoint curves (keypoint/geometry_kp.py) lie on the exported label boundaries (<= 0.05 px)
  6. the physics inner-shadow boundary from the label separates shadow from light in the rendered mask:
     from each boundary point, a step of SIDE_STEP_PX across the boundary (along its local normal) towards
     the rim (shadow side) must land on the labelled tank's shadow (part 6) and a step the other way on its
     lit roof (part 3). Points where a step leaves the roof (shadows thinner than the step, the near wall,
     the ladder) or where the boundary runs along the shadow direction (crescent tips, so "towards the rim"
     is undefined) cannot be tested and are skipped; >= SEPARATION_MIN of the testable points must separate correctly. Images with
     fewer than MIN_TESTABLE testable points (e.g. a sub-pixel sliver of shadow at a high sun) are counted
     as untestable, not as passed or failed.
"""
import argparse
import glob
import json
import os
import sys
import types

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "keypoint"))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))   # geometry_kp needs no OpenCV for the label checks
import geometry_kp as G  # noqa: E402

ROOF_GROUND_MIN = 0.45
SIDE_STEP_PX = 1.5
SEPARATION_MIN = 0.95
MIN_TESTABLE = 8


def lum(img):
    return img[..., :3].astype(np.float32).mean(-1) / 255.0


def shadow_separation(red, green, pts, delta_1):
    """
    (testable points, share that separate shadow from light) for the label inner-shadow boundary. The shadow
    region lies between the boundary and the rim, i.e. against the image shift delta_1 of the rim's shadow.
    """
    p = np.asarray(pts, float)
    d = np.asarray(delta_1, float)
    if len(p) < 3 or np.linalg.norm(d) < 1e-6:   # a 1-2 point sliver has no boundary direction: untestable
        return 0, None
    u = d / np.linalg.norm(d)
    h, w = red.shape
    # local boundary normal from the neighbouring points, oriented to the shadow side (against delta_1)
    tang = np.gradient(p, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    across = -(nrm @ u)                       # > 0: the normal points to the shadow side
    nrm *= np.sign(across)[:, None]
    clear = np.abs(across) >= 0.3             # boundary not running along the shadow direction

    def part(q):
        ix = np.clip(np.floor(q[:, 0]).astype(int), 0, w - 1)
        iy = np.clip(np.floor(q[:, 1]).astype(int), 0, h - 1)
        return np.where(red[iy, ix] == 1, green[iy, ix].astype(np.int16), -1)

    shadow_side, lit_side = part(p + SIDE_STEP_PX * nrm), part(p - SIDE_STEP_PX * nrm)
    roof = (3, 6)
    testable = clear & np.isin(shadow_side, roof) & np.isin(lit_side, roof)
    n = int(testable.sum())
    if n == 0:
        return 0, None
    ok = (shadow_side[testable] == 6) & (lit_side[testable] == 3)
    return n, float(ok.mean())


def check_image(root, name, v):
    errs, info = [], {}
    im_path = os.path.join(root, v["image_metadata"]["relative_path"])
    mask_path = os.path.join(root, v["image_metadata"].get("mask_relative_path") or "")
    label_path = os.path.join(root, v["image_metadata"].get("label_relative_path") or "")
    for what, p in (("image", im_path), ("mask", mask_path), ("label", label_path)):
        if not os.path.isfile(p):
            return [f"{what} missing"], info
    img, mask = np.asarray(Image.open(im_path).convert("RGB")), np.asarray(Image.open(mask_path).convert("RGB"))
    if img.shape[:2] != (600, 800) or mask.shape[:2] != (600, 800):
        errs.append(f"size image {img.shape[:2]} mask {mask.shape[:2]}")
    clean = os.path.join(root, "images_clean", os.path.relpath(im_path, os.path.join(root, "images")))
    bright = lum(np.asarray(Image.open(clean).convert("RGB")) if os.path.isfile(clean) else img)

    req = v["satellite_sensor"].get("requested_meters_per_pixel")
    mpp = v["image_geometry"]["meters_per_pixel"]
    if req and mpp < req - 1e-4:
        errs.append(f"mpp {mpp} below requested {req}")

    tg, w = v["tank_geometry"], v["world_truth"]
    pairs = (("seal_gap_meters", "seal_gap_m"), ("roof_tilt_degrees", "roof_tilt_deg"), ("edge_type", "eta_edge"),
             ("edge_height_meters", "edge_offset_m"))
    for a, b in pairs:
        if a in tg and tg[a] != w.get(b) and not (a == "roof_tilt_degrees" and not tg[a] and not w.get(b)):
            errs.append(f"{a} {tg[a]} != world {w.get(b)}")
    if "roof_offset_meters" in tg and [round(x, 6) for x in tg["roof_offset_meters"]] != \
            [round(w["roof_offset_x_m"], 6), round(w["roof_offset_y_m"], 6)]:
        errs.append("roof offset differs from world truth")

    red, green = mask[..., 0], mask[..., 1]
    lit_roof = (red == 1) & (green == 3)
    ground = (red == 0) & (green == 0)
    if lit_roof.sum() >= 20 and ground.sum() >= 1000:
        ratio = float(np.median(bright[lit_roof]) / max(np.median(bright[ground]), 1e-3))
        info["roof_ground"] = ratio
        if ratio < ROOF_GROUND_MIN:
            errs.append(f"lit roof dark: roof/ground brightness {ratio:.2f} < {ROOF_GROUND_MIN}")

    lab = json.load(open(label_path))
    _, _, _, chk = G.keypoints_from_label(lab)
    worst = max(chk.values()) if chk else 0.0
    info["kp_dev"] = worst
    if worst > G.CHECK_TOL_PX:
        errs.append(f"keypoint curve {worst:.3f} px off the label boundary")
    pt = lab["projected_truth"]
    n_test, share = shadow_separation(red, green, pt.get("inner_shadow_boundary") or [], pt["delta_1_px"])
    if n_test >= MIN_TESTABLE:
        info["shadow_separation"] = share
        if share < SEPARATION_MIN:
            errs.append(f"label inner-shadow boundary separates shadow from light at only {share:.0%} of "
                        f"{n_test} testable points")
    else:
        info["shadow_untestable"] = 1
    return errs, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dataset folder with images/, masks/, labels/")
    a = ap.parse_args()
    n = bad = 0
    stats = {"roof_ground": [], "kp_dev": [], "shadow_separation": []}
    untestable = 0
    for meta_path in sorted(glob.glob(os.path.join(a.root, "images", "*", "metadata.json"))):
        for name, v in sorted(json.load(open(meta_path)).items()):
            errs, info = check_image(a.root, name, v)
            n += 1
            for k in stats:
                if k in info:
                    stats[k].append(info[k])
            untestable += info.get("shadow_untestable", 0)
            if errs:
                bad += 1
                print(f"FAIL {name}: " + "; ".join(errs))
    for k, vals in stats.items():
        if vals:
            print(f"  {k:16s} n={len(vals):5d}  min {np.min(vals):.3f}  median {np.median(vals):.3f}  max {np.max(vals):.3f}")
    print(f"  shadow test untestable (visible shadow too thin or hidden): {untestable} images")
    print(f"\n{n - bad}/{n} images pass")
    sys.exit(1 if bad or not n else 0)


if __name__ == "__main__":
    main()
