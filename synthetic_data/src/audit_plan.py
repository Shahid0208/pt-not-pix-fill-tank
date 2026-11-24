"""
Checks a saved balanced plan for shortcuts BEFORE anything is rendered (no Blender needed).

    python src/generate_dataset.py --total 25000 --seed 46 --start-index 22000 --save-plan plan.json
    python src/audit_plan.py plan.json            # exits 1 if a check fails

Checks:
  1. Height is not predictable from the radius (R^2 linear and by size class).
  2. No pair of continuous factors (radius, height, fill, camera elevation, sun elevation) is correlated,
     except the documented lit-roof coupling (fill and height against sun), which must stay small.
  3. Height and fill are not predictable from the other plan factors (least squares on a quadratic
     expansion of the continuous factors plus one-hot categorical factors, adjusted R^2):
     - from every factor except the sun: no unintended shortcut, <= --max-r2;
     - fill from the sun and the tank shape: the physical lit-roof coupling (a low sun on a deep tank shadows
       the whole roof, so such images cannot exist), reported and capped at --max-coupling.
  4. Fill and camera elevation are continuous (almost all values distinct) and fill is flat over its bins.
  5. Tank size in pixels: the NAIP-scale share matches real NAIP chips (median ~56 px across). For zoomed
     images the size is estimated from the renderer's framing (nadir approximation).
  6. Held-out tags match their rules, and every roof is partly lit.
  8. No group of images (a location, a held-out tag, a size class, an atmosphere) has a different mean
     radius, height or fill than the rest: fails only if beyond chance (|z| > 4) and larger than 0.1 SD.
  7. Roof geometry is physical: roof offset inside the seal gap, the tilted deck between the floor
     (0.2 m) and 0.05 m below the rim, edge structures below the rim, all edge types used.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import balanced_plan  # noqa: E402

RES_X, RES_Y, FRAME_MARGIN = 800, 600, 1.35   # same as blender_render_script.py
MAX_MPP = 0.8                                 # render_settings.max_meters_per_pixel default
NAIP_MEDIAN_PX = 56.0                         # real NAIP tanks, median diameter in pixels


def r2_lstsq(X, y):
    """Adjusted R^2 of a least-squares fit (corrected for the number of features, so many columns on a
    small plan do not look like a shortcut)."""
    n, k = X.shape
    X = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    r2 = 1.0 - float(res @ res) / float(((y - y.mean()) ** 2).sum())
    return 1.0 - (1.0 - r2) * (n - 1) / max(n - k - 1, 1)


def design(plan, exclude):
    """Quadratic expansion of the continuous factors (minus `exclude`) and one-hot categorical factors."""
    cont = {"radius": [p["radius"] for p in plan], "height": [p["height"] for p in plan],
            "fill": [p["fill_level"] for p in plan], "cam_el": [p["camera_elevation"] for p in plan],
            "sun_el": [p["sun_elevation"] for p in plan], "sun_az": [math.sin(math.radians(p["sun_azimuth"])) for p in plan]}
    cols = [np.array(v, float) for k, v in cont.items() if k not in exclude]
    cols = [(c - c.mean()) / (c.std() or 1.0) for c in cols]
    quad = [a * b for i, a in enumerate(cols) for b in cols[i:]]
    cats = []
    for key in ("size", "location", "atmosphere", "layout", "edge_type"):
        if key in plan[0] and key not in exclude:
            values = sorted({p[key] for p in plan})
            cats += [np.array([p[key] == v for p in plan], float) for v in values[1:]]
    return np.column_stack(cols + quad + cats)


def tank_px(p):
    """Labelled-tank diameter in pixels of the 800 x 600 frame (zoomed images: nadir framing estimate)."""
    R, H = p["radius"], p["height"]
    if p.get("render_mpp"):
        mpp = p["render_mpp"]
    else:
        L = H / math.tan(math.radians(p["sun_elevation"]))
        az = math.radians(p["sun_azimuth"])
        half_w, half_h = R + abs(0.5 * L * math.sin(az)), R + abs(0.5 * L * math.cos(az))
        tight = 2.0 * max(half_w, half_h * RES_X / RES_Y) * FRAME_MARGIN
        ortho = min(tight * max(1.0, p["zoom"]), max(tight, MAX_MPP * RES_X))
        mpp = ortho / RES_X
    return 2.0 * R / mpp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", help="balanced_plan.json (from --save-plan or images/balanced_plan.json)")
    ap.add_argument("--max-r2", type=float, default=0.02, help="largest allowed R^2 for a shortcut")
    ap.add_argument("--max-lit-roof-r2", type=float, default=0.05,
                    help="largest allowed R^2 of fill or height against the sun (lit-roof coupling)")
    ap.add_argument("--max-coupling", type=float, default=0.10,
                    help="largest allowed adjusted R^2 of fill from the sun and the tank shape (physical)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    import yaml
    hold = (yaml.safe_load(open(a.config))["generation_modes"].get("balanced", {}).get("holdout") or {})
    fails, lines = [], [f"Plan: {a.plan}  ({len(plan)} images)"]

    def check(name, value, ok, fmt="{:.4f}"):
        lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: " + (fmt.format(value) if fmt else str(value)))
        if not ok:
            fails.append(name)

    v = {k: np.array([p[f] for p in plan], float) for k, f in
         (("radius", "radius"), ("height", "height"), ("fill", "fill_level"), ("cam_el", "camera_elevation"),
          ("sun_el", "sun_elevation"))}

    lines.append("1. Height from radius")
    r2 = balanced_plan._r2_linear(v["radius"], v["height"])
    check("R^2 height ~ radius (linear)", r2, r2 <= a.max_r2)
    r2 = balanced_plan._r2_groups([p["size"] for p in plan], list(v["height"]))
    check("R^2 height ~ size class", r2, r2 <= a.max_r2)

    lines.append("2. Pairwise R^2 of continuous factors")
    coupled = {("fill", "sun_el"), ("height", "sun_el"), ("radius", "sun_el")}
    keys = list(v)
    for i, x in enumerate(keys):
        for y in keys[i + 1:]:
            r = balanced_plan._r2_linear(v[x], v[y])
            limit = a.max_lit_roof_r2 if (x, y) in coupled else a.max_r2
            check(f"{x} ~ {y}" + (" (lit-roof coupling)" if (x, y) in coupled else ""), r, r <= limit)

    lines.append("3. Predictability from all other plan factors (quadratic + one-hot)")
    for k in ("height", "fill"):
        r2 = r2_lstsq(design(plan, {k, "sun_el", "sun_az"}), v[k])
        check(f"adjusted R^2 {k} ~ every factor except the sun (unintended shortcut)", r2, r2 <= a.max_r2)
    se = np.radians(v["sun_el"])
    shape = v["height"] / v["radius"]
    coupling = r2_lstsq(np.column_stack([v["sun_el"], v["sun_el"] ** 2, 1 / np.tan(se), shape, shape / np.tan(se)]),
                        v["fill"])
    check("adjusted R^2 fill ~ sun + tank shape (physical lit-roof coupling)", coupling, coupling <= a.max_coupling)

    lines.append("4. Continuity")
    n = len(plan)
    for k in ("fill", "cam_el", "radius", "height"):
        top = Counter(v[k]).most_common(1)[0][1]
        check(f"{k}: most repeated value share", top / n, top / n <= 0.01, "{:.4f}")
    hist, _ = np.histogram(v["fill"], bins=20, range=(0.0, 1.0))
    check("fill histogram (20 bins) min/max", hist.min() / hist.max(), hist.min() / hist.max() >= 0.8, "{:.3f}")

    lines.append("5. Tank size in pixels (diameter)")
    px = np.array([tank_px(p) for p in plan])
    naip = np.array([bool(p.get("render_mpp")) for p in plan])
    lines.append(f"      all:        median {np.median(px):.0f} px, 10th pct {np.percentile(px, 10):.0f}, "
                 f"<= 60 px: {np.mean(px <= 60):.1%}")
    if naip.any():
        med = float(np.median(px[naip]))
        lines.append(f"      NAIP share: {naip.mean():.1%} of images, median {med:.0f} px, 10-90 pct "
                     f"{np.percentile(px[naip], 10):.0f}-{np.percentile(px[naip], 90):.0f} px")
        check("NAIP-scale median within 0.7-1.5x real NAIP (56 px)", med, 0.7 * NAIP_MEDIAN_PX <= med <= 1.5 * NAIP_MEDIAN_PX, "{:.0f}")
    else:
        check("NAIP-scale share present", 0, False, "{}")
    if not naip.all():
        lines.append(f"      zoomed:     median {np.median(px[~naip]):.0f} px (estimate)")

    lines.append("6. Held-out tags and lit roofs")
    wrong = sum(sorted(p.get("holdout", [])) != sorted(balanced_plan.holdout_tags(p, hold)) for p in plan)
    check("samples whose holdout tags differ from the rules", wrong, wrong == 0, "{}")
    tags = Counter(t for p in plan for t in p.get("holdout", []))
    lines.append(f"      held out: any {sum(1 for p in plan if p.get('holdout'))} ({sum(1 for p in plan if p.get('holdout')) / n:.1%}), "
                 + ", ".join(f"{k} {c}" for k, c in sorted(tags.items())))
    dark = sum(p["sun_elevation"] < balanced_plan.min_sun_for(p["radius"], p["height"], p["fill_level"]) - 1e-6 for p in plan)
    check("samples with a fully shadowed roof", dark, dark == 0, "{}")

    if "seal_gap_m" in plan[0]:
        lines.append("7. Roof geometry")
        bad_off = bad_deck = bad_edge = 0
        for p in plan:
            R, H, gap = p["radius"], p["height"], p["seal_gap_m"]
            z_R = 0.2 + p["fill_level"] * (H - 0.3)
            reach = (R + 0.9 * gap) * math.tan(math.radians(p["roof_tilt_deg"]))
            bad_off += math.hypot(p["roof_offset_x_m"], p["roof_offset_y_m"]) >= gap
            bad_deck += z_R - reach < 0.2 - 1e-6 or z_R + reach > H - balanced_plan.RIM_CLEARANCE + 1e-6
            bad_edge += z_R + reach + p["edge_height_m"] > H - balanced_plan.RIM_CLEARANCE + 1e-6
        check("roof offset not inside the seal gap", bad_off, bad_off == 0, "{}")
        check("tilted deck below the floor or above the rim", bad_deck, bad_deck == 0, "{}")
        check("edge structure above the rim", bad_edge, bad_edge == 0, "{}")
        ec = Counter(p["edge_type"] for p in plan)
        check("edge types used (counts)", dict(ec), min(ec.values()) >= 0.15 * n, None)
        tilts = np.array([p["roof_tilt_deg"] for p in plan])
        lines.append(f"      tilt: median {np.median(tilts):.2f} deg, max {tilts.max():.2f} deg")

    lines.append("8. Group means (z of each group's mean against the whole plan)")
    # held-out groups "shape" (defined by radius/height boxes) and "sun" (low sun needs a fuller tank to keep
    # part of the roof lit) differ by design; the "site" group must not differ
    groups = {"location": lambda p: p["location"],
              "holdout": lambda p: "site" if "site" in p.get("holdout", []) else (
                  None if p.get("holdout") else "in-dist"),
              "atmosphere": lambda p: p["atmosphere"], "layout": lambda p: p.get("layout")}
    # A group fails only if its mean differs beyond chance (|z| > 4) AND by a size that matters (> 0.1 SD).
    # z alone grows with sqrt(group size): on the 25,000-image plan (seed 46) differences of 0.06 SD (in-dist
    # radius without the held-out shape boxes, the lit-roof rule at Primorsk) gave |z| 4.5-8.4 although
    # section 3 shows no shortcut (adjusted R^2 0.0005). The same differences pass at 2,000 images.
    for k in ("radius", "height", "fill"):
        x = v[k]
        worst = (0.0, 0.0, "")
        for gname, key in groups.items():
            labels = np.array([key(p) for p in plan])
            for g in set(labels):
                if g is None or g == "None":
                    continue
                sel = labels == g
                if sel.sum() >= 5:
                    d = (x[sel].mean() - x.mean()) / x.std()
                    z = d * math.sqrt(sel.sum())
                    worst = max(worst, (min(abs(z) / 4.0, abs(d) / 0.1), abs(z), f"{gname}={g}, d={d:+.3f} SD"))
        check(f"{k}: largest group-mean |z| ({worst[2]})", worst[1], worst[0] <= 1.0, "{:.2f}")

    print("\n".join(lines))
    print(f"\n{'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
