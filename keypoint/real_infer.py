"""Run the synthetic-trained keypoint model on REAL NAIP floating-roof tanks (no fill truth). GPU.

    python keypoint/real_infer.py --run runs/keypoint/kp_r18_s0 --out runs/keypoint/kp_r18_s0/real

Real data has tank boxes, location, capture date and 0.6 m/px, but NO fill truth, NO tank height, NO time of
day (so no sun angles) and NO camera angles. Therefore:
  * shadow direction: rotation search — the model is run on crops rotated to every candidate shadow azimuth
    (10° steps, then 2° refinement) and the most confident orientation (inner + ground shadow keypoints) wins.
    Sun prior: only shadow azimuths the sun can cast on that date and place with the sun >= MIN_SUN_ELEV (NAIP is
    flown at sun >= 30°; lowered to the day's maximum - 5° if the sun never gets that high) are searched. The
    unconstrained choice is kept too (shadow_az_free, prior_override_deg).
    Validated on synthetic test tanks where the true direction is known (synthetic_rotation_check.csv).
  * sun elevation: the time of day at which the sun's azimuth (= shadow azimuth − 180°) is reached at that
    place and date (NOAA solar position); elevation at that time. Mismatch > 15° = impossible direction.
  * camera: NAIP is an orthophoto -> straight-down assumed (b = 0), so M2 = M3 here.
  * M1 (handoff §2.2 boxed): phi = 1 - d_int/d_ext                       (no metadata, no height)
    M3 (handoff §2.3, nadir): Delta_R = d_int / |a_s|, |a_s| = cot(theta) / GSD   (metres)
    tank height from the ground shadow: H_est = d_ext / |a_s|
    M3 fill with a height prior from diameter: phi_M3 = 1 - Delta_R / H_prior  (independent of the ground shadow)
    roof height z_R = phi_M1 * H_est (z_G = 0); liquid height h_L = z_R - z_F (z_F = 0.2 m);
    volume as in the synthetic generator / calc_handoff.py: V = pi R^2 h_L
No truth -> failures are flagged by internal consistency (see FLAGS). Outputs: per_tank.csv, crops/ (the model
input crop + keypoints for plotting), summary.json, synthetic_rotation_check.csv.
"""
import argparse
import csv
import glob
import json
import math
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry_kp as G  # noqa: E402
from model import KeypointNet, decode_heatmaps  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
GSD = 0.6                      # NAIP "060" = 0.6 m/px
WALL = 0.35                    # m, wall thickness assumed as in the synthetic tanks
Z_F = 0.2                      # m, floor elevation as in the synthetic tanks
EFR = "external_floating_roof_tank"
MIN_SUN_ELEV = 30.0            # NAIP acquisition rule: sun elevation >= 30°
# height prior from diameter: the synthetic tank sizes (D 20/40/60/80 m -> H 12/14/16/22 m), interpolated
H_PRIOR_D = np.array([20.0, 40.0, 60.0, 80.0])
H_PRIOR_H = np.array([12.0, 14.0, 16.0, 22.0])


# ------------------------------------------------------------------------------------------------
# NOAA solar position (copied from synthetic_data/src/geometry.py so this job is self-contained)
# ------------------------------------------------------------------------------------------------
def solar_position(latitude, longitude, dt_utc):
    t = (dt_utc.timestamp() / 86400.0 + 2440587.5 - 2451545.0) / 36525.0
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    m_r = math.radians(m)
    c = (math.sin(m_r) * (1.914602 - t * (0.004817 + 0.000014 * t)) + math.sin(2 * m_r) * (0.019993 - 0.000101 * t)
         + math.sin(3 * m_r) * 0.000289)
    omega = math.radians(125.04 - 1934.136 * t)
    app_long = math.radians(l0 + c - 0.00569 - 0.00478 * math.sin(omega))
    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    eps = math.radians(eps0 + 0.00256 * math.cos(omega))
    decl = math.asin(math.sin(eps) * math.sin(app_long))
    y = math.tan(eps / 2.0) ** 2
    l0_r = math.radians(l0)
    eot = 4.0 * math.degrees(y * math.sin(2 * l0_r) - 2 * e * math.sin(m_r) + 4 * e * y * math.sin(m_r) * math.cos(2 * l0_r)
                             - 0.5 * y * y * math.sin(4 * l0_r) - 1.25 * e * e * math.sin(2 * m_r))
    utc_min = dt_utc.hour * 60.0 + dt_utc.minute + dt_utc.second / 60.0
    ha = math.radians(((utc_min + eot + 4.0 * longitude) % 1440.0) / 4.0 - 180.0)
    lat = math.radians(latitude)
    cz = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    elev = 90.0 - math.degrees(math.acos(max(-1.0, min(1.0, cz))))
    az = (math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(lat) - math.tan(decl) * math.cos(lat))) + 180.0) % 360.0
    return elev, az


def sun_from_azimuth(lat, lon, date_str, sun_az):
    """Local mean solar time (05:00-19:00, 1-min steps) at which the sun is closest to azimuth sun_az above 5°
    elevation. Returns (elevation, local_hours, azimuth mismatch in degrees)."""
    day = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
    best = (None, None, 1e9)
    for minute in range(5 * 60, 19 * 60 + 1):
        h = minute / 60.0
        el, az = solar_position(lat, lon, day + timedelta(hours=h - lon / 15.0))
        if el < 5:
            continue
        d = abs((az - sun_az + 180) % 360 - 180)
        if d < best[2]:
            best = (el, h, d)
    return best


def reachable_shadow_az(lat, lon, date_str, min_elev=MIN_SUN_ELEV, pad=5):
    """Boolean mask over integer shadow azimuths 0..359: True if the sun, on that date and place, reaches the
    opposite azimuth (+-pad deg) while at least min_elev high (lowered to the day's maximum - 5° if needed)."""
    day = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
    pos = [solar_position(lat, lon, day + timedelta(hours=m / 60.0 - lon / 15.0)) for m in range(0, 24 * 60, 4)]
    thr = min(min_elev, max(el for el, _ in pos) - 5.0)
    mask = np.zeros(360, bool)
    for el, az in pos:
        if el >= thr:
            c = int(round(az + 180)) % 360
            mask[[(c + d) % 360 for d in range(-pad, pad + 1)]] = True
    return mask, thr


# ------------------------------------------------------------------------------------------------
def shadow_dir(az):
    """Image direction (north-up, x east, y south) of a ground azimuth (clockwise from north)."""
    a = math.radians(az)
    return np.array([math.sin(a), -math.cos(a)])


def context_quality(img, center, R_px, f=2.0):
    """Within the 4R x 4R context around the tank: share of NAIP no-data (black) pixels and share outside the chip."""
    H, W = img.shape[:2]
    h = f * R_px
    x0, x1, y0, y1 = center[0] - h, center[0] + h, center[1] - h, center[1] + h
    area = (2 * h) ** 2
    ix0, ix1, iy0, iy1 = max(int(x0), 0), min(int(np.ceil(x1)), W), max(int(y0), 0), min(int(np.ceil(y1)), H)
    sub = img[iy0:iy1, ix0:ix1]
    nodata = float((sub.max(axis=2) <= 8).sum()) / area
    outside = 1.0 - float(sub.shape[0] * sub.shape[1]) / area
    return nodata, max(outside, 0.0)


def make_input(img, center, R_px, az, size=256):
    M = G.crop_matrix(center, R_px, shadow_dir(az), 2.0, size)   # 4 R across, shadow pointing down
    s = abs(np.linalg.det(M[:, :2])) ** 0.5
    src = cv2.GaussianBlur(img, (0, 0), 0.5 * (1.0 / s - 1.0) + 1e-3) if s < 1 else img
    crop = cv2.warpAffine(src, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(114, 114, 114))
    x = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD
    return crop, M, x.transpose(2, 0, 1)


@torch.no_grad()
def run_batch(model, xs, device):
    x = torch.from_numpy(np.stack(xs)).to(device)
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        hm, val = model(x)
    hm, prob = hm.float(), torch.sigmoid(val.float())
    peak = hm.flatten(2).amax(-1)
    score = (prob * peak)[:, 17:49].mean(1)                          # inner + ground shadow confidence
    return hm.cpu().numpy(), prob.cpu().numpy(), score.cpu().numpy()


def rotation_search(model, img, center, R_px, device, allowed=None, coarse=10, fine=2):
    """Returns (constrained, free) results; free = best over all directions, constrained = best over `allowed`
    (boolean mask over integer azimuths). Without a mask both are the same."""
    azs = list(range(0, 360, coarse))
    out = [make_input(img, center, R_px, a) for a in azs]
    _, _, sc = run_batch(model, [o[2] for o in out], device)

    def refine(i, mask):
        far = [s for a, s in zip(azs, sc) if abs((a - azs[i] + 180) % 360 - 180) > 45]
        margin = float((sc[i] - max(far)) / max(sc[i], 1e-6)) if far else 0.0
        fazs = [(azs[i] + d) % 360 for d in range(-coarse + fine, coarse, fine)]
        if mask is not None:
            fazs = [a for a in fazs if mask[a]] or [azs[i]]
        o = [make_input(img, center, R_px, a) for a in fazs]
        hm, prob, sc2 = run_batch(model, [x[2] for x in o], device)
        j = int(np.argmax(sc2))
        return {"az": fazs[j], "score": float(sc2[j]), "margin": margin, "crop": o[j][0], "M": o[j][1],
                "hm": hm[j], "prob": prob[j], "coarse_scores": sc.tolist()}

    i_free = int(np.argmax(sc))
    free = refine(i_free, None)
    ok = [k for k, a in enumerate(azs) if allowed is not None and allowed[a]]
    if not ok:
        return free, free
    i_c = max(ok, key=lambda k: sc[k])
    return refine(i_c, allowed), free


def measure(kp, prob, rho):
    """d_int (inner shadow vs inner lip) and d_ext (ground shadow vs rim), px, from valid keypoints.
    rho = r_inner / R (inner lip of the wall, which casts the roof shadow)."""
    c = kp[0]
    rim = kp[G.GROUP_SLICE["rim"]]
    inner = kp[G.GROUP_SLICE["inner_shadow"]]
    ground = kp[G.GROUP_SLICE["ground_shadow"]]
    rim_in = c + rho * (rim - c)
    ui, ug = prob[17:33] > 0.5, prob[33:49] > 0.5
    d_int = float(np.median(np.linalg.norm((inner - rim_in)[ui], axis=1))) if ui.any() else np.nan
    d_ext = float(np.median(np.linalg.norm((ground - rim)[ug], axis=1))) if ug.any() else np.nan
    return d_int, d_ext, int(ui.sum()), int(ug.sum())


def load_model(run, device):
    ck = torch.load(Path(run) / "weights" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = KeypointNet(cfg["num_kp"], cfg["deconv_layers"], pretrained=False).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


# ------------------------------------------------------------------------------------------------
def synthetic_check(model, data_root, labels, device, out, limit=0, shard=0, nshards=1):
    """Rotation search on synthetic test crops whose true shadow direction is 'down' (az 180 in the crop frame).
    The sun prior is built from the scene's own place and date, rotated into the crop frame."""
    recs = [r for r in json.load(open(Path(data_root) / "keypoints.json")) if r["split"] == "test"]
    if limit:
        recs = recs[:limit]
    recs = recs[shard::nshards]
    rows = []
    for r in recs:
        img = cv2.imread(str(Path(data_root) / "crops" / f"{r['image']}.png"))
        center, R_ctx = np.array([160.0, 160.0]), 64.0
        lab = json.load(open(Path(labels) / r["folder"] / f"{r['image']}.json"))
        loc = lab["location"]
        mask_n, _ = reachable_shadow_az(loc["latitude"], loc["longitude"], loc["date"].replace("-", ""))
        true_shadow_az = (lab["solar_conditions"]["sun_azimuth_degrees"] + 180) % 360    # ground azimuth
        mask = np.roll(mask_n, int(round(180 - true_shadow_az)))                          # crop frame: truth at 180
        res, free = rotation_search(model, img, center, R_ctx, device, mask)
        err = (res["az"] - 180 + 180) % 360 - 180
        err_free = (free["az"] - 180 + 180) % 360 - 180
        kp = G.apply_affine(G.invert_affine(res["M"]), decode_heatmaps(res["hm"][None], 2.0)[0])
        rho = r["physics"]["r_inner_m"] / r["physics"]["R_m"]
        d_int, d_ext, ni, ng = measure(kp, res["prob"], rho)
        # same at the true orientation
        crop, M, x = make_input(img, center, R_ctx, 180)
        hm, prob, _ = run_batch(model, [x], device)
        kp0 = G.apply_affine(G.invert_affine(M), decode_heatmaps(hm, 2.0)[0])
        d_int0, d_ext0, _, _ = measure(kp0, prob[0], rho)
        p = r["physics"]
        rows.append({"image": r["image"], "az_error_deg": err, "az_error_free_deg": err_free,
                     "score": res["score"], "score_free": free["score"], "margin": res["margin"],
                     "true_phi": 1 - p["Delta_R_m"] / p["H_G_m"],
                     "phi_M1_searched": 1 - d_int / d_ext if np.isfinite(d_int / d_ext) else np.nan,
                     "phi_M1_true_rot": 1 - d_int0 / d_ext0 if np.isfinite(d_int0 / d_ext0) else np.nan,
                     "sun_elevation": p["sun_elevation_deg"]})
    with open(out / "synthetic_rotation_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    s = rotation_stats(rows)
    print("synthetic rotation check:", s)
    return s


def rotation_stats(rows):
    s = {"n": len(rows)}
    sun = np.array([float(r["sun_elevation"]) for r in rows])
    for tag, key in (("prior", "az_error_deg"), ("free", "az_error_free_deg")):
        e = np.array([abs(float(r[key])) for r in rows])
        for sub, m in (("all", np.ones(len(e), bool)), ("sun30_70", (sun >= 30) & (sun < 70))):
            if m.any():
                s[f"{tag}_{sub}"] = {"n": int(m.sum()), "median_abs_az_error": float(np.median(e[m])),
                                     "within_10deg": float((e[m] <= 10).mean()), "within_30deg": float((e[m] <= 30).mean()),
                                     "flipped_gt_90deg": float((e[m] > 90).mean())}
    return s


def merge_shards(out, n):
    """Combine the outputs of n shard runs (<out>/shard_<k>/) into <out>: CSVs, crops and summary."""
    shards = [out / f"shard_{k}" for k in range(n)]
    (out / "crops").mkdir(parents=True, exist_ok=True)
    for name in ("per_tank.csv", "synthetic_rotation_check.csv"):
        rows = [r for d in shards for r in csv.DictReader(open(d / name))]
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        if name == "synthetic_rotation_check.csv":
            rot = rows
        else:
            n_tanks = len(rows)
    for d in shards:
        for f in (d / "crops").glob("*.npz"):
            f.rename(out / "crops" / f.name)
    sums = [json.load(open(d / "summary.json")) for d in shards]
    skipped = {k: sum(x["skipped"][k] for x in sums) for k in sums[0]["skipped"]}
    summary = {"synthetic_rotation_check": rotation_stats(rot), "n_tanks": n_tanks, "skipped": skipped,
               "shards": n, **{k: sums[0][k] for k in ("gsd", "wall_m", "height_prior")}}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(f"merged {n} shards: {n_tanks} tanks, {len(rot)} synthetic checks -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/keypoint/kp_r18_s0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--xmls", default="data/xmls")
    ap.add_argument("--chars", default="data/image_characteristics.csv")
    ap.add_argument("--list", default="all", help="'all' = every real chip, or a split list such as yolo_data/real/test.txt")
    ap.add_argument("--splits", default="yolo_data/real", help="folder with train/val/test.txt (only to label the split)")
    ap.add_argument("--synthetic", default="keypoint_data/synthetic")
    ap.add_argument("--labels", default="synthetic_data/labels")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only N synthetic and N real images")
    ap.add_argument("--shard", type=int, default=0, help="run only every --nshards-th tank/check, starting at this one")
    ap.add_argument("--nshards", type=int, default=1, help="number of parallel shards (one per GPU)")
    ap.add_argument("--merge", type=int, default=0, help="combine N shard outputs <out>/shard_<k>/ into <out> and stop")
    a = ap.parse_args()
    out = Path(a.out or Path(a.run) / "real")
    if a.merge:
        merge_shards(out, a.merge)
        return
    (out / "crops").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.run, device)

    summary = {"synthetic_rotation_check": synthetic_check(model, a.synthetic, a.labels, device, out, a.limit,
                                                           a.shard, a.nshards)}

    chars = {}
    with open(a.chars) as f:
        for r in csv.DictReader(f):
            chars[r["image_name"]] = r
    split_of = {}
    for sp in ("train", "val", "test"):
        f = Path(a.splits) / f"{sp}.txt"
        if f.exists():
            split_of.update({Path(l.strip()).stem: sp for l in open(f) if l.strip()})
    if a.list == "all":
        names = sorted(Path(x).stem for x in glob.glob(str(Path(a.xmls) / "*.xml")))
    else:
        names = [Path(l.strip()).stem for l in open(a.list) if l.strip()]
    if a.limit:
        names = [n for n in names if (Path(a.xmls) / f"{n}.xml").exists() and EFR in (Path(a.xmls) / f"{n}.xml").read_text()][:a.limit]
    names = names[a.shard::a.nshards]
    rows, skipped = [], {"truncated": 0, "no_metadata": 0, "nodata": 0}
    for name in names:
        xml = Path(a.xmls) / f"{name}.xml"
        if not xml.exists() or EFR not in xml.read_text():
            continue
        try:
            root = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        boxes = [o for o in root.findall("object") if o.findtext("name") == EFR]
        if not boxes:
            continue
        img = cv2.imread(str(Path(a.images) / f"{name}.jpg"))
        H, W = img.shape[:2]
        date = root.findtext("capture_date")
        ch = chars.get(name)
        for bi, o in enumerate(boxes):
            bb = o.find("bndbox")
            x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            tid = f"{name}_{bi}"
            if x0 <= 1 or y0 <= 1 or x1 >= W - 2 or y1 >= H - 2:
                skipped["truncated"] += 1
                continue
            if ch is None:
                skipped["no_metadata"] += 1
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            R_px = (x1 - x0 + y1 - y0) / 4
            fx, fy = cx / W, cy / H
            lat = float(ch["nw_lat_image_coord"]) + fy * (float(ch["se_lat_image_coord"]) - float(ch["nw_lat_image_coord"]))
            lon = float(ch["nw_lon_image_coord"]) + fx * (float(ch["se_lon_image_coord"]) - float(ch["nw_lon_image_coord"]))
            allowed, thr = reachable_shadow_az(lat, lon, date)
            nodata, outside = context_quality(img, (cx, cy), R_px)
            if nodata > 0.005:                 # NAIP no-data (black) pixels in the crop: they look like shadow
                skipped["nodata"] += 1
                continue
            res, free = rotation_search(model, img, np.array([cx, cy]), R_px, device, allowed)
            kp = G.apply_affine(G.invert_affine(res["M"]), decode_heatmaps(res["hm"][None], 2.0)[0])
            R_m = R_px * GSD
            d_int, d_ext, ni, ng = measure(kp, res["prob"], max(R_m - WALL, 0.5 * R_m) / R_m)
            sun_az = (res["az"] - 180) % 360
            el, hours, mismatch = sun_from_azimuth(lat, lon, date, sun_az)
            D = 2 * R_px * GSD
            row = {"tank_id": tid, "image": name, "split": split_of.get(name, ""), "date": date, "lat": lat, "lon": lon, "R_px": R_px, "diameter_m": D,
                   "shadow_az_est": res["az"], "sun_az_est": sun_az, "rot_score": res["score"], "rot_margin": res["margin"],
                   "crop_outside_chip": outside,
                   "shadow_az_free": free["az"], "rot_score_free": free["score"],
                   "prior_override_deg": abs((res["az"] - free["az"] + 180) % 360 - 180), "min_sun_elev_used": thr,
                   "sun_elev_est": el, "local_solar_time_est": hours, "az_mismatch_deg": mismatch,
                   "n_inner": ni, "n_ground": ng, "d_int_px": d_int, "d_ext_px": d_ext}
            H_prior = float(np.interp(D, H_PRIOR_D, H_PRIOR_H))
            row["H_prior_m"] = H_prior
            if el is not None and np.isfinite(d_int):
                a_s = 1.0 / math.tan(math.radians(el)) / GSD                     # px per metre, nadir camera
                row["a_s_px_per_m"] = a_s
                row["Delta_R_m"] = d_int / a_s
                row["phi_M3_raw"] = 1 - row["Delta_R_m"] / H_prior
                row["phi_M3"] = float(np.clip(row["phi_M3_raw"], 0, 1))
                if np.isfinite(d_ext):
                    row["H_est_m"] = d_ext / a_s
            if np.isfinite(d_int) and np.isfinite(d_ext):
                row["phi_M1_raw"] = 1 - d_int / d_ext
                row["phi_M1"] = float(np.clip(row["phi_M1_raw"], 0, 1))
                if "H_est_m" in row:
                    row["z_R_m"] = row["phi_M1"] * row["H_est_m"]
                    row["h_L_m"] = max(row["z_R_m"] - Z_F, 0.0)
                    row["V_m3"] = math.pi * R_m ** 2 * row["h_L_m"]
                    row["V_bbl"] = row["V_m3"] * 6.2898
                    row["V_capacity_m3"] = math.pi * R_m ** 2 * max(row["H_est_m"] - 0.3, 0.0)
            rows.append(row)
            if len(rows) % 500 == 0:
                print(f"  {len(rows)} tanks", flush=True)
            np.savez_compressed(out / "crops" / f"{tid}.npz", crop=res["crop"], kp=G.apply_affine(res["M"], kp),
                                prob=res["prob"], M=res["M"], coarse=np.array(res["coarse_scores"]))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out / "per_tank.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary.update({"n_tanks": len(rows), "skipped": skipped, "gsd": GSD, "wall_m": WALL,
                    "height_prior": dict(zip(H_PRIOR_D.tolist(), H_PRIOR_H.tolist()))})
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(f"real: {len(rows)} tanks processed, skipped {skipped} -> {out}")


if __name__ == "__main__":
    main()

