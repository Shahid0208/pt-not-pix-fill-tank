"""Fill fraction, height and volume with the formulas of the research handoff
(Sec. 3 of the paper), synthetic data only.

    python keypoint/calc_handoff.py --run runs/keypoint/kp_r18_s0 --data data/keypoint_data --labels data/SynTank-25k/labels

Notation (one meaning per symbol, as in the handoff):
  z_C  rim / caster top            z_R  floating-roof deck
  z_G  GROUND: the terrain the exterior shadow falls on (d_ext) and the reference of phi_R (handoff §2.1
       "reference ground/bottom elevation"; here ground and bottom are taken as the same plane, the ground,
       because that is the plane the boxed ratio measures against)
  z_F  tank FLOOR (lowest deck position). Not part of handoff §2; used only downstream for liquid height,
       volume and phi_F (the generator's label convention)

Handoff §2 as written:
  §2.1  phi_R = (z_R - z_G) / (z_C - z_G)  in [0, 1]              fill fraction (normalised roof state)
  §2.2  d_int = (z_C - z_R) cot(theta),  d_ext = (z_C - z_G) cot(theta)
        phi_R = 1 - d_int / d_ext                                 (boxed formula)
  §2.3  Gamma(z; xi, m) = F_phys(z; xi, m): state-to-observation compatibility

Methods (M3 is the primary method, chosen before any result was seen: the handoff prescribes the forward
model for non-ideal geometry, and the synthetic set includes tilted cameras):
  M1  §2.2 ratio          phi_R = 1 - d_int / d_ext                      no sun/camera metadata
  M2  §2.2 single shadow  z_C - z_R = d_int[m] tan(theta),  d_int[m] = d_int[px] * GSD   (ideal = nadir view)
  M3  §2.3 forward model  z_R = argmin_z sum_k Huber(| d_k - (z_C - z) a_s |)  over z in [z_G, z_C] (phi_R in [0, 1])
                          a_s = image displacement of the shadow per metre of roof depth (sun + camera metadata)

Every method's z_R is limited to [z_G, z_C], i.e. phi_R in [0, 1] as in §2.1.
Downstream (floor-referenced, not handoff §2):
  phi_F = (z_R - z_F) / (z_C - z_F)    = the generator's world_truth.phi_R (floor convention), for comparison only
  ullage = z_C - z_R;  h_L = max(z_R - z_F, 0);  fill_level = h_L / (H - 0.3);  V = pi R^2 h_L;  V_bbl = 6.2898 V.
h_L = z_R - z_F and V = pi R^2 h_L are the synthetic generator's definitions (deck on the liquid; outer
radius). They are used for synthetic evaluation only, not as real-tank or certified-inventory claims
(handoff §24 rules 2 and 3).

Static calibration (handoff H0.1): R, r_inner, H = z_C, z_F, z_G from the labels. Only the shadow is measured.
Outputs in <run>/calc_handoff/: oracle_all/, test/, val/ (per_tank.csv, summary.json, plots/), all_results.json.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry_kp as G  # noqa: E402

BBL_PER_M3 = 6.2898
METHODS = ("M3", "M1", "M2")
METHOD_NAME = {"M1": "§2.2 ratio\n1 - d_int/d_ext", "M2": "§2.2 single shadow\nd_int tan θ",
               "M3": "§2.3 forward\nmodel Γ(z)"}
QUANTS = ["phi_R", "phi_F", "z_R_m", "ullage_m", "h_L_m", "fill_level", "V_m3", "V_bbl"]
HUBER_PX = 1.0
Z_STEP = 0.005  # m, forward-model search grid


# ------------------------------------------------------------------------------------------------
# formulas
# ------------------------------------------------------------------------------------------------
def calibration_and_truth(label):
    tg, w = label["tank_geometry"], label["world_truth"]
    cal = {"R_m": w["radius_m"], "r_inner_m": w["radius_inner_m"], "H_m": tg["total_height_meters"],
           "z_C_m": w["z_C_m"], "z_F_m": w["z_F_m"], "z_G_m": w["z_G_m"]}
    z_R = w["z_R_m"]
    phi_F = (z_R - cal["z_F_m"]) / (cal["z_C_m"] - cal["z_F_m"])
    # the generator's own phi_R label is floor-referenced; make sure that is what it contains
    assert abs(phi_F - w["phi_R"]) < 1e-4, f"label phi_R {w['phi_R']} is not (z_R - z_F)/(z_C - z_F) = {phi_F}"
    truth = {"z_R_m": z_R, "phi_R": (z_R - cal["z_G_m"]) / (cal["z_C_m"] - cal["z_G_m"]),  # handoff §2.1, ground
             "phi_F": phi_F,                                                               # floor (label)
             "ullage_m": cal["z_C_m"] - z_R, "h_L_m": tg["liquid_height_meters"],
             "fill_level": tg["fill_level_percentage"] / 100.0, "V_m3": tg["volume_m3"], "V_bbl": tg["volume_barrels"]}
    return cal, truth


def from_z_R(z_R, cal):
    """Roof elevation -> all reported quantities. z_R is limited to [z_G, z_C], i.e. phi_R in [0, 1] (handoff §2.1)."""
    z_R = float(np.clip(z_R, cal["z_G_m"], cal["z_C_m"]))
    h_L = max(z_R - cal["z_F_m"], 0.0)
    V = math.pi * cal["R_m"] ** 2 * h_L
    return {"z_R_m": z_R, "phi_R": (z_R - cal["z_G_m"]) / (cal["z_C_m"] - cal["z_G_m"]),
            "phi_F": (z_R - cal["z_F_m"]) / (cal["z_C_m"] - cal["z_F_m"]),
            "ullage_m": cal["z_C_m"] - z_R, "h_L_m": h_L, "fill_level": float(np.clip(h_L / (cal["H_m"] - 0.3), 0, 1)),
            "V_m3": V, "V_bbl": V * BBL_PER_M3}


def displacements(kp, cal):
    kp = np.asarray(kp, float)
    c = kp[0]
    rim = kp[G.GROUP_SLICE["rim"]]
    inner = kp[G.GROUP_SLICE["inner_shadow"]]
    ground = kp[G.GROUP_SLICE["ground_shadow"]]
    rim_inner = c + (cal["r_inner_m"] / cal["R_m"]) * (rim - c)   # inner lip: caster of the shadow on the roof
    return inner - rim_inner, ground - rim                         # (16, 2) each, px


def forward_model_z(dvec, a_s, cal):
    """§2.3: the roof state z predicts the inner-shadow displacement Gamma(z) = (z_C - z) a_s for every
    anchor. Choose the z in [z_G, z_C] (phi_R in [0, 1]) whose prediction is most compatible with the observed displacements
    (Huber loss, robust to a few wrong keypoints). Returns (z, median residual px)."""
    zs = np.arange(cal["z_G_m"], cal["z_C_m"] + Z_STEP / 2, Z_STEP)
    pred = (cal["z_C_m"] - zs)[:, None, None] * a_s[None, None, :]          # (Z, 1, 2)
    r = np.linalg.norm(dvec[None] - pred, axis=2)                            # (Z, K)
    hub = np.where(r <= HUBER_PX, 0.5 * r ** 2, HUBER_PX * (r - 0.5 * HUBER_PX)).sum(1)
    i = int(np.argmin(hub))
    # parabolic refinement between grid points
    if 0 < i < len(zs) - 1:
        y0, y1, y2 = hub[i - 1], hub[i], hub[i + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den > 0 else 0.0
        z = zs[i] + float(np.clip(off, -1, 1)) * Z_STEP
    else:
        z = zs[i]
    return float(z), float(np.median(r[i]))


def estimate(kp, use_inner, use_ground, phys, cal):
    """All three methods for one tank. None if no inner-shadow keypoint is usable (abstain, handoff rule 9)."""
    use_inner, use_ground = np.asarray(use_inner, bool), np.asarray(use_ground, bool)
    if not use_inner.any():
        return None
    dvec, gvec = displacements(kp, cal)
    a_s = np.asarray(phys["a_s"], float)
    d_int = float(np.median(np.linalg.norm(dvec[use_inner], axis=1)))        # px
    out = {"n_inner": int(use_inner.sum()), "n_ground": int(use_ground.sum()), "d_int_px": d_int}
    z3, res = forward_model_z(dvec[use_inner], a_s, cal)
    out["M3"] = from_z_R(z3, cal)
    out["M3_residual_px"] = res
    out["M2"] = from_z_R(cal["z_C_m"] - d_int * phys["gsd_m_per_px"] * phys["tan_el"], cal)
    if use_ground.any():
        d_ext = float(np.median(np.linalg.norm(gvec[use_ground], axis=1)))
        out["d_ext_px"] = d_ext
        phi1 = 1.0 - d_int / max(d_ext, 1e-9)
        out["M1"] = from_z_R(cal["z_G_m"] + phi1 * (cal["z_C_m"] - cal["z_G_m"]), cal)
    return out


def tank_row(rec, label, kp, use_inner, use_ground):
    cal, truth = calibration_and_truth(label)
    phys = rec["physics"]
    row = {"image": rec["image"], "folder": rec["folder"], "R_m": cal["R_m"], "H_m": cal["H_m"],
           "sun_elevation": phys["sun_elevation_deg"], "camera_elevation": rec["camera_elevation"],
           "atmosphere": rec["atmosphere"], "V_capacity_m3": math.pi * cal["R_m"] ** 2 * (cal["H_m"] - 0.3)}
    row.update({f"true_{q}": truth[q] for q in QUANTS})
    e = estimate(kp, use_inner, use_ground, phys, cal)
    row["abstained"] = e is None
    if e is None:
        return row
    for k in ("n_inner", "n_ground", "d_int_px", "d_ext_px", "M3_residual_px"):
        if k in e:
            row[k] = e[k]
    for m in METHODS:
        if m in e:
            for q in QUANTS:
                row[f"{m}_{q}"] = e[m][q]
                row[f"{m}_err_{q}"] = e[m][q] - truth[q]
    return row


# ------------------------------------------------------------------------------------------------
# statistics (bootstrap over tanks, handoff §12)
# ------------------------------------------------------------------------------------------------
def boot_ci(x, n=1000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def stats(rows, m, q):
    rs = [r for r in rows if f"{m}_err_{q}" in r]
    if not rs:
        return None
    e = np.array([r[f"{m}_err_{q}"] for r in rs])
    a = np.abs(e)
    out = {"n": len(rs), "MAE": float(a.mean()), "MAE_ci": boot_ci(a), "bias": float(e.mean()),
           "RMSE": float(np.sqrt((e ** 2).mean())), "median_AE": float(np.median(a)), "max_AE": float(a.max())}
    if q in ("phi_R", "phi_F"):
        out["within_0.02"] = float((a <= 0.02).mean())
        out["within_0.05"] = float((a <= 0.05).mean())
    if q == "V_m3":
        cap = np.array([r["V_capacity_m3"] for r in rs])
        out["pct_capacity"] = float(100 * (a / cap).mean())
        out["pct_capacity_ci"] = [100 * c for c in boot_ci(a / cap)]
        vt = np.array([r["true_V_m3"] for r in rs])
        nz = vt > 0
        if nz.any():
            out["pct_true_volume_median"] = float(100 * np.median(a[nz] / vt[nz]))
    return out


def paired(rows, m_a, m_b, q="phi_R"):
    """Paired MAE difference m_a - m_b on the tanks both methods cover (handoff §12.1, §12.3)."""
    rs = [r for r in rows if f"{m_a}_err_{q}" in r and f"{m_b}_err_{q}" in r]
    if not rs:
        return None
    d = np.array([abs(r[f"{m_a}_err_{q}"]) - abs(r[f"{m_b}_err_{q}"]) for r in rs])
    return {"pair": f"{m_a} - {m_b}", "quantity": q, "n": len(rs),
            "MAE_a": float(np.mean([abs(r[f"{m_a}_err_{q}"]) for r in rs])),
            "MAE_b": float(np.mean([abs(r[f"{m_b}_err_{q}"]) for r in rs])),
            "diff": float(d.mean()), "diff_ci": boot_ci(d)}


def bin_of(name, r):
    if name == "sun_elevation":
        e = r["sun_elevation"]
        return next(lab for lo, hi, lab in [(0, 25, "10-25"), (25, 40, "25-40"), (40, 55, "40-55"),
                                            (55, 70, "55-70"), (70, 91, "70-90")] if lo <= e < hi)
    if name == "camera_elevation":
        return str(int(r["camera_elevation"]))
    return r["folder"]


def breakdown(rows, m="M3"):
    out = []
    for name in ("sun_elevation", "camera_elevation", "size"):
        bins = {}
        for r in rows:
            if not r["abstained"] and f"{m}_err_phi_R" in r:
                bins.setdefault(bin_of(name, r), []).append(r)
        for b in sorted(bins):
            rs = bins[b]
            out.append({"method": m, "factor": name, "bin": b, "n": len(rs),
                        "phi_MAE": stats(rs, m, "phi_R")["MAE"], "z_R_MAE_m": stats(rs, m, "z_R_m")["MAE"],
                        "V_MAE_m3": stats(rs, m, "V_m3")["MAE"], "V_pct_capacity": stats(rs, m, "V_m3")["pct_capacity"],
                        "small_n": len(rs) < 10})
    return out


def summarise(rows):
    s = {"n_tanks": len(rows), "n_abstained": int(sum(r["abstained"] for r in rows)),
         "coverage": float(np.mean([not r["abstained"] for r in rows]))}
    for m in METHODS:
        s[m] = {q: stats(rows, m, q) for q in QUANTS}
    s["paired"] = [p for p in (paired(rows, "M3", "M1"), paired(rows, "M3", "M2"), paired(rows, "M1", "M2")) if p]
    s["breakdown"] = breakdown(rows, "M3") + breakdown(rows, "M2")
    return s


# ------------------------------------------------------------------------------------------------
# io + plots
# ------------------------------------------------------------------------------------------------
def load_predictions(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d = out.setdefault(r["image"], {"pred": np.zeros((65, 2)), "gt": np.zeros((65, 2)),
                                            "prob": np.zeros(65), "v": np.zeros(65, int)})
            k = int(r["kp"])
            d["pred"][k] = [float(r["pred_x"]), float(r["pred_y"])]
            d["gt"][k] = [float(r["gt_x"]), float(r["gt_y"])]
            d["prob"][k] = float(r["validity_prob"])
            d["v"][k] = int(r["v"])
    return out


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


SIZE_COL = {"small_20x12": "#2a78d6", "medium_40x14": "#eb6834", "large_60x16": "#1baf7a", "extraLarge_80x22": "#eda100",
            "small_r06-12": "#2a78d6", "medium_r12-20": "#eb6834", "large_r20-30": "#1baf7a", "extraLarge_r30-45": "#eda100"}
SIZE_LAB = {"small_20x12": "20 m", "medium_40x14": "40 m", "large_60x16": "60 m", "extraLarge_80x22": "80 m", "small_r06-12": "12–24 m", "medium_r12-20": "24–40 m", "large_r20-30": "40–60 m", "extraLarge_r30-45": "60–90 m"}


def plots(rows, summ, out, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    rs = [r for r in rows if not r["abstained"]]

    def style(ax, title, xl, yl):
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.grid(True, color="#e6e6e6", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def scatter(q, label, fname, scale=1.0):
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        for size, col in SIZE_COL.items():
            sub = [r for r in rs if r["folder"] == size]
            if not sub:
                continue
            ax.scatter([r[f"true_{q}"] * scale for r in sub], [r[f"M3_{q}"] * scale for r in sub], s=16, color=col,
                       edgecolor="white", linewidth=0.4, label=SIZE_LAB[size])
        lim = max(max(r[f"true_{q}"] for r in rs), max(r[f"M3_{q}"] for r in rs)) * scale * 1.03
        ax.plot([0, lim], [0, lim], color="#999", lw=1)
        style(ax, f"{label}: M3 vs truth ({split})", f"true {label}", f"calculated {label}")
        ax.legend(fontsize=8, frameon=False, title="tank diameter", title_fontsize=8)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=120)
        plt.close(fig)

    scatter("phi_R", "fill fraction φ_R", "fill_scatter.png")
    scatter("z_R_m", "roof height z_R (m)", "height_scatter.png")
    scatter("V_m3", "volume (1000 m³)", "volume_scatter.png", 1e-3)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    v = [summ[m]["phi_R"]["MAE"] for m in METHODS]
    ax.bar(range(3), v, color="#2a78d6", width=0.6)
    for i, x in enumerate(v):
        ax.text(i, x, f"{x:.4f}\nn={summ[METHODS[i]]['phi_R']['n']}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(3), [f"{m}\n{METHOD_NAME[m]}" for m in METHODS], fontsize=8)
    ax.set_ylim(0, max(v) * 1.25)
    style(ax, f"Fill-fraction error by handoff formula ({split})", "", "φ_R MAE")
    fig.tight_layout()
    fig.savefig(out / "methods_comparison.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter([r["sun_elevation"] for r in rs], [abs(r["M3_err_phi_R"]) for r in rs], s=16, color="#2a78d6", label="M3")
    ax.scatter([r["sun_elevation"] for r in rs if "M1_err_phi_R" in r],
               [abs(r["M1_err_phi_R"]) for r in rs if "M1_err_phi_R" in r], s=12, color="#eb6834", marker="x", label="M1")
    style(ax, f"Fill-fraction error vs sun elevation ({split})", "sun elevation (deg)", "|φ_R error|")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "error_vs_sun_elevation.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    sizes = [s for s in SIZE_COL if any(r["folder"] == s for r in rs)]
    vals = [np.mean([abs(r["M3_err_V_m3"]) / r["V_capacity_m3"] * 100 for r in rs if r["folder"] == s]) for s in sizes]
    ax.bar([SIZE_LAB[s] for s in sizes], vals, color=[SIZE_COL[s] for s in sizes], width=0.6)
    for i, s in enumerate(sizes):
        ax.text(i, vals[i], f"{vals[i]:.1f}%\nn={sum(r['folder'] == s for r in rs)}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(vals) * 1.25)
    style(ax, f"Volume error as % of capacity, M3 ({split})", "tank diameter", "mean |V error| / capacity (%)")
    fig.tight_layout()
    fig.savefig(out / "volume_error_by_size.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="keypoint_data/synthetic")
    ap.add_argument("--labels", default="synthetic_data/labels")
    a = ap.parse_args()
    run, labels = Path(a.run), Path(a.labels)
    recs = {r["image"]: r for r in json.load(open(Path(a.data) / "keypoints.json"))}

    def label_of(rec):
        return json.load(open(labels / rec["folder"] / f"{rec['image']}.json"))

    root = run / "calc_handoff"
    root.mkdir(exist_ok=True)
    results = {}

    # oracle: true keypoints on all tanks (checks the formulas, no model involved)
    rows = []
    for rec in recs.values():
        ex = np.array(rec["exists"], bool) & np.array(rec["in_image"], bool)
        rows.append(tank_row(rec, label_of(rec), rec["kp_native"], ex[G.GROUP_SLICE["inner_shadow"]],
                             ex[G.GROUP_SLICE["ground_shadow"]]))
    d = root / "oracle_all"
    d.mkdir(exist_ok=True)
    write_csv(d / "per_tank.csv", rows)
    results["oracle_all"] = summarise(rows)
    json.dump(results["oracle_all"], open(d / "summary.json", "w"), indent=2)
    print("oracle_all:", len(rows), "tanks |", " ".join(f"{m} phi MAE {results['oracle_all'][m]['phi_R']['MAE']:.2e}"
                                                      for m in METHODS))

    # predicted keypoints: test and val separately (never train)
    for split in ("test", "val"):
        pred = load_predictions(run / split / "per_keypoint.csv")
        rows = []
        for img, p in pred.items():
            rec = recs[img]
            assert rec["split"] == split, f"{img} is not a {split} image"
            use = p["prob"] > 0.5
            rows.append(tank_row(rec, label_of(rec), p["pred"], use[G.GROUP_SLICE["inner_shadow"]],
                                 use[G.GROUP_SLICE["ground_shadow"]]))
        d = root / split
        d.mkdir(exist_ok=True)
        write_csv(d / "per_tank.csv", rows)
        summ = summarise(rows)
        write_csv(d / "breakdown.csv", summ["breakdown"])
        json.dump(summ, open(d / "summary.json", "w"), indent=2)
        plots(rows, summ, d / "plots", split)
        results[split] = summ
        m3 = summ["M3"]
        print(f"{split}: {len(rows)} tanks, abstained {summ['n_abstained']} | M3 phi MAE {m3['phi_R']['MAE']:.4f} "
              f"z_R MAE {m3['z_R_m']['MAE']:.3f} m, V {m3['V_m3']['pct_capacity']:.2f}% of capacity | "
              f"M1 phi {summ['M1']['phi_R']['MAE']:.4f} | M2 phi {summ['M2']['phi_R']['MAE']:.4f}")
    json.dump(results, open(root / "all_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
