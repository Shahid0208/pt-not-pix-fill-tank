"""Figures for real NAIP tanks: the synthetic-trained keypoint model + handoff formulas on REAL NAIP tanks.

    python keypoint/plot_real.py --real runs/keypoint/kp_r18_s0/real

Reads <real>/{per_tank.csv, summary.json, synthetic_rotation_check.csv, crops/*.npz} and data/images (chips).
Writes <real>/figures/ and <real>/failures/{failures.csv, groups.csv}.

Real data has no fill / height truth, so the synthetic error plots become consistency plots:
  * M1 (ratio 1 - d_int/d_ext, uses the ground shadow) vs M3 (sun angle + height prior, no ground shadow) agreement;
  * height from the ground shadow vs tank diameter;
  * shadow direction found by rotation search, validated on synthetic tanks (where the truth is known);
  * failures = rule-based consistency flags (FLAGS), not errors.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry_kp as G  # noqa: E402

# figure palette and style
MCOL = {"M3": "#2a78d6", "M1": "#eb6834"}
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
PRED_COL, SYN_COL, REAL_COL = "#e8344a", "#1baf7a", "#2a78d6"
KP_COL = {"rim": "#f2c14e", "inner_shadow": "#e8344a", "ground_shadow": "#2a78d6"}
GROUP_COL = {"A": "#e8344a", "B": "#a05cd6", "C": "#eb6834", "D": "#f2c14e", "E": "#2a78d6", "OK": "#1baf7a"}
DISAGREE = 0.15          # |phi_M1 - phi_M3| above this = methods disagree
H_OK = (6.0, 30.0)       # plausible floating-roof tank height, m
AZ_OK = 15.0             # sun azimuth mismatch above this = no sun position on that day casts this shadow
OVERRIDE = 45.0          # sun prior moved the model's own best direction by more than this = direction uncertain

GROUP_INFO = {
    "A": ("No shadow found", "No inner-shadow keypoint marked valid: no fill estimate (abstained)."),
    "B": ("Shadow direction uncertain",
          f"The model's own most confident direction is impossible for the sun on that date and place, and the sun "
          f"prior moved it by > {OVERRIDE:.0f}° (often a flip): the image alone does not show which way the shadow falls."),
    "C": ("No ground shadow (M1 impossible)",
          "Inner shadow found, ground shadow not: M1 and the height estimate are missing; only M3 (height prior) is left."),
    "D": ("Implausible height from the ground shadow",
          f"Height from the ground shadow outside {H_OK[0]:.0f}–{H_OK[1]:.0f} m: the ground-shadow keypoints are off "
          "(clutter, neighbouring tanks, shadow leaving the crop)."),
    "E": ("M1 and M3 disagree", f"|φ_M1 − φ_M3| > {DISAGREE} (unclipped), or one of them < −0.05 (roof shadow longer "
          "than the wall is tall): the two independent fill estimates do not agree."),
    "OK": ("Consistent", "All checks pass: valid direction, both shadows found, plausible height, M1 ≈ M3."),
}
GROUP_ORDER = ["A", "B", "C", "D", "E", "OK"]


def style(ax, title=None, xl=None, yl=None):
    if title:
        ax.set_title(title, fontsize=10, loc="left", color=INK)
    if xl:
        ax.set_xlabel(xl, fontsize=9, color=MUTED)
    if yl:
        ax.set_ylabel(yl, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(colors=MUTED, labelsize=8)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def read_rows(path):
    keep = ("tank_id", "image", "split", "date")
    with open(path) as f:
        return [{k: (v if k in keep else fnum(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def col(rows, k):
    return np.array([r.get(k, np.nan) for r in rows], float)


def save(fig, out, name):
    fig.savefig(out / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out / name)


# ------------------------------------------------------------------------------------------------
def classify(rows):
    """Rule-based consistency groups (first matching rule wins) and a severity used to sort within a group."""
    for r in rows:
        m1, m3 = r.get("phi_M1_raw", np.nan), r.get("phi_M3_raw", np.nan)
        d = abs(m1 - m3)                      # unclipped: two estimates clipped to 0 must not "agree"
        r["disagree"] = d
        H = r.get("H_est_m", np.nan)
        if not r["n_inner"] > 0:
            g, sev = "A", 1 - r["rot_score"]
        elif r["prior_override_deg"] > OVERRIDE or r["az_mismatch_deg"] > AZ_OK:
            g, sev = "B", r["prior_override_deg"]
        elif not r["n_ground"] > 0:
            g, sev = "C", 1 - r["rot_score"]
        elif not (H_OK[0] <= H <= H_OK[1]):
            g, sev = "D", abs(np.log(H / 15.0)) if H > 0 else 9.0
        elif d > DISAGREE or min(m1, m3) < -0.05:   # or a roof shadow longer than the wall is tall
            g, sev = "E", max(d, -min(m1, m3))
        else:
            g, sev = "OK", d
        r["group"], r["severity"] = g, float(sev)
    return rows


def load_crop(real, tid):
    z = np.load(real / "crops" / f"{tid}.npz")
    return z["crop"], z["kp"], z["prob"]


def draw_kp(ax, crop, kp, prob):
    ax.imshow(crop[..., ::-1])
    for grp in ("rim", "inner_shadow", "ground_shadow"):
        s = G.GROUP_SLICE[grp]
        use = prob[s] > 0.5 if grp != "rim" else np.ones(16, bool)
        ax.scatter(kp[s][use, 0], kp[s][use, 1], s=13, marker="x" if grp != "rim" else "o",
                   color=KP_COL[grp], linewidths=1.1, zorder=4)
    ax.plot(kp[0, 0], kp[0, 1], "+", color="white", ms=7, mew=1.2)
    ax.set_xticks([])
    ax.set_yticks([])


def kp_legend(fig, **kw):
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", color=KP_COL["rim"], label="rim"),
                        Line2D([], [], ls="none", marker="x", color=KP_COL["inner_shadow"], mew=1.3, label="inner shadow (on roof)"),
                        Line2D([], [], ls="none", marker="x", color=KP_COL["ground_shadow"], mew=1.3, label="ground shadow")],
               fontsize=9, frameon=False, ncol=3, **kw)


def fmt(x, f="{:.3f}"):
    return f.format(x) if np.isfinite(x) else "—"


def tank_text(r):
    return (f"φ M1 {fmt(r.get('phi_M1', np.nan))}  M3 {fmt(r.get('phi_M3', np.nan))}\n"
            f"H    {fmt(r.get('H_est_m', np.nan), '{:.1f}')} m (prior {r['H_prior_m']:.1f})\n"
            f"V    {fmt(r.get('V_m3', np.nan) / 1000, '{:.1f}')} k m³")


# ------------------------------------------------------------------------------------------------
def fig_numbers(rows, S, syn, out):
    n = len(rows)
    m1, m3 = col(rows, "phi_M1"), col(rows, "phi_M3")
    both = np.isfinite(m1) & np.isfinite(m3)
    d = np.abs(m1 - m3)[both]
    groups = {g: sum(r["group"] == g for r in rows) for g in GROUP_ORDER}
    rc = S["synthetic_rotation_check"]
    rp, rf = rc.get("prior_sun30_70", rc.get("prior_all")), rc.get("free_sun30_70", rc.get("free_all"))
    lines = [
        ("Real floating-roof tanks processed (whole tank in chip)", f"{n:,}"),
        ("Skipped: tank cut by chip border / no-data (black) pixels in crop",
         f"{S['skipped']['truncated']:,} / {S['skipped'].get('nodata', 0):,}"),
        ("Split (train / val / test chips of YOLO split)",
         " / ".join(f"{sum(r['split'] == s for r in rows):,}" for s in ("train", "val", "test"))),
        ("Median tank diameter", f"{np.nanmedian(col(rows, 'diameter_m')):.1f} m ({np.nanmedian(2 * col(rows, 'R_px')):.0f} px)"),
        ("Synthetic check, sun 30–70°: direction within ±10° (free → with sun prior)",
         f"{100 * rf['within_10deg']:.0f}% → {100 * rp['within_10deg']:.0f}%  (n={rp['n']})"),
        ("Synthetic check, sun 30–70°: flipped > 90° (free → with sun prior)",
         f"{100 * rf['flipped_gt_90deg']:.0f}% → {100 * rp['flipped_gt_90deg']:.0f}%"),
        (f"Real: model's own direction kept by the sun prior (≤ {OVERRIDE:.0f}°)", f"{100 * np.mean(col(rows, 'prior_override_deg') <= OVERRIDE):.0f}%"),
        ("Estimated sun elevation, median", f"{np.nanmedian(col(rows, 'sun_elev_est')):.0f}°"),
        ("Fill estimate: M3 / M1 / both", f"{np.isfinite(m3).mean() * 100:.0f}% / {np.isfinite(m1).mean() * 100:.0f}% / {both.mean() * 100:.0f}%"),
        ("Median fill φ: M3 / M1", f"{np.nanmedian(m3):.2f} / {np.nanmedian(m1):.2f}"),
        ("M1 vs M3: median |Δφ| / within ±0.10", f"{np.median(d):.3f} / {100 * np.mean(d <= 0.10):.0f}%" if d.size else "—"),
        ("Height from ground shadow, median (prior)", f"{np.nanmedian(col(rows, 'H_est_m')):.1f} m ({np.nanmedian(col(rows, 'H_prior_m')):.1f} m)"),
        ("Consistent tanks (all checks pass)", f"{groups['OK']:,} ({100 * groups['OK'] / max(n, 1):.0f}%)"),
    ] + [(f"Group {g}: {GROUP_INFO[g][0]}", f"{groups[g]:,}") for g in GROUP_ORDER[:-1]]
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(lines) + 0.8))
    ax.axis("off")
    t = ax.table(cellText=[[a, b] for a, b in lines], colLabels=["Real NAIP data (no fill truth)", "Value"],
                 colWidths=[0.68, 0.32], loc="upper left", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.35)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#dddddd")
        if i == 0:
            c.set_facecolor("#f2f2f2")
            c.set_text_props(weight="bold")
    ax.set_title("Synthetic-trained keypoint model on real tanks: key numbers", fontsize=11, loc="left", color=INK)
    save(fig, out, "fig1_numbers_table.png")


def fig_rotation(rows, syn, out):
    """Can the shadow direction be found without sun metadata? Synthetic check + real confidence."""
    fig, axes = plt.subplots(1, 4, figsize=(19, 3.9))
    ax = axes[0]
    sun = col(syn, "sun_elevation")
    bands = [(0, 30), (30, 50), (50, 70), (70, 91)]
    xs = np.arange(len(bands))
    for k, (key, lab, c) in enumerate((("az_error_free_deg", "free (image only)", "#999999"), ("az_error_deg", "with sun prior", SYN_COL))):
        e = np.abs(col(syn, key))
        v = [100 * np.mean(e[(sun >= lo) & (sun < hi)] <= 10) if ((sun >= lo) & (sun < hi)).any() else np.nan for lo, hi in bands]
        ax.bar(xs + (k - 0.5) * 0.38, v, 0.36, color=c, label=lab)
    ax.set_xticks(xs, [f"{lo}–{min(hi, 90)}°\nn={int(((sun >= lo) & (sun < hi)).sum())}" for lo, hi in bands], fontsize=8)
    ax.set_ylim(0, 105)
    style(ax, "Synthetic test tanks (truth known):\ndirection within ±10°, by true sun elevation", "sun elevation", "% of tanks")
    ax.set_ylim(0, 125)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    ax = axes[1]
    e = col(syn, "az_error_deg")
    m = (sun >= 30) & (sun < 70)
    ax.hist(e[m], bins=np.arange(-180, 181, 10), color=SYN_COL, edgecolor="white")
    style(ax, f"Synthetic, sun 30–70° (NAIP-like), with prior\nwithin ±10°: {100 * np.mean(np.abs(e[m]) <= 10):.0f}% · "
          f"flipped: {100 * np.mean(np.abs(e[m]) > 90):.0f}%", "found − true shadow azimuth (deg)", "tanks")
    ax = axes[2]
    ax.hist(col(syn, "score"), bins=30, color=SYN_COL, alpha=0.7, density=True, label=f"synthetic test (n={len(syn)})")
    ax.hist(col(rows, "rot_score"), bins=30, color=REAL_COL, alpha=0.6, density=True, label=f"real (n={len(rows)})")
    style(ax, "Model confidence at the chosen direction\n(mean validity × heatmap peak, shadow keypoints)", "confidence", "density")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[3]
    mm = col(rows, "prior_override_deg")
    ax.hist(mm, bins=np.arange(0, 182, 6), color=REAL_COL, edgecolor="white")
    ax.axvline(OVERRIDE, color=INK, lw=1, ls=":")
    style(ax, f"Real: how far the sun prior moved the model's choice\nkept (≤ {OVERRIDE:.0f}°): {100 * np.mean(mm <= OVERRIDE):.0f}% "
          "(larger = group B)", "free − constrained direction (deg)", "tanks")
    fig.tight_layout()
    save(fig, out, "fig2_shadow_direction.png")


def fig_sun(rows, out):
    ok = [r for r in rows if r["az_mismatch_deg"] <= AZ_OK]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    ax = axes[0]
    ax.hist(col(ok, "sun_elev_est"), bins=np.arange(10, 91, 4), color=REAL_COL, edgecolor="white")
    ax.axvline(70, color=PRED_COL, lw=1, ls="--")
    ax.text(71, ax.get_ylim()[1] * 0.9, "≥ 70°: high-error\nzone on synthetic", fontsize=7.5, color=PRED_COL, va="top")
    style(ax, "Estimated sun elevation at capture", "sun elevation (deg)", "tanks")
    ax = axes[1]
    ax.hist(col(ok, "local_solar_time_est"), bins=np.arange(6, 18.1, 0.5), color=REAL_COL, edgecolor="white")
    style(ax, "Estimated capture time (local solar time)\nfrom the shadow direction", "hour", "tanks")
    ax = axes[2]
    ax.scatter(col(ok, "lat"), col(ok, "sun_elev_est"), s=6, color=REAL_COL, alpha=0.5, edgecolor="none")
    style(ax, "Sun elevation vs latitude", "latitude (deg N)", "sun elevation (deg)")
    fig.suptitle(f"Sun position inferred from the shadow direction + date + location (tanks with a valid direction, n={len(ok)})",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out, "fig3_sun_estimate.png")


def fig_fill(rows, out):
    m1, m3 = col(rows, "phi_M1"), col(rows, "phi_M3")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    b = np.linspace(0, 1, 21)
    ax.hist(m3[np.isfinite(m3)], bins=b, color=MCOL["M3"], alpha=0.65, label=f"M3 (n={np.isfinite(m3).sum()})")
    ax.hist(m1[np.isfinite(m1)], bins=b, color=MCOL["M1"], alpha=0.55, label=f"M1 (n={np.isfinite(m1).sum()})")
    style(ax, "Estimated fill fraction φ_R", "φ_R (roof height / tank height)", "tanks")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    both = np.isfinite(m1) & np.isfinite(m3)
    ax.scatter(m3[both], m1[both], s=7, color=INK, alpha=0.35, edgecolor="none")
    ax.plot([0, 1], [0, 1], color=PRED_COL, lw=1)
    for s in (-DISAGREE, DISAGREE):
        ax.plot([0, 1], [s, 1 + s], color=PRED_COL, lw=0.8, ls=":")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    d = np.abs(m1 - m3)[both]
    style(ax, f"Agreement of the two independent methods (n={both.sum()})\nmedian |Δφ| {np.median(d):.3f} · "
          f"within ±{DISAGREE}: {100 * np.mean(d <= DISAGREE):.0f}%", "φ M3 (sun angle + height prior)", "φ M1 (shadow ratio)")
    ax = axes[2]
    ax.hist(np.clip((m1 - m3)[both], -0.6, 0.6), bins=np.linspace(-0.6, 0.6, 49), color=INK, alpha=0.75)
    ax.axvline(0, color=PRED_COL, lw=1)
    style(ax, "M1 − M3 (signed, clipped at ±0.6)", "Δφ", "tanks")
    fig.suptitle("Fill fraction on real tanks: no truth, so the two handoff formulas are compared with each other",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out, "fig4_fill_fraction.png")


def fig_height_volume(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    D, H = col(rows, "diameter_m"), col(rows, "H_est_m")
    ax = axes[0]
    f = np.isfinite(H)
    ax.scatter(D[f], np.clip(H[f], 0, 50), s=7, color=REAL_COL, alpha=0.4, edgecolor="none", label="from ground shadow")
    xs = np.linspace(10, 90, 50)
    ax.plot(xs, np.interp(xs, [20, 40, 60, 80], [12, 14, 16, 22]), color=PRED_COL, lw=1.5, label="height prior (synthetic sizes)")
    ax.axhspan(*H_OK, color="#1baf7a", alpha=0.07, label=f"plausible {H_OK[0]:.0f}–{H_OK[1]:.0f} m")
    style(ax, f"Tank height from the ground shadow\n(d_ext / |a_s|, n={f.sum()}; median {np.nanmedian(H):.1f} m)",
          "diameter from box (m)", "height (m, clipped at 50)")
    ax.legend(fontsize=7.5, frameon=False)
    ax = axes[1]
    V, C = col(rows, "V_m3"), col(rows, "V_capacity_m3")
    g = np.isfinite(V) & np.isfinite(C) & (C > 0)
    ax.scatter(C[g] / 1000, V[g] / 1000, s=7, color=INK, alpha=0.35, edgecolor="none")
    m = np.nanmax(C[g]) / 1000 if g.any() else 1
    ax.plot([0, m], [0, m], color=PRED_COL, lw=1, label="full (V = capacity)")
    ax.plot([0, m], [0, m / 2], color=PRED_COL, lw=0.8, ls=":", label="half full")
    style(ax, f"Volume vs capacity (n={g.sum()})\nV = πR²·h_L, h_L = φ_M1·H − 0.2 m", "capacity πR²(H − 0.3) (k m³)", "volume (k m³)")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[2]
    vb = col(rows, "V_bbl")
    vb = vb[np.isfinite(vb)]
    hi = np.percentile(vb, 99) / 1000 if vb.size else 1
    ax.hist(np.clip(vb / 1000, 0, hi), bins=40, color=REAL_COL, edgecolor="white")
    style(ax, f"Volume per tank (sum {vb.sum() / 1e6:.1f} M bbl over {vb.size} tanks)", "thousand barrels (clipped at 99th pct)", "tanks")
    fig.suptitle("Height and volume on real tanks (model definitions from the synthetic generator; no truth)",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out, "fig5_height_volume.png")


def fig_envelope(rows, out):
    """Synthetic: error vs sun elevation / size. Real: M1-M3 disagreement and abstention vs sun / size."""
    ok = [r for r in rows if r["az_mismatch_deg"] <= AZ_OK]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0))
    for ax, key, bins, xl in ((axes[0], "sun_elev_est", np.arange(20, 91, 10), "estimated sun elevation (deg)"),
                              (axes[1], "diameter_m", np.array([0, 15, 20, 25, 30, 40, 60, 100]), "tank diameter (m)")):
        x = col(ok, key)
        d = col(ok, "disagree")
        lab, med, q75, has, cnt = [], [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = (x >= lo) & (x < hi)
            if s.sum() < 3:
                continue
            dd = d[s][np.isfinite(d[s])]
            lab.append(f"{lo:.0f}–{hi:.0f}")
            med.append(np.median(dd) if dd.size else np.nan)
            q75.append(np.percentile(dd, 75) if dd.size else np.nan)
            has.append(dd.size / s.sum())
            cnt.append(int(s.sum()))
        xs = np.arange(len(lab))
        ax.bar(xs, med, 0.6, color=INK, alpha=0.8, label="median |φ_M1 − φ_M3|")
        ax.plot(xs, q75, "o", color=PRED_COL, ms=4, label="75th percentile")
        ax2 = ax.twinx()
        ax2.plot(xs, np.array(has) * 100, "-s", color=REAL_COL, ms=4, lw=1.2, label="% with both estimates")
        ax2.set_ylim(0, 105)
        ax2.tick_params(colors=REAL_COL, labelsize=8)
        ax2.spines["top"].set_visible(False)
        ax.set_xticks(xs, [f"{l}\nn={c}" for l, c in zip(lab, cnt)], fontsize=7.5)
        style(ax, f"Consistency vs {xl.split(' (')[0]}", xl, "|Δφ| (M1 vs M3)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    fig.suptitle("Operating envelope on real data: where the two methods agree and where estimates are missing",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out, "fig6_operating_envelope.png")


def fig_samples(rows, real, out):
    """Consistent tanks from best to worst agreement, then one of each failure group."""
    okr = sorted([r for r in rows if r["group"] == "OK"], key=lambda r: r["disagree"])
    pick = []
    if okr:
        n = len(okr)
        pick += [(okr[i], lab) for i, lab in ((0, "best agreement"), (n // 4, "consistent, 25th pct"),
                                              (n // 2, "consistent, median"), ((3 * n) // 4, "consistent, 75th pct"))]
    for g in ("B", "C", "D", "E"):
        R = sorted([r for r in rows if r["group"] == g], key=lambda r: -r["severity"])
        if R:
            pick.append((R[len(R) // 2], f"group {g}: {GROUP_INFO[g][0]}"))
    pick = pick[:8]
    fig, axes = plt.subplots(2, 4, figsize=(15, 9.4))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (r, lab) in zip(axes.ravel(), pick):
        ax.axis("on")
        crop, kp, prob = load_crop(real, r["tank_id"])
        draw_kp(ax, crop, kp, prob)
        ax.set_title(f"{lab}\nØ {r['diameter_m']:.0f} m ({2 * r['R_px']:.0f} px) · sun {fmt(r['sun_elev_est'], '{:.0f}')}° · "
                     f"{int(r['n_inner'])}/{int(r['n_ground'])} shadow kp", fontsize=8.5, loc="left", color=INK)
        ax.text(0.02, 0.02, tank_text(r), transform=ax.transAxes, fontsize=7.5, family="monospace", color=INK,
                va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
    kp_legend(fig, loc="upper right")
    fig.suptitle("Sample real tanks: shadow-aligned crops (found shadow direction points down), predicted keypoints",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.05, hspace=0.22)
    save(fig, out, "fig7_sample_tanks.png")


def fig_pipeline(rows, real, images, out):
    import cv2
    okr = sorted([r for r in rows if r["group"] == "OK" and 0.2 < r["phi_M1"] < 0.9 and r.get("crop_outside_chip", 0) == 0],
                 key=lambda r: r["disagree"])
    if not okr:
        return
    r = okr[len(okr) // 2]
    chip = cv2.imread(str(Path(images) / f"{r['image']}.jpg"))
    z = np.load(real / "crops" / f"{r['tank_id']}.npz")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    ax = axes[0]
    if chip is not None:
        ax.imshow(chip[..., ::-1])
        Minv = G.invert_affine(z["M"])
        c = G.apply_affine(Minv, np.array([[0, 0], [256, 0], [256, 256], [0, 256], [0, 0]], float))
        ax.plot(c[:, 0], c[:, 1], color=PRED_COL, lw=1.5)
        cen = G.apply_affine(Minv, z["kp"][:1])[0]
        L = 2.2 * r["R_px"]
        a = np.radians(r["shadow_az_est"])
        ax.annotate("", xy=(cen[0] + L * np.sin(a), cen[1] - L * np.cos(a)), xytext=(cen[0], cen[1]),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.6))
        ax.set_xlim(0, chip.shape[1])
        ax.set_ylim(chip.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("1. Real NAIP chip (512×512 px, 0.6 m/px)\nred = rotated crop, arrow = shadow direction found",
                 fontsize=9, loc="left", color=INK)
    draw_kp(axes[1], z["crop"], z["kp"], z["prob"])
    axes[1].set_title("2. Shadow-aligned crop + predicted keypoints\n(rotation search: most confident direction)",
                      fontsize=9, loc="left", color=INK)
    ax = axes[2]
    ax.axis("off")
    txt = (f"3. Measurement → numbers\n\n"
           f"date / place  {r['date']}  {r['lat']:.2f}N {abs(r['lon']):.2f}W\n"
           f"tank          Ø {r['diameter_m']:.1f} m ({2 * r['R_px']:.0f} px)\n"
           f"shadow az     {r['shadow_az_est']:.0f}°  → sun az {r['sun_az_est']:.0f}°\n"
           f"sun           {r['sun_elev_est']:.1f}° at {int(r['local_solar_time_est'])}:"
           f"{int(60 * (r['local_solar_time_est'] % 1)):02d} solar time\n"
           f"|a_s|         {r['a_s_px_per_m']:.3f} px per m (nadir)\n"
           f"d_int / d_ext {r['d_int_px']:.1f} / {r['d_ext_px']:.1f} px\n\n"
           f"M1  φ = 1 − d_int/d_ext   = {r['phi_M1']:.3f}\n"
           f"M3  φ = 1 − ΔR/H_prior    = {r['phi_M3']:.3f}\n"
           f"    ΔR = d_int/|a_s|      = {r['Delta_R_m']:.2f} m\n"
           f"H   = d_ext/|a_s|         = {r['H_est_m']:.1f} m (prior {r['H_prior_m']:.1f})\n"
           f"roof z_R = φ_M1·H         = {r['z_R_m']:.2f} m\n"
           f"volume πR²(z_R − 0.2)     = {r['V_m3']:,.0f} m³\n"
           f"                          = {r['V_bbl']:,.0f} bbl")
    ax.text(0.0, 0.98, txt, va="top", fontsize=8.8, family="monospace", color=INK)
    fig.suptitle(f"Pipeline example on a real tank (consistent, median agreement): {r['tank_id']}", fontsize=10.5,
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out, "fig8_pipeline_example.png")


def fig_failure_overview(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]})
    ax = axes[0]
    for g in GROUP_ORDER:
        R = [r for r in rows if r["group"] == g]
        ax.scatter(col(R, "sun_elev_est"), 2 * col(R, "R_px"), s=8, color=GROUP_COL[g], alpha=0.6, edgecolor="none",
                   label=f"{g}: {GROUP_INFO[g][0]}")
    style(ax, "Groups by estimated sun elevation and tank size", "sun elevation (deg)", "tank diameter (px)")
    ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, markerscale=2)
    ax = axes[1]
    for g in GROUP_ORDER:
        R = [r for r in rows if r["group"] == g]
        ax.hist(col(R, "rot_score"), bins=np.linspace(0, 1, 26), color=GROUP_COL[g], alpha=0.55, histtype="stepfilled")
    style(ax, "Model confidence by group", "confidence at chosen direction", "tanks")
    ax = axes[2]
    cnt = [sum(r["group"] == g for r in rows) for g in GROUP_ORDER]
    ax.barh(range(len(cnt)), cnt, color=[GROUP_COL[g] for g in GROUP_ORDER])
    for i, c in enumerate(cnt):
        ax.text(c, i, f" {c} ({100 * c / max(len(rows), 1):.0f}%)", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(cnt)), [f"{g}: {GROUP_INFO[g][0]}" for g in GROUP_ORDER], fontsize=7.5)
    ax.invert_yaxis()
    style(ax, f"Tanks per group (n={len(rows)})", "tanks", None)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, out, "overview.png")


def fig_group(g, rows, real, out, n_max=8):
    R = sorted([r for r in rows if r["group"] == g], key=lambda r: -r["severity"])
    if not R:
        return
    idx = sorted(set(np.linspace(0, len(R) - 1, min(n_max, len(R))).round().astype(int)))
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        r = R[i]
        ax.axis("on")
        crop, kp, prob = load_crop(real, r["tank_id"])
        draw_kp(ax, crop, kp, prob)
        ax.set_title(f"#{i + 1}/{len(R)} · Ø {2 * r['R_px']:.0f} px · sun {fmt(r['sun_elev_est'], '{:.0f}')}° · "
                     f"prior moved {r['prior_override_deg']:.0f}°\n{r['image'][:34]}", fontsize=8, loc="left", color=INK)
        ax.text(0.02, 0.02, tank_text(r), transform=ax.transAxes, fontsize=7.2, family="monospace", color=INK,
                va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
    kp_legend(fig, loc="lower right")
    fig.suptitle(f"Group {g}: {GROUP_INFO[g][0]} ({len(R)} tanks, worst → least severe)\n{GROUP_INFO[g][1]}",
                 fontsize=10, x=0.01, ha="left", color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.04, wspace=0.05, hspace=0.25)
    save(fig, out, f"group_{g}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="runs/keypoint/kp_r18_s0/real")
    ap.add_argument("--images", default="data/images")
    a = ap.parse_args()
    real = Path(a.real)
    out = real / "figures"
    fout = real / "failures"
    out.mkdir(exist_ok=True)
    fout.mkdir(exist_ok=True)
    rows = classify(read_rows(real / "per_tank.csv"))
    syn = read_rows(real / "synthetic_rotation_check.csv")
    S = json.load(open(real / "summary.json"))

    fig_numbers(rows, S, syn, out)
    fig_rotation(rows, syn, out)
    fig_sun(rows, out)
    fig_fill(rows, out)
    fig_height_volume(rows, out)
    fig_envelope(rows, out)
    fig_samples(rows, real, out)
    fig_pipeline(rows, real, a.images, out)
    fig_failure_overview(rows, fout)
    for g in GROUP_ORDER:
        fig_group(g, rows, real, fout)

    keys = ["tank_id", "group", "severity", "disagree", "split", "diameter_m", "R_px", "sun_elev_est", "prior_override_deg", "az_mismatch_deg",
            "rot_score", "n_inner", "n_ground", "phi_M1", "phi_M3", "H_est_m", "V_m3"]
    srt = sorted(rows, key=lambda r: (GROUP_ORDER.index(r["group"]), -r["severity"]))
    with open(fout / "failures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(srt)
    with open(fout / "groups.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "name", "tanks", "share", "rule"])
        for g in GROUP_ORDER:
            c = sum(r["group"] == g for r in rows)
            w.writerow([g, GROUP_INFO[g][0], c, round(c / max(len(rows), 1), 4), GROUP_INFO[g][1]])
    print("groups:", {g: sum(r["group"] == g for r in rows) for g in GROUP_ORDER})


if __name__ == "__main__":
    main()
