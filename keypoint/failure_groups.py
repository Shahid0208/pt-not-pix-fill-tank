"""Failure analysis of the handoff calculation (M3): sort, cluster into groups, sample sheets per group.

    python keypoint/failure_groups.py --run runs/keypoint/kp_r18_s0

Failure = test or val tank with |phi_R error| > 0.02 (the 0.02 failure threshold of the paper), plus the tanks
that abstained (no usable inner-shadow keypoint). Failures are clustered with k-means (k = 4, standardised
features, fixed seed); abstained tanks form their own group. Each cluster is named from its feature means.
Writes <run>/calc_handoff/failures/: failures.csv (sorted, with group), groups.csv, overview.png, group_*.png.
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry_kp as G  # noqa: E402
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
GT_COL, PRED_COL = "#1baf7a", "#e8344a"
SIZE_LAB = {"small_r06-12": "12–24 m", "medium_r12-20": "24–40 m", "large_r20-30": "40–60 m", "extraLarge_r30-45": "60–90 m"}


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


def load_kp(path):
    """per_keypoint.csv of keypoint/evaluate.py -> image -> predicted / true keypoints, probability, visibility."""
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

TOL = 0.02
FEATURES = ["sun", "tilt", "n_inner", "log_res", "log_shadow", "err", "false_pos", "log_rim"]
GROUP_COL = {"A": "#e8344a", "B": "#eda100", "C": "#2a78d6", "D": "#1baf7a", "E": "#666666"}
PHANTOM_COL = "#eda100"


def collect(run, recs):
    rows = []
    for split in ("test", "val"):
        kerr = collections.defaultdict(lambda: collections.defaultdict(list))
        false_pos = collections.Counter()
        with open(run / split / "per_keypoint.csv") as f:
            for r in csv.DictReader(f):
                if r["v"] == "2":
                    kerr[r["image"]][r["group"]].append(float(r["err_px"]))
                elif r["group"] == "inner_shadow" and float(r["validity_prob"]) > 0.5:
                    false_pos[r["image"]] += 1       # predicted inner-shadow point where no visible true point exists
        with open(run / "calc_handoff" / split / "per_tank.csv") as f:
            for r in csv.DictReader(f):
                abst = r["abstained"] == "True"
                err = None if abst else float(r["M3_err_phi_R"])
                if not abst and abs(err) <= TOL:
                    continue
                ke = kerr[r["image"]]
                p = recs[r["image"]]["physics"]
                g = lambda k: float(r[k]) if r.get(k) not in ("", None) else np.nan  # noqa: E731
                rows.append({
                    "image": r["image"], "split": split, "abstained": abst, "err_phi": err,
                    "abs_err_phi": abs(err) if err is not None else np.nan,
                    "true_phi": g("true_phi_R"), "calc_phi": g("M3_phi_R"), "M1_phi": g("M1_phi_R"),
                    "true_z_R": g("true_z_R_m"), "calc_z_R": g("M3_z_R_m"), "err_z_R": g("M3_err_z_R_m"),
                    "true_V": g("true_V_m3"), "calc_V": g("M3_V_m3"), "V_capacity": g("V_capacity_m3"),
                    "sun": g("sun_elevation"), "camera": g("camera_elevation"), "tilt": 90 - g("camera_elevation"),
                    "size": r["folder"], "atmosphere": r["atmosphere"],
                    "n_inner": g("n_inner") if not abst else 0, "residual_px": g("M3_residual_px"),
                    "true_shadow_px": p["delta_1_mag_px"], "R_px": p["R_px"], "false_pos": false_pos[r["image"]],
                    "kerr_inner": np.mean(ke["inner_shadow"]) if ke["inner_shadow"] else np.nan,
                    "kerr_rim": np.mean(ke["rim"]) if ke["rim"] else np.nan,
                    "kerr_center": np.mean(ke["center"]) if ke["center"] else np.nan,
                })
    return rows


def cluster(rows, k=4, seed=0):
    F = [r for r in rows if not r["abstained"]]
    X = np.array([[r["sun"], r["tilt"], r["n_inner"], np.log(r["residual_px"] + 0.05), np.log(r["true_shadow_px"] + 0.5),
                   r["err_phi"], r["false_pos"], np.log(r["kerr_rim"])] for r in F])
    Z = (X - X.mean(0)) / X.std(0)
    sil = {kk: float(silhouette_score(Z, KMeans(kk, n_init=50, random_state=seed).fit(Z).labels_)) for kk in range(2, 7)}
    lab = KMeans(k, n_init=50, random_state=seed).fit(Z).labels_
    stats = {}
    for c in range(k):
        R = [r for r, l in zip(F, lab) if l == c]
        stats[c] = {f: float(np.mean([r[f] for r in R])) for f in ("sun", "tilt", "n_inner", "false_pos", "true_shadow_px", "abs_err_phi")}
    # name the clusters from their feature means (decided by the physics, not by hand-picking tanks)
    left = set(stats)
    few = min(left, key=lambda c: stats[c]["n_inner"]); left.discard(few)
    phantom = max(left, key=lambda c: stats[c]["false_pos"]); left.discard(phantom)
    high = max(left, key=lambda c: stats[c]["sun"]); left.discard(high)
    tilted = left.pop()
    name = {few: "A", phantom: "B", high: "C", tilted: "D"}
    for r, l in zip(F, lab):
        r["group"] = name[l]
    for r in rows:
        if r["abstained"]:
            r["group"] = "E"
    return sil


GROUP_INFO = {   # what each cluster is; per-run counts and errors are in groups.csv and the figures
    "A": ("Shadow hidden by the wall: model uses phantom points",
          "Few inner-shadow points in the true geometry (high sun and/or tilted camera: the near wall hides the roof "
          "shadow), but the model marks some as valid anyway; the points it uses are phantoms, so φ is off."),
    "B": ("Phantom shadow on (nearly) full tanks",
          "The true roof shadow is under a pixel (roof at the rim), but the model marks inner-shadow keypoints as "
          "valid a few px inside the rim, so the tank reads as LESS full (negative error)."),
    "C": ("High sun, short shadow: pixel-precision limit",
          "High sun, keypoints found and placed within a few px, but the shadow per metre of roof depth is so "
          "short that a pixel or two of keypoint error is a large error in φ."),
    "D": ("Tilted camera, real shadow, just over tolerance",
          "Mostly tilted cameras, long visible shadows and good keypoints; errors just above the tolerance from a "
          "small shared rim/centre offset."),
    "E": ("Shadow hidden by the wall: model abstains (correct)",
          "The roof shadow is hidden or absent and the model marks no inner-shadow keypoint valid, so it abstains."),
}


def draw(ax, img, M, k):
    ax.imshow(img[..., ::-1])
    g = G.apply_affine(M, k["gt"])
    p = G.apply_affine(M, k["pred"])
    for grp in ("rim", "inner_shadow", "ground_shadow"):
        s = G.GROUP_SLICE[grp]
        vis = k["v"][s] == 2
        use = k["prob"][s] > 0.5
        ax.scatter(g[s][vis, 0], g[s][vis, 1], s=24, facecolors="none", edgecolors=GT_COL, linewidths=1.2, zorder=3)
        ok = use & vis
        ax.scatter(p[s][ok, 0], p[s][ok, 1], s=16, marker="x", color=PRED_COL, linewidths=1.2, zorder=4)
        ph = use & ~vis                                   # predicted where no visible true point exists
        ax.scatter(p[s][ph, 0], p[s][ph, 1], s=30, marker="^", facecolors="none", edgecolors=PHANTOM_COL,
                   linewidths=1.3, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])


def sample_sheet(gid, R, kp, recs, data_root, out, n_max=8):
    n_all = len(R)
    if not n_all:   # a failure group with no member in this run: no sheet
        return
    R = sorted(R, key=lambda r: -(r["abs_err_phi"] if np.isfinite(r["abs_err_phi"]) else 1))[:n_max]
    ncol = min(4, len(R))
    nrow = int(np.ceil(len(R) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 4.5 * nrow + 0.9), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), R):
        ax.axis("on")
        rec = recs[r["image"]]
        img = cv2.imread(str(data_root / "crops" / f"{r['image']}.png"))
        draw(ax, img, np.array(rec["M_ctx"]), kp[r["split"]][r["image"]])
        err = "abstained" if r["abstained"] else f"φ err {r['err_phi']:+.3f}"
        ax.set_title(f"{r['image'].split('_')[0]} ({r['split']}) · {err}\n{SIZE_LAB[r['size']]} · sun {r['sun']:.0f}° · camera {r['camera']:.0f}° · "
                     f"{r['atmosphere'].lower()}", fontsize=8, loc="left", color=INK)
        if r["abstained"]:
            txt = f"true φ  {r['true_phi']:.3f}\nno valid inner-shadow kp\ntrue shadow {r['true_shadow_px']:.1f} px"
        else:
            txt = (f"φ   {r['true_phi']:.3f} → {r['calc_phi']:.3f}\n"
                   f"z_R {r['true_z_R']:.2f} → {r['calc_z_R']:.2f} m\n"
                   f"kp used {int(r['n_inner'])} · phantom {r['false_pos']}\n"
                   f"shadow {r['true_shadow_px']:.1f} px · res {r['residual_px']:.2f} px")
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=7, family="monospace", va="bottom", color=INK,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.9))
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", mfc="none", mec=GT_COL, mew=1.2, label="true keypoint"),
                        Line2D([], [], ls="none", marker="x", color=PRED_COL, mew=1.2, label="predicted (valid, true point exists)"),
                        Line2D([], [], ls="none", marker="^", mfc="none", mec=PHANTOM_COL, mew=1.3, label="predicted valid, but no true point (phantom)")],
               loc="lower center", fontsize=8, frameon=False, ncol=3, bbox_to_anchor=(0.5, 0.0))
    title, desc = GROUP_INFO[gid]
    fig.suptitle(f"Group {gid}: {title}  ({len(R)} of {n_all} tanks shown, worst first)",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.text(0.01, 1 - 0.55 / (4.5 * nrow + 0.9), desc, fontsize=8.5, color=MUTED, ha="left", va="top", wrap=True)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.15 / (4.5 * nrow + 0.9), bottom=0.6 / (4.5 * nrow + 0.9),
                        wspace=0.04, hspace=0.22)
    fig.savefig(out / f"group_{gid}.png", dpi=125, facecolor="white")
    plt.close(fig)
    print("wrote", out / f"group_{gid}.png")


def overview(rows, n_total, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]})
    ax = axes[0]
    for gid in "ABCD":
        R = [r for r in rows if r["group"] == gid]
        ax.scatter([r["sun"] for r in R], [r["abs_err_phi"] for r in R], s=34, color=GROUP_COL[gid], edgecolor="white",
                   linewidth=0.5, label=f"{gid}: {GROUP_INFO[gid][0][:34]}…" if len(GROUP_INFO[gid][0]) > 34 else f"{gid}: {GROUP_INFO[gid][0]}", zorder=3)
    ax.set_yscale("log")
    ax.axhline(TOL, color="#999999", lw=1, ls=":")
    style(ax, "Failures: |φ error| vs sun elevation", "sun elevation (deg)", "|φ_R error| (log)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax = axes[1]
    for gid in "ABCD":
        R = [r for r in rows if r["group"] == gid]
        ax.scatter([max(r["true_shadow_px"], 0.05) for r in R], [r["err_phi"] for r in R], s=34, color=GROUP_COL[gid],
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xscale("log")
    ax.axhline(0, color=INK, lw=1)
    style(ax, "Signed error vs true shadow length", "true inner-shadow length (px, log)", "φ_R error (calc − true)")
    ax = axes[2]
    gids = "ABCDE"
    cnt = [sum(r["group"] == g for r in rows) for g in gids]
    ax.barh(range(5), cnt, color=[GROUP_COL[g] for g in gids])
    for i, c in enumerate(cnt):
        ax.text(c + 0.2, i, str(c), va="center", fontsize=8, color=INK)
    ax.set_yticks(range(5), [f"{g}: {GROUP_INFO[g][0][:30]}" + ("…" if len(GROUP_INFO[g][0]) > 30 else "") for g in gids], fontsize=7.5)
    ax.invert_yaxis()
    style(ax, f"Tanks per group ({len(rows)} failures of {n_total})", "tanks", None)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=125, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / "overview.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/keypoint/kp_r18_s0")
    ap.add_argument("--data", default="keypoint_data/synthetic")
    a = ap.parse_args()
    run, data_root = Path(a.run), Path(a.data)
    out = run / "calc_handoff" / "failures"
    out.mkdir(parents=True, exist_ok=True)
    recs = {r["image"]: r for r in json.load(open(data_root / "keypoints.json"))}
    rows = collect(run, recs)
    sil = cluster(rows)
    n_total = sum(1 for s in ("test", "val") for _ in csv.DictReader(open(run / "calc_handoff" / s / "per_tank.csv")))
    rows.sort(key=lambda r: -(r["abs_err_phi"] if np.isfinite(r["abs_err_phi"]) else 9))
    keys = list(rows[0].keys())
    with open(out / "failures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank"] + keys)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"rank": i, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}})
    groups = []
    for gid in "ABCDE":
        R = [r for r in rows if r["group"] == gid]
        fin = [r for r in R if not r["abstained"]]
        groups.append({"group": gid, "name": GROUP_INFO[gid][0], "n": len(R),
                       "test": sum(r["split"] == "test" for r in R), "val": sum(r["split"] == "val" for r in R),
                       "mean_abs_err_phi": np.mean([r["abs_err_phi"] for r in fin]) if fin else None,
                       "mean_signed_err_phi": np.mean([r["err_phi"] for r in fin]) if fin else None,
                       "mean_abs_err_z_R_m": np.mean([abs(r["err_z_R"]) for r in fin]) if fin else None,
                       "sun_mean": np.mean([r["sun"] for r in R]), "camera_mean": np.mean([r["camera"] for r in R]),
                       "n_inner_mean": np.mean([r["n_inner"] for r in R]), "phantom_kp_mean": np.mean([r["false_pos"] for r in R]),
                       "true_shadow_px_median": float(np.median([r["true_shadow_px"] for r in R])),
                       "rim_kp_err_px_mean": float(np.nanmean([r["kerr_rim"] for r in R])),
                       "residual_px_median": float(np.nanmedian([r["residual_px"] for r in R])) if fin else None,
                       "tanks": " ".join(r["image"][:5] for r in R)})
    with open(out / "groups.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(groups[0].keys()))
        w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in g.items()} for g in groups])
    json.dump({"tolerance": TOL, "n_tanks": n_total, "n_failures": len(rows), "silhouette_by_k": sil, "k_used": 4,
               "groups": groups}, open(out / "summary.json", "w"), indent=2, default=float)
    print(f"{len(rows)} failures of {n_total} tanks; silhouette {sil}")
    for g in groups:
        print(f"  {g['group']} n={g['n']} (test {g['test']}, val {g['val']}): {g['name']} | mean |φ err| {g['mean_abs_err_phi']}")
    kp = {s: load_kp(run / s / "per_keypoint.csv") for s in ("test", "val")}
    overview(rows, n_total, out)
    for gid in "ABCDE":
        sample_sheet(gid, [r for r in rows if r["group"] == gid], kp, recs, data_root, out)


if __name__ == "__main__":
    main()
