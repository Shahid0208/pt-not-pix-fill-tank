"""Evaluate a trained keypoint model on one split. Val and test are evaluated separately;
run test only once, after training.

    python keypoint/evaluate.py --run runs/keypoint/kp_r18_s0 --split val
    python keypoint/evaluate.py --run runs/keypoint/kp_r18_s0 --split test

Writes <run>/<split>/: metrics.json, summary.md, per_image.csv, per_keypoint.csv, k_sweep.csv,
fusion_compare.csv, envelope.csv, controls.csv, plots/, vis/.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry_kp as G  # noqa: E402
import physics_kp as P  # noqa: E402
from dataset import KeypointDataset, load_records  # noqa: E402
from model import KeypointNet  # noqa: E402
from train import predict  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
KS = (1, 2, 4, 8, 16)
SUBGROUPS = ("center", "rim", "inner_shadow", "ground_shadow", "roof_edge")


def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(x, n=1000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def nanmean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def style(ax, title, xl, yl):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xl, fontsize=9)
    ax.set_ylabel(yl, fontsize=9)
    ax.grid(True, color="#e6e6e6", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def run_model(model, root, split, cfg, device, rot_offset=0.0):
    ds = KeypointDataset(root, split, cfg, train=False, rot_offset_deg=rot_offset)
    dl = DataLoader(ds, batch_size=cfg["batch"], shuffle=False, num_workers=cfg["workers"])
    res = predict(model, dl, device, cfg)
    order = np.argsort(res["idx"])
    for k in ("kp", "prob", "v", "idx"):
        res[k] = res[k][order]
    return ds, res


def physics_rows(recs, kp_all, use_all, method="median", subset_fn=None):
    rows = []
    for i, r in enumerate(recs):
        use = use_all[i]
        sub = subset_fn(i, use) if subset_fn else None
        e = P.estimate(kp_all[i], use[G.GROUP_SLICE["inner_shadow"]], use[G.GROUP_SLICE["ground_shadow"]],
                       r["physics"], method, sub)
        rows.append(e)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--data", default=None, help="override the data root stored in the checkpoint")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out-name", default=None, help="output folder in the run (default: the split name)")
    args = ap.parse_args()

    run = Path(args.run)
    ck = torch.load(run / "weights" / args.weights, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if args.data:
        cfg["data"] = args.data
    if args.workers is not None:
        cfg["workers"] = args.workers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeypointNet(cfg["num_kp"], cfg["deconv_layers"], pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    out = run / (args.out_name or args.split)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "vis").mkdir(exist_ok=True)

    ds, res = run_model(model, cfg["data"], args.split, cfg, device)
    recs = ds.recs
    N = len(recs)
    gt = np.array([r["kp_native"] for r in recs])
    exists = np.array([r["exists"] for r in recs], bool)
    v = res["v"]
    vis = v == 2
    err = np.linalg.norm(res["kp"] - gt, axis=2)                       # (N, 65) native px
    R_px = np.array([r["physics"]["R_px"] for r in recs])
    pred_use = res["prob"] > 0.5

    # ---- 2. keypoint accuracy
    kp_metrics = {}
    for g in SUBGROUPS:
        s = G.GROUP_SLICE[g]
        e, m = err[:, s], vis[:, s]
        rr = np.repeat(R_px[:, None], s.stop - s.start, 1)
        ev = e[m]
        y, p = vis[:, s].ravel(), res["prob"][:, s].ravel()
        pr = p > 0.5
        kp_metrics[g] = {
            "n_visible": int(m.sum()), "mean_px": float(ev.mean()) if len(ev) else None,
            "median_px": float(np.median(ev)) if len(ev) else None,
            "pck_0.5px": float((ev <= 0.5).mean()) if len(ev) else None,
            "pck_1px": float((ev <= 1).mean()) if len(ev) else None,
            "pck_2px": float((ev <= 2).mean()) if len(ev) else None,
            "pck_0.05R": float((e[m] <= 0.05 * rr[m]).mean()) if len(ev) else None,
            "validity_precision": float((pr & y).sum() / max(pr.sum(), 1)),
            "validity_recall": float((pr & y).sum() / max(y.sum(), 1)),
            "validity_auroc": auroc(y, p),
        }

    # ---- 3. physics: oracle (GT keypoints, GT visibility) vs predicted (pred keypoints, pred validity)
    phys_true = np.array([P.phi_true(r["physics"]) for r in recs])   # handoff §2.1, ground-referenced
    dr_true = np.array([r["physics"]["Delta_R_m"] for r in recs])
    d1_true = np.array([r["physics"]["delta_1_mag_px"] for r in recs])
    oracle = physics_rows(recs, gt, vis)
    pred = physics_rows(recs, res["kp"], pred_use)

    def summarise(rows):
        phi = np.array([x["phi_R"] for x in rows])
        phr = np.array([x["phi_R_ratio"] for x in rows])
        d1 = np.array([x["d1_px"] for x in rows])
        dr = np.array([x["Delta_R_m"] for x in rows])
        e_phi = np.abs(phi - phys_true)
        return {
            "coverage": float(np.isfinite(phi).mean()),
            "d1_mae_px": nanmean(np.abs(d1 - d1_true)), "d1_mae_px_ci": boot_ci(np.abs(d1 - d1_true)),
            "Delta_R_mae_m": nanmean(np.abs(dr - dr_true)), "Delta_R_mae_m_ci": boot_ci(np.abs(dr - dr_true)),
            "phi_mae": nanmean(e_phi), "phi_mae_ci": boot_ci(e_phi),
            "phi_ratio_mae": nanmean(np.abs(phr - phys_true)), "phi_ratio_mae_ci": boot_ci(np.abs(phr - phys_true)),
            "coverage_at_0.05": float(np.nan_to_num(e_phi <= 0.05, nan=0).mean()),
        }, e_phi

    phys_or, _ = summarise(oracle)
    phys_pr, e_phi_pred = summarise(pred)

    # metadata-only baseline: ridge regression of phi_R on metadata, fitted on the train split only
    def feats(r):
        p = r["physics"]
        return [1.0, p["sun_elevation_deg"] / 90, p["sensor_offnadir_deg"] / 30, p["R_m"] / 40,
                p["gsd_m_per_px"], p["tan_el"] / 10]
    train_recs = load_records(cfg["data"], "train")
    X = np.array([feats(r) for r in train_recs])
    y = np.array([P.phi_true(r["physics"]) for r in train_recs])
    w = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ y)
    base = np.clip(np.array([feats(r) for r in recs]) @ w, 0, 1)
    base_err = np.abs(base - phys_true)
    phys_base = {"phi_mae": float(base_err.mean()), "phi_mae_ci": boot_ci(base_err)}

    # ---- fusion comparison
    fusion = []
    for meth in P.FUSIONS:
        fusion.append({"fusion": meth, "oracle_phi_mae": summarise(physics_rows(recs, gt, vis, meth))[0]["phi_mae"],
                       "pred_phi_mae": summarise(physics_rows(recs, res["kp"], pred_use, meth))[0]["phi_mae"]})

    # ---- 4. K-sweep: random subsets of K usable inner anchors, 20 draws each
    rng = np.random.default_rng(0)
    ksweep = []
    for K in KS:
        acc = {"oracle_phi": [], "pred_phi": [], "pred_d1": []}
        for _ in range(20):
            draws = {}

            def sub(i, use, K=K, draws=draws):
                cand = np.flatnonzero(use[G.GROUP_SLICE["inner_shadow"]])
                if len(cand) == 0:
                    return None
                key = i
                if key not in draws:
                    draws[key] = rng.choice(cand, min(K, len(cand)), replace=False)
                return draws[key]
            o = summarise(physics_rows(recs, gt, vis, "median", sub))[0]
            draws.clear()
            p = summarise(physics_rows(recs, res["kp"], pred_use, "median", sub))[0]
            acc["oracle_phi"].append(o["phi_mae"])
            acc["pred_phi"].append(p["phi_mae"])
            acc["pred_d1"].append(p["d1_mae_px"])
        ksweep.append({"K": K, **{k: float(np.nanmean(v_)) for k, v_ in acc.items()}})

    # ---- 5. operating envelope (predicted phi error)
    def bin_of(name, r):
        p = r["physics"]
        if name == "delta_1_px":
            d = p["delta_1_mag_px"]
            return "<2" if d < 2 else "2-5" if d < 5 else "5-20" if d < 20 else ">20"
        if name == "sun_elevation":
            e = p["sun_elevation_deg"]
            return next(lab for lo, hi, lab in [(0, 25, "10-25"), (25, 40, "25-40"), (40, 55, "40-55"),
                                                (55, 70, "55-70"), (70, 91, "70-90")] if lo <= e < hi)
        if name == "tank_size":
            return r["folder"]
        if name == "camera_elevation":
            return str(int(r["camera_elevation"]))
        if name == "atmosphere":
            return r["atmosphere"]
        if name == "fill":
            return f"{r['fill']:.2f}"
    envelope = []
    for name in ("delta_1_px", "sun_elevation", "tank_size", "camera_elevation", "atmosphere", "fill"):
        bins = {}
        for i, r in enumerate(recs):
            bins.setdefault(bin_of(name, r), []).append(i)
        for b in sorted(bins):
            idx = bins[b]
            envelope.append({"factor": name, "bin": b, "n": len(idx), "phi_mae": nanmean(e_phi_pred[idx]),
                             "coverage": float(np.isfinite(e_phi_pred[idx]).mean()),
                             "small_n": len(idx) < 10})

    # ---- 6. controls
    _, res_wrong = run_model(model, cfg["data"], args.split, cfg, device, rot_offset=90.0)
    wrong = physics_rows(recs, res_wrong["kp"], res_wrong["prob"] > 0.5)
    phys_wrong, _ = summarise(wrong)
    controls = [
        {"control": "none (predicted)", "phi_mae": phys_pr["phi_mae"], "coverage": phys_pr["coverage"]},
        {"control": "wrong_sun_direction_+90deg", "phi_mae": phys_wrong["phi_mae"], "coverage": phys_wrong["coverage"]},
        {"control": "metadata_only_baseline", "phi_mae": phys_base["phi_mae"], "coverage": 1.0},
    ]

    # ---- files
    metrics = {"split": args.split, "n_images": N, "weights": args.weights, "epoch": ck["epoch"],
               "keypoints": kp_metrics, "physics_oracle": phys_or, "physics_pred": phys_pr,
               "metadata_only_baseline": phys_base, "fusion": fusion, "k_sweep": ksweep,
               "envelope": envelope, "controls": controls}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)

    def write_csv(name, rows):
        with open(out / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    write_csv("fusion_compare.csv", fusion)
    write_csv("k_sweep.csv", ksweep)
    write_csv("envelope.csv", envelope)
    write_csv("controls.csv", controls)
    per_image = []
    for i, r in enumerate(recs):
        p = r["physics"]
        per_image.append({"image": r["image"], "folder": r["folder"], "fill": r["fill"],
                          "sun_elevation": p["sun_elevation_deg"], "camera_elevation": r["camera_elevation"],
                          "atmosphere": r["atmosphere"], "R_px": round(p["R_px"], 2),
                          "d1_true_px": round(d1_true[i], 4), "d1_pred_px": pred[i]["d1_px"],
                          "Delta_R_true_m": dr_true[i], "Delta_R_pred_m": pred[i]["Delta_R_m"],
                          "phi_true": phys_true[i], "phi_pred": pred[i]["phi_R"], "phi_pred_ratio": pred[i]["phi_R_ratio"],
                          "phi_abs_err": e_phi_pred[i], "n_inner_used": pred[i]["n_inner"],
                          "mean_kp_err_px": float(err[i][vis[i]].mean())})
    write_csv("per_image.csv", per_image)
    with open(out / "per_keypoint.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "kp", "group", "gt_x", "gt_y", "pred_x", "pred_y", "exists", "v", "err_px", "validity_prob"])
        for i, r in enumerate(recs):
            for k in range(G.NUM_KP):
                wr.writerow([r["image"], k, G.group_of(k), *np.round(gt[i, k], 3), *np.round(res["kp"][i, k], 3),
                             int(exists[i, k]), int(v[i, k]), round(float(err[i, k]), 4), round(float(res["prob"][i, k]), 4)])

    # ---- plots
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(KS, [r["pred_phi"] for r in ksweep], color=BLUE, lw=2, marker="o", label="predicted keypoints")
    ax.plot(KS, [r["oracle_phi"] for r in ksweep], color=ORANGE, lw=2, ls="--", marker="s", label="oracle (GT)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS, [str(k) for k in KS])
    style(ax, "phi_R error vs number of inner-shadow keypoints", "K", "phi_R MAE")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "plots" / "k_sweep.png", dpi=120)
    plt.close(fig)

    env = [e for e in envelope if e["factor"] == "delta_1_px"]
    order = ["<2", "2-5", "5-20", ">20"]
    env = sorted(env, key=lambda e: order.index(e["bin"]))
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar([e["bin"] for e in env], [e["phi_mae"] for e in env], color=BLUE, width=0.6)
    for j, e in enumerate(env):
        ax.text(j, (e["phi_mae"] if np.isfinite(e["phi_mae"]) else 0), f"n={e['n']}", ha="center", va="bottom", fontsize=8)
    style(ax, "phi_R error by inner-shadow displacement", "|delta_1| (native px)", "phi_R MAE")
    fig.tight_layout()
    fig.savefig(out / "plots" / "envelope_delta_px.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 4))
    ph = np.array([x["phi_R"] for x in pred])
    ax.plot([0, 1], [0, 1], color="#999", lw=1)
    ax.scatter(phys_true, ph, s=14, color=BLUE)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    style(ax, f"phi_R predicted vs true ({args.split})", "true phi_R", "predicted phi_R")
    fig.tight_layout()
    fig.savefig(out / "plots" / "phi_scatter.png", dpi=120)
    plt.close(fig)

    th = np.linspace(0, 10, 101)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    cols = {"center": "#555555", "rim": BLUE, "inner_shadow": ORANGE, "ground_shadow": AQUA, "roof_edge": "#eda100"}
    for g in SUBGROUPS:
        s = G.GROUP_SLICE[g]
        ev = err[:, s][vis[:, s]]
        if len(ev):
            ax.plot(th, [(ev <= t).mean() for t in th], color=cols[g], lw=2, label=g)
    style(ax, "PCK: share of keypoints within threshold", "threshold (native px)", "PCK")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "plots" / "pck_curves.png", dpi=120)
    plt.close(fig)

    # ---- visualisations: 10 best and 10 worst by mean keypoint error (drawn on the model input crop)
    mean_err = np.array([err[i][vis[i]].mean() if vis[i].any() else np.inf for i in range(N)])
    order = np.argsort(mean_err)
    picks = [("best", i) for i in order[:10]] + [("worst", i) for i in order[::-1][:10] if np.isfinite(mean_err[i])]
    for tag, i in picks:
        b = ds[i]
        img = b["img"].numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        M = b["M"].numpy()
        g_out, p_out = G.apply_affine(M, gt[i]) * 2, G.apply_affine(M, res["kp"][i]) * 2
        for k in range(G.NUM_KP):
            if vis[i, k]:
                cv2.circle(img, tuple(int(round(c)) for c in g_out[k]), 3, (0, 200, 0), -1)
            if pred_use[i, k]:
                cv2.circle(img, tuple(int(round(c)) for c in p_out[k]), 2, (0, 0, 255), -1)
        cv2.imwrite(str(out / "vis" / f"{tag}_{mean_err[i]:.2f}px_{recs[i]['image']}.png"), img)

    # ---- summary.md
    L = [f"# {args.split} results ({N} images, {args.weights}, epoch {ck['epoch']})", "",
         "| Group | n visible | mean px | median px | PCK@0.5 | PCK@1 | PCK@2 | PCK@0.05R | valid. P | valid. R | AUROC |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, n=3: "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"
    for g, m in kp_metrics.items():
        L.append(f"| {g} | {m['n_visible']} | {f(m['mean_px'], 2)} | {f(m['median_px'], 2)} | {f(m['pck_0.5px'])} | "
                 f"{f(m['pck_1px'])} | {f(m['pck_2px'])} | {f(m['pck_0.05R'])} | {f(m['validity_precision'])} | "
                 f"{f(m['validity_recall'])} | {f(m['validity_auroc'])} |")
    L += ["", "| Quantity | Oracle | Predicted | 95% CI (pred) |", "|---|---|---|---|"]
    for key, lab in [("d1_mae_px", "|delta_1| MAE (px)"), ("Delta_R_mae_m", "Delta_R MAE (m)"),
                     ("phi_mae", "phi_R MAE, metadata route"), ("phi_ratio_mae", "phi_R MAE, ratio route")]:
        ci = phys_pr[key + "_ci"]
        L.append(f"| {lab} | {f(phys_or[key], 5)} | {f(phys_pr[key], 4)} | [{f(ci[0], 4)}, {f(ci[1], 4)}] |")
    L.append(f"| Coverage (any inner kp) | {f(phys_or['coverage'])} | {f(phys_pr['coverage'])} | |")
    L.append(f"| Coverage@0.05 | {f(phys_or['coverage_at_0.05'])} | {f(phys_pr['coverage_at_0.05'])} | |")
    L.append(f"| Metadata-only baseline phi MAE | n/a | {f(phys_base['phi_mae'], 4)} | |")
    L += ["", "| K | oracle phi MAE | pred phi MAE | pred |delta_1| MAE px |", "|---|---|---|---|"]
    L += [f"| {r['K']} | {f(r['oracle_phi'], 5)} | {f(r['pred_phi'], 4)} | {f(r['pred_d1'], 3)} |" for r in ksweep]
    L += ["", "| Fusion | oracle phi MAE | pred phi MAE |", "|---|---|---|"]
    L += [f"| {r['fusion']} | {f(r['oracle_phi_mae'], 5)} | {f(r['pred_phi_mae'], 4)} |" for r in fusion]
    L += ["", "| Factor | Bin | n | phi MAE |", "|---|---|---|---|"]
    L += [f"| {e['factor']} | {e['bin']} | {e['n']}{' (n<10)' if e['small_n'] else ''} | {f(e['phi_mae'], 4)} |" for e in envelope]
    L += ["", "| Control | phi MAE | coverage |", "|---|---|---|"]
    L += [f"| {c['control']} | {f(c['phi_mae'], 4)} | {f(c['coverage'])} |" for c in controls]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
