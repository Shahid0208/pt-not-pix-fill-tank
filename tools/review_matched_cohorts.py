"""
Matched-cohort fill MAE: keypoints vs segmentation on the tanks both score. CPU, saved CSVs only.

    python tools/review_matched_cohorts.py --root runs --out runs/matched_mnv3.json
    # with the true-mask checks (tools/mask_floor_check.py):
    python tools/review_matched_cohorts.py --root runs --floor runs/mask_floor_128 --floor runs/mask_floor_256 ...
    # matched ResNet-18 U-Net, 256 px:
    python tools/review_matched_cohorts.py --root runs --seg seg_r18_256 --out runs/matched_r18_256.json

Test split, per seed s0-s3 (keypoints: keypoint/<run>/calc_handoff/test/per_tank.csv; segmentation:
seg/seg_mnv3_s<k>/per_image_handoff_test.csv, which holds seg M1/M2/M3 and the true-mask M1/M2 errors;
true-mask M3 = oracle_err in seg/seg_mnv3_s<k>/per_image_test.csv). True keypoints: calc_handoff/oracle_all.
For each method: coverage and MAE over each pipeline's own responders, and MAE on tanks both pipelines score.
Also: learned keypoints vs true masks (M3, M2) on shared tanks, and why mask M2 abstains (the tanks segmentation
M2 drops but M3 scores).
"""
import argparse
import csv
import json
import os
import statistics as st

KP = "keypoint/kp_r18_s{s}/calc_handoff/{f}/per_tank.csv"
SEG = "seg/seg_mnv3_s{s}/per_image_{f}.csv"
BANDS = [(0, 1e9, "all"), (0, 6, "<6 px"), (6, 12, "6-12 px"), (12, 1e9, ">=12 px")]


def rows(p):
    return {r["image"]: r for r in csv.DictReader(open(p))}


def a(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else abs(v)


def mae(xs):
    return sum(xs) / len(xs) if xs else None


def paired(ids, ea, eb):
    both = [i for i in ids if ea(i) is not None and eb(i) is not None]
    return {"n": len(both), "a": mae([ea(i) for i in both]), "b": mae([eb(i) for i in both])}


def msd(v):
    v = [x for x in v if x is not None]
    return {"mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0, "per_seed": v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default="runs/matched_cohorts.json")
    ap.add_argument("--seg", default="seg_mnv3",
                    help="segmentation run prefix under <root>/seg (matched model: seg_r18_256)")
    ap.add_argument("--floor", action="append", default=[],
                    help="output folder of tools/mask_floor_check.py (optional, repeatable; the last one is the "
                         "'floor' of kp_vs_true_masks_floor)")
    a_ = ap.parse_args()
    R = lambda p: os.path.join(a_.root, p)  # noqa: E731
    orc = rows(R(KP.format(s=0, f="oracle_all")))
    floors = {os.path.basename(os.path.normpath(f)): rows(os.path.join(f, "per_tank.csv")) for f in a_.floor}
    floor = floors[os.path.basename(os.path.normpath(a_.floor[-1]))] if a_.floor else None
    out = {"split": "test", "seg_runs": a_.seg, "seeds": {}}
    for s in range(4):
        k = rows(R(KP.format(s=s, f="test")))
        h = rows(R(SEG.format(s=s, f="handoff_test").replace("seg_mnv3", a_.seg)))
        g = rows(R(SEG.format(s=s, f="test").replace("seg_mnv3", a_.seg)))
        assert set(k) == set(h) == set(g), s
        ids = sorted(k)
        n = len(ids)
        kp = {m: (lambda i, m=m: a(k[i][f"{m}_err_phi_R"])) for m in ("M1", "M2", "M3")}
        sg = {"M1": lambda i: a(h[i]["seg_M1_err"]), "M2": lambda i: a(h[i]["seg_M2_err"]),
              "M3": lambda i: None if g[i]["seg_M3_abstain"] == "True" else a(g[i]["seg_M3_err"])}
        tm = {"M1": lambda i: a(h[i]["oracle_M1_err"]), "M2": lambda i: a(h[i]["oracle_M2_err"]),
              "M3": lambda i: None if g[i]["oracle_abstain"] == "True" else a(g[i]["oracle_err"])}
        tk = {m: (lambda i, m=m: a(orc[i][f"{m}_err_phi_R"])) for m in ("M1", "M2", "M3")}
        d = {"n_test": n}
        for m in ("M1", "M2", "M3"):
            kv = [kp[m](i) for i in ids if kp[m](i) is not None]
            sv = [sg[m](i) for i in ids if sg[m](i) is not None]
            p = paired(ids, kp[m], sg[m])
            d[m] = {"kp": {"n": len(kv), "coverage": len(kv) / n, "MAE": mae(kv)},
                    "seg": {"n": len(sv), "coverage": len(sv) / n, "MAE": mae(sv)},
                    "matched": {"n": p["n"], "kp": p["a"], "seg": p["b"]}}
            p = paired(ids, kp[m], tm[m])
            d[m]["kp_vs_true_masks"] = {"n": p["n"], "kp": p["a"], "true_masks": p["b"]}
            if floor is not None and m == "M3":
                fl = lambda i: (None if i not in floor or floor[i]["oracle_abstain"] == "True"  # noqa: E731
                                else a(floor[i]["oracle_err"]))
                p = paired(ids, kp[m], fl)
                d[m]["kp_vs_true_masks_floor"] = {"n": p["n"], "kp": p["a"], "true_masks_floor": p["b"]}
        if floors:   # learned KP vs true masks at each floor setting, tanks all of them score, by shadow width
            fe = {k: (lambda i, F=F: None if i not in F or F[i]["oracle_abstain"] == "True"
                      else a(F[i]["oracle_err"])) for k, F in floors.items()}
            allk = [i for i in ids if kp["M3"](i) is not None and all(f(i) is not None for f in fe.values())]
            d["M3_kp_vs_floors_by_width"] = {}
            for lo, hi, lab in BANDS:
                b = [i for i in allk if lo <= float(h[i]["inner_shadow_native_px"]) < hi]
                d["M3_kp_vs_floors_by_width"][lab] = {"n": len(b), "kp": mae([kp["M3"](i) for i in b]),
                                                      **{k: mae([f(i) for i in b]) for k, f in fe.items()}}
        drop = [i for i in ids if sg["M2"](i) is None and sg["M3"](i) is not None]
        keep = [i for i in ids if sg["M2"](i) is not None]
        d["seg_M2_dropped"] = {
            "n": len(drop),
            "median_camera_elevation": st.median(float(h[i]["camera_elevation_deg"]) for i in drop),
            "median_inner_shadow_px": st.median(float(h[i]["inner_shadow_native_px"]) for i in drop),
            "kp_M2_MAE_on_dropped": mae([kp["M2"](i) for i in drop if kp["M2"](i) is not None]),
            "kept_median_camera_elevation": st.median(float(h[i]["camera_elevation_deg"]) for i in keep),
            "kept_median_inner_shadow_px": st.median(float(h[i]["inner_shadow_native_px"]) for i in keep)}
        if s == 0:
            for m in ("M3", "M2"):
                tv = [tk[m](i) for i in ids if tk[m](i) is not None]
                tmv = [tm[m](i) for i in ids if tm[m](i) is not None]
                p = paired(ids, tk[m], tm[m])
                out[f"true_inputs_{m}"] = {"true_kp": {"n": len(tv), "MAE": mae(tv)},
                                           "true_masks": {"n": len(tmv), "MAE": mae(tmv)},
                                           "matched": {"n": p["n"], "true_kp": p["a"], "true_masks": p["b"]}}
        out["seeds"][f"s{s}"] = d
    out["true_kp_all_splits"] = {}
    for m in ("M1", "M2", "M3"):
        v = [x for x in (a(r[f"{m}_err_phi_R"]) for r in orc.values()) if x is not None]
        out["true_kp_all_splits"][m] = {"n": len(v), "MAE": mae(v)}
    S = out["seeds"].values()
    out["summary"] = {
        **{f"{m}_{w}_{c}": msd([x[m][w][c] for x in S]) for m in ("M1", "M2", "M3")
           for w in ("kp", "seg") for c in ("coverage", "MAE")},
        **{f"{m}_matched_{w}": msd([x[m]["matched"][w] for x in S]) for m in ("M1", "M2", "M3")
           for w in ("kp", "seg")},
        **{f"{m}_kp_vs_true_masks_{w}": msd([x[m]["kp_vs_true_masks"][w] for x in S]) for m in ("M2", "M3")
           for w in ("kp", "true_masks")},
        **({f"M3_by_width_{lab}_{k}": msd([x["M3_kp_vs_floors_by_width"][lab][k] for x in S])
            for _, _, lab in BANDS for k in ["n", "kp", *floors]} if floors else {}),
        "M3_kp_beats_true_masks_all_seeds": all(x["M3"]["kp_vs_true_masks"]["kp"]
                                                < x["M3"]["kp_vs_true_masks"]["true_masks"] for x in S),
        "M2_matched_kp_beats_seg_all_seeds": all(x["M2"]["matched"]["kp"] < x["M2"]["matched"]["seg"]
                                                 for x in S)}
    os.makedirs(os.path.dirname(a_.out), exist_ok=True)
    json.dump(out, open(a_.out, "w"), indent=1)
    sm = out["summary"]
    f = lambda d: f"{d['mean']:.4f} +- {d['sd']:.4f}"  # noqa: E731
    for m in ("M3", "M2", "M1"):
        print(f"{m}: KP {f(sm[f'{m}_kp_MAE'])} (cov {sm[f'{m}_kp_coverage']['mean']:.3f}) | SEG "
              f"{f(sm[f'{m}_seg_MAE'])} (cov {sm[f'{m}_seg_coverage']['mean']:.3f}) | matched KP "
              f"{f(sm[f'{m}_matched_kp'])} SEG {f(sm[f'{m}_matched_seg'])}")
    for m in ("M3", "M2"):
        print(f"{m} learned KP vs true masks: {f(sm[f'{m}_kp_vs_true_masks_kp'])} vs "
              f"{f(sm[f'{m}_kp_vs_true_masks_true_masks'])}; true inputs {out[f'true_inputs_{m}']}")
    print("seg M2 dropped (s0):", out["seeds"]["s0"]["seg_M2_dropped"])
    print({k: v for k, v in sm.items() if k.endswith("seeds")})


if __name__ == "__main__":
    main()
