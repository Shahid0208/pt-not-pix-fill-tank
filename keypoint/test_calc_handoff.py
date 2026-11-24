"""Checks for calc_handoff.py. Run before the calculation:  python keypoint/test_calc_handoff.py"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calc_handoff as C  # noqa: E402
import geometry_kp as G  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


def hand_tank(cam_tilt_b=(0.0, 0.0)):
    """GSD 0.5 m/px, sun from the south at 45 deg (shadow points north = image up), R 10, r_inner 9.65,
    H = z_C 12, z_F 0.2, z_G 0, roof 3 m below the rim (z_R = 9 m). Optional camera parallax b."""
    gsd, el, az = 0.5, 45.0, 180.0
    A = np.array([[1 / gsd, 0.0], [0.0, -1 / gsd]])
    s_hat = -np.array([math.sin(math.radians(az)), math.cos(math.radians(az))])
    a_s = A @ (s_hat / math.tan(math.radians(el))) - np.array(cam_tilt_b)
    cal = {"R_m": 10.0, "r_inner_m": 9.65, "H_m": 12.0, "z_C_m": 12.0, "z_F_m": 0.2, "z_G_m": 0.0}
    c = np.array([100.0, 100.0])
    psi = np.radians(az + np.arange(16) * 22.5)
    u = np.stack([np.sin(psi), np.cos(psi)], 1)
    rim = c + (A @ (cal["R_m"] * u).T).T
    inner = c + (A @ (cal["r_inner_m"] * u).T).T + 3.0 * a_s
    ground = rim + (cal["z_C_m"] - cal["z_G_m"]) * a_s
    kp = np.vstack([c[None], rim, inner, ground, rim])
    phys = {"a_s": a_s.tolist(), "gsd_m_per_px": gsd, "tan_el": math.tan(math.radians(el))}
    return kp, phys, cal


# 1. Hand tank, nadir camera: every handoff formula gives z_R = 9 m
kp, phys, cal = hand_tank()
e = C.estimate(kp, np.ones(16, bool), np.ones(16, bool), phys, cal)
for m in ("M1", "M2", "M3"):
    check(f"nadir hand tank: {m} z_R = 9.000 m", abs(e[m]["z_R_m"] - 9.0) < 2e-3, f"({e[m]['z_R_m']:.4f})")
check("§2.1 phi_R = (z_R - z_G)/(z_C - z_G) = 0.75", abs(e["M3"]["phi_R"] - 0.75) < 2e-4, f"({e['M3']['phi_R']:.5f})")
check("§2.2 boxed: 1 - d_int/d_ext = 0.75", abs(1 - e["d_int_px"] / e["d_ext_px"] - 0.75) < 1e-9)
check("phi_F = (z_R - z_F)/(z_C - z_F) = 8.8/11.8", abs(e["M3"]["phi_F"] - 8.8 / 11.8) < 2e-4, f"({e['M3']['phi_F']:.5f})")
check("h_L = z_R - z_F = 8.8 m; V = pi 10^2 8.8", abs(e["M3"]["h_L_m"] - 8.8) < 2e-3 and abs(e["M3"]["V_m3"] - math.pi * 100 * 8.8) < 1.0,
      f"(h_L {e['M3']['h_L_m']:.4f}, V {e['M3']['V_m3']:.1f})")
check("ullage = z_C - z_R = 3 m", abs(e["M3"]["ullage_m"] - 3.0) < 2e-3)
check("no usable inner keypoint -> abstain", C.estimate(kp, np.zeros(16, bool), np.ones(16, bool), phys, cal) is None)

# 1b. Ground vs floor: roof shadow as long as the full wall -> z_R = z_G, phi_R = 0 (not clipped at the floor);
#     h_L is never negative
kp0, phys0, cal0 = hand_tank()
deep = kp0.copy()
deep[17:33] = kp0[1:17] + (np.array(kp0[1:17]) - kp0[0]) * (cal0["r_inner_m"] / cal0["R_m"] - 1) + 12.0 * np.array(phys0["a_s"])
e0 = C.estimate(deep, np.ones(16, bool), np.ones(16, bool), phys0, cal0)
check("roof shadow = full wall: M1 phi_R = 0 (z_R = z_G)", abs(e0["M1"]["phi_R"]) < 1e-6, f"({e0['M1']['phi_R']:.5f})")
check("roof shadow = full wall: M3 phi_R = 0 (search reaches z_G)", abs(e0["M3"]["phi_R"]) < 2e-3, f"({e0['M3']['phi_R']:.5f})")
check("z_R below the floor: h_L = 0, V = 0", e0["M3"]["h_L_m"] == 0.0 and e0["M3"]["V_m3"] == 0.0)

# 2. Tilted camera (parallax b): M1 and M3 stay exact, M2 (ideal horizontal geometry) does not
kp, phys, cal = hand_tank(cam_tilt_b=(0.6, -0.4))
e = C.estimate(kp, np.ones(16, bool), np.ones(16, bool), phys, cal)
check("tilted hand tank: M1 exact", abs(e["M1"]["z_R_m"] - 9.0) < 2e-3, f"({e['M1']['z_R_m']:.4f})")
check("tilted hand tank: M3 exact", abs(e["M3"]["z_R_m"] - 9.0) < 2e-3, f"({e['M3']['z_R_m']:.4f})")
check("tilted hand tank: M2 biased (assumes nadir)", abs(e["M2"]["z_R_m"] - 9.0) > 0.05, f"({e['M2']['z_R_m']:.4f})")

# 3. Forward model is robust to wrong keypoints (Huber): corrupt 2 of 8 anchors by 15 px
kp, phys, cal = hand_tank()
bad = kp.copy()
bad[17 + 1] += [15.0, 0.0]
bad[17 + 2] += [0.0, 15.0]
use = np.zeros(16, bool)
use[:8] = True
e = C.estimate(bad, use, use, phys, cal)
check("M3 robust: 2 of 8 anchors 15 px off -> z_R within 0.1 m", abs(e["M3"]["z_R_m"] - 9.0) < 0.1, f"({e['M3']['z_R_m']:.4f})")

# 4. Real labels, true keypoints: M1 and M3 reproduce the generator truth
labels = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("synthetic_data/labels")
# the keypoint data made from these labels (prepare_synthetic_kp.py output)
recs = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "keypoint_data/synthetic/keypoints.json"))
# Level roofs (all v1 tanks; v2 tanks with tilt < 0.01 deg): the formulas assume a level deck, so true keypoints
# must reproduce the truth exactly. Tilted v2 decks: the deck height changes by R tan(tilt) from the centre to the
# edge, so the level-roof formula may miss z_R by up to that much (a real model error, reported by calc_handoff);
# it must never exceed R tan(tilt) + 0.01 m.
worst = {m: {"z_R_m": 0.0, "phi_R": 0.0, "phi_F": 0.0, "h_L_m": 0.0, "V_cap": 0.0} for m in ("M1", "M3")}
n = n_tilt = 0
tilt_excess = {m: -1.0 for m in ("M1", "M3")}
step = 15 if len(recs) > 1000 else 1
for rec in recs[::step]:
    lab = json.load(open(labels / rec["folder"] / f"{rec['image']}.json"))
    ex = np.array(rec["exists"], bool) & np.array(rec["in_image"], bool)
    row = C.tank_row(rec, lab, rec["kp_native"], ex[G.GROUP_SLICE["inner_shadow"]], ex[G.GROUP_SLICE["ground_shadow"]])
    if row["abstained"]:
        continue
    tilt = lab["world_truth"].get("roof_tilt_deg") or 0.0
    if tilt >= 0.01:
        n_tilt += 1
        bound = lab["world_truth"]["radius_m"] * math.tan(math.radians(tilt)) + 0.01
        for m in tilt_excess:
            if f"{m}_err_z_R_m" in row:
                tilt_excess[m] = max(tilt_excess[m], abs(row[f"{m}_err_z_R_m"]) - bound)
        continue
    n += 1
    for m in worst:
        if f"{m}_err_z_R_m" not in row:
            continue
        for q in ("z_R_m", "phi_R", "phi_F", "h_L_m"):
            worst[m][q] = max(worst[m][q], abs(row[f"{m}_err_{q}"]))
        worst[m]["V_cap"] = max(worst[m]["V_cap"], abs(row[f"{m}_err_V_m3"]) / row["V_capacity_m3"])
check(f"level roofs available for the exactness check ({n})", n > 0)
for m in worst:
    check(f"real labels, level roofs ({n} tanks), true keypoints: {m} matches truth (z_R, h_L < 0.01 m; V < 0.1% capacity)",
          worst[m]["z_R_m"] < 0.01 and worst[m]["h_L_m"] < 0.01 and worst[m]["phi_R"] < 1e-3 and worst[m]["phi_F"] < 1e-3 and worst[m]["V_cap"] < 1e-3,
          str({k: f"{v:.1e}" for k, v in worst[m].items()}))
if n_tilt:
    for m in tilt_excess:
        check(f"real labels, tilted roofs ({n_tilt} tanks), true keypoints: {m} z_R error <= R tan(tilt) + 0.01 m",
              tilt_excess[m] <= 0.0, f"(largest excess {tilt_excess[m]:+.4f} m)")

print("ALL PASSED" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
