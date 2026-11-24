"""Keypoints -> displacement d_k -> Delta_R -> phi_R, with robust fusion over anchors.

phi_R is the handoff §2.1 fill fraction, referenced to the GROUND z_G (as in calc_handoff.py):
  phi_R = (z_R - z_G)/(z_C - z_G) = 1 - Delta_R/H_G,   Delta_R = z_C - z_R,  H_G = z_C - z_G
The generator's label phys["phi_R"] is floor-referenced, 1 - Delta_R/H_F; it is reported as phi_F only.
"""
import numpy as np

import geometry_kp as G

FUSIONS = ("mean", "median", "trimmed", "huber", "ransac")


def fuse(x, method="median", ransac_tol=None):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan
    if method == "mean":
        return float(x.mean())
    if method == "median":
        return float(np.median(x))
    if method == "trimmed":
        k = int(0.2 * len(x))
        xs = np.sort(x)
        return float(xs[k:len(x) - k].mean()) if len(x) - 2 * k > 0 else float(np.median(x))
    if method == "huber":
        mu = np.median(x)
        s = 1.4826 * np.median(np.abs(x - mu)) + 1e-9
        for _ in range(20):
            r = (x - mu) / (1.345 * s)
            w = np.where(np.abs(r) <= 1, 1.0, 1.0 / np.maximum(np.abs(r), 1e-12))
            mu = float((w * x).sum() / w.sum())
        return mu
    if method == "ransac":
        tol = ransac_tol if ransac_tol is not None else 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-9
        best = max(range(len(x)), key=lambda i: (np.abs(x - x[i]) <= tol).sum())
        return float(x[np.abs(x - x[best]) <= tol].mean())
    raise ValueError(method)


def phi_true(phys):
    """Ground truth handoff §2.1 phi_R (ground-referenced) from the label's Delta_R and H_G."""
    return 1.0 - phys["Delta_R_m"] / phys["H_G_m"]


def anchor_displacements(kp, phys):
    """Per anchor: inner-shadow displacement along the shadow direction (px), and the inner and ground
    displacement magnitudes (px) used by the metadata-free ratio route."""
    kp = np.asarray(kp, float)
    c = kp[0]
    rim = kp[G.GROUP_SLICE["rim"]]
    inner = kp[G.GROUP_SLICE["inner_shadow"]]
    ground = kp[G.GROUP_SLICE["ground_shadow"]]
    rim_in = c + (phys["r_inner_m"] / phys["R_m"]) * (rim - c)   # inner lip from the static radii
    a = np.asarray(phys["a_s"], float)
    a_hat = a / max(np.linalg.norm(a), 1e-12)
    d = inner - rim_in
    return d @ a_hat, np.linalg.norm(d, axis=1), np.linalg.norm(ground - rim, axis=1)


def estimate(kp, use_inner, use_ground, phys, method="median", subset=None):
    """Return dict with d1_px (fused |delta_1|), Delta_R_m, phi_R (metadata route, 1 - Delta_R/H_G),
    phi_R_ratio (handoff §2.2 boxed, 1 - d_int/d_ext) and phi_F (metadata route, floor-referenced)."""
    proj, mag_in, mag_gd = anchor_displacements(kp, phys)
    use_inner = np.asarray(use_inner, bool).copy()
    if subset is not None:
        keep = np.zeros(G.NK, bool)
        keep[subset] = True
        use_inner &= keep
    out = {"n_inner": int(use_inner.sum()), "d1_px": np.nan, "Delta_R_m": np.nan, "phi_R": np.nan,
           "phi_R_ratio": np.nan, "phi_F": np.nan}
    if use_inner.any():
        tol = 1.0  # px, for RANSAC
        d1 = fuse(proj[use_inner], method, ransac_tol=tol)
        dr = d1 / max(phys["a_s_mag"], 1e-12)
        out.update(d1_px=d1, Delta_R_m=dr, phi_R=float(np.clip(1.0 - dr / phys["H_G_m"], 0.0, 1.0)),
                   phi_F=float(np.clip(1.0 - dr / phys["H_F_m"], 0.0, 1.0)))
        use_ground = np.asarray(use_ground, bool)
        if use_ground.any():
            ratio = fuse(mag_in[use_inner], method) / max(fuse(mag_gd[use_ground], method), 1e-9)
            out["phi_R_ratio"] = float(np.clip(1.0 - ratio, 0.0, 1.0))
    return out
