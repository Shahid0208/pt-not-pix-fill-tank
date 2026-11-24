"""
Fill fraction from an inner-shadow mask: the keypoint pipeline's M3 forward model (calc_handoff.py, handoff
§2.3), applied to regions instead of keypoints.

Orthographic camera (as in keypoint/geometry_kp.py): a world point X maps to c + A X_xy + b (X_z - z_C), with c
the image of the tank axis at rim height z_C. For a level roof at height z, in the image:
  lip(q)   q inside the inner-lip ellipse          |A^-1 (q - c)|                     <= r_inner
  roof(q)  q inside the roof-edge ellipse at z     |A^-1 (q - c - b (z - z_C))|      <= r_roof
  lit(q)   q inside the lip's shadow outline at z  |A^-1 (q - c - (z_C - z) a_s)|    <= r_inner
The roof is seen through the lip opening, and a roof point is lit when the sun ray from it passes through
the lip, so the predicted inner shadow is S(z) = lip & roof & ~lit.
z_R = argmin over z in [z_G, z_C] of |S(z) xor P| (P = mask pixels of class 2 inside the lip), i.e. the roof
height whose shadow best explains the mask. Pixels are scored by coverage (SS x SS sub-samples) against the
mask or the predicted probability, so the fit is sub-pixel. With less than MIN_SHADOW_PX of inner shadow the
height is not determined and the tank is marked `abstain` (the keypoint pipeline abstains when no inner-shadow
point is usable).

Static calibration as in calc_handoff.py (handoff H0.1): rim centre c, R, r_inner, roof radius, z_C, z_G and the
sun/camera vectors come from the labels; only the shadow is measured. Roof offset and tilt are not modelled
(level-roof assumption, as for the keypoint M3).
"""
import numpy as np

Z_STEP = 0.01   # m
INNER = 2


SS = 4             # sub-samples per pixel side: pixel coverage of the modelled shadow, for sub-pixel fits
MIN_SHADOW_PX = 10  # abstain below this much inner shadow (crop px): the roof height is not determined


def _grid(meta, ss=1):
    """Native-pixel coordinates of every crop pixel's ss x ss sub-sample centres: (n, n, ss*ss, 2)."""
    c = meta["crop"]
    n = c["size"]
    o = (np.arange(ss) + 0.5) / ss
    sub = np.stack(np.meshgrid(o, o), -1).reshape(-1, 2)                     # (ss*ss, 2)
    px = np.stack(np.meshgrid(np.arange(n), np.arange(n)), -1)[:, :, None]   # (n, n, 1, 2)
    return (px + sub) / c["k"] + np.array([c["x0"], c["y0"]])


def _frame(meta, ss=SS):
    """Sub-samples mapped to metres in the rim plane (w = A^-1 (q - c)), the pixels touching the lip, and the
    roof-centre and shadow shifts per metre of depth below the rim, in the same metric frame."""
    Ainv = np.linalg.inv(np.asarray(meta["A"], float))
    w = (_grid(meta, ss) - np.asarray(meta["rim_center_native"], float)) @ Ainv.T
    lip_sub = (w ** 2).sum(-1) <= meta["r_inner_m"] ** 2
    return w, lip_sub, Ainv @ np.asarray(meta["b"], float), Ainv @ np.asarray(meta["a_s"], float)


def _coverage(w, lip_sub, depth, u_b, u_a, r_roof, r_in):
    """Fraction of each pixel's sub-samples in shadow for a roof `depth` m below the rim: seen through the lip,
    inside the roof edge (centre shifted by -depth u_b) and outside the lip's shadow outline (shifted by
    depth u_a)."""
    roof = ((w + depth * u_b) ** 2).sum(-1) <= r_roof ** 2
    lit = ((w - depth * u_a) ** 2).sum(-1) <= r_in ** 2
    return (lip_sub & roof & ~lit).mean(-1)


def shadow_model(meta, z):
    """Crop-sized predicted inner-shadow mask (coverage >= 0.5) for a roof at height z, and the lip mask."""
    w, lip_sub, u_b, u_a = _frame(meta)
    cov = _coverage(w, lip_sub, meta["z_C_m"] - z, u_b, u_a, meta["roof_radius_m"], meta["r_inner_m"])
    return cov >= 0.5, lip_sub.any(-1)


def fill_from_mask(target, meta):
    """target: (size, size) class map (inner shadow = class 2) or inner-shadow probability in [0, 1].
    Returns z_R, phi_R (ground-referenced, as the keypoint M3), phi_F (floor-referenced), the fit's mismatch
    (mean |coverage - target| over lip pixels), the evidence (target sum in the lip) and `abstain`."""
    t = np.asarray(target)
    P = (t == INNER).astype(float) if t.dtype.kind in "iu" else t.astype(float)
    w, lip_sub, u_b, u_a = _frame(meta)
    lip = lip_sub.any(-1)
    wl, ls, Pl = w[lip], lip_sub[lip], P[lip]
    zc, zg, zf = meta["z_C_m"], meta["z_G_m"], meta["z_F_m"]
    rr, ri = meta["roof_radius_m"], meta["r_inner_m"]
    evidence = float(Pl.sum())

    def costs(zs):
        return np.array([np.abs(_coverage(wl, ls, zc - z, u_b, u_a, rr, ri) - Pl).sum() for z in zs])

    zs = np.arange(zg, zc + 1e-9, 10 * Z_STEP)                    # coarse: 10 cm
    cz = costs(zs)
    i = int(np.argmin(cz))
    zs = np.arange(max(zg, zs[i] - 0.15), min(zc, zs[i] + 0.15) + 1e-9, Z_STEP)   # fine: 1 cm
    cz = costs(zs)
    best = np.flatnonzero(cz <= cz.min() + 1e-9)                   # flat minimum: its middle
    z = float(zs[best[len(best) // 2]])
    return {"z_R": z, "phi_R": (z - zg) / (zc - zg), "phi_F": (z - zf) / (zc - zf),
            "mismatch": float(cz.min() / max(lip.sum(), 1)), "evidence_px": evidence,
            "abstain": evidence < MIN_SHADOW_PX}


def area_ratio_fill(pred_cls, sun_elevation_deg):
    """The original pipeline's heuristic (utils.calculate_oil_volume), reported for reference only:
    1 - A_int / (A_ext sin(theta)) with A_ext = ground-shadow pixels, A_int = inner-shadow pixels."""
    a_ext, a_int = np.sum(pred_cls == 1), np.sum(pred_cls == 2)
    if a_ext == 0:
        return 0.0
    return float(np.clip(1 - a_int / (a_ext * np.sin(np.radians(sun_elevation_deg))), 0, 1))


# ------------------------------------------------------------------------------------------------------------
# Handoff §2.2 from masks: displacements d_int (inner shadow) and d_ext (ground shadow), no sun/camera metadata.
#   d_int: translation t1 of the inner-lip ellipse whose crescent (lip minus shifted lip) matches the inner shadow
#   d_ext: translation d3 of the rim disc whose sweep (rim disc moved by s d3, s in [0, 1]: a capsule in the rim
#          frame) matches the target's ground shadow
# With an orthographic camera both are multiples of the same image vector a_s: t1 = (z_C - z_R) a_s and
# d3 = (z_C - z_G) a_s, so M1: phi_R = 1 - |t1| / |d3| (handoff boxed formula) and M2: z_C - z_R = |t1| GSD tan(el).
# Pixels are mapped to the rim frame w = A^-1 (q - c) (calibration: rim centre c and the ellipse's A), so a
# circle of radius r is |w| <= r and an image translation t is the shift A^-1 t.
# ------------------------------------------------------------------------------------------------------------
def _frame_points(q_native, meta):
    Ainv = np.linalg.inv(np.asarray(meta["A"], float))
    return (q_native - np.asarray(meta["rim_center_native"], float)) @ Ainv.T


def _dir_search(cost, r_max, n_mag=120, u0=None, n_dir=36):
    """Minimise cost(u) over 2D shifts u. Coarse scan: every 360/n_dir degrees x n_mag magnitudes (or, with u0,
    magnitudes along u0 only), then two local angle x magnitude refinement rounds. Returns (u, cost)."""
    if u0 is None:
        dirs = [np.array([np.cos(a), np.sin(a)]) for a in np.arange(n_dir) * 2 * np.pi / n_dir]
    else:
        dirs = [u0 / max(np.linalg.norm(u0), 1e-9)]
    mags = np.linspace(0, r_max, n_mag)
    best_u, best_c = np.zeros(2), cost(np.zeros(2))
    for d in dirs:
        for m in mags[1:]:
            c = cost(m * d)
            if c < best_c:
                best_u, best_c = m * d, c
    for rnd in range(2):
        step_a = np.radians(5 if u0 is None else 2) / (4 ** rnd)
        step_m = r_max / n_mag / (4 ** rnd)
        a_c, m_c = np.arctan2(best_u[1], best_u[0]), np.linalg.norm(best_u)
        for da in np.arange(-6, 7) * step_a:
            for dm in np.arange(-4, 5) * step_m:
                mm = max(m_c + dm, 0)
                u = mm * np.array([np.cos(a_c + da), np.sin(a_c + da)])
                c = cost(u)
                if c < best_c:
                    best_u, best_c = u, c
    return best_u, best_c


def fit_inner_displacement(target, meta, direction_px=None):
    """t1 (native px) from an inner-shadow class map or probability on the meta's crop. None if < MIN_SHADOW_PX.
    direction_px: the fitted ground-shadow displacement d3; t1 is searched along it (both are multiples of a_s)."""
    t = np.asarray(target)
    P = (t == INNER).astype(float) if t.dtype.kind in "iu" else t.astype(float)
    w = _frame_points(_grid(meta, 2), meta)                      # (n, n, 4, 2)
    ri = meta["r_inner_m"]
    lip_sub = (w ** 2).sum(-1) <= ri ** 2
    lip = lip_sub.any(-1)
    wl, ls, Pl = w[lip], lip_sub[lip], P[lip]
    if Pl.sum() < MIN_SHADOW_PX:
        return None
    def cost(u):
        cov = (ls & (((wl - u) ** 2).sum(-1) > ri ** 2)).mean(-1)
        return np.abs(cov - Pl).sum()
    u0 = None
    if direction_px is not None:
        u0 = np.linalg.inv(np.asarray(meta["A"], float)) @ np.asarray(direction_px, float)
    u, c = _dir_search(cost, 2 * ri, u0=u0)
    if np.linalg.norm(u) < 1e-6 or c >= 0.9 * cost(np.zeros(2)):   # no shift explains the shadow: not measurable
        return None
    return np.asarray(meta["A"], float) @ u                          # rim frame -> image px


def fit_ground_displacement(cls_mosaic, q_native, valid, meta, step=2):
    """d3 (native px) from a ground-shadow class map on a mosaic (q_native: (H, W, 2) native coords of its
    pixels). Only ground-shadow blobs touching the tank are used (neighbours' shadows are dropped); the tank top
    (|w| <= R) is not scored. Returns (d3 or None, reason)."""
    import cv2
    R = meta["R_m"]
    O = (cls_mosaic == 1) & valid
    w_full = _frame_points(q_native, meta)
    near = (w_full ** 2).sum(-1) <= (1.15 * R) ** 2
    n, lab = cv2.connectedComponents(O.astype(np.uint8), connectivity=8)
    keep = np.unique(lab[O & near])
    O = np.isin(lab, keep[keep > 0])
    if O.sum() < MIN_SHADOW_PX:
        return None, "no ground shadow at the tank"
    def make_cost(st):
        sl = (slice(None, None, st), slice(None, None, st))
        w, Os, vs = w_full[sl], O[sl], valid[sl]
        score = vs & ((w ** 2).sum(-1) > R ** 2)
        ws, Ob = w[score], Os[score]

        def cost(u):
            uu = max(float(u @ u), 1e-12)
            s = np.clip(ws @ u / uu, 0, 1)
            return np.count_nonzero((((ws - s[:, None] * u) ** 2).sum(-1) <= R ** 2) ^ Ob)
        return cost
    r_max = np.sqrt((w_full[valid] ** 2).sum(-1).max())
    u, _ = _dir_search(make_cost(2 * step), r_max, n_mag=100)           # coarse grid, all directions
    fine = make_cost(step)
    u, c = _dir_search(fine, r_max, n_mag=400, u0=u if np.linalg.norm(u) > 1e-6 else None)   # refine along it
    if np.linalg.norm(u) < 1e-6 or c >= 0.9 * fine(np.zeros(2)):
        return None, "ground shadow not measurable (hidden by the tank or not shadow-shaped)"
    tip = u + R * u / max(np.linalg.norm(u), 1e-9)                   # far end of the shadow, rim frame
    inside = (((w_full - tip) ** 2).sum(-1) <= (0.1 * R) ** 2) & valid   # a valid mosaic pixel at the tip
    if not inside.any():
        return None, "shadow tip outside the mosaic"
    return np.asarray(meta["A"], float) @ u, "ok"


def fill_m1_m2(t1, d3, meta):
    """Handoff §2.2. M1 = 1 - |t1|/|d3| (no metadata); M2 from |t1| GSD tan(sun elevation) (nadir assumption).
    Both limited to phi_R in [0, 1]."""
    zc, zg = meta["z_C_m"], meta["z_G_m"]
    out = {}
    if t1 is not None:
        d_int = float(np.linalg.norm(t1))
        z2 = zc - d_int * meta["gsd_m_per_px"] * np.tan(np.radians(meta["sun_elevation_deg"]))
        out["M2_phi_R"] = float(np.clip((z2 - zg) / (zc - zg), 0, 1))
        out["d_int_px"] = d_int
        if d3 is not None:
            d_ext = float(np.linalg.norm(d3))
            out["d_ext_px"] = d_ext
            out["M1_phi_R"] = float(np.clip(1 - d_int / max(d_ext, 1e-9), 0, 1))
    return out
