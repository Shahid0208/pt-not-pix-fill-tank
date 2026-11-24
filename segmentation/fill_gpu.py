"""
GPU versions of the fill fits in fill_from_mask.py (same maths, same search, same tie-breaking), for the
evaluation scripts: every candidate roof height (M3) or displacement (M1/M2) of a tank is scored in one
batched tensor operation instead of a Python loop. float64 throughout, so pixel tests at the exact boundary
(<= r^2) decide as on the CPU; check_fill_gpu.py compares the two on real data.

    from fill_gpu import fill_from_mask_gpu, fill_m1_m2_gpu
"""
import numpy as np
import torch

import fill_from_mask as F

DT = torch.float64


def _t(x, dev):
    return torch.as_tensor(np.asarray(x, np.float64), dtype=DT, device=dev)


def _first_min(base, costs):
    """Index of the first strict improvement in order: -1 if no candidate beats `base` (as the CPU loop with
    `if c < best_c`), else the earliest index among the minimal costs."""
    m = torch.min(costs)
    if not bool(m < base):
        return -1
    return int(torch.nonzero(costs == m)[0, 0])


# ------------------------------------------------------------------------------------------------ M3
def fill_from_mask_gpu(target, meta, dev, zchunk=24):
    """GPU version of fill_from_mask.fill_from_mask (same return dict)."""
    t = np.asarray(target)
    P = (t == F.INNER).astype(float) if t.dtype.kind in "iu" else t.astype(float)
    w, lip_sub, u_b, u_a = F._frame(meta)
    lip = lip_sub.any(-1)
    wl, ls, Pl = _t(w[lip], dev), torch.as_tensor(lip_sub[lip], device=dev), _t(P[lip], dev)
    ub, ua = _t(u_b, dev), _t(u_a, dev)
    zc, zg, zf = meta["z_C_m"], meta["z_G_m"], meta["z_F_m"]
    rr2, ri2 = meta["roof_radius_m"] ** 2, meta["r_inner_m"] ** 2
    evidence = float(P[lip].sum())

    def costs(zs):
        out = []
        for k in range(0, len(zs), zchunk):
            d = _t(zc - zs[k:k + zchunk], dev)[:, None, None, None]               # (Z,1,1,1) depth
            roof = ((wl[None] + d * ub) ** 2).sum(-1) <= rr2                       # (Z,N,S)
            lit = ((wl[None] - d * ua) ** 2).sum(-1) <= ri2
            cov = (ls[None] & roof & ~lit).to(DT).mean(-1)                          # (Z,N)
            out.append((cov - Pl[None]).abs().sum(-1))
        return torch.cat(out).cpu().numpy()

    zs = np.arange(zg, zc + 1e-9, 10 * F.Z_STEP)
    cz = costs(zs)
    i = int(np.argmin(cz))
    zs = np.arange(max(zg, zs[i] - 0.15), min(zc, zs[i] + 0.15) + 1e-9, F.Z_STEP)
    cz = costs(zs)
    best = np.flatnonzero(cz <= cz.min() + 1e-9)
    z = float(zs[best[len(best) // 2]])
    return {"z_R": z, "phi_R": (z - zg) / (zc - zg), "phi_F": (z - zf) / (zc - zf),
            "mismatch": float(cz.min() / max(lip.sum(), 1)), "evidence_px": evidence,
            "abstain": evidence < F.MIN_SHADOW_PX}


# ------------------------------------------------------------------------------------------------ M1 / M2
def _dir_search_gpu(cost_batch, r_max, dev, n_mag=120, u0=None, n_dir=36):
    """Batched _dir_search: same candidates in the same order, same strict-improvement rule."""
    if u0 is None:
        a = torch.arange(n_dir, dtype=DT, device=dev) * 2 * np.pi / n_dir
        dirs = torch.stack([torch.cos(a), torch.sin(a)], -1)
    else:
        dirs = _t(u0 / max(np.linalg.norm(u0), 1e-9), dev)[None]
    mags = torch.linspace(0, r_max, n_mag, dtype=DT, device=dev)[1:]
    cand = (dirs[:, None, :] * mags[None, :, None]).reshape(-1, 2)              # dir-major, as the CPU loops
    best_u = torch.zeros(2, dtype=DT, device=dev)
    best_c = cost_batch(best_u[None])[0]
    cc = cost_batch(cand)
    k = _first_min(best_c, cc)
    if k >= 0:
        best_u, best_c = cand[k], cc[k]
    for rnd in range(2):
        step_a = np.radians(5 if u0 is None else 2) / (4 ** rnd)
        step_m = r_max / n_mag / (4 ** rnd)
        a_c = torch.atan2(best_u[1], best_u[0])
        m_c = torch.linalg.norm(best_u)
        da = torch.arange(-6, 7, dtype=DT, device=dev) * step_a
        dm = torch.arange(-4, 5, dtype=DT, device=dev) * step_m
        mm = torch.clamp(m_c + dm, min=0)                                           # (9,)
        ang = a_c + da                                                              # (13,)
        cand = torch.stack([mm[None, :] * torch.cos(ang)[:, None], mm[None, :] * torch.sin(ang)[:, None]],
                           -1).reshape(-1, 2)                                       # da outer, dm inner
        cc = cost_batch(cand)
        k = _first_min(best_c, cc)
        if k >= 0:
            best_u, best_c = cand[k], cc[k]
    return best_u.cpu().numpy(), float(best_c)


def fit_inner_displacement_gpu(target, meta, dev, direction_px=None, kchunk=512):
    t = np.asarray(target)
    P = (t == F.INNER).astype(float) if t.dtype.kind in "iu" else t.astype(float)
    w = F._frame_points(F._grid(meta, 2), meta)
    ri = meta["r_inner_m"]
    lip_sub = (w ** 2).sum(-1) <= ri ** 2
    lip = lip_sub.any(-1)
    if P[lip].sum() < F.MIN_SHADOW_PX:
        return None
    wl, ls, Pl = _t(w[lip], dev), torch.as_tensor(lip_sub[lip], device=dev), _t(P[lip], dev)

    def cost_batch(U):                                                              # (K,2) -> (K,)
        out = []
        for k in range(0, len(U), kchunk):
            u = U[k:k + kchunk][:, None, None, :]
            cov = (ls[None] & (((wl[None] - u) ** 2).sum(-1) > ri ** 2)).to(DT).mean(-1)
            out.append((cov - Pl[None]).abs().sum(-1))
        return torch.cat(out)

    u0 = None
    if direction_px is not None:
        u0 = np.linalg.inv(np.asarray(meta["A"], float)) @ np.asarray(direction_px, float)
    u, c = _dir_search_gpu(cost_batch, 2 * ri, dev, u0=u0)
    c0 = float(cost_batch(torch.zeros((1, 2), dtype=DT, device=dev))[0])
    if np.linalg.norm(u) < 1e-6 or c >= 0.9 * c0:
        return None
    return np.asarray(meta["A"], float) @ u


def fit_ground_displacement_gpu(cls_mosaic, q_native, valid, meta, dev, step=2, kchunk=64):
    import cv2
    R = meta["R_m"]
    O = (cls_mosaic == 1) & valid
    w_full = F._frame_points(q_native, meta)
    near = (w_full ** 2).sum(-1) <= (1.15 * R) ** 2
    n, lab = cv2.connectedComponents(O.astype(np.uint8), connectivity=8)
    keep = np.unique(lab[O & near])
    O = np.isin(lab, keep[keep > 0])
    if O.sum() < F.MIN_SHADOW_PX:
        return None, "no ground shadow at the tank"

    def make_cost(st):
        sl = (slice(None, None, st), slice(None, None, st))
        w, Os, vs = w_full[sl], O[sl], valid[sl]
        score = vs & ((w ** 2).sum(-1) > R ** 2)
        ws, Ob = _t(w[score], dev), torch.as_tensor(Os[score], device=dev)

        def cost_batch(U):
            out = []
            for k in range(0, len(U), kchunk):
                u = U[k:k + kchunk]                                                 # (K,2)
                uu = torch.clamp((u * u).sum(-1), min=1e-12)                        # (K,)
                s = torch.clamp(ws @ u.T / uu, 0, 1)                                # (M,K)
                d2 = ((ws[:, None, :] - s[..., None] * u[None]) ** 2).sum(-1)       # (M,K)
                out.append(((d2 <= R ** 2) ^ Ob[:, None]).sum(0).to(DT))
            return torch.cat(out)
        return cost_batch

    r_max = np.sqrt((w_full[valid] ** 2).sum(-1).max())
    u, _ = _dir_search_gpu(make_cost(2 * step), r_max, dev, n_mag=100)
    fine = make_cost(step)
    u, c = _dir_search_gpu(fine, r_max, dev, n_mag=400, u0=u if np.linalg.norm(u) > 1e-6 else None)
    c0 = float(fine(torch.zeros((1, 2), dtype=DT, device=dev))[0])
    if np.linalg.norm(u) < 1e-6 or c >= 0.9 * c0:
        return None, "ground shadow not measurable (hidden by the tank or not shadow-shaped)"
    tip = u + R * u / max(np.linalg.norm(u), 1e-9)
    inside = (((w_full - tip) ** 2).sum(-1) <= (0.1 * R) ** 2) & valid
    if not inside.any():
        return None, "shadow tip outside the mosaic"
    return np.asarray(meta["A"], float) @ u, "ok"
