"""Keypoint definition and sun-aligned crop transform.

65 keypoints per tank, in this order:
    0           center          rim centre (image of the tank axis at z_C)
    1..16       rim             outer rim, z_C, radius R
    17..32      inner_shadow    inner-lip caster (radius r_inner) translated by delta_1
    33..48      ground_shadow   outer rim translated by delta_3
    49..64      roof_edge       roof perimeter (radius roof_radius) around the roof centre; with a visible edge
                                structure, the top of that structure (its own radius and height)
Anchor k sits at world azimuth psi_k = sun_azimuth + k * 22.5 deg (k = 0 faces the sun), so an index
has the same physical role in every image. World frame: x east, y north; acquisition.A_jacobian maps
world metres to image pixels (x right, y down).
"""
import math

import numpy as np

NK = 16
GROUPS = ["center", "rim", "inner_shadow", "ground_shadow", "roof_edge"]
GROUP_SLICE = {"center": slice(0, 1), "rim": slice(1, 17), "inner_shadow": slice(17, 33),
               "ground_shadow": slice(33, 49), "roof_edge": slice(49, 65)}
NUM_KP = 65
BOUNDARY_TOL_PX = 0.5      # a shadow/roof keypoint exists if it lies on the exported boundary
CHECK_TOL_PX = 0.05        # every exported boundary point must lie on the analytic curve


def group_of(i):
    for g, s in GROUP_SLICE.items():
        if s.start <= i < s.stop:
            return g


def flip_index():
    """Index map for a horizontal flip of a sun-aligned crop: anchor k <-> anchor (16 - k) mod 16."""
    idx = [0]
    for start in (1, 17, 33, 49):
        idx += [start + ((NK - k) % NK) for k in range(NK)]
    return np.array(idx)


def _unit_world(psi_deg):
    p = np.radians(psi_deg)
    return np.stack([np.sin(p), np.cos(p)], axis=-1)  # (east, north) for an azimuth clockwise from north


def _seg_dist(pts, poly):
    """Distance from each point to a sampled boundary polyline. Consecutive samples further apart than
    3x the median spacing are treated as a gap (separate visible segments), not joined."""
    poly = np.asarray(poly, float)
    pts = np.asarray(pts, float)
    if len(poly) == 0:
        return np.full(len(pts), np.inf)
    d = np.sqrt(((pts[:, None] - poly[None]) ** 2).sum(-1)).min(1)
    if len(poly) < 2:
        return d
    a, b = poly[:-1], poly[1:]
    seg = b - a
    L = np.linalg.norm(seg, axis=1)
    keep = L <= 3.0 * max(np.median(L), 1e-6)
    a, seg, L2 = a[keep], seg[keep], (L[keep] ** 2)[None]
    if len(a) == 0:
        return d
    t = np.clip(((pts[:, None] - a[None]) * seg[None]).sum(-1) / np.maximum(L2, 1e-12), 0, 1)
    proj = a[None] + t[..., None] * seg[None]
    return np.minimum(d, np.sqrt(((pts[:, None] - proj) ** 2).sum(-1)).min(1))


def _curve_dist(pts, centre, A, r):
    """Distance from points to the affine image of a horizontal circle of radius r (dense sampling)."""
    t = np.radians(np.arange(0.0, 360.0, 0.02))
    curve = centre + (A @ (r * np.stack([np.cos(t), np.sin(t)]))).T
    pts = np.asarray(pts, float)
    out = np.empty(len(pts))
    for i in range(0, len(pts), 64):
        out[i:i + 64] = np.sqrt(((pts[i:i + 64, None] - curve[None]) ** 2).sum(-1)).min(1)
    return out


# Edge structures whose top ring sits inside the roof perimeter (as in synthetic_data/src/v3/physics.py)
EDGE_RADIUS_INSET = {"none": 0.0, "rim_plate": 0.0, "seal": 0.0, "foam_dam": 0.6, "pontoon_edge": 0.0}


def _is_level_legacy(w):
    """v1 roof: centred, level, no edge structure (the closed-form translations below are exact)."""
    return (not w.get("roof_tilt_deg") and not w.get("roof_offset_x_m") and not w.get("roof_offset_y_m")
            and w.get("eta_edge", "none") == "none")


def _general_curves(lab, psi_deg):
    """
    Image points of the inner shadow and the roof edge at world azimuths psi for any v2 roof: an offset
    roof centre, a tilted deck (the inner-lip shadow lands on the tilted plane) and a visible edge
    structure (its top ring at its own radius and height). Orthographic camera: a world point X maps to
    c + A X_xy + b (X_z - z_C), with c the image of the tank axis at the rim height z_C.
    """
    acq, w, pt = lab["acquisition"], lab["world_truth"], lab["projected_truth"]
    A, b = np.array(acq["A_jacobian"], float), np.array(acq["b_vector"], float)
    c = np.array([pt["rim_ellipse"]["cx_px"], pt["rim_ellipse"]["cy_px"]], float)
    z_C, z_R = w["z_C_m"], w["z_R_m"]
    off = np.array([w.get("roof_offset_x_m") or 0.0, w.get("roof_offset_y_m") or 0.0])
    t, a = math.radians(w.get("roof_tilt_deg") or 0.0), math.radians(w.get("roof_tilt_azimuth_deg") or 0.0)
    n = np.array([math.sin(t) * math.sin(a), math.sin(t) * math.cos(a), math.cos(t)])   # roof plane normal
    roof_c = np.array([off[0], off[1], z_R])
    se, sa = math.radians(acq["sun_elevation_deg"]), math.radians(acq["sun_azimuth_deg"])
    d_sun = np.array([math.cos(se) * math.sin(sa), math.cos(se) * math.cos(sa), math.sin(se)])
    u = _unit_world(np.asarray(psi_deg, float))

    def image(X):
        return c + (A @ X[:, :2].T).T + np.outer(X[:, 2] - z_C, b)

    lip = np.column_stack([w["radius_inner_m"] * u, np.full(len(u), z_C)])
    s = ((lip - roof_c) @ n) / (d_sun @ n)
    inner = lip - s[:, None] * d_sun                                       # lip shadow on the roof plane
    edge_type = w.get("eta_edge", "none")
    r_edge = w.get("eta_edge_radius_m", w["roof_radius_m"] - EDGE_RADIUS_INSET.get(edge_type, 0.0))
    e_xy = off + r_edge * u
    e_z = z_R - ((e_xy[:, 0] - off[0]) * n[0] + (e_xy[:, 1] - off[1]) * n[1]) / n[2] + (w.get("edge_offset_m") or 0.0)
    return image(inner), image(np.column_stack([e_xy, e_z]))


def _general_dist(pts, curve):
    pts = np.asarray(pts, float)
    out = np.empty(len(pts))
    for i in range(0, len(pts), 64):
        out[i:i + 64] = np.sqrt(((pts[i:i + 64, None] - curve[None]) ** 2).sum(-1)).min(1)
    return out


def keypoints_from_label(lab):
    """Return (pts[65, 2] native px, exists[65] bool, physics dict, check dict)."""
    acq, w, pt = lab["acquisition"], lab["world_truth"], lab["projected_truth"]
    A = np.array(acq["A_jacobian"], float)
    c = np.array([pt["rim_ellipse"]["cx_px"], pt["rim_ellipse"]["cy_px"]], float)
    d1 = np.array(pt["delta_1_px"], float)
    d3 = np.array(pt["delta_3_px"], float)
    droof = np.array(pt["delta_roof_center_px"], float)
    R, r_in, r_roof = w["radius_m"], w["radius_inner_m"], w["roof_radius_m"]
    psi = acq["sun_azimuth_deg"] + np.arange(NK) * 360.0 / NK
    u = _unit_world(psi)  # (16, 2)
    rim = c + (A @ (R * u).T).T
    ground = rim + d3
    level = _is_level_legacy(w)
    if level:
        inner = c + (A @ (r_in * u).T).T + d1
        roof = c + droof + (A @ (r_roof * u).T).T
    else:
        inner, roof = _general_curves(lab, psi)
    pts = np.vstack([c[None], rim, inner, ground, roof])

    exists = np.ones(NUM_KP, bool)
    bnd = {"inner_shadow": pt["inner_shadow_boundary"], "ground_shadow": pt["ground_shadow_boundary"],
           "roof_edge": pt["roof_edge_boundary"]}
    for g, poly in bnd.items():
        s = GROUP_SLICE[g]
        exists[s] = _seg_dist(pts[s], poly) <= BOUNDARY_TOL_PX

    # Independent check: every exported boundary point must lie on the analytic curve used above.
    check = {}
    dense = None if level else _general_curves(lab, np.arange(0.0, 360.0, 0.02))
    for g, (cen, r) in {"inner_shadow": (c + d1, r_in), "ground_shadow": (c + d3, R),
                        "roof_edge": (c + droof, r_roof)}.items():
        poly = np.asarray(bnd[g], float)
        if not len(poly):
            check[g] = 0.0
        elif dense is not None and g != "ground_shadow":
            check[g] = float(_general_dist(poly, dense[0 if g == "inner_shadow" else 1]).max())
        else:
            check[g] = float(_curve_dist(poly, cen, A, r).max())

    el = math.radians(acq["sun_elevation_deg"])
    toward_sun = _unit_world(acq["sun_azimuth_deg"])
    shadow_img = A @ (-toward_sun)                      # horizontal shadow direction in the image
    phys = {
        "a_s": [float(x) for x in acq["a_s_vector"]], "a_s_mag": float(np.linalg.norm(acq["a_s_vector"])),  # px per metre (acq["a_s_magnitude"] is in other units)
        "b": [float(x) for x in acq["b_vector"]],
        "shadow_dir_img": [float(x) for x in shadow_img / np.linalg.norm(shadow_img)],
        "R_m": R, "r_inner_m": r_in, "roof_radius_m": r_roof,
        "R_px": float(pt["rim_ellipse"]["a_px"]),
        "H_F_m": w["H_F_m"], "H_G_m": w["H_G_m"], "Delta_R_m": w["Delta_R_m"], "phi_R": w["phi_R"],
        "delta_1_px": d1.tolist(), "delta_1_mag_px": float(np.linalg.norm(d1)),
        "delta_3_px": d3.tolist(), "delta_3_mag_px": float(np.linalg.norm(d3)),
        "gsd_m_per_px": float(acq["gsd_m_per_px"]),
        "sun_elevation_deg": acq["sun_elevation_deg"], "sun_azimuth_deg": acq["sun_azimuth_deg"],
        "sensor_offnadir_deg": acq["sensor_offnadir_deg"], "tan_el": math.tan(el),
    }
    return pts, exists, phys, check


def crop_matrix(centre, R_px, shadow_dir_img, ctx_factor, out_size, rot_offset_deg=0.0):
    """2x3 affine native px -> crop px. The crop is centred on the rim centre, spans 2*ctx_factor*R_px
    native pixels, and is rotated so the image shadow direction points down (+y)."""
    ang = math.atan2(shadow_dir_img[1], shadow_dir_img[0])  # current angle of the shadow direction
    rot = math.pi / 2 - ang + math.radians(rot_offset_deg)  # rotate it onto +y
    s = out_size / (2.0 * ctx_factor * R_px)
    cr, sr = math.cos(rot) * s, math.sin(rot) * s
    L = np.array([[cr, -sr], [sr, cr]])
    t = np.array([out_size / 2.0, out_size / 2.0]) - L @ np.asarray(centre, float)
    return np.hstack([L, t[:, None]])


def apply_affine(M, pts):
    pts = np.asarray(pts, float)
    return pts @ M[:, :2].T + M[:, 2]


def invert_affine(M):
    L = M[:, :2]
    Li = np.linalg.inv(L)
    return np.hstack([Li, (-Li @ M[:, 2])[:, None]])


def compose(M2, M1):
    """Affine that applies M1 then M2."""
    return np.hstack([M2[:, :2] @ M1[:, :2], (M2[:, :2] @ M1[:, 2] + M2[:, 2])[:, None]])
