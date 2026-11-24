"""
Layer A (world truth) and Layer B (projected truth) for one floating-roof tank scene.

Notation, used for every key:
  z_C  rim / caster elevation (top of the wall)       z_R  floating-roof deck elevation
  z_F  tank floor elevation (lowest deck position)    z_G  exterior terrain elevation (0 at the tank)
  Delta_R = z_C - z_R      H_F = z_C - z_F      H_G = z_C - z_G      phi_R = (z_R - z_F) / H_F

World axes: +x East, +y North, +z Up (metres). Compass bearings clockwise from North.
Image axes: x right, y down (pixels), origin at the top-left pixel corner.

The satellite camera is orthographic, so the projection model y = A x + b z + c is EXACT here
(not an approximation): A and b are read straight from the camera axes.
"""
import math

import numpy as np

WALL_THICKNESS = 0.35   # must match blender_render_script.WALL_THICKNESS
ROOF_THICKNESS = 0.2
FRAME_MARGIN = 1.35     # must match blender_render_script.FRAME_MARGIN
EDGE_RADIUS_INSET = {"none": 0.0, "rim_plate": 0.0, "seal": 0.0, "foam_dam": 0.6, "pontoon_edge": 0.0}
CHANNELS = ("inner_shadow", "roof_edge", "ground_shadow", "tank_base")
BETA_STEP_DEG = 0.5     # rim sampling for arcs and dense boundary points


def compass(az_deg):
    a = np.radians(az_deg)
    return np.stack([np.sin(a), np.cos(a)], axis=-1)


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v * 0.0


def plane_normal(tilt_deg, tilt_az_deg):
    t, a = math.radians(tilt_deg), math.radians(tilt_az_deg)
    return np.array([math.sin(t) * math.sin(a), math.sin(t) * math.cos(a), math.cos(t)])


# ---------------------------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------------------------
class Camera:
    """Exact orthographic camera. From a Blender export (rendered levels) or built in Python (level A)."""

    def __init__(self, right, up, loc, ortho_scale, res_x, res_y):
        self.right, self.up, self.loc = np.asarray(right, float), np.asarray(up, float), np.asarray(loc, float)
        self.back = np.cross(self.right, self.up)        # unit vector towards the camera
        self.ortho_scale, self.res_x, self.res_y = float(ortho_scale), int(res_x), int(res_y)
        self.scale = self.res_x / self.ortho_scale        # px per metre in the image plane
        s = self.scale
        self.A = s * np.array([[self.right[0], self.right[1]], [-self.up[0], -self.up[1]]])
        self.b = s * np.array([self.right[2], -self.up[2]])
        self.c = self.project(np.zeros(3))

    @classmethod
    def from_blender(cls, cam):
        m = np.array(cam["matrix_world"], float)
        return cls(m[:3, 0], m[:3, 1], m[:3, 3], cam["ortho_scale"], cam["res_x"], cam["res_y"])

    @classmethod
    def for_scene(cls, w):
        """Same framing as blender_render_script.setup_camera(..., gsd): used for level A (no render)."""
        el, az = math.radians(w["sensor_elevation_deg"]), math.radians(w["sensor_azimuth_deg"])
        se, sa = math.radians(w["sun_elevation_deg"]), math.radians(w["sun_azimuth_deg"])
        R, H = w["radius_outer_R_m"], w["z_C_m"]
        L = H / math.tan(se)
        off = np.array([-0.5 * L * math.sin(sa), -0.5 * L * math.cos(sa), H / 2.0])
        to_cam = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
        loc = off + 250.0 * to_cam  # same position as Blender: offset target + 250 m towards the satellite
        forward = _unit(off - loc)
        north = np.array([0.0, 1.0, 0.0])
        up = north - north.dot(forward) * forward
        up = _unit(up) if np.linalg.norm(up) > 1e-6 else np.array([0.0, 0.0, 1.0])
        right = np.cross(up, -forward)
        sdx, sdy = -L * math.sin(sa), -L * math.cos(sa)
        half_w = half_h = 0.0
        for i in range(128):
            a = 2 * math.pi * i / 128
            x, y = R * math.cos(a), R * math.sin(a)
            for p in ((x, y, 0.0), (x, y, H), (x + sdx, y + sdy, 0.0)):
                rel = np.array(p) - off
                half_w, half_h = max(half_w, abs(rel.dot(right))), max(half_h, abs(rel.dot(up)))
        gsd = w["gsd_m_per_px"]
        res_x = max(64, int(math.ceil(2 * half_w * FRAME_MARGIN / gsd)))
        res_y = max(64, int(math.ceil(2 * half_h * FRAME_MARGIN / gsd)))
        return cls(right, up, loc, res_x * gsd, res_x, res_y)

    def project(self, P):
        P = np.asarray(P, float)
        rel = P - self.loc
        x = self.res_x / 2.0 + rel @ self.right * self.scale
        y = self.res_y / 2.0 - rel @ self.up * self.scale
        return np.stack([x, y], axis=-1)

    def to_dict(self):
        return {"right": self.right.tolist(), "up": self.up.tolist(), "loc": self.loc.tolist(),
                "ortho_scale": self.ortho_scale, "res_x": self.res_x, "res_y": self.res_y}


# ---------------------------------------------------------------------------------------------
# Layer A: world truth
# ---------------------------------------------------------------------------------------------
def world_truth(spec):
    """Exact physical scene parameters. No rendering involved."""
    R, H = float(spec["R"]), float(spec["H"])
    z_C, z_F, z_G = H, float(spec["z_F"]), 0.0
    H_F, H_G = z_C - z_F, z_C - z_G
    phi = float(spec["phi_R"])
    z_R = z_F + phi * H_F
    g = float(spec["seal_gap"])
    R_in = R - WALL_THICKNESS
    r_roof = R_in - g
    edge_type = spec.get("edge_type", "none")
    hx, hy = spec.get("roof_offset", (0.0, 0.0))
    tilt, tilt_az = spec.get("roof_tilt", (0.0, 0.0))
    slope, slope_az = spec.get("terrain_slope", (0.0, 0.0))
    return {
        "radius_outer_R_m": R, "radius_inner_m": R_in, "wall_thickness_m": WALL_THICKNESS,
        "roof_radius_m": r_roof, "seal_gap_g_m": g,
        "z_C_m": z_C, "z_F_m": z_F, "z_G_m": z_G, "z_R_m": z_R,
        "H_F_m": H_F, "H_G_m": H_G, "H_F_equals_H_G": abs(H_F - H_G) < 1e-9,
        "Delta_R_m": z_C - z_R, "phi_R": phi,
        "roof_offset_x_m": float(hx), "roof_offset_y_m": float(hy),
        "roof_concentric": abs(hx) < 1e-9 and abs(hy) < 1e-9,
        "roof_tilt_deg": float(tilt), "roof_tilt_azimuth_deg": float(tilt_az) if tilt else None,
        "terrain_slope_deg": float(slope), "terrain_slope_azimuth_deg": float(slope_az) if slope else None,
        "eta_edge_type": edge_type, "eta_edge_height_m": float(spec.get("edge_height", 0.0)),
        "eta_edge_radius_m": r_roof - EDGE_RADIUS_INSET.get(edge_type, 0.0),
        "roof_edge_offset_m": float(spec.get("edge_height", 0.0)),
        "caster_type": "rim_top",
        "caster_height_true_m": z_C,
        "caster_height_assumed_m": z_C + float(spec.get("caster_height_error", 0.0)),
        "center_world_x_m": 0.0, "center_world_y_m": 0.0,
        "sun_elevation_deg": float(spec["sun_el"]), "sun_azimuth_deg": float(spec["sun_az"]),
        "sensor_elevation_deg": float(spec["cam_el"]), "sensor_azimuth_deg": float(spec["cam_az"]),
        "sensor_offnadir_deg": 90.0 - float(spec["cam_el"]),
        "gsd_m_per_px": float(spec["gsd"]),
    }


class Scene:
    """Geometry helpers for one scene (world truth + exact camera)."""

    def __init__(self, w, cam):
        self.w, self.cam = w, cam
        se, sa = math.radians(w["sun_elevation_deg"]), math.radians(w["sun_azimuth_deg"])
        self.d_sun = np.array([math.cos(se) * math.sin(sa), math.cos(se) * math.cos(sa), math.sin(se)])
        self.k = 1.0 / math.tan(se)                                 # cot(theta)
        self.s_hat = -compass(w["sun_azimuth_deg"])                 # shadow direction, horizontal
        self.q_s = self.k * self.s_hat
        self.n_roof = plane_normal(w["roof_tilt_deg"], w["roof_tilt_azimuth_deg"] or 0.0)
        self.n_ter = plane_normal(w["terrain_slope_deg"], w["terrain_slope_azimuth_deg"] or 0.0)
        self.roof_c = np.array([w["roof_offset_x_m"], w["roof_offset_y_m"], w["z_R_m"]])
        self.to_cam = cam.back
        self.t = 0.0 if w["sensor_elevation_deg"] >= 89.999 else 1.0 / math.tan(math.radians(w["sensor_elevation_deg"]))
        # parallax direction (away from the sensor) in the ground plane
        self.u_hat = -compass(w["sensor_azimuth_deg"]) if self.t > 0 else np.zeros(2)
        self.A, self.b = cam.A, cam.b
        self.a_s = self.A @ self.q_s - self.b                       # image shift per metre of caster height

    # --- planes ------------------------------------------------------------------------------
    def shadow_on_plane(self, P, Q, n):
        t = ((P - Q) @ n) / (self.d_sun @ n)
        return P - t[..., None] * self.d_sun

    def terrain_z(self, xy):
        n = self.n_ter
        return -(xy[..., 0] * n[0] + xy[..., 1] * n[1]) / n[2]

    def roof_z(self, xy):
        n, c = self.n_roof, self.roof_c
        return c[2] - ((xy[..., 0] - c[0]) * n[0] + (xy[..., 1] - c[1]) * n[1]) / n[2]

    # --- visibility ----------------------------------------------------------------------------
    def through_opening(self, P):
        """True if the line of sight from P (inside the tank) to the camera clears the near wall."""
        v = self.to_cam
        if v[2] < 1e-9:
            return np.zeros(P.shape[:-1], bool)
        s = (self.w["z_C_m"] - P[..., 2]) / v[2]
        top = P[..., :2] + s[..., None] * v[:2]
        return np.linalg.norm(top, axis=-1) <= self.w["radius_inner_m"] + 1e-9

    def clear_of_tank(self, P):
        """True if the line of sight from a ground point P to the camera misses the tank cylinder."""
        v, R, z_C = self.to_cam, self.w["radius_outer_R_m"], self.w["z_C_m"]
        p, vh = P[..., :2], v[:2]
        a = vh @ vh
        if a < 1e-12:
            return np.linalg.norm(p, axis=-1) > R
        bq = 2 * (p @ vh)
        cq = np.sum(p * p, axis=-1) - R * R
        disc = bq * bq - 4 * a * cq
        ok = np.ones(P.shape[:-1], bool)
        hit = disc > 0
        sq = np.sqrt(np.where(hit, disc, 0.0))
        s1, s2 = (-bq - sq) / (2 * a), (-bq + sq) / (2 * a)
        s_in = np.maximum(s1, 0.0)
        z_in = P[..., 2] + s_in * v[2]
        blocked = hit & (s2 > 0) & (z_in <= z_C)
        ok[blocked] = False
        return ok

    # --- the four channels ----------------------------------------------------------------------
    def curves(self, beta_deg):
        """World points of rim anchors and channel boundaries at rim bearings beta."""
        w = self.w
        u = compass(beta_deg)
        z_C = w["z_C_m"]
        n = len(beta_deg)
        rim_in = np.column_stack([w["radius_inner_m"] * u, np.full(n, z_C)])
        rim_out = np.column_stack([w["radius_outer_R_m"] * u, np.full(n, z_C)])
        roof_pt = np.array([0.0, 0.0, 0.0])
        inner = self.shadow_on_plane(rim_in, self.roof_c, self.n_roof)
        ground = self.shadow_on_plane(rim_out, roof_pt, self.n_ter)
        e_xy = self.roof_c[:2] + w["eta_edge_radius_m"] * u
        edge = np.column_stack([e_xy, self.roof_z(e_xy) + w["eta_edge_height_m"]])
        b_xy = w["radius_outer_R_m"] * u
        base = np.column_stack([b_xy, self.terrain_z(b_xy)])
        return {"rim_inner": rim_in, "rim_outer": rim_out, "inner_shadow": inner, "roof_edge": edge,
                "ground_shadow": ground, "tank_base": base}

    def channel_masks(self, beta_deg, c):
        """valid = the point is a boundary of this channel; visible = valid and seen by the camera."""
        w = self.w
        u = compass(beta_deg)
        d_in = np.linalg.norm(c["inner_shadow"][:, :2] - self.roof_c[:2], axis=1)
        v_inner = d_in <= w["roof_radius_m"]
        v_edge = np.ones(len(beta_deg), bool)
        v_ground = (u @ self.s_hat) > 0
        vh = self.to_cam[:2]
        v_base = (u @ vh) > 1e-9 if np.linalg.norm(vh) > 1e-6 else np.zeros(len(beta_deg), bool)
        valid = {"inner_shadow": v_inner, "roof_edge": v_edge, "ground_shadow": v_ground, "tank_base": v_base}
        seen = {"inner_shadow": self.through_opening(c["inner_shadow"]),
                "roof_edge": self.through_opening(c["roof_edge"]),
                "ground_shadow": self.clear_of_tank(c["ground_shadow"]),
                "tank_base": np.ones(len(beta_deg), bool)}
        return valid, {k: valid[k] & seen[k] for k in valid}

    def channel_direction(self, ch):
        """Unit image direction a_hat of the channel translation."""
        v = self.a_s if ch in ("inner_shadow", "ground_shadow") else -self.b
        return _unit(v), float(np.linalg.norm(v))

    def anchor_key(self, ch):
        return "rim_inner" if ch in ("inner_shadow", "roof_edge") else "rim_outer"

    def channel_centre(self, ch):
        """World point that the channel's circle is centred on (for the channel translation vector)."""
        w = self.w
        c_top = np.array([0.0, 0.0, w["z_C_m"]])
        if ch == "inner_shadow":
            return self.shadow_on_plane(c_top[None], self.roof_c, self.n_roof)[0]
        if ch == "ground_shadow":
            return self.shadow_on_plane(c_top[None], np.zeros(3), self.n_ter)[0]
        if ch == "roof_edge":
            return np.array([*self.roof_c[:2], self.roof_c[2] + w["eta_edge_height_m"]])
        return np.array([0.0, 0.0, w["z_G_m"]])


# ---------------------------------------------------------------------------------------------
# Layer B: projected truth
# ---------------------------------------------------------------------------------------------
def _segments(mask):
    """Contiguous True runs of a circular boolean array, as (start, end) index pairs (inclusive)."""
    n = len(mask)
    if not mask.any():
        return []
    if mask.all():
        return [(0, n - 1)]
    start = int(np.argmin(mask))  # begin scanning at a False so runs do not wrap
    runs, cur = [], None
    for j in range(1, n + 1):
        i = (start + j) % n
        if mask[i] and cur is None:
            cur = i
        if not mask[i] and cur is not None:
            runs.append((cur, (i - 1) % n))
            cur = None
    if cur is not None:
        runs.append((cur, (start + n) % n))
    return runs


def _run_indices(run, n):
    s, e = run
    return np.arange(s, s + ((e - s) % n) + 1) % n


def _polyline_intersections(P, Q):
    """Intersections of polylines P (N,2) and Q (M,2). Returns (point, index_in_Q). Vectorised."""
    if len(P) < 2 or len(Q) < 2:
        return []
    p, r = P[:-1, None, :], (P[1:] - P[:-1])[:, None, :]
    q, sv = Q[None, :-1, :], (Q[1:] - Q[:-1])[None, :, :]
    den = r[..., 0] * sv[..., 1] - r[..., 1] * sv[..., 0]
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * sv[..., 1] - qp[..., 1] * sv[..., 0]) / den
        v = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / den
    hit = (np.abs(den) > 1e-12) & (t >= 0) & (t <= 1) & (v >= 0) & (v <= 1)
    i, j = np.nonzero(hit)
    return [(P[a] + t[a, b] * (P[a + 1] - P[a]), int(b)) for a, b in zip(i, j)]


def projected_truth(scene):
    """Exact subpixel image geometry of rim and all four channels."""
    cam = scene.cam
    betas = np.arange(0.0, 360.0, BETA_STEP_DEG)
    c = scene.curves(betas)
    valid, vis = scene.channel_masks(betas, c)
    px = {k: cam.project(v) for k, v in c.items()}
    rim_c = cam.project(np.array([0.0, 0.0, scene.w["z_C_m"]]))
    out = {"gt_rim_center_x_px": float(rim_c[0]), "gt_rim_center_y_px": float(rim_c[1]), "channels": {}}
    n = len(betas)
    for ch in CHANNELS:
        a_hat, a_mag = scene.channel_direction(ch)
        centre_px = cam.project(scene.channel_centre(ch))
        delta = centre_px - rim_c
        rec = {"delta_x_px": float(delta[0]), "delta_y_px": float(delta[1]),
               "delta_magnitude_px": float(np.linalg.norm(delta)), "d_k_px": float(delta @ a_hat),
               "direction_x": float(a_hat[0]), "direction_y": float(a_hat[1]), "sensitivity_px_per_m": a_mag,
               "arc_visible_fraction": float(vis[ch].mean())}
        runs = _segments(vis[ch])
        rec["arc_segments_deg"] = [[float(betas[s]), float(betas[e])] for s, e in runs]
        if runs:
            main = max(runs, key=lambda r: (r[1] - r[0]) % n)
            idx = _run_indices(main, n)
            pts = px[ch][idx]
            p1, p2 = pts[0], pts[-1]
            mid = 0.5 * (p1 + p2)
            arc_mid = pts[len(pts) // 2]
            chord = p2 - p1
            sag = abs(_cross2(chord, arc_mid - p1)) / max(np.linalg.norm(chord), 1e-9)
            all_idx = np.concatenate([_run_indices(r, n) for r in runs])
            # where the full channel curve meets the rim curve (e.g. the crescent tips), closed loops
            loop = np.vstack([px[ch], px[ch][:1]])
            rim_loop = np.vstack([px[scene.anchor_key(ch)], px[scene.anchor_key(ch)][:1]])
            inter = _polyline_intersections(loop, rim_loop)
            rec.update({
                "arc_start_deg": float(betas[main[0]]), "arc_end_deg": float(betas[main[1]]),
                "arc_length_px": float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))),
                "chord_p1_x_px": float(p1[0]), "chord_p1_y_px": float(p1[1]),
                "chord_p2_x_px": float(p2[0]), "chord_p2_y_px": float(p2[1]),
                "chord_length_px": float(np.linalg.norm(chord)),
                "chord_midpoint_x_px": float(mid[0]), "chord_midpoint_y_px": float(mid[1]),
                "sagitta_px": float(sag),
                "rim_intersections": [{"x_px": float(p[0]), "y_px": float(p[1]),
                                       "rim_angle_deg": float(betas[j])} for p, j in inter],
                "boundary_points_px": np.round(px[ch][all_idx[::2]], 3).tolist(),
            })
        else:
            rec.update({k: None for k in ("arc_start_deg", "arc_end_deg", "arc_length_px", "chord_p1_x_px",
                                          "chord_p1_y_px", "chord_p2_x_px", "chord_p2_y_px", "chord_length_px",
                                          "chord_midpoint_x_px", "chord_midpoint_y_px", "sagitta_px")})
            rec.update({"rim_intersections": [], "boundary_points_px": []})
        hidden = valid[ch] & ~vis[ch]
        rec["occluded_by"] = "wall" if hidden.any() else "none"
        out["channels"][ch] = rec
    # parallax of the roof centre vs the visible edge structure
    roof_c_px = cam.project(scene.roof_c)
    rim_axis_px = cam.project(np.array([*scene.roof_c[:2], scene.w["z_C_m"]]))
    edge_c_px = cam.project(scene.channel_centre("roof_edge"))
    out["gt_delta_roof_center_x_px"], out["gt_delta_roof_center_y_px"] = (roof_c_px - rim_axis_px).tolist()
    out["gt_delta_roof_edge_x_px"], out["gt_delta_roof_edge_y_px"] = (edge_c_px - rim_axis_px).tolist()
    # caster-shadow correspondences every 15 degrees on the visible arcs
    pairs = []
    for ch, caster in (("inner_shadow", "rim_inner"), ("ground_shadow", "rim_outer")):
        for j in np.where(vis[ch])[0][::int(15 / BETA_STEP_DEG)]:
            pairs.append({"channel": ch, "caster_world": np.round(c[caster][j], 4).tolist(),
                          "shadow_world": np.round(c[ch][j], 4).tolist(),
                          "caster_px": np.round(px[caster][j], 3).tolist(),
                          "shadow_px": np.round(px[ch][j], 3).tolist()})
    out["gt_caster_shadow_pairs"] = pairs
    return out, {"betas": betas, "curves": c, "px": px, "valid": valid, "visible": vis}


# ---------------------------------------------------------------------------------------------
# Acquisition-derived fields and observability
# ---------------------------------------------------------------------------------------------
def acquisition_fields(scene):
    w = scene.w
    k, t = scene.k, scene.t
    cosb = float(scene.s_hat @ scene.u_hat) if t > 0 else 1.0
    beta = math.degrees(math.acos(max(-1.0, min(1.0, cosb)))) if t > 0 else 0.0
    mag = math.sqrt(max(k * k + t * t - 2 * k * t * cosb, 0.0))
    return {
        "sun_elevation_deg": w["sun_elevation_deg"], "sun_azimuth_deg": w["sun_azimuth_deg"],
        "sensor_offnadir_deg": w["sensor_offnadir_deg"], "sensor_azimuth_deg": w["sensor_azimuth_deg"],
        "gsd_m_per_px": w["gsd_m_per_px"],
        "k": k, "t": t, "beta_deg": beta, "a_s_magnitude": mag,
        "a_s_vector": scene.a_s.tolist(), "b_vector": scene.b.tolist(), "A_jacobian": scene.A.tolist(),
        "affine_approx_valid": True,  # the satellite camera is orthographic: y = A x + b z + c is exact
        "near_nadir_flag": t < 0.05,
        "hotspot_flag": t > 0 and beta < 10.0 and abs(k - t) < 0.1 * max(k, t),
        "same_side_geometry": t > 0 and cosb > 0,
    }


def observability(scene, sigma_px=0.5):
    """Observability measures. sigma_px is the expected boundary localisation noise (from Layer C)."""
    a_s, b, A = scene.a_s, scene.b, scene.A
    na, nb = float(np.linalg.norm(a_s)), float(np.linalg.norm(b))
    dot = float(abs(_unit(a_s) @ _unit(b))) if na > 1e-9 and nb > 1e-9 else None
    # unknowns (Delta_R, h_x, h_y); rows: inner-shadow shift (2), roof-edge shift (2)
    J = np.zeros((4, 3))
    J[0:2, 0] = a_s
    J[2:4, 0] = -b
    J[2:4, 1:3] = A
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > 1e-6 * sv[0]))
    J1 = np.concatenate([a_s, -b])
    H_F = scene.w["H_F_m"]
    sens = math.sqrt(na * na + nb * nb)
    shadow_par = scene.A @ scene.q_s
    return {
        "a_s_vector_x": float(a_s[0]), "a_s_vector_y": float(a_s[1]), "a_s_magnitude": na,
        "b_vector_x": float(b[0]), "b_vector_y": float(b[1]), "parallax_sensitivity_t": scene.t,
        "state_jacobian_shadow": na, "state_jacobian_parallax": nb,
        "shadow_parallax_dot": dot,
        "condition_number_joint": float(sv[0] / sv[-1]) if sv[-1] > 1e-9 else None,
        "jacobian_rank_with_offset": rank,
        "identifiable_single_view": bool(np.linalg.norm(J1) * H_F > sigma_px),
        "identifiable_with_offset": bool(rank == 3 and sv[-1] * H_F > sigma_px),
        "shadow_parallax_cancellation": bool(na < 0.2 * np.linalg.norm(shadow_par)),
        "resolution_limit_phi_R": float(sigma_px / (H_F * sens)) if sens > 0 else None,
        "sigma_px_assumed": sigma_px,
    }
