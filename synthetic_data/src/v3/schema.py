"""
Label blocks of a render (world truth, projected truth, anchors, observability) from the exact Blender camera.

legacy_layers() is called by render.py for every image. Units: metres, degrees, image pixels (x right, y down,
origin at the top-left corner).
"""
import math

import numpy as np

from . import physics as ph

SCHEMA_VERSION = "3.0"

CHANNEL_SHORT = {"inner_shadow": "shadow", "roof_edge": "roof", "ground_shadow": "gnd", "tank_base": "base"}
DELTA_INDEX = {"inner_shadow": 1, "roof_edge": 2, "ground_shadow": 3, "tank_base": 4}


def _f(x, nd=4):
    if x is None:
        return None
    return round(float(x), nd)


def _vec(v, nd=4):
    return [round(float(c), nd) for c in v]


def rim_ellipse(scene):
    """Exact image ellipse of the outer rim circle (orthographic camera: the image of a horizontal circle
    of radius R is an ellipse with semi-axes R * singular values of the 2x2 Jacobian A)."""
    w, cam = scene.w, scene.cam
    c = cam.project(np.array([0.0, 0.0, w["z_C_m"]]))
    U, S, _ = np.linalg.svd(scene.A)
    R = w["radius_outer_R_m"]
    return {"cx_px": _f(c[0]), "cy_px": _f(c[1]), "a_px": _f(R * S[0]), "b_px": _f(R * S[1]),
            "angle_deg": _f(math.degrees(math.atan2(U[1, 0], U[0, 0])) % 180.0, 3)}


def _arc_len(pts):
    pts = np.asarray(pts, float)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0


# ---------------------------------------------------------------------------------------------
# Layer A / B / C / D blocks
# ---------------------------------------------------------------------------------------------
def world_truth_block(world, spec):
    return {
        "tank_center_world": [world["center_world_x_m"], world["center_world_y_m"], world["z_G_m"]],
        "radius_m": world["radius_outer_R_m"], "radius_inner_m": world["radius_inner_m"],
        "roof_radius_m": world["roof_radius_m"],
        "z_C_m": world["z_C_m"], "z_R_m": world["z_R_m"], "z_F_m": world["z_F_m"], "z_G_m": world["z_G_m"],
        "Delta_R_m": world["Delta_R_m"], "H_F_m": world["H_F_m"], "H_G_m": world["H_G_m"], "phi_R": world["phi_R"],
        "roof_offset_x_m": world["roof_offset_x_m"], "roof_offset_y_m": world["roof_offset_y_m"],
        "roof_tilt_deg": world["roof_tilt_deg"], "roof_tilt_azimuth_deg": world["roof_tilt_azimuth_deg"],
        "seal_gap_m": world["seal_gap_g_m"], "edge_offset_m": world["roof_edge_offset_m"],
        "eta_edge": world["eta_edge_type"], "terrain_slope_deg": world["terrain_slope_deg"],
        "terrain_slope_azimuth_deg": world["terrain_slope_azimuth_deg"],
        # The shadow caster is the top of the wall: its height above the terrain that receives the
        # ground shadow (H_G). caster_height_assumed_m is what the estimators were told.
        "caster_height_m": world["caster_height_true_m"] - world["z_G_m"],
        "caster_height_assumed_m": world["caster_height_assumed_m"] - world["z_G_m"],
        "caster_type": world["caster_type"],
    }


def projected_truth_block(scene, pt, aux, mask_paths=None, occluders=None):
    """Projected truth: boundary curves and measurements, from the exact camera (subpixel, pre-rasterisation)."""
    ch_rec = pt["channels"]
    rim_c = np.array([pt["gt_rim_center_x_px"], pt["gt_rim_center_y_px"]])
    betas = aux["betas"]
    step = max(1, int(round(2.0 / (betas[1] - betas[0]))))
    rim_pts = aux["px"]["rim_outer"]
    rim_loop = np.vstack([rim_pts, rim_pts[:1]])
    out = {"rim_ellipse": rim_ellipse(scene),
           "rim_arc_points": np.round(rim_pts[::step], 3).tolist(),
           "arc_len_rim_px": _f(_arc_len(rim_loop), 3)}
    for ch in ph.CHANNELS:
        r = ch_rec[ch]
        name = {"inner_shadow": "inner_shadow_boundary", "roof_edge": "roof_edge_boundary",
                "ground_shadow": "ground_shadow_boundary", "tank_base": "tank_base_boundary"}[ch]
        out[name] = r["boundary_points_px"]
        i = DELTA_INDEX[ch]
        out[f"delta_{i}_px"] = [_f(r["delta_x_px"]), _f(r["delta_y_px"])]
        out[f"delta_{i}_magnitude_px"] = _f(r["delta_magnitude_px"])
        out[f"arc_len_{CHANNEL_SHORT[ch]}_px"] = _f(r["arc_length_px"] or 0.0, 3)
        out[f"arc_visible_fraction_{CHANNEL_SHORT[ch]}"] = _f(r["arc_visible_fraction"])
    d1 = np.array(out["delta_1_px"], float)
    out["shadow_arc_center_px"] = _vec(rim_c + d1)
    out["shadow_offset_x_px"], out["shadow_offset_y_px"] = _f(d1[0]), _f(d1[1])
    out["delta_roof_center_px"] = [_f(pt["gt_delta_roof_center_x_px"]), _f(pt["gt_delta_roof_center_y_px"])]
    out["delta_roof_center_x_px"], out["delta_roof_center_y_px"] = out["delta_roof_center_px"]
    mp = mask_paths or {}
    for ch, key in (("inner_shadow", "inner_shadow"), ("roof_edge", "roof_edge"),
                    ("ground_shadow", "ground_shadow"), ("tank_base", "base")):
        out[f"visibility_mask_{key}"] = mp.get(f"mask_{ch}")
    occ = occluders or {}
    out["occlusion_flags"] = {("base" if ch == "tank_base" else ch):
                              bool(ch_rec[ch]["occluded_by"] != "none" or occ.get(ch)) for ch in ph.CHANNELS}
    out["occluded_by"] = {("base" if ch == "tank_base" else ch):
                          sorted(set(([ch_rec[ch]["occluded_by"]] if ch_rec[ch]["occluded_by"] != "none" else []) +
                                     list(occ.get(ch) or []))) for ch in ph.CHANNELS}
    a_hat = ph._unit(scene.a_s)
    u_img = ph._unit(scene.A @ scene.u_hat) if scene.t > 0 else np.zeros(2)
    out["translation_direction_a_s"] = _vec(a_hat)
    out["translation_direction_u_hat"] = _vec(u_img)
    out["translation_direction_a_v"] = _vec(ph._unit(-scene.b))
    out["a_s_px_per_m"] = _vec(scene.a_s)
    out["b_px_per_m"] = _vec(scene.b)
    k, t = scene.k, scene.t
    cosb = float(scene.s_hat @ scene.u_hat) if t > 0 else 1.0
    out["effective_shadow_sensitivity"] = _f(math.sqrt(max(k * k + t * t - 2 * k * t * cosb, 0.0)))
    out["effective_parallax_sensitivity"] = _f(t)
    ell = out["rim_ellipse"]
    out["diameter_px"] = _f(2.0 * ell["a_px"], 3)
    out["radius_px"] = _f(ell["a_px"], 3)
    out["radius_m"] = scene.w["radius_outer_R_m"]
    # Per channel: translation, arc, chord, sagitta, crescent tips (rim intersections), visible segments
    out["channels"] = {ch: {k: (_f(v) if isinstance(v, float) else v) for k, v in ch_rec[ch].items()
                            if k != "boundary_points_px"} for ch in ph.CHANNELS}
    out["caster_shadow_pairs"] = pt["gt_caster_shadow_pairs"]
    return out


# ---------------------------------------------------------------------------------------------
# Per-scene files
# ---------------------------------------------------------------------------------------------
def anchor_annotations(scene_id, tank_id, scene, pt, sets, mask_paths=None, occluders=None):
    """Anchor annotations. The canonical anchor list is the information-weighted K=16 set of every channel."""
    anchors, summary = [], {}
    occluders = occluders or {}
    for ch in ph.CHANNELS:
        sel = sets.get((ch, "information_weighted", 16)) or []
        r = pt["channels"][ch]
        arc = float(r["arc_length_px"] or 0.0)
        for a in sel:
            vis = a.get("actual_visibility") or a.get("predicted_visibility")
            anchors.append({
                "anchor_id": a["anchor_id"], "channel": ch,
                "anchor_angle_deg": _f(a["anchor_angle_deg"], 3),
                "anchor_x_px": _f(a["anchor_x_px"], 3), "anchor_y_px": _f(a["anchor_y_px"], 3),
                "shadow_point_x_px": _f(a["true_boundary_x_px"], 3), "shadow_point_y_px": _f(a["true_boundary_y_px"], 3),
                "translation_px": [_f(a["true_translation_x_px"], 3), _f(a["true_translation_y_px"], 3)],
                "translation_magnitude_px": _f(a["true_translation_magnitude_px"], 3),
                "arc_length_contribution_px": _f(arc / max(len(sel), 1), 3),
                "information_weight": _f(a["info_weight"], 4),
                "boundary_normal": [_f(a["normal_x"]), _f(a["normal_y"])],
                "predicted_visibility": a["predicted_visibility"],
                "actual_visibility": a.get("actual_visibility"),
                "validity_label": bool(a.get("valid", True)) if a.get("valid") is not None else vis == "visible",
                "occlusion_type": None if a.get("occlusion_cause", "none") == "none" else a["occlusion_cause"],
                "failure_type": None if a.get("failure_type", "none") in ("none", None) else a["failure_type"],
                "annotation_confidence": a.get("annotation_confidence", 1.0),
                "observed_boundary_px": ([_f(a["obs_boundary_x_px"], 3), _f(a["obs_boundary_y_px"], 3)]
                                         if a.get("obs_boundary_x_px") is not None else None),
                "strip_path": a.get("strip_path"),
            })
        frac = r["arc_visible_fraction"]
        hidden = r["occluded_by"] != "none" or bool(occluders.get(ch))
        vis = "occluded" if frac == 0 else ("partially_visible" if hidden else "visible")
        fails = [a.get("failure_type") for a in sel if a.get("failure_type") not in (None, "none")]
        summary[ch] = {"arc_len_px": _f(arc, 3), "n_anchors": len(sel),
                       "mean_info_weight": _f(np.mean([a["info_weight"] for a in sel]), 4) if sel else None,
                       "visibility": vis,
                       "failure": (max(set(fails), key=fails.count) if fails else
                                   ((occluders.get(ch) or [None])[0] if hidden and occluders.get(ch) else None))}
    return {"image_id": scene_id, "tank_id": tank_id, "annotator": "blender_gt",
            "annotation_type": "simulator_exact", "rim_ellipse": rim_ellipse(scene),
            "anchors": anchors, "channel_summary": summary,
            "visibility_masks": {k: v for k, v in (mask_paths or {}).items() if k.startswith("mask_")}}


def observability_diagnostics(scene_id, scene, acq, obs, pt, sets, spec, occluders=None):
    """Observability diagnostics (shadow sensitivity is the dimensionless |a_s| = sqrt(k^2 + t^2 - 2 k t cos beta))."""
    k, t = scene.k, scene.t
    a_mag = acq["a_s_magnitude"]
    beta = acq["beta_deg"]
    relation = ("nadir" if t < 1e-6 else "same_side" if beta < 60 else "opposite_side" if beta > 120 else "perpendicular")
    hot = max(0.0, 1.0 - a_mag / (k + t)) if (k + t) > 0 else None
    ch_px = {}
    for ch in ph.CHANNELS:
        r = pt["channels"][ch]
        ch_px[ch] = _f(r["delta_magnitude_px"], 3) if r["arc_visible_fraction"] > 0 else None
    weights = [a["info_weight"] for a in (sets.get(("inner_shadow", "information_weighted", 16)) or [])]
    diam_px = 2.0 * scene.w["radius_outer_R_m"] / scene.w["gsd_m_per_px"]
    occ = occluders or {}
    occl = "none" if not any(occ.values()) else ("heavy" if sum(bool(v) for v in occ.values()) >= 2 else "partial")
    multi = []
    if not obs["identifiable_with_offset"]:
        multi += ["Delta_R_with_roof_offset"]
    if t < 0.05:
        multi += ["parallax_channel"]
    return {
        "scene_id": scene_id,
        "shadow_sensitivity_a_s": _f(a_mag), "shadow_sensitivity_px_per_m": _f(obs["a_s_magnitude"]),
        "parallax_sensitivity_t": _f(t), "parallax_sensitivity_px_per_m": _f(obs["state_jacobian_parallax"]),
        "k_cot_sun_elevation": _f(k),
        "sun_sensor_azimuth_relation": relation, "beta_deg": _f(beta, 3),
        "hotspot_proximity": _f(hot), "hotspot_flag": acq["hotspot_flag"],
        "channel_translation_px": ch_px,
        "information_weights_per_anchor": [_f(x) for x in weights],
        "state_jacobian_norm": _f(math.hypot(obs["a_s_magnitude"], obs["state_jacobian_parallax"])),
        "condition_number": _f(obs["condition_number_joint"]),
        "jacobian_rank_with_offset": obs["jacobian_rank_with_offset"],
        "parallax_shadow_cancellation_risk": obs["shadow_parallax_cancellation"],
        "near_nadir_parallax_risk": acq["near_nadir_flag"],
        "lateral_offset_identifiable": obs["identifiable_with_offset"],
        "identifiable_single_view": obs["identifiable_single_view"],
        "resolution_limit_phi_R": _f(obs["resolution_limit_phi_R"], 6),
        "sigma_px_assumed": _f(obs["sigma_px_assumed"]),
        "multi_view_required_for": multi,
        "operating_condition_bins": {
            "shadow_sensitivity": "low" if a_mag < 0.5 else "medium" if a_mag < 1.0 else "high",
            "parallax_sensitivity": "low" if t < 0.1 else "medium" if t < 0.3 else "high",
            "sun_sensor_relation": relation,
            "tank_diameter_px": "small" if diam_px < 80 else "medium" if diam_px < 200 else "large",
            "contrast": {"CLEAR": "easy", "HAZE": "medium", "THIN_CLOUD": "hard"}.get(spec["atmosphere"], "unknown"),
            "occlusion": occl,
        },
    }


def legacy_layers(radius, height, fill_level, sun_el, sun_az, cam_el, cam_az, atmosphere, camera_export, full=None,
                  seal_gap=0.05, roof_offset=(0.0, 0.0), roof_tilt=(0.0, 0.0), edge_type="none", edge_height=0.0):
    """
    v3 truth layers for an image from the balanced generator (generate_dataset.py / render_tank), whose
    roof follows the legacy fill mapping: deck top z_R = 0.2 + fill * (H - 0.3), floor z_F = 0.2 (the
    lowest deck), terrain z_G = 0. The roof geometry (seal gap, roof offset and tilt, edge structure) is the
    tank's own; the defaults are the v1 tank (0.05 m gap, centred level deck, no edge). camera_export is the "camera" block
    of the Blender --export-json file. Returns (world_truth, projected_truth summary, observability).
    If full is a dict, it is filled with the complete record (boundary curves, rim points, caster-shadow
    pairs, K-point anchors) for the per-image label file.
    """
    z_F = 0.2
    spec = {"R": radius, "H": height, "z_F": z_F, "phi_R": fill_level * (height - 0.3) / (height - z_F),
            "seal_gap": seal_gap, "edge_type": edge_type, "edge_height": edge_height,
            "roof_offset": tuple(roof_offset), "roof_tilt": tuple(roof_tilt), "sun_el": sun_el, "sun_az": sun_az,
            "cam_el": cam_el, "cam_az": cam_az, "atmosphere": atmosphere,
            "gsd": camera_export["ortho_scale"] / camera_export["res_x"]}
    world = ph.world_truth(spec)
    scene = ph.Scene(world, ph.Camera.from_blender(camera_export))
    pt, aux = ph.projected_truth(scene)
    proj = projected_truth_block(scene, pt, aux)
    if full is not None:
        from . import measure
        sigma = math.sqrt(1.0 / 12.0)
        sets = measure.anchor_sets(scene, aux, sigma, "image", "tank")
        full.update({"world_truth": world_truth_block(world, spec), "projected_truth": dict(proj),
                     "acquisition": ph.acquisition_fields(scene),
                     "anchor_annotations": anchor_annotations("image", "tank", scene, pt, sets)})
    for key in ("rim_arc_points", "inner_shadow_boundary", "roof_edge_boundary", "ground_shadow_boundary",
                "tank_base_boundary", "caster_shadow_pairs") + tuple(k for k in proj if k.startswith("visibility_mask")):
        proj.pop(key, None)   # keep metadata.json small: the exact curves follow from the camera and world truth
    acq = ph.acquisition_fields(scene)
    obs = observability_diagnostics(None, scene, acq, ph.observability(scene, math.sqrt(1.0 / 12.0)), pt, {},
                                    {"atmosphere": atmosphere})
    obs.pop("scene_id")
    obs.pop("information_weights_per_anchor")
    return world_truth_block(world, spec), proj, obs


