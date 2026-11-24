"""
K-point anchors on the visible arc of each boundary channel (uniform_visible and information_weighted schemes),
written to the per-image label.
"""
import numpy as np

from .physics import CHANNELS, _segments, _run_indices

K_VALUES = (1, 2, 4, 8, 16)
SCHEMES = ("uniform_visible", "information_weighted")


# ---------------------------------------------------------------------------------------------
# Anchors (Layer B truth per anchor)
# ---------------------------------------------------------------------------------------------
def _normals(px):
    t = np.gradient(px, axis=0)
    t[0], t[-1] = px[1] - px[-1], px[0] - px[-2]
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def place_anchors(scene, aux, ch, K, scheme, sigma_px, scene_id, tank_id):
    betas, vis = aux["betas"], aux["visible"][ch]
    n = len(betas)
    runs = _segments(vis)
    if not runs:
        return []
    a_hat, _ = scene.channel_direction(ch)
    bpx, apx = aux["px"][ch], aux["px"][scene.anchor_key(ch)]
    normals = _normals(bpx)
    cos_na = normals @ a_hat
    normals = normals * np.sign(np.where(cos_na == 0, 1, cos_na))[:, None]
    cos_na = np.abs(cos_na)
    info = cos_na ** 2 / sigma_px ** 2
    main = max(runs, key=lambda r: (r[1] - r[0]) % n)
    main_idx = _run_indices(main, n)
    all_idx = np.concatenate([_run_indices(r, n) for r in runs])
    if scheme == "uniform_visible":
        picks = [int(main_idx[min(len(main_idx) - 1, int((j + 0.5) / K * len(main_idx)))]) for j in range(K)]
    else:
        order = all_idx[np.argsort(-info[all_idx])]
        span = len(all_idx) * (betas[1] - betas[0])
        min_sep = max(span / (2.0 * K), 1.0)
        picks = []
        for i in order:
            if all(abs(((betas[i] - betas[p]) + 180) % 360 - 180) >= min_sep for p in picks):
                picks.append(int(i))
            if len(picks) == K:
                break
    ends = [betas[r[0]] for r in runs] + [betas[r[1]] for r in runs]
    p1, p2 = bpx[main_idx[0]], bpx[main_idx[-1]]
    chord = p2 - p1
    cl = max(np.linalg.norm(chord), 1e-9)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(bpx[main_idx], axis=0), axis=1))])
    pos_in_main = {int(i): j for j, i in enumerate(main_idx)}
    out = []
    short = "uv" if scheme == "uniform_visible" else "iw"
    for k, i in enumerate(picks):
        tr = bpx[i] - apx[i]
        j = pos_in_main.get(i)
        out.append({
            "anchor_id": f"{scene_id}_{ch}_{short}_K{K}_k{k:03d}", "scene_id": scene_id, "tank_id": tank_id,
            "channel": ch, "k_index": k, "K_total": len(picks), "sampling_scheme": scheme,
            "anchor_angle_deg": float(betas[i]), "anchor_x_px": float(apx[i][0]), "anchor_y_px": float(apx[i][1]),
            "arc_pos_norm": float(j / max(len(main_idx) - 1, 1)) if j is not None else None,
            "dist_from_tip_deg": float(min(abs(((betas[i] - e) + 180) % 360 - 180) for e in ends)),
            "true_boundary_x_px": float(bpx[i][0]), "true_boundary_y_px": float(bpx[i][1]),
            "true_translation_x_px": float(tr[0]), "true_translation_y_px": float(tr[1]),
            "true_translation_magnitude_px": float(np.linalg.norm(tr)), "true_d_k_px": float(tr @ a_hat),
            "channel_dir_x": float(a_hat[0]), "channel_dir_y": float(a_hat[1]),
            "normal_x": float(normals[i][0]), "normal_y": float(normals[i][1]),
            "info_weight": float(info[i]), "cos_n_a": float(cos_na[i]),
            "chord_to_anchor_px": float(min(cum[j], cum[-1] - cum[j])) if j is not None else None,
            "sagitta_at_anchor_px": float(abs(chord[0] * (bpx[i] - p1)[1] - chord[1] * (bpx[i] - p1)[0]) / cl),
            "predicted_visibility": "grazing" if cos_na[i] < 0.2 else "visible",
            "actual_visibility": None, "occlusion_cause": "none",
            "annotation_confidence": 1.0,
        })
    return out


def anchor_sets(scene, aux, sigma_px, scene_id, tank_id):
    """All K x scheme anchor sets for every channel: {(channel, scheme, K): [anchors]}."""
    return {(ch, sch, K): place_anchors(scene, aux, ch, K, sch, sigma_px, scene_id, tank_id)
            for ch in CHANNELS for sch in SCHEMES for K in K_VALUES}


