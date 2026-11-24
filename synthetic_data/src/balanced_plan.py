"""
Plans a dataset of a fixed size in which every factor is uniformly distributed:
tank size, height, fill level, location (country), sun-height band, atmosphere and camera elevation.
Continuous factors are balanced on bins and drawn uniformly inside them, so the marginals are even
without repeated values (v1 used 4 fixed sizes, 8 fill levels and 4 camera angles, which a model can
snap to or read shortcuts from).

- size class x height bin x fill bin is fully crossed with equal counts. A size class is a radius range:
  the radius is drawn log-uniformly inside it and the height uniformly inside its height bin, so radius
  and height are independent (v1 had one fixed height per radius).
- location x sun band: each location and each band gets the same total. Bands a location's sun
  never reaches (e.g. above 53 deg at Primorsk) stay empty, and the other locations make up those bands.
  The real date and time for each sample are drawn from the times the sun is in that band there.
- atmosphere x camera-elevation bin is fully crossed with near-equal counts.
The three groups are shuffled against each other, and the shuffle with the most even pairwise
combinations (e.g. size x country, fill x sun band) is kept.

Then per image: camera azimuth, ground colour, layout, zoom OR a fixed NAIP ground resolution
(naip_scale_share), the labelled tank's roof geometry (tank_variation) and its held-out tags (holdout:
shape box, low sun, held-out site; the site gets its own balanced sub-plan). The legacy settings
(fixed tank_sizes, fill_levels, camera_angles, no tank_variation / holdout) still plan a v1-style set.

Every image must show part of the roof in sunlight. When the sun is too low for the tank's ullage, the
rim shadows the whole roof, and an empty tank looks the same as a quarter-full one, so the fill label
cannot be read from the image. Such (size, fill, sun) combinations are never planned (see min_sun_for).
Check a saved plan for shortcuts with src/audit_plan.py before rendering.
"""
import math
import random
import itertools
from collections import Counter
from datetime import date, timedelta

import geometry


def sun_candidates(location, bands, time_range, year=2025, day_step=2, minute_step=10):
    """All (date, local solar time, elevation, azimuth) in the time range, grouped by sun band."""
    pools = {i: [] for i in range(len(bands))}
    t0, t1 = time_range
    for d in range(0, 365, day_step):
        day = (date(year, 1, 1) + timedelta(days=d)).isoformat()
        minutes = int(round(t0 * 60))
        while minutes <= int(round(t1 * 60)):
            lst = minutes / 60.0
            sun = geometry.sun_for_location(location, day, lst)
            for i, (lo, hi) in enumerate(bands):
                if lo <= sun["elevation"] < hi:
                    pools[i].append((day, lst, sun["elevation"], sun["azimuth"]))
                    break
            minutes += minute_step
    return pools


# The internal shadow may cover at most this fraction of the roof diameter, so a lit part always remains
MAX_SHADOW_FRACTION = 0.85

# On tilted views the wall nearest the satellite hides part of the roof. At least this fraction of the
# roof must be both visible and sunlit (or 70% of what a nadir view would show, if that is smaller), so
# the fill can be read from the image. 3% was tested and the sunlit sliver was not visible at 60 deg.
MIN_VISIBLE_LIT_ROOF = 0.10


def _window(preset, default):
    """Acquisition window (local solar hours) for a location: its own acquisition_window, else the default."""
    w = preset.get("acquisition_window")
    if isinstance(w, str):  # the fallback YAML parser leaves lists as text
        w = [float(v) for v in w.strip("[] ").split(",")]
    return tuple(float(v) for v in w) if w else default


def min_sun_for(radius, height, fill):
    """Lowest sun elevation (deg) at which part of the roof is still lit for this tank and fill."""
    ullage = height - (fill * (height - 0.3) + 0.2)  # same geometry as the render and the metadata
    return math.degrees(math.atan2(max(ullage, 0.0), MAX_SHADOW_FRACTION * 2.0 * radius))


def _split_evenly(total, n):
    return [total // n + (1 if i < total % n else 0) for i in range(n)]


def _balanced_counts(total, dims, rng):
    """
    Counts for a fully crossed table with dims (e.g. sizes x height bins x fill bins) summing to total, with
    EVERY factor's marginal as even as possible on its own. When total does not divide evenly, each extra
    unit goes to a cell whose factor levels have received the fewest extras so far, ties broken at random
    (seeded rng). Filling cells in index order instead would give a small sub-plan (e.g. 20 images of a
    held-out site over 4 x 4 x 20 cells) only the lowest height bin.
    """
    cells = list(itertools.product(*[range(n) for n in dims]))
    base, extra = divmod(total, len(cells))
    counts = {c: base for c in cells}
    extras = [[0] * n for n in dims]
    free = cells[:]
    rng.shuffle(free)
    for _ in range(extra):
        best = min(range(len(free)), key=lambda i: sum(extras[d][free[i][d]] for d in range(len(dims))))
        c = free.pop(best)
        counts[c] += 1
        for d in range(len(dims)):
            extras[d][c[d]] += 1
    return counts


def allocate(row_totals, col_totals, feasible, iterations=500):
    """Integer table with the given row/column sums, zero where infeasible, as even as possible."""
    rows, cols = len(row_totals), len(col_totals)
    # Iterative proportional fitting on the feasibility mask gives the most even fractional table
    x = [[1.0 if feasible[r][c] else 0.0 for c in range(cols)] for r in range(rows)]
    for _ in range(iterations):
        for r in range(rows):
            s = sum(x[r])
            if s > 0:
                x[r] = [v * row_totals[r] / s for v in x[r]]
        for c in range(cols):
            s = sum(x[r][c] for r in range(rows))
            if s > 0:
                for r in range(rows):
                    x[r][c] *= col_totals[c] / s
    table = [[int(math.floor(v)) for v in row] for row in x]
    # Hand out the remaining units to the largest fractional parts that both sums still need
    order = sorted(((x[r][c] - table[r][c], r, c) for r in range(rows) for c in range(cols) if feasible[r][c]),
                   reverse=True)
    while True:
        row_need = [row_totals[r] - sum(table[r]) for r in range(rows)]
        col_need = [col_totals[c] - sum(table[r][c] for r in range(rows)) for c in range(cols)]
        if not any(row_need) and not any(col_need):
            return table
        for _, r, c in order:
            if row_need[r] > 0 and col_need[c] > 0:
                table[r][c] += 1
                break
        else:
            # Row r and column c both need a unit but (r, c) is infeasible (e.g. Russia and a
            # high-sun band). Route it through another cell: r takes c2 from row r2, r2 takes c.
            r = next(i for i in range(rows) if row_need[i] > 0)
            c = next(j for j in range(cols) if col_need[j] > 0)
            swap = next(((r2, c2) for r2 in range(rows) for c2 in range(cols)
                         if r2 != r and c2 != c and feasible[r][c2] and feasible[r2][c] and table[r2][c2] > 0),
                        None)
            if swap:
                r2, c2 = swap
                table[r][c2] += 1
                table[r2][c2] -= 1
                table[r2][c] += 1
                continue
            raise ValueError("No balanced location x sun-band table exists for these settings "
                             "(a band is reachable at too few locations).")


def _range(v):
    """A [lo, hi] setting as two floats (the fallback YAML parser leaves lists as text)."""
    if isinstance(v, str):
        v = v.strip("[] ").split(",")
    return tuple(float(x) for x in v)


def _bins(lo, hi, n):
    return [(lo + (hi - lo) * k / n, lo + (hi - lo) * (k + 1) / n) for k in range(n)]


def _bin_label(b):
    return f"{b[0]:g}-{b[1]:g}"


def tank_shapes(config, s):
    """
    (size classes, height bins). A class with radius_range gets a log-uniform radius and the height comes
    from the shared height_range, split into height_bins bins. Legacy classes (fixed radius and height, no
    radius_range) keep both fixed: height bins is then None.
    """
    sizes = list(config["tank_sizes"].items())
    continuous = [("radius_range" in size) for _, size in sizes]
    if any(continuous) and not all(continuous):
        raise ValueError("tank_sizes: give every class a radius_range, or none of them (legacy fixed sizes)")
    if not all(continuous):
        return sizes, None
    lo, hi = _range(s.get("height_range", [9.0, 22.0]))
    return sizes, _bins(lo, hi, int(s.get("height_bins", 4)))


def fill_levels(s):
    """Fill bins [(lo, hi), ...] when fill_range is set (continuous), else the legacy fixed fill_levels."""
    if "fill_range" in s:
        return _bins(*_range(s["fill_range"]), int(s.get("fill_bins", 20))), True
    return list(s.get("fill_levels", [0.0, 0.25, 0.5, 0.75, 1.0])), False


def camera_levels(s):
    """Camera-elevation bins when camera_elevation_range is set (continuous), else the legacy camera_angles."""
    if "camera_elevation_range" in s:
        return _bins(*_range(s["camera_elevation_range"]), int(s.get("camera_elevation_bins", 6))), True
    return (s.get("camera_angles") or [{"elevation": 90.0, "azimuth": 0.0}, {"elevation": 80.0, "azimuth": 135.0},
                                       {"elevation": 70.0, "azimuth": 270.0}]), False


def holdout_tags(p, hold):
    """
    Held-out designs this sample belongs to (written into its metadata as "holdout"; such images never go
    to train or val): "shape" (radius/height inside a held-out box), "sun" (sun below sun_below) and
    "site" (a held-out location).
    """
    tags = []
    for r_lo, r_hi, h_lo, h_hi in (hold.get("shape_boxes") or []):
        if r_lo <= p["radius"] <= r_hi and h_lo <= p["height"] <= h_hi:
            tags.append("shape")
            break
    if hold.get("sun_below") is not None and p["sun_elevation"] < float(hold["sun_below"]):
        tags.append("sun")
    if p["location"] in (hold.get("locations") or []):
        tags.append("site")
    return tags


def _core_plan(config, s, names, total, seed, tries):
    """
    The balanced part of the plan for these locations: size class x height bin x fill, location x sun band,
    atmosphere x camera, each sample with its continuous radius, height, fill and camera elevation, and a
    real date and time at which its roof is partly lit.
    """
    presets = config["locations"]
    sizes, height_bins = tank_shapes(config, s)
    n_hbins = len(height_bins) if height_bins else 1
    fills, fill_cont = fill_levels(s)
    cameras, cam_cont = camera_levels(s)
    bands = [tuple(b) for b in s.get("sun_bands", [[10, 25], [25, 40], [40, 55], [55, 70], [70, 90]])]
    time_range = tuple(s.get("local_solar_time_range", [8.0, 16.0]))
    atmospheres = s.get("atmospheres") or ["CLEAR", "HAZE", "THIN_CLOUD"]
    min_elev = float(config.get("sun_limits", {}).get("min_sun_elevation", 10.0))
    bands = [(max(lo, min_elev), hi) for lo, hi in bands]

    rng = random.Random(seed)
    # Each location can have its own acquisition window (e.g. Primorsk at 60 N: ~10:30 satellites only)
    pools = {n: sun_candidates(presets[n], bands, _window(presets[n], time_range)) for n in names}
    # Bands that none of these locations reach are left out (e.g. a held-out site at 52 N has no 70-90 sun)
    keep = [b for b in range(len(bands)) if any(pools[n][b] for n in names)]
    bands = [bands[b] for b in keep]
    pools = {n: {i: pools[n][b] for i, b in enumerate(keep)} for n in names}
    feasible = [[len(pools[n][b]) > 0 for b in range(len(bands))] for n in names]
    table = allocate(_split_evenly(total, len(names)), _split_evenly(total, len(bands)), feasible)
    top_sun = {(li, bi): max((c[2] for c in pools[names[li]][bi]), default=-1.0)
               for li in range(len(names)) for bi in range(len(bands))}
    highest = max(top_sun.values())

    # Group 1: size class x height bin x fill, fully crossed (columns are height bin x fill). Each sample
    # draws its own radius (log-uniform in its class), height and fill (uniform in their bins). A draw whose
    # roof would be fully shadowed even at the highest sun these locations offer is drawn again inside the
    # same class and bins (only tall, narrow, near-empty tanks at high-latitude sites are affected).
    shape_rng = random.Random(f"shape-{seed}")
    cell_rng = random.Random(f"cells-{seed}")

    def draw(si, hi, fi):
        size = sizes[si][1]
        for _ in range(200):
            if height_bins:
                r_lo, r_hi = _range(size["radius_range"])
                radius = round(math.exp(shape_rng.uniform(math.log(r_lo), math.log(r_hi))), 2)
                height = round(shape_rng.uniform(*height_bins[hi]), 2)
            else:
                radius, height = float(size["radius"]), float(size["height"])
            fill = round(shape_rng.uniform(*fills[fi]), 4) if fill_cont else float(fills[fi])
            if min_sun_for(radius, height, fill) <= highest or not (height_bins or fill_cont):
                break
        return radius, height, fill

    group1 = []
    for (si, hi, fi), k in sorted(_balanced_counts(total, (len(sizes), n_hbins, len(fills)), cell_rng).items()):
        for _ in range(k):
            group1.append((si, hi, fi) + draw(si, hi, fi))
    # Group 2: location x sun band from the balanced table
    group2 = [(r, c) for r in range(len(names)) for c in range(len(bands)) for _ in range(table[r][c])]
    # Group 3: atmosphere x camera, fully crossed
    group3 = [cell for cell, k in sorted(_balanced_counts(total, (len(atmospheres), len(cameras)), cell_rng).items())
              for _ in range(k)]

    # Shuffle groups 2 and 3 against group 1; keep the most even pairwise mix
    def pair_score(g2, g3):
        rows = [(a[0], a[1], a[2], b[0], b[1], c[0], c[1]) for a, b, c in zip(group1, g2, g3)]
        worst = 0.0
        for i, j in itertools.combinations(range(7), 2):
            if {i, j} <= {0, 1, 2} or {i, j} in ({3, 4}, {5, 6}):
                continue  # pairs fixed by construction
            counts = Counter((r[i], r[j]) for r in rows)
            ci, cj = Counter(r[i] for r in rows), Counter(r[j] for r in rows)
            for (a, b), n in counts.items():
                expected = ci[a] * cj[b] / total
                worst = max(worst, abs(n - expected) / max(expected, 1.0))
            # a pair cell that should exist but is empty
            for a in ci:
                for b in cj:
                    if (a, b) not in counts and ci[a] * cj[b] / total >= 1.0:
                        worst = max(worst, 1.0)
        return worst

    # Lowest usable sun for each sample's tank and fill
    min_sun = [min_sun_for(radius, height, fill) for _, _, _, radius, height, fill in group1]

    def ok(i, cell):
        return top_sun[cell] >= min_sun[i]

    def repair(g2):
        """Swap location-band cells between samples until every sample has a lit roof."""
        for i in range(total):
            if ok(i, g2[i]):
                continue
            for j in rng.sample(range(total), total):
                if ok(i, g2[j]) and ok(j, g2[i]):
                    g2[i], g2[j] = g2[j], g2[i]
                    break
            else:
                return False
        return True

    best = None
    for _ in range(tries):
        g2, g3 = group2[:], group3[:]
        rng.shuffle(g2)
        rng.shuffle(g3)
        if not repair(g2):
            continue
        score = pair_score(g2, g3)
        if best is None or score < best[0]:
            best = (score, g2, g3)
    if best is None:
        raise ValueError("Could not plan a set where every roof is partly lit; widen the sun bands or time range.")
    _, group2, group3 = best

    # Draw a real date/time per sample, spread over each location-band pool
    cam_rng = random.Random(f"camera-{seed}")
    used = {}
    plan = []
    for idx, ((si, hi, fi, radius, height, fill), (li, bi), (ai, ci)) in enumerate(zip(group1, group2, group3)):
        name = names[li]
        # Only times when the sun is high enough for this tank and fill to keep part of its roof lit. Each
        # location-band pool is used in one shuffled order (every sample has its own lowest usable sun),
        # so dates and times repeat only once the pool runs out.
        full = pools[name][bi]
        if (name, bi) not in used:
            order = list(range(len(full)))
            rng.shuffle(order)
            used[(name, bi)] = order
        order = used[(name, bi)]
        pick = next((k for k in range(len(order) - 1, -1, -1) if full[order[k]][2] >= min_sun[idx]), None)
        if pick is not None:
            day, lst, elev, az = full[order.pop(pick)]
        else:
            day, lst, elev, az = rng.choice([c for c in full if c[2] >= min_sun[idx]])
        size_key, size = sizes[si]
        if cam_cont:
            cam_el, cam_az, cam_bin = round(cam_rng.uniform(*cameras[ci]), 2), 0.0, _bin_label(cameras[ci])
        else:
            cam_el, cam_az = cameras[ci]["elevation"], cameras[ci]["azimuth"]
            cam_bin = f"{cam_el:g}"
        plan.append({
            "size": size_key,
            "folder": size.get("folder", f"size_{size_key}"),
            "radius": radius,
            "height": height,
            "height_bin": _bin_label(height_bins[hi]) if height_bins else f"{height:g}",
            "fill_level": fill,
            "fill_bin": _bin_label(fills[fi]) if fill_cont else f"{fill:g}",
            "location": name,
            "country": presets[name].get("country", ""),
            "sun_band": _bin_label(bands[bi]),
            "date": day,
            "local_solar_time": lst,
            "sun_elevation": round(elev, 3),
            "sun_azimuth": round(az, 3),
            "camera_elevation": cam_el,
            "camera_bin": cam_bin,
            "camera_azimuth": cam_az,
            "atmosphere": atmospheres[ai],
        })
    return plan


def build_plan(config, total=1000, seed=42, tries=300, start_index=0):
    s = config["generation_modes"].get("balanced", {})
    presets = config["locations"]
    hold = s.get("holdout") or {}
    site_names = list(hold.get("locations") or [])
    names = [n for n in (s.get("locations") or sorted(presets)) if n not in site_names]
    # Held-out sites get their own balanced sub-plan (location_share of the images), so the training
    # locations keep their even share and the held-out site is never mixed into their tables
    n_site = int(round(total * float(hold.get("location_share", 0.0)))) if site_names else 0
    plan = _core_plan(config, s, names, total - n_site, seed, tries)
    if n_site:
        plan += _core_plan(config, s, site_names, n_site, f"{seed}-site", tries)
    for i, p in enumerate(plan):
        p["index"] = i
    _add_realism(plan, config, s, seed)
    _add_tank_variation(plan, s, seed)
    _, height_bins = tank_shapes(config, s)
    if height_bins:
        # Neighbouring tanks draw their own heights from the same range (blender --neighbour-height-range)
        for p in plan:
            p["neighbour_height_range"] = [height_bins[0][0], height_bins[-1][1]]
    for p in plan:
        # start_index numbers a follow-up batch after an existing one (b1500, b1501, ...) so names and
        # scene ids never collide when the batches are merged
        p["index"] += start_index
        scale = f"_g{p['render_mpp']:.2f}" if p.get("render_mpp") else f"_z{p['zoom']:.1f}"
        p["filename"] = (f"b{p['index']:04d}_{p['location']}_{p['date'].replace('-', '')}_lst"
                         f"{int(p['local_solar_time']):02d}{int(round((p['local_solar_time'] % 1) * 60)):02d}"
                         f"_r{p['radius']:.1f}_h{p['height']:.1f}_fill_{p['fill_level']:.3f}"
                         f"_camelev{int(p['camera_elevation'])}"
                         f"_camaz{int(p['camera_azimuth'])}{scale}_{p['atmosphere'].lower()}.png")
        # Label notation. Legacy roof mapping: deck top z_R = 0.2 + fill (H - 0.3), floor z_F = 0.2
        H = p["height"]
        z_R = 0.2 + p["fill_level"] * (H - 0.3)
        p.update({"scene_id": f"b{p['index']:04d}", "tank_id": f"tank_synthetic_b{p['index']:04d}",
                  "z_C_m": H, "z_F_m": 0.2, "z_G_m": 0.0, "z_R_m": round(z_R, 4), "Delta_R_m": round(H - z_R, 4),
                  "phi_R": round((z_R - 0.2) / (H - 0.2), 5),
                  "sensor_offnadir_deg": round(90.0 - p["camera_elevation"], 3),
                  "experiment_id": "balanced", "counterfactual_group_id": None, "random_seed": seed,
                  "holdout": holdout_tags(p, hold)})
    return plan


# Every part of the floating roof (deck and edge structure) stays at least this far below the rim
RIM_CLEARANCE = 0.05


def _add_tank_variation(plan, s, seed):
    """
    Per-tank geometry of the labelled tank (tank_variation in the config): seal gap, a roof offset inside
    the seal gap, a small deck tilt and a visible roof-edge structure (types shared out evenly). Without
    the setting every tank keeps the legacy values: 0.05 m gap, centred level deck, no edge structure.

    Physical limits (a real roof rests on its legs on the floor and cannot rise above the rim):
    - the tilt is drawn below the largest tilt that keeps the whole deck top between the floor level
      (0.2 m, the deck at fill 0) and RIM_CLEARANCE below the rim, so near-empty and near-full tanks
      are (almost) level;
    - an edge structure's top stays RIM_CLEARANCE below the rim, so on a near-full tank it is lower.
    """
    v = s.get("tank_variation") or {}
    rng = random.Random(f"tank-{seed}")
    edges = v.get("edge_types") or {"none": 0.0}
    names = list(edges)
    edge_list = [names[i % len(names)] for i in range(len(plan))]
    rng.shuffle(edge_list)
    jitter = float(v.get("edge_height_jitter", 0.0))
    for p, edge in zip(plan, edge_list):
        if v:
            R, H = p["radius"], p["height"]
            z_R = 0.2 + p["fill_level"] * (H - 0.3)          # deck top at the centre (legacy mapping)
            gap = round(rng.uniform(*_range(v.get("seal_gap_range", [0.05, 0.05]))), 3)
            # The roof can only move inside its seal gap (keep 10 % of it free so it never touches the wall)
            off, ang = rng.uniform(0.0, 0.9 * gap), rng.uniform(0.0, 2.0 * math.pi)
            t_lo, t_hi = _range(v.get("roof_tilt_range", [0.0, 0.0]))
            room = max(0.0, min(z_R - 0.2, H - RIM_CLEARANCE - z_R))   # deck may move this far up/down
            t_hi = min(t_hi, math.degrees(math.atan(room / (R + 0.9 * gap))))
            # rounded DOWN to 0.001 deg so rounding never pushes the deck past its limit (rounding to nearest and
            # then flooring the rounded value let 0.02472 -> 0.025 > t_hi = 0.02475 through; 25k plan, seed 46)
            tilt = math.floor(rng.uniform(min(t_lo, t_hi), t_hi) * 1000) / 1000
            tilt_az = round(rng.uniform(0.0, 360.0), 1)
            edge_h = round(float(edges[edge]) * (1.0 + rng.uniform(-jitter, jitter)), 3) if edges[edge] else 0.0
            deck_high = z_R + (R + 0.9 * gap) * math.tan(math.radians(tilt))
            # rounded DOWN to mm, so rounding never lifts the edge above the limit
            edge_h = math.floor(max(0.0, min(edge_h, H - RIM_CLEARANCE - deck_high)) * 1000) / 1000
        else:
            gap, off, ang, tilt, tilt_az, edge, edge_h = 0.05, 0.0, 0.0, 0.0, 0.0, "none", 0.0
        p.update({"seal_gap_m": gap, "roof_offset_x_m": round(off * math.sin(ang), 4),
                  "roof_offset_y_m": round(off * math.cos(ang), 4), "roof_tilt_deg": tilt,
                  "roof_tilt_azimuth_deg": tilt_az, "edge_type": edge, "edge_height_m": edge_h})


# Layout types understood by blender_render_script.py (see LAYOUTS there)
LAYOUT_TYPES = ["dike_rows", "staggered_rows", "clusters", "refinery_mixed", "single_row", "isolated"]


def _add_realism(plan, config, s, seed):
    """
    Per-image variation that real imagery has and a fixed preset does not:
    - camera azimuth: a satellite can view a site from any direction, so a fixed azimuth per tilt
      would let a model learn "tilt 80 deg always leans the same way". Drawn uniformly per image.
    - ground colour: soil, gravel and concrete differ between sites and seasons, so the preset colour
      is varied in brightness and tint.
    Uses its own random stream, so dates, times, sizes and fills are the same as without it.

    A random azimuth can put the satellite on the side away from the sun, where the near wall hides the
    small sunlit part of a deep roof. Azimuths are redrawn until at least MIN_VISIBLE_LIT_ROOF of the roof
    (or 70% of the nadir value, if smaller) is visible and lit (azimuth is not a balanced factor, so the
    distribution is unchanged).
    """
    rng = random.Random(f"realism-{seed}")
    random_azimuth = s.get("randomize_camera_azimuth", True)
    brightness = float(s.get("ground_brightness_jitter", 0.15))
    tint = float(s.get("ground_tint_jitter", 0.04))
    presets = config["locations"]
    for p in plan:
        if random_azimuth and p["camera_elevation"] < 89.99:
            ullage = max(p["height"] - (p["fill_level"] * (p["height"] - 0.3) + 0.2), 0.0)
            _, nadir_shadowed = geometry.visible_roof_shadow(p["radius"], ullage, p["sun_elevation"],
                                                             p["sun_azimuth"], 90.0, 0.0, n=61)
            need = min(MIN_VISIBLE_LIT_ROOF, 0.7 * (1.0 - nadir_shadowed))
            best = None
            for _ in range(36):
                az = round(rng.uniform(0.0, 360.0), 1)
                seen, shadowed = geometry.visible_roof_shadow(p["radius"], ullage, p["sun_elevation"], p["sun_azimuth"],
                                                              p["camera_elevation"], az, n=61)
                lit = seen * (1.0 - shadowed)
                if best is None or lit > best[0]:
                    best = (lit, az)
                if lit >= need:
                    break
            p["camera_azimuth"] = best[1]
        base = str(presets[p["location"]].get("ground_color", "") or "0.75,0.72,0.65")
        rgb = [float(v) for v in base.split(",")]
        b = 1.0 + rng.uniform(-brightness, brightness)
        rgb = [min(1.0, max(0.0, v * b * (1.0 + rng.uniform(-tint, tint)))) for v in rgb]
        p["ground_color"] = ",".join(f"{v:.3f}" for v in rgb)

    # Tank-farm layout type: shared out in proportion to the weights (equal by default), then shuffled
    weights = s.get("layout_weights") or {name: 1.0 for name in LAYOUT_TYPES}
    total_w = sum(weights.values())
    shares = {k: len(plan) * w / total_w for k, w in weights.items()}
    counts = {k: int(v) for k, v in shares.items()}
    for k in sorted(shares, key=lambda k: shares[k] - counts[k], reverse=True)[:len(plan) - sum(counts.values())]:
        counts[k] += 1
    layouts = [k for k, n in counts.items() for _ in range(n)]
    rng.shuffle(layouts)
    for p, layout in zip(plan, layouts):
        p["layout"] = layout

    # NAIP pixel scale: naip_scale_share of the images are rendered directly at a ground resolution from
    # naip_mpp_range (render_mpp, evenly spread), so tanks have as many pixels as in real NAIP chips.
    # The others use zoom: evenly spread on a log scale over the range (as many 1-2x as 2-4x images), so
    # the set has close-ups of one tank as well as wide views of a whole farm. Off-centre framing is uniform.
    order = list(range(len(plan)))
    rng.shuffle(order)
    n_naip = int(round(len(plan) * float(s.get("naip_scale_share", 0.0))))
    naip, zoomed = order[:n_naip], order[n_naip:]
    m_lo, m_hi = _range(s.get("naip_mpp_range", [0.5, 0.7]))
    for k, i in enumerate(naip):
        t = (k + rng.random()) / len(naip)
        plan[i]["render_mpp"] = round(m_lo + t * (m_hi - m_lo), 4)
        plan[i]["zoom"] = None
    lo, hi = (float(v) for v in s.get("zoom_range", [1.0, 4.0]))
    for k, i in enumerate(zoomed):
        t = (k + rng.random()) / len(zoomed)
        plan[i]["zoom"] = round(math.exp(math.log(lo) + t * (math.log(hi) - math.log(lo))), 3)
        plan[i]["render_mpp"] = None
    for p in plan:
        p["frame_shift_x"] = round(rng.uniform(-1.0, 1.0), 3)
        p["frame_shift_y"] = round(rng.uniform(-1.0, 1.0), 3)


def _r2_linear(x, y):
    """R^2 of the least-squares line y ~ x (the squared Pearson correlation)."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx, syy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
    return sxy * sxy / (sxx * syy) if sxx > 0 and syy > 0 else (1.0 if syy == 0 else 0.0)


def _r2_groups(groups, y):
    """R^2 of predicting y by its group mean (share of the variance of y explained by the group)."""
    my = sum(y) / len(y)
    total = sum((b - my) ** 2 for b in y)
    if total == 0:
        return 1.0
    sums, counts = Counter(), Counter()
    for g, b in zip(groups, y):
        sums[g] += b
        counts[g] += 1
    return 1.0 - sum((b - sums[g] / counts[g]) ** 2 for g, b in zip(groups, y)) / total


def summary(plan):
    """Text tables of every factor and the pairwise minimum, for checking uniformity."""
    total = len(plan)
    lines = [f"Total images: {total}"]
    first = plan[0] if plan else {}
    factors = ["size", "height_bin" if "height_bin" in first else None,
               "fill_bin" if "fill_bin" in first else "fill_level", "country", "sun_band", "atmosphere",
               "camera_bin" if "camera_bin" in first else "camera_elevation"]
    factors = [f for f in factors if f]
    for f in factors:
        c = Counter(p[f] for p in plan)
        lines.append(f"  {f:17s} " + "  ".join(f"{k}: {v}" for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))))
    if plan and "zoom" in plan[0]:
        zb = Counter("1-1.5" if p["zoom"] < 1.5 else "1.5-2" if p["zoom"] < 2 else "2-3" if p["zoom"] < 3 else "3+"
                     for p in plan if p["zoom"] is not None)
        n_naip = sum(1 for p in plan if p.get("render_mpp"))
        lines.append("  zoom              " + "  ".join(f"{k}: {zb[k]}" for k in ("1-1.5", "1.5-2", "2-3", "3+") if k in zb)
                     + (f"  NAIP scale (0.5-0.7 m/px): {n_naip}" if n_naip else ""))
    if plan and "edge_type" in plan[0]:
        ec = Counter(p["edge_type"] for p in plan)
        lines.append("  edge_type         " + "  ".join(f"{k}: {v}" for k, v in sorted(ec.items())))
    if plan and "holdout" in plan[0]:
        hc = Counter(t for p in plan for t in p["holdout"])
        n_any = sum(1 for p in plan if p["holdout"])
        lines.append(f"  held out          any: {n_any}  " + "  ".join(f"{k}: {v}" for k, v in sorted(hc.items())))
    if plan and "layout" in plan[0]:
        lc = Counter(p["layout"] for p in plan)
        lines.append("  layout            " + "  ".join(f"{k}: {v}" for k, v in sorted(lc.items())))
    country_band = Counter((p["country"], p["sun_band"]) for p in plan)
    bands = sorted({p["sun_band"] for p in plan}, key=lambda b: float(b.split("-")[0]))
    lines.append("  country x sun band (deg):")
    lines.append("    " + f"{'':14s}" + "".join(f"{b:>8s}" for b in bands))
    for country in sorted({p["country"] for p in plan}):
        lines.append("    " + f"{country:14s}" + "".join(f"{country_band.get((country, b), 0):8d}" for b in bands))
    radii, heights = [p["radius"] for p in plan], [p["height"] for p in plan]
    fills, cams = [p["fill_level"] for p in plan], [p["camera_elevation"] for p in plan]
    lines.append(f"  radius {min(radii):.1f}-{max(radii):.1f} m ({len(set(radii))} distinct), "
                 f"height {min(heights):.1f}-{max(heights):.1f} m ({len(set(heights))} distinct), "
                 f"fill {len(set(fills))} distinct, camera elevation {len(set(cams))} distinct")
    # How well the height can be predicted from the radius (v1: 1.0). Should be about 0.
    lines.append(f"  R^2 of height from radius: linear {_r2_linear(radii, heights):.4f}, "
                 f"from size class {_r2_groups([p['size'] for p in plan], heights):.4f}")
    pair_mins = []
    for a, b in itertools.combinations(factors, 2):
        counts = Counter((p[a], p[b]) for p in plan)
        pair_mins.append((min(counts.values()), a, b))
    lines.append("  smallest non-empty pair combination: " +
                 ", ".join(f"{a} x {b} = {n}" for n, a, b in sorted(pair_mins)[:3]))
    return "\n".join(lines)
