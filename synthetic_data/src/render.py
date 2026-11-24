"""Render one tank scene with Blender (blender_render_script.py) and write its image, mask, label and metadata.

render_tank() is called by generate_dataset.py for every entry of the balanced plan; load_config() reads config.yaml.
"""
import os
import subprocess
import json
import math
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry

# 1. Configuration Loading (With robust PyYAML fallback)
def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.yaml")
    
    # Base defaults
    config = {
        "directories": {"output_base_dir": "images"},
        "tank_sizes": {
            "small": {"radius": 10.0, "height": 12.0, "folder": "small_20x12"},
            "medium": {"radius": 20.0, "height": 14.0, "folder": "medium_40x14"},
            "large": {"radius": 30.0, "height": 16.0, "folder": "large_60x16"},
            "extraLarge": {"radius": 40.0, "height": 22.0, "folder": "extraLarge_80x22"}
        },
        "render_settings": {"engine": "CYCLES"}
    }
    
    # Try PyYAML
    try:
        import yaml
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                # Merge loaded config with defaults
                for section in yaml_config:
                    if section in config and isinstance(yaml_config[section], dict):
                        config[section].update(yaml_config[section])
                    else:
                        config[section] = yaml_config[section]
        return config
    except Exception:
        pass

    # Simple line-by-line fallback parser for config.yaml if PyYAML is not installed
    if not os.path.exists(config_path):
        return config

    try:
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        current_section = None
        current_subsection = None
        
        for line in lines:
            line = line.split('#')[0].strip() # strip comments
            if not line:
                continue
            if line.endswith(':'):
                section_name = line[:-1].strip()
                if line.startswith('  '):
                    current_subsection = section_name
                else:
                    current_section = section_name
                    current_subsection = None
                continue
            
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip()
                val_str = parts[1].strip()
                
                # Clean value quotes and convert types
                val = val_str
                if val_str.startswith('"') and val_str.endswith('"'):
                    val = val_str[1:-1]
                elif val_str.startswith("'") and val_str.endswith("'"):
                    val = val_str[1:-1]
                else:
                    try:
                        if '.' in val_str:
                            val = float(val_str)
                        else:
                            val = int(val_str)
                    except ValueError:
                        pass
                
                # Store value in structured dict
                if current_section == "directories":
                    config["directories"][key] = val
                elif current_section == "tank_sizes":
                    if current_subsection:
                        if current_subsection not in config["tank_sizes"]:
                            config["tank_sizes"][current_subsection] = {}
                        config["tank_sizes"][current_subsection][key] = val
                elif current_section == "render_settings":
                    config["render_settings"][key] = val
                elif current_section == "locations" and current_subsection:
                    config.setdefault("locations", {}).setdefault(current_subsection, {})[key] = val
    except Exception:
        pass
    return config

# Load initial config
config = load_config()

def get_size_folder(radius, height):
    """
    Looks up the appropriate size folder from config.yaml: the class whose radius_range holds the radius,
    or (legacy fixed classes) the class with the same radius and height.
    Falls back to dynamic 'custom_DxH' if no class matches.
    """
    diameter = radius * 2.0
    
    # Check configured tank sizes
    for size_key, size_info in config["tank_sizes"].items():
        rr = size_info.get("radius_range")
        if rr is not None:
            if isinstance(rr, str):  # the fallback YAML parser leaves lists as text
                rr = rr.strip("[] ").split(",")
            lo, hi = (float(v) for v in rr)
            if lo <= radius <= hi:
                return size_info.get("folder", f"size_{size_key}")
            continue
        ref_radius = size_info.get("radius")
        ref_height = size_info.get("height")
        if ref_radius is not None and ref_height is not None:
            ref_diameter = ref_radius * 2.0
            if abs(diameter - ref_diameter) < 1.0 and abs(height - ref_height) < 1.0:
                return size_info.get("folder", f"size_{size_key}")
                
    return f"custom_{int(diameter)}x{int(height)}"

def crescent_shadow_area_ratio(shadow_width: float, diameter: float) -> float:
    """
    Returns the fraction of the roof area covered by the internal (crescent) shadow.

    The rim casts a circle of the same diameter onto the roof, shifted by shadow_width
    along the shadow direction. The lit part of the roof is the lens where the roof and
    the shifted circle overlap; the shadow is the rest of the roof.
    """
    if diameter <= 0:
        return 0.0
    u = min(max(shadow_width / diameter, 0.0), 1.0)  # offset / diameter, in [0,1]
    if u <= 0:
        return 0.0
    if u >= 1:
        return 1.0
    lens_ratio = (2.0 / math.pi) * (math.acos(u) - u * math.sqrt(1.0 - u * u))
    return 1.0 - lens_ratio

def render_tank(
    filename: str,
    base_dir: str = "",
    radius: float = 20.0,
    height: float = 15.0,
    fill_level: float = 0.5,
    sun_elevation: float = 45.0,
    sun_azimuth: float = 135.0,
    camera_elevation: float = 90.0,
    camera_azimuth: float = 0.0,
    engine: str = "",
    atmosphere: str = "CLEAR",
    location: str = "",
    latitude: float | None = None,
    longitude: float | None = None,
    date: str = "",
    local_solar_time: float = 10.5,
    ground_color: str = "",
    zoom: float = 1.0,
    frame_shift_x: float = 0.0,
    frame_shift_y: float = 0.0,
    layout: str = "",
    folder: str = "",
    neighbour_height_range: list | None = None,
    render_mpp: float = 0.0,
    seal_gap: float = 0.05,
    roof_offset_x: float = 0.0,
    roof_offset_y: float = 0.0,
    roof_tilt: float = 0.0,
    roof_tilt_azimuth: float = 0.0,
    edge_type: str = "none",
    edge_height: float = 0.0,
    holdout: list | None = None
) -> str:
    """
    Renders a 3D model of an oil storage tank and its shadows using Blender in headless mode.
    
    Parameters:
    - filename: Name of the output image file (e.g. 'tank_fill_0.5.png').
    - base_dir: Base directory path. If empty, the workspace directory is used.
    - radius: Radius of the cylindrical tank in meters.
    - height: Total height of the tank shell in meters.
    - fill_level: Height of the floating roof (0.0 = empty tank, 1.0 = full tank).
    - sun_elevation: Solar elevation angle in degrees.
    - sun_azimuth: Solar azimuth angle in degrees.
    - camera_elevation: Satellite viewing elevation angle in degrees (90 = nadir).
    - camera_azimuth: Direction from the tank to the satellite in degrees (0 = North, 90 = East).
    - engine: Rendering engine ('EEVEE' or 'CYCLES'). Defaults to configuration.yaml choice.
    - atmosphere: 'CLEAR', 'HAZE' or 'THIN_CLOUD' (shadows stay visible in all three; 'OVERCAST' is an old alias of 'THIN_CLOUD').
    - location: Name of a location preset in config.yaml (e.g. 'saudi_ras_tanura', 'china_zhoushan',
      'russia_primorsk', 'usa_cushing'). With `date`, the sun position is computed for that place and
      time, and sun_elevation / sun_azimuth are ignored. The preset also sets the ground colour.
    - latitude, longitude: A custom place instead of a preset (degrees, North and East positive).
    - date: 'YYYY-MM-DD'. Needed whenever a location or latitude/longitude is given.
    - local_solar_time: Local mean solar time in hours (10.5 = 10:30, the usual overpass time of
      sun-synchronous imaging satellites).
    - ground_color: Optional 'r,g,b' in 0-1, overriding the preset's ground colour.
    - zoom: 1 = tight frame around the tank and its shadow; larger values show more of the tank farm
      (more tanks, fewer pixels per tank), limited by render_settings.max_meters_per_pixel.
    - frame_shift_x, frame_shift_y: -1..1, move the labelled tank off centre (it always stays in frame).
    - layout: tank-farm layout type (dike_rows, staggered_rows, clusters, refinery_mixed, single_row,
      isolated); empty = random.
    - folder: output sub-folder (e.g. the plan's size class); empty = looked up from the radius.
    - neighbour_height_range: [lo, hi] m; neighbouring tanks draw their heights from it instead of
      scaling the labelled tank's height (which would let a model read that height off the neighbours).
    - render_mpp: > 0 renders at this ground resolution (m/px) instead of zooming (e.g. NAIP, ~0.6).
    - seal_gap, roof_offset_x/y, roof_tilt, roof_tilt_azimuth, edge_type, edge_height: geometry of the
      labelled tank's floating roof (defaults = the legacy tank: 0.05 m gap, centred level deck, no edge).
      The roof offset must stay inside the seal gap. They go into the v3 truth layers as well.
    - holdout: held-out design tags for this image (e.g. ["shape"]), stored in metadata "holdout".

    Images are always north-up (image-up = North, image-right = East).
    """
    # Reload config dynamically in case the user edited config.yaml during runtime
    global config
    config = load_config()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "blender_render_script.py")
    
    # Path to the downloaded portable Blender binary
    portable_blender = os.path.join(current_dir, "..", "blender_app", "blender")
    blender_bin = portable_blender if os.path.exists(portable_blender) else "blender"
    
    if not os.path.exists(script_path):
        return f"Error: Blender script not found at {script_path}"

    # Sun from a place and time (overrides sun_elevation / sun_azimuth)
    location_info = None
    vegetation = None  # 0 = bare (desert) .. 1 = lush; from the location preset, else random per image
    if location or latitude is not None or longitude is not None:
        if location:
            preset = config.get("locations", {}).get(location)
            if preset is None:
                return f"Error: unknown location '{location}'. Known: {sorted(config.get('locations', {}))}"
            preset = dict(preset)
        elif latitude is None or longitude is None:
            return "Error: give both latitude and longitude."
        else:
            preset = {"name": "custom", "country": "", "latitude": latitude, "longitude": longitude}
        if not date:
            return "Error: a date ('YYYY-MM-DD') is required to compute the sun position for a location."
        try:
            sun = geometry.sun_for_location(preset, date, float(local_solar_time))
        except ValueError as ex:
            return f"Error: bad date '{date}': {ex}"
        min_elev = float(config.get("sun_limits", {}).get("min_sun_elevation", 10.0))
        if sun["elevation"] < min_elev:
            return (f"Error: sun elevation at {location or 'this place'} on {date} at local solar time "
                    f"{local_solar_time} is {sun['elevation']:.1f} deg, below the {min_elev} deg minimum "
                    f"(shadows would be too long to frame, or it is night).")
        sun_elevation = round(sun["elevation"], 3)
        sun_azimuth = round(sun["azimuth"], 3)
        if not ground_color:
            ground_color = str(preset.get("ground_color", ""))
        if preset.get("vegetation") not in (None, ""):
            vegetation = float(preset["vegetation"])
        location_info = {
            "preset": location or "custom",
            "name": preset.get("name", location),
            "country": preset.get("country", ""),
            "latitude": float(preset["latitude"]),
            "longitude": float(preset["longitude"]),
            "date": date,
            "local_mean_solar_time_hours": float(local_solar_time),
            "utc_time": sun["utc_time"],
            "local_clock_time": sun["local_clock_time"],
            "utc_offset_hours": sun["utc_offset_hours"],
            "solar_declination_degrees": round(sun["declination"], 4),
            "equation_of_time_minutes": round(sun["equation_of_time_minutes"], 3),
            "geometric_sun_elevation_degrees": round(sun["geometric_elevation"], 3),
        }
    elif not 0.0 < sun_elevation <= 90.0:
        return f"Error: sun_elevation must be in (0, 90], got {sun_elevation}."
    if zoom < 1.0:
        return f"Error: zoom must be >= 1, got {zoom}."
    if not 0.0 < camera_elevation <= 90.0:
        return f"Error: camera_elevation must be in (0, 90], got {camera_elevation}."
    if not ground_color:
        ground_color = "0.75,0.72,0.65"
    atmosphere = (atmosphere or "CLEAR").upper()
    if atmosphere == "OVERCAST":  # old name; fully overcast scenes have no shadows, so it now means thin cloud
        atmosphere = "THIN_CLOUD"
    if atmosphere not in ("CLEAR", "HAZE", "THIN_CLOUD"):
        return f"Error: atmosphere must be CLEAR, HAZE or THIN_CLOUD, got '{atmosphere}'."

    # Resolve output directories
    workspace_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if not base_dir:
        base_dir = workspace_dir
        
    size_folder = folder or get_size_folder(radius, height)
    output_dir_name = config["directories"].get("output_base_dir", "images")
    target_dir = os.path.join(base_dir, output_dir_name, size_folder)
    
    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, filename)
    # Segmentation masks live in a parallel masks/ tree so they are never mistaken for images
    write_masks = str(config["render_settings"].get("write_masks", True)).lower() not in ("false", "0", "no")
    mask_rel = f"masks/{size_folder}/{filename}" if write_masks else None
    mask_path = os.path.join(base_dir, mask_rel) if mask_rel else ""

    # Use configuration defaults for engine if empty
    if not engine:
        engine = config["render_settings"].get("engine", "EEVEE")

    # Build argument list
    cmd = [
        blender_bin,
        "--background",
        "--python-exit-code", "1",
        "--python", script_path,
        "--",
        "--radius", str(radius),
        "--height", str(height),
        "--fill-level", str(fill_level),
        "--sun-elevation", str(sun_elevation),
        "--sun-azimuth", str(sun_azimuth),
        "--camera-elevation", str(camera_elevation),
        "--camera-azimuth", str(camera_azimuth),
        "--engine", engine,
        "--atmosphere", atmosphere,
        "--ground-color", ground_color,
        # FULL: tank-farm scene (bunds, roads, pipes, neighbouring tanks, roof details); CLEAN: lone tank
        "--realism", str(config["render_settings"].get("scene_realism", "FULL")).upper(),
        "--samples", str(int(config["render_settings"].get("samples", 64))),
        "--zoom", str(zoom),
        "--shift-x", str(frame_shift_x),
        "--shift-y", str(frame_shift_y),
        "--max-mpp", str(float(config["render_settings"].get("max_meters_per_pixel", 0.8))),
        "--max-tanks", str(int(config["render_settings"].get("max_neighbour_tanks", 150))),
        "--mask-output", mask_path,
        # exact camera for the v3 truth layers (removed after reading)
        "--export-json", output_path + ".geo.json",
        "--output", output_path
    ]
    if os.environ.get("OILTANK_CYCLES_DEVICE"):        # set by generate_dataset.py --device
        cmd += ["--device", os.environ["OILTANK_CYCLES_DEVICE"]]
    if os.environ.get("OILTANK_THREADS"):              # set by generate_dataset.py --threads
        cmd[1:1] = ["--threads", os.environ["OILTANK_THREADS"]]
    if vegetation is not None:
        cmd += ["--vegetation", str(vegetation)]
    if layout:
        cmd += ["--layout", layout]
    if neighbour_height_range:
        cmd += ["--neighbour-height-range", ",".join(str(float(v)) for v in neighbour_height_range)]
    if render_mpp and render_mpp > 0:
        cmd += ["--mpp", str(float(render_mpp))]
    offset = math.hypot(roof_offset_x, roof_offset_y)
    if offset > 0 and offset >= seal_gap:
        return f"Error: roof offset {offset:.3f} m must be smaller than the seal gap {seal_gap} m."
    cmd += ["--seal-gap", str(float(seal_gap)), "--roof-offset-x", str(float(roof_offset_x)),
            "--roof-offset-y", str(float(roof_offset_y)), "--roof-tilt", str(float(roof_tilt)),
            "--roof-tilt-azimuth", str(float(roof_tilt_azimuth)), "--edge-type", edge_type,
            "--edge-height", str(float(edge_height))]
    timeout = int(config["render_settings"].get("timeout_seconds", 300))

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
        if mask_path and not os.path.exists(mask_path):
            return f"Error: Blender exited without writing the mask '{mask_path}'.\nLog tail:\n{(completed.stdout + completed.stderr)[-2000:]}"
        if not os.path.exists(output_path):
            return f"Error: Blender exited without writing '{output_path}'.\nLog tail:\n{(completed.stdout + completed.stderr)[-2000:]}"

        # Image scale chosen by the Blender script's camera framing
        ortho_scale = None
        pixels = {}
        scene_info = None
        # The scene block comes from the geometry file Blender writes. The SCENE_JSON log line is only a
        # fallback: with many neighbour tanks it is ~16 KB long, and Blender's own render messages can be
        # interleaved into it, which once stopped a long run with "JSONDecodeError: Extra data".
        try:
            with open(output_path + ".geo.json") as f:
                scene_info = json.load(f).get("scene")
        except (OSError, ValueError):
            scene_info = None
        for line in completed.stdout.splitlines():
            if line.startswith("SCENE_JSON="):
                if scene_info is None:
                    try:  # take the first complete JSON object and ignore anything appended to the line
                        scene_info, _ = json.JSONDecoder().raw_decode(line.split("=", 1)[1])
                    except ValueError:
                        scene_info = {"error": "scene block could not be read from the Blender log"}
            elif line.startswith("CAMERA_ORTHO_SCALE="):
                ortho_scale = float(line.split("=", 1)[1])
            elif line.startswith("PIXEL_"):
                key, val = line[len("PIXEL_"):].split("=", 1)
                pixels[key.lower()] = [round(float(c), 2) for c in val.split(",")]

        # v3 truth layers (world / projected / observability) from the exact Blender camera
        v3_layers = None
        full_record = None
        geo_path = output_path + ".geo.json"
        try:
            with open(geo_path) as f:
                cam_export = json.load(f)["camera"]
            from v3 import schema as v3_schema
            full_record = {}
            v3_layers = v3_schema.legacy_layers(radius, height, fill_level, sun_elevation, sun_azimuth,
                                                camera_elevation, camera_azimuth, atmosphere, cam_export,
                                                full=full_record, seal_gap=seal_gap,
                                                roof_offset=(roof_offset_x, roof_offset_y),
                                                roof_tilt=(roof_tilt, roof_tilt_azimuth),
                                                edge_type=edge_type, edge_height=edge_height)
        except Exception as ex:  # numpy missing or an old Blender script: keep the legacy labels only
            v3_layers = {"error": f"v3 truth layers not computed: {ex}"}
        finally:
            if os.path.exists(geo_path):
                os.remove(geo_path)

        # Calculate derived metadata fields
        diameter = radius * 2.0
        liquid_height = fill_level * (height - 0.3)
        ullage = height - (fill_level * (height - 0.3) + 0.2)
        volume_m3 = math.pi * radius**2 * liquid_height
        volume_barrels = volume_m3 * 6.2898  # 1 m³ ≈ 6.2898 bbl
        shadow_dir = (sun_azimuth + 180.0) % 360.0
        off_nadir = 90.0 - camera_elevation
        
        sun_elev_rad = math.radians(sun_elevation)
        tan_elev = math.tan(sun_elev_rad)
        
        external_shadow_len = height / tan_elev if tan_elev > 0 else 0.0
        internal_shadow_w = ullage / tan_elev if tan_elev > 0 else 0.0
        # The shadow cannot extend past the far wall: cap at the tank diameter
        internal_shadow_w = min(internal_shadow_w, diameter)
        shadow_area_ratio = crescent_shadow_area_ratio(internal_shadow_w, diameter)

        # What the north-up camera sees. Off-nadir, ground directions are foreshortened, so image
        # directions differ slightly from compass bearings; lengths are in image-plane metres.
        basis = geometry.camera_basis(camera_elevation, camera_azimuth)
        ext_img_len, shadow_img_dir = geometry.to_image(geometry.ground_vector(external_shadow_len, shadow_dir), basis)
        int_img_len, _ = geometry.to_image(geometry.ground_vector(internal_shadow_w, shadow_dir), basis)
        _, sun_img_dir = geometry.to_image(geometry.ground_vector(1.0, sun_azimuth), basis)
        rim_img_len, rim_img_dir = geometry.to_image((0.0, 0.0, height), basis)
        visible_roof, visible_shadow = geometry.visible_roof_shadow(
            radius, max(ullage, 0.0), sun_elevation, sun_azimuth, camera_elevation, camera_azimuth)
        m_per_px = ortho_scale / 800.0 if ortho_scale else None
        px = (lambda m: round(m / m_per_px, 2)) if m_per_px else (lambda m: None)

        metadata_entry = {
            "image_metadata": {
                "filename": filename,
                "relative_path": f"{output_dir_name}/{size_folder}/{filename}",
                # Same size as the image. Red = tank instance id (1 = this labelled tank, others in
                # scene.neighbour_tanks[*].instance_id, 0 = background); green = part (1 shell outside,
                # 2 shell inside, 3 floating roof, 4 fixed roof, 5 ladder/platform); blue unused.
                "mask_relative_path": mask_rel,
                "generation_timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "tank_geometry": {
                "size_class": size_folder.split('_')[0] if '_' in size_folder else "Custom",
                "diameter_meters": float(diameter),
                "radius_meters": float(radius),
                "total_height_meters": float(height),
                "fill_level_percentage": float(fill_level * 100.0),
                "liquid_height_meters": float(round(liquid_height, 3)),
                "ullage_meters": float(round(ullage, 3)),
                "volume_m3": float(round(volume_m3, 1)),
                "volume_barrels": float(round(volume_barrels, 0)),
                # Floating-roof geometry of this tank (v1: 0.05 m gap, centred level deck, no edge)
                "seal_gap_meters": float(seal_gap),
                "roof_offset_meters": [float(roof_offset_x), float(roof_offset_y)],
                "roof_tilt_degrees": float(roof_tilt),
                "roof_tilt_azimuth_degrees": float(roof_tilt_azimuth),
                "edge_type": edge_type,
                "edge_height_meters": float(edge_height)
            },
            "solar_conditions": {
                "sun_elevation_degrees": float(sun_elevation),
                "sun_azimuth_degrees": float(sun_azimuth),
                "shadow_direction_degrees": float(shadow_dir),
                "sun_angle_radians": float(round(sun_elev_rad, 4)),
                # "location": computed from a place, date and time (see the location block); "explicit": given directly
                "sun_source": "location" if location_info else "explicit"
            },
            "satellite_sensor": {
                "camera_elevation_degrees": float(camera_elevation),
                "camera_azimuth_degrees": float(camera_azimuth),
                "off_nadir_angle_degrees": float(off_nadir),
                # zoom is ignored (null) when the image was rendered at a fixed ground resolution
                "zoom": None if render_mpp and render_mpp > 0 else float(zoom),
                "requested_meters_per_pixel": float(render_mpp) if render_mpp and render_mpp > 0 else None,
                "frame_shift": [float(frame_shift_x), float(frame_shift_y)],
                # Width of the image in metres (orthographic camera); divide by 800 for metres per pixel
                "ortho_scale_meters": float(round(ortho_scale, 4)) if ortho_scale is not None else None,
                "rendering_engine": engine,
                "atmospheric_condition": atmosphere
            },
            # null when the sun angles were given directly instead of a place and time
            "location": location_info,
            "image_geometry": {
                "orientation": "north_up",
                "image_up_azimuth_degrees": 0.0,
                "meters_per_pixel": round(m_per_px, 5) if m_per_px else None,
                # Directions are clockwise from image-up. Pixels are (x right, y down) from the top-left.
                "sun_direction_image_degrees": round(sun_img_dir, 3),
                "shadow_direction_image_degrees": round(shadow_img_dir, 3),
                "external_shadow_length_image_meters": round(ext_img_len, 3),
                "external_shadow_length_pixels": px(ext_img_len),
                "internal_shadow_width_image_meters": round(int_img_len, 3),
                "internal_shadow_width_pixels": px(int_img_len),
                # Off-nadir, the tank top appears shifted away from the satellite (relief displacement)
                "rim_displacement_direction_image_degrees": round(rim_img_dir, 3) if rim_img_len > 1e-6 else None,
                "rim_displacement_image_meters": round(rim_img_len, 3),
                "tank_base_center_pixel": pixels.get("base"),
                "tank_rim_center_pixel": pixels.get("rim"),
                "roof_center_pixel": pixels.get("roof"),
            },
            # Scene around the labelled tank. neighbour_tanks lists every other tank visible in the image
            # (pixel centres, radius and clipped box), so a detector can be trained on all of them.
            "scene": scene_info,
            "schema_version": "3.0",
            # Held-out designs this image belongs to ("shape", "sun", "site"); such images go only to the
            # matching test_ood_* split, never to train or val. Empty for in-distribution images.
            "holdout": list(holdout or []),
            # label notation: z_C rim, z_R deck, z_F floor (lowest deck, 0.2 m), z_G terrain
            "world_truth": v3_layers[0] if isinstance(v3_layers, tuple) else v3_layers,
            "projected_truth": v3_layers[1] if isinstance(v3_layers, tuple) else None,
            "observability": v3_layers[2] if isinstance(v3_layers, tuple) else None,
            "analytical_ground_truth": {
                "external_shadow_length_meters": float(round(external_shadow_len, 3)),
                "max_internal_shadow_width_meters": float(round(internal_shadow_w, 3)),
                "internal_shadow_area_ratio": float(round(shadow_area_ratio, 3)),
                # Off-nadir the wall nearest the satellite hides part of the roof (and of its shadow)
                "visible_roof_ratio": round(visible_roof, 3),
                "visible_internal_shadow_ratio": round(visible_shadow, 3)
            }
        }

        # Save metadata
        metadata_path = os.path.join(target_dir, "metadata.json")
        metadata_dict = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
            except Exception as ex:
                # Never start over from an empty dict here: that would silently drop every earlier entry
                return (f"Error: could not read existing '{metadata_path}' ({ex}). The image was rendered "
                        f"but its metadata was NOT saved. Repair or move the file, then render again.")

        metadata_dict[filename] = metadata_entry

        # Per-image label file with every raw quantity (labels/<size folder>/<image name>.json): the
        # metadata entry plus the full boundary curves, rim points, chords, caster-shadow pairs and anchors
        if full_record:
            label_dir = os.path.join(base_dir, "labels", size_folder)
            os.makedirs(label_dir, exist_ok=True)
            label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".json")
            for anc in full_record["anchor_annotations"]["anchors"]:
                anc["anchor_id"] = anc["anchor_id"].replace("image_", os.path.splitext(filename)[0] + "_", 1)
            full_record["anchor_annotations"]["image_id"] = os.path.splitext(filename)[0]
            record = dict(metadata_entry, world_truth=full_record["world_truth"],
                          projected_truth=full_record["projected_truth"], acquisition=full_record["acquisition"],
                          anchor_annotations=full_record["anchor_annotations"])
            with open(label_path + ".tmp", "w") as f:
                json.dump(record, f, indent=1)
            os.replace(label_path + ".tmp", label_path)
            metadata_entry["image_metadata"]["label_relative_path"] = f"labels/{size_folder}/{os.path.basename(label_path)}"

        # Write to a temporary file and swap it in, so an interrupted run cannot leave a half-written file
        tmp_path = metadata_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(metadata_dict, f, indent=4)
        os.replace(tmp_path, metadata_path)

        where = (f"Location: {location_info['name']} ({location_info['country']}) {date} "
                 f"LMST {local_solar_time} -> sun elev {sun_elevation:.2f}, az {sun_azimuth:.2f}\n"
                 if location_info else "")
        return (
            f"Success: Image augmented and saved to '{output_path}'\n"
            f"{where}"
            f"Metadata successfully saved in '{metadata_path}'\n"
            f"Folder category: {output_dir_name}/{size_folder}\n"
            f"Using Blender executable: {blender_bin}"
        )
    except subprocess.TimeoutExpired:
        return (f"Error: Rendering command timed out (max {timeout} seconds exceeded). Raise "
                f"render_settings.timeout_seconds or lower render_settings.samples in config.yaml.")
    except subprocess.CalledProcessError as e:
        return (
            f"Error: Blender execution failed with exit code {e.returncode}.\n"
            f"Command run: {' '.join(cmd)}\n"
            f"Stdout:\n{e.stdout}\n"
            f"Stderr:\n{e.stderr}"
        )
