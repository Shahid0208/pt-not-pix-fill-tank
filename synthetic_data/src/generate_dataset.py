"""
Renders a balanced synthetic dataset: --total images (default 1000) in which tank size, fill level,
country, sun-height band, atmosphere and camera angle are each evenly spread (see balanced_plan.py).
Each finished render is then passed through sensor_sim.py so it looks like real pan-sharpened
satellite imagery (GSD resampling, colour bleed, noise, sharpening, JPEG).

    python src/generate_dataset.py --dry-run          # show the plan's distribution only
    python src/generate_dataset.py --total 1000       # render (safe to stop and restart)
    python src/generate_dataset.py --keep-clean       # also keep the untouched renders
    python src/generate_dataset.py --no-sensor-sim    # clean renders only

Large batches on a cluster: one plan, split over many jobs, numbered after an existing set:
    python src/generate_dataset.py --total 25000 --seed 46 --start-index 22000 --save-plan plan.json   # SynTank-25k
    python src/generate_dataset.py --plan plan.json --seed 46 --shard 3/4 --base-dir shard03 --device CUDA --threads 10
Each shard renders every N-th image of the same plan into its own folder; merge the folders afterwards.
"""
import os
import sys
import json
import shutil
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render import render_tank, load_config
import balanced_plan
import sensor_sim

SENSOR_LOG = "sensor_sim_done.json"


def load_sensor_log(images_dir):
    path = os.path.join(images_dir, SENSOR_LOG)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as ex:
        sys.exit(f"[FAILED] {path} cannot be read ({ex}). Without it, images could be degraded twice.")


def save_sensor_log(images_dir, log):
    path = os.path.join(images_dir, SENSOR_LOG)
    with open(path + ".tmp", "w") as f:
        json.dump(log, f, indent=1)
    os.replace(path + ".tmp", path)


def degrade(p, path, images_dir, clean_dir, sensor_log, seed, config):
    """
    Apply the satellite look once per image, recording it so restarts never apply it twice.
    The render's metres-per-pixel comes from its metadata entry, so every image ends up at a real
    satellite resolution; the settings used are written back into that entry.
    """
    key = f"{p['folder']}/{p['filename']}"
    if key in sensor_log:
        return
    meta_path = os.path.join(images_dir, p["folder"], "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    entry = meta.get(p["filename"], {})
    mpp = (entry.get("image_geometry") or {}).get("meters_per_pixel")
    if clean_dir:
        dst = os.path.join(clean_dir, p["folder"], p["filename"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(path, dst)
    info = sensor_sim.apply(path, seed=seed, config=config, meters_per_pixel=mpp)
    info["ground_color"] = p.get("ground_color")
    # label field names alongside the sensor_sim ones
    info["gsd_simulated_m"] = info.get("effective_gsd_meters")
    info["noise_sigma"] = info.get("read_noise")
    info["applied_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    entry.setdefault("satellite_sensor", {})["post_processing"] = info
    meta[p["filename"]] = entry
    with open(meta_path + ".tmp", "w") as f:
        json.dump(meta, f, indent=4)
    os.replace(meta_path + ".tmp", meta_path)
    sensor_log[key] = info
    save_sensor_log(images_dir, sensor_log)


def main():
    parser = argparse.ArgumentParser(description="Render a balanced synthetic oil-tank dataset with Blender.")
    parser.add_argument("--total", type=int, default=1000, help="Number of images (default 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed; the same seed always gives the same plan")
    parser.add_argument("--base-dir", type=str, default="",
                        help="Folder that gets the images/ tree (default: synthetic_data/)")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan's distribution without rendering")
    parser.add_argument("--no-sensor-sim", action="store_true", help="Skip the satellite-sensor post-processing")
    parser.add_argument("--keep-clean", action="store_true",
                        help="Copy each untouched render to images_clean/ before post-processing")
    parser.add_argument("--plan", type=str, default="",
                        help="Load this saved balanced_plan.json instead of planning again (planning 20,000 "
                             "images takes minutes; save it once with --save-plan, then give it to every shard)")
    parser.add_argument("--save-plan", type=str, default="", help="Write the plan to this file and exit")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Number the images from here (e.g. 1500 to follow an existing 1,500-image set)")
    parser.add_argument("--shard", type=str, default="",
                        help="K/N: render only every N-th image of the plan, starting at K (0-based)")
    parser.add_argument("--device", type=str, default="", choices=["", "CPU", "CUDA", "OPTIX"],
                        help="Cycles device (default: CPU, as in config)")
    parser.add_argument("--threads", type=int, default=0, help="Blender render threads (0 = all cores)")
    args = parser.parse_args()
    if args.device:
        os.environ["OILTANK_CYCLES_DEVICE"] = args.device
    if args.threads:
        os.environ["OILTANK_THREADS"] = str(args.threads)

    config = load_config()
    engine = config["render_settings"].get("engine", "CYCLES")
    workspace_dir = os.path.abspath(args.base_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    images_dir = os.path.join(workspace_dir, config["directories"].get("output_base_dir", "images"))
    clean_dir = os.path.join(workspace_dir, "images_clean") if args.keep_clean else ""
    use_sim = not args.no_sensor_sim and sensor_sim.settings(config).get("enabled", True)

    print("Planning (scanning the sun path at each location)...")
    if args.plan:
        with open(args.plan) as f:
            plan = json.load(f)
        print(f"Loaded {len(plan)} planned images from {args.plan}")
    else:
        plan = balanced_plan.build_plan(config, total=args.total, seed=args.seed, start_index=args.start_index)
        print(balanced_plan.summary(plan))
    if args.save_plan:
        with open(args.save_plan, "w") as f:
            json.dump(plan, f, indent=1)
        print(f"Plan written to {args.save_plan}")
        return
    full_plan = plan
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        assert 0 <= k < n, "--shard must be K/N with 0 <= K < N"
        plan = plan[k::n]
        print(f"Shard {k}/{n}: {len(plan)} of {len(full_plan)} images")
    print(f"Engine: {engine}   Output: {images_dir}   Sensor simulation: {'on' if use_sim else 'off'}")
    if args.dry_run:
        return

    # Existing metadata per size folder: used to resume an interrupted run and to spot leftovers
    existing = {}
    for folder in sorted({p["folder"] for p in plan}):
        meta_path = os.path.join(images_dir, folder, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    existing[folder] = json.load(f)
            except Exception as ex:
                sys.exit(f"[FAILED] {meta_path} cannot be read ({ex}). Repair or move it before generating.")
    planned = {p["filename"] for p in plan}
    leftovers = sum(len(set(entries) - planned) for entries in existing.values())
    if leftovers:
        print(f"WARNING: {leftovers} images in {images_dir} are not part of this plan (other seed, total or "
              f"settings). run_split_on_host.py would mix them in and the set would no longer be balanced. "
              f"Move the old images/ folder away, or use --base-dir.")

    os.makedirs(images_dir, exist_ok=True)
    with open(os.path.join(images_dir, "balanced_plan.json"), "w") as f:
        json.dump(full_plan, f, indent=2)

    sensor_log = load_sensor_log(images_dir)
    if not use_sim and sensor_log:
        print(f"NOTE: {len(sensor_log)} images here were already post-processed; mixing clean and "
              f"processed images makes the set inconsistent.")

    done = failed = 0
    for n, p in enumerate(plan, 1):
        path = os.path.join(images_dir, p["folder"], p["filename"])
        # Skip only finished images: the PNG exists AND its metadata entry was written. A PNG without an
        # entry comes from an interrupted render (it may also lack the atmosphere veil), so render it again.
        finished = os.path.exists(path) and p["filename"] in existing.get(p["folder"], {})
        if not finished:
            print(f"[{n}/{len(plan)}] {p['filename']}")
            # A re-render replaces any earlier processed version, so forget that it was processed
            sensor_log.pop(f"{p['folder']}/{p['filename']}", None)
            try:
                result = render_tank(
                    filename=p["filename"],
                    base_dir=workspace_dir,
                    radius=p["radius"],
                    height=p["height"],
                    fill_level=p["fill_level"],
                    camera_elevation=p["camera_elevation"],
                    camera_azimuth=p["camera_azimuth"],
                    engine=engine,
                    atmosphere=p["atmosphere"],
                    location=p["location"],
                    date=p["date"],
                    local_solar_time=p["local_solar_time"],
                    ground_color=p.get("ground_color", ""),
                    zoom=p.get("zoom") or 1.0,
                    render_mpp=p.get("render_mpp") or 0.0,
                    layout=p.get("layout", ""),
                    frame_shift_x=p.get("frame_shift_x", 0.0),
                    frame_shift_y=p.get("frame_shift_y", 0.0),
                    folder=p["folder"],
                    neighbour_height_range=p.get("neighbour_height_range"),
                    seal_gap=p.get("seal_gap_m", 0.05),
                    roof_offset_x=p.get("roof_offset_x_m", 0.0),
                    roof_offset_y=p.get("roof_offset_y_m", 0.0),
                    roof_tilt=p.get("roof_tilt_deg", 0.0),
                    roof_tilt_azimuth=p.get("roof_tilt_azimuth_deg", 0.0),
                    edge_type=p.get("edge_type", "none"),
                    edge_height=p.get("edge_height_m", 0.0),
                    holdout=p.get("holdout"),
                )
            except Exception as ex:  # one bad image must not stop a long run
                result = f"Error: {type(ex).__name__}: {ex}"
            if not (result.startswith("Success") and os.path.exists(path)):
                failed += 1
                print(f"   [FAILED] {result}")
                continue

        if use_sim:
            try:
                degrade(p, path, images_dir, clean_dir, sensor_log, args.seed, config)
            except Exception as ex:
                failed += 1
                print(f"   [FAILED] sensor simulation on {p['filename']}: {ex}")
                continue
        done += 1

    print(f"\nDone: {done}/{len(plan)} images in {images_dir}" + (f", {failed} FAILED" if failed else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
