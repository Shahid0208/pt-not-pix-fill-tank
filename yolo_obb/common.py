"""Shared class list and YOLO-OBB label helpers for the real and synthetic prep scripts."""
from pathlib import Path

import numpy as np

# The 7 classes of the real NAIP tank dataset (PASCAL VOC <name>). Order = YOLO class id.
CLASS_NAMES = [
    "closed_roof_tank",
    "narrow_closed_roof_tank",
    "external_floating_roof_tank",
    "sedimentation_tank",
    "spherical_tank",
    "water_tower",
    "undefined_object",
]
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Synthetic tank kinds (labels/*.json scene.neighbour_tanks[*].kind) -> real class.
# The labelled tank (mask instance 1) is always a floating-roof tank.
SYNTHETIC_KIND_TO_CLASS = {
    "floating": "external_floating_roof_tank",
    "fixed_cone": "closed_roof_tank",
}

SPLITS = ("train", "val", "test")


def box_to_corners(xmin, ymin, xmax, ymax):
    """Axis-aligned box -> 4 corners (clockwise from top-left), shape (4, 2)."""
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float64)


def format_obb_line(class_id, corners, width, height):
    """One YOLO-OBB label line: `cls x1 y1 x2 y2 x3 y3 x4 y4`, coords normalised to [0, 1]."""
    pts = np.asarray(corners, dtype=np.float64).copy()
    pts[:, 0] = np.clip(pts[:, 0] / width, 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / height, 0.0, 1.0)
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in pts.reshape(-1))


def write_label(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_split_list(path: Path, image_paths, root: Path):
    """Image list for data.yaml. Entries are `./rel/path` so Ultralytics resolves them
    against the list file's folder, which keeps the dataset relocatable (e.g. to a local disk)."""
    rels = sorted(f"./{p.relative_to(root).as_posix()}" for p in image_paths)
    path.write_text("\n".join(rels) + ("\n" if rels else ""))
