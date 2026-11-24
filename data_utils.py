"""File-name helper for the real NAIP chips."""
from collections import namedtuple

ImageNameParts = namedtuple("ImageNameParts", ["tile", "quad_id", "zone", "resolution", "date", "row", "col"])


def parse_image_name(filename: str) -> ImageNameParts:
    """m_<quad>_<quarter>_<zone>_<res>_<date>_<row>_<col> -> its parts; tile = the source NAIP tile."""
    parts = filename.split("_")
    if len(parts) < 8:
        raise ValueError(f"Expected at least 8 underscore-delimited parts, got {len(parts)}")
    return ImageNameParts("_".join(parts[:6]), "_".join(parts[1:3]), parts[3], parts[4], parts[5], parts[6], parts[7])
