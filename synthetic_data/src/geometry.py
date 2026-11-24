"""
Pure-Python scene geometry shared by render.py and generate_dataset.py (no bpy needed).

Conventions used everywhere in the pipeline:
- World axes: +X = East, +Y = North, +Z = Up. Distances in metres.
- Azimuths are compass bearings: 0 = North, 90 = East, measured clockwise.
- sun_azimuth is the direction from the tank TO the sun.
- camera_azimuth is the direction from the tank TO the satellite.
- Images are rendered north-up: the image's up axis is North projected onto the image plane,
  and the right axis is East (a normal map view, not mirrored).
- Image-space directions are measured clockwise from image-up, so on a nadir image they equal
  compass bearings. On an off-nadir image the tilt foreshortens the ground, so they differ slightly.
"""
import math
from datetime import datetime, timedelta, timezone


# --- Sun position (NOAA solar calculator algorithm, accurate to about 0.01 deg for 1900-2100) ---

def _julian_day(dt_utc):
    return dt_utc.timestamp() / 86400.0 + 2440587.5


def solar_position(latitude, longitude, dt_utc):
    """
    Apparent sun position seen from (latitude, longitude) at dt_utc (timezone-aware or naive UTC).
    Longitude is positive East. Returns a dict with elevation (with atmospheric refraction),
    azimuth (clockwise from North), declination and equation of time (minutes).
    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    t = (_julian_day(dt_utc) - 2451545.0) / 36525.0

    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    m_r = math.radians(m)
    c = (math.sin(m_r) * (1.914602 - t * (0.004817 + 0.000014 * t))
         + math.sin(2 * m_r) * (0.019993 - 0.000101 * t)
         + math.sin(3 * m_r) * 0.000289)
    omega = math.radians(125.04 - 1934.136 * t)
    app_long = math.radians(l0 + c - 0.00569 - 0.00478 * math.sin(omega))
    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    eps = math.radians(eps0 + 0.00256 * math.cos(omega))
    decl = math.asin(math.sin(eps) * math.sin(app_long))

    y = math.tan(eps / 2.0) ** 2
    l0_r = math.radians(l0)
    eot = 4.0 * math.degrees(
        y * math.sin(2 * l0_r) - 2 * e * math.sin(m_r) + 4 * e * y * math.sin(m_r) * math.cos(2 * l0_r)
        - 0.5 * y * y * math.sin(4 * l0_r) - 1.25 * e * e * math.sin(2 * m_r))

    utc_minutes = dt_utc.hour * 60.0 + dt_utc.minute + dt_utc.second / 60.0
    true_solar_minutes = (utc_minutes + eot + 4.0 * longitude) % 1440.0
    hour_angle = math.radians(true_solar_minutes / 4.0 - 180.0)

    lat = math.radians(latitude)
    cos_zen = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    zenith = math.acos(max(-1.0, min(1.0, cos_zen)))
    elevation = 90.0 - math.degrees(zenith)
    azimuth = (math.degrees(math.atan2(
        math.sin(hour_angle),
        math.cos(hour_angle) * math.sin(lat) - math.tan(decl) * math.cos(lat))) + 180.0) % 360.0

    return {
        "elevation": elevation + _refraction(elevation),
        "geometric_elevation": elevation,
        "azimuth": azimuth,
        "declination": math.degrees(decl),
        "equation_of_time_minutes": eot,
    }


def _refraction(elev):
    """NOAA atmospheric refraction correction in degrees (raises the apparent sun)."""
    if elev > 85.0:
        return 0.0
    te = math.tan(math.radians(elev))
    if elev > 5.0:
        arcsec = 58.1 / te - 0.07 / te ** 3 + 0.000086 / te ** 5
    elif elev > -0.575:
        arcsec = 1735.0 + elev * (-518.2 + elev * (103.4 + elev * (-12.79 + elev * 0.711)))
    else:
        arcsec = -20.774 / te
    return arcsec / 3600.0


def utc_from_local_mean_solar_time(date_str, local_mean_solar_hours, longitude):
    """
    Sun-synchronous satellites (Sentinel-2, Landsat, WorldView, Pleiades, ...) pass over at a fixed
    LOCAL MEAN SOLAR TIME (for example 10:30), whatever the country's time zone. Convert it to UTC:
    UTC = local mean solar time - longitude / 15 h.
    """
    day = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return day + timedelta(hours=local_mean_solar_hours - longitude / 15.0)


def sun_for_location(location, date_str, local_mean_solar_hours):
    """Sun position for a location preset (dict with latitude, longitude, ...) at a local solar time."""
    lat, lon = float(location["latitude"]), float(location["longitude"])
    dt_utc = utc_from_local_mean_solar_time(date_str, local_mean_solar_hours, lon)
    sun = solar_position(lat, lon, dt_utc)
    offset = float(location.get("utc_offset_hours", round(lon / 15.0)))
    sun["utc_time"] = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    sun["local_clock_time"] = (dt_utc + timedelta(hours=offset)).strftime("%Y-%m-%dT%H:%M:%S")
    sun["utc_offset_hours"] = offset
    return sun


# --- Camera and image-space geometry ---

def _norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def camera_basis(camera_elevation, camera_azimuth):
    """
    Right/up/forward unit vectors of the north-up orthographic satellite camera.
    Must match setup_camera() in blender_render_script.py.
    """
    el, az = math.radians(camera_elevation), math.radians(camera_azimuth)
    to_cam = (math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el))
    forward = tuple(-c for c in to_cam)
    north = (0.0, 1.0, 0.0)
    up = tuple(n - _dot(north, forward) * f for n, f in zip(north, forward))
    if math.sqrt(_dot(up, up)) < 1e-6:  # only if looking exactly along North (horizontal camera)
        up = (0.0, 0.0, 1.0)
    up = _norm(up)
    right = _cross(up, tuple(-f for f in forward))  # Blender camera: X = Y x Z, Z = -forward
    return right, up, forward


def ground_vector(length, azimuth_deg):
    az = math.radians(azimuth_deg)
    return (length * math.sin(az), length * math.cos(az), 0.0)


def to_image(vec, basis):
    """Project a world vector onto the image plane. Returns (length_m, direction_deg_cw_from_up)."""
    right, up, _ = basis
    x, y = _dot(vec, right), _dot(vec, up)
    length = math.hypot(x, y)
    direction = math.degrees(math.atan2(x, y)) % 360.0 if length > 1e-9 else 0.0
    return length, direction


def crescent_ratio(width, diameter):
    """Fraction of a circle of this diameter not covered by the same circle shifted by `width`."""
    u = max(0.0, min(1.0, width / diameter)) if diameter > 0 else 0.0
    return 1.0 - (2.0 / math.pi) * (math.acos(u) - u * math.sqrt(1.0 - u * u))


def visible_roof_shadow(radius, ullage, sun_elevation, sun_azimuth, camera_elevation, camera_azimuth, n=161):
    """
    What the camera actually sees on the floating roof, by sampling an n x n grid over the roof disc.

    A roof point p is lit if the ray towards the sun clears the rim: p moved by ullage/tan(sun_elev)
    towards the sun must lie inside the rim circle. It is visible if the same test passes towards
    the camera. Off-nadir, the wall nearest the satellite hides a crescent of roof, and part of the
    internal shadow can be hidden behind it.

    Returns (visible_roof_ratio, visible_shadow_ratio): the fraction of the roof the camera sees,
    and the fraction of that visible roof that is in shadow.
    """
    def shift(elev, az):
        t = math.tan(math.radians(elev))
        d = ullage / t if t > 1e-9 else 1e9
        a = math.radians(az)
        return d * math.sin(a), d * math.cos(a)

    sx, sy = shift(sun_elevation, sun_azimuth)
    cx, cy = shift(camera_elevation, camera_azimuth)
    r2 = radius * radius
    roof = visible = visible_shadow = 0
    step = 2.0 * radius / (n - 1)
    for i in range(n):
        x = -radius + i * step
        for j in range(n):
            y = -radius + j * step
            if x * x + y * y > r2:
                continue
            roof += 1
            if (x + cx) ** 2 + (y + cy) ** 2 <= r2:
                visible += 1
                if (x + sx) ** 2 + (y + sy) ** 2 > r2:
                    visible_shadow += 1
    if roof == 0:
        return 0.0, 0.0
    return visible / roof, (visible_shadow / visible if visible else 0.0)
