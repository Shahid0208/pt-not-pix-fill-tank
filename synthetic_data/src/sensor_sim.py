"""
Makes a clean Blender render look like a delivered very-high-resolution satellite product
(defaults calibrated to NAIP aerial imagery, 0.6 m; a sensor_sim block can switch to WorldView /
Pleiades-class 0.3-0.5 m pan-sharpened RGB).

The image is processed in place and keeps its pixel size, so tank positions / boxes / masks in
metadata.json stay valid.

Chain (randomised per image, but deterministic for a given seed + filename):
  1. Optics blur (MTF) and resampling to a real satellite ground sample distance (GSD, 0.3-0.5 m),
     based on the render's own metres-per-pixel, so small tanks really are only ~50-70 px across
  2. Pan-sharpening: detail comes from the sharp panchromatic band, colour from a multispectral
     band ~4x coarser (WorldView-3: 0.31 m pan vs 1.24 m MS) -> soft colour bleed at edges
  3. Sensor noise (signal-dependent shot noise + read noise), strongest in shadows
  4. Vendor processing: dynamic-range stretch, gamma, muted saturation, white-balance drift,
     MTF-compensation sharpening (gives the faint halos seen around bright tank roofs)
  5. JPEG compression at satellite resolution (most imagery is served as JPEG tiles), then
     resampling back to the render size

Every range can be overridden with a "sensor_sim" block in the config, e.g.
    "sensor_sim": {"target_gsd_m": [0.3, 0.6], "jpeg_quality": [60, 85]}
"""
import hashlib
import io
import os

import numpy as np
from PIL import Image, ImageFilter

# Calibrated against real NAIP imagery of floating-roof tanks (the project's real dataset: USDA aerial,
# 0.6 m/px; 73 tanks at Cushing OK, Pasadena / Nederland / Corpus Christi TX). Measured there: lit/shadow
# luminance 2.23 (ground) and 1.99 (roof crescent), shadows bluish (B/R 1.20 ground, 1.42 roof), darkest
# shadow ~62, luminance 1st/99th percentile 66 / 229, noise ~1.4 DN, saturation 0.14, shadow edges
# 2.5-3.5 px (10-90 %), no sharpening halos. The old WorldView-style defaults (strong contrast stretch,
# heavy noise and sharpening) pushed the shadow ratios to 3.7 / 5.9.
DEFAULTS = {
    "enabled": True,
    # Ground sample distance of the simulated sensor, in metres per pixel (NAIP 0.6 m; WorldView-2/3,
    # GeoEye, Pleiades 0.3-0.5 m: override with a sensor_sim block). The render's own m/px comes from metadata.
    "target_gsd_m": [0.5, 0.7],
    # Used only when the render's m/px is unknown: sensor GSD / render GSD
    "resolution_factor": [1.5, 3.0],
    "ms_to_pan_ratio": 4,               # multispectral GSD / panchromatic GSD
    "psf_sigma_px": [0.6, 1.0],         # optics blur, in sensor pixels (gives the 2.5-3.5 px shadow edges)
    "shot_noise": [0.00005, 0.0002],    # variance per unit of signal
    "read_noise": [0.003, 0.006],       # std dev, signal range 0..1 (~1-1.5 DN)
    "stretch_percentiles": [0.5, 99.5], # dynamic range adjustment
    "stretch_strength": [0.0, 0.2],     # 0 = no stretch, 1 = full stretch
    "gamma": [0.95, 1.05],
    "saturation": [1.0, 1.2],
    "white_balance_drift": 0.03,        # +/- per-channel gain
    "sharpen_percent": [0, 25],         # light: real tank images show no halos
    "jpeg_quality": [80, 95],           # the real chips are JPEG
}

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def settings(config):
    s = dict(DEFAULTS)
    s.update((config or {}).get("sensor_sim", {}) or {})
    return s


def _rng(seed, name):
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _pick(rng, value):
    if isinstance(value, (list, tuple)):
        return float(rng.uniform(value[0], value[1]))
    return float(value)


def _per_channel(arr, fn):
    return np.stack([np.asarray(fn(Image.fromarray(np.ascontiguousarray(arr[..., c]), "F")))
                     for c in range(arr.shape[2])], axis=-1).astype(np.float32)


def _resize(arr, size, resample):
    return _per_channel(arr, lambda im: im.resize(size, resample))


def _blur(arr, sigma):
    """Separable Gaussian blur on a float HxWxC array (edge-padded)."""
    if sigma <= 0.05:
        return arr
    r = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    out = np.pad(arr, ((r, r), (r, r), (0, 0)), mode="edge")
    out = sum(k[i] * out[i:i + out.shape[0] - 2 * r] for i in range(2 * r + 1))
    out = sum(k[i] * out[:, i:i + out.shape[1] - 2 * r] for i in range(2 * r + 1))
    return out.astype(np.float32)


def _luma(arr):
    return arr @ _LUMA


def apply(path, seed=42, config=None, meters_per_pixel=None):
    """
    Degrade the PNG at `path` in place. `meters_per_pixel` is the render's scale (metadata
    image_geometry.meters_per_pixel); with it, every image ends up at a real satellite GSD whatever
    the tank size. Returns the parameters used.
    """
    s = settings(config)
    rng = _rng(seed, os.path.basename(path))

    with Image.open(path) as im:
        img = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]

    if meters_per_pixel:
        target = _pick(rng, s["target_gsd_m"])
        factor = max(1.0, target / float(meters_per_pixel))
    else:
        factor = max(1.0, _pick(rng, s["resolution_factor"]))
    effective_gsd = factor * float(meters_per_pixel) if meters_per_pixel else None
    psf = _pick(rng, s["psf_sigma_px"])
    pan_size = (max(8, round(w / factor)), max(8, round(h / factor)))
    ratio = float(s["ms_to_pan_ratio"])
    ms_size = (max(4, round(pan_size[0] / ratio)), max(4, round(pan_size[1] / ratio)))

    # 1. Optics + sampling at the panchromatic GSD
    pan_rgb = _resize(_blur(img, psf * factor), pan_size, Image.BOX)

    # 2. Pan-sharpening (Brovey-style): colour from the coarse MS band, detail from pan luminance
    ms = _resize(_resize(pan_rgb, ms_size, Image.BOX), pan_size, Image.BICUBIC)
    y_pan = _luma(pan_rgb)
    y_ms = _luma(ms)
    out = ms * (y_pan / np.maximum(y_ms, 1e-3))[..., None]

    # 3. Sensor noise: mostly shared across channels (pan noise), a little per channel
    shot = _pick(rng, s["shot_noise"])
    read = _pick(rng, s["read_noise"])
    sigma = np.sqrt(shot * np.clip(y_pan, 0, 1) + read ** 2)
    out += (rng.standard_normal(y_pan.shape, dtype=np.float32) * sigma)[..., None]
    out += rng.standard_normal(out.shape, dtype=np.float32) * (0.35 * read)

    # 4. Vendor processing
    lo, hi = np.percentile(_luma(out), s["stretch_percentiles"])
    strength = _pick(rng, s["stretch_strength"])
    if hi - lo > 1e-3:
        stretched = (out - lo) / (hi - lo) * 0.96 + 0.02
        out = out + strength * (stretched - out)
    out = np.clip(out, 0, 1) ** _pick(rng, s["gamma"])
    y = _luma(out)[..., None]
    out = y + _pick(rng, s["saturation"]) * (out - y)
    drift = float(s["white_balance_drift"])
    out *= 1 + rng.uniform(-drift, drift, size=3).astype(np.float32)
    out8 = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
    sharpen = int(_pick(rng, s["sharpen_percent"]))
    out8 = out8.filter(ImageFilter.UnsharpMask(radius=1.0, percent=sharpen, threshold=0))

    # 5. JPEG at satellite resolution (blocks are 8x8 satellite pixels, as in real tiles), then back
    #    to the render size so pixel labels stay valid
    quality = int(round(_pick(rng, s["jpeg_quality"])))
    buf = io.BytesIO()
    out8.save(buf, "JPEG", quality=quality, subsampling=2)
    buf.seek(0)
    final = Image.open(buf).convert("RGB").resize((w, h), Image.BICUBIC)

    tmp = path + ".tmp.png"
    final.save(tmp, "PNG")
    os.replace(tmp, path)

    return {"effective_gsd_meters": round(effective_gsd, 4) if effective_gsd else None,
            "resolution_factor": round(factor, 3), "psf_sigma_px": round(psf, 3),
            "shot_noise": round(shot, 5), "read_noise": round(read, 4),
            "sharpen_percent": sharpen, "jpeg_quality": quality}