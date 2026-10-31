"""Improved main_side ionosphere, from the two bands' unwrapped time series.

Same two-frequency estimator as :func:`.gslc.estimate_iono_main_side`, with the changes
that made it competitive on the ab36 frames (variant v5 ``kwp_s5km_c10``):

1. Area referencing of B to A, per date (median(B - A) = 0 over the mask), instead of
   one reference pixel, whose noise becomes a frame-wide offset amplified ~11 times.
2. Masking on both bands' temporal coherence and similarity before estimation.
3. Band consistency: pixels whose A - B residual leaves its large-scale part by more
   than 0.2 cycle are dropped per date (and masked on every date when that happens on at
   least half of them), so the kernel fills them instead of smearing them into a
   bullseye.
4. Normalized convolution at the ionosphere's scale, weighted by the inverse variance of
   the dispersive phase from both bands' coherence (Cramer-Rao).

Works on a 90 m grid (3 x 3 of A's 30 m); the screen is written back on A's grid, in
meters, referenced to A's reference pixel. Bad-date checks are not part of this recipe.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.ndimage import binary_dilation, gaussian_filter

from .gslc import estimate_iono_main_side
from .options import IonosphereOptions
from .utils import (
    disp_to_phase,
    freq_to_wavelength,
    phase_to_disp,
    read_reference_point,
)

logger = logging.getLogger(__name__)

__all__ = ["run_main_side_recipe"]

MULTILOOK = 3
TCOH_MIN, SIM_MIN = 0.6, 0.5
WEIGHT_MAX = 50.0
CONSISTENCY_CYCLES = 0.2
PERSISTENT_FRACTION = 0.5
LOOKS = (9, 5)  # looks at 90 m: A's 30 m, B's 40 m


def _multilook(a: np.ndarray, n: int = MULTILOOK) -> np.ndarray:
    r, c = (a.shape[0] // n) * n, (a.shape[1] // n) * n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a[:r, :c].reshape(r // n, n, c // n, n), axis=(1, 3))


def _read_a(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1).astype("float32")
    a[a == 0] = np.nan
    return _multilook(a).astype("float32")


def _read_on(path: Path, shape, transform, crs, zero_is_nodata=False) -> np.ndarray:
    out = np.full(shape, np.nan, "float32")
    with rasterio.open(path) as src:
        a = src.read(1).astype("float32")
        if zero_is_nodata:
            a[a == 0] = np.nan
        elif src.nodata is not None:
            a[a == src.nodata] = np.nan
        reproject(
            a,
            out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    return out


def _large_scale(r: np.ndarray, ok: np.ndarray, block: int) -> np.ndarray:
    """Return the block-median part of r, gap-filled and smoothed back to r's grid."""
    h, w = r.shape
    rows, cols = -(-h // block), -(-w // block)
    pad = np.full((rows * block, cols * block), np.nan)
    pad[:h, :w] = np.where(ok, r, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(pad.reshape(rows, block, cols, block), axis=(1, 3))
    m = np.isfinite(med)
    num = gaussian_filter(np.where(m, med, 0.0), 1.5)
    den = gaussian_filter(m.astype(float), 1.5)
    with np.errstate(all="ignore"):
        coarse = np.where(den > 1e-3, num / den, np.nan)
    return np.repeat(np.repeat(coarse, block, 0), block, 1)[:h, :w]


def _dispersive_weight(
    tcoh_a: np.ndarray, tcoh_b: np.ndarray, f_a: float, f_b: float
) -> np.ndarray:
    """Return 1/var of the dispersive phase from both coherences, median-normalized."""
    m_inv = np.linalg.inv(np.array([[1.0, 1.0], [f_b / f_a, f_a / f_b]]))
    var = np.zeros_like(tcoh_a, dtype="float64")
    for coef, looks, g in zip(m_inv[1], LOOKS, (tcoh_a, tcoh_b), strict=True):
        g = np.clip(np.nan_to_num(g, nan=0.05), 0.05, 0.995)
        var = var + coef**2 * (1 - g**2) / (2 * looks * g**2)
    w = 1.0 / var
    return np.minimum(w / np.median(w[np.isfinite(w)]), WEIGHT_MAX)


def _residual_cycles(r: np.ndarray, ok: np.ndarray, block: int) -> np.ndarray:
    return np.nan_to_num((r - _large_scale(r, ok, block)) / (2 * np.pi))


def run_main_side_recipe(
    ts_dir_a: str | Path,
    ts_dir_b: str | Path,
    quality_a: tuple[Path, Path],
    quality_b: tuple[Path, Path],
    out_dir: str | Path,
    output_timeseries_paths: list[Path],
    f_a: float,
    f_b: float,
    options: IonosphereOptions,
) -> list[Path]:
    """Write the improved main_side correction for every output pair; return the paths.

    `quality_a` and `quality_b`: each band's (temporal coherence, similarity) rasters.
    """
    ts_dir_a, ts_dir_b, out_dir = Path(ts_dir_a), Path(ts_dir_b), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wl_a, wl_b = freq_to_wavelength(f_a), freq_to_wavelength(f_b)
    pairs = sorted(p.stem for p in output_timeseries_paths)

    with rasterio.open(ts_dir_a / f"{pairs[0]}.tif") as src:
        profile = src.profile
        full_shape = (src.height, src.width)
        full_transform = src.transform
    transform = full_transform * full_transform.scale(MULTILOOK, MULTILOOK)
    shape = (full_shape[0] // MULTILOOK, full_shape[1] // MULTILOOK)
    crs = profile["crs"]
    px = abs(transform.a)
    sigma_px = options.main_side_sigma_km * 1000 / px
    block = round(options.main_side_consistency_km * 1000 / px)

    tcoh_a, sim_a = (_read_on(p, shape, transform, crs) for p in quality_a)
    tcoh_b, sim_b = (_read_on(p, shape, transform, crs) for p in quality_b)
    mask = (tcoh_a > TCOH_MIN) & (np.abs(sim_a) > SIM_MIN)
    mask &= (tcoh_b > TCOH_MIN) & (np.abs(sim_b) > SIM_MIN)
    weight = _dispersive_weight(tcoh_a, tcoh_b, f_a, f_b)

    def read_pair(pair):
        a = _read_a(ts_dir_a / f"{pair}.tif")[: shape[0], : shape[1]]
        b = _read_on(ts_dir_b / f"{pair}.tif", shape, transform, crs, True)
        ok = mask & np.isfinite(a) & np.isfinite(b)
        if not ok.any():
            raise ValueError(f"{pair}: no valid pixels in both bands")
        b = b - float(np.median((b - a)[ok]))
        return (
            ok,
            disp_to_phase(np.where(ok, a, 0.0), wl_a),
            disp_to_phase(np.where(ok, b, 0.0), wl_b),
        )

    n_bad, n_ok = np.zeros(shape), np.zeros(shape)
    for pair in pairs:
        ok, phi_a, phi_b = read_pair(pair)
        res = _residual_cycles(phi_b - (f_b / f_a) * phi_a, ok, block)
        n_bad += ok & (np.abs(res) > CONSISTENCY_CYCLES)
        n_ok += ok
    persistent = (n_ok >= 4) & (n_bad / np.maximum(n_ok, 1) >= PERSISTENT_FRACTION)
    mask &= ~binary_dilation(persistent, iterations=1)
    w = np.where(mask, weight, 0.0)

    row, col = read_reference_point(ts_dir_a / "reference_point.txt")
    row, col = row // MULTILOOK, col // MULTILOOK
    outputs = []
    for pair in pairs:
        ok, phi_a, phi_b = read_pair(pair)
        res = _residual_cycles(phi_b - (f_b / f_a) * phi_a, ok, block)
        bad = binary_dilation(np.abs(res) > CONSISTENCY_CYCLES, iterations=1) & ok
        ok &= ~bad
        disp, _ = estimate_iono_main_side(f_a, f_b, phi_a, phi_b)
        iono = phase_to_disp(disp, wl_a).astype("float64")
        num = gaussian_filter(np.where(ok, iono, 0.0) * w, sigma_px)
        den = gaussian_filter(np.where(ok, w, 0.0), sigma_px)
        with np.errstate(invalid="ignore", divide="ignore"):
            screen = np.where(den > 1e-6, num / den, np.nan)
        window = screen[max(row - 1, 0) : row + 2, max(col - 1, 0) : col + 2]
        if not np.isfinite(window).any():
            raise ValueError(f"{pair}: no screen at the reference pixel")
        screen -= np.nanmean(window)

        full = np.full(full_shape, np.nan, "float32")
        reproject(
            screen.astype("float32"),
            full,
            src_transform=transform,
            src_crs=crs,
            dst_transform=full_transform,
            dst_crs=crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        output = out_dir / f"{pair}_iono.tif"
        out_profile = profile | {
            "dtype": "float32",
            "nodata": np.nan,
            "compress": "deflate",
            "count": 1,
        }
        with rasterio.open(output, "w", **out_profile) as dst:
            dst.write(full, 1)
            dst.set_band_unit(1, "meters")
            dst.update_tags(ionosphere_method="main_side_recipe")
        outputs.append(output)
        logger.info("%s: main_side recipe, %.2f %% dropped", pair, bad.mean() * 100)
    return outputs
