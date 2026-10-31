"""Robust smoothing of the per-epoch ionospheric screen (requirement R-F8).

Replaces the gap-fill + Gaussian of the first main_diff implementation with:

1. **Background clipping.** Each epoch is compared with its own ~10 km background (block
   medians, smoothed); pixels further than `clip_cycles` from it are dropped before
   smoothing. A few bad pixels at a coast or a hole boundary otherwise become a ±1000 mm
   bullseye after a 2 km kernel.
2. **Empirical noise weights.** The RMS over epochs of each pixel's short-wavelength
   phase (phase minus its 10 km smooth) measures what the pixel contributes beyond the
   screen; weights are median(noise²)/noise², capped. It halves the noise of the
   near-range strip and of the interior on track 042 (2026-10-11).
3. **Normalized convolution with a support floor.** G(w·x)/G(w) at the screen scale;
   where the kernel's support (fraction of its mass on trusted pixels) is below
   `min_support` the value comes from a coarser kernel instead, and where even that has
   no support the pixel is left NaN: the screen is never extrapolated across a gap
   larger than the coarse kernel (no freqB coverage, open water).

The combined screen (A and D already merged) is what is smoothed, so non-dispersive
terms have cancelled before any filtering (otherwise the two bands' different smoothing
leaks them back in).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import ndimage

__all__ = [
    "near_range_band",
    "background_clip",
    "block_reduce_mean",
    "empirical_noise",
    "noise_weights",
    "normalized_convolution",
    "smooth_screen",
]


def normalized_convolution(
    x: np.ndarray,
    w: np.ndarray,
    sigma_px: float,
    min_support: float | None = None,
    fallback_sigma_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted normalized convolution G(w x)/G(w); returns (smoothed, support).

    `support` is G(valid)/G(1) in [0, 1]. With `min_support` and `fallback_sigma_px`,
    pixels whose support is below the floor take the coarser kernel's value; pixels
    below the floor for the coarse kernel too are NaN.
    """
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    wx = np.where(ok, w * x, 0.0).astype(np.float64)
    ww = np.where(ok, w, 0.0).astype(np.float64)
    num = ndimage.gaussian_filter(wx, sigma_px, mode="constant")
    den = ndimage.gaussian_filter(ww, sigma_px, mode="constant")
    support = ndimage.gaussian_filter(ok.astype(np.float64), sigma_px, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    if min_support is not None:
        low = support < min_support
        if fallback_sigma_px is not None and low.any():
            num2 = ndimage.gaussian_filter(wx, fallback_sigma_px, mode="constant")
            den2 = ndimage.gaussian_filter(ww, fallback_sigma_px, mode="constant")
            sup2 = ndimage.gaussian_filter(
                ok.astype(np.float64), fallback_sigma_px, mode="constant"
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                coarse = np.where(
                    (den2 > 0) & (sup2 >= min_support), num2 / den2, np.nan
                )
            out = np.where(low, coarse, out)
        else:
            out = np.where(low, np.nan, out)
    return out.astype(np.float32), support.astype(np.float32)


def block_reduce_mean(
    x: np.ndarray, ok: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean over k×k blocks of the valid pixels; returns (mean, fraction valid)."""
    h, w = x.shape
    hh, ww = -(-h // k), -(-w // k)
    pad_x = np.zeros((hh * k, ww * k))
    pad_n = np.zeros((hh * k, ww * k))
    pad_x[:h, :w] = np.where(ok, x, 0.0)
    pad_n[:h, :w] = ok
    s = pad_x.reshape(hh, k, ww, k).sum(axis=(1, 3))
    n = pad_n.reshape(hh, k, ww, k).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan), n / (k * k)


def _upsample(x: np.ndarray, k: int, shape: tuple[int, int]) -> np.ndarray:
    return np.repeat(np.repeat(x, k, 0), k, 1)[: shape[0], : shape[1]]


def background_clip(
    x: np.ndarray,
    ok: np.ndarray,
    block_px: int,
    max_dev: float,
    smooth_blocks: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop pixels further than `max_dev` from the block-median background.

    Returns (kept, background).
    """
    h, w = x.shape
    k = max(int(block_px), 1)
    hh, ww = -(-h // k), -(-w // k)
    pad = np.full((hh * k, ww * k), np.nan)
    pad[:h, :w] = np.where(ok, x, np.nan)
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(pad.reshape(hh, k, ww, k), axis=(1, 3))
    m = np.isfinite(med)
    num = ndimage.gaussian_filter(np.where(m, med, 0.0), smooth_blocks, mode="constant")
    den = ndimage.gaussian_filter(m.astype(float), smooth_blocks, mode="constant")
    with np.errstate(all="ignore"):
        coarse = np.where(den > 1e-3, num / den, np.nan)
    background = _upsample(coarse, k, (h, w))
    dev = np.abs(x - background)
    kept = ok & np.isfinite(background) & (dev <= max_dev)
    kept = ndimage.binary_erosion(kept, iterations=1) | (
        kept & ~ok
    )  # one-pixel border around drops
    return kept, background.astype(np.float32)


def empirical_noise(
    epochs: Iterable[np.ndarray],
    ok: np.ndarray,
    sigma_px: float,
    reduce: int = 5,
    min_epochs: int = 4,
) -> np.ndarray:
    """RMS over epochs of (phase − its masked smooth at `sigma_px`), on a coarser grid.

    The grid is `reduce` times coarser than the input.

    `epochs` yields one array per epoch (same shape as `ok`; NaN invalid). Returns the
    noise upsampled to the input shape (NaN where fewer than `min_epochs` epochs were
    valid). The value is the noise of the `reduce`×`reduce` block mean (white noise:
    pixel noise / reduce), which is the scale the screen-size kernel sees; only ratios
    enter the weights.

    Do not propagate this value to the smoothed screen as if pixels were independent:
    the screen noise is spatially correlated, and on the ab36 frames the scatter at a 2
    km kernel is 4-5x the independent-pixel propagation (3x at 400 m). It is a relative
    quality map, not an absolute uncertainty of the correction.
    """
    shape = ok.shape
    ss = nn = None
    for phi in epochs:
        valid = ok & np.isfinite(phi)
        red, frac = block_reduce_mean(phi, valid, reduce)
        rok = frac > 0
        if ss is None:
            ss, nn = np.zeros(red.shape), np.zeros(red.shape)
        sm, _ = normalized_convolution(red, rok.astype(float), sigma_px / reduce)
        hp = np.where(rok & np.isfinite(sm), red - sm, 0.0)
        ss += hp**2
        nn += rok & np.isfinite(sm)
    if ss is None or nn is None:
        raise ValueError("no epochs")
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.where(nn >= min_epochs, np.sqrt(ss / np.maximum(nn, 1)), np.nan)
    return _upsample(rms, reduce, shape).astype(np.float32)


def noise_weights(noise: np.ndarray, ok: np.ndarray, cap: float = 50.0) -> np.ndarray:
    """Return median(noise²)/noise² over `ok`, capped at `cap`; 1 where unknown."""
    var = noise.astype(np.float64) ** 2
    med = np.nanmedian(var[ok & np.isfinite(var)])
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(np.isfinite(var) & (var > 0), np.minimum(med / var, cap), 1.0)
    return w.astype(np.float32)


def near_range_band(
    footprint: np.ndarray, incidence: np.ndarray, pixel_m: float, km: float
) -> np.ndarray:
    """Pixels within `km` of the footprint's outer edge on the near-range side.

    The near-range side is where the incidence is below the frame median.

    The first kilometres of near range carry a frequency-A phase term that is not
    ionosphere (seasonal, band-dependent; track-042 study 2026-10-21); the screen must
    not be estimated there.
    """
    foot = ndimage.binary_fill_holes(footprint)
    inc = np.where(np.isfinite(incidence) & (incidence > 0), incidence, np.nan)
    # the near-range edge is the part of the footprint boundary at the lowest incidence
    # (lowest decile); distance is measured from that edge only, so the along-track
    # frame ends are not excluded
    boundary = foot & ~ndimage.binary_erosion(foot, border_value=0)
    near_edge = boundary & (inc <= np.nanquantile(inc[foot], 0.1))
    if not near_edge.any():
        return np.zeros_like(foot)
    dist = ndimage.distance_transform_edt(~near_edge) * pixel_m / 1000.0
    return foot & (dist < km)


def smooth_screen(
    iono: np.ndarray,
    trusted: np.ndarray,
    footprint: np.ndarray,
    weights: np.ndarray | None,
    sigma_px: float,
    min_support: float = 0.2,
    fallback_sigma_px: float | None = None,
    clip_max_dev: float | None = None,
    clip_block_px: int = 250,
    noise: np.ndarray | None = None,
    exclude_noise_factor: float | None = None,
    exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust smoothing of one epoch's screen.

    Returns (smoothed, quality_mask, support).

    `smoothed` is NaN outside `footprint` and wherever neither kernel has enough
    support. `exclude_mask` marks pixels that are never used as sources (the near-range
    band); they are filled by the kernels. With `noise` and `exclude_noise_factor`,
    pixels whose empirical noise exceeds `factor` × the median are not trusted at all
    (not merely down-weighted): at a swath edge they are the only pixels inside the fine
    kernel, so a weight cannot stop their bias from reaching the screen; exclusion makes
    the coarse kernel fill them from the interior instead. This is what removes the
    near-range strip (3–4× the interior noise, biased).
    """
    ok = trusted & footprint & np.isfinite(iono)
    if (
        exclude_mask is not None
    ):  # e.g. the near-range band (near_range_band): never a source for the screen
        ok &= ~exclude_mask
    if noise is not None and exclude_noise_factor is not None:
        med = np.nanmedian(noise[ok & np.isfinite(noise)])
        ok &= ~(np.isfinite(noise) & (noise > exclude_noise_factor * med))
    if clip_max_dev is not None:
        ok, _ = background_clip(iono, ok, clip_block_px, clip_max_dev)
    w = (
        np.ones(iono.shape, np.float32)
        if weights is None
        else np.where(np.isfinite(weights), weights, 1.0)
    )
    out, support = normalized_convolution(
        np.where(ok, iono, np.nan),
        np.where(ok, w, 0.0),
        sigma_px,
        min_support,
        fallback_sigma_px,
    )
    out = np.where(footprint, out, np.nan).astype(np.float32)
    return out, ok, support
