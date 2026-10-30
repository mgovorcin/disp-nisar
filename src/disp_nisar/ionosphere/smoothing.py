"""Robust smoothing of the ionospheric screen, selected by filtering.method: robust.

1. Background clipping. Each epoch is compared with its own coarse background, and
   pixels too far from it are dropped. A few bad pixels at a coast otherwise become a
   large bullseye after a 2 km kernel.
2. Empirical noise weights. The RMS over epochs of each pixel's short-wavelength phase
   measures what the pixel adds beyond the screen. Pixels far noisier than the median
   are excluded outright: at a swath edge they are all the fine kernel sees, so a
   weight cannot stop their bias from reaching the screen.
3. Near-range exclusion. The first kilometers of near range carry a frequency-A phase
   term that is not ionosphere, so the screen is not estimated there.
4. Normalized convolution with a support floor. Where the fine kernel's support is too
   low the value comes from a coarser kernel; where that has none the pixel stays NaN,
   so the screen is never extrapolated across a gap wider than the coarse kernel.

The combined screen is what is smoothed, so non-dispersive terms have already cancelled.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from dolphin.utils import take_looks
from scipy import ndimage

from .ionosphere_filter import (
    _convolve,
    _float_array,
    _gaussian_kernel,
    _mask_array,
    _normalized_filter,
)
from .options import IonosphereFilterOptions

_TRUNCATE = 4.0


def normalized_convolution(
    values: np.ndarray,
    weights: np.ndarray,
    sigma_px: float | tuple[float, float],
    min_support: float | None = None,
    fallback_sigma_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a weighted Gaussian mean with a floor on the kernel's support.

    Parameters
    ----------
    values, weights : np.ndarray
        Samples, and their weights. Non-finite samples and non-positive weights do
        not contribute.
    sigma_px : float or tuple[float, float]
        Gaussian standard deviation in pixels, or (row, column) values.
    min_support : float, optional
        Pixels whose kernel has less of its mass on valid samples take the coarser
        kernel's value, or NaN when no coarser kernel is given.
    fallback_sigma_px : float, optional
        Coarser kernel. Pixels below the floor for this kernel too are NaN.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Smoothed values and the fine kernel's support in [0, 1], both float32.

    """

    def kernels(sigma: float | tuple[float, float]) -> tuple[np.ndarray, ...]:
        rows, cols = np.broadcast_to(sigma, 2)
        return _gaussian_kernel(rows, _TRUNCATE), _gaussian_kernel(cols, _TRUNCATE)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    fine = kernels(sigma_px)
    out = _normalized_filter(values, weights, fine)
    support = _convolve(valid.astype(np.float64), fine)
    if min_support is not None:
        low = support < min_support
        coarse = np.full(values.shape, np.nan)
        if fallback_sigma_px is not None and low.any():
            wide = kernels(fallback_sigma_px)
            coarse = _normalized_filter(values, weights, wide)
            coarse[_convolve(valid.astype(np.float64), wide) < min_support] = np.nan
        out = np.where(low, coarse, out)
    return out.astype(np.float32), support.astype(np.float32)


def _block_mean(
    values: np.ndarray, valid: np.ndarray, looks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean of the valid pixels of each block, and their count."""
    total = take_looks(
        np.where(valid, values, 0.0), looks, looks, "nansum", edge_strategy="pad"
    )
    count = take_looks(valid.astype(float), looks, looks, "nansum", edge_strategy="pad")
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0), (
        count
    )


def _upsample(values: np.ndarray, looks: int, shape: tuple[int, int]) -> np.ndarray:
    return np.repeat(np.repeat(values, looks, 0), looks, 1)[: shape[0], : shape[1]]


def background_clip(
    values: np.ndarray,
    valid: np.ndarray,
    block_px: int,
    max_dev: float,
    smooth_blocks: float = 1.5,
) -> np.ndarray:
    """Drop pixels further than `max_dev` from a smoothed block-median background.

    Returns
    -------
    np.ndarray
        Pixels kept, without the one-pixel border around everything dropped.

    """
    block_px = max(int(block_px), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = take_looks(
            np.where(valid, values, np.nan),
            block_px,
            block_px,
            "nanmedian",
            edge_strategy="pad",
        )
    known = np.isfinite(median)
    numerator = ndimage.gaussian_filter(
        np.where(known, median, 0.0), smooth_blocks, mode="constant"
    )
    denominator = ndimage.gaussian_filter(
        known.astype(float), smooth_blocks, mode="constant"
    )
    coarse = np.divide(
        numerator,
        denominator,
        out=np.full(median.shape, np.nan),
        where=denominator > 1e-3,
    )
    background = _upsample(coarse, block_px, values.shape)
    with np.errstate(invalid="ignore"):
        kept = valid & (np.abs(values - background) <= max_dev)
    return ndimage.binary_erosion(kept)


def empirical_noise(
    epochs: Iterable[np.ndarray],
    valid: np.ndarray,
    sigma_px: float,
    reduce: int = 5,
    min_epochs: int = 4,
) -> np.ndarray:
    """Return the RMS over epochs of each block's phase minus its smooth at `sigma_px`.

    This is a relative quality map, not an uncertainty of the correction: screen noise
    is spatially correlated, and on the ab36 frames the scatter at a 2 km kernel is 4-5
    times the independent-pixel propagation.

    Parameters
    ----------
    epochs : Iterable[np.ndarray]
        One phase array per epoch, NaN where invalid.
    valid : np.ndarray
        Pixels that may contribute.
    sigma_px : float
        Smoothing scale removed from each epoch, in input pixels.
    reduce : int
        Block size. The value is the noise of the block mean.
    min_epochs : int
        Blocks with fewer valid epochs are NaN.

    Returns
    -------
    np.ndarray
        Noise as float32, on the input grid.

    Raises
    ------
    ValueError
        If no epochs are given.

    """
    total = count = None
    for phase in epochs:
        good = valid & np.isfinite(phase)
        mean, n = _block_mean(phase, good, reduce)
        known = n > 0
        if total is None or count is None:
            total, count = np.zeros(mean.shape), np.zeros(mean.shape)
        smooth, _ = normalized_convolution(mean, known.astype(float), sigma_px / reduce)
        keep = known & np.isfinite(smooth)
        total += np.where(keep, mean - smooth, 0.0) ** 2
        count += keep
    if total is None or count is None:
        raise ValueError("Empirical noise needs at least one epoch")
    rms = np.where(count >= min_epochs, np.sqrt(total / np.maximum(count, 1)), np.nan)
    return _upsample(rms, reduce, valid.shape).astype(np.float32)


def noise_weights(noise: np.ndarray, cap: float = 50.0) -> np.ndarray:
    """Return median(noise^2) / noise^2 capped at `cap`, and 1 where it is unknown."""
    variance = noise.astype(np.float64) ** 2
    known = np.isfinite(variance) & (variance > 0)
    median = np.median(variance[known])
    weights = np.ones(noise.shape, dtype=np.float32)
    weights[known] = np.minimum(median / variance[known], cap)
    return weights


def near_range_band(
    footprint: np.ndarray, incidence: np.ndarray, pixel_m: float, km: float
) -> np.ndarray:
    """Return the footprint within `km` of its near-range edge.

    The edge is the footprint boundary at the lowest decile of incidence, so the
    along-track frame ends are not excluded.
    """
    filled = ndimage.binary_fill_holes(footprint)
    angle = np.where(np.isfinite(incidence) & (incidence > 0), incidence, np.nan)
    boundary = filled & ~ndimage.binary_erosion(filled, border_value=0)
    with np.errstate(invalid="ignore"):
        near_edge = boundary & (angle <= np.nanquantile(angle[filled], 0.1))
    if not near_edge.any():
        return np.zeros_like(filled)
    distance_km = ndimage.distance_transform_edt(~near_edge) * pixel_m / 1000.0
    return filled & (distance_km < km)


def smooth_screen(
    phase: np.ndarray,
    trusted: np.ndarray,
    footprint: np.ndarray,
    weights: np.ndarray | None,
    sigma_px: float | tuple[float, float],
    min_support: float = 0.2,
    fallback_sigma_px: float | None = None,
    clip_max_dev: float | None = None,
    clip_block_px: int = 250,
    noise: np.ndarray | None = None,
    exclude_noise_factor: float | None = None,
    exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth one epoch's screen from the pixels that survive every exclusion.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The screen, NaN outside `footprint` and where neither kernel has enough
        support, and the source pixels actually used.

    """
    sources = trusted & footprint & np.isfinite(phase)
    if exclude_mask is not None:
        sources &= ~exclude_mask
    if noise is not None and exclude_noise_factor is not None:
        known = sources & np.isfinite(noise)
        if known.any():
            limit = exclude_noise_factor * np.median(noise[known])
            sources &= ~(np.isfinite(noise) & (noise > limit))
    if clip_max_dev is not None:
        sources = background_clip(phase, sources, clip_block_px, clip_max_dev)
    used = np.ones(phase.shape, np.float32) if weights is None else weights
    out, _ = normalized_convolution(
        np.where(sources, phase, np.nan),
        np.where(sources, used, 0.0),
        sigma_px,
        min_support,
        fallback_sigma_px,
    )
    return np.where(footprint, out, np.nan).astype(np.float32), sources


def filter_ionosphere_robust(
    phase: np.ndarray,
    *,
    donor_mask: np.ndarray,
    footprint: np.ndarray,
    pixel_size_m: tuple[float, float],
    settings: IonosphereFilterOptions,
    sigma_m: tuple[float, float],
    noise: np.ndarray | None = None,
    exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply robust smoothing to an aligned unwrapped phase array.

    Parameters
    ----------
    phase : np.ndarray
        Real unwrapped ionospheric phase, in radians, on a single grid.
    donor_mask, footprint : np.ndarray
        Original observation support and allowed output domain, respectively.
    pixel_size_m, sigma_m : tuple[float, float]
        Pixel spacing and Gaussian standard deviations in (row, column) order.
    settings : IonosphereFilterOptions
        Support floor, coarse kernel, clipping, and exclusion controls.
    noise : np.ndarray, optional
        Empirical noise of the stack. Without it, pixels are weighted equally.
    exclude_mask : np.ndarray, optional
        Pixels never used as sources, such as the near-range band.

    Returns
    -------
    np.ndarray
        Smoothed phase as float32.

    """
    data = _float_array(phase)
    domain = _mask_array(footprint, data.shape)
    valid = _mask_array(donor_mask, data.shape) & domain & np.isfinite(data)
    if not valid.any():
        return np.full(data.shape, np.nan, np.float32)
    known_noise = noise is not None and bool(np.isfinite(noise).any())
    out, _ = smooth_screen(
        data,
        valid,
        domain,
        noise_weights(noise) if noise is not None and known_noise else None,
        (sigma_m[0] / pixel_size_m[0], sigma_m[1] / pixel_size_m[1]),
        min_support=settings.robust_min_support,
        fallback_sigma_px=settings.robust_fallback_sigma_m / pixel_size_m[1],
        clip_max_dev=(
            2 * np.pi * settings.robust_clip_cycles
            if settings.robust_clip_cycles is not None
            else None
        ),
        clip_block_px=int(settings.robust_clip_block_m / pixel_size_m[1]),
        noise=noise if known_noise else None,
        exclude_noise_factor=settings.robust_exclude_noise_factor,
        exclude_mask=exclude_mask,
    )
    return out
