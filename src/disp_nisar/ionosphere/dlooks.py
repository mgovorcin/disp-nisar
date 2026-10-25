"""Coarse unwrapping of the double-difference D (A·conj(B)).

The ionosphere is smooth, so D carries no information below a few hundred metres, while
every cycle error in D costs f_A f_B/(f_B² − f_A²) ≈ 11.7 cycles of ionosphere.
Unwrapping D on a grid `d_looks` times coarser than the B grid (400 m for d_looks=10 at
40 m) removes most of the D noise before unwrapping, makes the unwrapping of D a small
fraction of the A unwrapping cost, and leaves far fewer places for a cycle error to
start. A keeps its native grid: its scale factor is only ~0.51.

Only donor pixels (valid in both bands) contribute to a coarse block; the 1+0j interior
placeholders that the pair preparation writes for interpolation never do. The unwrapped
coarse D is filled by nearest neighbour inside the coarse footprint (at most
`max_fill_px` coarse pixels), bilinearly upsampled back to the B grid, and the donor
mask is reduced to the pixels where D has a value.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from scipy import ndimage

from ._main_diff_io import Grid


def block_sum(x: np.ndarray, k: int) -> np.ndarray:
    """Sum over k×k blocks; the last partial row/column of blocks is zero-padded."""
    h, w = x.shape
    hh, ww = -(-h // k), -(-w // k)
    dtype = np.complex128 if np.iscomplexobj(x) else np.float64
    pad = np.zeros((hh * k, ww * k), dtype=dtype)
    pad[:h, :w] = x
    return pad.reshape(hh, k, ww, k).sum(axis=(1, 3))


def coarse_grid(grid: Grid, k: int) -> Grid:
    """Return the k× coarser grid that shares the fine grid's origin."""
    return Grid(
        -(-grid.height // k),
        -(-grid.width // k),
        grid.crs,
        grid.transform * Affine.scale(k),
    )


def reduce_pair_d(z, donor, cor, sim, inside, k: int, min_fraction: float):
    """Reduce one prepared pair's D layers by k×k phasor averaging over donor pixels.

    Returns coarse (ifg, cor, sim, donor, inside, count). ``ifg`` follows the
    prepared-pair convention: unit phasor where coarse donor, 1+0j placeholder where
    inside but not donor, 0 outside.
    """
    donor = donor.astype(bool)
    n = block_sum(donor.astype(np.float64), k)
    nz = np.maximum(n, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(donor, z / np.maximum(np.abs(z), 1e-12), 0)
    zc = block_sum(unit, k) / nz
    corc = block_sum(np.where(donor, cor, 0.0), k) / nz
    simc = block_sum(np.where(donor, sim, 0.0), k) / nz
    insidec = block_sum(inside.astype(np.float64), k) > 0
    mag = np.abs(zc)
    donorc = (n >= max(1.0, min_fraction * k * k)) & insidec & (mag > 0)
    ifg = np.where(
        insidec, np.where(donorc, zc / np.maximum(mag, 1e-12), 1 + 0j), 0
    ).astype(np.complex64)
    # the block-mean phasor magnitude is the coherence of the averaged D; keep the
    # smaller of it and the mean pixel correlation so the unwrapper's cost follows the
    # real quality of the coarse pixel
    corc = np.where(donorc, np.clip(np.minimum(corc, mag), 0, 1), 0).astype(np.float32)
    simc = np.where(donorc, simc, 0).astype(np.float32)
    return ifg, corc, simc, donorc, insidec, n


def upsample_unwrapped(
    unw_c, cc_c, insidec, cgrid: Grid, grid: Grid, max_fill_px: int = 5
):
    """Fill and upsample the coarse unwrapped D onto the fine grid.

    NaNs in the coarse footprint are nearest-filled (at most `max_fill_px`); the phase
    is upsampled bilinearly and the connected components by nearest neighbour.
    """
    ok = np.isfinite(unw_c)
    if not ok.any():
        raise ValueError("coarse D unwrapping produced no finite phase")
    dist, (iy, ix) = ndimage.distance_transform_edt(~ok, return_indices=True)
    keep = insidec & (dist <= max_fill_px)
    filled = np.where(keep, unw_c[iy, ix], np.nan).astype(np.float32)
    cc_f = np.where(keep, cc_c[iy, ix], 0).astype(cc_c.dtype)
    unw = np.full((grid.height, grid.width), np.nan, np.float32)
    reproject(
        filled,
        unw,
        src_transform=cgrid.transform,
        src_crs=cgrid.crs,
        src_nodata=np.nan,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    cc = np.zeros((grid.height, grid.width), cc_c.dtype)
    reproject(
        cc_f,
        cc,
        src_transform=cgrid.transform,
        src_crs=cgrid.crs,
        src_nodata=0,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return unw, cc


def unwrap_d_coarse(prepared: dict, folder, cfg, options, unwrap_fn):
    """Unwrap D at `options.d_looks` looks and publish it on the fine grid.

    Writes D.unw.tif and D.unw.conncomp.tif.

    `unwrap_fn` is dolphin.unwrap.unwrap (injected for testing). The pair's donor raster
    is rewritten so that donors are exactly the pixels where the upsampled D is finite.
    """
    k = int(options.d_looks)
    grid = Grid.from_file(prepared["D_ifg"])

    def rd(key):
        with rasterio.open(prepared[key]) as s:
            return s.read(1)

    z, donor, cor, sim, inside = (
        rd("D_ifg"),
        rd("donor") == 1,
        rd("D_cor"),
        rd("D_sim"),
        rd("inside") == 1,
    )
    ifg, corc, simc, donorc, insidec, _ = reduce_pair_d(
        z, donor, cor, sim, inside, k, options.d_looks_min_fraction
    )
    cg = coarse_grid(grid, k)
    paths = {}
    for name, arr, dtype, nod in (
        ("D_ifg_c", ifg, "complex64", 0),
        ("D_cor_c", corc, "float32", None),
        ("D_sim_c", simc, "float32", None),
        ("inside_c", insidec.astype("uint8"), "uint8", None),
    ):
        p = folder / f"{name}.tif"
        with rasterio.open(p, "w", **cg.profile(dtype, nod)) as dst:
            dst.write(arr, 1)
        paths[name] = p
    unw_c_path, cc_c_path = unwrap_fn(
        ifg_filename=paths["D_ifg_c"],
        corr_filename=paths["D_cor_c"],
        unw_filename=folder / "D_c.unw.tif",
        nlooks=options.nlooks * k * k,
        mask_filename=paths["inside_c"],
        similarity_filename=paths["D_sim_c"],
        unwrap_options=cfg,
        unw_nodata=np.nan,
        ccl_nodata=0,
        scratchdir=folder / "scratch_D",
        delete_scratch=not options.keep_intermediates,
    )
    with rasterio.open(unw_c_path) as s:
        unw_c = s.read(1)
    with rasterio.open(cc_c_path) as s:
        cc_c = s.read(1)
    unw, cc = upsample_unwrapped(unw_c, cc_c, insidec, cg, grid)
    donor_f = donor & np.isfinite(unw)
    unw = np.where(inside & np.isfinite(unw), unw, np.nan).astype(np.float32)
    out_unw, out_cc = folder / "D.unw.tif", folder / "D.unw.conncomp.tif"
    with rasterio.open(out_unw, "w", **grid.profile("float32", np.nan)) as dst:
        dst.write(unw, 1)
        dst.update_tags(
            d_looks=str(k), d_looks_min_fraction=str(options.d_looks_min_fraction)
        )
    with rasterio.open(out_cc, "w", **grid.profile(str(cc.dtype), 0)) as dst:
        dst.write(cc, 1)
    with rasterio.open(prepared["donor"], "w", **grid.profile("uint8", None)) as dst:
        dst.write(donor_f.astype("uint8"), 1)
    if not options.keep_intermediates:
        for p in (*paths.values(), unw_c_path, cc_c_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    return out_unw, out_cc
