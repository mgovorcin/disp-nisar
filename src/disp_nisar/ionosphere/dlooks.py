"""Coarse unwrapping of the double difference D = A * conj(B).

The ionosphere is smooth, so D carries no information below a few hundred meters,
while every cycle error in D costs f_A f_B / (f_B^2 - f_A^2), about 11.7 cycles of
ionosphere. Unwrapping D on a grid d_looks times coarser than the B grid removes most
of the D noise first and leaves fewer places for a cycle error to start. A keeps its
native grid: its scale factor is only about 0.51.

Only donor pixels contribute to a coarse block; the 1+0j interior placeholders that
pair preparation writes never do. The unwrapped coarse D is filled by nearest neighbor
inside the coarse footprint, upsampled bilinearly to the B grid, and the donor mask is
reduced to the pixels where D has a value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from affine import Affine
from dolphin import io
from dolphin.utils import take_looks
from dolphin.workflows.config import UnwrapOptions
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage

from ._main_diff_io import Grid
from .options import IonosphereOptions
from .utils import WORKING_TIFF_OPTIONS


def _block_sum(values: np.ndarray, looks: int) -> np.ndarray:
    """Sum over looks-by-looks blocks, counting the padded edge as zero."""
    return take_looks(values, looks, looks, func_type="nansum", edge_strategy="pad")


def coarse_grid(grid: Grid, looks: int) -> Grid:
    """Return the grid `looks` times coarser that shares the fine grid's origin."""
    return Grid(
        -(-grid.height // looks),
        -(-grid.width // looks),
        grid.crs,
        grid.transform * Affine.scale(looks),
    )


def reduce_pair_d(
    z: np.ndarray,
    donor: np.ndarray,
    cor: np.ndarray,
    sim: np.ndarray,
    inside: np.ndarray,
    looks: int,
    min_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce one prepared pair's D layers by phasor averaging over donor pixels.

    Parameters
    ----------
    z, cor, sim : np.ndarray
        Prepared D interferogram, correlation, and similarity on the B grid.
    donor, inside : np.ndarray
        Prepared donor and footprint masks.
    looks : int
        Reduction factor along both axes.
    min_fraction : float
        Smallest donor fraction of a block that still counts as an observation.

    Returns
    -------
    tuple[np.ndarray, ...]
        Coarse interferogram, correlation, similarity, donor, and footprint. The
        interferogram holds unit phasors at coarse donors, 1+0j elsewhere inside the
        footprint, and 0 outside it, as the prepared pairs do.

    """
    count = _block_sum(donor.astype(np.float64), looks)
    denominator = np.maximum(count, 1.0)
    unit = np.zeros(z.shape, dtype=np.complex128)
    np.divide(z, np.abs(z), out=unit, where=donor & (np.abs(z) > 0))
    mean_z = _block_sum(unit, looks) / denominator
    mean_cor = _block_sum(np.where(donor, cor, 0.0), looks) / denominator
    mean_sim = _block_sum(np.where(donor, sim, 0.0), looks) / denominator
    coarse_inside = _block_sum(inside.astype(np.float64), looks) > 0
    magnitude = np.abs(mean_z)
    coarse_donor = (
        (count >= max(1.0, min_fraction * looks * looks))
        & coarse_inside
        & (magnitude > 0)
    )
    ifg = np.where(
        coarse_inside,
        np.where(coarse_donor, mean_z / np.maximum(magnitude, 1e-12), 1 + 0j),
        0,
    ).astype(np.complex64)
    # The block's phasor magnitude is the coherence of the averaged D. Keeping the
    # smaller of it and the mean correlation makes the unwrapper's cost follow the
    # real quality of the coarse pixel.
    coarse_cor = np.where(
        coarse_donor, np.clip(np.minimum(mean_cor, magnitude), 0, 1), 0
    )
    coarse_sim = np.where(coarse_donor, mean_sim, 0)
    return (
        ifg,
        coarse_cor.astype(np.float32),
        coarse_sim.astype(np.float32),
        coarse_donor,
        coarse_inside,
    )


def upsample_unwrapped(
    coarse_unw: np.ndarray,
    coarse_inside: np.ndarray,
    coarse: Grid,
    fine: Grid,
    max_fill_px: int = 5,
) -> np.ndarray:
    """Fill the coarse unwrapped D and upsample it bilinearly onto the fine grid.

    Gaps inside the coarse footprint take their nearest value, up to `max_fill_px`
    coarse pixels away; anything further stays NaN.

    Raises
    ------
    ValueError
        If the coarse D has no finite phase.

    """
    finite = np.isfinite(coarse_unw)
    if not finite.any():
        raise ValueError("Coarse D unwrapping produced no finite phase")
    distance, (rows, cols) = ndimage.distance_transform_edt(
        ~finite, return_indices=True
    )
    keep = coarse_inside & (distance <= max_fill_px)
    filled = np.where(keep, coarse_unw[rows, cols], np.nan).astype(np.float32)
    fine_unw = np.full((fine.height, fine.width), np.nan, np.float32)
    reproject(
        filled,
        fine_unw,
        src_transform=coarse.transform,
        src_crs=coarse.crs,
        src_nodata=np.nan,
        dst_transform=fine.transform,
        dst_crs=fine.crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return fine_unw


def unwrap_d_coarse(
    prepared: dict[str, Path],
    folder: Path,
    unwrap_options: UnwrapOptions,
    options: IonosphereOptions,
    unwrap: Callable,
) -> Path:
    """Unwrap D at `options.d_looks` looks and write it on the fine grid.

    Parameters
    ----------
    prepared : dict[str, Path]
        Outputs from pair preparation. The donor raster is rewritten so donors are
        exactly the pixels where the upsampled D is finite.
    folder : Path
        Working directory for the coarse rasters and the fine result.
    unwrap_options : dolphin.workflows.config.UnwrapOptions
        Configuration of the D unwrap.
    options : IonosphereOptions
        Looks, donor fraction, and intermediate-file settings.
    unwrap : Callable
        The dolphin unwrapper.

    Returns
    -------
    Path
        Unwrapped D in radians on the fine grid.

    """
    looks = options.d_looks
    fine = Grid.from_file(prepared["D_ifg"])
    coarse = coarse_grid(fine, looks)
    donor = io.load_gdal(prepared["donor"]) == 1
    inside = io.load_gdal(prepared["inside"]) == 1
    ifg, cor, sim, _, coarse_inside = reduce_pair_d(
        io.load_gdal(prepared["D_ifg"]),
        donor,
        io.load_gdal(prepared["D_cor"]),
        io.load_gdal(prepared["D_sim"]),
        inside,
        looks,
        options.d_looks_min_fraction,
    )

    def write(path: Path, array: np.ndarray, grid: Grid, nodata: float | None) -> Path:
        io.write_arr(
            arr=array,
            output_name=path,
            geotransform=grid.transform.to_gdal(),
            projection=grid.crs.to_wkt(),
            dtype=array.dtype,
            nbands=1,
            nodata=nodata,
            options=WORKING_TIFF_OPTIONS,
        )
        return path

    coarse_unw_path, _ = unwrap(
        ifg_filename=write(folder / "D_coarse_ifg.tif", ifg, coarse, 0),
        corr_filename=write(folder / "D_coarse_cor.tif", cor, coarse, None),
        unw_filename=folder / "D_coarse.unw.tif",
        nlooks=options.nlooks * looks * looks,
        mask_filename=write(
            folder / "inside_coarse.tif",
            coarse_inside.astype(np.uint8),
            coarse,
            None,
        ),
        similarity_filename=write(folder / "D_coarse_sim.tif", sim, coarse, None),
        unwrap_options=unwrap_options,
        unw_nodata=np.nan,
        ccl_nodata=0,
        scratchdir=folder / "scratch",
        delete_scratch=not options.keep_intermediates,
    )
    fine_unw = upsample_unwrapped(
        io.load_gdal(coarse_unw_path, masked=True).astype(np.float32).filled(np.nan),
        coarse_inside,
        coarse,
        fine,
    )
    finite = np.isfinite(fine_unw)
    write(prepared["donor"], (donor & finite).astype(np.uint8), fine, None)
    return write(
        folder / "D.unw.tif",
        np.where(inside & finite, fine_unw, np.nan).astype(np.float32),
        fine,
        np.nan,
    )
