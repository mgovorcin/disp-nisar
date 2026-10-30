"""A/(A-B) estimation with fixed date-based output paths.

Resume checks final-file existence only. Changing inputs, reference, grid or
settings requires resume=False (for the entire rerun), or removal of affected
outputs and all dependent stages. Hash/attempt caches are not auto-migrated.
Preparation/inversion are windowed; gap filling uses one full B-grid array.

Dolphin/SNAPHU still controls unwrap memory. Phase-linked SLCs are consumed through
existing stitched IFGs; independent A/B displacement series are not mixed here.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject
from rasterio.windows import Window

from ._main_diff_io import (
    Grid,
    atomic_json,
    check_reduction,
    read_window,
    windows,
)
from .gslc_mask import prepare_gslc_mask_cache
from .inversion import build_design_matrix, invert_common_phase_block
from .mask import apply_similarity_mask_and_fill
from .options import IonosphereOptions
from .resampling import average_to_window, resample_phasor_to_match, source_window

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BandInputs:
    """Existing stitched Dolphin outputs for one frequency."""

    interferograms: Sequence[Path]
    correlations: Sequence[Path]
    similarity: Path


def _date_pair(path: Path) -> tuple:
    from opera_utils import get_dates

    dates = [
        d.replace(hour=0, minute=0, second=0, microsecond=0) for d in get_dates(path)
    ]
    if len(dates) != 2 or dates[0] >= dates[1]:
        raise ValueError(f"Expected one ordered acquisition pair: {path}")
    return tuple(dates)


def _index(paths: Sequence[Path]) -> dict:
    result: dict[tuple, Path] = {}
    for p in paths:
        pair = _date_pair(p)
        if pair in result:
            raise ValueError(f"Duplicate pair {pair}: {result[pair]}, {p}")
        result[pair] = Path(p)
    return result


def estimate_iono_main_diff(
    f_a: float, f_b: float, phase_a: np.ndarray, phase_d: np.ndarray
) -> np.ndarray:
    """Estimate ionospheric phase at frequency A from A and A-minus-B.

    Parameters
    ----------
    f_a, f_b : float
        Distinct positive center frequencies in Hz.
    phase_a, phase_d : np.ndarray
        Matched unwrapped phase or phase time series, in radians, sharing spatial
        and temporal references, support, and phase convention.

    Returns
    -------
    np.ndarray
        Ionospheric phase at frequency A, in the input phase convention.

    """
    if not (np.isfinite(f_a) and np.isfinite(f_b) and min(f_a, f_b) > 0):
        raise ValueError("Center frequencies must be finite and positive")
    if abs(f_a - f_b) <= 1e-8 * max(f_a, f_b):
        raise ValueError("Center frequencies are indistinguishable")
    if phase_a.shape != phase_d.shape:
        raise ValueError("A and D shapes differ")
    return (
        f_b / (f_a + f_b) * phase_a.astype(np.float64)
        + f_a * f_b / (f_b * f_b - f_a * f_a) * phase_d.astype(np.float64)
    ).astype(np.float32)


def _same_grid(path, grid):
    if Grid.from_file(Path(path)) != grid:
        raise ValueError(f"Grid differs from the band's stitched IFG: {path}")


def _prepare_pair(inputs, masks, target, folder, options):
    source = Grid.from_file(inputs["ifg_A"])
    check_reduction(source, target)
    specs = {
        "A_ifg": "complex64",
        "D_ifg": "complex64",
        "A_cor": "float32",
        "D_cor": "float32",
        "A_sim": "float32",
        "D_sim": "float32",
        "donor": "uint8",
        "inside": "uint8",
    }
    outputs = {n: folder / f"{n}.tif" for n in specs}
    with ExitStack() as stack:
        readers = {n: stack.enter_context(rasterio.open(p)) for n, p in inputs.items()}
        mr = {n: stack.enter_context(rasterio.open(p)) for n, p in masks.items()}
        writers = {
            n: stack.enter_context(
                rasterio.open(
                    outputs[n], "w", **target.profile(dtype, 0 if "ifg" in n else None)
                )
            )
            for n, dtype in specs.items()
        }
        for win in windows(target, options.block_size):
            sw = source_window(source, target, win)
            native = {}
            for band, w in (("A", sw), ("B", win)):
                z = read_window(readers[f"ifg_{band}"], w)
                q = read_window(readers[f"cor_{band}"], w)
                sim = read_window(readers[f"sim_{band}"], w)
                valid = (read_window(mr[f"{band}_ref_valid"], w) == 1) & (
                    read_window(mr[f"{band}_sec_valid"], w) == 1
                )
                inside = (read_window(mr[f"{band}_ref_inside"], w) == 1) & (
                    read_window(mr[f"{band}_sec_inside"], w) == 1
                )
                good = (
                    valid
                    & inside
                    & np.isfinite(z)
                    & (np.abs(z) > 0)
                    & np.isfinite(q)
                    & (q >= options.correlation_threshold)
                    & np.isfinite(sim)
                    & (sim >= options.similarity_threshold)
                )
                # Numeric quality is retained at donors; the mask never becomes quality.
                native[band] = (
                    z,
                    np.where(good, np.clip(q, 0, 1), 0),
                    np.where(good, sim, 0),
                    good,
                    inside,
                )
            za, qa, sa, ga, ia = native["A"]
            zb, qb, sb, gb, ib = native["B"]
            ua, good_a, support = resample_phasor_to_match(
                za,
                ga,
                source,
                sw,
                target,
                win,
                options.min_valid_fraction,
                options.min_phasor_magnitude,
            )
            qa = average_to_window(qa, source, sw, target, win) / np.maximum(
                support, 1e-12
            )
            sa = average_to_window(sa, source, sw, target, win) / np.maximum(
                support, 1e-12
            )
            inside = (average_to_window(ia, source, sw, target, win) > 0) & ib
            good_a &= inside
            good_d = good_a & gb
            ub = np.zeros(zb.shape, np.complex64)
            np.divide(zb, np.abs(zb), out=ub, where=gb)
            # Interior placeholders permit interpolation, but NEVER become observations.
            ai = np.where(inside, np.where(good_a, ua, 1 + 0j), 0)
            di = np.where(inside, np.where(good_d, ua * np.conj(ub), 1 + 0j), 0)
            data = {
                "A_ifg": ai,
                "D_ifg": di,
                "A_cor": np.where(good_a, qa, 0),
                "D_cor": np.where(good_d, np.clip(qa * qb, 0, 1), 0),
                "A_sim": np.where(good_a, sa, 0),
                "D_sim": np.where(good_d, np.minimum(sa, sb), 0),
                "donor": good_d,
                "inside": inside,
            }
            for name, a in data.items():
                writers[name].write(a.astype(specs[name]), 1, window=win)
    return outputs


def _process_pair(inputs, masks, folder, target, unwrap_options, options):
    """Prepare and unwrap one pair, publishing its files only once it completed.

    Returns (ready, status).
    """
    import shutil

    folder.mkdir(parents=True, exist_ok=True)
    # These are the files consumed by reference selection and inversion.
    ready = {
        "A_unw": folder / "A.unw.tif",
        "D_unw": folder / "D.unw.tif",
        "donor": folder / "donor.tif",
    }
    if options.resume and all(p.is_file() for p in ready.values()):
        return ready, "reused"
    # One fixed working directory, not a new attempt directory each time.
    work = folder / "_partial"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    generated = _unwrap_pair(
        _prepare_pair(inputs, masks, target, work, options),
        work,
        unwrap_options,
        options,
    )
    ready = {}
    for key_name, path in generated.items():
        destination = folder / path.name
        path.replace(destination)
        ready[key_name] = destination
    if not options.keep_intermediates:
        shutil.rmtree(work)
    return ready, "processed"


def _process_pairs(tasks, target, unwrap_options, options):
    """Run `_process_pair` over all pairs, in `options.n_workers` spawned processes.

    Results keep the input order, so reference selection and inversion see the same job
    list whatever the worker count. Spawn (not fork) keeps GDAL handles out of the
    children.
    """
    n = len(tasks)
    if options.n_workers <= 1 or n <= 1:
        jobs = []
        for i, (name, inputs, masks, folder) in enumerate(tasks, 1):
            ready, status = _process_pair(
                inputs, masks, folder, target, unwrap_options, options
            )
            logger.info("[%d/%d] %s: %s", i, n, name, status)
            jobs.append(ready)
        return jobs
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = [None] * n
    with ProcessPoolExecutor(
        min(options.n_workers, n), mp_context=mp.get_context("spawn")
    ) as ex:
        futures = {
            ex.submit(
                _process_pair,
                inputs,
                masks,
                folder,
                target,
                unwrap_options,
                options,
            ): k
            for k, (name, inputs, masks, folder) in enumerate(tasks)
        }
        for done, fut in enumerate(as_completed(futures), 1):
            k = futures[fut]
            jobs[k], status = fut.result()
            logger.info("[%d/%d] %s: %s", done, n, tasks[k][0], status)
    return jobs


def _unwrap_pair(prepared, folder, unwrap_options, options):
    from dolphin.unwrap import unwrap

    method = getattr(
        unwrap_options.unwrap_method, "value", unwrap_options.unwrap_method
    )
    if method not in ("snaphu", "whirlwind"):
        raise ValueError(f"main_diff supports snaphu or whirlwind; received {method!r}")
    result = dict(prepared)
    for band in ("A", "D"):
        published = folder.parent / f"{band}.unw.tif"
        published_cc = folder.parent / f"{band}.unw.conncomp.tif"
        if options.resume and published.is_file() and published_cc.is_file():
            # Per-band resume: a band already unwrapped for this pair is reused (the
            # pair-level resume above needs all three published files; this lets a
            # D-only rerun keep A).
            import shutil

            for src in (published, published_cc):
                shutil.copy2(src, folder / src.name)
            result[f"{band}_unw"] = folder / published.name
            result[f"{band}_cc"] = folder / published_cc.name
            continue
        cfg = unwrap_options.model_copy(deep=True)
        if method == "whirlwind":
            # Use the existing GSLC/quality-aware Dolphin interpolation.
            # Avoid a second interpolation inside Whirlwind.
            cfg.whirlwind_options.interpolate = False
        cfg.run_unwrap = True
        cfg.run_interpolation = True
        cfg.zero_where_masked = False
        cfg.preprocess_options.interpolation_cor_threshold = (
            options.correlation_threshold
            if band == "A"
            else options.correlation_threshold**2
        )
        cfg.preprocess_options.interpolation_similarity_threshold = (
            options.similarity_threshold
        )
        if band == "D" and options.d_looks > 1:
            from .dlooks import unwrap_d_coarse

            unw, cc = unwrap_d_coarse(prepared, folder, cfg, options, unwrap)
            result[f"{band}_unw"], result[f"{band}_cc"] = Path(unw), Path(cc)
            continue
        unw, cc = unwrap(
            ifg_filename=prepared[f"{band}_ifg"],
            corr_filename=prepared[f"{band}_cor"],
            unw_filename=folder / f"{band}.unw.tif",
            nlooks=options.nlooks,
            mask_filename=prepared["inside"],
            similarity_filename=prepared[f"{band}_sim"],
            unwrap_options=cfg,
            unw_nodata=np.nan,
            ccl_nodata=0,
            scratchdir=folder / f"scratch_{band}",
            delete_scratch=not options.keep_intermediates,
        )
        result[f"{band}_unw"], result[f"{band}_cc"] = Path(unw), Path(cc)
    if not options.keep_intermediates:
        for name, path in prepared.items():
            if name != "donor":
                path.unlink()
                result.pop(name)
    return result


def _sample(path, reference):
    with rasterio.open(path) as ds:
        return float(read_window(ds, Window(reference[1], reference[0], 1, 1))[0, 0])


def select_common_reference(
    jobs: list[dict[str, Path]],
    grid: Grid,
    origin: tuple[int, int],
    output: Path,
    block_size: int,
) -> tuple[int, int]:
    """Choose a common donor after all A/D unwrapping, writing candidates on disk.

    Require original donors with finite A/D phase in every pair.
    Connected-component labels are not used. Avoid a full-frame distance transform.
    """
    with rasterio.open(output, "w", **grid.profile("uint8")) as ds:
        for win in windows(grid, block_size):
            ds.write(
                np.ones((int(win.height), int(win.width)), np.uint8), 1, window=win
            )
    for job in jobs:
        with ExitStack() as stack:
            ds = stack.enter_context(rasterio.open(output, "r+"))
            readers = {
                n: stack.enter_context(rasterio.open(job[n]))
                for n in ("donor", "A_unw", "D_unw")
            }
            for win in windows(grid, block_size):
                good = (ds.read(1, window=win) > 0) & (
                    read_window(readers["donor"], win) == 1
                )
                for band in ("A", "D"):
                    phase = read_window(readers[f"{band}_unw"], win)
                    good &= np.isfinite(phase)
                ds.write(good.astype(np.uint8), 1, window=win)
    best = None
    with rasterio.open(output) as ds:
        for win in windows(grid, block_size):
            rr, cc = np.nonzero(ds.read(1, window=win))
            if not len(rr):
                continue
            rr, cc = rr + int(win.row_off), cc + int(win.col_off)
            distance = (rr - origin[0]) ** 2 + (cc - origin[1]) ** 2
            i = int(np.argmin(distance))
            candidate = int(distance[i]), int(rr[i]), int(cc[i])
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError(
            "No common A/D reference survives donor and finite-phase checks; completed"
            " unwraps are retained"
        )
    return best[1], best[2]


def _invert_window(win, jobs, design, offsets, f_a, f_b):
    """Read one window of every pair and invert it (A, D and the ionosphere series).

    Opens its own readers so it can run in a worker process. GDAL's block cache is
    capped: its default (5 % of RAM per process) times the worker count can exceed the
    machine.
    """
    with ExitStack() as stack:
        stack.enter_context(rasterio.Env(GDAL_CACHEMAX=256))
        readers = [
            {
                n: stack.enter_context(rasterio.open(j[n]))
                for n in ("donor", "A_unw", "D_unw")
            }
            for j in jobs
        ]
        valid = np.stack([read_window(r["donor"], win) == 1 for r in readers])
        phases = {}
        for band in ("A", "D"):
            phases[band] = np.stack(
                [read_window(r[f"{band}_unw"], win) for r in readers]
            )
            phases[band] -= offsets[band][:, None, None]
    a, d = invert_common_phase_block(phases["A"], phases["D"], design, valid)
    iono = estimate_iono_main_diff(f_a, f_b, a, d)
    zero = np.where(np.isfinite(iono).all(axis=0), 0, np.nan)[None].astype(np.float32)
    return (
        win,
        np.concatenate((zero, iono)),
        np.concatenate((zero, d.astype(np.float32))),
    )


def _invert_jobs(
    jobs,
    pairs,
    grid,
    reference,
    f_a,
    f_b,
    outputs,
    output_pairs,
    block_size,
    d_outputs=None,
    n_workers=1,
):
    """Invert the A and D pair networks and write the per-output-pair ionosphere.

    The ionosphere is in radians at f_A. When `d_outputs` is given (one path per output
    pair), the per-epoch inverted D is written as well: the band difference per epoch,
    which is what the epoch line test (epoch_qc) runs on in the main_diff path.
    """
    dates = sorted({d for pair in pairs for d in pair})
    idx = {d: i for i, d in enumerate(dates)}
    design = build_design_matrix(pairs, dates)
    if np.linalg.matrix_rank(design) != len(dates) - 1:
        raise ValueError("A/D network is disconnected")
    offsets = {
        b: np.array([_sample(j[f"{b}_unw"], reference) for j in jobs])
        for b in ("A", "D")
    }

    with ExitStack() as stack:
        writers = [
            stack.enter_context(
                rasterio.open(p, "w", **grid.profile("float32", np.nan))
            )
            for p in outputs
        ]
        for dst in writers:
            dst.set_band_unit(1, "radians")
            dst.update_tags(
                phase_convention="dolphin_ifg_ref_times_conj_sec",
                ionosphere_method="main_diff",
            )
        d_writers = []
        if d_outputs is not None:
            d_writers = [
                stack.enter_context(
                    rasterio.open(p, "w", **grid.profile("float32", np.nan))
                )
                for p in d_outputs
            ]
            for dst in d_writers:
                dst.set_band_unit(1, "radians")
                dst.update_tags(
                    phase_convention="dolphin_ifg_ref_times_conj_sec",
                    content="per-epoch A minus B phase",
                )

        def write(win, series, series_d):
            for dst, (ref, sec) in zip(writers, output_pairs, strict=True):
                dst.write(series[idx[sec]] - series[idx[ref]], 1, window=win)
            if d_writers:
                for dst, (ref, sec) in zip(d_writers, output_pairs, strict=True):
                    dst.write(series_d[idx[sec]] - series_d[idx[ref]], 1, window=win)

        wins = list(windows(grid, block_size))
        if n_workers <= 1:
            for win in wins:
                write(*_invert_window(win, jobs, design, offsets, f_a, f_b))
        else:
            # Windows are independent: solve them in spawned workers, write in this
            # process (one writer per raster). Output is identical to the sequential
            # path.
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(
                n_workers, mp_context=mp.get_context("spawn")
            ) as ex:
                futures = [
                    ex.submit(_invert_window, w, jobs, design, offsets, f_a, f_b)
                    for w in wins
                ]
                for fut in as_completed(futures):
                    write(*fut.result())


def _export_correction(source, match, output, f_a, reference_a, block_size):
    # Same matrix/sign as Dolphin inversion; convert raw phase by -lambda/(4*pi)
    # exactly once. The A-B phase has no independent wavelength.
    target = Grid.from_file(match)
    factor = -299792458.0 / f_a / (4 * np.pi)
    with (
        rasterio.open(source) as src,
        WarpedVRT(
            src,
            crs=target.crs,
            transform=target.transform,
            width=target.width,
            height=target.height,
            resampling=Resampling.bilinear,
            nodata=np.nan,
            dtype="float32",
        ) as vrt,
    ):
        offset = float(
            read_window(vrt, Window(reference_a[1], reference_a[0], 1, 1))[0, 0]
        )
        if not np.isfinite(offset):
            raise ValueError(
                "Product reference has no supported ionosphere estimate. Choose a"
                " nominal A reference inside common support; unwrap cache is retained."
                " No neighboring-reference fallback is applied."
            )
        with rasterio.open(output, "w", **target.profile("float32", np.nan)) as dst:
            dst.set_band_unit(1, "meters")
            dst.update_tags(
                ionosphere_method="main_diff",
                reference_row=reference_a[0],
                reference_col=reference_a[1],
            )
            for win in windows(target, block_size):
                dst.write(
                    ((read_window(vrt, win) - offset) * factor).astype(np.float32),
                    1,
                    window=win,
                )


def _fill_and_export(
    name,
    src,
    match,
    output_path,
    timeseries_dir,
    pair_masks,
    target,
    fill_settings,
    f_a,
    reference_a,
    options,
    near_band_path=None,
    noise_path=None,
):
    """Fill one output pair's raw ionosphere and export it as the product correction.

    Independent per output, so it can run in a worker process; `pair_masks` holds the
    four (date, band) inside-mask paths; the robust smoothing reads the near-range band
    and the empirical noise from `near_band_path` and `noise_path`. GDAL's block cache
    is capped so several workers fit in memory.
    """
    with rasterio.Env(GDAL_CACHEMAX=256):
        filled_path = timeseries_dir / f"{name}_iono_B_filled.rad.tif"
        if options.resume and filled_path.is_file():
            logger.info("%s: reusing filled ionosphere", name)
        else:
            with rasterio.open(src) as ds:
                iono = ds.read(1, masked=True).astype(np.float32).filled(np.nan)
            footprint = np.ones(iono.shape, dtype=bool)
            for mask_path in pair_masks:
                with rasterio.open(mask_path) as mask_ds:
                    with WarpedVRT(
                        mask_ds,
                        crs=target.crs,
                        transform=target.transform,
                        width=target.width,
                        height=target.height,
                        resampling=Resampling.average,
                        src_nodata=255,
                        nodata=255,
                        dtype="float32",
                    ) as mask_vrt:
                        inside = mask_vrt.read(1, masked=True).filled(0)
                footprint &= inside > 0
            trusted = footprint & np.isfinite(iono)
            if not trusted.any():
                raise ValueError(f"{name}: no valid ionosphere samples for filling")

            # Existing helper erodes invalid stripes along columns. Check donors
            # before calling it: a Gaussian fill with no donors is undefined.
            from scipy.ndimage import maximum_filter1d

            radius = fill_settings["mask_erosion_px"]
            eroded_donors = footprint & ~maximum_filter1d(
                (~trusted).astype(np.uint8),
                size=2 * radius + 1,
                axis=1,
            ).astype(bool)
            if not eroded_donors.any():
                raise ValueError(f"{name}: no donors remain after mask erosion")

            if options.smoothing == "robust":
                from .smoothing import noise_weights, smooth_screen

                noise = noise_w = None
                if noise_path is not None:
                    with rasterio.open(noise_path) as ds:
                        noise = ds.read(1)
                    noise_w = noise_weights(noise, np.isfinite(noise))
                iono_filled, quality_mask, _ = smooth_screen(
                    iono,
                    trusted,
                    footprint,
                    noise_w,
                    options.smooth_sigma_px,
                    min_support=options.smooth_min_support,
                    fallback_sigma_px=options.smooth_fallback_sigma_px,
                    clip_max_dev=(2 * np.pi * options.background_clip_cycles) or None,
                    clip_block_px=options.background_block_px,
                    noise=noise,
                    exclude_noise_factor=options.exclude_noise_factor or None,
                    exclude_mask=(
                        np.load(near_band_path) if near_band_path is not None else None
                    ),
                )
            else:
                iono_filled, quality_mask = apply_similarity_mask_and_fill(
                    iono=iono,
                    sim_mask=trusted,
                    existing_mask=footprint,
                    **fill_settings,
                )
            logger.info(
                "%s: ionosphere valid before=%d, after=%d",
                name,
                np.count_nonzero(trusted),
                np.count_nonzero(np.isfinite(iono_filled)),
            )
            temporary_filled = filled_path.with_name(f"{filled_path.stem}.partial.tif")
            with rasterio.open(
                temporary_filled,
                "w",
                **target.profile("float32", np.nan),
            ) as dst:
                dst.write(iono_filled.astype(np.float32), 1)
                dst.set_band_unit(1, "radians")
                dst.update_tags(
                    ionosphere_method="main_diff",
                    phase_convention="dolphin_ifg_ref_times_conj_sec",
                    smoothing=options.smoothing,
                )
            temporary_filled.replace(filled_path)

        temporary_output = output_path.with_name(f"{output_path.stem}.partial.tif")
        _export_correction(
            filled_path,
            Path(match),
            temporary_output,
            f_a,
            reference_a,
            options.block_size,
        )
        temporary_output.replace(output_path)


def run_main_diff_estimation(
    *,
    band_a: BandInputs,
    band_b: BandInputs,
    gslc_by_date: dict,
    output_timeseries_paths: Sequence[Path],
    reference_a: tuple[int, int],
    f_a: float,
    f_b: float,
    out_dir: Path,
    unwrap_options,
    options: IonosphereOptions,
) -> list[Path]:
    """Prepare, unwrap, reference, invert, and export A/(A-B) corrections.

    Parameters
    ----------
    band_a, band_b : BandInputs
        Full matching stitched IFG networks on their respective native grids.
    gslc_by_date : dict
        Acquisition datetime (midnight) to the actual local raw GSLC used in PL.
    output_timeseries_paths : sequence[Path]
        Nominal products whose date pairs, output order and grids must be matched.
    reference_a : tuple[int, int]
        Existing nominal A spatial reference on its product grid.
    f_a, f_b : float
        Constant center frequencies in Hz; caller checks acquisition consistency.
    out_dir : Path
        Stable mask/pair/inversion cache directory.
    unwrap_options : dolphin.workflows.config.UnwrapOptions
        SNAPHU settings, with interpolation enabled and explicit donor thresholds.
    options : IonosphereOptions
        DISP-specific donor, memory and restart options.

    Returns
    -------
    list[Path]
        Correction rasters in meters, ordered like output_timeseries_paths.
        Original nominal data and their reference metadata are not modified.

    """
    import fcntl

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "main_diff is already running in this directory"
            ) from exc
        return _run_locked(
            band_a,
            band_b,
            gslc_by_date,
            output_timeseries_paths,
            reference_a,
            f_a,
            f_b,
            out_dir,
            unwrap_options,
            options,
        )


def _run_locked(
    band_a,
    band_b,
    gslc_by_date,
    products,
    reference_a,
    f_a,
    f_b,
    out_dir,
    unwrap_options,
    options,
):
    if not products:
        return []
    final_paths = [
        out_dir
        / "corrections"
        / ("_".join(d.strftime("%Y%m%d") for d in _date_pair(Path(p))) + "_iono.tif")
        for p in products
    ]
    if len(set(final_paths)) != len(final_paths):
        raise ValueError("Duplicate output date pairs")
    if options.resume and all(p.is_file() for p in final_paths):
        logger.info("Reusing all existing ionosphere corrections")
        return final_paths
    estimate_iono_main_diff(f_a, f_b, np.zeros(1), np.zeros(1))
    maps = {
        "ifg_A": _index(band_a.interferograms),
        "cor_A": _index(band_a.correlations),
        "ifg_B": _index(band_b.interferograms),
        "cor_B": _index(band_b.correlations),
    }
    pairs = sorted(maps["ifg_A"])
    if not pairs or any(set(m) != set(pairs) for m in maps.values()):
        raise ValueError("A/B IFG and correlation networks must match exactly")
    output_pairs = [_date_pair(Path(p)) for p in products]
    dates = sorted({d for pair in pairs for d in pair})
    if any(d not in dates for pair in output_pairs for d in pair):
        raise ValueError("Output requests an epoch absent from the common network")
    if missing := set(dates) - set(gslc_by_date):
        raise ValueError(f"Missing raw GSLC masks for acquisitions: {missing}")
    grids = {b: Grid.from_file(maps[f"ifg_{b}"][pairs[0]]) for b in ("A", "B")}
    target = grids["B"]
    check_reduction(grids["A"], target)
    for b, band in (("A", band_a), ("B", band_b)):
        for p in (*band.interferograms, *band.correlations, band.similarity):
            _same_grid(p, grids[b])
    pg = Grid.from_file(Path(products[0]))
    if pg.crs != target.crs:
        raise ValueError("Product CRS differs from B CRS")
    for p in products:
        _same_grid(p, pg)
    r, c = reference_a
    if not (0 <= r < pg.height and 0 <= c < pg.width):
        raise ValueError("Nominal reference outside A product grid")
    x, y = pg.transform * (c + 0.5, r + 0.5)
    cb, rb = (~target.transform) * (x, y)
    origin = int(np.floor(rb)), int(np.floor(cb))
    if not (0 <= origin[0] < target.height and 0 <= origin[1] < target.width):
        raise ValueError("Nominal A reference is outside the B grid")
    raw_paths = [
        out_dir
        / "timeseries"
        / ("_".join(d.strftime("%Y%m%d") for d in pair) + "_iono_B.rad.tif")
        for pair in output_pairs
    ]
    inversion_ready = options.resume and all(p.is_file() for p in raw_paths)
    need_masks = not inversion_ready or any(
        not final.is_file()
        and not (
            out_dir
            / "timeseries"
            / ("_".join(d.strftime("%Y%m%d") for d in pair) + "_iono_B_filled.rad.tif")
        ).is_file()
        for pair, final in zip(output_pairs, final_paths, strict=True)
    )
    cached_masks = (
        {
            (d, b): prepare_gslc_mask_cache(
                Path(gslc_by_date[d]),
                b,
                grids[b],
                out_dir / "masks",
                options.mask_reduction,
                options.block_size,
                options.mask_work_mb,
                resume=options.resume,
            )
            for d in dates
            for b in ("A", "B")
        }
        if need_masks
        else {}
    )
    tasks = []
    for pair in [] if inversion_ready else pairs:
        name = "_".join(d.strftime("%Y%m%d") for d in pair)
        inputs = {n: m[pair] for n, m in maps.items()}
        inputs.update(sim_A=Path(band_a.similarity), sim_B=Path(band_b.similarity))
        masks = {
            f"{b}_{role}_{kind}": cached_masks[d, b][kind]
            for b in ("A", "B")
            for role, d in zip(("ref", "sec"), pair, strict=True)
            for kind in ("valid", "inside")
        }
        tasks.append((name, inputs, masks, out_dir / "pairs" / name))
    jobs = _process_pairs(tasks, target, unwrap_options, options)

    fill_settings = {
        "fill_method": "gaussian",
        "smooth_sigma": 50.0,
        "mask_erosion_px": 3,
    }
    timeseries_dir = out_dir / "timeseries"
    correction_dir = out_dir / "corrections"
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    correction_dir.mkdir(parents=True, exist_ok=True)
    names = ["_".join(d.strftime("%Y%m%d") for d in pair) for pair in output_pairs]
    intermediate = [timeseries_dir / f"{name}_iono_B.rad.tif" for name in names]

    if options.resume and all(p.is_file() for p in intermediate):
        logger.info("Reusing existing ionosphere inversion results")
    else:
        reference = select_common_reference(
            jobs,
            target,
            origin,
            timeseries_dir / "common_candidates.tif",
            options.block_size,
        )
        atomic_json(
            timeseries_dir / "reference.json",
            {"reference_b": list(reference), "reference_a": list(reference_a)},
        )
        temporary = [p.with_name(f"{p.stem}.partial.tif") for p in intermediate]
        _invert_jobs(
            jobs,
            pairs,
            target,
            reference,
            f_a,
            f_b,
            temporary,
            output_pairs,
            options.block_size,
            n_workers=options.n_workers,
        )
        for source, destination in zip(temporary, intermediate, strict=True):
            source.replace(destination)

    near_band_path = None
    if options.smoothing == "robust" and options.near_range_exclude_km > 0:
        # the incidence raster (static layers) is optional input; without it the band
        # cannot be located
        inc_path = getattr(options, "incidence_path", None) or (
            out_dir / "incidence.tif"
        )
        if Path(inc_path).is_file():
            from .smoothing import near_range_band

            inc = np.full((target.height, target.width), np.nan, "float32")
            with rasterio.open(inc_path) as src:
                reproject(
                    src.read(1).astype("float32"),
                    inc,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=target.transform,
                    dst_crs=target.crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.average,
                )
            foot = np.zeros((target.height, target.width), bool)
            for p in intermediate:
                with rasterio.open(p) as ds:
                    foot |= np.isfinite(
                        ds.read(1, masked=True).astype(np.float32).filled(np.nan)
                    )
            near_band = near_range_band(
                foot, inc, abs(target.transform.a), options.near_range_exclude_km
            )
            logger.info(
                "near-range band of %.1f km excluded from the screen: %.1f %% of the"
                " footprint",
                options.near_range_exclude_km,
                100 * near_band[foot].mean(),
            )
            near_band_path = correction_dir / "near_range_band.npy"
            np.save(near_band_path, near_band)
        else:
            logger.warning(
                "near_range_exclude_km set but no incidence raster at %s; band not"
                " excluded",
                inc_path,
            )
    noise_path = None
    if options.smoothing == "robust" and options.noise_weights:
        from .smoothing import empirical_noise

        noise_path = correction_dir / "ionosphere_noise.rad.tif"
        if not (options.resume and noise_path.is_file()):

            def _epochs():
                for p in intermediate:
                    with rasterio.open(p) as ds:
                        yield ds.read(1, masked=True).astype(np.float32).filled(np.nan)

            any_ok = np.zeros((target.height, target.width), bool)
            for p in intermediate:
                with rasterio.open(p) as ds:
                    any_ok |= np.isfinite(
                        ds.read(1, masked=True).astype(np.float32).filled(np.nan)
                    )
            noise = empirical_noise(_epochs(), any_ok, options.smooth_fallback_sigma_px)
            with rasterio.open(
                noise_path, "w", **target.profile("float32", np.nan)
            ) as dst:
                dst.write(noise, 1)
                dst.set_band_unit(1, "radians")
                dst.update_tags(
                    description="RMS over epochs of the screen minus its 10 km smooth"
                )
    outputs = []
    tasks = []
    for i, (src, match, name) in enumerate(
        zip(intermediate, products, names, strict=True)
    ):
        output_path = correction_dir / f"{name}_iono.tif"
        outputs.append(output_path)
        if options.resume and output_path.is_file():
            logger.info("%s: reusing correction", name)
            continue
        pair_masks = (
            [
                cached_masks[date, band]["inside"]
                for date in output_pairs[i]
                for band in ("A", "B")
            ]
            if cached_masks
            else []
        )
        tasks.append(
            (
                name,
                src,
                match,
                output_path,
                timeseries_dir,
                pair_masks,
                target,
                fill_settings,
                f_a,
                reference_a,
                options,
                near_band_path,
                noise_path,
            )
        )
    if options.n_workers <= 1 or len(tasks) <= 1:
        for t in tasks:
            _fill_and_export(*t)
    else:
        # Outputs are independent: fill and export them in spawned workers.
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            min(options.n_workers, len(tasks)), mp_context=mp.get_context("spawn")
        ) as ex:
            for _ in ex.map(_fill_and_export, *zip(*tasks)):
                pass
    return outputs
