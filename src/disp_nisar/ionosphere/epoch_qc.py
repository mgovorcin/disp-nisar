"""Per-epoch quality control for the two-band ionosphere and the displacement.

Two straight-line tests, calibrated on the 14 California frames and Alaska 106_A_037 of
the ab36 campaign (track-042 ionosphere study, Oct 2026):

1. **Band-difference line.** r = phi_B - (f_B/f_A) phi_A, in cycles. Non-dispersive
   terms cancel in r and the ionosphere is smooth at kilometre scale, so after a robust
   plane the along-track profile of r has no jump. A GSLC/burst artefact that steps the
   two bands by different amounts leaves a jump at constant azimuth. An epoch is flagged
   when |jump| > `step_min_cycles` (0.02) and |jump| > `ratio_min` (50) × the robust
   noise of the jump series. Real artefacts scored 65-462; smooth ionosphere, island
   offsets and region-wide unwrapping disagreements scored at most 38. Size is not a
   general test; sharpness is.
2. **Displacement line.** The (ionosphere-corrected) displacement through the same test,
   in mm. It catches an acquisition corrupted the same way in both bands, which cancels
   in r. An epoch is flagged when the ratio is at least `ratio_min` (15), |step| is at
   least `step_min_mm` (45) and the step position is not shared by at least
   `shared_fraction` (1/3) of the epochs: a position that recurs on most epochs is a
   geographic feature (an island, a frame-internal boundary), not a bad acquisition.

Flagged epochs are excluded from the ionosphere estimate and the velocity; they are
never interpolated.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "EpochFlag",
    "StepResult",
    "along_track_step",
    "along_track_unit",
    "band_difference_cycles",
    "coherence_streak_ratio",
    "flag_band_difference",
    "flag_d_series",
    "flag_displacement",
    "flag_screens",
    "flag_streak_pairs",
    "read_bad_epochs",
    "streak_epochs",
    "write_bad_epochs",
]


@dataclass(frozen=True)
class StepResult:
    """Largest along-track jump of a field.

    Holds the jump (field units), its position and the robust noise of the jump series.
    The position is absolute along-track km, (x·E + y·N)/1000 in the grid's projected
    metres, so it can be compared between epochs of the same frame.
    """

    step: float
    position_km: float
    noise: float

    @property
    def ratio(self) -> float:
        """Sharpness of the jump: |step| over the robust noise (NaN without noise)."""
        if not np.isfinite(self.noise) or self.noise <= 0:
            return np.nan
        return abs(self.step) / self.noise


@dataclass(frozen=True)
class EpochFlag:
    """One flagged epoch with the reason code and the statistic that triggered it."""

    epoch: str
    reason: str
    statistic: float
    position_km: float


def along_track_unit(los_east: np.ndarray, los_north: np.ndarray) -> np.ndarray:
    """Horizontal along-track direction (E, N) from the mean horizontal LOS unit vector.

    The along-track direction is perpendicular to the horizontal look direction for a
    side-looking SAR; the sign (ascending vs descending) does not matter for a step
    test.
    """
    e_arr = np.asarray(los_east, dtype="float64")
    n_arr = np.asarray(los_north, dtype="float64")
    valid = (
        np.isfinite(e_arr) & np.isfinite(n_arr) & ~((e_arr == 0) & (n_arr == 0))
    )  # (0, 0) = nodata
    if not valid.any():
        raise ValueError("LOS unit vectors give no horizontal look direction")
    e, n = float(e_arr[valid].mean()), float(n_arr[valid].mean())
    v = np.array([-n, e])
    norm = np.hypot(*v)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("LOS unit vectors give no horizontal look direction")
    return v / norm


def band_difference_cycles(
    phi_a: np.ndarray, phi_b: np.ndarray, f_a: float, f_b: float
) -> np.ndarray:
    """Return r = phi_B - (f_B/f_A) phi_A in cycles.

    r is zero for any non-dispersive signal and -0.086 x the ionosphere.
    """
    return (phi_b - (f_b / f_a) * phi_a) / (2 * np.pi)


def along_track_step(
    values: np.ndarray,
    ok: np.ndarray,
    transform,
    along: Sequence[float],
    sub: int = 3,
    bin_km: float = 1.0,
    window: int = 5,
    min_per_bin: int = 100,
    min_points: int = 5000,
) -> StepResult:
    """Largest jump in the along-track profile of `values`, after a robust plane.

    Parameters
    ----------
    values, ok : arrays
        Field to test and its validity mask (same shape).
    transform : affine.Affine
        Pixel-edge geotransform of `values` (north-up, projected metres).
    along : (E, N)
        Horizontal along-track unit vector (`along_track_unit`).
    sub : int
        Pixel subsampling of the valid points.
    bin_km, window : float, int
        Profile bin length and the half-window (in bins) of the two medians whose
        difference is the jump.
    min_per_bin, min_points : int
        A bin with fewer points is left empty; with fewer valid points overall the
        result is all-NaN.

    """
    nan = StepResult(np.nan, np.nan, np.nan)
    yy, xx = np.nonzero(ok)
    yy, xx = yy[::sub], xx[::sub]
    if len(yy) < min_points:
        return nan
    x = transform.c + transform.a * (xx + 0.5)
    y = transform.f + transform.e * (yy + 0.5)
    v = values[yy, xx].astype("float64")
    design = np.c_[np.ones_like(x), (x - x.mean()) / 1e5, (y - y.mean()) / 1e5]
    keep = np.isfinite(v)
    if keep.sum() < min_points:
        return nan
    res = v.copy()
    for _ in range(2):
        coef = np.linalg.lstsq(design[keep], v[keep], rcond=None)[0]
        res = v - design @ coef
        med = np.nanmedian(res[keep])
        mad = 1.4826 * np.nanmedian(np.abs(res[keep] - med))
        keep = np.isfinite(res) & (np.abs(res - med) < 4 * mad + 1e-9)
    s = (x * along[0] + y * along[1]) / (1000.0 * bin_km)
    s0 = s.min()
    b = np.floor(s - s0).astype(int)
    nb = int(b.max()) + 1
    order = np.argsort(b[keep], kind="stable")
    bs, rs = b[keep][order], res[keep][order]
    starts = np.searchsorted(bs, np.arange(nb + 1))
    profile = np.full(nb, np.nan)
    for i in range(nb):
        if starts[i + 1] - starts[i] >= min_per_bin:
            profile[i] = np.median(rs[starts[i] : starts[i + 1]])
    jumps = np.full(nb, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(window, nb - window):
            jumps[i] = np.nanmedian(profile[i : i + window]) - np.nanmedian(
                profile[i - window : i]
            )
    if not np.isfinite(jumps).any():
        return nan
    i = int(np.nanargmax(np.abs(jumps)))
    noise = 1.4826 * np.nanmedian(np.abs(jumps - np.nanmedian(jumps)))
    return StepResult(float(jumps[i]), float((i + s0) * bin_km), float(noise))


def flag_band_difference(
    result: StepResult, step_min_cycles: float = 0.02, ratio_min: float = 50.0
) -> bool:
    """Rule 1: a sharp jump in the band difference, in cycles."""
    return bool(
        np.isfinite(result.step)
        and abs(result.step) > step_min_cycles
        and result.ratio > ratio_min
    )


def _unshared(
    results: Sequence[StepResult],
    step_min: float,
    ratio_min: float,
    shared_fraction: float,
    position_tolerance_km: float,
) -> list[bool]:
    """Flag steps that are sharp (ratio), large (|step|) and at an unshared position.

    Unshared means fewer than `shared_fraction` of the epochs put their step there.

    A position is 'shared' when at least `shared_fraction` of the epochs put their
    largest step within `position_tolerance_km` of it: a recurring position is a
    frame-edge or static term, not a bad acquisition.
    """
    pos = np.array([r.position_km for r in results], dtype="float64")
    n = len(results)
    flags = []
    for r in results:
        if not (
            np.isfinite(r.step) and abs(r.step) >= step_min and r.ratio >= ratio_min
        ):
            flags.append(False)
            continue
        shared = np.sum(np.abs(pos - r.position_km) <= position_tolerance_km) / max(
            n, 1
        )
        flags.append(bool(shared < shared_fraction))
    return flags


def flag_displacement(
    results: Sequence[StepResult],
    step_min_mm: float = 45.0,
    ratio_min: float = 15.0,
    shared_fraction: float = 1 / 3,
    position_tolerance_km: float = 2.0,
) -> list[bool]:
    """Apply rule 2 over all epochs of a frame (results in mm).

    A flag needs a sharp, large step at a position the other epochs do not share.
    """
    return _unshared(
        results, step_min_mm, ratio_min, shared_fraction, position_tolerance_km
    )


def flag_d_series(
    d_paths: Sequence,
    los_east: np.ndarray,
    los_north: np.ndarray,
    ratio_min: float = 20.0,
    step_min_cycles: float = 0.02,
    shared_fraction: float = 1 / 3,
    position_tolerance_km: float = 5.0,
) -> tuple[list[StepResult], list[EpochFlag]]:
    """Apply rule 1 on the main_diff path, to the per-epoch inverted D (A·conj(B)).

    `d_paths` are the per-epoch D rasters written by the inversion's ``d_outputs``
    (radians of D, unfilled, same grid; the stem's second date is the epoch). D is the
    band difference itself, so the r thresholds of rule 1 apply in D cycles without the
    screen's 11.7x scaling. The inversion spreads a one-date artefact over the network,
    which lowers its sharpness: on the ab36 frames the known line artefacts reach ratio
    27-35 (not the ~300 of a band time series), and frame-edge terms reach the same
    ratio at a recurring position on many epochs. Hence ratio >= 20, |step| >= 0.02
    cycle and an unshared position (3 of 3 artefacts caught, 0 false flags over 61
    epochs on 114_D_069, 085_D_070, 034_A_021).
    """
    import rasterio

    along = along_track_unit(los_east, los_north)
    results, epochs = [], []
    for p in d_paths:
        p = Path(p)
        with rasterio.open(p) as src:
            arr = src.read(1, masked=True).astype("float64").filled(np.nan)
            tr = src.transform
        cycles = arr / (2 * np.pi)
        results.append(along_track_step(cycles, np.isfinite(cycles), tr, along))
        epochs.append(p.stem.split("_")[1] if "_" in p.stem else p.stem)
    keep = _unshared(
        results, step_min_cycles, ratio_min, shared_fraction, position_tolerance_km
    )
    flags = [
        EpochFlag(e, "d_line", float(r.ratio), r.position_km)
        for e, r, k in zip(epochs, results, keep)
        if k
    ]
    return results, flags


def coherence_streak_ratio(
    cor: np.ndarray,
    pixel_m: float,
    along: np.ndarray,
    block_m: float = 210.0,
    highpass_m: float = 20000.0,
    lam_min_m: float = 1000.0,
    lam_max_m: float = 20000.0,
    half_deg: float = 10.0,
) -> float:
    """Anisotropy of one pair's coherence loss: along / across-track power, 1-20 km.

    Bands of coherence loss at constant azimuth (a GSLC line artefact, or
    misregistration from a steep along-track ionospheric gradient) put their power on
    the along-track wavenumber axis. `cor` on a north-up grid (rows southward), `along`
    the (east, north) unit vector. Isotropic texture gives ~1, but the level is
    frame-dependent (medians 0.2-4.5 on the ab36 frames); the six 114_D_069 pairs
    containing the 2026-02-22 line artefact give 9.6-11.1 against a frame median of
    0.44. Returns NaN when the central window is less than 30 % valid.
    """
    from scipy.ndimage import gaussian_filter

    c = np.where(np.isfinite(cor) & (cor > 0), cor, np.nan).astype("float64")
    k = max(1, int(round(block_m / pixel_m)))
    h, w = (c.shape[0] // k) * k, (c.shape[1] // k) * k
    v = c[:h, :w].reshape(h // k, k, w // k, k)
    ok = np.isfinite(v)
    with np.errstate(invalid="ignore"):
        b = np.where(ok, v, 0).sum(axis=(1, 3)) / np.maximum(ok.sum(axis=(1, 3)), 1)
    b = np.where(ok.mean(axis=(1, 3)) >= 0.5, 1.0 - b, np.nan)
    px = pixel_m * k
    okb = np.isfinite(b)
    sig = highpass_m / px
    num = gaussian_filter(np.where(okb, b, 0.0), sig)
    den = gaussian_filter(okb.astype(float), sig)
    x = np.where(okb & (den > 0.1), b - num / np.maximum(den, 1e-9), np.nan)
    n = 2 ** int(np.log2(min(x.shape)))
    r0, c0 = (x.shape[0] - n) // 2, (x.shape[1] - n) // 2
    win = x[r0 : r0 + n, c0 : c0 + n]
    okw = np.isfinite(win)
    if okw.mean() < 0.3:
        return float("nan")
    win = (
        np.where(okw, win - np.nanmean(win), 0.0)
        * np.hanning(n)[:, None]
        * np.hanning(n)[None, :]
    )
    P = np.abs(np.fft.fft2(win)) ** 2
    f = np.fft.fftfreq(n, px)
    kx = np.broadcast_to(f[None, :], (n, n))
    kn = np.broadcast_to(-f[:, None], (n, n))
    kr = np.hypot(kx, kn)
    band = (kr >= 1.0 / lam_max_m) & (kr <= 1.0 / lam_min_m)
    cosang = np.abs(kx * along[0] + kn * along[1]) / np.maximum(kr, 1e-12)
    pa = P[band & (cosang >= np.cos(np.radians(half_deg)))].mean()
    pc = P[band & (cosang <= np.sin(np.radians(half_deg)))].mean()
    return float(pa / pc)


def flag_streak_pairs(
    ratios: dict[str, float], factor: float = 3.0, floor: float = 2.0
) -> list[str]:
    """Pairs whose coherence-loss anisotropy is anomalous for their frame (sorted).

    A pair is flagged when its ratio exceeds `factor` x the frame's median and `floor`.

    The ratio's level is a property of the frame (terrain, coast and look geometry give
    medians from 0.2 to 4.5 on the ab36 frames), so a fixed threshold flags whole
    frames; an artefact pair stands out from its own frame. An epoch shared by most
    flagged pairs is the artefact date (see `streak_epochs`).
    """
    vals = np.array([r for r in ratios.values() if np.isfinite(r)])
    if vals.size == 0:
        return []
    thr = max(factor * float(np.median(vals)), floor)
    return sorted(p for p, r in ratios.items() if np.isfinite(r) and r > thr)


def streak_epochs(
    flagged_pairs: Sequence[str], all_pairs: Sequence[str], min_fraction: float = 0.5
) -> list[str]:
    """Epochs shared by the flagged pairs.

    An epoch is returned when at least `min_fraction` of the pairs containing it (and at
    least two) were flagged.
    """
    from collections import Counter

    total = Counter(d for p in all_pairs for d in p.split("_")[:2])
    hit = Counter(d for p in flagged_pairs for d in p.split("_")[:2])
    return sorted(d for d, n in hit.items() if n >= min_fraction * total[d] and n >= 2)


def write_bad_epochs(path: str | Path, flags: Iterable[EpochFlag]) -> Path:
    """Write ``epoch,reason,statistic,position_km``, one row per flag.

    An epoch may have several rows.
    """
    path = Path(path)
    rows = sorted(flags, key=lambda f: (f.epoch, f.reason))
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "reason", "statistic", "position_km"])
        for f in rows:
            w.writerow(
                [f.epoch, f.reason, f"{f.statistic:.4g}", f"{f.position_km:.1f}"]
            )
    return path


def read_bad_epochs(path: str | Path) -> set[str]:
    """Epochs listed in a bad-epochs CSV (or a bare one-epoch-per-line text file)."""
    path = Path(path)
    if not path.exists():
        return set()
    text = path.read_text().strip()
    if not text:
        return set()
    epochs = set()
    for line in text.splitlines():
        first = line.split(",")[0].strip()
        if first and first != "epoch":
            epochs.add(first)
    return epochs


def flag_screens(
    screen_paths: Sequence,
    los_east: np.ndarray,
    los_north: np.ndarray,
    ratio_min: float = 50.0,
    step_min_cycles: float = 0.25,
) -> tuple[list[StepResult], list[EpochFlag]]:
    """Apply rule 1 to the per-epoch ionospheric screens themselves.

    The screen is the band difference scaled by f_A f_B/(f_B² − f_A²) ≈ 11.7, so a
    straight-line artefact of 0.9 cycle in r appears as ~10 cycles in the screen; the
    sharpness test (ratio) is scale-free and `step_min_cycles` is the r threshold (0.02)
    scaled by the same factor, rounded to 0.25. On the ab36 frames this placement
    catches few artefacts; prefer `flag_d_series`.

    `screen_paths` are rasters of the unfilled per-epoch screen in radians at f_A (same
    grid); the file stem's second date is the epoch. Returns the per-epoch step results
    and the flags.
    """
    import rasterio

    along = along_track_unit(los_east, los_north)
    results, flags = [], []
    for p in screen_paths:
        p = Path(p)
        with rasterio.open(p) as src:
            arr = src.read(1, masked=True).astype("float64").filled(np.nan)
            tr = src.transform
        cycles = arr / (2 * np.pi)
        res = along_track_step(cycles, np.isfinite(cycles), tr, along)
        results.append(res)
        epoch = p.stem.split("_")[1] if "_" in p.stem else p.stem
        if (
            np.isfinite(res.step)
            and abs(res.step) > step_min_cycles
            and res.ratio > ratio_min
        ):
            flags.append(
                EpochFlag(epoch, "screen_line", float(res.ratio), res.position_km)
            )
    return results, flags
