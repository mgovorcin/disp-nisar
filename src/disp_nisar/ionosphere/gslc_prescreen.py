"""Epoch quality control on the GSLCs, before phase linking.

A bad acquisition (a GSLC line artefact, a partly decorrelated frame) contaminates every
interferogram that contains it, so it is cheapest to find it before phase linking. For
each date t this module forms multilooked interferograms of both bands directly from the
GSLC HDF5 files, takes their band difference D = ifg_A · conj(ifg_B) (non-dispersive
phase cancels), and scores two signatures against a reference pair that does not contain
t:

- **Line** (phase, no unwrapping): for every direction, D's phasors are binned along the
  line normal and the phase jumps between adjacent bins are combined over the date's
  pairs so that a one-date artefact adds. Interior date: (j(t-1,t) − j(t,t+1)) / 2
  against the skip pair (t-1,t+1). Newest date (forward mode): (j(t-1,t) + j(t-2,t)) / 2
  against (t-2,t-1). Score = max over bins of (|c| − |j_ref|) / robust noise(c),
  maximised over direction.
- **Decorrelation**: median over pixels of log coh_ref minus the mean log coherence of
  the date's pairs; > 0 when the date's pairs are less coherent than the reference.

A date is flagged when its line score is >= `line_min` *and* >= `line_rel_min` times
the frame's median first-pass line score (real ionosphere raises every date's score, as
on Alaska frames; the relative test needs >= 3 scored dates), or when its
decorrelation is >= `coherence_drop_min`. After a flag, the other dates are re-scored
against their nearest unflagged neighbours, so a bad neighbour neither leaks into a
clean date nor hides an adjacent artefact (re-scoring can need pairs more than three
dates apart).

Thresholds (2026-10-28, seven ab36 frames, 34 scored dates): the three confirmed line
artefacts scored 1.85-2.07x the frame median (one only after its adjacent artefact was
flagged) and one decorrelated 0.48; real Alaska ionosphere scored up to 1.37x; winter
decorrelation on clean dates reached 0.35. A permanent decorrelated area (ocean) raises
every date's line score (034_A_019: ~200-280), so the relative test keeps such frames
unflagged but blind; blanking low-coherence pixels was tried and removes the artefacts'
signal too. Two adjacent dates with an identical step cannot be told apart from a step
on the dates around them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .epoch_qc import EpochFlag

__all__ = [
    "PairMaps",
    "band_difference",
    "coherence_drop",
    "date_line_score",
    "multilook_pair",
    "prescreen_dates",
]

_GRID = "science/LSAR/GSLC/grids/frequency{band}/"


@dataclass(frozen=True)
class PairMaps:
    """Multilooked two-band maps of one date pair on a common coarse grid."""

    d: np.ndarray  # unit phasor of ifg_A · conj(ifg_B), NaN where invalid
    coherence: np.ndarray  # min of the two bands' coherence


def _block_sum(x: np.ndarray, ky: int, kx: int) -> np.ndarray:
    h, w = (x.shape[0] // ky) * ky, (x.shape[1] // kx) * kx
    return x[:h, :w].reshape(h // ky, ky, w // kx, kx).sum(axis=(1, 3))


def _window_sums(args):
    import h5py

    f1, f2, prefix, pol, r0, r1, nc, ky, kx = args
    with (
        h5py.File(f1, "r", locking=False) as h1,
        h5py.File(f2, "r", locking=False) as h2,
    ):
        z1 = h1[prefix + pol][r0:r1, :nc]
        z2 = h2[prefix + pol][r0:r1, :nc]
        m1 = h1[prefix + "mask"][r0:r1, :nc]
        m2 = h2[prefix + "mask"][r0:r1, :nc]
    ok = (m1 > 0) & (m1 < 255) & (m2 > 0) & (m2 < 255) & (z1 != 0) & (z2 != 0)
    z1 = np.where(ok, z1, 0)
    z2 = np.where(ok, z2, 0)
    return (
        r0,
        _block_sum(z1 * np.conj(z2), ky, kx),
        _block_sum(np.abs(z1) ** 2, ky, kx),
        _block_sum(np.abs(z2) ** 2, ky, kx),
        _block_sum(ok.astype(np.float32), ky, kx),
    )


def _multilook_band(f1, f2, band, pol, looks_y, row_stride, window_rows, executor):
    import h5py

    prefix = _GRID.format(band=band)
    with (
        h5py.File(f1, "r", locking=False) as h1,
        h5py.File(f2, "r", locking=False) as h2,
    ):
        dx = float(h1[prefix + "xCoordinateSpacing"][()])
        dy = float(h1[prefix + "yCoordinateSpacing"][()])
        nr, nc = (
            min(a, b) for a, b in zip(h1[prefix + pol].shape, h2[prefix + pol].shape)
        )
    kx = max(1, int(round(looks_y * abs(dy) / abs(dx))))
    rows_out = nr // looks_y
    shape = (rows_out, nc // kx)
    s = np.zeros(shape, np.complex128)
    p1 = np.zeros(shape)
    p2 = np.zeros(shape)
    n = np.zeros(shape)
    step = window_rows - window_rows % looks_y
    starts = list(range(0, rows_out * looks_y, step))[::row_stride]
    jobs = [
        (
            str(f1),
            str(f2),
            prefix,
            pol,
            r0,
            min(r0 + step, rows_out * looks_y),
            nc,
            looks_y,
            kx,
        )
        for r0 in starts
    ]
    results = executor.map(_window_sums, jobs) if executor else map(_window_sums, jobs)
    for r0, s_, a, b, c in results:
        o = r0 // looks_y
        s[o : o + s_.shape[0]] = s_
        p1[o : o + s_.shape[0]] = a
        p2[o : o + s_.shape[0]] = b
        n[o : o + s_.shape[0]] = c
    valid = n >= 0.5 * looks_y * kx
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.abs(s) / np.sqrt(np.maximum(p1 * p2, 1e-30))
        ifg = s / np.maximum(np.abs(s), 1e-30)
    return np.where(valid, ifg, np.nan), np.where(valid, coh, np.nan)


def multilook_pair(
    gslc_ref: str | Path,
    gslc_sec: str | Path,
    looks_y: int = 64,
    pol: str = "HH",
    row_stride: int = 1,
    window_rows: int = 1024,
    executor=None,
) -> PairMaps:
    """Multilooked band difference D and coherence of one pair, read from two GSLCs.

    Both bands are averaged to blocks of `looks_y` rows and the same ground width (A at
    5 m: looks_y × looks_y; B at 40 × 5 m: looks_y × looks_y/8). `row_stride` > 1 reads
    only every n-th window of `window_rows` rows (the line test bins along a tilted
    direction, so a subsample still covers every bin). `executor` (a process pool)
    parallelises the window reads; HDF5 decompression dominates the cost.
    """
    ia, ca = _multilook_band(
        gslc_ref, gslc_sec, "A", pol, looks_y, row_stride, window_rows, executor
    )
    ib, cb = _multilook_band(
        gslc_ref, gslc_sec, "B", pol, looks_y, row_stride, window_rows, executor
    )
    h, w = min(ia.shape[0], ib.shape[0]), min(ia.shape[1], ib.shape[1])
    return PairMaps(
        band_difference(ia[:h, :w], ib[:h, :w]), np.fmin(ca[:h, :w], cb[:h, :w])
    )


def band_difference(ifg_a: np.ndarray, ifg_b: np.ndarray) -> np.ndarray:
    """Return D = ifg_A · conj(ifg_B) as a unit phasor (NaN where either is NaN)."""
    d = ifg_a * np.conj(ifg_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        return d / np.abs(d)


def _jumps(d: np.ndarray, w: np.ndarray, theta: float) -> np.ndarray:
    """Adjacent-bin phase jumps (cycles) of D binned along the normal at angle theta."""
    yy, xx = np.mgrid[0 : d.shape[0], 0 : d.shape[1]]
    s = xx * np.cos(theta) + yy * np.sin(theta)
    b = np.floor(s - s.min()).astype(int)
    ok = np.isfinite(d) & np.isfinite(w) & (w > 0)
    nb = int(b.max()) + 1
    wt = w[ok] ** 2
    p = np.bincount(b[ok], weights=(d[ok] * wt).real, minlength=nb) + 1j * np.bincount(
        b[ok], weights=(d[ok] * wt).imag, minlength=nb
    )
    n = np.bincount(b[ok], minlength=nb)
    full = 0.6 * np.median(n[n > 0])
    good = (n[1:] >= full) & (n[:-1] >= full)
    j = np.angle(p[1:] * np.conj(p[:-1])) / (2 * np.pi)
    return np.where(good, j, np.nan)


def date_line_score(
    pairs: Sequence[tuple[PairMaps, float]],
    reference: PairMaps,
    coarse_deg: float = 1.0,
    fine_deg: float = 0.1,
) -> tuple[float, float]:
    """Largest straight-line step that belongs to one date; returns (score, angle_deg).

    `pairs` holds the date's pairs with the sign its artefact takes in each (−1 where
    the date is the reference acquisition, +1 where it is the secondary); `reference` is
    a pair without the date. Maps must share one coarse grid.
    """

    def at(deg: float) -> float:
        th = np.radians(deg)
        stack = np.array([sign * _jumps(p.d, p.coherence, th) for p, sign in pairs])
        cnt = np.isfinite(stack).sum(axis=0)
        comb = np.where(cnt > 0, np.nansum(stack, axis=0) / np.maximum(cnt, 1), np.nan)
        ref = _jumps(reference.d, reference.coherence, th)
        ok = np.nonzero(np.isfinite(comb) & np.isfinite(ref))[0]
        if ok.size < 20:
            return 0.0
        ok = ok[(ok > ok.min() + 3) & (ok < ok.max() - 3)]
        c = comb[ok] - np.median(comb[ok])
        r = ref[ok] - np.median(ref[ok])
        noise = 1.4826 * np.median(np.abs(c))
        return float(np.max((np.abs(c) - np.abs(r)) / max(noise, 1e-9)))

    coarse = [(at(a), a) for a in np.arange(0.0, 180.0, coarse_deg)]
    best, a0 = max(coarse)
    fine = [(at(a), a) for a in np.arange(a0 - coarse_deg, a0 + coarse_deg, fine_deg)]
    best, a1 = max([(best, a0), *fine])
    return best, float(a1)


def coherence_drop(pairs: Sequence[PairMaps], reference: PairMaps) -> float:
    """Median log coherence of the reference pair minus the date's pairs' mean."""

    def lc(c):
        return np.log(np.clip(c, 1e-3, 1.0))

    ok = np.isfinite(reference.coherence)
    for p in pairs:
        ok &= np.isfinite(p.coherence)
    mean_pairs = np.mean([lc(p.coherence[ok]) for p in pairs], axis=0)
    return float(np.median(lc(reference.coherence[ok]) - mean_pairs))


def prescreen_dates(
    gslc_by_date: dict[str, str | Path],
    line_min: float = 14.0,
    line_rel_min: float = 1.6,
    coherence_drop_min: float = 0.4,
    looks_y: int = 64,
    row_stride: int = 1,
    n_workers: int = 1,
) -> tuple[list[dict], list[EpochFlag]]:
    """Score every date with a reference pair and return (scores, flags).

    Interior dates are first scored with the triplet (t-1, t, t+1), which with the
    newest date's (t-2, t-1, t) sets the frame median; after any flag, unflagged dates
    are re-scored against their nearest unflagged neighbours (a, t, b), and the newest
    date against the two most recent unflagged dates (a, b, t). The oldest date has no
    reference pair and is not scored.
    Each pair is formed once; `reference` in each score names the pair (a, b).
    """
    dates = sorted(gslc_by_date)
    cache: dict[tuple[str, str], PairMaps] = {}
    executor = None
    if n_workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        executor = ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn"))

    def pair(a: str, b: str) -> PairMaps:
        if (a, b) not in cache:
            cache[a, b] = multilook_pair(
                gslc_by_date[a],
                gslc_by_date[b],
                looks_y,
                row_stride=row_stride,
                executor=executor,
            )
        return cache[a, b]

    def common(maps: list[PairMaps]) -> list[PairMaps]:
        h = min(m.d.shape[0] for m in maps)
        w = min(m.d.shape[1] for m in maps)
        return [PairMaps(m.d[:h, :w], m.coherence[:h, :w]) for m in maps]

    def score(t: str, mode: str, ref_dates: str, maps: list[PairMaps], sign: float):
        p1, p2, ref = common(maps)
        line, angle = date_line_score([(p1, 1.0), (p2, sign)], ref)
        drop = coherence_drop([p1, p2], ref)
        return {
            "epoch": t,
            "mode": mode,
            "reference": ref_dates,
            "line_score": line,
            "angle_deg": angle,
            "coherence_drop": drop,
        }

    def flag(s: dict, median: float) -> list[EpochFlag]:
        out = []
        rel = s["line_score"] / median if median > 0 else np.inf
        s["line_rel"] = rel
        if s["line_score"] >= line_min and rel >= line_rel_min:
            out.append(EpochFlag(s["epoch"], "gslc_line", s["line_score"], np.nan))
        if s["coherence_drop"] >= coherence_drop_min:
            out.append(
                EpochFlag(s["epoch"], "gslc_decorrelated", s["coherence_drop"], np.nan)
            )
        return out

    def neighbours(k: int, bad: set[str]) -> tuple[str, str, str] | None:
        """(mode, a, b) for date k skipping flagged dates; None if it is unscorable."""
        before = [d for d in dates[:k] if d not in bad]
        after = [d for d in dates[k + 1 :] if d not in bad]
        if k == len(dates) - 1:
            return ("newest", before[-2], before[-1]) if len(before) >= 2 else None
        # an interior date with no unflagged date on one side keeps its last score
        return ("interior", before[-1], after[0]) if before and after else None

    def score_at(k: int, spec: tuple[str, str, str]) -> dict:
        mode, a, b = spec
        t = dates[k]
        if mode == "newest":
            return score(t, mode, f"{a}_{b}", [pair(b, t), pair(a, t), pair(a, b)], 1.0)
        return score(t, mode, f"{a}_{b}", [pair(a, t), pair(t, b), pair(a, b)], -1.0)

    by_date: dict[str, dict] = {}
    used: dict[str, tuple[str, str, str]] = {}
    flags: list[EpochFlag] = []
    try:
        for k in range(1, len(dates) - 1):
            used[dates[k]] = ("interior", dates[k - 1], dates[k + 1])
            by_date[dates[k]] = score_at(k, used[dates[k]])
        lines = [s["line_score"] for s in by_date.values()]
        if len(dates) >= 3:
            # the newest date's plain triplet joins the median only (more dates make it
            # robust when bad dates are many); it is flagged from its re-score below
            k = len(dates) - 1
            lines.append(
                score_at(k, ("newest", dates[k - 2], dates[k - 1]))["line_score"]
            )
        median = float(np.median(lines)) if len(lines) >= 3 else 0.0
        for s in by_date.values():
            flags += flag(s, median)
        # Re-score unflagged dates against their nearest unflagged neighbours until no
        # new flag appears, so adjacent bad dates cannot hide each other; the newest
        # date is first scored here.
        changed = True
        while changed:
            changed = False
            bad = {f.epoch for f in flags}
            for k in range(1, len(dates)):
                t = dates[k]
                spec = neighbours(k, bad)
                if t in bad or spec is None or used.get(t) == spec:
                    continue
                used[t] = spec
                by_date[t] = score_at(k, spec)
                new = flag(by_date[t], median)
                if new:
                    flags += new
                    changed = True
                    break
        scores = [by_date[d] for d in dates if d in by_date]
    finally:
        if executor is not None:
            executor.shutdown()
    return scores, flags
