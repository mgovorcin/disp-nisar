"""Per-epoch admissibility of a two-band GSLC stack for the ionosphere estimate.

Before any phase is touched, each acquisition is checked against the stack (requirement
R-F6/R-F10; edge cases S2 missing B, S3 mode change, S4 CRID change, E8 partial frame):

- ``missing_B``: the epoch has a frequency-A granule but no frequency-B granule;
- ``mode_change``: the centre frequency of A or B differs from the stack's majority
  value, or the radar mode / polarization field of the granule name differs from the
  majority (20 vs 40 vs 77 MHz modes have different f_A, and 77 MHz has no B at all);
- ``crid_change``: the granule's CRID (composite release id, ``_P05023_``) differs
  from the majority. This is a suspect, not a rejection: calibration changed between
  CRIDs and the epoch deserves a look in the line tests, so the reason is reported
  with ``reject=False``.

A flagged ``missing_B`` or ``mode_change`` epoch is excluded from the dual-band estimate
(the displacement can still be produced from A, without an ionosphere layer, and the
product flag says so). The stack must keep at least ``min_epochs`` admissible epochs,
otherwise the ionosphere stage raises a clear input error instead of failing inside a
block loop.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from opera_utils import get_dates

__all__ = ["EpochRecord", "AdmissibilityFlag", "parse_granule", "check_admissibility"]

_CRID = re.compile(r"_(P\d{5})_")
_MODE = re.compile(r"_(\d{4})_([A-Z]{4})_[AB]_\d{8}T")


@dataclass(frozen=True)
class EpochRecord:
    """One acquisition of the stack: its granules, centre frequencies, CRID and mode."""

    epoch: str
    file_a: Path | None
    file_b: Path | None
    f_a: float | None
    f_b: float | None
    crid: str | None
    mode: str | None


@dataclass(frozen=True)
class AdmissibilityFlag:
    """One reason an epoch is rejected (or, with ``reject=False``, only suspected)."""

    epoch: str
    reason: str
    detail: str
    reject: bool


def parse_granule(path: str | Path) -> tuple[str, str | None, str | None]:
    """Return (epoch YYYYMMDD, CRID, mode+pol) from a NISAR GSLC granule name.

    CRID and mode are None when the name does not carry them.
    """
    name = Path(path).name
    epoch = get_dates(name)[0].strftime("%Y%m%d")
    crid = m.group(1) if (m := _CRID.search(name)) else None
    mode = f"{m.group(1)}_{m.group(2)}" if (m := _MODE.search(name)) else None
    return epoch, crid, mode


def _majority(values):
    vals = [v for v in values if v is not None]
    return Counter(vals).most_common(1)[0][0] if vals else None


def check_admissibility(
    files_a: Sequence[str | Path],
    files_b: Sequence[str | Path] | None,
    center_frequencies: Callable[[Path], tuple[float, float]] | None = None,
    freq_tolerance_hz: float = 1e3,
    min_epochs: int = 3,
) -> tuple[list[EpochRecord], list[AdmissibilityFlag]]:
    """Build the epoch table and the flags.

    Raises ValueError on duplicate epochs or too few admissible ones.

    `center_frequencies(path) -> (f_A, f_B)` is read from each frequency-A granule when
    given (it is `disp_nisar.ionosphere.gslc.get_center_frequencies` in production;
    tests inject a stub).
    """
    by_a: dict[str, Path] = {}
    for f in files_a:
        e = parse_granule(f)[0]
        if e in by_a:
            raise ValueError(
                f"Duplicate frequency-A acquisition {e}: {by_a[e].name}, {Path(f).name}"
            )
        by_a[e] = Path(f)
    by_b: dict[str, Path] = {}
    for f in files_b or []:
        e = parse_granule(f)[0]
        if e in by_b:
            raise ValueError(
                f"Duplicate frequency-B acquisition {e}: {by_b[e].name}, {Path(f).name}"
            )
        by_b[e] = Path(f)
    records = []
    for e in sorted(by_a):
        fa = fb = None
        if center_frequencies is not None:
            try:
                fa, fb = center_frequencies(by_a[e])
            except (
                Exception
            ) as exc:  # a granule without a frequencyB group reads as B missing
                fa, fb = _first_or_none(center_frequencies, by_a[e]), None
                if fa is None:
                    raise ValueError(
                        f"Cannot read centre frequencies of {by_a[e].name}"
                    ) from exc
        _, crid, mode = parse_granule(by_a[e])
        records.append(EpochRecord(e, by_a[e], by_b.get(e), fa, fb, crid, mode))
    flags: list[AdmissibilityFlag] = []
    maj_fa = _majority([_round(r.f_a, freq_tolerance_hz) for r in records])
    maj_fb = _majority([_round(r.f_b, freq_tolerance_hz) for r in records])
    maj_mode = _majority([r.mode for r in records])
    maj_crid = _majority([r.crid for r in records])
    for r in records:
        if files_b is not None and r.file_b is None:
            flags.append(
                AdmissibilityFlag(r.epoch, "missing_B", "no frequency-B granule", True)
            )
        if r.f_b is None and center_frequencies is not None:
            flags.append(
                AdmissibilityFlag(
                    r.epoch,
                    "missing_B",
                    "no frequencyB centre frequency (single-band mode)",
                    True,
                )
            )
        if (
            maj_fa is not None
            and r.f_a is not None
            and _round(r.f_a, freq_tolerance_hz) != maj_fa
        ):
            flags.append(
                AdmissibilityFlag(
                    r.epoch,
                    "mode_change",
                    f"f_A {r.f_a / 1e6:.3f} MHz vs stack {maj_fa / 1e6:.3f}",
                    True,
                )
            )
        if (
            maj_fb is not None
            and r.f_b is not None
            and _round(r.f_b, freq_tolerance_hz) != maj_fb
        ):
            flags.append(
                AdmissibilityFlag(
                    r.epoch,
                    "mode_change",
                    f"f_B {r.f_b / 1e6:.3f} MHz vs stack {maj_fb / 1e6:.3f}",
                    True,
                )
            )
        if maj_mode is not None and r.mode is not None and r.mode != maj_mode:
            flags.append(
                AdmissibilityFlag(
                    r.epoch, "mode_change", f"mode {r.mode} vs stack {maj_mode}", True
                )
            )
        if maj_crid is not None and r.crid is not None and r.crid != maj_crid:
            flags.append(
                AdmissibilityFlag(
                    r.epoch, "crid_change", f"CRID {r.crid} vs stack {maj_crid}", False
                )
            )
    rejected = {f.epoch for f in flags if f.reject}
    admissible = [r for r in records if r.epoch not in rejected]
    if len(admissible) < min_epochs:
        raise ValueError(
            f"Only {len(admissible)} admissible two-band epochs of {len(records)} (need"
            f" {min_epochs}); "
            + "; ".join(f"{f.epoch}:{f.reason}" for f in flags if f.reject)
        )
    return records, flags


def _round(v, tol):
    return None if v is None else float(np.round(v / tol) * tol)


def _first_or_none(fn, path):
    try:
        return fn(path)[0]
    except Exception:
        return None
