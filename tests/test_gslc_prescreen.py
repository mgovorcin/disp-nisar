"""GSLC pre-screen on small synthetic two-band GSLC stacks."""

import json
from pathlib import Path

import h5py
import numpy as np

from disp_nisar.ionosphere.gslc_prescreen import prescreen_dates
from disp_nisar.main import _prescreen_gslc_epochs
from disp_nisar.pge_runconfig import AlgorithmParameters

ROWS, COLS_A, COLS_B = 512, 512, 64  # A at 5 m, B at 40 x 5 m: same ground footprint
DATES = [f"202601{d:02d}" for d in (1, 13, 25)] + ["20260206", "20260218"]


def _write(path: Path, za: np.ndarray, zb: np.ndarray) -> None:
    with h5py.File(path, "w") as h:
        for band, z, dx in (("A", za, 5.0), ("B", zb, 40.0)):
            g = h.create_group(f"science/LSAR/GSLC/grids/frequency{band}")
            g["HH"] = z.astype(np.complex64)
            g["mask"] = np.ones(z.shape, np.uint8)
            g["xCoordinateSpacing"] = dx
            g["yCoordinateSpacing"] = -5.0
            g["xCoordinates"] = 500000.0 + dx * np.arange(z.shape[1])
            g["yCoordinates"] = 4000000.0 - 5.0 * np.arange(z.shape[0])


def _stack(
    tmp_path, line_date=None, decorrelated_date=None, every_date_line=False, seed=0
):
    """Build a synthetic stack, optionally with one bad date.

    Each band has coherent speckle (independent between bands) plus small noise.
    `line_date` adds 0.6 rad to band A only on one side of a tilted straight line;
    `decorrelated_date` replaces half of the frame with new speckle in both bands;
    `every_date_line` gives each date its own line at a different offset (frame-wide
    ionosphere rather than a one-date artefact).
    """
    rng = np.random.default_rng(seed)

    def cplx(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    sa, sb = cplx((ROWS, COLS_A)), cplx((ROWS, COLS_B))
    ya, xa = np.mgrid[0:ROWS, 0:COLS_A] * 5.0
    yb, xb = np.mgrid[0:ROWS, 0:COLS_B]
    yb, xb = yb * 5.0, xb * 40.0 + 17.5
    files = {}
    for d in DATES:
        za = sa + 0.3 * cplx(sa.shape)
        zb = sb + 0.3 * cplx(sb.shape)
        if every_date_line:
            off = 600.0 + 300.0 * DATES.index(d)
            za = za * np.exp(1j * 0.6 * ((ya - 0.2 * xa) > off))
        if d == line_date:
            za = za * np.exp(1j * 0.6 * ((ya - 0.2 * xa) > 1200.0))
        if d == decorrelated_date:
            za = np.where(ya > 1280.0, cplx(sa.shape), za)
            zb = np.where(yb > 1280.0, cplx(sb.shape), zb)
        files[d] = tmp_path / f"gslc_{d}.h5"
        _write(files[d], za, zb)
    return files


def test_clean_stack_is_not_flagged(tmp_path):
    scores, flags = prescreen_dates(_stack(tmp_path), looks_y=8)
    assert flags == []
    assert {s["epoch"] for s in scores} == set(
        DATES[1:]
    )  # the oldest date has no reference pair


def test_interior_line_artefact_is_flagged_on_its_date_only(tmp_path):
    scores, flags = prescreen_dates(_stack(tmp_path, line_date=DATES[2]), looks_y=8)
    assert [(f.epoch, f.reason) for f in flags] == [(DATES[2], "gslc_line")]


def test_newest_date_line_artefact_is_flagged(tmp_path):
    scores, flags = prescreen_dates(_stack(tmp_path, line_date=DATES[-1]), looks_y=8)
    assert (DATES[-1], "gslc_line") in [(f.epoch, f.reason) for f in flags]
    assert all(f.epoch == DATES[-1] for f in flags)


def test_decorrelated_date_is_flagged(tmp_path):
    scores, flags = prescreen_dates(
        _stack(tmp_path, decorrelated_date=DATES[2]), looks_y=8
    )
    assert (DATES[2], "gslc_decorrelated") in [(f.epoch, f.reason) for f in flags]
    assert all(f.epoch == DATES[2] for f in flags if f.reason == "gslc_decorrelated")


def test_flagged_neighbour_does_not_leak_into_newest_date(tmp_path):
    scores, flags = prescreen_dates(_stack(tmp_path, line_date=DATES[-3]), looks_y=8)
    assert [(f.epoch, f.reason) for f in flags] == [(DATES[-3], "gslc_line")]
    newest = next(s for s in scores if s["mode"] == "newest")
    assert newest["epoch"] == DATES[-1]
    assert newest["reference"] == f"{DATES[-4]}_{DATES[-2]}"  # skips the flagged t-2


def test_lines_on_every_date_are_not_flagged(tmp_path):
    scores, flags = prescreen_dates(_stack(tmp_path, every_date_line=True), looks_y=8)
    assert all(s["line_score"] >= 14.0 for s in scores if s["mode"] == "interior")
    assert [f for f in flags if f.reason == "gslc_line"] == []


def test_prescreen_is_off_by_default():
    assert AlgorithmParameters().gslc_prescreen is False


def test_workflow_prescreen_skips_compressed_and_writes_report(tmp_path):
    files = _stack(tmp_path, line_date=DATES[2])
    renamed = []
    for d, f in files.items():  # date in the name as get_dates expects
        g = f.with_name(f"NISAR_GSLC_{d}T000000_{d}T000010.h5")
        f.rename(g)
        renamed.append(g)
    compressed = tmp_path / "compressed_20251201T000000_20251201T000010.h5"
    compressed.write_bytes(b"")  # unreadable: would fail if it were scored
    out = tmp_path / "report" / "gslc_prescreen.json"
    flags = _prescreen_gslc_epochs(
        [compressed, *renamed], out, row_stride=1, n_workers=1, looks_y=8
    )
    assert [(f.epoch, f.reason) for f in flags] == [(DATES[2], "gslc_line")]
    report = json.loads(out.read_text())
    assert report["flags"][0]["epoch"] == DATES[2]
    assert {s["epoch"] for s in report["scores"]} == set(DATES[1:])
