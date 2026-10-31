"""Improved main_side on small synthetic two-band time series."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from disp_nisar.ionosphere.main_side_recipe import run_main_side_recipe
from disp_nisar.ionosphere.options import IonosphereOptions
from disp_nisar.ionosphere.utils import freq_to_wavelength, phase_to_disp

F_A, F_B = 1.2390e9, 1.2935e9
ROWS, COLS = 300, 300  # A at 30 m; B at 40 m over the same 9 km
PAIRS = [f"20260101_202601{d:02d}" for d in (13, 25)] + [
    f"20260101_202602{d:02d}" for d in (6, 18)
]
ORIGIN = (500000.0, 4000000.0)


def _write(path: Path, a: np.ndarray, px: float) -> None:
    profile = {
        "driver": "GTiff",
        "height": a.shape[0],
        "width": a.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32611",
        "transform": from_origin(*ORIGIN, px, px),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(a.astype("float32"), 1)


def _iono_rad(px: float, shape: tuple[int, int], k: int) -> np.ndarray:
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]] * px
    return (k + 1) * 0.4 * (x / 9000.0 + 0.5 * y / 9000.0)


def _stack(tmp_path: Path, b_offset=0.0, patch=False):
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "reference_point.txt").write_text("150,150\n")
    shape_b = (225, 225)
    for k, pair in enumerate(PAIRS):
        phi_a = _iono_rad(30.0, (ROWS, COLS), k) + 1.0
        phi_b = (F_A / F_B) * _iono_rad(40.0, shape_b, k) + (F_B / F_A) * 1.0
        b = phase_to_disp(phi_b, freq_to_wavelength(F_B)) + b_offset * (k + 1)
        if patch:  # bands disagree by 0.4 cycle over a 1 km patch, on every date
            b[100:125, 100:125] += 0.4 * freq_to_wavelength(F_B) / 2
        _write(
            dir_a / f"{pair}.tif", phase_to_disp(phi_a, freq_to_wavelength(F_A)), 30.0
        )
        _write(dir_b / f"{pair}.tif", b, 40.0)
    quality = []
    for name, shape, px in (("a", (ROWS, COLS), 30.0), ("b", shape_b, 40.0)):
        tc, sim = tmp_path / f"tc_{name}.tif", tmp_path / f"sim_{name}.tif"
        _write(tc, np.full(shape, 0.9), px)
        _write(sim, np.full(shape, 0.8), px)
        quality.append((tc, sim))
    return dir_a, dir_b, quality


def _run(tmp_path, **kw):
    dir_a, dir_b, (quality_a, quality_b) = _stack(tmp_path, **kw)
    return run_main_side_recipe(
        dir_a,
        dir_b,
        quality_a,
        quality_b,
        tmp_path / "out",
        [Path(f"{p}.tif") for p in PAIRS],
        F_A,
        F_B,
        IonosphereOptions(main_side_sigma_km=0.5, main_side_consistency_km=1.5),
    )


def _error_mm(path: Path, k: int) -> np.ndarray:
    """Return the screen's error in mm, away from the 2 km edges the kernel loses."""
    with rasterio.open(path) as src:
        a = src.read(1).astype("float64")
    truth = phase_to_disp(_iono_rad(30.0, (ROWS, COLS), k), freq_to_wavelength(F_A))
    return ((a - truth) - (a - truth)[150, 150])[60:240, 60:240] * 1000


def test_recovers_the_screen_and_writes_meters(tmp_path):
    outputs = _run(tmp_path)
    assert [p.name for p in outputs] == [f"{p}_iono.tif" for p in PAIRS]
    for k, path in enumerate(outputs):
        assert np.nanmax(np.abs(_error_mm(path, k))) < 3.0
    with rasterio.open(outputs[0]) as src:
        assert src.shape == (ROWS, COLS)
        assert src.units[0] == "meters"
        assert src.tags()["ionosphere_method"] == "main_side_recipe"


def test_area_referencing_removes_a_per_date_band_offset(tmp_path):
    outputs = _run(tmp_path, b_offset=0.02)
    for k, path in enumerate(outputs):
        assert np.nanmax(np.abs(_error_mm(path, k))) < 3.0


def test_inconsistent_patch_does_not_become_a_bullseye(tmp_path):
    outputs = _run(tmp_path, patch=True)
    for k, path in enumerate(outputs):
        assert np.nanmax(np.abs(_error_mm(path, k))) < 3.0


def test_stock_estimator_stays_the_default():
    assert IonosphereOptions().main_side_recipe == "stock"
