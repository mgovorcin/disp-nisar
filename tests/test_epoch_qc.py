import numpy as np
from rasterio.transform import from_origin

from disp_nisar.ionosphere.epoch_qc import (
    EpochFlag,
    StepResult,
    along_track_step,
    along_track_unit,
    band_difference_cycles,
    flag_band_difference,
    flag_displacement,
    read_bad_epochs,
    write_bad_epochs,
)

TR = from_origin(500000.0, 4000000.0, 90.0, 90.0)
SHAPE = (400, 400)


def _xy():
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    return TR.c + TR.a * (xx + 0.5), TR.f + TR.e * (yy + 0.5)


def _field(step=0.0, line_frac=0.5, seed=0, along=(0.0, 1.0)):
    """Plane + noise, plus a step across a line perpendicular to `along`.

    The step of size `step` sits at fraction `line_frac` of the along-track extent.
    """
    rng = np.random.default_rng(seed)
    x, y = _xy()
    s = x * along[0] + y * along[1]
    f = (
        1e-6 * (x - x.mean())
        + 3e-6 * (y - y.mean())
        + 0.01 * rng.standard_normal(SHAPE)
    )
    thr = s.min() + line_frac * (s.max() - s.min())
    return f + step * (s > thr)


def test_along_track_unit_is_perpendicular_to_look():
    e = np.full((5, 5), -0.6)
    n = np.full((5, 5), 0.1)
    v = along_track_unit(e, n)
    assert abs(np.hypot(*v) - 1) < 1e-12
    assert abs(v @ np.array([-0.6, 0.1]) / np.hypot(-0.6, 0.1)) < 1e-12


def test_band_difference_cancels_nondispersive():
    f_a, f_b = 1239.0e6, 1293.5e6
    nd = np.linspace(0, 20, 50)
    assert np.allclose(band_difference_cycles(nd, (f_b / f_a) * nd, f_a, f_b), 0)
    d = 2 * np.pi * np.ones(50)  # one cycle of ionosphere at A
    r = band_difference_cycles(d, (f_a / f_b) * d, f_a, f_b)
    assert np.allclose(r, (f_a / f_b - f_b / f_a))  # about -0.086


def test_smooth_field_is_not_flagged():
    f = _field(step=0.0)
    ok = np.ones(SHAPE, bool)
    r = along_track_step(f, ok, TR, (0.0, 1.0))
    assert np.isfinite(r.step)
    assert not flag_band_difference(r)
    assert r.ratio < 10


def test_straight_line_is_flagged_with_position():
    f = _field(step=0.9, line_frac=0.4)
    ok = np.ones(SHAPE, bool)
    r = along_track_step(f, ok, TR, (0.0, 1.0))
    # the robust plane absorbs part of a step (the calibrated artefacts measured
    # 0.89-0.93 of a cycle); the decision is on sharpness (ratio), not size
    assert 0.6 < abs(r.step) < 1.0
    assert r.ratio > 50
    assert flag_band_difference(r)
    # position is absolute along-track km, (x E + y N) / 1000: the line sits 40 % up the
    # 36 km frame
    x, y = _xy()
    s_km = y / 1000.0
    expected = s_km.min() + 0.4 * (s_km.max() - s_km.min())
    assert abs(r.position_km - expected) < 2.0


def test_too_few_points_gives_nan():
    f = _field()
    ok = np.zeros(SHAPE, bool)
    ok[:30, :30] = True
    r = along_track_step(f, ok, TR, (0.0, 1.0))
    assert np.isnan(r.step) and not flag_band_difference(r)


def test_displacement_rule_ignores_shared_positions():
    shared = [
        StepResult(60.0, 12.0, 2.0) for _ in range(9)
    ]  # same island offset on every epoch
    one_off = StepResult(80.0, 25.0, 2.0)  # a real bad acquisition elsewhere
    small = StepResult(30.0, 5.0, 1.0)  # large ratio but below 45 mm
    flags = flag_displacement([*shared, one_off, small])
    assert flags == [False] * 9 + [True, False]


def test_write_and_read_bad_epochs(tmp_path):
    p = write_bad_epochs(
        tmp_path / "bad_epochs.csv",
        [
            EpochFlag("20260222", "band_difference_line", 288.0, 41.2),
            EpochFlag("20260831", "displacement_line", 62.0, 10.0),
            EpochFlag("20260222", "displacement_line", 20.0, 41.0),
        ],
    )
    text = p.read_text().splitlines()
    assert text[0] == "epoch,reason,statistic,position_km"
    assert len(text) == 4
    assert read_bad_epochs(p) == {"20260222", "20260831"}
    (tmp_path / "bad_dates.txt").write_text("20260101\n20260113\n")
    assert read_bad_epochs(tmp_path / "bad_dates.txt") == {"20260101", "20260113"}
    assert read_bad_epochs(tmp_path / "missing.csv") == set()


def test_flag_screens_on_rasters(tmp_path):
    import rasterio

    from disp_nisar.ionosphere.epoch_qc import flag_screens

    prof = {
        "driver": "GTiff",
        "height": SHAPE[0],
        "width": SHAPE[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32610",
        "transform": TR,
        "nodata": np.nan,
    }
    # a real artefact is a line of constant azimuth, i.e. perpendicular to the
    # along-track direction that the LOS vectors imply; build the synthetic line with
    # that same direction
    e = np.full((5, 5), -0.6)
    n = np.full((5, 5), 0.1)
    along = along_track_unit(e, n)
    paths = []
    for epoch, step in (("20260101", 0.0), ("20260113", 11.0), ("20260125", 0.0)):
        f = tmp_path / f"20251201_{epoch}_iono_B.rad.tif"
        with rasterio.open(f, "w", **prof) as dst:
            dst.write(
                (
                    _field(step=step, line_frac=0.5, along=tuple(along)) * 2 * np.pi
                ).astype("float32"),
                1,
            )
        paths.append(f)
    results, flags = flag_screens(paths, e, n)
    assert len(results) == 3
    assert [f.epoch for f in flags] == ["20260113"]
    assert flags[0].reason == "screen_line" and flags[0].statistic > 50


def test_along_track_unit_accepts_a_zero_component():
    v = along_track_unit(np.full((3, 3), -0.99), np.zeros((3, 3)))
    assert abs(abs(v[1]) - 1) < 1e-9  # pure east look -> along-track is north-south


def test_flag_d_series_skips_recurring_edge_steps(tmp_path):
    import rasterio

    from disp_nisar.ionosphere.epoch_qc import flag_d_series

    prof = {
        "driver": "GTiff",
        "height": SHAPE[0],
        "width": SHAPE[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32610",
        "transform": TR,
        "nodata": np.nan,
    }
    e = np.full((5, 5), -0.6)
    n = np.full((5, 5), 0.1)
    along = tuple(along_track_unit(e, n))
    # six epochs share an edge step at 20 % of the frame; one epoch carries a sharp line
    # at 60 %; one a line that is sharp but below 0.02 cycle
    cases = [(f"202601{d:02d}", 0.4, 0.2) for d in range(1, 7)] + [
        ("20260120", 0.5, 0.6),
        ("20260125", 0.012, 0.8),
    ]
    paths = []
    for epoch, step_cyc, frac in cases:
        f = tmp_path / f"20251201_{epoch}_D.rad.tif"
        with rasterio.open(f, "w", **prof) as dst:
            dst.write(
                (_field(step=step_cyc, line_frac=frac, along=along) * 2 * np.pi).astype(
                    "float32"
                ),
                1,
            )
        paths.append(f)
    results, flags = flag_d_series(paths, e, n)
    assert len(results) == len(cases)
    assert [f.epoch for f in flags] == ["20260120"]
    assert flags[0].reason == "d_line"


def test_coherence_streak_ratio_separates_constant_azimuth_bands():
    from disp_nisar.ionosphere.epoch_qc import coherence_streak_ratio

    rng = np.random.default_rng(3)
    shape = (1200, 1200)  # 30 m pixels, 36 km
    e, n = np.full((5, 5), -0.6), np.full((5, 5), 0.1)
    along = along_track_unit(e, n)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    x, y = xx * 30.0, -yy * 30.0
    s = x * along[0] + y * along[1]  # metres along track
    base = np.clip(0.9 + 0.05 * rng.standard_normal(shape), 0, 1)
    clean = coherence_streak_ratio(base, 30.0, along)
    bands = np.clip(base - 0.3 * (np.sin(2 * np.pi * s / 4000.0) > 0.7), 0, 1)
    streaky = coherence_streak_ratio(bands, 30.0, along)
    assert clean < 2.0
    assert streaky > 3.0


def test_streak_epochs_names_the_shared_date():
    from disp_nisar.ionosphere.epoch_qc import flag_streak_pairs, streak_epochs

    pairs = [
        "20260117_20260129",
        "20260129_20260210",
        "20260129_20260222",
        "20260210_20260222",
        "20260222_20260306",
        "20260210_20260306",
        "20260306_20260318",
    ]
    ratios = {p: 10.0 if "20260222" in p else 0.8 for p in pairs}
    flagged = flag_streak_pairs(ratios)
    assert flagged == ["20260129_20260222", "20260210_20260222", "20260222_20260306"]
    assert streak_epochs(flagged, pairs) == ["20260222"]


def test_flag_streak_pairs_is_relative_to_the_frame():
    from disp_nisar.ionosphere.epoch_qc import flag_streak_pairs

    textured = {
        f"p{k}": 4.0 + 0.1 * (k % 5) for k in range(40)
    }  # an anisotropic frame, no artefact
    assert flag_streak_pairs(textured) == []
    quiet = {f"p{k}": 0.4 + 0.02 * (k % 5) for k in range(40)}
    quiet["bad"] = 1.6  # 4x the median but under the floor
    assert flag_streak_pairs(quiet) == []
    quiet["bad"] = 9.0
    assert flag_streak_pairs(quiet) == ["bad"]
