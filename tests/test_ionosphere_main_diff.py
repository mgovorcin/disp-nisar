"""Run main_diff end to end on a synthetic stack with a known ionosphere.

The answer is an observation rather than a recording: the phases are built from a
chosen ionosphere and non-dispersive signal, so the correction must return that
ionosphere. Only the unwrapper is replaced, since these phases do not wrap.
"""

from datetime import datetime

import h5py
import numpy as np
import pytest
import rasterio
from affine import Affine
from dolphin.workflows.config import UnwrapOptions
from rasterio.crs import CRS

from disp_nisar.ionosphere.ionosphere_filter import apply_ionosphere_filter
from disp_nisar.ionosphere.main_diff import (
    BandInputs,
    estimate_iono_main_diff,
    run_main_diff_estimation,
)
from disp_nisar.ionosphere.options import IonosphereFilterOptions, IonosphereOptions
from disp_nisar.ionosphere.utils import freq_to_wavelength

F_A, F_B = 1.2395e9, 1.2935e9
CRS_ = CRS.from_epsg(32611)
X0, Y0 = 500000.0, 3800000.0
# The real ratio on NISAR 20+5 frames is non-integer (30 m to 40 m)
SHAPE_A, TRANSFORM_A = (120, 160), Affine(30, 0, X0, 0, -30, Y0)
SHAPE_B, TRANSFORM_B = (90, 120), Affine(40, 0, X0, 0, -40, Y0)
DATES = ["20260101", "20260113", "20260125"]
PAIRS = [(0, 1), (1, 2), (0, 2)]
REFERENCE_A = (60, 80)
# Ionosphere at each epoch, as radians per frame width along x and y
IONO_RAMPS = [(0.0, 0.0), (0.6, -0.3), (-0.4, 0.5)]
OFFSETS = [0.0, 0.3, -0.4]
WIDTH_M = 4800.0


def _xy(shape, transform):
    rows, cols = np.mgrid[: shape[0], : shape[1]]
    x, y = transform * (cols + 0.5, rows + 0.5)
    return x - X0, y - Y0


def _iono(epoch, x, y):
    gx, gy = IONO_RAMPS[epoch]
    return (gx * x + gy * y) / WIDTH_M


def _write(path, array, transform, nodata=None):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=CRS_,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)
    return path


def _band_inputs(folder, shape, transform, phase_of_pair):
    folder.mkdir()
    ifgs, cors = [], []
    for i, j in PAIRS:
        name = f"{DATES[i]}_{DATES[j]}"
        phase = phase_of_pair(i, j, *_xy(shape, transform))
        ifgs.append(
            _write(
                folder / f"{name}.int.tif",
                np.exp(1j * phase).astype(np.complex64),
                transform,
            )
        )
        cors.append(
            _write(folder / f"{name}.cor.tif", np.full(shape, 0.9, "f4"), transform)
        )
    similarity = _write(folder / "similarity.tif", np.full(shape, 0.9, "f4"), transform)
    return BandInputs(ifgs, cors, similarity)


def _write_gslc(path):
    with h5py.File(path, "w") as f:
        for band, shape, step in (("A", SHAPE_A, 30.0), ("B", SHAPE_B, 40.0)):
            group = f.create_group(f"science/LSAR/GSLC/grids/frequency{band}")
            group["mask"] = np.ones(shape, np.uint8)
            group["xCoordinates"] = X0 + step / 2 + step * np.arange(shape[1])
            group["yCoordinates"] = Y0 - step / 2 - step * np.arange(shape[0])
            group["xCoordinateSpacing"] = step
            group["yCoordinateSpacing"] = -step
            group["projection"] = 32611
    return path


@pytest.fixture
def unwrap_calls(monkeypatch):
    """Replace the unwrapper with the wrapped phase, which never wraps here."""
    calls = []

    def fake_unwrap(*, ifg_filename, unw_filename, **_kwargs):
        calls.append(ifg_filename)
        with rasterio.open(ifg_filename) as ds:
            phase = np.angle(ds.read(1)).astype(np.float32)
            profile = ds.profile
        profile.update(dtype="float32", nodata=np.nan)
        with rasterio.open(unw_filename, "w", **profile) as dst:
            dst.write(phase, 1)
        return unw_filename, None

    monkeypatch.setattr("dolphin.unwrap.unwrap", fake_unwrap)
    return calls


@pytest.fixture
def stack(tmp_path):
    def phase_a(i, j, x, y):
        return OFFSETS[j] - OFFSETS[i] + _iono(j, x, y) - _iono(i, x, y)

    def phase_b(i, j, x, y):
        nondispersive = (OFFSETS[j] - OFFSETS[i]) * F_B / F_A
        return nondispersive + (_iono(j, x, y) - _iono(i, x, y)) * F_A / F_B

    products = [
        _write(
            tmp_path / f"{DATES[0]}_{DATES[k]}.tif",
            np.zeros(SHAPE_A, "f4"),
            TRANSFORM_A,
            nodata=np.nan,
        )
        for k in (1, 2)
    ]
    return {
        "band_a": _band_inputs(tmp_path / "A", SHAPE_A, TRANSFORM_A, phase_a),
        "band_b": _band_inputs(tmp_path / "B", SHAPE_B, TRANSFORM_B, phase_b),
        "gslc_by_date": {
            datetime.strptime(d, "%Y%m%d"): _write_gslc(
                tmp_path / f"NISAR_L2_GSLC_{d}.h5"
            )
            for d in DATES
        },
        "output_timeseries_paths": products,
        "reference_a": REFERENCE_A,
        "f_A": F_A,
        "f_B": F_B,
        "out_dir": tmp_path / "iono",
        "unwrap_options": UnwrapOptions(unwrap_method="snaphu"),
    }


def _options(**kwargs):
    kwargs.setdefault(
        "filtering",
        IonosphereFilterOptions(sigma_m=(120.0, 120.0), donor_erosion_px=(0, 0)),
    )
    return IonosphereOptions(method="main_diff", **kwargs)


def _read(path):
    with rasterio.open(path) as ds:
        return ds.read(1)


def test_estimator_recovers_the_ionosphere():
    rng = np.random.default_rng(0)
    nondispersive, iono = rng.normal(size=(2, 4, 5))
    phase_a = nondispersive + iono
    phase_b = nondispersive * F_B / F_A + iono * F_A / F_B
    out = estimate_iono_main_diff(F_A, F_B, phase_a, phase_a - phase_b)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, iono, atol=1e-5)


@pytest.mark.parametrize(
    ("f_a", "f_b", "message"),
    [
        (F_A, F_A, "indistinguishable"),
        (-F_A, F_B, "finite and positive"),
        (F_A, np.nan, "finite and positive"),
    ],
)
def test_estimator_rejects_bad_frequencies(f_a, f_b, message):
    with pytest.raises(ValueError, match=message):
        estimate_iono_main_diff(f_a, f_b, np.zeros((1, 1)), np.zeros((1, 1)))


def test_estimator_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        estimate_iono_main_diff(F_A, F_B, np.zeros((1, 1)), np.zeros((1, 2)))


@pytest.mark.parametrize("product", [0, 1])
def test_correction_returns_the_known_ionosphere(stack, unwrap_calls, product):
    outputs = run_main_diff_estimation(**stack, options=_options())

    assert [p.name for p in outputs] == [
        "20260101_20260113_iono.tif",
        "20260101_20260125_iono.tif",
    ]
    assert len(unwrap_calls) == 2 * len(PAIRS)
    with rasterio.open(outputs[product]) as ds:
        assert ds.units == ("meters",)
        assert ds.tags()["ionosphere_method"] == "main_diff"
        correction = ds.read(1)
    assert correction.shape == SHAPE_A
    assert correction[REFERENCE_A] == pytest.approx(0.0, abs=1e-6)

    x, y = _xy(SHAPE_A, TRANSFORM_A)
    epoch = product + 1
    iono = _iono(epoch, x, y)
    expected = -freq_to_wavelength(F_A) / (4 * np.pi) * (iono - iono[REFERENCE_A])
    interior = (slice(25, -25), slice(25, -25))
    # 1 mm, against a signal of about 10 mm across the interior
    assert np.ptp(expected[interior]) > 0.005
    np.testing.assert_allclose(correction[interior], expected[interior], atol=1e-3)


def test_native_d_gives_the_same_correction_on_a_smooth_ionosphere(stack, unwrap_calls):
    coarse = run_main_diff_estimation(**stack, options=_options(d_looks=10))
    expected = _read(coarse[0])
    assert any("D_coarse_ifg" in str(call) for call in unwrap_calls)
    unwrap_calls.clear()

    native = run_main_diff_estimation(
        **stack, options=_options(d_looks=1, resume=False)
    )

    assert not any("D_coarse_ifg" in str(call) for call in unwrap_calls)
    interior = (slice(25, -25), slice(25, -25))
    np.testing.assert_allclose(
        _read(native[0])[interior], expected[interior], atol=1e-3
    )


def test_changing_d_looks_stops_a_resume(stack, unwrap_calls):
    run_main_diff_estimation(**stack, options=_options(d_looks=10))
    count = len(unwrap_calls)
    with pytest.raises(ValueError, match="used different"):
        run_main_diff_estimation(**stack, options=_options(d_looks=4))
    assert len(unwrap_calls) == count


def _robust_options(**kwargs):
    return _options(
        filtering=IonosphereFilterOptions(
            method="robust",
            sigma_m=(120.0, 120.0),
            robust_fallback_sigma_m=500.0,
            robust_clip_block_m=500.0,
            **kwargs,
        )
    )


def _write_incidence(tmp_path):
    x, _ = _xy(SHAPE_B, TRANSFORM_B)
    return _write(
        tmp_path / "incidence.tif", (33.0 + x / 400.0).astype("f4"), TRANSFORM_B
    )


def test_robust_filter_returns_the_known_ionosphere(stack, unwrap_calls, tmp_path):
    outputs = run_main_diff_estimation(
        **stack,
        options=_robust_options(robust_near_range_exclude_km=0.4),
        incidence_file=_write_incidence(tmp_path),
    )

    assert len(unwrap_calls) == 2 * len(PAIRS)
    correction = _read(outputs[1])
    x, y = _xy(SHAPE_A, TRANSFORM_A)
    iono = _iono(2, x, y)
    expected = -freq_to_wavelength(F_A) / (4 * np.pi) * (iono - iono[REFERENCE_A])
    interior = (slice(35, -35), slice(35, -35))
    np.testing.assert_allclose(correction[interior], expected[interior], atol=1e-3)
    band = _read(stack["out_dir"] / "timeseries" / "near_range_band.tif")
    # 400 m is 10 pixels from the near edge, away from the frame's corners
    assert band[45, :10].all()
    assert not band[45, 12:].any()


def test_robust_filter_runs_without_an_incidence_layer(stack, unwrap_calls):
    assert not unwrap_calls
    outputs = run_main_diff_estimation(**stack, options=_robust_options())
    assert not (stack["out_dir"] / "timeseries" / "near_range_band.tif").exists()
    assert np.isfinite(_read(outputs[0])).any()


def test_robust_filter_weights_equally_with_too_few_epochs(stack, unwrap_calls):
    assert not unwrap_calls
    run_main_diff_estimation(**stack, options=_robust_options())
    noise = stack["out_dir"] / "timeseries" / "ionosphere_noise.rad.tif"
    assert np.isnan(_read(noise)).all()


def test_noise_weights_can_be_turned_off(stack, unwrap_calls):
    assert not unwrap_calls
    run_main_diff_estimation(
        **stack, options=_robust_options(robust_noise_weights=False)
    )
    assert not (stack["out_dir"] / "timeseries" / "ionosphere_noise.rad.tif").exists()


def test_robust_layers_reach_the_filter(stack, unwrap_calls, tmp_path, monkeypatch):
    seen = []

    def recording_filter(phase, **kwargs):
        seen.append(kwargs)
        return apply_ionosphere_filter(phase, **kwargs)

    monkeypatch.setattr(
        "disp_nisar.ionosphere.main_diff.apply_ionosphere_filter", recording_filter
    )
    run_main_diff_estimation(
        **stack,
        options=_robust_options(robust_near_range_exclude_km=0.4),
        incidence_file=_write_incidence(tmp_path),
    )

    assert len(unwrap_calls) == 2 * len(PAIRS)
    assert len(seen) == 2
    assert seen[0]["noise"].shape == SHAPE_B
    assert seen[0]["exclude_mask"].dtype == bool
    assert seen[0]["exclude_mask"][45, :10].all()


def test_retuning_robust_settings_rebuilds_its_layers(stack, unwrap_calls, tmp_path):
    incidence = _write_incidence(tmp_path)
    run_main_diff_estimation(
        **stack,
        options=_robust_options(robust_near_range_exclude_km=0.4),
        incidence_file=incidence,
    )
    count = len(unwrap_calls)
    band_path = stack["out_dir"] / "timeseries" / "near_range_band.tif"
    narrow = np.count_nonzero(_read(band_path))

    run_main_diff_estimation(
        **stack,
        options=_robust_options(robust_near_range_exclude_km=1.2),
        incidence_file=incidence,
    )

    assert len(unwrap_calls) == count
    assert np.count_nonzero(_read(band_path)) > narrow


def test_resume_reuses_finished_work(stack, unwrap_calls):
    first = run_main_diff_estimation(**stack, options=_options())
    before = _read(first[0])
    count = len(unwrap_calls)

    second = run_main_diff_estimation(**stack, options=_options())

    assert len(unwrap_calls) == count
    np.testing.assert_array_equal(_read(second[0]), before)


def test_retuning_the_filter_does_not_redo_the_unwrapping(stack, unwrap_calls):
    first = run_main_diff_estimation(**stack, options=_options())
    before = _read(first[0])
    count = len(unwrap_calls)

    options = _options(
        filtering=IonosphereFilterOptions(
            method="smooth_iono", sigma_m=(400.0, 400.0), donor_erosion_px=(0, 0)
        )
    )
    second = run_main_diff_estimation(**stack, options=options)

    assert len(unwrap_calls) == count
    assert not np.allclose(_read(second[0]), before, equal_nan=True)


@pytest.mark.parametrize(
    ("change", "changed"),
    [
        ({"options": _options(correlation_threshold=0.5)}, "options"),
        ({"reference_a": (40, 40)}, "reference_a"),
    ],
)
def test_resume_stops_when_settings_changed(stack, unwrap_calls, change, changed):
    """Regression: old pairs were silently reused after a settings change."""
    run_main_diff_estimation(**stack, options=_options())
    count = len(unwrap_calls)

    kwargs = {"options": _options(), **stack, **change}
    with pytest.raises(ValueError, match=rf"used different \['{changed}'\]"):
        run_main_diff_estimation(**kwargs)
    assert len(unwrap_calls) == count


def test_resume_stops_when_an_input_changed(stack, unwrap_calls):
    assert not unwrap_calls
    run_main_diff_estimation(**stack, options=_options())
    similarity = stack["band_b"].similarity
    _write(similarity, np.full(SHAPE_B, 0.8, "f4"), TRANSFORM_B)
    with pytest.raises(ValueError, match=r"used different \['inputs'\]"):
        run_main_diff_estimation(**stack, options=_options())


def test_resume_stops_on_results_without_a_record(stack, unwrap_calls):
    assert not unwrap_calls
    (stack["out_dir"] / "pairs").mkdir(parents=True)
    with pytest.raises(ValueError, match="no settings record"):
        run_main_diff_estimation(**stack, options=_options())


def test_resume_false_discards_old_results(stack, unwrap_calls):
    run_main_diff_estimation(**stack, options=_options())
    count = len(unwrap_calls)

    run_main_diff_estimation(
        **stack, options=_options(correlation_threshold=0.5, resume=False)
    )

    assert len(unwrap_calls) == 2 * count


def test_networks_must_match(stack, unwrap_calls):
    assert not unwrap_calls
    band_b = stack["band_b"]
    stack["band_b"] = BandInputs(
        band_b.interferograms[:-1], band_b.correlations[:-1], band_b.similarity
    )
    with pytest.raises(ValueError, match="networks must match exactly"):
        run_main_diff_estimation(**stack, options=_options())


def test_missing_gslc_is_reported(stack, unwrap_calls):
    assert not unwrap_calls
    stack["gslc_by_date"].pop(next(iter(stack["gslc_by_date"])))
    with pytest.raises(ValueError, match="Missing raw GSLC masks"):
        run_main_diff_estimation(**stack, options=_options())


def test_reference_outside_the_product_is_rejected(stack, unwrap_calls):
    assert not unwrap_calls
    stack["reference_a"] = (5000, 0)
    with pytest.raises(ValueError, match="outside A product grid"):
        run_main_diff_estimation(**stack, options=_options())


def test_bands_must_stay_on_one_grid(stack, unwrap_calls):
    assert not unwrap_calls
    similarity = stack["band_b"].similarity
    _write(similarity, np.full((10, 10), 0.9, "f4"), TRANSFORM_B)
    with pytest.raises(ValueError, match="Grid differs from the band"):
        run_main_diff_estimation(**stack, options=_options())


def test_b_grid_must_not_be_finer_than_a(stack, unwrap_calls):
    assert not unwrap_calls
    stack["band_a"], stack["band_b"] = stack["band_b"], stack["band_a"]
    with pytest.raises(ValueError, match="must not be finer"):
        run_main_diff_estimation(**stack, options=_options())


def test_other_unwrappers_are_rejected(stack, unwrap_calls):
    assert not unwrap_calls
    stack["unwrap_options"] = UnwrapOptions(unwrap_method="spurt")
    with pytest.raises(ValueError, match="supports snaphu or whirlwind"):
        run_main_diff_estimation(**stack, options=_options())


def test_a_reference_in_a_gap_stops_the_export(stack, unwrap_calls):
    """No correction can be referenced there, so none may be published."""
    assert not unwrap_calls
    options = _options(
        filtering=IonosphereFilterOptions(
            sigma_m=(120.0, 120.0), donor_erosion_px=(0, 0), restore_mask=True
        ),
    )
    similarity = stack["band_a"].similarity
    values = np.full(SHAPE_A, 0.9, "f4")
    values[50:70, 70:90] = 0.0
    _write(similarity, values, TRANSFORM_A)
    with pytest.raises(ValueError, match="no supported ionosphere estimate"):
        run_main_diff_estimation(**stack, options=options)
