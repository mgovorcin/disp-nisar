"""Robust smoothing of the ionospheric screen."""

import numpy as np
import pytest

from disp_nisar.ionosphere.ionosphere_filter import apply_ionosphere_filter
from disp_nisar.ionosphere.options import IonosphereFilterOptions, IonosphereOptions
from disp_nisar.ionosphere.smoothing import (
    background_clip,
    empirical_noise,
    near_range_band,
    noise_weights,
    normalized_convolution,
    smooth_screen,
)


def test_convolution_recovers_a_constant_and_reports_support():
    x = np.full((120, 120), 3.0)
    x[:, :40] = np.nan
    out, support = normalized_convolution(x, np.ones_like(x), 5.0)
    assert np.allclose(out[:, 60:], 3.0, atol=1e-6)
    assert np.isnan(out[:, :15]).all()  # beyond 4 sigma the kernel has no mass on data
    assert support[60, 100] > 0.99
    assert support[60, 40] < 0.6


def test_support_floor_falls_back_then_gives_nan():
    _, cols = np.mgrid[:200, :200]
    x = 0.01 * cols.astype(float)
    x[:, :120] = np.nan
    out, _ = normalized_convolution(
        x, np.ones_like(x), 10.0, min_support=0.3, fallback_sigma_px=40.0
    )
    assert np.isfinite(out[100, 150])
    assert np.isnan(out[100, 5])  # even the coarse kernel has no support
    # 20 px into the gap the coarse kernel gives a mean of the data, no extrapolation
    assert 1.3 < out[100, 100] < 1.6


def test_support_floor_without_a_fallback_gives_nan():
    x = np.full((100, 100), 1.0)
    x[:, :50] = np.nan
    out, _ = normalized_convolution(x, np.ones_like(x), 5.0, min_support=0.3)
    assert np.isnan(out[50, 45])
    assert out[50, 60] == pytest.approx(1.0)


def test_background_clip_drops_a_bullseye_seed_only():
    x = 0.05 * np.random.default_rng(0).standard_normal((300, 300))
    x[150:153, 150:153] = 6.0
    kept = background_clip(x, np.ones(x.shape, bool), block_px=30, max_dev=1.0)
    assert not kept[150:153, 150:153].any()
    assert kept.mean() > 0.97


def test_noise_finds_a_noisy_strip_and_weights_it_down():
    rng = np.random.default_rng(1)
    shape = (200, 200)
    sigma = np.full(shape, 0.1)
    sigma[:, :50] = 0.5
    epochs = [sigma * rng.standard_normal(shape) for _ in range(8)]

    noise = empirical_noise(epochs, np.ones(shape, bool), sigma_px=40.0, reduce=5)

    # The noise of the 5 x 5 block mean: pixel noise / 5
    assert abs(np.nanmedian(noise[:, 100:]) - 0.1 / 5) < 0.006
    assert abs(np.nanmedian(noise[:, :45]) - 0.5 / 5) < 0.03
    weights = noise_weights(noise)
    assert np.median(weights[:, :45]) < 0.1
    assert 0.5 < np.median(weights[:, 100:]) < 2.0


def test_noise_needs_enough_epochs():
    epochs = [np.ones((50, 50))] * 3
    noise = empirical_noise(epochs, np.ones((50, 50), bool), sigma_px=10.0)
    assert np.isnan(noise).all()


def test_noise_needs_an_epoch():
    with pytest.raises(ValueError, match="at least one epoch"):
        empirical_noise([], np.ones((5, 5), bool), sigma_px=10.0)


def test_screen_is_not_extrapolated_beyond_the_coarse_kernel():
    rows, cols = np.mgrid[:300, :300]
    truth = 0.002 * cols + 0.001 * rows
    phase = truth + 0.05 * np.random.default_rng(2).standard_normal(truth.shape)
    trusted = cols > 150

    out, sources = smooth_screen(
        phase,
        trusted,
        np.ones(phase.shape, bool),
        None,
        sigma_px=10.0,
        min_support=0.3,
        fallback_sigma_px=30.0,
    )

    assert np.isnan(out[150, 20])
    assert np.isfinite(out[150, 140])
    assert np.nanmax(np.abs(out[:, 200:] - truth[:, 200:])) < 0.05
    assert sources.sum() == trusted.sum()


def test_noise_exclusion_removes_an_edge_bias_that_weights_cannot():
    rows, cols = np.mgrid[:300, :300]
    truth = 0.001 * rows
    rng = np.random.default_rng(3)
    phase = truth + 0.02 * rng.standard_normal(truth.shape)
    strip = cols < 30
    phase[strip] += 0.5
    noise = np.where(strip, 0.08, 0.02)
    everywhere = np.ones(phase.shape, bool)
    kernels = {"sigma_px": 10.0, "min_support": 0.2, "fallback_sigma_px": 40.0}
    weights = noise_weights(noise)

    weighted, _ = smooth_screen(phase, everywhere, everywhere, weights, **kernels)
    excluded, sources = smooth_screen(
        phase,
        everywhere,
        everywhere,
        weights,
        noise=noise,
        exclude_noise_factor=3.0,
        **kernels,
    )

    edge = cols < 10
    assert np.nanmean(np.abs(weighted[edge] - truth[edge])) > 0.2
    assert np.nanmean(np.abs(excluded[edge] - truth[edge])) < 0.05
    assert not sources[strip].any()
    assert sources[~strip].all()


def _near_range():
    rows, cols = np.mgrid[:200, :300]
    footprint = (cols > 20) & (cols < 280)
    # Incidence rises with column, so near range is the left edge
    return rows, cols, footprint, 33.0 + 10.0 * (cols - 20) / 260.0


def test_near_range_band_covers_the_near_edge_only():
    _, _, footprint, incidence = _near_range()
    band = near_range_band(footprint, incidence, pixel_m=100.0, km=6.0)
    assert band[:, 21:80].all()
    assert not band[:, 110:].any()


def test_near_range_band_is_filled_from_the_interior():
    rows, _, footprint, incidence = _near_range()
    band = near_range_band(footprint, incidence, pixel_m=100.0, km=6.0)
    truth = 0.001 * rows
    phase = truth + np.where(band, 0.5, 0.0)

    out, sources = smooth_screen(
        phase,
        footprint,
        footprint,
        None,
        sigma_px=10.0,
        min_support=0.2,
        fallback_sigma_px=40.0,
        exclude_mask=band,
    )

    assert not sources[band].any()
    assert np.nanmean(np.abs(out[band] - truth[band])) < 0.05


def _robust(**kwargs):
    return IonosphereOptions(
        method="main_diff",
        filtering=IonosphereFilterOptions(
            method="robust",
            robust_fallback_sigma_m=2000.0,
            robust_clip_block_m=2000.0,
            **kwargs,
        ),
    )


def _filter(phase, options, **kwargs):
    everywhere = np.ones(phase.shape, bool)
    kwargs.setdefault("donor_mask", np.isfinite(phase))
    return apply_ionosphere_filter(
        phase,
        footprint=everywhere,
        pixel_size_m=(40.0, 40.0),
        options=options,
        **kwargs,
    )


def test_dispatch_recovers_a_plane():
    rows, cols = np.mgrid[:200, :200]
    plane = 0.002 * rows - 0.001 * cols
    out = _filter(plane, _robust(sigma_m=(200.0, 200.0)))
    np.testing.assert_allclose(out[40:-40, 40:-40], plane[40:-40, 40:-40], atol=1e-3)


def test_dispatch_defaults_to_a_2_km_kernel():
    rng = np.random.default_rng(4)
    noisy = rng.standard_normal((200, 200)) * 0.1
    default = _filter(noisy, _robust())
    explicit = _filter(noisy, _robust(sigma_m=(2000.0, 2000.0)))
    fine = _filter(noisy, _robust(sigma_m=(100.0, 100.0)))
    np.testing.assert_array_equal(default, explicit)
    assert not np.allclose(default, fine, equal_nan=True)


def test_dispatch_applies_the_exclusion_mask():
    rows, cols = np.mgrid[:200, :200]
    phase = np.full((200, 200), 1.0)
    band = cols < 20
    phase[band] = 5.0
    out = _filter(phase, _robust(sigma_m=(100.0, 100.0)), exclude_mask=band)
    np.testing.assert_allclose(out[:, 30:], 1.0, atol=1e-5)
    assert np.nanmax(out) < 1.1


def test_dispatch_tolerates_a_noise_map_without_values():
    plane = np.full((100, 100), 2.0)
    unknown = np.full(plane.shape, np.nan, np.float32)
    out = _filter(plane, _robust(sigma_m=(100.0, 100.0)), noise=unknown)
    np.testing.assert_allclose(out[20:-20, 20:-20], 2.0, atol=1e-5)


def test_dispatch_without_donors_gives_nan():
    plane = np.full((100, 100), np.nan)
    assert np.isnan(_filter(plane, _robust())).all()


def test_dispatch_excludes_pixels_by_their_noise():
    rows, cols = np.mgrid[:200, :200]
    phase = 0.0 * rows + 1.0
    strip = cols < 20
    phase[strip] = 3.0
    noise = np.where(strip, 0.5, 0.05).astype(np.float32)
    options = _robust(sigma_m=(100.0, 100.0), robust_clip_cycles=None)

    excluded = _filter(phase, options, noise=noise)
    kept = _filter(phase, options)

    assert np.nanmax(excluded) < 1.1
    assert np.nanmax(kept) > 2.0
