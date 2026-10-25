import numpy as np

from disp_nisar.ionosphere.smoothing import (
    background_clip,
    block_reduce_mean,
    empirical_noise,
    noise_weights,
    normalized_convolution,
    smooth_screen,
)


def test_nc_recovers_constant_and_reports_support():
    x = np.full((120, 120), 3.0)
    x[:, :40] = np.nan
    out, sup = normalized_convolution(x, np.ones_like(x), 5.0)
    assert np.allclose(out[:, 60:], 3.0, atol=1e-6)
    assert np.isnan(
        out[:, :15]
    ).all()  # beyond 4 sigma from any data the Gaussian mass is zero
    assert sup[60, 100] > 0.99 and sup[60, 40] < 0.6


def test_support_floor_falls_back_then_gives_nan():
    yy, xx = np.mgrid[0:200, 0:200]
    x = 0.01 * xx.astype(float)
    x[:, :120] = np.nan  # a 120 px gap on the left
    out, _ = normalized_convolution(
        x, np.ones_like(x), 10.0, min_support=0.3, fallback_sigma_px=40.0
    )
    assert np.isfinite(out[100, 150])
    assert np.isnan(out[100, 5])  # even the coarse kernel has no support there
    # 20 px into the gap: the fine kernel's support (~0.03) is below the floor, the
    # coarse one's (~0.3) is not, so the value is the coarse kernel's weighted mean of
    # the data (x between 120 and 200), i.e. a value inside the data range, not the ramp
    # extrapolated down to 1.0
    assert np.isfinite(out[100, 100]) and 1.3 < out[100, 100] < 1.6
    # at the data edge every symmetric kernel has support 0.5, so a floor of 0.5 would
    # fill nothing outside


def test_background_clip_drops_bullseye_seed_only():
    rng = np.random.default_rng(0)
    x = 0.05 * rng.standard_normal((300, 300))
    x[150:153, 150:153] = (
        6.0  # a few bad pixels: 1 cycle of D is 6.28, this is ~that in screen units
    )
    ok = np.ones(x.shape, bool)
    kept, bg = background_clip(x, ok, block_px=30, max_dev=1.0)
    assert not kept[150:153, 150:153].any()
    assert kept.mean() > 0.97
    assert np.nanmax(np.abs(bg)) < 0.2


def test_empirical_noise_and_weights():
    rng = np.random.default_rng(1)
    shape = (200, 200)
    sig = np.full(shape, 0.1)
    sig[:, :50] = 0.5  # a noisy strip
    epochs = [sig * rng.standard_normal(shape) for _ in range(8)]
    ok = np.ones(shape, bool)
    noise = empirical_noise(epochs, ok, sigma_px=40.0, reduce=5)
    # the noise is that of the 5x5 block mean: pixel noise / 5
    assert abs(np.nanmedian(noise[:, 100:]) - 0.1 / 5) < 0.006
    assert abs(np.nanmedian(noise[:, :45]) - 0.5 / 5) < 0.03
    w = noise_weights(noise, ok)
    assert np.nanmedian(w[:, :45]) < 0.1 and 0.5 < np.nanmedian(w[:, 100:]) < 2.0


def test_smooth_screen_never_extrapolates_beyond_coarse_kernel():
    yy, xx = np.mgrid[0:300, 0:300]
    truth = 0.002 * xx + 0.001 * yy
    iono = truth + 0.05 * np.random.default_rng(2).standard_normal(truth.shape)
    footprint = np.ones(iono.shape, bool)
    trusted = xx > 150  # no freqB coverage on the left half
    out, qmask, sup = smooth_screen(
        iono,
        trusted,
        footprint,
        None,
        sigma_px=10.0,
        min_support=0.3,
        fallback_sigma_px=30.0,
    )
    assert np.isnan(out[150, 20])  # 130 px from data: not filled
    assert np.isfinite(out[150, 140])  # within the coarse kernel: filled from the right
    assert np.nanmax(np.abs(out[:, 200:] - truth[:, 200:])) < 0.05
    assert qmask.sum() == trusted.sum()


def test_block_reduce_mean_fraction():
    x = np.arange(16.0).reshape(4, 4)
    ok = np.ones((4, 4), bool)
    ok[0, 0] = False
    m, f = block_reduce_mean(x, ok, 2)
    assert m.shape == (2, 2) and f[0, 0] == 0.75 and f[1, 1] == 1.0
    assert m[0, 0] == (1 + 4 + 5) / 3


def test_noise_exclusion_removes_edge_bias_that_weights_cannot():
    """A biased, noisy strip at the footprint edge.

    With weights only, the fine kernel at the edge sees nothing but strip pixels and
    keeps the bias; with exclusion the coarse kernel fills the strip from the interior.
    """
    yy, xx = np.mgrid[0:300, 0:300]
    truth = 0.001 * yy
    rng = np.random.default_rng(3)
    iono = truth + 0.02 * rng.standard_normal(truth.shape)
    strip = xx < 30
    iono[strip] += (
        0.5
        + 0.08
        * rng.standard_normal(strip.sum())
        .reshape(-1)[: strip.sum()]
        .reshape(iono[strip].shape)
        if False
        else 0.5
    )
    noise = np.where(strip, 0.08, 0.02)
    foot = np.ones(iono.shape, bool)
    w = noise_weights(noise, foot)
    weighted, _, _ = smooth_screen(
        iono, foot, foot, w, sigma_px=10.0, min_support=0.2, fallback_sigma_px=40.0
    )
    excluded, qmask, _ = smooth_screen(
        iono,
        foot,
        foot,
        w,
        sigma_px=10.0,
        min_support=0.2,
        fallback_sigma_px=40.0,
        noise=noise,
        exclude_noise_factor=3.0,
    )
    edge = strip & (xx < 10)
    assert (
        np.nanmean(np.abs(weighted[edge] - truth[edge])) > 0.2
    )  # weights alone keep most of the 0.5 bias
    assert (
        np.nanmean(np.abs(excluded[edge] - truth[edge])) < 0.05
    )  # excluded: filled from the interior
    assert not qmask[strip].any() and qmask[~strip].all()


def test_near_range_band_and_exclusion():
    from disp_nisar.ionosphere.smoothing import near_range_band

    yy, xx = np.mgrid[0:200, 0:300]
    foot = (xx > 20) & (xx < 280)
    inc = (
        33.0 + 10.0 * (xx - 20) / 260.0
    )  # incidence rises with x: near range is the left edge
    band = near_range_band(foot, inc, pixel_m=100.0, km=6.0)
    assert band[:, 21:80].all()  # first 6 km on the near side
    assert not band[
        :, 110:
    ].any()  # far side and the interior untouched; along-track ends are not a near edge
    assert not band[100, 90:].any()
    truth = 0.001 * yy
    iono = truth + np.where(band, 0.5, 0.0)  # a biased near-range band
    out, qmask, _ = smooth_screen(
        iono,
        foot,
        foot,
        None,
        sigma_px=10.0,
        min_support=0.2,
        fallback_sigma_px=40.0,
        exclude_mask=band,
    )
    assert not qmask[band].any()
    assert (
        np.nanmean(np.abs(out[band] - truth[band])) < 0.05
    )  # filled from the interior, bias gone
