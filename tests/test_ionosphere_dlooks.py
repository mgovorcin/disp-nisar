"""Coarse-D unwrapping helpers (disp_nisar.ionosphere.dlooks)."""

import numpy as np
import rasterio
from rasterio.transform import from_origin

from disp_nisar.ionosphere._main_diff_io import Grid
from disp_nisar.ionosphere.dlooks import (
    block_sum,
    coarse_grid,
    reduce_pair_d,
    upsample_unwrapped,
)


def _grid(h=40, w=48, res=40.0):
    return Grid(
        h,
        w,
        rasterio.crs.CRS.from_epsg(32610),
        from_origin(500000.0, 4000000.0, res, res),
    )


def test_block_sum_pads_partial_blocks():
    x = np.ones((10, 11))
    s = block_sum(x, 4)
    assert s.shape == (3, 3)
    assert s[0, 0] == 16 and s[2, 2] == 2 * 3  # last block: 2 rows x 3 cols


def test_reduce_pair_d_ignores_placeholders_and_thresholds_donors():
    h, w, k = 40, 48, 4
    yy, xx = np.mgrid[0:h, 0:w]
    phase = 0.01 * xx + 0.02 * yy
    z = np.exp(1j * phase).astype(np.complex64)
    donor = np.ones((h, w), bool)
    donor[:, :8] = False  # two coarse columns with no donors at all
    donor[::2, 8:12] = False  # one coarse column with half donors
    inside = np.ones((h, w), bool)
    z = np.where(donor, z, 1 + 0j)  # placeholders exactly as _prepare_pair writes them
    cor = np.full((h, w), 0.8, np.float32)
    sim = np.full((h, w), 0.7, np.float32)
    ifg, corc, simc, donorc, insidec, n = reduce_pair_d(
        z, donor, cor, sim, inside, k, min_fraction=0.6
    )
    assert ifg.shape == (10, 12)
    assert not donorc[:, :2].any() and np.all(
        ifg[:, :2] == 1 + 0j
    )  # inside, no donors -> placeholder
    assert not donorc[:, 2].any()  # half the pixels < min_fraction 0.6 -> not a donor
    assert donorc[:, 3:].all()
    # the coarse phase equals the mean phase of the donors, not pulled toward 0 by
    # placeholders
    blk = phase[0:4, 12:16]
    assert abs(np.angle(ifg[0, 3]) - np.angle(np.exp(1j * blk).mean())) < 1e-5
    assert np.all(corc[donorc] > 0) and np.all(corc[~donorc] == 0)
    assert np.all(simc[donorc] == np.float32(0.7))
    assert n[0, 3] == 16


def test_upsample_round_trips_a_smooth_field_and_limits_fill():
    g = _grid(40, 48)
    k = 4
    cg = coarse_grid(g, k)
    assert (cg.height, cg.width) == (10, 12)
    cy, cx = np.mgrid[0:10, 0:12]
    unw_c = (0.5 * cx + 0.2 * cy).astype(np.float32)
    cc_c = np.ones((10, 12), np.uint16)
    insidec = np.ones((10, 12), bool)
    unw_c[:, 0:2] = np.nan  # two coarse columns missing, within the fill limit
    unw_c[9, 11] = np.nan
    unw, cc = upsample_unwrapped(unw_c, cc_c, insidec, cg, g, max_fill_px=2)
    assert unw.shape == (40, 48) and cc.dtype == np.uint16
    fy, fx = np.mgrid[0:40, 0:48]
    truth = 0.5 * (fx + 0.5) / k - 0.5 * 0.5 + 0.2 * (fy + 0.5) / k - 0.2 * 0.5
    interior = (fx > 12) & (fx < 40) & (fy > 4) & (fy < 32)
    assert np.nanmax(np.abs(unw - truth)[interior]) < 0.05
    assert np.isfinite(unw[:, 0:8]).all()  # filled from the nearest coarse column
    # beyond the fill limit nothing is invented
    unw_c2 = unw_c.copy()
    unw_c2[:, 0:6] = np.nan
    unw2, _ = upsample_unwrapped(unw_c2, cc_c, insidec, cg, g, max_fill_px=2)
    assert np.isnan(unw2[:, 0:8]).all() and np.isfinite(unw2[:, 20:]).all()
