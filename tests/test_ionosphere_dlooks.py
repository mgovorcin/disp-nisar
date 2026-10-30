"""Coarse unwrapping of the double difference."""

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from disp_nisar.ionosphere._main_diff_io import Grid
from disp_nisar.ionosphere.dlooks import (
    coarse_grid,
    reduce_pair_d,
    upsample_unwrapped,
)

FINE = Grid(40, 48, CRS.from_epsg(32610), Affine(40, 0, 500000, 0, -40, 4000000))


def test_coarse_grid_keeps_the_origin_and_rounds_up():
    coarse = coarse_grid(Grid(41, 48, FINE.crs, FINE.transform), 4)
    assert (coarse.height, coarse.width) == (11, 12)
    assert coarse.transform * (0, 0) == FINE.transform * (0, 0)
    assert coarse.transform.a == 160


def _prepared(donor):
    rows, cols = np.mgrid[: donor.shape[0], : donor.shape[1]]
    phase = 0.01 * cols + 0.02 * rows
    # Placeholders exactly as pair preparation writes them
    z = np.where(donor, np.exp(1j * phase), 1 + 0j).astype(np.complex64)
    cor = np.full(donor.shape, 0.8, np.float32)
    sim = np.full(donor.shape, 0.7, np.float32)
    return phase, z, cor, sim


def test_reduction_ignores_placeholders_and_thresholds_donors():
    donor = np.ones((40, 48), bool)
    donor[:, :8] = False  # two coarse columns with no donors
    donor[::2, 8:12] = False  # one coarse column with half donors
    phase, z, cor, sim = _prepared(donor)

    ifg, coarse_cor, coarse_sim, coarse_donor, _ = reduce_pair_d(
        z, donor, cor, sim, np.ones(donor.shape, bool), 4, min_fraction=0.6
    )

    assert ifg.shape == (10, 12)
    assert not coarse_donor[:, :3].any()
    assert coarse_donor[:, 3:].all()
    assert np.all(ifg[:, :2] == 1 + 0j)
    # The mean phase of the donors, not pulled toward zero by placeholders
    expected = np.angle(np.exp(1j * phase[:4, 12:16]).mean())
    assert np.angle(ifg[0, 3]) == pytest.approx(expected, abs=1e-5)
    assert np.all(coarse_cor[coarse_donor] > 0)
    assert np.all(coarse_cor[~coarse_donor] == 0)
    np.testing.assert_allclose(coarse_sim[coarse_donor], 0.7)


def test_partial_edge_blocks_use_only_their_pixels():
    donor = np.ones((42, 49), bool)
    _, z, cor, sim = _prepared(donor)

    ifg, _, _, coarse_donor, inside = reduce_pair_d(
        z, donor, cor, sim, donor, 4, min_fraction=0.2
    )

    assert ifg.shape == (11, 13)
    assert inside.all()
    # 2 of the 16 pixels is below the 20 % floor, 8 is above it
    assert not coarse_donor[-1, -1]
    assert coarse_donor[-1, 0]
    assert np.isfinite(ifg).all()


def test_block_outside_the_footprint_is_zero():
    donor = np.zeros((40, 48), bool)
    _, z, cor, sim = _prepared(donor)
    ifg, *_ = reduce_pair_d(z, donor, cor, sim, donor, 4, min_fraction=0.1)
    assert np.all(ifg == 0)


def test_upsampling_round_trips_a_smooth_field():
    coarse = coarse_grid(FINE, 4)
    rows, cols = np.mgrid[:10, :12]
    unw = (0.5 * cols + 0.2 * rows).astype(np.float32)

    fine = upsample_unwrapped(unw, np.ones(unw.shape, bool), coarse, FINE)

    rows, cols = np.mgrid[:40, :48]
    truth = 0.5 * ((cols + 0.5) / 4 - 0.5) + 0.2 * ((rows + 0.5) / 4 - 0.5)
    interior = (slice(4, 32), slice(4, 40))
    assert np.abs(fine - truth)[interior].max() < 0.05


def test_upsampling_fills_small_gaps_only():
    coarse = coarse_grid(FINE, 4)
    unw = np.ones((10, 12), np.float32)
    unw[:, :6] = np.nan
    inside = np.ones(unw.shape, bool)

    fine = upsample_unwrapped(unw, inside, coarse, FINE, max_fill_px=2)

    assert np.isnan(fine[:, :8]).all()  # beyond the limit nothing is invented
    assert np.isfinite(fine[:, 20:]).all()


def test_upsampling_stops_without_any_phase():
    unw = np.full((10, 12), np.nan, np.float32)
    with pytest.raises(ValueError, match="no finite phase"):
        upsample_unwrapped(unw, np.ones(unw.shape, bool), coarse_grid(FINE, 4), FINE)
