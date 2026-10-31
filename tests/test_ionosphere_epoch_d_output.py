"""_invert_jobs writes the per-epoch inverted D series when d_outputs is given."""

from datetime import datetime, timedelta

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from disp_nisar.ionosphere._main_diff_io import Grid
from disp_nisar.ionosphere.main_diff import _invert_jobs


def test_invert_jobs_writes_the_per_epoch_d_series(tmp_path):
    rng = np.random.default_rng(5)
    dates = [datetime(2026, 1, 1) + timedelta(days=12 * k) for k in range(6)]
    pairs = [
        (dates[i], dates[j]) for i in range(6) for j in range(i + 1, min(i + 3, 6))
    ]
    grid = Grid(30, 40, CRS.from_epsg(32610), from_origin(500000, 4000000, 40, 40))
    prof = {
        "driver": "GTiff",
        "height": 30,
        "width": 40,
        "count": 1,
        "crs": grid.crs,
        "transform": grid.transform,
    }
    a_true = rng.normal(size=(len(dates), 30, 40)).astype("float32")
    d_true = 0.1 * rng.normal(size=(len(dates), 30, 40)).astype("float32")
    jobs = []
    for k, (r, s) in enumerate(pairs):
        i, j = dates.index(r), dates.index(s)
        paths = {n: tmp_path / f"p{k}_{n}.tif" for n in ("donor", "A_unw", "D_unw")}
        for name, field in (("A_unw", a_true), ("D_unw", d_true)):
            with rasterio.open(
                paths[name], "w", dtype="float32", nodata=np.nan, **prof
            ) as dst:
                dst.write(field[j] - field[i], 1)
        with rasterio.open(paths["donor"], "w", dtype="uint8", **prof) as dst:
            dst.write(np.ones((30, 40), "uint8"), 1)
        jobs.append(paths)
    ref = (15, 20)
    out_pairs = [(dates[0], d) for d in dates[1:]]
    outs = [tmp_path / f"iono_{k}.tif" for k in range(len(out_pairs))]
    d_outs = [tmp_path / f"d_{k}.tif" for k in range(len(out_pairs))]
    _invert_jobs(
        jobs, pairs, grid, ref, 1.239e9, 1.2935e9, outs, out_pairs, 16, d_outputs=d_outs
    )
    for k, d in enumerate(dates[1:]):
        expected = d_true[k + 1] - d_true[0]
        expected = expected - expected[ref]
        with rasterio.open(d_outs[k]) as src:
            np.testing.assert_allclose(src.read(1), expected, atol=1e-5)
