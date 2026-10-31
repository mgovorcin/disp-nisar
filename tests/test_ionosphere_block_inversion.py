"""invert_common_phase_block equals a per-pixel least-squares solve."""

from datetime import datetime, timedelta

import numpy as np

from disp_nisar.ionosphere.inversion import (
    build_design_matrix,
    invert_common_phase_block,
)


def test_block_inversion_matches_per_pixel_lstsq_with_mixed_patterns():
    rng = np.random.default_rng(7)
    dates = [datetime(2026, 1, 1) + timedelta(days=12 * k) for k in range(8)]
    pairs = [
        (dates[i], dates[j]) for i in range(8) for j in range(i + 1, min(i + 4, 8))
    ]
    G = build_design_matrix(pairs, dates)
    n, rows, cols = len(pairs), 9, 11
    truth = rng.normal(size=(G.shape[1], rows * cols))
    a = (G @ truth + 0.01 * rng.normal(size=(n, rows * cols))).reshape(n, rows, cols)
    d = (G @ (0.1 * truth)).reshape(n, rows, cols)
    valid = (
        rng.random((n, rows, cols)) > 0.15
    )  # many distinct patterns, some rank-deficient
    out_a, out_d = invert_common_phase_block(a, d, G, valid)
    fa, fd, fv = a.reshape(n, -1), d.reshape(n, -1), valid.reshape(n, -1)
    for p in range(rows * cols):
        use = fv[:, p]
        sub = G[use]
        r, c = divmod(p, cols)
        if sub.shape[0] < G.shape[1] or np.linalg.matrix_rank(sub) < G.shape[1]:
            assert np.isnan(out_a[:, r, c]).all()
            continue
        ref_a = np.linalg.lstsq(sub, fa[use, p], rcond=None)[0]
        ref_d = np.linalg.lstsq(sub, fd[use, p], rcond=None)[0]
        np.testing.assert_allclose(out_a[:, r, c], ref_a, atol=1e-5)
        np.testing.assert_allclose(out_d[:, r, c], ref_d, atol=1e-5)


def test_invert_jobs_parallel_windows_equal_sequential(tmp_path):
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    from disp_nisar.ionosphere._main_diff_io import Grid
    from disp_nisar.ionosphere.main_diff import _invert_jobs

    rng = np.random.default_rng(11)
    dates = [datetime(2026, 1, 1) + timedelta(days=12 * k) for k in range(6)]
    pairs = [
        (dates[i], dates[j]) for i in range(6) for j in range(i + 1, min(i + 3, 6))
    ]
    grid = Grid(40, 52, CRS.from_epsg(32610), from_origin(500000, 4000000, 40, 40))
    prof = {
        "driver": "GTiff",
        "height": 40,
        "width": 52,
        "count": 1,
        "crs": grid.crs,
        "transform": grid.transform,
    }
    truth = rng.normal(size=(len(dates), 40, 52)).astype("float32")
    jobs = []
    for k, (r, s) in enumerate(pairs):
        i, j = dates.index(r), dates.index(s)
        paths = {n: tmp_path / f"p{k}_{n}.tif" for n in ("donor", "A_unw", "D_unw")}
        with rasterio.open(
            paths["A_unw"], "w", dtype="float32", nodata=np.nan, **prof
        ) as dst:
            dst.write(truth[j] - truth[i], 1)
        with rasterio.open(
            paths["D_unw"], "w", dtype="float32", nodata=np.nan, **prof
        ) as dst:
            dst.write(0.1 * (truth[j] - truth[i]), 1)
        with rasterio.open(paths["donor"], "w", dtype="uint8", **prof) as dst:
            dst.write((rng.random((40, 52)) > 0.1).astype("uint8"), 1)
        jobs.append(paths)
    out_pairs = [(dates[0], d) for d in dates[1:]]
    res = {}
    for workers in (1, 3):
        outs = [tmp_path / f"w{workers}_{k}.tif" for k in range(len(out_pairs))]
        _invert_jobs(
            jobs,
            pairs,
            grid,
            (20, 26),
            1.239e9,
            1.2935e9,
            outs,
            out_pairs,
            16,
            n_workers=workers,
        )
        res[workers] = np.stack([rasterio.open(p).read(1) for p in outs])
    np.testing.assert_array_equal(res[1], res[3])
