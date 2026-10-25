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
