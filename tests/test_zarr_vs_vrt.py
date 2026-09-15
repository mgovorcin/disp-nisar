"""Benchmark: ZarrStack vs VRTStack read performance on real NISAR GSLC data.

Compares I/O throughput for sequential, random, and repeated block access
patterns. All tests are skipped when the real data files are not present.

Run with output:
    pytest tests/test_zarr_vs_vrt.py -v -s
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths — adjust if your data lives elsewhere
# ---------------------------------------------------------------------------
ZARR_STORE = Path(
    "/u/aurora-r0/govorcin/03_DISP_NI/NISAR-DB/DISP/workflow/062_A_017/slc_stack.zarr"
)
GSLC_DIR = Path(
    "/u/aurora-r0/govorcin/03_DISP_NI/NISAR-DB/GSLC_validation/data/T062_017/GSLC"
)
SUBDATASET = "/science/LSAR/GSLC/grids/frequencyA/HH"

DATA_AVAILABLE = ZARR_STORE.exists() and GSLC_DIR.exists()
pytestmark = pytest.mark.skipif(
    not DATA_AVAILABLE, reason="Real NISAR GSLC data not available"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def zarr_stack():
    from disp_nisar._zarr import ZarrStack

    return ZarrStack(ZARR_STORE, freq="A", pol="HH")


@pytest.fixture(scope="module")
def vrt_stack(tmp_path_factory):
    from dolphin.io import VRTStack

    files = sorted(GSLC_DIR.glob("NISAR_L2_PR_GSLC_*.h5"))
    if not files:
        pytest.skip("No GSLC HDF5 files found")
    tmp = tmp_path_factory.mktemp("vrt")
    return VRTStack(
        file_list=files,
        outfile=tmp / "slc_stack.vrt",
        subdataset=SUBDATASET,
        sort_files=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_blocks(reader, blocks: list[tuple[int, int, int, int]]) -> float:
    """Read all blocks and return wall-clock seconds."""
    t0 = time.perf_counter()
    for r0, r1, c0, c1 in blocks:
        _ = reader[:, r0:r1, c0:c1]
    return time.perf_counter() - t0


def _benchmark(
    reader,
    blocks: list[tuple[int, int, int, int]],
    n_warmup: int = 1,
    n_repeat: int = 5,
) -> tuple[float, float]:
    """Return (mean_s_per_block, min_s_per_block) after warmup reads."""
    for _ in range(n_warmup):
        _read_blocks(reader, blocks)
    times = [_read_blocks(reader, blocks) / len(blocks) for _ in range(n_repeat)]
    return float(np.mean(times)), float(np.min(times))


def _report(label: str, zarr_mean: float, vrt_mean: float) -> None:
    ratio = vrt_mean / zarr_mean if zarr_mean > 0 else float("inf")
    winner = "zarr" if zarr_mean < vrt_mean else "VRT "
    print(
        f"\n{label}\n"
        f"  ZarrStack: {zarr_mean * 1000:7.1f} ms/block\n"
        f"  VRTStack:  {vrt_mean  * 1000:7.1f} ms/block\n"
        f"  VRT/zarr = {ratio:.2f}x   winner: {winner}"
    )


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def test_correctness(zarr_stack, vrt_stack):
    """ZarrStack and VRTStack return the same values for valid (non-nodata) pixels."""
    row, col, size = 10_000, 20_000, 512
    zarr_data = zarr_stack[:, row : row + size, col : col + size]
    vrt_data = vrt_stack[:, row : row + size, col : col + size]

    assert zarr_data.shape == vrt_data.shape, "Shape mismatch"

    # ZarrStack zeroes nodata pixels (mask != 1); compare only where zarr is non-zero
    valid = np.abs(zarr_data) > 0
    assert valid.any(), "No valid pixels in test block — choose a different block"

    np.testing.assert_allclose(
        zarr_data[valid],
        vrt_data[valid].astype(zarr_data.dtype),
        rtol=1e-5,
        atol=0,
        err_msg="ZarrStack and VRTStack disagree on valid pixels",
    )


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [256, 512, 1024])
def test_benchmark_single_block(zarr_stack, vrt_stack, block_size, capsys):
    """Single repeated block — measures decompression / decode throughput."""
    row, col = 10_000, 20_000
    blocks = [(row, row + block_size, col, col + block_size)]
    px = block_size**2

    zarr_mean, _ = _benchmark(zarr_stack, blocks, n_warmup=2, n_repeat=10)
    vrt_mean, _ = _benchmark(vrt_stack, blocks, n_warmup=2, n_repeat=10)

    with capsys.disabled():
        _report(
            f"Single block {block_size}×{block_size}  ({px / 1e6:.2f} M px)",
            zarr_mean,
            vrt_mean,
        )
        print(
            f"  Zarr throughput: {px / zarr_mean / 1e6:.0f} M px/s\n"
            f"  VRT  throughput: {px / vrt_mean  / 1e6:.0f} M px/s"
        )


def test_benchmark_sequential_scan(zarr_stack, vrt_stack, capsys):
    """Sequential column scan — 20 adjacent 512×512 blocks (row-contiguous)."""
    size = 512
    col = 20_000
    blocks = [(r, r + size, col, col + size) for r in range(0, 20 * size, size)]

    zarr_mean, _ = _benchmark(zarr_stack, blocks, n_warmup=1, n_repeat=3)
    vrt_mean, _ = _benchmark(vrt_stack, blocks, n_warmup=1, n_repeat=3)

    with capsys.disabled():
        _report(
            f"Sequential scan  ({len(blocks)} × {size}×{size})", zarr_mean, vrt_mean
        )


def test_benchmark_random_access(zarr_stack, vrt_stack, capsys):
    """Random spatial access — 20 non-contiguous 512×512 blocks."""
    rng = np.random.default_rng(42)
    size = 512
    ny, nx = zarr_stack.shape[1], zarr_stack.shape[2]
    rows = rng.integers(0, ny - size, size=20)
    cols = rng.integers(0, nx - size, size=20)
    blocks = [
        (int(r), int(r) + size, int(c), int(c) + size) for r, c in zip(rows, cols)
    ]

    zarr_mean, _ = _benchmark(zarr_stack, blocks, n_warmup=1, n_repeat=3)
    vrt_mean, _ = _benchmark(vrt_stack, blocks, n_warmup=1, n_repeat=3)

    with capsys.disabled():
        _report(f"Random access  ({len(blocks)} × {size}×{size})", zarr_mean, vrt_mean)


def test_benchmark_cache_cold_vs_warm(zarr_stack, vrt_stack, capsys):
    """Compare first-read (cold) vs repeated-read (warm) latency."""
    size = 512
    # Use a rarely-touched region near the centre of the raster
    ny, nx = zarr_stack.shape[1], zarr_stack.shape[2]
    row, col = ny // 2, nx // 2
    blocks = [(row, row + size, col, col + size)]

    # cold: no warmup, single read
    zarr_cold = _read_blocks(zarr_stack, blocks)
    vrt_cold = _read_blocks(vrt_stack, blocks)

    # warm: after cache is primed
    zarr_warm, _ = _benchmark(zarr_stack, blocks, n_warmup=3, n_repeat=5)
    vrt_warm, _ = _benchmark(vrt_stack, blocks, n_warmup=3, n_repeat=5)

    with capsys.disabled():
        print(
            f"\nCold vs warm  ({size}×{size}):\n"
            f"  Zarr  cold={zarr_cold * 1000:.1f}ms  warm={zarr_warm * 1000:.1f}ms\n"
            f"  VRT   cold={vrt_cold  * 1000:.1f}ms  warm={vrt_warm  * 1000:.1f}ms"
        )
