"""main_diff pair loop: the n_workers pool keeps input order and propagates failures."""

from pathlib import Path

import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from disp_nisar.ionosphere._main_diff_io import Grid
from disp_nisar.ionosphere.main_diff import _process_pairs
from disp_nisar.ionosphere.options import IonosphereOptions

GRID = Grid(8, 8, CRS.from_epsg(32610), from_origin(500000, 4000000, 40, 40))


def _ready_tasks(tmp_path: Path, n: int):
    tasks: list[tuple] = []
    for k in range(n):
        folder = tmp_path / "pairs" / f"p{k:02d}"
        folder.mkdir(parents=True)
        for f in ("A.unw.tif", "D.unw.tif", "donor.tif"):
            (folder / f).write_bytes(b"")
        tasks.append((f"p{k:02d}", {}, {}, folder))
    return tasks


@pytest.mark.parametrize("workers", [1, 3])
def test_process_pairs_keeps_input_order(tmp_path, workers):
    tasks = _ready_tasks(tmp_path, 7)
    jobs = _process_pairs(
        tasks, GRID, None, IonosphereOptions(method="main_diff", n_workers=workers)
    )
    assert [j["D_unw"].parent.name for j in jobs] == [t[0] for t in tasks]


def test_process_pairs_propagates_worker_failure(tmp_path):
    tasks = _ready_tasks(tmp_path, 3)
    broken = tmp_path / "pairs" / "broken"
    tasks.insert(1, ("broken", {"ifg_A": tmp_path / "missing.tif"}, {}, broken))
    with pytest.raises(Exception):
        _process_pairs(
            tasks, GRID, None, IonosphereOptions(method="main_diff", n_workers=2)
        )
