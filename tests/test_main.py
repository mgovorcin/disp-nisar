from contextlib import chdir
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from dolphin import io

from disp_nisar import main
from disp_nisar.cli.run import run_main
from disp_nisar.ionosphere.options import IonosphereOptions

TEST_DATA_DIR = Path(__file__).parent / "data/delivery_data_small"


def test_run_main():
    config_file = TEST_DATA_DIR / "config_files/runconfig_historical.yaml"
    if not config_file.exists():
        pytest.skip(f"Test data does not exist at {TEST_DATA_DIR}")

    with chdir(TEST_DATA_DIR):
        run_main(config_file="config_files/runconfig_historical.yaml")


def _main_diff_inputs(
    method="main_diff",
    product_type="DISP_NISAR_HISTORICAL",
    gunw_files=None,
    iono_parameters=Path("algorithm_parameters_freqB.yaml"),
    frequency="frequencyA",
    gslc_name="NISAR_L2_GSLC_20260101.h5",
):
    pge_runconfig = SimpleNamespace(
        get_ionosphere_options=lambda: IonosphereOptions(method=method),
        dynamic_ancillary_file_group=SimpleNamespace(
            gunw_files=gunw_files,
            ionosphere_algorithm_parameters_file=iono_parameters,
        ),
        input_file_group=SimpleNamespace(frequency=frequency),
        primary_executable=SimpleNamespace(product_type=product_type),
    )
    return SimpleNamespace(cslc_file_list=[Path(gslc_name)]), pge_runconfig


@pytest.mark.parametrize(
    "kwargs",
    [
        # Each of these ran the split-spectrum workflow before main_diff was added
        {"method": "main_side"},
        {"method": "main_side", "product_type": "DISP_NISAR_FORWARD"},
        {"method": "main_side", "iono_parameters": None},
        # GUNW inputs always win, whatever the method
        {"gunw_files": [Path("gunw.h5")]},
    ],
)
def test_main_diff_is_opt_in(kwargs):
    cfg, pge_runconfig = _main_diff_inputs(**kwargs)
    assert main._get_main_diff_options(cfg, pge_runconfig) is None


def test_default_method_keeps_the_split_spectrum_workflow():
    assert IonosphereOptions().method == "main_side"


def test_main_diff_selected():
    cfg, pge_runconfig = _main_diff_inputs()
    assert main._get_main_diff_options(cfg, pge_runconfig).method == "main_diff"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frequency": "frequencyB"}, "nominal frequencyA"),
        ({"product_type": "DISP_NISAR_FORWARD"}, "historical GSLC stacks only"),
        ({"gslc_name": "compressed_20260101.h5"}, "historical GSLC stacks only"),
        ({"iono_parameters": None}, "explicit frequencyB algorithm file"),
    ],
)
def test_main_diff_rejects_unsupported_runs(kwargs, message):
    cfg, pge_runconfig = _main_diff_inputs(**kwargs)
    with pytest.raises(ValueError, match=message):
        main._get_main_diff_options(cfg, pge_runconfig)


def test_nodata_mask_keeps_valid_zeros(tmp_path):
    """The reference pixel of a displacement raster is a valid zero."""
    image = tmp_path / "20260101_20260113.tif"
    data = np.array([[0.0, 1.0], [np.nan, -2.0]], dtype=np.float32)
    io.write_arr(arr=data, output_name=image, nodata=np.nan)

    output = tmp_path / "mask.tif"
    main._create_nodata_mask(image, output)

    np.testing.assert_array_equal(io.load_gdal(output), [[1, 1], [0, 1]])
