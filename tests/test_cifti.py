import shutil
import subprocess
import nibabel as nib
import numpy as np
import pytest
import importlib_resources
from comet import cifti, utils

# Parameter space
ATLASES = ["schaefer", "glasser", "gordon"]
RESOLUTIONS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
NETWORKS = [7, 17]
SUBCORTICALS = [None, "S1", "S2", "S3", "S4"]
KONG = [False, True]

def _wb_command():
    wb = shutil.which("wb_command")
    if wb is None:
        raise RuntimeError(
            "wb_command not found on PATH."
            "Please make sure Connectome Workbench is installed and wb_command is available."
        )
    return wb

def _atlas_filename(atlas: str, resolution: int, networks: int, subcortical, kong: bool) -> str:
    filenames = {
        "schaefer_c": "Schaefer2018_{parcels}Parcels_{kong}{networks}Networks_order.dlabel.nii",
        "schaefer":   "Schaefer2018_{parcels}Parcels_{networks}Networks_order_Tian_Subcortex_{subcortical}.dlabel.nii",
        "glasser":    "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR_Tian_Subcortex_{subcortical}.dlabel.nii",
        "gordon":     "Gordon333.32k_fs_LR_Tian_Subcortex_{subcortical}.dlabel.nii",
    }

    if atlas == "schaefer" and subcortical is None:
        return filenames["schaefer_c"].format(parcels=resolution, networks=networks, kong="Kong2022_" if kong else "")
    return filenames[atlas].format(parcels=resolution, networks=networks, subcortical=subcortical)

def _cases():
    # Introduce some conditions to skip invalid combinations
    for atlas in ATLASES:
        for resolution in RESOLUTIONS:
            for networks in NETWORKS:
                for subcortical in SUBCORTICALS:
                    for kong in KONG:
                        # General parameter checks
                        if resolution not in RESOLUTIONS:
                            continue
                        if networks not in NETWORKS:
                            continue
                        if subcortical not in SUBCORTICALS:
                            continue
                        if kong not in KONG:
                            continue
                        # Specific invalid combinations
                        if (atlas == "schaefer" and subcortical is not None and resolution not in (100, 200, 400)):
                            continue
                        if (atlas in ("glasser", "gordon") and subcortical is None):
                            continue
                        if (atlas == "schaefer" and resolution == 100 and networks == 17 and subcortical is not None):
                            continue
                        if (atlas == "glasser" and resolution > 100) or (atlas == "gordon" and resolution > 100):
                            continue
                        if (atlas == "glasser" and networks > 7) or (atlas == "gordon" and networks > 7):
                            continue
                        if (atlas == "schaefer" and networks == 7 and kong == True) or (atlas != "schaefer" and kong == True):
                            continue
            
                        yield (atlas, resolution, networks, subcortical, kong)

CASES = list(_cases())

# Fixtures
@pytest.fixture(scope="module")
def cifti_data():
    return utils.load_testdata(data="cifti")

@pytest.fixture(scope="module")
def cifti_file():
    # Real filesystem path to the packaged dtseries
    with importlib_resources.path("comet.data.tests", "cifti.dtseries.nii") as p:
        yield str(p)

@pytest.fixture
def atlas_file(atlas, resolution, networks, subcortical, kong):
    atlas_name = _atlas_filename(atlas, resolution, networks, subcortical, kong)
    with importlib_resources.path("comet.data.atlas", atlas_name) as p:
        yield str(p)

# Tests
@pytest.mark.parametrize("atlas,resolution,networks,subcortical,kong", CASES)
def test_parcellate_matches_workbench(cifti_file, atlas_file, tmp_path, atlas, resolution, networks, subcortical, kong):
    # Comet
    ts = cifti.parcellate(cifti_file, atlas=atlas, resolution=resolution, networks=networks, subcortical=subcortical, kong=kong, standardize=False)

    # Workbench
    out_file = tmp_path / "wb.ptseries.nii"
    cmd = [_wb_command(), "-cifti-parcellate", cifti_file, atlas_file, "COLUMN", str(out_file)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    ts_wb = np.asarray(nib.load(str(out_file)).get_fdata(), dtype=np.float64)
    # Checks
    assert ts.shape == ts_wb.shape
    assert np.allclose(ts, ts_wb, rtol=1e-6, atol=1e-6)
