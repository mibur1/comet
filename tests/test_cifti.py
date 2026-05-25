import shutil
import subprocess
import nibabel as nib
import numpy as np
import pytest
import importlib_resources
from comet import cifti, utils

# Parameter space
ATLASES = ["schaefer", "yan", "glasser", "gordon"]
RESOLUTIONS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
NETWORKS = [7, 17]
SUBCORTICALS = [None, "S1", "S2", "S3", "S4"]
KONG = [False, True]

# Number of Tian subcortical parcels per scale, and fixed cortical counts.
N_SUB = {None: 0, "S1": 16, "S2": 32, "S3": 50, "S4": 54}
N_CORT = {"glasser": 360, "gordon": 333}  # schaefer/yan have `resolution` cortical parcels

wb_missing = shutil.which("wb_command") is None


def _wb_command():
    wb = shutil.which("wb_command")
    if wb is None:
        raise RuntimeError(
            "wb_command not found on PATH. "
            "Please make sure Connectome Workbench is installed and wb_command is available."
        )
    return wb


def _n_cortical(atlas, resolution):
    return N_CORT.get(atlas, resolution)


def _cortical_filename(atlas, resolution, networks, kong):
    """Published dlabel that supplies the cortical parcels for an atlas.

    Schaefer/Yan ship cortical-only files. Glasser/Gordon are only published as Tian-combined
    files, so the (scale-invariant) S1 file is used and its leading 16 Tian columns are dropped.
    """
    if atlas == "schaefer":
        return "Schaefer2018_{p}Parcels_{k}{n}Networks_order.dlabel.nii".format(
            p=resolution, n=networks, k="Kong2022_" if kong else "")
    if atlas == "yan":
        return "{p}Parcels_Yeo2011_{n}Networks.dlabel.nii".format(p=resolution, n=networks)
    if atlas == "glasser":
        return ("Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors"
                ".32k_fs_LR_Tian_Subcortex_S1.dlabel.nii")
    if atlas == "gordon":
        return "Gordon333.32k_fs_LR_Tian_Subcortex_S1.dlabel.nii"


def _gordon_tian_filename(subcortical):
    return "Gordon333.32k_fs_LR_Tian_Subcortex_{s}.dlabel.nii".format(s=subcortical)


def _cases():
    """Every valid atlas/parameter combination after the cortical+subcortical refactor.

    Every atlas works both cortical-only (subcortical=None) and combined with any Tian scale.
    Resolution/networks only matter for Schaefer and Yan; kong only for Schaefer (and forces
    17 networks, so the kong+7 cases are dropped as duplicates of kong+17).
    """
    for atlas in ATLASES:
        resolutions = RESOLUTIONS if atlas in ("schaefer", "yan") else [100]
        networks_list = NETWORKS if atlas in ("schaefer", "yan") else [7]
        kong_list = KONG if atlas == "schaefer" else [False]
        for resolution in resolutions:
            for networks in networks_list:
                for subcortical in SUBCORTICALS:
                    for kong in kong_list:
                        if kong and networks == 7:
                            continue
                        yield (atlas, resolution, networks, subcortical, kong)


CASES = list(_cases())


# Fixtures
@pytest.fixture(scope="module")
def cifti_data():
    # Vertex data array (T, 91282) for the structural checks.
    return utils.load_testdata(data="cifti")


@pytest.fixture(scope="module")
def cifti_file():
    # Real filesystem path to the packaged dtseries (needed by wb_command).
    with importlib_resources.path("comet.data.tests", "cifti.dtseries.nii") as p:
        yield str(p)


def _wb_parcellate(wb, cifti_file, atlas_filename, out_path):
    """Parcellate the dtseries with a published dlabel using wb_command (ground truth)."""
    with importlib_resources.path("comet.data.atlas", atlas_filename) as dlabel:
        subprocess.run([wb, "-cifti-parcellate", cifti_file, str(dlabel), "COLUMN", str(out_path)],
                       check=True, capture_output=True, text=True)
    return np.asarray(nib.load(str(out_path)).get_fdata(), dtype=np.float64)


def _expected(wb, cifti_file, atlas, resolution, networks, subcortical, kong, tmp_path):
    """Reference parcellation assembled from published component files via wb_command.

    Mirrors the runtime assembly independently: cortex from the cortical source file and,
    when requested, the Tian subcortex from the Gordon+Tian file placed first (matching the
    published combined atlases). For Glasser/Gordon the cortical source is the S1 file with its
    16 leading Tian columns dropped.
    """
    cort = _wb_parcellate(wb, cifti_file, _cortical_filename(atlas, resolution, networks, kong),
                          tmp_path / "wb_cort.ptseries.nii")
    if atlas in ("glasser", "gordon"):
        cort = cort[:, N_SUB["S1"]:]
    if subcortical is None:
        return cort
    sub = _wb_parcellate(wb, cifti_file, _gordon_tian_filename(subcortical),
                         tmp_path / "wb_sub.ptseries.nii")[:, :N_SUB[subcortical]]
    return np.concatenate([sub, cort], axis=1)


# Tests
@pytest.mark.parametrize("atlas,resolution,networks,subcortical,kong", CASES)
def test_shapes_and_layout(cifti_data, atlas, resolution, networks, subcortical, kong):
    # comet forces kong to 17 networks; match that for the expected counts.
    if atlas == "schaefer" and networks == 7 and kong:
        networks = 17
    n_parcels = _n_cortical(atlas, resolution) + N_SUB[subcortical]

    ts, labels, vlab = cifti.parcellate(cifti_data, atlas=atlas, resolution=resolution,
                                        networks=networks, subcortical=subcortical, kong=kong,
                                        return_labels=True)
    assert ts.shape[1] == n_parcels == len(labels)

    if subcortical is None:
        assert vlab.size == 64984  # cortical-only 64984-vertex surface map
    else:
        assert vlab.size == 91282  # cortex + subcortex grayordinates
        # Subcortical parcels are keyed first (leading columns).
        n_sub = N_SUB[subcortical]
        assert all(lab.endswith(("-lh", "-rh")) for lab in labels[:n_sub])
        assert not any(lab.endswith(("-lh", "-rh")) for lab in labels[n_sub:])

    # Labels-only mode (no data) returns identical labels and vertex map.
    _, labels2, vlab2 = cifti.parcellate(None, atlas=atlas, resolution=resolution,
                                         networks=networks, subcortical=subcortical, kong=kong,
                                         return_labels=True)
    assert labels2 == labels
    assert np.array_equal(vlab2, vlab)


@pytest.mark.skipif(wb_missing, reason="wb_command (Connectome Workbench) not on PATH")
@pytest.mark.parametrize("atlas,resolution,networks,subcortical,kong", CASES)
def test_parcellate_matches_workbench(cifti_file, tmp_path, atlas, resolution, networks, subcortical, kong):
    wb = _wb_command()

    # comet forces kong to 17 networks; use the same cortical source file for the reference.
    if atlas == "schaefer" and networks == 7 and kong:
        networks = 17

    # comet (also downloads/caches the component files the reference needs below).
    ts = cifti.parcellate(cifti_file, atlas=atlas, resolution=resolution, networks=networks,
                          subcortical=subcortical, kong=kong, standardize=False)

    expected = _expected(wb, cifti_file, atlas, resolution, networks, subcortical, kong, tmp_path)

    assert ts.shape == expected.shape
    assert np.allclose(ts, expected, rtol=1e-6, atol=1e-6)
