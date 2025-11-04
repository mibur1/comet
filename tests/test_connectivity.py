import pytest
import teneto
import numpy as np
from comet import connectivity, utils

import pydfc
from pydfc.data_loader import TIME_SERIES
from pydfc.dfc_methods import WINDOWLESS

np.random.seed(0)

# Fixture to load data once for all tests
@pytest.fixture(scope="module")
def ts():
    ts = utils.load_example("time_series.txt")
    return ts

"""
Connectivity methods are compared to their implementation in the original packages (teneto, pydfc, DynamicBC, etc.).
    - If possible, the packages are used directly with real fMRI data as included in utils.load_example("time_series.txt")
    - Otherwise, precomputed dFC matrices are loaded (calculated for only the first 25 ROIs of fMRI data to save space)

Precomputed dFC matrices:
    - SpatialDistance: Teneto fails with newer scipy versions (module 'scipy.spatial.distance' has no attribute 'kulsinski')
        Implementation: teneto.timeseries.derive_temporalnetwork(ts.T, params={"method": "distance", "distance": "euclidean"})
    - FlexibleLeastSquares
        Paper:  https://pmc.ncbi.nlm.nih.gov/articles/PMC4268585/
        Code:   https://github.com/guorongwu/DynamicBC/
    - Dynamic Conditional Correlation
        Paper:  https://doi.org/10.1016/j.neuroimage.2014.06.052
        Code:   https://github.com/canlab/Lindquist_Dynamic_Correlation
    - LeiDA
        Paper:  https://www.sciencedirect.com/science/article/pii/S1053811922008370?via%3Dihub
        Code:   https://github.com/anders-s-olsen/psilocybin_dynamic_FC
    - Edge Time Series
        Paper:  https://www.sciencedirect.com/science/article/pii/S1053811922007066
        Code:   https://github.com/brain-networks/edge-ts/blob/master/main.m
"""

def test_SlidingWindow(ts):
    """Test that SlidingWindow() gives same result as Teneto's sliding window."""
    dfc_comet = connectivity.SlidingWindow(ts, windowsize=15, diagonal=1).estimate()
    dfc_teneto = teneto.timeseries.derive_temporalnetwork(ts.T, params={"method": "slidingwindow", "windowsize": 15})
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)

def test_Jackknife(ts):
    """Test that Jackknife() gives same result as Teneto's jackknife correlation."""
    dfc_comet = connectivity.Jackknife(ts, windowsize=1, diagonal=1).estimate()
    dfc_teneto = teneto.timeseries.derive_temporalnetwork(ts.T, params={"method": "jackknife"})
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)

def test_MTD(ts):
    """Test that MTD() gives same result as Teneto's multiply temporal derivative."""
    dfc_comet = connectivity.TemporalDerivatives(ts, windowsize=7, diagonal=1).estimate()
    dfc_teneto = teneto.timeseries.derive_temporalnetwork(ts.T, params={"method": "multiplytemporalderivative", "windowsize": 7})
    i = np.arange(dfc_teneto.shape[0]) # set diagonal to 1
    dfc_teneto[i, i, :] = 1
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)

def test_PhaseSynchronization(ts):
    """Test that PhaseSynchrony() gives same result as Teneto's instantaneous phase synchronization."""
    dfc_comet = connectivity.PhaseSynchronization(ts, method="teneto", diagonal=1).estimate()
    dfc_teneto = teneto.timeseries.derive_temporalnetwork(ts.T, params={"method": "instantaneousphasesync"})
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-4)

def test_SpatialDistance(ts):
    """Test that SpatialDistance() gives same result as Teneto's spatial distance."""
    dfc_comet = connectivity.SpatialDistance(ts[:,:5], dist="euclidean", diagonal=1).estimate()
    dfc_teneto = utils.load_testdata(data="connectivity")["sd"]
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)

def test_FlexibleLeastSquares(ts):
    """Test that FlexibleLeastSquares() gives same result as DynamicBC toolbox."""
    dfc_comet = connectivity.FlexibleLeastSquares(ts[:,:5], mu=100, num_cores=1, progress_bar=False).estimate()
    dfc_dynamicBC = utils.load_testdata(data="connectivity")["fls"]
    assert np.allclose(dfc_comet, dfc_dynamicBC, atol=1e-2)

def test_LeiDA(ts):
    """Test that LeiDA() gives same result as MATLAB implementation."""
    leida = connectivity.LeiDA(ts[:,:5])
    dfc = leida.estimate()
    v1_comet = leida.V1
    v1_mat = utils.load_testdata(data="connectivity")["leida"]
    assert np.allclose(v1_comet, v1_mat, atol=1e-4)

def test_ets(ts):
    """Test that EdgeConnectivity() gives same result as MATLAB implementation."""
    ets_comet = connectivity.EdgeConnectivity(ts[:,:5], method="eTS").estimate()
    ets_mat = utils.load_testdata(data="connectivity")["ets"]
    assert np.allclose(ets_comet, ets_mat, atol=1e-6)

def test_DCC(ts):
    """Test that DCC() gives similar result as MATLAB implementation."""
    dfc_comet = connectivity.DCC(ts[:,:5], diagonal=1).estimate()
    dfc_dcc = utils.load_testdata(data="connectivity")["dcc"]
    corr = np.corrcoef(dfc_comet.flatten(), dfc_dcc.flatten())[0,1]
    assert corr > 0.9

def test_clustering_equivalence(ts):
    """Test that SlidingWindow() + kmeans_cluster gives same result as SlidingWindowClustering()."""
    dfc_sw = connectivity.SlidingWindow(ts, windowsize=29, stepsize=1, shape="gaussian", diagonal=1).estimate()
    state_tc, _, _ = utils.kmeans_cluster(dfc_sw, num_states=5, subject_clusters=5, strategy="two_level", random_state=42)
    state_tc_swc, _ = connectivity.SlidingWindowClustering(ts, n_states=5, subject_clusters=5, windowsize=29, stepsize=1, random_state=42).estimate()
    assert np.allclose(state_tc[0], state_tc_swc[0], atol=1e-6)

"""
def test_KSVD(ts):
    state_tc, states = connectivity.KSVD(ts, n_states=5).estimate()
    
    locs = np.zeros((ts.shape[0], 3))
    labels = list(np.zeros(ts.shape[0])) if labels is None else labels

    dataobj = TIME_SERIES(data=ts, subj_id=["0"], Fs=1/tr, locs=locs, node_labels=labels)
    
    windowless = WINDOWLESS(n_states=5, random_state=0)
    windowless_obj = windowless.estimate_dFC(time_series=dataobj.get_subj_ts(subjs_id=["0"])) 

    # Extract dFC information
    dfc_data = np.transpose(windowless.get_dFC_mat(), (1, 2, 0))  # 3D dFC matrix (roi x roi x state_estimates)
    dfc_states = dfc_obj.FCSs_                                 # Dict with state matrices. Keys: "FCS1", FC2S", etc.
    dfc_state_tc = dfc_obj.state_TC()                          # State time courses"""
