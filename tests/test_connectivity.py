import pytest
import numpy as np

from comet import connectivity, utils
import teneto
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
    - FlexibleLeastSquares: DynamicBC toolbox in MATLAB
        Paper:   https://pmc.ncbi.nlm.nih.gov/articles/PMC4268585/
        Toolbox: https://github.com/guorongwu/DynamicBC/
    - Dynamic Conditional Correlation: DCC package in MATLAB
        Paper:   https://doi.org/10.1016/j.neuroimage.2014.06.052
        Toolbox: https://github.com/canlab/Lindquist_Dynamic_Correlation
    - LeiDA: LeiDA package in MATLAB
        Paper:   https://www.nature.com/articles/s41598-017-05425-7
        Toolbox: https://github.com/juanitacabral/LEiDA

"""

def test_SlidingWindow(ts):
    dfc_comet = connectivity.SlidingWindow(ts, windowsize=15, diagonal=1).estimate()
    dfc_teneto = teneto.timeseries.derive_temporalnetwork(ts.T, params={"method": "slidingwindow", "windowsize": 15})
    print(dfc_comet.shape, dfc_teneto.shape)
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)

def test_Jackknife(ts):
    dfc_comet = connectivity.Jackknife(ts, windowsize=1, diagonal=1).estimate()
    dfc_teneto = teneto.timeseries.derive_temporalnetwork(ts.T, params={"method": "jackknife"})
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)

def test_SpatialDistance(ts):
    dfc_comet = connectivity.SpatialDistance(ts, dist="euclidean", diagonal=1).estimate()
    dfc_teneto = utils.load_testdata(data="connectivity")["spatial_distance"]
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)

def test_MTD(ts):
    dfc_comet = connectivity.TemporalDerivatives(ts, windowsize=7, diagonal=1).estimate()
    dfc_teneto = teneto.timeseries.derive_temporalnetwork(ts.T, params={"method": "multiplytemporalderivative", "windowsize": 7})
    i = np.arange(dfc_teneto.shape[0]) # set diagonal to 1
    dfc_teneto[i, i, :] = 1
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)

def test_PhaseSynchronization(ts):
    dfc_comet = connectivity.PhaseSynchrony(ts, method="teneto", diagonal=1).estimate()
    dfc_teneto = teneto.timeseries.derive_temporalnetwork(ts.T, params={"method": "instantaneousphasesync"})
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-4)

def test_FlexibleLeastSquares(ts):
    dfc_comet = connectivity.FlexibleLeastSquares(ts, mu=100).estimate()
    dfc_dynamicBC = utils.load_testdata(data="connectivity")["fls"]
    assert np.allclose(dfc_comet, dfc_dynamicBC, atol=1e-2)

def test_DCC(ts):
    dfc_comet = connectivity.DCC(ts, diagonal=1).estimate()
    dfc_dcc = utils.load_testdata(data="connectivity")["dcc"]
    corr = np.corrcoef(dfc_comet.flatten(), dfc_dcc.flatten())[0,1]
    assert corr > 0.95

def test_LeiDA(ts):
    dfc_comet = connectivity.LeiDA(ts, diagonal=1).estimate()
    dfc_LeiDA =
    assert np.allclose(dfc_comet, dfc_LeiDA, atol=1e-6)

def test_WaveletCoherence(ts):
    dfc_comet = connectivity.WaveletCoherence(ts, diagonal=1).estimate()
    dfc_wcoh =
    assert np.allclose(dfc_comet, dfc_wcoh, atol=1e-6)

def test_EdgeTimeSeries(ts):
    dfc_comet = connectivity.EdgeConnectivity(ts, diagonal=1).estimate()
    dfc_ets =
    assert np.allclose(dfc_comet, dfc_ets, atol=1e-6)

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

