import pytest
import numpy as np

from comet import connectivity
import teneto
import pydfc
from pydfc.data_loader import TIME_SERIES
from pydfc.dfc_methods import WINDOWLESS

np.random.seed(0)

# Fixture to load data once for all tests
@pytest.fixture(scope="module")
def ts():
    T = 500
    # Generate a sequence of time-varying covariance values
    time = np.arange(T)
    changing_covariance = 0.5 * np.sin(2 * np.pi * time / 100)

    # Initialize time series arrays
    series1 = np.zeros(T)
    series2 = np.zeros(T)

    # Generate the time series
    for t in range(1, T):
        mean = [0, 0]
        cov = [[1, changing_covariance[t]], [changing_covariance[t], 1]]
        series1[t], series2[t] = np.random.multivariate_normal(mean, cov)

    ts = np.column_stack((series1, series2))
    return ts

def test_SlidingWindow(ts):
    dfc_comet = connectivity.SlidingWindow(ts, windowsize=15).estimate()[0,1,:]
    dfc_teneto = teneto.timeseries.derive_temporalnetwork(ts.T, params={"method": "slidingwindow", "windowsize": 15})[0,1,:]
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)

def test_Jackknife(ts):
    dfc_comet = connectivity.Jackknife(ts, windowsize=1).estimate()[0,1,:]
    dfc_teneto = teneto.timeseries.derive_temporalnetwork(ts.T, params={"method": "jackknife"})[0,1,:]
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)

"""
# Teneto implementation is not working with newer scipy versions anymore
def test_SpatialDistance(ts):
    dfc_comet = connectivity.SpatialDistance(ts, dist="euclidean").estimate()[0,1,:]
    dfc_teneto = teneto.timeseries.derive_temporalnetwork(ts.T, params={"method": "distance", "distance": "euclidean"})[0,1,:]
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)"""

"""
def test_FlexibleLeastSquares(ts):
    dfc_comet = connectivity.FlexibleLeastSquares(ts, windowsize=7).estimate()[0,1,:]
    dfc_teneto =
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)

def test_PhaseSynchrony(ts):
    dfc_comet = connectivity.PhaseSynchrony(ts).estimate()[0,1,:]
    dfc_teneto =
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)

def test_LeiDA(ts):
    dfc_comet = connectivity.LeiDA(ts).estimate()[0,1,:]
    dfc_teneto =
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)

def test_WaveletCoherence(ts):
    dfc_comet = connectivity.WaveletCoherence(ts).estimate()[0,1,:]
    dfc_teneto =
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)

def test_DCC(ts):
    dfc_comet = connectivity.DCC(ts).estimate()[0,1,:]
    dfc_teneto =
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)

def test_EdgeTimeSeries(ts):
    dfc_comet = connectivity.EdgeTimeSeries(ts).estimate()[0,1,:]
    dfc_teneto =
    assert np.allclose(dfc_comet, dfc_teneto, atol=1e-6)"""

def test_KSVD(ts):

    ts= 
    state_tc, states = connectivity.KSVD(ts, n_states=5).estimate()
    

    locs = np.zeros((ts.shape[0], 3))
    labels = list(np.zeros(ts.shape[0])) if labels is None else labels

    dataobj = TIME_SERIES(data=ts, subj_id=["0"], Fs=1/tr, locs=locs, node_labels=labels)
    
    windowless = WINDOWLESS(n_states=5, random_state=0)
    windowless_obj = windowless.estimate_dFC(time_series=dataobj.get_subj_ts(subjs_id=["0"])) 

    # Extract dFC information
    dfc_data = np.transpose(windowless.get_dFC_mat(), (1, 2, 0))  # 3D dFC matrix (roi x roi x state_estimates)
    dfc_states = dfc_obj.FCSs_                                 # Dict with state matrices. Keys: "FCS1", FC2S", etc.
    dfc_state_tc = dfc_obj.state_TC()                          # State time courses

