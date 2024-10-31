import pytest
import numpy as np
from comet import connectivity
import teneto

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