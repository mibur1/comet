import pytest
import numpy as np
from comet import network

def _random_fc(n, rng):
    x = rng.standard_normal((50, n))
    return np.corrcoef(x, rowvar=False)

def test_nsp_static_shapes():
    rng = np.random.default_rng(0)
    C = _random_fc(8, rng)

    nsp = network.NestedSpectralPartition(C, type="static")
    nsp.estimate()
    assert nsp.H_int.shape == nsp.H_seg.shape == nsp.H_bal.shape == (1,)
    assert np.allclose(nsp.H_bal, nsp.H_int - nsp.H_seg)

    C3 = np.stack([C, _random_fc(8, rng)])
    nsp3 = network.NestedSpectralPartition(C3, type="static")
    nsp3.estimate()
    assert nsp3.H_int.shape == (2,)
    assert nsp3.eigvecs.shape == (2, 8, 8)
    assert np.isclose(nsp3.H_int[0], nsp.H_int[0]) # subject 0 matches the 2D input

def test_nsp_dynamic_shapes():
    rng = np.random.default_rng(1)
    C3 = np.stack([_random_fc(8, rng) for _ in range(5)], axis=2) # (N, N, T)

    nsp = network.NestedSpectralPartition(C3, type="dynamic")
    assert (nsp.S, nsp.N, nsp.T) == (1, 8, 5) # nodes are not mistaken for subjects
    nsp.estimate()
    assert nsp.H_int.shape == nsp.H_seg.shape == (1, 5)
    assert nsp.eigvecs.shape == (1, 8, 8, 5)

    C4 = np.stack([C3, C3[:, :, ::-1]]) # (S, N, N, T)
    nsp4 = network.NestedSpectralPartition(C4, type="dynamic")
    nsp4.estimate()
    assert nsp4.H_int.shape == (2, 5)
    assert np.allclose(nsp4.H_int[0], nsp.H_int[0])
    assert np.allclose(nsp4.H_int[1], nsp.H_int[0, ::-1])

    nsp4.calibrate(C3.mean(axis=2))
    nsp4.dynamic_measures()
    assert np.all(np.isfinite(nsp4.H_bal_cal))
    assert np.allclose(nsp4.Dwell_In + nsp4.Dwell_Seg, 1)

def test_nsp_fully_integrated():
    # All-ones matrix: a single module with eigenvalue N, all remaining eigenvalues are 0
    nsp = network.NestedSpectralPartition(np.ones((6, 6)), type="static")
    nsp.estimate()
    assert np.isclose(nsp.H_int[0], 1.0)
    assert np.isclose(nsp.H_seg[0], 0.0)

def test_nsp_invalid_input():
    with pytest.raises(ValueError):
        network.NestedSpectralPartition(np.ones((6, 6)), type="dynamic")
    with pytest.raises(ValueError):
        network.NestedSpectralPartition(np.ones((2, 6, 6, 3)), type="static")
    with pytest.raises(ValueError):
        network.NestedSpectralPartition(np.ones((6, 5, 3)), type="dynamic")
