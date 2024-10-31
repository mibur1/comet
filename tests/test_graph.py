import pytest
import sys
import numpy as np
from scipy.io import loadmat
from comet import graph

# Fixture to load data once for all tests
@pytest.fixture(scope="module")
def data():
    wd = sys.path[0]
    """
    Keys: 'W', 'W_abs', 'W_bin', 'W_fix', 'W_len', 'W_norm', 'W_prop',
        'avg_clustering_onella', 'avg_path_bin', 'avg_path_wei', 'dist_bin', 'dist_wei',
        'eff_bin', 'eff_wei', 'matching_ind', 'swp_bin', 'swp_wei', 'trans_und_bin', 'trans_und_wei'
    """
    data = loadmat(f'{wd}/files/network_measures.mat')
    data["W_prop"] = np.ascontiguousarray(data["W_prop"], dtype=np.float64)
    data["W_bin"] = np.ascontiguousarray(data["W_bin"], dtype=np.float64)
    data["W_len"] = np.ascontiguousarray(data["W_len"], dtype=np.float64)
    return data

def test_handle_negative_weights():
    W = np.random.uniform(-1, 1, (10, 10))
    W_abs = graph.handle_negative_weights(W, type="absolute")
    W_disc = graph.handle_negative_weights(W, type="discard")
    W_expected_disc = np.where(W < 0, 0, W)
    assert np.all(W_abs >= 0)
    assert np.array_equal(W_disc, W_expected_disc)

def test_threshold(data):
    W = data['W']
    W_abs_expected = data['W_abs']
    W_abs_actual = graph.threshold(W, type="absolute", threshold=0.6)
    W_prop_expected = data['W_prop']
    W_prop_actual = graph.threshold(W, type="density", density=0.4)
    assert np.array_equal(W_abs_actual, W_abs_expected)
    assert np.array_equal(W_prop_actual, W_prop_expected)

def test_binarise(data):
    W_prop = data['W_prop']
    W_bin_expected = data['W_bin']
    W_bin_actual = graph.binarise(W_prop)
    assert np.array_equal(W_bin_actual, W_bin_expected)

def test_normalise(data):
    W_prop = data['W_prop']
    W_norm_expected = data['W_norm']
    W_norm_actual = graph.normalise(W_prop)
    assert np.array_equal(W_norm_actual, W_norm_expected)

def test_invert(data):
    W_prop = data['W_prop']
    W_len_expected = data['W_len']
    W_len_actual = graph.invert(W_prop)
    assert np.array_equal(W_len_actual, W_len_expected)

def test_symmetrise():
    W = np.random.uniform(-1, 1, (10, 10))
    W_sym = graph.symmetrise(W)
    assert np.array_equal(W_sym, W_sym.T)

def test_logtransform():
    # TODO
    pass

def test_randomise(data):
    # TODO
    pass

def test_regular_matrix():
    # TODO
    pass

def test_postproc(data):
    W_prop = data['W_prop']
    W_fix_expected = data['W_fix']
    W_fix_actual = graph.postproc(W_prop)
    assert np.allclose(W_fix_actual, W_fix_expected, atol=1e-6)

def test_distance_wei(data):
    W_len = data['W_len']
    dist_wei_expected = data['dist_wei']
    dist_wei_actual = graph.distance_wei(W_len)
    assert np.array_equal(dist_wei_actual, dist_wei_expected)
def test_distance_bin(data):
    W_bin = data['W_bin']
    dist_bin_expected = data['dist_bin']
    dist_bin_actual = graph.distance_bin(W_bin)
    assert np.array_equal(dist_bin_actual, dist_bin_expected)

def test_avg_shortest_path_wei(data):
    W_len = data['W_len']
    avg_path_wei_expected = data['avg_path_wei']
    avg_path_wei_actual = graph.avg_shortest_path(W_len)
    assert np.allclose(avg_path_wei_actual, avg_path_wei_expected, atol=1e-3)

def test_avg_shortest_path_bin(data):
    W_bin = data['W_bin']
    avg_path_bin_expected = data['avg_path_bin']
    avg_path_bin_actual = graph.avg_shortest_path(W_bin)
    assert np.allclose(avg_path_bin_actual, avg_path_bin_expected, atol=1e-3)

def test_transitivity_und_wei(data):
    W_prop = data['W_prop']
    trans_und_wei_expected = data['trans_und_wei']
    trans_und_wei_actual = graph.transitivity_und(W_prop)
    assert np.allclose(trans_und_wei_actual, trans_und_wei_expected, atol=1e-3)

def test_transitivity_und_bin(data):
    W_bin = data['W_bin']
    trans_und_bin_expected = data['trans_und_bin']
    trans_und_bin_actual = graph.transitivity_und(W_bin)
    assert np.allclose(trans_und_bin_actual, trans_und_bin_expected, atol=1e-3)

def test_efficiency_wei(data):
    W_prop = data['W_prop']
    eff_wei_expected = data['eff_wei']
    eff_wei_actual = graph.efficiency(W_prop)
    assert np.allclose(eff_wei_actual, eff_wei_expected, atol=1e-6)

def test_efficiency_bin(data):
    W_bin = data['W_bin']
    eff_bin_expected = data['eff_bin']
    eff_bin_actual = graph.efficiency(W_bin)
    assert np.allclose(eff_bin_actual, eff_bin_expected, atol=1e-6)

def test_avg_clustering_onella(data):
    W_prop = data['W_prop']
    avg_clustering_onella_expected = data['avg_clustering_onella']
    avg_clustering_onella_actual = graph.avg_clustering_onella(W_prop)
    assert np.allclose(avg_clustering_onella_actual, avg_clustering_onella_expected, atol=1e-2)

def test_matching_ind_und(data):
    W_bin = data['W_bin']
    matching_ind_expected = data['matching_ind']
    matching_ind_actual = graph.matching_ind_und(W_bin)
    assert np.allclose(matching_ind_actual, matching_ind_expected, atol=1e-3)

def test_small_world_propensity_wei(data):
    W_prop = data['W_prop']
    swp_wei_expected = data['swp_wei']
    swp_wei_actual = graph.small_world_propensity(W_prop)[0]
    print("SWP:", swp_wei_actual, swp_wei_expected)
    assert np.allclose(swp_wei_actual, swp_wei_expected, atol=1e-2)

def test_small_world_propensity_bin(data):
    W_bin = data['W_bin']
    swp_bin_expected = data['swp_bin']
    swp_bin_actual = graph.small_world_propensity(W_bin)[0]
    assert np.allclose(swp_bin_actual, swp_bin_expected, atol=1e-2)
