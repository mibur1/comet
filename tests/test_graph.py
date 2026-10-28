import pytest
import numpy as np
from comet import graph, utils

# Fixture to load data once for all tests
@pytest.fixture(scope="module")
def data():
    """
    Keys: 'W', 'W_abs', 'W_bin', 'W_fix', 'W_len', 'W_norm', 'W_prop',
        'avg_clustering_onella', 'avg_path_bin', 'avg_path_wei', 'dist_bin', 'dist_wei',
        'eff_bin', 'eff_wei', 'matching_ind', 'swp_bin', 'swp_wei', 'trans_und_bin', 'trans_und_wei'
    """
    data = utils.load_testdata(data="graph")
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

def test_fisher_z():
    x = np.random.randn(100, 10)
    R = np.corrcoef(x, rowvar=False)
    Z = graph.fisher_z(R)
    off = ~np.eye(10, dtype=bool)
    assert np.allclose(Z[off], np.arctanh(R[off]))
    assert np.all(np.diag(Z) == 0)
    assert np.all(np.diag(R) != 0) # input untouched

    R3 = np.stack([R, R], axis=2)
    assert np.allclose(graph.fisher_z(R3), np.stack([Z, Z], axis=2))

    R_perfect = R.copy()
    R_perfect[0, 1] = R_perfect[1, 0] = 1.0
    with pytest.warns(UserWarning):
        assert np.all(np.isfinite(graph.fisher_z(R_perfect)))

    with pytest.raises(ValueError):
        graph.fisher_z(R + 1.5)

def test_preprocess(data):
    W = data['W']
    W_expected = graph.postproc(graph.threshold(graph.handle_negative_weights(W, type="absolute"), type="density", density=0.4))
    W_actual = graph.preprocess(W, negative_weights="absolute", threshold_type="density", density=0.4)
    assert np.array_equal(W_actual, W_expected)
    assert np.array_equal(graph.preprocess(W, clean=False), W) # all steps off

    x = np.random.randn(100, 10)
    R = np.stack([np.corrcoef(x, rowvar=False)] * 3, axis=2)
    R_expected = graph.postproc(graph.invert(graph.normalise(graph.threshold(graph.handle_negative_weights(
        graph.fisher_z(R), type="discard"), type="absolute", threshold=0.05))))
    R_actual = graph.preprocess(R, fisher=True, negative_weights="discard", threshold_type="absolute",
                                threshold_value=0.05, scaling="normalise", distance="invert")
    assert np.array_equal(R_actual, R_expected)

    with pytest.raises(ValueError):
        graph.preprocess(W, binary=True, scaling="normalise")
    with pytest.raises(ValueError):
        graph.preprocess(W, threshold_type="absolute")

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

def test_system_segregation():
    # test case of the reference implementation (mychan24/system-segregation-and-graph-tools, test/testR.R)
    m = np.array([[1, 4, 2, 2], [4, 1, 3, 2], [2, 3, 1, 8], [2, 2, 8, 1]], dtype=float)
    S, W, B = graph.system_segregation(m, [1, 1, 2, 2], return_components=True)
    assert np.allclose([S, W, B], [0.625, 6.0, 2.25])

    # per system, string labels, single-node system -> NaN, and the pooled value is size-weighted
    ci = np.array(["a", "a", "a", "b", "b", "c"])
    G = np.where(ci[:, None] == ci[None, :], 1.0, 0.5)
    np.fill_diagonal(G, 0)
    S_sys = graph.system_segregation(G, ci, per_system=True)
    assert np.allclose(S_sys[:2], [0.5, 0.5]) and np.isnan(S_sys[2])
    assert np.isclose(graph.system_segregation(G, ci), 0.5)

    # 3D input: one value per slice, scale invariant
    G3 = np.stack([G, 2 * G], axis=-1)
    assert np.allclose(graph.system_segregation(G3, ci), [0.5, 0.5])
    assert graph.system_segregation(G3, ci, per_system=True).shape == (3, 2)

def test_strengths_und():
    import bct
    W = np.random.uniform(-1, 1, (10, 10))
    W = (W + W.T) / 2
    assert np.allclose(graph.strengths_und(W), bct.strengths_und(W))
    Spos, Sneg = graph.strengths_und(W, sign=True)
    assert np.allclose(Spos, bct.strengths_und_sign(W)[0]) and np.allclose(Sneg, bct.strengths_und_sign(W)[1])

    # 3D input: one column per slice
    W3 = np.stack([W, 2 * W], axis=-1)
    assert np.allclose(graph.strengths_und(W3)[:, 1], 2 * bct.strengths_und(W))
    assert graph.strengths_und(W3, sign=True)[0].shape == (10, 2)

    # normalise: mean weight per possible link, diagonal ignored
    W0 = W.copy()
    np.fill_diagonal(W0, 0)
    assert np.allclose(graph.strengths_und(W, normalise=True), W0.sum(axis=0) / 9)
    assert np.allclose(graph.strengths_und(W, sign=True, normalise=True)[0], Spos / 9)
