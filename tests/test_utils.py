import pytest
import numpy as np
from comet import utils

@pytest.fixture(scope="module")
def summary():
    """
    Create 3 subjects, each with two 'true' states (values 1 and 2)
    and different dwell proportions.
    """
    dfc_list = []
    dwells = [(30, 70), (50, 50), (20, 30, 50)]
    for i in range(3):
        if i < 2:
            # subjects 1 & 2: single switch
            m1 = np.full((2, 2, dwells[i][0]), 1.0)
            m2 = np.full((2, 2, dwells[i][1]), 2.0)
            dfc = np.concatenate([m1, m2], axis=2)
        else:
            # subject 3: 1→2→1 pattern
            m1 = np.full((2, 2, dwells[i][0]), 1.0)
            m2 = np.full((2, 2, dwells[i][1]), 2.0)
            m3 = np.full((2, 2, dwells[i][2]), 1.0)
            dfc = np.concatenate([m1, m2, m3], axis=2)
        dfc_list.append(dfc)

    dfc_array = np.stack(dfc_list, axis=0)  # shape (3, 2, 2, 100)
    state_tc, states, inertia = utils.kmeans_cluster(dfc_array, num_states=2, random_state=0)
    
    return utils.summarise_state_tc(state_tc)

def test_dwell(summary):
    # Average length (in time points) of contiguous runs in each state per subject
    expected_dwell = np.array([[70., 30.],
                               [50., 50.],
                               [30., 35.]])
    assert np.allclose(summary["dwell_times"], expected_dwell)

def test_trans(summary):
    # Row-stochastic transition probability matrices: P(next = j | current = i)
    expected_trans = np.array([[[1.0,   0.0 ], [1/30,  29/30]],
                               [[1.0,   0.0 ], [1/50,  49/50]],
                               [[29/30, 1/30], [1/69,  68/69]]])
    assert np.allclose(summary["transitions"], expected_trans)

def test_fo(summary):
    # Fraction of total time spent in each state
    expected_fo = np.array([[0.7, 0.3],
                            [0.5, 0.5],
                            [0.3, 0.7]])
    assert np.allclose(summary["fractional_occupancy"], expected_fo)

def test_counts(summary):
    # Raw counts of transitions i → j.
    expected_counts = np.array([[[69, 0], [1, 29]],
                                [[49, 0], [1, 49]],
                                [[29, 1], [1, 68]]])
    assert np.array_equal(summary["transition_counts"], expected_counts)

def test_switches(summary):
    # Total number of switches (state changes) per subject.
    expected_sum = np.array([1, 1, 2])
    assert np.array_equal(summary["transitions_sum"], expected_sum)

def test_switch_rate(summary):
    # Fraction of time steps that involve a switch.
    expected_rate = np.array([1/99, 1/99, 2/99])
    assert np.allclose(summary["switch_rate"], expected_rate)
