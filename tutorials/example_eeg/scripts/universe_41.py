import os
import comet
import numpy as np

base_value = 10
if 'MNE' == 'EEGlab':
    base_value += 2
if 'IAF' == 'IAF':
    base_value += 5
if 'Pz' == 'Pz':
    base_value += 7
if 'IAF' == 'IAF' and 'Pz' == 'Pz':
    base_value += 2

# Generate 50 outcome values (mock differences in alpha power)
power_diffs = [base_value + np.random.normal(0, 35) for _ in range(50)]

result = {
    "power_diffs": [round(power_diff, 3) for power_diff in power_diffs],
}

comet.utils.save_universe_results(result)