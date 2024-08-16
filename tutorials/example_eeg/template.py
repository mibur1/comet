# Template script containing the required data for multiverse analysis.
# This file is used by the GUI, users should directly interact with their own multiverse script.

forking_paths = {
    'electrode': ['Pz', 'O1', 'O2', 'P3', 'P4'],
    'ica': [True],
    'normalization': ['min-max', 'relative', 'z-score', 'baseline'],
    'rejection': [80, 90, 100],
    'software': ['MNE', 'FieldTrip', 'EEGLab']}
def analysis_template():
    import os
    import comet
    import numpy as np

    base_value = 2
    if {{normalization}} == 'z-score':
        base_value += 0.3
    if {{electrode}} == 'Pz':
        base_value += 1.0
    if {{rejection}} == 80:
        base_value += 0.4
    outcome = base_value + np.random.normal(0, 0.5)

    result = {
        "P100 amplitude\ndifference across\nconditions (μV)": round(outcome, 3)
    }

    comet.data.save_universe_results(result, universe=os.path.abspath(__file__))
