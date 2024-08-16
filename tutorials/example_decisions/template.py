# Template script containing the required data for multiverse analysis.
# This file is used by the GUI, users should directly interact with their own multiverse script.

forking_paths = {
    'booleans': [True, False],
    'dfc_measures': [   {   'args': {'time_series': 'ts'},
                            'func': 'comet.connectivity.LeiDA',
                            'name': 'LeiDA'},
                        {   'args': {'time_series': 'ts', 'windowsize': 11},
                            'func': 'comet.connectivity.Jackknife',
                            'name': 'JC11'}],
    'graph_measures': [   {   'args': {'W': 'W', 'local': True},
                              'func': 'bct.efficiency_wei',
                              'name': 'eff_bct'},
                          {   'args': {'W': 'W', 'local': True},
                              'func': 'comet.graph.efficiency',
                              'name': 'eff_comet'}],
    'numbers': [1, 2, 4.2],
    'strings': ['Hello', 'world']}

invalid_paths = [('Hello', 4.2), ('world', 'eff_bct')]

def analysis_template():
    import numpy as np
    import bct
    import comet

    print(f"Decision 1: {{strings}}")
    print(f"Decision 2: {{numbers}}")
    print(f"Decision 3:{{booleans}}")

    # Load example data and calculate dFC + local efficiency
    ts = comet.data.load_example()
    dfc = {{dfc_measures}}
    dfc = dfc[0] if isinstance(dfc, tuple) else dfc #required as LeiDA returns multiple outputs

    efficiency = np.zeros((ts.shape[0], dfc.shape[1]))
    for i in range(dfc.shape[2]):
        W = dfc[:, :, i]
        W = np.abs(W)
        efficiency[i] = {{graph_measures}}

    print("Universe finished.")
