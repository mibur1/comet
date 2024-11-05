# Template script containing the required data for multiverse analysis.
# This file is used/overwritten by the GUI, users should usually directly interact with their own multiverse script.

from comet.multiverse import Multiverse

forking_paths = {
    "dfc_method": [
    {
        "name": "TSW21",
        "func": "comet.connectivity.SlidingWindow",
        "args": {
            "Option": "CONT Sliding Window",
            "time_series": "ts_hp",
            "windowsize": 21,
            "stepsize": 1,
            "shape": "gaussian",
            "std": 7.0,
            "diagonal": 0,
            "standardize": False,
            "fisher_z": False,
            "tril": False
        }
    },
    {
        "name": "SD",
        "func": "comet.connectivity.SpatialDistance",
        "args": {
            "Option": "CONT Spatial Distance",
            "time_series": "ts_hp",
            "dist": "euclidean",
            "diagonal": 0,
            "standardize": False,
            "fisher_z": False,
            "tril": False
        }
    },
    {
        "name": "MTD7",
        "func": "comet.connectivity.TemporalDerivatives",
        "args": {
            "Option": "CONT Multiplication of Temporal Derivatives",
            "time_series": "ts_hp",
            "windowsize": 7,
            "diagonal": 0,
            "standardize": False,
            "fisher_z": False,
            "tril": False
        }
    },
    {
        "name": "PSc",
        "func": "comet.connectivity.PhaseSynchrony",
        "args": {
            "Option": "CONT Phase Synchronization",
            "time_series": "ts_hp",
            "method": "crp",
            "diagonal": 0,
            "standardize": False,
            "fisher_z": False,
            "tril": False
        }
    },
    {
        "name": "FLS",
        "func": "comet.connectivity.FlexibleLeastSquares",
        "args": {
            "Option": "CONT Flexible Least Squares",
            "time_series": "ts_hp",
            "standardizeData": True,
            "mu": 50.0,
            "num_cores": 8,
            "diagonal": 0,
            "standardize": False,
            "fisher_z": False,
            "tril": False
        }
    }
],
    "tr": [0.72],
    "segment_length": [10],
    "delay": [6, 10],
    "density": [0.5, 0.25],
    "binarise": [True, False],
    "graph_measure": [
    {
        "name": "participation",
        "func": "bct.participation_coef",
        "args": {
            "Option": "BCT Participation coef",
            "W": "W",
            "ci": "louvain",
            "degree": "undirected"
        }
    },
    {
        "name": "clustering",
        "func": "bct.clustering_coef",
        "args": {
            "Option": "BCT Clustering coefficient",
            "W": "W"
        }
    },
    {
        "name": "efficiency",
        "func": "comet.graph.efficiency",
        "args": {
            "Option": "COMET Efficiency",
            "G": "",
            "local": True
        }
    }
],
    "svc_kernel": ['linear', 'rbf'],
}

def analysis_template():
    import comet
    import os
    import bct
    import numpy as np
    from tqdm import tqdm
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    from joblib import Parallel, delayed

    #####################################
    # 1. LOAD DATA AND EXTRACT PARAMETERS
    data_sim = comet.data.load_example(ftype="pkl")
    ts_sim = data_sim[0] # time series data
    time = data_sim[1]   # time in seconds
    onsets = data_sim[2] # trial onsets in seconds
    labels = data_sim[3] # trial labels (connectivity state)

    ###############################################
    # 2. DFC CALCULATION (DECISION: DFC METHOD)
    # Preprocessing. Phase-based methods require band-pass filtering, amplitude-based methods require high-pass filtering.
    ts_bp = comet.data.clean(ts_sim, confounds=None, t_r={{tr}}, detrend=True, standardize=False, \
                             high_pass=0.03, low_pass=0.07) # band pass (narrow-band signal for Hilbert transform)
    ts_hp = comet.data.clean(ts_sim, confounds=None, t_r={{tr}}, detrend=True, standardize=False, \
                             high_pass=0.01)                # high pass (for amplitude based methods)

    # Estimate dFC
    dfc_ts = {{dfc_method}}

    #######################################
    # 3. SEGMENT DATA (DECISION: DELAY)
    segments = []
    for i in onsets:
        segment = [i+j+{{delay}} for j in range(0, {{segment_length}})]
        segments.append(segment)

    segments = np.asarray(segments).astype(int)
    labels = np.asarray(labels).astype(int)

    # IMPORTANT! Handle the different lenghts of dfc time series as windowing methods will produce shorter dFC time series
    windowsize = ts_sim.shape[0] - dfc_ts.shape[2] + 1
    offset = windowsize // 2
    segments = np.asarray(segments) - offset

    index = []
    features = []
    behaviour = []

    # Get the trial segments (this only checks if we are outside the allowed bounds, otherwise we just keep all segments/labels)
    for segment, label in zip(segments, labels):
        if segment[0] > 0 and segment[-1] < dfc_ts.shape[2]: # make sure the trial is covered by the dfc data
            matrices = dfc_ts[:,:,segment]
            matrix = np.mean(matrices, axis=2) # average over the dFC estimates to reduce noise and get a single estimate for each trial

            features.append(matrix)
            behaviour.append(label)
            index.append(segment)
        else:
            raise ValueError(f"Segment {segment} not covered by data, aborting calculations.")

    index = np.asarray(index)
    features = np.asarray(features)
    behaviour = np.asarray(behaviour)

    ####################################################################
    # 4. CALCULATE GRAPH MEASURES (DECISIONS: DENSITY, BINARISATION)
    def compute_graph_measures(t, features, index, density, binarise):
        W = np.asarray(features[t, :, :]).copy()
        W = comet.graph.handle_negative_weights(W, type="absolute")
        W = comet.graph.threshold(W, type="density", density=density)
        W = comet.graph.postproc(W)

        graph_results = {{graph_measure}}

        return graph_results


    graph_results = Parallel(n_jobs=4)(delayed(compute_graph_measures)(t, features, index, {{density}}, {{binarise}}) for t in tqdm(range(features.shape[0])))

    # Unpack the results
    features = []
    for result in graph_results:
        features.append(result)

    ##############################################
    # 5. CLASSIFICATION (DECISION: SVC KERNEL)
    features = np.asarray(features)
    labels = behaviour

    # Initialize the SVC
    svc = SVC(kernel={{svc_kernel}})

    # Perform 5-fold cross-validation
    accuracy = []
    skf = StratifiedKFold(n_splits=5)

    for train_index, test_index in skf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)

        accuracy.append(accuracy_score(y_test, y_pred))

    accuracy = np.asarray(accuracy)

    # comet.data.save_universe_results will save a dictionary to .pkl file.
    # The universe argument is needed for it to know its universe number. I will have to figure out a more elegant way to do this in the future...
    result = {
        "prediction\naccuracy": accuracy
    }

    comet.data.save_universe_results(result, universe=os.path.abspath(__file__))