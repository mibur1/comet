# Script for running the multiverse analysis.
# This file is used/overwritten by the GUI, users should usually interact with their own multiverse script (located one folder above).

from comet.multiverse import Multiverse

forking_paths = {
    "pipeline": ['cpac', 'ccs', 'dparsf', 'niak'],
    "parcellation": ['rois_aal', 'rois_cc200', 'rois_dosenbach160'],
    "band_pass": [True, False],
    "global_signal": [True, False],
    "connectivity": [
    {
        "name": "pearson",
        "func": "comet.connectivity.Static_Pearson(ts).estimate()"
    },
    {
        "name": "partial",
        "func": "comet.connectivity.Static_Partial(ts).estimate()"
    }
],
    "regularisation": [0.25, 1.0],
}

def analysis_template():
    import comet
    import numpy as np
    from nilearn import datasets
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    # List of subjects
    SUB_IDS = [50012, 50014, 50015, 50016, 50020, 50022, 50023, 50024, 50025, 50027, # controls
               50030, 50031, 50032, 50033, 50034, 50035, 50036, 50037, 50038, 50040] # autism
    #SUB_IDS = None # Use all subjects

    # Get data (if available, it will be loaded from disk)
    data = datasets.fetch_abide_pcp(SUB_ID=SUB_IDS, data_dir="/home/mibur/comet/tutorials/abide_data", verbose=0, 
                                    pipeline={{pipeline}},
                                    derivatives={{parcellation}},
                                    band_pass_filtering={{band_pass}},
                                    global_signal_regression={{global_signal}})

    time_series = data[{{parcellation}}]
    diagnosis = data["phenotypic"]["DX_GROUP"]

    # Calculate FC
    tri_ix = None
    features = []

    for ts in time_series:
        FC = {{connectivity}}

        if tri_ix == None:
            tri_ix = np.triu_indices_from(FC, k=1)
        
        feat_vec = FC[tri_ix]
        features.append(feat_vec)

    # Prepare features (FC estimates) and target (autism/control)
    X = np.vstack(features)
    X[np.isnan(X)] = 0.0
    y = np.array(diagnosis)

    # Classification model
    model = Pipeline([('scaler', StandardScaler()), ('reg', LogisticRegression(penalty='l2', C={{regularisation}}, tol=1e-3))])
    cv = StratifiedKFold(n_splits=5)
    accuracies = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    # Save the results
    comet.utils.save_universe_results({"accuracy": accuracies})