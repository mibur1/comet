import comet
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

#####################################
# 1. LOAD DATA AND EXTRACT PARAMETERS
data_sim = comet.utils.load_example("simulation.pkl")
ts_sim = data_sim[0] # time series data
time = data_sim[1]   # time in seconds
onsets = data_sim[2] # trial onsets in seconds
labels = data_sim[3] # trial labels (connectivity state)

###############################################
# 2. DFC CALCULATION (DECISION: DFC METHOD)
# Preprocessing. Phase-based methods require band-pass filtering, amplitude-based methods require high-pass filtering.
ts_bp = comet.utils.clean(ts_sim, confounds=None, t_r=0.72, detrend=True, standardize=False, \
                         high_pass=0.03, low_pass=0.07) # band pass (narrow-band signal for Hilbert transform)
ts_hp = comet.utils.clean(ts_sim, confounds=None, t_r=0.72, detrend=True, standardize=False, \
                         high_pass=0.01)                # high pass (for amplitude based methods)

# Estimate dFC
dfc_ts = comet.connectivity.PhaseSynchrony(ts_hp, **{'method': 'crp'}).estimate()

#######################################
# 3. SEGMENT DATA (DECISION: DELAY)
segments = []
for i in onsets:
    segment = [i+j+6 for j in range(0, 10)]
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
    G = np.asarray(features[t, :, :]).copy()
    G = comet.graph.handle_negative_weights(G, type="absolute")
    G = comet.graph.threshold(G, type="density", density=density)
    G = comet.graph.postproc(G)

    graph_results = comet.graph.participation_coef(W, **{'ci': 'louvain'})

    return graph_results

graph_results = Parallel(n_jobs=4)(delayed(compute_graph_measures)(t, features, index, 0.25, False) for t in tqdm(range(features.shape[0])))

# Unpack the results
features = []
for result in graph_results:
    features.append(result)

##############################################
# 5. CLASSIFICATION (DECISION: SVC KERNEL)
features = np.asarray(features)
labels = behaviour

# Initialize the SVC
svc = SVC(kernel=linear)

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

# Save the result (pediction accuracy for each fold)
result = {"prediction\naccuracy": accuracy}
comet.utils.save_universe_results(result)