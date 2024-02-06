import comet
import os
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

#####################################
# 1. LOAD DATA AND EXTRACT PARAMETERS
data_sim = comet.data.load_example(type="pkl")
ts_sim = data_sim[0] # time series data
time = data_sim[1]   # time in seconds
onsets = data_sim[2] # trial onsets in seconds
labels = data_sim[3] # trial labels (connectivity state)


###############################################
# 2. DFC CALCULATION (ECISION: DFC METHOD)
# Preprocessing. Phase-based methods require band-pass filtering, amplitude-based methods require high-pass filtering.
ts_bp = comet.data.clean(ts_sim, confounds=None, t_r=0.72, detrend=True, standardize=False, high_pass=0.03, low_pass=0.07) # band pass (narrow-band signal for Hilbert transform)
ts_hp = comet.data.clean(ts_sim, confounds=None, t_r=0.72, detrend=True, standardize=False, high_pass=0.01)                # high pass (for amplitude based methods)
dfc_ts = comet.methods.SpatialDistance(ts_hp, **{'dist': 'euclidean'}).connectivity() # estimate dFC
   

#######################################
# 3. SEGMENT DATA (DECISION: DELAY) 
segments = []
for i in onsets:
    segment = [i+j+10 for j in range(0, 10)]
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
graph_results = Parallel(n_jobs=8)(delayed(comet.graph.compute_graph_measures)(t, features, index, 0.25, True) for t in tqdm(range(features.shape[0])))

# Unpack the results
participation, clustering, efficiency = [], [], []
for result in graph_results:
    participation.append(result["participation"])
    clustering.append(result["clustering"])
    efficiency.append(result["efficiency"])

# Save as array and make sure we have no NaNs or infs
participation = np.nan_to_num(np.asarray(participation), nan=0.0, posinf=0.0, neginf=0.0)
clustering = np.nan_to_num(np.asarray(clustering), nan=0.0, posinf=0.0, neginf=0.0)
efficiency = np.nan_to_num(np.asarray(efficiency), nan=0.0, posinf=0.0, neginf=0.0)


##############################################
# 5. CLASSIFICATION (DECISION: SVC KERNEL)
features = np.hstack([participation, clustering, efficiency])
labels = behaviour

# Initialize the SVC
svc = SVC(kernel='linear')
    
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

# comet.data.save_results will save any data to a .pkl file.
# The universe argument is needed for it to know its universe number, will have to figure out a more elegant way to do this in the future...
comet.data.save_results(accuracy, universe=os.path.basename(__file__))
print("Done.")