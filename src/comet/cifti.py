import os
import sys
import urllib
import numpy as np
import pandas as pd
import nibabel as nib
nib.imageglobals.logger.setLevel(40)
import importlib_resources
from matplotlib import pyplot as plt
from collections import namedtuple
from scipy.io import loadmat
from nilearn.signal import clean

"""
SECTION: Functions for parcellating cifti data
"""
def parcellate(dtseries, atlas="schaefer_200_cortical", method=np.mean, standardize=True):
    """
    Parcellate cifti data (.dtseries.nii) using a given atlas.

    The parcellated time series is calculated as the mean over all grayordinates within a parcel.

    Parameters
    ----------
    dtseries : nibabel.cifti2.cifti2.Cifti2Image
        nibabel cifti image object

    atlas : string
        name of the atlas to use for parcellation. Options are:
            - schaefer_{x}_cortical     with x = 100, 200, 300 400, 500, 600, 700, 800, 900, 1000
            - schaefer_{x}_subcortical  with x = 100, 200, 300 400, 500, 600, 700, 800, 900, 1000
            - glasser_mmp_subcortical

    method : function
        function to use for parcellation. Only available option is np.mean

    standardize : bool
        Standardize the time series to zero (temporal) mean and unit
        standard deviation before(!) parcellation.

    Returns
    -------
    ts_parc : TxP np.ndarray
        parcellated time series data
    """

    if isinstance(dtseries, nib.cifti2.cifti2.Cifti2Image):
        ts = dtseries.get_fdata()
    elif isinstance(dtseries, np.ndarray) or isinstance(dtseries, np.memmap):
        ts = dtseries
    elif isinstance(dtseries, str):
        data = nib.load(dtseries)
        ts = data.get_fdata()
    else:
        print("Error: Input must be a nibabel cifti image object or a numpy memmap object")
        return

    rois, keys, _, _ = _get_atlas(atlas)

    # Schaefer cortical includes the medial wall which we have to insert into the data
    if atlas.startswith("schaefer") and atlas.endswith("_cortical"):
        with importlib_resources.path("comet.data.atlas", "fs_LR_32k_medial_mask.mat") as maskdir:
            medial_mask = loadmat(maskdir)['medial_mask'].squeeze().astype(bool)
        idx = np.where(medial_mask == 0)[0]

        # prepare idices and insert them into the HCP data
        for i, value in enumerate(idx):
            idx[i] = value - i

        cortical_vertices = 59412 # HCP data has 59412 cortical vertices
        ts = ts[:,:cortical_vertices]
        ts = np.insert(ts, idx, np.nan, axis=1)

    # Standardize the time series
    # TODO: Check if it should be done somewhere else
    if standardize:
        ts = _stdize(ts)

    # Parcellation
    n = np.sum(keys!=0)
    ts_parc = np.zeros((len(ts), n), dtype=ts.dtype)

    i = 0
    for k in keys:
        if k!=0:
            ts_parc[:, i] = method(ts[:, rois==k], axis=1)
            i += 1

    return ts_parc

def _get_atlas(atlas_name, debug=False):
    """
    Helper functio: Get and prepare a CIFTI-2 atlas for parcellation.

    Parameters
    ----------
    atlas_name : string
        name of the atlas to use for parcellation. Options are:
            - schaefer_{x}_cortical     with x = 100, 200, 300 400, 500, 600, 700, 800, 900, 1000
            - schaefer_{x}_subcortical  with x = 100, 200, 300 400, 500, 600, 700, 800, 900, 1000
            - glasser_mmp_subcortical
    debug : bool, optional
        Flag to provide additional debugging information. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - rois : np.ndarray
            ROI indices for each vertex.
        - keys : np.ndarray
            Keys of the atlas.
        - labels : list
            Labels of the atlas.
        - rgba : list
            RGBA values of each label.
    """
    base_urls = {
        "schaefer_cortical": "https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_{parcels}Parcels_17Networks_order.dlabel.nii",
        "schaefer_subcortical": "https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Cortex-Subcortex/Schaefer2018_{parcels}Parcels_7Networks_order_Tian_Subcortex_S4.dlabel.nii",
        "glasser_mmp_subcortical": "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors_with_Atlas_ROIs2.32k_fs_LR.dlabel.nii"
    }

    # Prepare and check atlas file names
    if "schaefer" in atlas_name:
        try:
            parts = atlas_name.split("_")
            parcels = parts[1]
            atlas_type = f"schaefer_{parts[2]}"
            url = base_urls[atlas_type].format(parcels=parcels)

        except (IndexError, KeyError):
            raise ValueError(f"Invalid atlas name format or unsupported type '{atlas_name}'.")

    elif atlas_name in base_urls:
        url = base_urls[atlas_name]

    else:
        raise ValueError(f"Atlas '{atlas_name}' not recognized. Please choose a valid atlas name.")

    atlas_file_name = f"{atlas_name}.dlabel.nii" if "schaefer" in atlas_name else base_urls[atlas_name]

    # Download (or load) the atlas
    with importlib_resources.path("comet.data.atlas", atlas_file_name) as atlas_path:
        if not atlas_path.exists():
            if "glasser" in atlas_name:
                raise FileNotFoundError(
                    f"Glasser atlas file '{atlas_file_name}' was not found\n"
                    f"  Please download manually from: https://balsa.wustl.edu/file/87B9N\n"
                    f"  and place it in the comet/src/comet/resources/atla folder.")
            else:
                urllib.request.urlretrieve(url, atlas_path)
                print(f"Atlas not available. Downloading to: {atlas_path}")

        atlas = nib.load(str(atlas_path))

    # Prepare the atlas
    # Usually for dlabel.nii files we have the following header stucture
    #       axis 0: LabelAxis
    #       axix 1: BrainModelAxis
    # print(atlas.header.get_axis(0), atlas.header.get_axis(1))

    # Roi numbers for each vertex
    rois = atlas.dataobj[0].astype(int).squeeze()

    if debug:
        brainmodelAxis = atlas.header.get_axis(1)
        for idx, (name, _slice, _bm) in enumerate(brainmodelAxis.iter_structures()):
            print(idx, str(name), _slice, _bm)

    index_map = atlas.header.get_index_map(0)
    named_map=list(index_map.named_maps)[0]
    keys = []
    labels = []
    rgba = []

    # Iterate over label_table items and get relevat values
    for key, label in named_map.label_table.items():
        labels.append(label.label)
        rgba.append(label.rgba)
        keys.append(key)

    keys = np.asarray(keys)

    return (rois, keys, labels, rgba)

def _stdize(ts):
    """
    Helper function: Standardize time series to zero (temporal) mean and unit standard deviation.

    Parameters
    ----------
    ts : np.ndarray
        Time series data

    Returns
    -------
    ts : np.ndarray
        Standardized time series data
    """

    return (ts - np.mean(ts,axis=0))/np.std(ts,axis=0)

def _get_fdata(dtseries):
    """
    Helper function: Get the cifti data

    Parameters
    ----------
    dtseries : nibabel.cifti2.cifti2.Cifti2Image
        nibabel cifti image objet

    Returns
    -------
    fdata : np.ndarray
        Time series data of the cifti image
    """

    data = nib.load(dtseries)
    return data.get_fdata()

def _get_header(dtseries):
    """
    Helper function: Get the cifti header

    Parameters
    ----------
    dtseries : nibabel.cifti2.cifti2.Cifti2Image
        nibabel cifti image objet

    Returns
    -------
    header : nibabel.cifti2.cifti2.Cifti2Header
        nibabel cifti header object
    """

    data = nib.load(dtseries)
    return data.header

def _get_labels(atlas):
    """
    Helper function: Get atlas labels and default rgba values

    Parameters
    ----------
    atlas : nibabel.cifti2.cifti2.Cifti2Image
        nibabel cifti image objet

    Returns
    -------
    labels : tuple
        Tuple containing the labes and rgba values
    """

    _, _, labels, rgba = _get_atlas(atlas)
    return labels, rgba


"""
SECTION: Class for handling HCP WM data
"""
class Hcp():
    def __init__(self, path, task="WM", TR=0.72, ts_length=405):
        assert(os.path.exists(path)), f"{path} does not seem to exist"
        self.path = path
        self.task = task
        self.TR = TR
        self.ts_length = ts_length

    ################
    # Public methods

    def get_subjects(self):
        """
        Get all subject IDs available the dataset
        """
        folder_names = []
        for entry in os.scandir(self.path):
            folder_names.append(entry.name)
        folder_names.sort(key=int)
        return folder_names

    def get_fdata(self, subject):
        """
        Get the cifti 2 data
        """
        dtseries = self._get_filenames(subject).dtseries
        data = [nib.load(run) for run in dtseries]
        return [data[0].get_fdata(), data[1].get_fdata()]

    def get_header(self, subject):
        """
        Get the cifti 2 headers
        """
        dtseries = self._get_filenames(subject).dtseries
        data = [nib.load(run) for run in dtseries]
        return [data[0].header, data[1].header]

    def get_labels(self, atlas):
        """
        Get atlas labels and default rgba values
        """
        _, _, labels, rgba = self._prepare_atlas(atlas)
        return labels, rgba

    def get_confounds(self, subject):
        """
        Get 12 motion parameter confounds
        """
        data = self._get_filenames(subject).confounds
        confounds = [np.loadtxt(file) for file in data]
        return confounds

    def get_experiment_data(self, subject):
        """
        Get WM task data
        """
        # Onset times for each condition and for all trials
        # https://www.humanconnectome.org/hcp-protocols-ya-task-fmri
        conditionFiles = self._get_filenames(subject).conditions
        conditionOnsets = {}
        for conditionFile in conditionFiles:
            if os.path.isfile(conditionFile):
                with open(conditionFile, 'r') as file:
                    line = file.readline()
                    try:
                        start = float(line.split('\t', 2)[0])
                        end = start + float(line.split('\t', 2)[1])
                        split_path = conditionFile.split("/")[-3:] # contains the last two folders and filename
                        conditionName = f"{split_path[0][-2:]}_{split_path[2][:-4]}"
                        conditionOnsets[conditionName] = [start, end]
                    except ValueError:
                        print("Error reading start and end time from EV files")

        # Split in RL/LR scans
        rl_conditions = {key[3:]: value for key, value in conditionOnsets.items() if key.startswith("RL_")}
        lr_condition = {key[3:]: value for key, value in conditionOnsets.items() if key.startswith("LR_")}
        conditionOnsets = [rl_conditions, lr_condition]

        trialFiles = self._get_filenames(subject).trials
        trialOnsets = {}
        for trialFile in trialFiles:
            if os.path.isfile(trialFile):
                with open(trialFile, 'r') as file:
                    resp = []
                    for line in file:
                        resp.append(float(line.split('\t', 2)[0]))
                    split_path = trialFile.split("/")[-3:] # contains the last two folders and filename
                    onsetName = f"{split_path[0][-2:]}_{split_path[2][:-4]}"
                    trialOnsets[onsetName] = resp

        # Split in RL/LR scans
        rl_onsets = {key[3:]: value for key, value in trialOnsets.items() if key.startswith("RL_")}
        lr_onsets = {key[3:]: value for key, value in trialOnsets.items() if key.startswith("LR_")}
        trialOnsets = [rl_onsets, lr_onsets]

        # Detailed E-Prime data
        tables = self._get_filenames(subject).tables
        columns = ["BlockType","TargetType", "StimType", "Procedure[Block]", "Stim.OnsetTime", "Cue2Back.OnsetTime", "CueTarget.OnsetTime", "Stim.RT", "Block", "Stim.ACC", "Stimulus[Block]"]
        expdata = [pd.read_csv(table, delimiter='\t')[columns] for table in tables]
        ####################### Columns: E-Prime key variables #######################
        # Trial related
        #   BlockType - 0-Back, 2-Back
        #   StimType - Tools, Body, Face, Place
        #   TargetType - nonjure, lure, target
        #
        # Time related
        #   Procedure[Block] - TRSyncPROC, Cue2BackPROC, Cue0BackPROC. TrialsPROC, Fix15secPROC
        #   Stim.OnsetTime - start of stimulus presentation per trial
        #   Cue2Back.OnsetTime - cue onset time for 2-back trials, start of block
        #   CueTarget.OnsetTime - cue onset time for 0-back trials, start of a block
        #   Stim.RT - reaction time from start of response period for each trial
        #
        # Stimulus related
        #   Block - 1, 1, 1, 1, 2, 3, 4, ..., 93
        #   Stim.ACC - 1 or 0 (correct/incorrect trials)
        #   Stimulus[Block] - filename of the stimulus
        #
        # Comments: - CorrectResponse marks correct button, Stim.RESP, Stim.CRESP could be used to check which kind of error was made
        #           - These variables are currently not in use
        ##############################################################################

        return conditionOnsets, trialOnsets, expdata

    def segment_experiment(self, conditionOnset, trialOnset, delay=0):
        """
        Calculate indices for 3 TRs after each stimulus onset to be used for time series segmentation.
        This is the maximum amount one can get without an overlap of datapoints.
        """
        cs = []
        ics = []
        csb = []
        icsb = []

        for conditionOnsets, trialOnsets in zip(conditionOnset, trialOnset):
            correct_segments = np.zeros((len(trialOnsets['all_bk_cor']), 3)).astype(int)
            incorrect_segments = np.zeros((len(trialOnsets['all_bk_err']), 3)).astype(int)

            # Transform onset times into "TR time" and get the following three indices
            for t, onset in enumerate(trialOnsets['all_bk_cor']):
                correct_segments[t,:] = np.linspace(np.ceil(onset/self.TR) + delay, np.ceil(onset/self.TR) + delay + 2, 3)

            for t, onset in enumerate(trialOnsets['all_bk_err']):
                incorrect_segments[t,:] = np.linspace(np.ceil(onset/self.TR)+ delay, np.ceil(onset/self.TR) + delay + 2, 3)

            # Get information about which condition/block the segments belong to (so we can later separate 0- and 2-back trials)
            correct_segments_block = np.zeros((len(trialOnsets['all_bk_cor']))).astype(int)
            incorrect_segments_block = np.zeros((len(trialOnsets['all_bk_err']))).astype(int)
            for blockname in conditionOnsets:
                onset = (conditionOnsets[blockname][0]/self.TR) + delay
                offset = (conditionOnsets[blockname][1]/self.TR) + delay

                for i in range(correct_segments.shape[0]):
                    segTimeCorr = correct_segments[i,0]
                    if segTimeCorr > onset and segTimeCorr < offset:
                        correct_segments_block[i] = int(blockname[0])

                for i in range(incorrect_segments.shape[0]):
                    segTimeIncorr = incorrect_segments[i,0]
                    if segTimeIncorr > onset and segTimeIncorr < offset:
                        incorrect_segments_block[i] = int(blockname[0])

            cs.append(correct_segments)
            ics.append(incorrect_segments)
            csb.append(correct_segments_block)
            icsb.append(incorrect_segments_block)

        #return correct_segments, incorrect_segments, correct_segments_block, incorrect_segments_block
        return cs, ics, csb, icsb

    def get_segments(self, subject, condition=None, delay=0):
        """
        The data is segmented into individual trials. Individual conditions and the delay for the segments can be specified.
        returns: 3 TR indices and the labels for every trial
        """
        ### Behavioral data ###
        # Time (in seconds) for the onset of all blocks/conditions and individual trials as dicts, read from the EVs .txt files (sorted in correct/incorrect responses)
        conditionOnsets, trialOnsets, expData = self.get_experiment_data(subject)

        # Using the time stamps and labels, we get the corresponding indices for TRs during which the responses happened
        # We get indices for 3 TRs after the response and also the corresponding label (correct -> 1, incorrect -> 0), this is the maximum amount of data without overlap
        # The delay argument can shift the indices to the right to account for hemodynamic delay
        correct_segments, incorrect_segments, correct_blocks, incorrect_blocks = self.segment_experiment(conditionOnsets, trialOnsets, delay=delay)

        # Plot the experiment
        #hcp.plot_experiment(subject, conditionOnsets, trialOnsets)

        # Combine the segments and labels in one matrix
        combined_segments = [np.vstack([cs, ics]) for cs, ics in zip(correct_segments, incorrect_segments)]
        combined_labels = [np.hstack([np.ones(len(cs)), np.zeros(len(ics))]) for cs, ics in zip(correct_segments, incorrect_segments)]

        # We can specify the model to only include 0-back or 2-back trials by taking this subset here
        combined_blocks = [np.hstack([cb, icb]) for cb, icb in zip(correct_blocks, incorrect_blocks)]

        if condition == "2back":
            nbk_mask = [(cb == 2) for cb in combined_blocks]
        elif condition == "0back":
            nbk_mask = [(cb == 0) for cb in combined_blocks]
        elif condition == "both":
            return combined_segments, combined_labels
        else:
            print("Error: Condition must be one of 2back, 0back, or both")
            quit()

        blocked_segments = [cb[msk] for cb, msk in zip(combined_segments, nbk_mask)]
        blocked_labels = [cl[msk] for cl, msk in zip(combined_labels, nbk_mask)]

        return blocked_segments, blocked_labels

    def clean(self, time_series, runs=None, detrend=False, confounds=None, standardize=False, standardize_confounds=True, filter='butterworth', low_pass=None, high_pass=None, t_r=0.72, ensure_finite=False):
        """
        Standard nilearn cleaning of the time series
        """
        ts_clean = []
        for ts, conf in zip(time_series, confounds):
            ts_clean.append(clean(ts, detrend=detrend, confounds=conf, standardize=standardize, standardize_confounds=standardize_confounds, filter=filter, low_pass=low_pass, high_pass=high_pass, t_r=t_r, ensure_finite=ensure_finite))
        return ts_clean

    def plot_experiment(self, subject):
        """
        Plot the exeriment time series for a subject
        """
        fig, ax = plt.subplots(2,1, figsize=(18,10))
        fontsize = 12
        fig.suptitle(f"Working memory task for subject {subject}", fontweight="bold", fontsize=fontsize)
        titles = ["Scan 1 (RL)", "Scan 2 (LR)"]
        blockOnsets, trialOnsets, expData = self.get_experiment_data(subject)

        # Plot both scans
        for i in range(2):
            # Define  parameters
            scan = [i+1 for i in range(self.ts_length)]
            runTime = [self.TR + i * self.TR for i in range(self.ts_length)]
            event = [None] * self.ts_length
            response_cor = []
            response_err = []
            response_nlr = []

            # Onset for each block (+ estimated end point 27.5s later), code all TRs in between
            for block, blockTime in blockOnsets[i].items():
                for t in range(self.ts_length):
                    if runTime[t] > blockTime[0] and runTime[t] < blockTime[1]:
                        event[t] = block

            # Trial onset times
            for block, onset in trialOnsets[i].items():
                if block == "0bk_cor" or block == "2bk_cor":
                    response_cor.append(onset)
                elif block == "0bk_err" or block == "2bk_err":
                    response_err.append(onset)
                else:
                    response_nlr.append(onset)

            # Scale onset times with TR length (720ms) for plotting
            response_cor = np.concatenate(response_cor)
            response_cor = response_cor / self.TR
            response_err = np.concatenate(response_err)
            response_err = response_err / self.TR
            response_nlr = np.concatenate(response_nlr)
            response_nlr = response_nlr / self.TR

            # Create a dataframe for plotting
            df = pd.DataFrame({'scan': scan, 'time': runTime, 'event': event})
            task_dict = {None: 0, "0-Back": 1, "2-Back": 2}
            color_dict = {0: "#A9A9A9", 1: "#2ca02c", 2: "#1f77b4"}
            task_list =[]

            for value in df['event']:
                if value is None:
                    task_list.append(task_dict[value])
                elif value.startswith('0bk_'):
                    task_list.append(task_dict["0-Back"])
                elif value.startswith('2bk_'):
                    task_list.append(task_dict["2-Back"])

            # Plotting
            ax[i].tick_params(labelsize=fontsize)
            ax[i].set_xlim(0,432)
            ax[i].set_title(titles[i], y=0.95)

            # Create horizontal barplots with widths of 1 and stack them to create "time series"-like visualization
            for t in range(len(df)):
                block_type = task_list[t]
                ax[i].barh(y = 1, width=1, left=t, color=color_dict[block_type])

            # Indivdual scan times
            #timetr = np.linspace(0,404,405)
            #ax[i].vlines(timetr, 0.6, 1.4, color="black", linestyles="")

            # Stimulus onsets
            ax[i].vlines(response_cor, 0.6, 1.4, color="darkgreen", linewidth=2)
            ax[i].vlines(response_err, 0.6, 1.4, color="firebrick", linewidth=2)


            # Dummy bars for the legend
            bar_0back = ax[i].barh(y = 1, width=0, left=0, color=color_dict[1])
            bar_2back = ax[i].barh(y = 1, width=0, left=0, color=color_dict[2])

            # Remove border
            for spine in ['left', 'right', 'top', 'bottom']:
                ax[i].spines[spine].set_visible(False)

            # Remove axis labels and ticks
            ax[i].yaxis.set_ticks_position('none')
            ax[i].set_yticklabels([])

        # Figure legend
        ax[0].legend([bar_0back, bar_2back], ["0-Back", "2-Back"], title="Task blocks:")

        plt.xlabel("TRs", fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(sys.path[0]+f"/images/trial_data/{subject}.jpg")

        return fig, ax

    ##################
    # Internal methods

    def _get_filenames(self, subject=None):
        """
        Get the path for the dtseries files for a specified subject
        """
        filedir = os.path.dirname(os.path.realpath(__file__))
        maindirs = (f"{self.path}/{subject}/MNINonLinear/Results/tfMRI_WM_RL/", f"{self.path}/{subject}/MNINonLinear/Results/tfMRI_WM_LR/")
        runs = ("tfMRI_WM_RL_Atlas_MSMAll.dtseries.nii", "tfMRI_WM_LR_Atlas_MSMAll.dtseries.nii")

        # data files
        dtseries = [f"{dir}{run}" for dir, run in zip(maindirs, runs)]
        confounds = [f"{dir}Movement_Regressors.txt" for dir in maindirs]

        # event related files
        EVs = [f"{dir}EVs/" for dir in maindirs]
        conditionFiles = ["2bk_tools.txt", "0bk_body.txt", "2bk_faces.txt", "0bk_tools.txt", "2bk_body.txt", "2bk_places.txt", "0bk_faces.txt", "0bk_places.txt"]
        trialFiles = ["0bk_cor.txt", "0bk_err.txt", "0bk_nlr.txt", "2bk_cor.txt", "2bk_err.txt", "2bk_nlr.txt", "all_bk_cor.txt", "all_bk_err.txt"]

        conditions = [ev + conditionFile for ev in EVs for conditionFile in conditionFiles]
        trials = [ev + trialFile for ev in EVs for trialFile in trialFiles]

        tabnames = ["WM_run1_TAB.txt", "WM_run2_TAB.txt", "WM_run3_TAB.txt", "WM_run4_TAB.txt"]
        #tables = [f"{dir}{tabname}" for dir, tabname in zip(maindirs, tabnames)]
        tables = [f"{dir}{tabname}" for dir in maindirs for tabname in tabnames if os.path.exists(f"{dir}{tabname}")]

        # Store everything in a named tuple for dot notation accessability
        Data = namedtuple("Data", ["filedir", "dtseries", "confounds", "conditions", "trials", "tables"])
        filenames = Data(filedir, dtseries, confounds, conditions, trials, tables)

        return filenames