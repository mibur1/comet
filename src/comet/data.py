import os
import sys
import numpy as np
import pandas as pd
import pkg_resources
import nibabel as nib
nib.imageglobals.logger.setLevel(40)
from scipy.io import loadmat
from matplotlib import pyplot as plt
from nilearn import signal
from collections import namedtuple
from .methods import *
import importlib.resources as pkg_resources

def load_example():
    """
    Simulated time series for testing purposes
    """
    with pkg_resources.path("voyager.example_data", "simulation.txt") as file_path:
        data = np.loadtxt(file_path)
    return data

def clean(time_series, runs=None, detrend=False, confounds=None, standardize=False, standardize_confounds=True, filter='butterworth', low_pass=None, high_pass=None, t_r=0.72, ensure_finite=False):
    """
    Standard nilearn cleaning of the time series
    """
    return signal.clean(time_series, detrend=detrend, confounds=confounds, standardize=standardize, standardize_confounds=standardize_confounds, filter=filter, low_pass=low_pass, high_pass=high_pass, t_r=t_r, ensure_finite=ensure_finite)

        

class Hcp():
    def __init__(self, path, task="WM", TR=0.72, ts_length=405):
        assert(os.path.exists(path)), f"{path} does not seem to exist"
        self.path = path
        self.task = task
        self.TR = TR
        self.ts_length = ts_length
    
    ###############################################
    # Internal methods that the user shouldn't need
    
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
    
    def _prepare_atlas(self, atlas_name, debug=False):
        """
        Prepare a cifti 2 atlas to be used in parcellation
        """
        atlasdir = self._get_filenames().filedir
        if atlas_name == "glasser":
            atlas = nib.load(f"{atlasdir}/atlas/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors_with_Atlas_ROIs2.32k_fs_LR.dlabel.nii")
        elif atlas_name == "schaefer_kong":
            atlas = nib.load(f"{atlasdir}/atlas/Schaefer2018_200Parcels_Kong2022_17Networks_order.dlabel.nii")
        elif atlas_name == "schaefer_tian":
            atlas = nib.load(f"{atlasdir}/atlas/Schaefer2018_200Parcels_17Networks_order_Tian_Subcortex_S4.dlabel.nii")
        else:
            sys.exit("Atlas must be any of glasser, schaefer_kong, or schaefer_tian")
        
        # Usually for dlabel.nii files we have the following header stucture
        #       axis 0: LabelAxis
        #       axix 1: BrainModelAxis
        # print(atlas.header.get_axis(0), atlas.header.get_axis(1))

        # Roi numbers for each vertex
        rois = atlas.dataobj[0].astype(int).squeeze()

        if debug:
            brainmodelAxis = atlas.header.get_axis(1)
            for idx, (name, slice, bm) in enumerate(brainmodelAxis.iter_structures()):
                print(idx, str(name), slice)

        index_map = atlas.header.get_index_map(0)
        named_map=list(index_map.named_maps)[0]

        keys = []
        labels = []
        rgba = []
  
        # Iterate over label_table and get relevat values
        for i in range(len(named_map.label_table)):
            roi = named_map.label_table[i]
            labels.append(roi.label)
            rgba.append(roi.rgba)
            keys.append(roi.key)
        keys = np.asarray(keys)
        
        return (rois, keys, labels, rgba)
    
    def _standardize(self, ts):
        """
        Standardize to zero (temporal) mean and unit standard deviation.
        """
        return (ts - np.mean(ts,axis=0))/np.std(ts,axis=0)
    
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
 
    def parcellate(self, time_series, atlas="glasser", method=np.mean, standardize=True):
        """
        Parcellation, calculates the mean over all grayordinates within a parcel. Z-standardization is enabled by default
        """
        rois, keys, labels, rgba = self._prepare_atlas(atlas)
        maskdir = self._get_filenames().filedir
        
        ts_parcellated = []
        # We have two runs for each subject (RL, LR)
        for ts in time_series:
            # schaefer_kong includes the medial wall which we have to insert into the data
            if atlas == "schaefer_kong":
                medial_mask = loadmat(f"{maskdir}/atlas/fs_LR_32k_medial_mask.mat")['medial_mask'].squeeze().astype(bool)
                idx = np.where(medial_mask == 0)[0]

                # prepare idices and insert them into the HCP data
                for i in range(len(idx)):
                    idx[i] = idx[i] - i

                cortical_vertices = 59412 # HCP data has 59412 cortical vertices
                ts = ts[:,:cortical_vertices]
                ts = np.insert(ts, idx, np.nan, axis=1)

            # Standardize the time series
            # TODO: Check if it should be done somewhere else
            if standardize:
                ts = self._standardize(ts)
            
            # Parcellation
            n = np.sum(keys!=0)
            ts_parc = np.zeros((len(ts), n), dtype=ts.dtype)

            i = 0
            for k in keys:
                if k!=0:
                    ts_parc[:, i] = method(ts[:, rois==k], axis=1)
                    i += 1

            ts_parcellated.append(ts_parc)

        return ts_parcellated
    
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
    
    def get_dfc(self, time_series, method, params=None):
        """
        Calculate dynamic functional connectivity matrices for one of the specified methods.
        Required parameters can be passed as a dict:
        params = {
            "windowsize" : 15,
            "windowshape": "gaussian",
            "distance"   : "euclidean"}
        """
        if method == "sw":
            dfc = [SlidingWindow(ts, windowsize=params['windowsize'], shape=params['windowshape']).connectivity() for ts in time_series]
        if method == "jackknife":
            dfc = [Jackknife(ts).connectivity() for ts in time_series]
        if method == "mtd":
            dfc = [TemporalDerivatives(ts, windowsize=params['windowsize']).connectivity() for ts in time_series]
        if method == "spatialdist":
            dfc = [SpatialDistance(ts, params['distance']).connectivity() for ts in time_series]
        if method == "phasesync":
            dfc = [PhaseSynchrony(ts).connectivity() for ts in time_series]
        if 'dfc' not in locals():
            raise ValueError("The method must be any of sw, jackknife, spatialdist, mtd, or phasesync")

        return dfc
    
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
    
    ####################
    # Plotting functions

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
    
    def plot_dfc(self, dfc_estimates, time_point):
        """
        Plot the DFC matrix at a given timepoint
        """
        fig, ax = plt.subplots()
        cax = ax.imshow(dfc_estimates[:,:,time_point])
        fig.colorbar(cax)
        ax.set_title(f"DFC estimate (t={time_point})")
        plt.show()

        return fig, ax