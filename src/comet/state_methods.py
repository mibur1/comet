from pydfc.dfc_methods import *

#from pydfc.time_series import TIME_SERIES
#from pydfc.dfc import DFC


'''
These classes bring the state based method from https://github.com/neurodatascience/dFC/ into the Comet framework
'''
class Sliding_Window(BaseDFCMethod):
    name = "Sliding Window (state-based)"
    options = {"shape": ["rectangular", "gaussian", "hamming"]}

    '''
    Sliding Window
    '''
    def __init__(self, time_series, **params):
        self.time_series = time_series
        self.logs_ = ''
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = ['measure_name', 'is_state_based', 'sw_method', 'tapered_window',
            'W', 'n_overlap', 'normalization',
            'num_select_nodes', 'num_time_point', 'Fs_ratio',
            'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
            else:
                self.params[params_name] = None

        self.params['measure_name'] = 'SlidingWindow'
        self.params['is_state_based'] = False

        assert self.params['sw_method'] in self.sw_methods_name_lst, \
            "sw_method not recognized."

    def connectivity(self):
        measure = SLIDING_WINDOW(**self.params)
        dFC = measure.estimate_dFC(time_series=self.time_series)
        return dFC
    
class Time_Freq(BaseDFCMethod):
    name = "Time-frequency (state-based)"
    options = {"shape": ["rectangular", "gaussian", "hamming"]}

    '''
    Time-Frequency
    '''
    def __init__(self, time_series, coi_correction=True, **params):
        self.time_series = time_series
        self.logs_ = ''
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = ['measure_name', 'is_state_based', 'TF_method', 'coi_correction',
            'n_jobs', 'verbose', 'backend',
            'normalization', 'num_select_nodes', 'num_time_point',
            'Fs_ratio', 'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
            else:
                self.params[params_name] = None
        
        self.params['measure_name'] = 'Time-Freq'
        self.params['is_state_based'] = False
        self.params['coi_correction'] = coi_correction

    def connectivity(self):
        measure = TIME_FREQ(**self.params)
        dFC = measure.estimate_dFC(time_series=self.time_series)
        return dFC
    
class Cap(BaseDFCMethod):
    name = "Co-activation patterns (state-based)"
    options = {"shape": ["rectangular", "gaussian", "hamming"]}

    '''
    Co-activation patterns
    '''
    def __init__(self, time_series, **params):
        self.time_series = time_series
        self.logs_ = ''
        self.FCS_ = []
        self.mean_act = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = ['measure_name', 'is_state_based', 'n_states',
            'n_subj_clstrs', 'normalization', 'num_subj', 'num_select_nodes', 'num_time_point',
            'Fs_ratio', 'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
            else:
                self.params[params_name] = None

        self.params['measure_name'] = 'CAP'
        self.params['is_state_based'] = True

    def connectivity(self, subj_id=None):
        measure = CAP(**self.params)
        measure.estimate_FCS(time_series=self.time_series)
        dFC = measure.estimate_dFC(time_series=self.time_series.get_subj_ts(subjs_id=subj_id))
        return dFC
    
class Sliding_Window_Clustr(BaseDFCMethod):
    name = "Sliding Window Clustering (state-based)"
    options = {"shape": ["rectangular", "gaussian", "hamming"]}

    '''
    Sliding Window Clustering
    '''
    def __init__(self, time_series, clstr_distance="euclidean", **params):
        self.time_series = time_series

        assert clstr_distance=='euclidean' or clstr_distance=='manhattan', \
            "Clustering distance not recognized. It must be either \
                euclidean or manhattan."
        self.logs_ = ''
        self.TPM = []
        self.FCS_ = []
        self.mean_act = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = ['measure_name', 'is_state_based', 'clstr_base_measure', 'sw_method', 'tapered_window',
            'clstr_distance', 'coi_correction',
            'n_subj_clstrs', 'W', 'n_overlap', 'n_states', 'normalization',
            'n_jobs', 'verbose', 'backend',
            'num_subj', 'num_select_nodes', 'num_time_point', 'Fs_ratio',
            'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
            else:
                self.params[params_name] = None
        
        self.params['measure_name'] = 'Clustering'
        self.params['is_state_based'] = True
        self.params['clstr_distance'] = clstr_distance

        assert self.params['clstr_base_measure'] in self.base_methods_name_lst, \
            "Base method not recognized."

    def connectivity(self, subj_id=None):
        measure = SLIDING_WINDOW_CLUSTR(**self.params)
        measure.estimate_FCS(time_series=self.time_series)
        dFC = measure.estimate_dFC(time_series=self.time_series.get_subj_ts(subjs_id=subj_id))
        return dFC
    
class Hmm_Cont(BaseDFCMethod):
    name = "Continuous Hidden Markov Model (state-based)"
    options = {"shape": ["rectangular", "gaussian", "hamming"]}

    '''
    Continuous Hidden Markov Model
    '''
    def __init__(self, time_series, clstr_distance="euclidean", **params):
        self.time_series = time_series

        self.logs_ = ''
        self.TPM = []
        self.FCS_ = []
        self.mean_act = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = ['measure_name', 'is_state_based', 'n_states', 'hmm_iter',
            'normalization', 'num_subj', 'num_select_nodes', 'num_time_point',
            'Fs_ratio', 'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
            else:
                self.params[params_name] = None

        self.params['measure_name'] = 'ContinuousHMM'
        self.params['is_state_based'] = True

    def connectivity(self, subj_id=None):
        measure = HMM_CONT(**self.params)
        measure.estimate_FCS(time_series=self.time_series)
        dFC = measure.estimate_dFC(time_series=self.time_series.get_subj_ts(subjs_id=subj_id))
        return dFC
    
class Hmm_Disc(BaseDFCMethod):
    name = "Sliding Window Clustering (state-based)"
    options = {"shape": ["rectangular", "gaussian", "hamming"]}

    '''
    Sliding Window Clustering
    '''
    def __init__(self, time_series, clstr_distance="euclidean", **params):
        self.time_series = time_series

        self.logs_ = ''
        self.TPM = []
        self.FCS_ = []
        self.mean_act = []
        self.swc = None
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        
        self.params_name_lst = ['measure_name', 'is_state_based', 'clstr_base_measure', 'sw_method', 'tapered_window',
            'dhmm_obs_state_ratio', 'coi_correction', 'hmm_iter',
            'n_jobs', 'verbose', 'backend',
            'n_subj_clstrs', 'W', 'n_overlap', 'n_states', 'normalization',
            'num_subj', 'num_select_nodes', 'num_time_point', 'Fs_ratio',
            'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
            else:
                self.params[params_name] = None
        
        self.params['measure_name'] = 'DiscreteHMM'
        self.params['is_state_based'] = True

        assert self.params['clstr_base_measure'] in self.base_methods_name_lst, \
            "Base measure not recognized."

    def connectivity(self, subj_id=None):
        measure = HMM_DISC(**self.params)
        measure.estimate_FCS(time_series=self.time_series)
        dFC = measure.estimate_dFC(time_series=self.time_series.get_subj_ts(subjs_id=subj_id))
        return dFC
    
class Windowless(BaseDFCMethod):
    name = "Windowless (state-based)"
    options = {"shape": ["rectangular", "gaussian", "hamming"]}

    '''
    Windowless
    '''
    def __init__(self, time_series, clstr_distance="euclidean", **params):
        self.time_series = time_series

        self.logs_ = ''
        self.TPM = []
        self.FCS_ = []
        self.mean_act = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = ['measure_name', 'is_state_based', 'n_states',
            'normalization', 'num_subj', 'num_select_nodes', 'num_time_point',
            'Fs_ratio', 'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
            else:
                self.params[params_name] = None

        self.params['measure_name'] = 'Windowless'
        self.params['is_state_based'] = True

    def connectivity(self, subj_id=None):
        measure = WINDOWLESS(**self.params)
        measure.estimate_FCS(time_series=self.time_series)
        dFC = measure.estimate_dFC(time_series=self.time_series.get_subj_ts(subjs_id=subj_id))
        return dFC