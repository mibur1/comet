import os
import random
import numpy as np
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
from scipy.stats import zscore
from scipy.spatial import distance
from scipy.signal import windows, hilbert
from scipy.linalg import eigh, solve, inv
from scipy.optimize import minimize
from sklearn.metrics import mutual_info_score
from statsmodels.stats.weightstats import DescrStatsW
from pycwt import cwt, Morlet
from pydfc.dfc_methods import *
from typing import Literal

from joblib import Parallel, delayed
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

'''
Abstract class template for all dynamic functional connectivity methods
Abstract methods need to be overriden in the child classes
'''
class ConnectivityMethod(metaclass=ABCMeta):
    def __init__(self, time_series, diagonal=0, standardize=False, fisher_z=False, tril=False):
        self.time_series = time_series.astype("float32")
        self.T = time_series.shape[0] # T timepoints
        self.P = time_series.shape[1] # P parcels
        self.diagonal = diagonal
        self.standardize = standardize
        self.fisher_z = fisher_z
        self.tril = tril

    @abstractmethod
    def connectivity(self):
        raise NotImplementedError("This method should be implemented in each child class.")

    def postproc(self, R_mat):
        # Fisher z-transformation
        if self.fisher_z:
            R_mat = np.clip(R_mat, -1 + np.finfo(float).eps, 1 - np.finfo(float).eps)
            R_mat = np.arctanh(R_mat)
        
        # z-standardize
        if self.standardize and len(R_mat.shape) == 3:
            R_mat = zscore(R_mat, axis=2)
        
        # Set main diagonal
        if len(R_mat.shape) == 3:
            for estimate in range(R_mat.shape[2]):
                np.fill_diagonal(R_mat[:,:, estimate], self.diagonal)
        else:
            np.fill_diagonal(R_mat, self.diagonal)
        
        # Get lower triangle to save space
        if self.tril:
            if len(R_mat.shape) == 3:
                mask = np.tril(R_mat[:, :, 0], k=-1) != 0
                R_mat = np.array([matrix_2d[mask] for matrix_2d in R_mat.transpose(2, 0, 1)])
            else:
                mask = np.tril(R_mat, k=-1) != 0
                R_mat = R_mat[mask]

        return R_mat

'''
Continuous dFC methods
'''
class SlidingWindow(ConnectivityMethod):
    name = "CONT Sliding Window"
    options = {"shape": ["rectangular", "gaussian", "hamming"]}

    '''
    Sliding Window
        Most widely used method, which involves sliding a window over the data.
        Cavariance is estimated for each windowed section.
    '''
    def __init__(self,
                 time_series: np.ndarray,
                 windowsize: int = 29,
                 shape: Literal["rectangular", "gaussian", "hamming"] = "rectangular",
                 std: float = 10,
                 diagonal: int = 0,
                 standardize: bool = False,
                 fisher_z: bool = False,
                 tril: bool = False):
        
        super().__init__(time_series, diagonal, standardize, fisher_z, tril)
        self.windowsize = windowsize
        self.shape = shape
        self.std = std
        
        self.N_estimates = self.T - self.windowsize + 1 # N possible estimates given the window size
        self.R_mat = np.full((self.P,self.P, self.N_estimates), np.nan)

        assert self.windowsize <= self.T, "windowsize is larger than time series"

    def _weights(self):
        '''
        Create weights for sliding window. Can be:
            - rectangular
            - gaussian
            - hamming
            - TODO: rectangular convolved with gaussian
        '''
        if self.shape == 'rectangular':
            weights_init = np.zeros(self.T)
            weights_init[0:self.windowsize] = np.ones(self.windowsize)
            weights = np.array([np.roll(weights_init, i) for i in range(0, self.T + 1 - self.windowsize)])     
        
        if self.shape == 'gaussian':
            weights_init = np.zeros(self.T)
            weights_init[0:self.windowsize] = windows.gaussian(self.windowsize, self.std)
            weights = np.array([np.roll(weights_init, i) for i in range(0, self.T + 1 - self.windowsize)])    

        if self.shape == 'hamming':
            weights_init = np.zeros(self.T)
            weights_init[0:self.windowsize] = windows.hamming(self.windowsize)
            weights = np.array([np.roll(weights_init, i) for i in range(0, self.T + 1 - self.windowsize)]) 
    
        return weights

    def connectivity(self):
        '''
        Calculate sliding window correlation
        '''
        print("Calculating Sliding Window Correlation, please wait...")
        weights = self._weights()
        self.R_mat = np.full((self.P,self.P, self.N_estimates), np.nan) 

        for estimate in range(self.N_estimates):
            self.R_mat[:,:,estimate] = DescrStatsW(self.time_series, weights[estimate, :]).corrcoef

        self.R_mat = self.postproc(self.R_mat)

        return self.R_mat

class Jackknife(ConnectivityMethod):
    name = "CONT Jackknife Correlation"
    options = {}
    '''
    Jackknife correlation:
        Richter CG, Thompson WH, Bosman CA, Fries P. A jackknife approach to quantifying single-trial 
        correlation between covariance-based metrics undefined on a single-trial basis.
        https://doi.org/10.1016/j.neuroimage.2015.04.040
    '''
    def __init__(self,
                 time_series: np.ndarray,
                 windowsize: int = 1,
                 diagonal: int = 0,
                 standardize: bool = False,
                 fisher_z: bool = False,
                 tril: bool = False):
        
        super().__init__(time_series, diagonal, standardize, fisher_z, tril)
        self.windowsize = windowsize

        self.N_estimates = self.T - self.windowsize + 1 # N possible estimates given the window size
        self.R_mat = np.full((self.P,self.P, self.N_estimates), np.nan)

        assert self.windowsize <= self.T, "windowsize is larger than time series"
    
    def _weights(self):
        '''
        Create logical weight vectors for jackknife correlation
        Example: 01111 -> 10111 -> 11011 -> 11101 -> 1110
        '''
        weights_init = np.ones(self.T)
        weights_init[0:self.windowsize] = 0
        weights = np.array([np.roll(weights_init, i) for i in range(self.N_estimates)])
        return weights.astype(bool)

    def connectivity(self):
        '''
        Calculate jackknife correlation
        '''
        print("Calculating Jackknife Correlation, please wait...")
        weights = self._weights()

        for estimate in range(self.N_estimates):
            ts_jackknife = self.time_series[weights[estimate,:],:]
            self.R_mat[:,:,estimate] = np.corrcoef(ts_jackknife.T) * -1 # correlation estimation and sign inversion
        
        self.R_mat = self.postproc(self.R_mat)

        return self.R_mat

class SpatialDistance(ConnectivityMethod):
    name = "CONT Spatial Distance"
    options = {"dist": ["euclidean"]}
    '''
    Spatial Distance:
        William Hedley Thompson, Per Brantefors, Peter Fransson. From static 
        to temporal network theory: Applications to functional brain connectivity. 
        https://doi.org/10.1162/NETN_a_00011
    '''
    def __init__(self,
                 time_series: np.ndarray,
                 dist: Literal["euclidean", "cosine", "cityblock"] = "euclidean",
                 diagonal: int = 0,
                 standardize: bool = False,
                 fisher_z: bool = False,
                 tril: bool = False):
        
        super().__init__(time_series, diagonal, standardize, fisher_z, tril)
        self.distance = self._distance_functions(dist)

        self.N_estimates = self.T
        self.R_mat = np.full((self.P,self.P, self.N_estimates), np.nan)
    
    def _distance_functions(self, dist):
        options = {'euclidean': distance.euclidean,
                   'cosine'   : distance.cosine,
                   'cityblock': distance.cityblock}
        return options[dist]
    
    def _weights(self):
        '''
        Calculate adjacency matrix for spatial distance
        '''
        weights = np.array([self.distance(self.time_series[n, :], self.time_series[t, :]) for n in np.arange(0, self.T) for t in np.arange(0, self.T)])
        weights = np.reshape(weights, [self.T, self.T])
        np.fill_diagonal(weights, np.nan)
        weights = 1 / weights
        weights = (weights - np.nanmin(weights)) / (np.nanmax(weights) - np.nanmin(weights))
        np.fill_diagonal(weights, 1)
        return weights

    def connectivity(self):
        '''
        Calculate spatial distance correlation
        '''
        print("Calculating Spatial Distance, please wait...")
        weights = self._weights() # in this case this is the distance matrix

        for estimate in range(self.N_estimates):
            self.R_mat[:,:,estimate] = DescrStatsW(self.time_series, weights[estimate, :]).corrcoef
        
        self.R_mat = self.postproc(self.R_mat)

        return self.R_mat

class TemporalDerivatives(ConnectivityMethod):
    name = "CONT Multiplication of Temporal Derivatives"
    options = {}
    '''
    Multiplication of temporal derivatives:
        Shine JM, Koyejo O, Bell PT, Gorgolewski KJ, Gilat M, Poldrack RA. Estimation of
        dynamic functional connectivity using Multiplication of Temporal Derivatives.
        https://doi.org/10.1016/j.neuroimage.2015.07.064.
    '''
    def __init__(self,
                 time_series: np.ndarray,
                 windowsize: int = 7,
                 diagonal: int = 0,
                 standardize: bool = False,
                 fisher_z: bool = False,
                 tril: bool = False):
        
        super().__init__(time_series, diagonal, standardize, fisher_z, tril)
        self.windowsize = windowsize
        
        self.N_estimates = self.T - self.windowsize
        self.R_mat = np.full((self.P,self.P, self.N_estimates), np.nan)

        assert self.windowsize <= self.T, "windowsize is larger than time series"

    def connectivity(self):
        '''
        Calculate multiplication of temproral derivatives
        '''
        print("Calculating Multiplication of Temporal Derivatives, please wait...")
        derivatives = self.time_series[1:, :] - self.time_series[:-1, :]
        derivatives = derivatives / np.std(derivatives, axis=0)
        coupling = np.array([derivatives[:, i] * derivatives[:, j] for i in range(self.P) for j in range(self.P)]) # multiplicative coupling

        # Convolve with rectangular kernel for smoothing
        kernel = np.ones(self.windowsize) / self.windowsize
        smooth_coupling = np.full((self.P * self.P, self.N_estimates), np.nan)
        
        for i in range(self.P * self.P):
            smooth_coupling[i,:] = np.convolve(coupling[i,:], kernel, mode='valid')

        smooth_coupling = np.reshape(smooth_coupling, [self.P, self.P, self.N_estimates]) #reshape into PxPxN connectivity matrix
        self.R_mat = self.postproc(smooth_coupling)

        return self.R_mat

class FlexibleLeastSquares(ConnectivityMethod):
    name = "CONT Flexible Least Squares"
    options = {}
    '''
    Flexible Least Squares:
        Liao, W., Wu, G. R., Xu, Q., Ji, G. J., Zhang, Z., Zang, Y. F., & Lu, G. (2014). 
        DynamicBC: a MATLAB toolbox for dynamic brain connectome analysis. Brain connectivity, 4(10), 780-790.
        https://doi.org/10.1089/brain.2014.0253
    '''
    def __init__(self,
                 time_series: np.ndarray,
                 standardizeData: bool = True,
                 mu: float = 100,
                 num_cores: int = 16,
                 diagonal: int = 0,
                 standardize: bool = False,
                 fisher_z: bool = False,
                 tril: bool = False):
        
        super().__init__(time_series, diagonal, standardize, fisher_z, tril)
        self.N_estimates = self.T
        self.R_mat = np.full((self.P,self.P, self.N_estimates), np.nan)
        self.standardizeData = standardizeData
        self.mu = mu
        self.num_cores = num_cores

    def _calculateBetas(self, X, y):
        N, K = X.shape
        G = np.zeros((N*K, N))
        A = np.zeros((N*K, N*K))
        mui = self.mu * np.eye(K)
        ind = np.arange(K)

        for i in range(N):
            G[ind, i] = X[i, :]
            if i == 0:
                Ai = X[i, :].T @ X[i, :] + mui
                A[ind, ind] = Ai
                A[ind, ind + K] = -mui
            elif i != N - 1:
                Ai = X[i, :].T @ X[i, :] + 2 * mui
                A[ind, ind] = Ai
                A[ind, ind + K] = -mui
                A[ind, ind - K] = -mui
            else:
                Ai = X[i, :].T @ X[i, :] + mui
                A[ind, ind] = Ai
                A[ind, ind - K] = -mui
            ind += K

        betas = solve(A, G.T @ y)
        betas = np.reshape(betas, (K, N)).T

        return betas.squeeze()
    
    def _calculateBetasForPair(self, i):
        beta_i = np.zeros((self.P, self.T))
        for j in range(self.P):
            beta_i[j, :] = self._calculateBetas(self.time_series[:, i].reshape(-1,1), self.time_series[:, j].reshape(-1,1))
        return i, beta_i

    # Flexible least squares algorithm as implemented in the DynamicBC toolbox
    def connectivity(self):
        print("Calculating Flexible Least Squares, please wait...")
        # Standardize time series
        if self.standardizeData:
            self.time_series = (self.time_series - np.mean(self.time_series, axis=0)) / np.std(self.time_series, axis=0)

        # Calculate functional connectivity
        T, P = self.time_series.shape
        beta = np.zeros((P,P,T))

        # FLS beta estimation
        results = Parallel(n_jobs=self.num_cores)(delayed(self._calculateBetasForPair)(i) for i in tqdm(range(self.P)))
        
        for i, beta_i in results:
            beta[i, :, :] = beta_i

        # Symmetrize and return the resulting 3D array
        for i in range(T):
            beta[:, :, i] = (beta[:, :, i] + beta[:, :, i].T) / 2

        beta = self.postproc(beta)

        return beta

class PhaseSynchrony(ConnectivityMethod):
    name = "CONT Phase Synchronization"
    options = {"method": ["crp", "pcoh", "teneto"]}
    '''
    Instantaneous Phase Synchrony:
        Honari, H., Choe, A. S., & Lindquist, M. A. (2021). Evaluating phase synchronization methods in fMRI: 
        A comparison study and new approaches. NeuroImage, 228, 117704. https://doi.org/10.1016/j.neuroimage.2020.117704
    '''
    def __init__(self,
                 time_series: np.ndarray,
                 method: Literal["crp", "pcoh", "teneto"] = "crp",
                 diagonal: int = 0,
                 standardize: bool = False,
                 fisher_z: bool = False,
                 tril: bool = False):
        
        super().__init__(time_series, diagonal, standardize, fisher_z, tril)
        self.N_estimates = self.T
        self.R_mat = np.full((self.P,self.P, self.N_estimates), np.nan)
        self.method = method

    def connectivity(self):
        '''
        Calculate instantaneous phase synchrony
        CARE: Hilbert transform needs narrowband signal to produce meaningful results
        '''
        print("Calculating Phase Synchronization, please wait...")
        analytic_signal = hilbert(self.time_series.transpose())
        instantaneous_phase = np.angle(analytic_signal)
        
        ips = np.full((self.P,self.P, self.N_estimates), np.nan)
        for i in range(self.P):
            for j in range(self.P):
                if self.method == "crp":
                    ips[i, j, :] = np.cos(instantaneous_phase[i] - instantaneous_phase[j]) # cosine of the relative phase
                if self.method == "phcoh":
                    ips[i, j, :] = 1 - np.abs(np.sin(instantaneous_phase[i] - instantaneous_phase[j])) # phase coherence
                if self.method == "teneto":
                    ips[i, j, :] = 1 - np.sin(np.abs(instantaneous_phase[i] - instantaneous_phase[j])/2) # teneto implementation
                

        self.R_mat = self.postproc(ips)
        return self.R_mat

class LeiDA(ConnectivityMethod):
    name = "CONT Leading Eigenvector Dynamics"
    options = {}
    '''
    Leading Eigenvector Dynamics:
        Cabral, J., Vidaurre, D., Marques, P., Magalhães, R., Silva Moreira, P., Miguel Soares, J., ... & Kringelbach, M. L. (2017). 
        Cognitive performance in healthy older adults relates to spontaneous switching between states of functional connectivity during rest. 
        Scientific reports, 7(1), 5135. https://doi.org/10.1038/s41598-017-05425-7
        
        Olsen, A. S., Lykkebo-Valløe, A., Ozenne, B., Madsen, M. K., Stenbæk, D. S., Armand, S., ... & Fisher, P. M. (2022). Psilocybin modulation 
        of time-varying functional connectivity is associated with plasma psilocin and subjective effects. Neuroimage, 264, 119716.
        https://doi.org/10.1016/j.neuroimage.2022.119716

        Vohryzek, J., Deco, G., Cessac, B., Kringelbach, M. L., & Cabral, J. (2020). Ghost attractors in spontaneous brain activity: 
        Recurrent excursions into functionally-relevant BOLD phase-locking states. Frontiers in systems neuroscience, 14, 20.
        https://doi.org/10.3389/fnsys.2020.00020
    '''
    def __init__(self,
                 time_series: np.ndarray,
                 flip_eigenvectors: bool = False,
                 diagonal: int = 0,
                 standardize: bool = False,
                 fisher_z: bool = False,
                 tril: bool = False):
        
        super().__init__(time_series, diagonal, standardize, fisher_z, tril)

        self.N_estimates = self.T
        self.R_mat = np.full((self.P,self.P, self.N_estimates), np.nan)
        self.flip_eigenvectors = flip_eigenvectors
        self.res = []

    def connectivity(self):
        print("Calculating LeiDA, please wait...")
        # Compute BOLD phases using Hilbert Transform
        instantaneous_phase = np.angle(hilbert(self.time_series.transpose())).transpose()

        # Compute the leading eigenvector for each time point
        for n in range(self.N_estimates):
            # Compute coherence matrix
            cohmat = np.cos(np.subtract.outer(instantaneous_phase[n,:], instantaneous_phase[n,:]))

            # Compute the eigenvectors and eigenvalues
            eigenvalues, eigenvectors = eigh(cohmat)

            # The leading eigenvector is the one corresponding to the largest eigenvalue
            V1 = eigenvectors[:, -1]

            if self.flip_eigenvectors:
                # Make sure the largest component is negative
                if np.mean(V1 > 0) > 0.5:
                    V1 = -V1

            self.R_mat[:, :, n] = np.outer(V1, V1)

        self.R_mat = self.postproc(self.R_mat)

        return self.R_mat, V1

class WaveletCoherence(ConnectivityMethod):
    name = "CONT Wavelet Coherence"
    options = {"method": ["weighted"]}
    '''
    Instantaneous Wavelet Coherence:
        Jacob Billings, Manish Saggar, Jaroslav Hlinka, Shella Keilholz, Giovanni Petri; Simplicial and 
        topological descriptions of human brain dynamics. Network Neuroscience 2021; 5 (2): 549–568. 
        https://doi.org/10.1162/netn_a_00190
    '''
    def __init__(self,
                 time_series: np.ndarray,
                 method: Literal["weighted"] = "weighted",
                 TR: float = 0.72,
                 fmin: float = 0.007,
                 fmax: float = 0.15,
                 n_scales: int = 15,
                 drop_scales: int = 2,
                 drop_timepoints: int = 50,
                 diagonal: int = 0,
                 standardize: bool = False,
                 fisher_z: bool = False,
                 tril: bool = False):
        
        super().__init__(time_series, diagonal, standardize, fisher_z, tril)

        self.N_estimates = self.T
        self.R_mat = np.full((self.P,self.P, self.N_estimates), np.nan)
        self.method = method
        self.TR = TR
        self.fmin = fmin
        self.fmax = fmax
        self.n_scales = n_scales
        self.drop_scales = drop_scales
        self.drop_timepoints = drop_timepoints

    def connectivity(self):
        print("Calculating Wavelet Coherence, please wait...")
        # Time series dimensions
        P = self.time_series.shape[1]
        T = self.time_series.shape[0]
        
        # Initial parameters
        dt = self.TR # TR of the data
        s0 = 1 / self.fmax # Smallest scale of the wavelet
        J = self.n_scales - 1 # Scales range from s0 up to s0 * 2**(J * dj), which gives a total of (J + 1) scales
        dj = np.log2(self.fmax / self.fmin) / J  # Spacing between discrete scales to achieve the desired frequency range
        mother_wavelet = Morlet(6) # Morlet wavelet with omega_0 = 6

        iwc = np.full((P, P, J+1, T), np.nan) # resulting wavelet coherence matrices
        dfc = np.full((P, P, T), np.nan) # resulting dfc matrices (weighted average of iwc across scales)
        
        W_list = []
        S_list = []
        for i in range(P):
            # Calculate continuous wavelet transform
            y_normal = (self.time_series[:,i] - self.time_series[:,i].mean()) / self.time_series[:,i].std()
            W, s, freq, coi, fft, fftfregs = cwt(y_normal, dt, dj=dj, s0=s0, J=J, wavelet=mother_wavelet)
            scales = np.ones([1, y_normal.size]) * s[:, None]
            S = mother_wavelet.smooth(np.abs(W) ** 2 / scales, dt, dj, s) # Smooth the wavelet spectra

            W_list.append(W)
            S_list.append(S)

        y_normal = (self.time_series[:,0] - self.time_series[:,0].mean()) / self.time_series[:,i].std()
        _, s1, _, _, _, _ = cwt(y_normal, dt, dj=dj, s0=s0, J=J, wavelet=mother_wavelet)
        scales = np.ones([1, y_normal.size]) * s1[:, None]

        for i in tqdm(range(P)):
            for j in range(i, P):
                W1 = W_list[i]
                W2 = W_list[j]
                S1 = S_list[i]
                S2 = S_list[j]
                
                # Calculate wavelet coherence
                W12 = W1 * np.conj(W2)
                S12 = mother_wavelet.smooth(W12 / scales, dt, dj, s1)
                WCT = np.abs(S12) ** 2 / (S1 * S2)
                iwc[i,j,:,:] = WCT
                iwc[j,i,:,:] = WCT

                if self.method == "weighted":
                # Calculate DFC as weighted average using cross wavelet power 
                    CWP = np.abs(W1 * np.conj(W2)) # cross wavelet power
                    CWP = CWP[self.drop_scales:-self.drop_scales, self.drop_timepoints:-self.drop_timepoints] # drop outer scales and outer time points
                    WCT = WCT[self.drop_scales:-self.drop_scales, self.drop_timepoints:-self.drop_timepoints] # drop outer scales and outer time points
                    cross_power = CWP / np.sum(CWP, axis=0) # normalize
                    
                    dfc[i, j, self.drop_timepoints:-self.drop_timepoints] = 1 - (cross_power * WCT).sum(axis=0) # dfc as in eq. 1
                    dfc[j, i, self.drop_timepoints:-self.drop_timepoints] = 1 - (cross_power * WCT).sum(axis=0) # dfc as in eq. 1
                else:
                    raise NotImplementedError("Other methods not yet implemented")

        # Get rid of empty time points
        dfc = dfc[:,:, self.drop_timepoints:-self.drop_timepoints]
        
        dfc = self.postproc(dfc)
        
        #return iwc, dfc
        return dfc

class DCC(ConnectivityMethod):
    name = "CONT Dynamic Conditional Correlation"
    options = {}
    '''
    Dynamic Conditional Correlation:
        Lindquist, M. A., Xu, Y., Nebel, M. B., & Caffo, B. S. (2014). Evaluating dynamic bivariate correlations 
        in resting-state fMRI: a comparison study and a new approach. NeuroImage, 101, 531-546.
        https://doi.org/10.1016/j.neuroimage.2014.06.052
    '''

    def __init__(self,
                 time_series: np.ndarray,
                 num_cores: int = 16,
                 standardizeData: bool = True,
                 diagonal: int = 0,
                 standardize: bool = False,
                 fisher_z: bool = False,
                 tril: bool = False):
        
        super().__init__(time_series, diagonal, standardize, fisher_z, tril)

        self.N_estimates = self.T
        self.R_mat = np.full((self.P,self.P, self.N_estimates), np.nan)
        self.standardizeData = standardizeData
        self.num_cores = num_cores

    def _loglikelihood_garch11(self, theta, data):
        """Calculates a real number proportional to the negative log-likelihood of the GARCH(1,1) model

        Parameters
        ----------
        theta : 1-by-3 vector
            GARCH(1,1) parameter vector
        data: 1-by-n vector
            one dimensional time series

        Returns
        -------
        output : float
            real number proportional to the negative log-likelihood (to be minimized)
        """
        T = len(data)
        h = np.zeros_like(data)
        h[0] = np.mean(data ** 2)
        
        output = 0
        eps = 1e-15
        for t in range(1,T):
            # h[t-1] neds to be > 0 and extremely small values can cause an overflow
            if h[t-1] > eps:
                output += data[t-1] ** 2 / h[t-1] + np.log(h[t-1])

            h[t] = theta[0] + theta[1] * data[t-1] ** 2 + theta[2] * h[t-1]

        return output

    def _rToEpsilon(self, r, theta):
        """Calculates the standardized residual vector and standardized conditional volatility vector (eq. 25)

        Parameters
        ----------
        r: 1-by-n vector
            one dimensional time series
        theta : 1-by-3 vector
            fitted GARCH(1,1) parameters

        Returns
        -------
        epsilon : n-by-1 vector
            standardized residual vector
        d : 1-by-n vector
            estimated conditional volatility vector
        """
        T = len(r)
        epsilon = np.zeros_like(r)
        d = np.zeros_like(r)
        d[0] = np.mean(r ** 2)

        for t in range(T-1):
            epsilon[t] = r[t] / np.sqrt(d[t])
            d[t+1] = theta[0] + theta[1] * r[t] ** 2 + theta[2] * d[t]

        epsilon[T-1] = r[T-1] / np.sqrt(d[T-1])
        
        return epsilon, d

    def _epsilonToR(self, epsilon, theta):
        """Calculates the time-varying conditional correlation matrices from standardized returns

        Parameters
        ----------
        epsilon : T-by-N matrix
            standardized residual time series
        theta : 1-by-2 vector
            dcc parameter vector

        Returns
        -------
        R : N-by-N-by-T matrix
            conditional correlation matrix
        """
        T, N = epsilon.shape
        S2 = np.corrcoef(epsilon.T)
        SS = S2 * (1 - np.sum(theta))
        Q = S2

        R = np.zeros((N, N, T))
        for t in range(T):
            temp = epsilon[t, :] * np.sqrt(np.diag(Q))
            Q = SS + theta[0] * np.outer(temp, temp) + theta[1] * Q
            R[:, :, t] = np.diag(1. / np.sqrt(np.diag(Q))) @ Q @ np.diag(1. / np.sqrt(np.diag(Q)))

        return R

    def _LcOriginal(self, epsilon, theta):
        """Calculates the correlation component of the log-likelihood (eq. 29)

        Parameters
        ----------
        epsilon : T-by-N matrix
            standardized residual time series
        theta : 1-by-2 vector
            dcc parameter vector

        Returns
        -------
        output : float
            real number proportional to the correlation component in negative log-likelihood (to be minimized)
        """
        T, N = epsilon.shape
        S2 = np.cov(epsilon, rowvar=False)
        SS = S2 * (1 - sum(theta))
        Q = S2.copy()
        R = np.diag(1/np.sqrt(np.diag(Q))) @ Q @ np.diag(1/np.sqrt(np.diag(Q)))

        output = 0
        for t in range(T):
            output += np.log(np.linalg.det(R)) + epsilon[t, :] @ np.linalg.inv(R) @ epsilon[t, :]
            temp = epsilon[t, :] * np.sqrt(np.diag(Q))
            Q = SS + theta[0] * np.outer(temp, temp) + theta[1] * Q
            R = np.diag(1/np.sqrt(np.diag(Q))) @ Q @ np.diag(1/np.sqrt(np.diag(Q)))

        return output
    
    def _compute_garch(self, n):
        ts_n = np.ascontiguousarray(self.time_series[:, n])
        constraints = {'type': 'ineq', 'fun': lambda x: np.array([1 - x[0] - x[1] - x[2], x[0], x[1], x[2]])} 
        bounds = ((0, 1), (0, 1), (0, 1))
        theta0 = (0.25, 0.25, 0.25)
        res = minimize(self._loglikelihood_garch11, theta0, args=(ts_n), constraints=constraints, bounds=bounds)
        
        ep, d = self._rToEpsilon(ts_n, res.x)
        return res.x, ep, d
    
    def connectivity(self):
        """DCC algorithm
        
        Parameters
        ----------
        theta : T-by-N matrix
            fMRI time series data

        Returns
        -------
        H : N*N*T matrix 
            estimated dynamic conditional covariance matrices
        R : N*N*T matrix 
            estimated dynamic conditional correlation matrices
        Theta : N*3 matrix
            GARCH(1,1) parameters
        X : 1*N vector
            DCC parameters
        """
        print("Calculating Dynamic Conditional Correlation, please wait...")
        T, N = self.time_series.shape # T timepoints x N parcels
        ts = self.time_series - np.mean(self.time_series, axis=0) # Demean

        # Initialize output data
        H = np.zeros((N,N,T)) # conditional covariance matrices
        R = np.zeros((N,N,T)) # conditional correlation matrices
        Theta = np.zeros((N,3)) # GARCH(1,1) parameters

        # Initialize intermediate parameters
        epsilon = np.zeros_like(ts)
        D = np.zeros_like(ts)

        # Fit a univariate GARCH process for each n
        results = Parallel(n_jobs=self.num_cores)(delayed(self._compute_garch)(n) for n in tqdm(range(N)))

        for n, (theta, ep, d) in enumerate(results):
            Theta[n, :] = theta
            epsilon[:, n] = ep
            D[:, n] = d
            
        # Estimate parameter X for dynamic correlation matrix
        constraints = {'type': 'ineq', 'fun': lambda x: np.array([1 - x[0] - x[1], x[0], x[1]])}  
        bounds = ((0, 1), (0, 1))
        x0 = (0.25, 0.25)

        for attempt in range(5):
            try: 
                res = minimize(lambda x: self._LcOriginal(epsilon, x), x0, constraints=constraints, bounds=bounds)
                break
            except:
                print(f"Attempt {attempt + 1} failed, trying new random initial values")
                x0 = (random.uniform(0.25, 0.5), random.uniform(0.25, 0.5))
                if attempt == 4:
                    print("All attempts failed.")
        X = res.x

        # Time-varying conditional correlation matrices
        R = self._epsilonToR(epsilon, X)
        
        # Time-varying conditional covariance matrices
        for t in tqdm(range(T)):
            H[:,:,t] = np.diag(np.sqrt(D[t,:])) @ R[:,:,t] @ np.diag(np.sqrt(D[t,:]))


        R = self.postproc(R)
        
        #return H, R, Theta, X
        return R

class Edge_centric_connectivity(ConnectivityMethod):
    name = "CONT Edge-centric Connectivity"
    options = {}

    def __init__(self,
                 time_series: np.ndarray,
                 standardizeData: bool = True,
                 vlim: float = 3, # for plotting in the GUI, not used in the method itself
                 diagonal: int = 0,
                 standardize: bool = False,
                 fisher_z: bool = False,
                 tril: bool = False):
        
        super().__init__(time_series, diagonal, standardize, fisher_z, tril)
        self.standardizeData = standardizeData
    
    def connectivity(self):
        
        z = zscore(self.time_series, axis=0, ddof=1) if self.standardizeData else self.time_series
        u, v = np.triu_indices(self.time_series.shape[1], k=1)
        a = np.multiply(z[:, u], z[:, v]) # edge time series
 
        b = a.T @ a # inner product
        c = np.sqrt(np.diag(b)) # Square root of the diagonal elements (variance) to normalize
        d = np.outer(c, c) ## Create the normalization matrix
        dfc = b / d # Element-wise division to get the correlation matrix
        
        dfc = self.postproc(dfc)

        return dfc, (a, u, v)
    
'''
State based dFC methods. Basically wrapper functions to bring methods from https://github.com/neurodatascience/dFC/ into the Comet framework
'''
class Sliding_Window(BaseDFCMethod):
    name = "CONT Sliding Window (pydfc)"
    options = {"clstr_distance": ["euclidean"]}
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
    name = "CONT Time-frequency (pydfc)"
    options = {}

    '''
    Time-Frequency
    '''
    def __init__(self, time_series, num_cores=8, coi_correction=True, **params):
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
        self.params['n_jobs'] = num_cores

    def connectivity(self):
        measure = TIME_FREQ(**self.params)
        dFC = measure.estimate_dFC(time_series=self.time_series)
        return dFC
    
class Cap(BaseDFCMethod):
    name = "STATE Co-activation patterns"
    options = {}

    '''
    Co-activation patterns
    '''
    def __init__(self, time_series, subject=0, n_states=5, n_subj_clusters=5, normalization=True, **params):
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
        self.params['subject'] = subject
        self.params['n_subj_clstrs'] = n_subj_clusters
        self.params['n_states'] = n_states
        self.params['normalization'] = normalization

        self.params['subjects'] = list(self.time_series.data_dict.keys())
        self.sub_id = self.params['subjects'][self.params['subject']]

    def connectivity(self):
        measure = CAP(**self.params)
        measure.estimate_FCS(time_series=self.time_series)
        dFC = measure.estimate_dFC(time_series=self.time_series.get_subj_ts(subjs_id=self.sub_id))
        return dFC
    
class Sliding_Window_Clustr(BaseDFCMethod):
    name = "STATE Sliding Window Clustering"
    options = {"clstr_distance": ["euclidean", "manhattan"], }

    '''
    Sliding Window Clustering
    '''
    def __init__(self, time_series, subject=0, windowsize=44, n_overlap=0.5, tapered_window=True, n_states=5, n_subj_clusters=5, normalization=True, clstr_distance="euclidean"):
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
                self.params[params_name] = None
        
        self.params['measure_name'] = 'Clustering'
        self.params['is_state_based'] = True
        self.params['clstr_base_measure'] = 'SlidingWindow'
        self.params["sw_method"] = 'pear_corr'
        self.params['subject'] = subject
        self.params["tapered_window"] = tapered_window
        self.params['clstr_distance'] = clstr_distance
        self.params['n_subj_clstrs'] = n_subj_clusters
        self.params['W'] = windowsize
        self.params['n_overlap'] = n_overlap
        self.params['n_states'] = n_states
        self.params['normalization'] = normalization

        self.params['subjects'] = list(self.time_series.data_dict.keys())
        self.sub_id = self.params['subjects'][self.params['subject']]

        assert self.params['clstr_base_measure'] in self.base_methods_name_lst, \
            "Base method not recognized."

    def connectivity(self):
        measure = SLIDING_WINDOW_CLUSTR(**self.params)
        measure.estimate_FCS(time_series=self.time_series)
        dFC = measure.estimate_dFC(time_series=self.time_series.get_subj_ts(subjs_id=self.sub_id))
        return dFC
    
class Hmm_Cont(BaseDFCMethod):
    name = "STATE Continuous Hidden Markov Model"
    options = {}

    '''
    Continuous Hidden Markov Model
    '''
    def __init__(self, time_series, subject=0, n_states=5, iterations=20, normalization=True, **params):
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
        self.params['subject'] = subject
        self.params['n_states'] = n_states
        self.params['hmm_iter'] = iterations
        self.params['normalization'] = normalization

        self.params['subjects'] = list(self.time_series.data_dict.keys())
        self.sub_id = self.params['subjects'][self.params['subject']]

    def connectivity(self):
        measure = HMM_CONT(**self.params)
        measure.estimate_FCS(time_series=self.time_series)
        dFC = measure.estimate_dFC(time_series=self.time_series.get_subj_ts(subjs_id=self.sub_id))
        return dFC
    
class Hmm_Disc(BaseDFCMethod):
    name = "STATE Discrete Hidden Markov Model"
    options = {"clstr_base_measure": ["SlidingWindow"], "sw_method": ["pear_corr"]}

    '''
    Discrete Hidden Markov Model
    '''
    def __init__(self, time_series, subject=0, windowsize=44, n_overlap=0.5, clstr_base_measure="SlidingWindow", sw_method="pear_corr", tapered_window=True, iterations= 20, dhmm_obs_state_ratio=16/24, n_states=5, n_subj_clusters=5, normalization=True, **params):
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
        self.params['subject'] = subject
        self.params['W'] = windowsize
        self.params['n_overlap'] = n_overlap
        self.params['clstr_base_measure'] = clstr_base_measure
        self.params['sw_method'] = sw_method
        self.params['tapered_window'] = tapered_window
        self.params['hmm_iter'] = iterations
        self.params['dhmm_obs_state_ratio'] = dhmm_obs_state_ratio
        self.params['n_subj_clstrs'] = n_subj_clusters
        self.params['n_states'] = n_states
        self.params['normalization'] = normalization

        self.params['subjects'] = list(self.time_series.data_dict.keys())
        self.sub_id = self.params['subjects'][self.params['subject']]

        assert self.params['clstr_base_measure'] in self.base_methods_name_lst, \
            "Base measure not recognized."

    def connectivity(self):
        measure = HMM_DISC(**self.params)
        measure.estimate_FCS(time_series=self.time_series)
        dFC = measure.estimate_dFC(time_series=self.time_series.get_subj_ts(subjs_id=self.sub_id))
        return dFC
    
class Windowless(BaseDFCMethod):
    name = "STATE Windowless"
    options = {}

    '''
    Windowless
    '''
    def __init__(self, time_series, subject=0, n_states=5, n_subj_clusters=5, normalization=True, **params):
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
        self.params['subject'] = subject
        self.params['n_subj_clstrs'] = n_subj_clusters
        self.params['n_states'] = n_states
        self.params['normalization'] = normalization

        self.params['subjects'] = list(self.time_series.data_dict.keys())
        self.sub_id = self.params['subjects'][self.params['subject']]

    def connectivity(self):
        measure = WINDOWLESS(**self.params)
        measure.estimate_FCS(time_series=self.time_series)
        dFC = measure.estimate_dFC(time_series=self.time_series.get_subj_ts(subjs_id=self.sub_id))
        return dFC

'''
Static FC methods
'''
class Static_Pearson(ConnectivityMethod):
    name = "STATIC Pearson Correlation"
    options = {}

    def __init__(self,
                 time_series: np.ndarray,
                 diagonal: int = 0,
                 standardize: bool = False,
                 fisher_z: bool = False,
                 tril: bool = False):
        
        super().__init__(time_series, diagonal, standardize, fisher_z, tril)

    def connectivity(self):
        fc = np.corrcoef(self.time_series.T) 
        fc = self.postproc(fc)
        return fc  

class Static_Partial(ConnectivityMethod):
    name = "STATIC Partial Correlation"
    options = {}

    def __init__(self,
                 time_series: np.ndarray,
                 diagonal: int = 0,
                 standardize: bool = False,
                 fisher_z: bool = False,
                 tril: bool = False):
        
        super().__init__(time_series, diagonal, standardize, fisher_z, tril)

    def connectivity(self):
        corr = np.corrcoef(self.time_series.T)
        precision = inv(corr)
        fc = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
        fc = self.postproc(fc)
        return fc
    
class Static_Mutual_Info(ConnectivityMethod):
    name = "STATIC Mutual Information"
    options = {}

    def __init__(self, 
                 time_series: np.ndarray,
                 num_bins: int = 10,
                 diagonal: int = 0,
                 standardize: bool = False,
                 fisher_z: bool = False,
                 tril: bool = False):
        
        super().__init__(time_series, diagonal, standardize, fisher_z, tril)
        self.num_bins = num_bins

    def connectivity(self):
        assert self.num_bins is not None, "Number of bins must be specified for mutual information method"

        binned_data = np.zeros_like(self.time_series, dtype=int)

        # Determine the bin edges and bin the data for each time series
        for i in range(self.P):
            bin_edges = np.histogram_bin_edges(self.time_series[:, i], bins=self.num_bins)
            binned_data[:, i] = np.digitize(self.time_series[:, i], bins=bin_edges, right=False)

        fc = np.zeros((self.P, self.P))

        for i in range(self.P):
            for j in range(i + 1, self.P):
                mi = mutual_info_score(binned_data[:, i], binned_data[:, j])
                fc[i, j] = mi
                fc[j, i] = mi

        fc = self.postproc(fc)
        return fc
