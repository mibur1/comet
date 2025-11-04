import random
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from typing import Literal, Union
from abc import ABCMeta, abstractmethod
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from statsmodels.stats.weightstats import DescrStatsW
from pycwt import cwt, Morlet
from hmmlearn import hmm
from ksvd import ApproximateKSVD
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from sklearn.covariance import LedoitWolf
from scipy.stats import zscore
from scipy.spatial import distance
from scipy.signal import windows, hilbert
from scipy.linalg import eigh, solve, det, inv
from scipy.optimize import minimize

"""
SECTION: Class template for all dynamic functional connectivity methods.
         New methods should be implemented as child classes.
"""
class ConnectivityMethod(metaclass=ABCMeta):
    """
    Base class for all dynamic functional connectivity methods.

    Attributes
    ----------
    time_series : np.ndarray
        Time series data.
    T : int
        Number of timepoints.
    P : int
        Number of parcels.
    diagonal : int or float
        Value to set on the diagonal of connectivity matrices.
    fisher_z : bool
        Whether to apply Fisher z-transformation.
    tril : bool
        Whether to return only the lower triangle of the matrices.
    """
    def __init__(self, time_series, diagonal=0, fisher_z=False, tril=False):
        """
        Initializes the ConnectivityMethod with the given parameters.

        Parameters
        ----------
        time_series : np.ndarray
            The input time series data.
        diagonal : int or float, optional
            Value to set on the diagonal of connectivity matrices. Default is 0.
        fisher_z : bool, optional
            Whether to apply Fisher z-transformation. Default is False.
        tril : bool, optional
            Whether to return only the lower triangle of the matrices. Default is False.
        """
        self.time_series = time_series
        self.diagonal = diagonal
        self.fisher_z = fisher_z
        self.tril = tril

        # Shape checks and conversion to array
        if isinstance(self.time_series, list) or (isinstance(self.time_series, np.ndarray) and self.time_series.dtype == object):
            if len(self.time_series) == 0:
                raise ValueError("time_series is empty.")
            seq = [np.asarray(x, dtype="float32") for x in self.time_series]
            shapes = {a.shape for a in seq}
            if not all(a.ndim == 2 for a in seq):
                raise ValueError("All subject time series must be 2D arrays (T, P).")
            if len(shapes) != 1:
                raise ValueError(f"All subject time series must have the same shape (T, P); got {shapes}.")
            self.time_series = np.stack(seq, axis=0)  # (S, T, P)

        # Case B: proper ndarray → just ensure float32
        elif isinstance(self.time_series, np.ndarray):
            self.time_series = self.time_series.astype("float32", copy=False)

        else:
            raise ValueError("time_series must be a 2D ndarray, 3D ndarray, or a list of 2D ndarrays.")

        # Prepare the data and create some variables
        self.time_series = self.time_series.astype("float32")
        if np.ndim(self.time_series) == 3 :
            self.n_subjects = self.time_series.shape[0]
            self.T = self.time_series.shape[1] # Timepoints
            self.P = self.time_series.shape[2] # Parcels (brain regions)
            self.time_series3D = self.time_series
            # Reshape the data to 2D by stacking all subjects (n_subjects * T, P)
            self.time_series = self.time_series.reshape(-1, self.time_series.shape[-1])

        elif np.ndim(self.time_series) == 2:
            self.n_subjects = 1
            self.T = self.time_series.shape[0]
            self.P = self.time_series.shape[1]
            self.time_series3D = self.time_series.reshape(1, self.T, self.P)
        else:
            raise ValueError("Input data must be either a 2D array, 3D array, or a list of 2D arrays.")

        return

    # All subclasses must implement an estimate() method and should use postproc() for simple post-processing
    @abstractmethod
    def estimate(self):
        """
        Abstract method to compute the connectivity matrix.
        This method should be implemented in each child class.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the child class.
        """
        raise NotImplementedError("This method should be implemented in each child class.")

    def postproc(self):
        """
        Post-process the connectivity matrix with optional Fisher z-transformation,
        z-standardization, diagonal setting, and lower triangle extraction.

        Returns
        -------
        np.ndarray
            The post-processed connectivity matrix.
        """
        if hasattr(self, "dfc"):
            dfc = self.dfc
        elif hasattr(self, "fc"):
            dfc = self.fc
        else:
            raise AttributeError("No connectivity estimates available.")

        # Fisher z-transformation
        if self.fisher_z:
            dfc = np.clip(dfc, -1 + np.finfo(float).eps, 1 - np.finfo(float).eps)
            dfc = np.arctanh(dfc)

        # Set main diagonal
        if len(dfc.shape) == 3:
            for estimate in range(dfc.shape[2]):
                np.fill_diagonal(dfc[:,:, estimate], self.diagonal)
        else:
            np.fill_diagonal(dfc, self.diagonal)

        # Get lower triangle to save space
        if self.tril:
            if len(dfc.shape) == 3:
                mask = np.tril(dfc[:, :, 0], k=-1) != 0
                dfc = np.array([matrix_2d[mask] for matrix_2d in dfc.transpose(2, 0, 1)])
            else:
                mask = np.tril(dfc, k=-1) != 0
                dfc = dfc[mask]

        return dfc
    

"""
SECTION: Continuous dFC methods
"""
class SlidingWindow(ConnectivityMethod):
    """
    Sliding Window connectivity method.

    This is the most widely used dynamic functional connectivity method. It involves sliding
    a window over the data. Covariance is estimated for each windowed section.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series data.
    windowsize : int, optional
        Size of the sliding window. Default is 29.
    stepsize : int, optional
        Step size for sliding the window. Default is 1.
    shape : {'rectangular', 'gaussian', 'hamming'}, optional
        Shape of the window. Default is 'rectangular'.
    std : float, optional
        Standard deviation for the Gaussian window. Default is 10.
    diagonal : int, optional
        Value to set on the diagonal of connectivity matrices. Default is 0.
    fisher_z : bool, optional
        Whether to apply Fisher z-transformation. Default is False.
    tril : bool, optional
        Whether to return only the lower triangle of the matrices. Default is False.
    """
    name = "CONT Sliding Window"

    def __init__(self,
                 time_series: np.ndarray,
                 windowsize: int = 29,
                 stepsize: int = 1,
                 shape: Literal["rectangular", "gaussian", "hamming"] = "rectangular",
                 std: float = 10,
                 diagonal: int = 0,
                 fisher_z: bool = False,
                 tril: bool = False):

        super().__init__(time_series, diagonal, fisher_z, tril)
        self.windowsize = windowsize
        self.stepsize = stepsize
        self.shape = shape
        self.std = std

        self.N_estimates = (self.T - self.windowsize) // self.stepsize + 1 # N possible estimates given the window and step size
        self.dfc = np.full((self.P,self.P, self.N_estimates), np.nan)

        if not self.windowsize <= self.T:
            raise ValueError("windowsize is larger than time series")

    def _weights(self):
        """
        Create weights for sliding window.

        The window shape can be:
            - rectangular
            - gaussian
            - hamming

        Returns
        -------
        np.ndarray
            Weights for each windowed section.
        """
        weights_init = np.zeros(self.T)

        if self.shape == 'rectangular':
            weights_init[0:self.windowsize] = np.ones(self.windowsize)
        if self.shape == 'gaussian':
            weights_init[0:self.windowsize] = windows.gaussian(self.windowsize, self.std)
        if self.shape == 'hamming':
            weights_init[0:self.windowsize] = windows.hamming(self.windowsize)

        weights = np.array([np.roll(weights_init, i) for i in range(0, self.T + 1 - self.windowsize, self.stepsize)])

        return weights

    def centers(self):
        """
        Calculate the central index of each window so dynamic functional connectivity (dFC)
        estimates can be related to the original time series.

        Returns
        -------
        np.ndarray
            Central index of each window.
        """
        centers = np.arange(self.windowsize // 2, self.T - self.windowsize // 2, self.stepsize)
        return centers

    def estimate(self):
        """
        Calculate sliding window correlation.

        Returns
        -------
        np.ndarray
            Dynamic functional connectivity as a PxPxN array.
        """
        weights = self._weights()

        for estimate in range(self.N_estimates):
            self.dfc[:,:,estimate] = DescrStatsW(self.time_series, weights[estimate, :]).corrcoef

        self.dfc = self.postproc()
        return self.dfc

class Jackknife(ConnectivityMethod):
    """
    Jackknife correlation method.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series data.
    windowsize : int, optional
        Size of the sliding window. Default is 1.
    stepsize : int, optional
        Step size for sliding the window. Default is 1.
    diagonal : int, optional
        Value to set on the diagonal of connectivity matrices. Default is 0.
    fisher_z : bool, optional
        Whether to apply Fisher z-transformation. Default is False.
    tril : bool, optional
        Whether to return only the lower triangle of the matrices. Default is False.

    References
    ----------
    Richter CG, Thompson WH, Bosman CA, Fries P. A jackknife approach to quantifying single-trial
    correlation between covariance-based metrics undefined on a single-trial basis.
    https://doi.org/10.1016/j.neuroimage.2015.04.040
    """
    name = "CONT Jackknife Correlation"

    def __init__(self,
                 time_series: np.ndarray,
                 windowsize: int = 1,
                 stepsize: int = 1,
                 diagonal: int = 0,
                 fisher_z: bool = False,
                 tril: bool = False):

        super().__init__(time_series, diagonal, fisher_z, tril)
        self.windowsize = windowsize
        self.stepsize = stepsize

        self.N_estimates = (self.T - self.windowsize) // self.stepsize + 1 # N possible estimates given the window size
        self.dfc = np.full((self.P,self.P, self.N_estimates), np.nan)

        if not self.windowsize <= self.T:
            raise ValueError("windowsize is larger than time series")

    def _weights(self):
        """
        Create logical weight vectors for jackknife correlation.

        Example:
        01111 -> 10111 -> 11011 -> 11101 -> 11110

        Returns
        -------
        np.ndarray
            Boolean weights for each windowed section.
        """
        weights_init = np.ones(self.T)
        weights_init[0:self.windowsize] = 0
        weights = np.array([np.roll(weights_init, i) for i in range(0, self.T + 1 - self.windowsize, self.stepsize)])
        return weights.astype(bool)

    def centers(self):
        """
        Calculate the central index of each window so dynamic functional connectivity (dFC)
        estimates can be related to the original time series.

        Returns
        -------
        np.ndarray
            Central index of each window.
        """
        centers = np.arange(self.windowsize // 2, self.T - self.windowsize // 2, self.stepsize)
        return centers

    def estimate(self):
        """
        Calculate jackknife correlation.

        Returns
        -------
        np.ndarray
            Dynamic functional connectivity as a PxPxN array.
        """
        weights = self._weights()

        for estimate in range(self.N_estimates):
            ts_jackknife = self.time_series[weights[estimate,:],:]
            self.dfc[:,:,estimate] = np.corrcoef(ts_jackknife.T) * -1 # correlation estimation and sign inversion

        self.dfc = self.postproc()

        return self.dfc
    
class SpatialDistance(ConnectivityMethod):
    """
    Spatial Distance connectivity method.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series data.
    dist : {'euclidean', 'cosine', 'cityblock'}, optional
        Type of distance metric to use. Default is 'euclidean'.
    diagonal : int, optional
        Value to set on the diagonal of connectivity matrices. Default is 0.
    fisher_z : bool, optional
        Whether to apply Fisher z-transformation. Default is False.
    tril : bool, optional
        Whether to return only the lower triangle of the matrices. Default is False.

    References
    ----------
    William Hedley Thompson, Per Brantefors, Peter Fransson. From static
    to temporal network theory: Applications to functional brain connectivity.
    https://doi.org/10.1162/NETN_a_00011
    """
    name = "CONT Spatial Distance"

    def __init__(self,
                 time_series: np.ndarray,
                 dist: Literal["euclidean", "cosine", "cityblock"] = "euclidean",
                 diagonal: int = 0,
                 fisher_z: bool = False,
                 tril: bool = False):

        super().__init__(time_series, diagonal, fisher_z, tril)
        self.distance = self._distance_functions(dist)

        self.N_estimates = self.T
        self.dfc = np.full((self.P,self.P, self.N_estimates), np.nan)

    def _distance_functions(self, dist):
        """
        Get the distance function based on the specified metric.

        Parameters
        ----------
        dist : str
            The type of distance metric to use.

        Returns
        -------
        function
            The corresponding distance function.
        """
        options = {'euclidean': distance.euclidean,
                   'cosine'   : distance.cosine,
                   'cityblock': distance.cityblock}
        return options[dist]

    def _weights(self):
        """
        Calculate adjacency matrix for spatial distance.

        Returns
        -------
        np.ndarray
            The spatial distance adjacency matrix.
        """
        weights = np.array([self.distance(self.time_series[n, :], self.time_series[t, :]) for n in np.arange(0, self.T) for t in np.arange(0, self.T)])
        weights = np.reshape(weights, [self.T, self.T])
        np.fill_diagonal(weights, np.nan)
        weights = 1 / weights
        weights = (weights - np.nanmin(weights)) / (np.nanmax(weights) - np.nanmin(weights))
        np.fill_diagonal(weights, 1)
        return weights

    def estimate(self):
        """
        Calculate spatial distance correlation.

        Returns
        -------
        np.ndarray
            Dynamic functional connectivity as a PxPxN array.
        """
        weights = self._weights() # in this case this is the distance matrix

        for estimate in range(self.N_estimates):
            self.dfc[:,:,estimate] = DescrStatsW(self.time_series, weights[estimate, :]).corrcoef

        self.dfc = self.postproc()

        return self.dfc
    
class TemporalDerivatives(ConnectivityMethod):
    """
    Multiplication of Temporal Derivatives connectivity method.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series data.
    windowsize : int, optional
        Size of the sliding window. Default is 7.
    diagonal : int, optional
        Value to set on the diagonal of connectivity matrices. Default is 0.
    fisher_z : bool, optional
        Whether to apply Fisher z-transformation. Default is False.
    tril : bool, optional
        Whether to return only the lower triangle of the matrices. Default is False.

    References
    ----------
    Shine JM, Koyejo O, Bell PT, Gorgolewski KJ, Gilat M, Poldrack RA. Estimation of
    dynamic functional connectivity using Multiplication of Temporal Derivatives.
    https://doi.org/10.1016/j.neuroimage.2015.07.064.
    """
    name = "CONT Multiplication of Temporal Derivatives"

    def __init__(self,
                 time_series: np.ndarray,
                 windowsize: int = 7,
                 diagonal: int = 0,
                 fisher_z: bool = False,
                 tril: bool = False):

        super().__init__(time_series, diagonal, fisher_z, tril)
        self.windowsize = windowsize

        self.N_estimates = self.T - self.windowsize
        self.dfc = np.full((self.P,self.P, self.N_estimates), np.nan)

        if not self.windowsize <= self.T:
            raise ValueError("windowsize is larger than time series")

    def centers(self):
        """
        Calculate the central index of each window so dynamic functional connectivity (dFC)
        estimates can be related to the original time series.

        Returns
        -------
        np.ndarray
            Central index of each window.
        """
        centers = np.arange(self.windowsize // 2 + 1, self.T - self.windowsize // 2)
        return centers

    def estimate(self):
        """
        Calculate multiplication of temporal derivatives.

        Returns
        -------
        np.ndarray
            Dynamic functional connectivity as a PxPxN array.
        """
        derivatives = self.time_series[1:, :] - self.time_series[:-1, :]
        derivatives = derivatives / np.std(derivatives, axis=0)
        coupling = np.array([derivatives[:, i] * derivatives[:, j] for i in range(self.P) for j in range(self.P)]) # multiplicative coupling

        # Convolve with rectangular kernel for smoothing
        kernel = np.ones(self.windowsize) / self.windowsize
        smooth_coupling = np.full((self.P * self.P, self.N_estimates), np.nan)

        for i in range(self.P * self.P):
            smooth_coupling[i,:] = np.convolve(coupling[i,:], kernel, mode='valid')

        self.dfc = np.reshape(smooth_coupling, [self.P, self.P, self.N_estimates]) #reshape into PxPxN connectivity matrix
        self.dfc = self.postproc()

        return self.dfc

class FlexibleLeastSquares(ConnectivityMethod):
    """
    Flexible Least Squares connectivity method.

    This implementation estimates dynamic functional connectivity using the
    Flexible Least Squares (FLS) algorithm as described in the DynamicBC toolbox.

    For each source region i, the FLS linear system is built once and solved
    against all target time series simultaneously. Parallel execution across
    regions is supported via joblib.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series data of shape (T, P), where T is the number
        of time points and P the number of regions.
    standardizeData : bool, optional
        Whether to standardize the time series column-wise before estimation.
        Default is True.
    mu : float, optional
        Regularization parameter controlling the smoothness of betas.
        Default is 100.0.
    num_cores : int, optional
        Number of CPU cores to use for parallel processing.
        Set to 1 for single-core execution. Default is 4.
    progress_bar : bool, optional
        Whether to display a progress bar during estimation. Default is True.
    diagonal : int, optional
        Value to set on the diagonal of connectivity matrices. Default is 0.
    fisher_z : bool, optional
        Whether to apply Fisher z-transformation in post-processing.
        Default is False.
    tril : bool, optional
        Whether to return only the lower triangle of the matrices.
        Default is False.

    Attributes
    ----------
    dfc : np.ndarray
        The dynamic connectivity estimates of shape (P, P, T).
    N_estimates : int
        Number of estimates (equals the number of time points T).

    References
    ----------
    Liao, W., Wu, G. R., Xu, Q., Ji, G. J., Zhang, Z., Zang, Y. F., & Lu, G. (2014).
    DynamicBC: a MATLAB toolbox for dynamic brain connectome analysis.
    Brain Connectivity, 4(10), 780–790.
    https://doi.org/10.1089/brain.2014.0253
    """

    name = "CONT Flexible Least Squares"

    def __init__(self, time_series: np.ndarray, standardizeData: bool = True, mu: float = 100.0,
                 num_cores: int = 4, progress_bar: bool = True, diagonal: int = 0,
                 fisher_z: bool = False, tril: bool = False):
        super().__init__(time_series, diagonal, fisher_z, tril)
        self.standardizeData = bool(standardizeData)
        self.mu = float(mu)
        self.num_cores = int(num_cores)
        self.progress_bar = bool(progress_bar)
        self.N_estimates = self.T
        self.dfc = np.full((self.P, self.P, self.N_estimates), np.nan)

    def _betas(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute FLS betas for one predictor X against all targets Y.

        Parameters
        ----------
        X : np.ndarray
            Predictor column of shape (T, 1) for a fixed source region i.
        Y : np.ndarray
            Full time series of shape (T, P), all target regions.

        Returns
        -------
        np.ndarray
            Betas of shape (P, T) for the given source i to all targets.
        """
        N, K = X.shape
        G = np.zeros((N * K, N))
        A = np.zeros((N * K, N * K))
        mui = self.mu * np.eye(K)
        ind = np.arange(K)

        for t in range(N):
            G[ind, t] = X[t, :]
            if t == 0:
                Ai = X[t, :].T @ X[t, :] + mui
                A[ind, ind] = Ai
                if N > 1:
                    A[ind, ind + K] = -mui
            elif t != N - 1:
                Ai = X[t, :].T @ X[t, :] + 2 * mui
                A[ind, ind] = Ai
                A[ind, ind + K] = -mui
                A[ind, ind - K] = -mui
            else:
                Ai = X[t, :].T @ X[t, :] + mui
                A[ind, ind] = Ai
                A[ind, ind - K] = -mui
            ind += K

        RHS = G.T @ Y
        with threadpool_limits(limits=1):
            betas = solve(A, RHS, assume_a='sym')  # (T, P)
        return betas.T  # (P, T)

    @staticmethod
    def _compute_i(i: int, ts: np.ndarray, mu: float):
        """
        Worker function to compute betas for a single source region i.

        Parameters
        ----------
        i : int
            Source region index.
        ts : np.ndarray
            Standardized time series, shape (T, P).
        mu : float
            Regularization parameter.

        Returns
        -------
        tuple
            (i, beta_i) where beta_i has shape (P, T).
        """
        N = ts.shape[0]
        X = ts[:, i].reshape(N, 1)
        K = 1
        G = np.zeros((N * K, N))
        A = np.zeros((N * K, N * K))
        mui = mu * np.eye(K)
        ind = np.arange(K)

        for t in range(N):
            G[ind, t] = X[t, :]
            if t == 0:
                Ai = X[t, :].T @ X[t, :] + mui
                A[ind, ind] = Ai
                if N > 1:
                    A[ind, ind + K] = -mui
            elif t != N - 1:
                Ai = X[t, :].T @ X[t, :] + 2 * mui
                A[ind, ind] = Ai
                A[ind, ind + K] = -mui
                A[ind, ind - K] = -mui
            else:
                Ai = X[t, :].T @ X[t, :] + mui
                A[ind, ind] = Ai
                A[ind, ind - K] = -mui
            ind += K

        RHS = G.T @ ts
        with threadpool_limits(limits=1):
            betas = solve(A, RHS, assume_a='sym')
        return i, betas.T

    def estimate(self) -> np.ndarray:
        """
        Estimate dynamic functional connectivity using FLS.

        Returns
        -------
        np.ndarray
            Dynamic connectivity as a (P, P, T) array, symmetrized
            and post-processed according to class settings.
        """
        # Standardize if requested
        if self.standardizeData:
            mu = np.mean(self.time_series, axis=0, keepdims=True)
            sd = np.std(self.time_series, axis=0, keepdims=True)
            sd = np.where(sd == 0, 1.0, sd)
            self.time_series = (self.time_series - mu) / sd

        T, P = self.time_series.shape
        beta = np.zeros((P, P, T), dtype=float)

        if self.num_cores == 1:
            rng = tqdm(range(P), desc="FLS (single core)") if self.progress_bar else range(P)
            for i in rng:
                beta[i, :, :] = self._betas(self.time_series[:, i].reshape(T, 1), self.time_series)
        else:
            if self.progress_bar:
                with tqdm_joblib(total=P, desc=f"FLS ({self.num_cores} cores)"):
                    results = Parallel(n_jobs=self.num_cores, prefer="processes", batch_size=1)(
                        delayed(FlexibleLeastSquares._compute_i)(i, self.time_series, self.mu) for i in range(P)
                    )
            else:
                results = Parallel(n_jobs=self.num_cores, prefer="processes", batch_size=1)(
                    delayed(FlexibleLeastSquares._compute_i)(i, self.time_series, self.mu) for i in range(P)
                )
            
            for i, beta_i in results:
                beta[i, :, :] = beta_i

        self.dfc = 0.5 * (beta + beta.transpose(1, 0, 2))
        self.dfc = self.postproc()
        
        return self.dfc

class PhaseSynchronization(ConnectivityMethod):
    """
    Instantaneous Phase Synchronization methods.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series data.
    method : {'crp', 'phcoh', 'teneto'}, optional
        The phase synchrony method to use. Default is 'crp'.
    diagonal : int, optional
        Value to set on the diagonal of connectivity matrices. Default is 0.
    fisher_z : bool, optional
        Whether to apply Fisher z-transformation. Default is False.
    tril : bool, optional
        Whether to return only the lower triangle of the matrices. Default is False.

    References
    ----------
    Honari, H., Choe, A. S., & Lindquist, M. A. (2021). Evaluating phase synchronization methods in fMRI:
    A comparison study and new approaches. NeuroImage, 228, 117704. https://doi.org/10.1016/j.neuroimage.2020.117704
    """
    name = "CONT Phase Synchronization"

    def __init__(self,
                 time_series: np.ndarray,
                 method: Literal["crp", "phcoh", "teneto"] = "crp",
                 diagonal: int = 0,
                 fisher_z: bool = False,
                 tril: bool = False):

        super().__init__(time_series, diagonal, fisher_z, tril)
        self.N_estimates = self.T
        self.dfc = np.full((self.P,self.P, self.N_estimates), np.nan)
        self.method = method

    def estimate(self):
        analytic = hilbert(self.time_series, axis=0)
        phi = np.angle(np.asarray(analytic))                 # (N, P)
        N, P = phi.shape

        # All pairwise phase diffs for all times in one shot: (N, P, P)
        dphi = phi[:, :, None] - phi[:, None, :]

        if self.method == "crp":
            M = np.cos(dphi)                                  # (N, P, P)
        elif self.method == "phcoh":
            M = 1 - np.abs(np.sin(dphi))
        elif self.method == "teneto":
            M = 1 - np.sin(np.abs(dphi) / 2)
        else:
            raise ValueError(f"Unknown estimation method: {self.method}")

        ips = np.moveaxis(M, 0, 2)                            # -> (P, P, N)

        # Enforce symmetry (in case of tiny asymmetries)
        self.dfc = 0.5 * (ips + ips.transpose(1, 0, 2))

        self.dfc = self.postproc()
        return self.dfc

class LeiDA(ConnectivityMethod):
    """
    Leading Eigenvector Dynamics.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series data.
    flip_eigenvectors : bool, optional
        The sign of each leading eigenvector is adjusted so that the majority of its elements are negative. 
        This ensures consistent orientation of eigenvectors across time points and subjects. Default is True.
    diagonal : int, optional
        Value to set on the diagonal of connectivity matrices. Default is 0.
    fisher_z : bool, optional
        Whether to apply Fisher z-transformation. Default is False.
    tril : bool, optional
        Whether to return only the lower triangle of the matrices. Default is False.

    References
    ----------
    Cabral, J., Vidaurre, D., Marques, P., Magalhães, R., Silva Moreira, P., Miguel Soares, J., ... & Kringelbach, M. L. (2017).
    Cognitive performance in healthy older adults relates to spontaneous switching between states of functional connectivity during rest.
    Scientific reports, 7(1), 5135. https://doi.org/10.1038/s41598-017-05425-7

    Olsen, A. S., Lykkebo-Valløe, A., Ozenne, B., Madsen, M. K., Stenbæk, D. S., Armand, S., ... & Fisher, P. M. (2022). Psilocybin modulation
    of time-varying functional connectivity is associated with plasma psilocin and subjective effects. Neuroimage, 264, 119716.
    https://doi.org/10.1016/j.neuroimage.2022.119716

    Vohryzek, J., Deco, G., Cessac, B., Kringelbach, M. L., & Cabral, J. (2020). Ghost attractors in spontaneous brain activity:
    Recurrent excursions into functionally-relevant BOLD phase-locking states. Frontiers in systems neuroscience, 14, 20.
    https://doi.org/10.3389/fnsys.2020.00020
    """
    name = "CONT Leading Eigenvector Dynamics"

    def __init__(self,
                 time_series: np.ndarray,
                 flip_eigenvectors: bool = True,
                 diagonal: int = 0,
                 fisher_z: bool = False,
                 tril: bool = False):

        super().__init__(time_series, diagonal, fisher_z, tril)

        self.N_estimates = self.T
        self.dfc = np.full((self.P,self.P, self.N_estimates), np.nan)
        self.flip_eigenvectors = flip_eigenvectors
        self.V1 = np.full((self.N_estimates, self.P), np.nan)

    def estimate(self):
        """
        Calculate Leading Eigenvector Dynamics Analysis (LeiDA).
        The leading eigenvectors are saved in as the V1 attribute.

        Returns
        -------
        np.ndarray
            Dynamic functional connectivity as a PxPxN array.
        """
        analytic_signal = hilbert(self.time_series, axis=0)
        instantaneous_phase = np.angle(np.asarray(analytic_signal))

        # Compute the leading eigenvector for each time point
        for n in range(self.N_estimates):
            cohmat = np.cos(np.subtract.outer(instantaneous_phase[n,:], instantaneous_phase[n,:])) # Compute coherence matrix
            _, eigenvectors = eigh(cohmat) # Compute the eigenvectors (ignore eigenvalues)
            V1 = eigenvectors[:, -1] # The leading eigenvector is the one corresponding to the largest eigenvalue

            if self.flip_eigenvectors:
                # Make sure the largest component is negative
                if np.mean(V1 > 0) > 0.5:
                    V1 = -V1

            self.dfc[:, :, n] = np.outer(V1, V1)
            self.V1[n, :] = V1

        self.dfc = self.postproc()
        return self.dfc

class WaveletCoherence(ConnectivityMethod):
    """
    Instantaneous Wavelet Coherence.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series data.
    method : {'weighted'}, optional
        The method to use for calculating wavelet coherence. Default is 'weighted'.
    TR : float, optional
        Repetition time of the data. Default is 0.72.
    fmin : float, optional
        Minimum frequency for wavelet transform. Default is 0.007.
    fmax : float, optional
        Maximum frequency for wavelet transform. Default is 0.15.
    n_scales : int, optional
        Number of scales for wavelet transform. Default is 15.
    drop_scales : int, optional
        Number of scales to drop from the edges. Default is 2.
    drop_timepoints : int, optional
        Number of time points to drop from the edges. Default is 50.
    diagonal : int, optional
        Value to set on the diagonal of connectivity matrices. Default is 0.
    fisher_z : bool, optional
        Whether to apply Fisher z-transformation. Default is False.
    tril : bool, optional
        Whether to return only the lower triangle of the matrices. Default is False.

    References
    ----------
    Jacob Billings, Manish Saggar, Jaroslav Hlinka, Shella Keilholz, Giovanni Petri; Simplicial and
    topological descriptions of human brain dynamics. Network Neuroscience 2021; 5 (2): 549–568.
    https://doi.org/10.1162/netn_a_00190
    """
    name = "CONT Wavelet Coherence"

    def __init__(self,
                 time_series: np.ndarray,
                 method: Literal["weighted"] = "weighted",
                 TR: float = 0.72,
                 fmin: float = 0.007,
                 fmax: float = 0.15,
                 n_scales: int = 15,
                 drop_scales: int = 2,
                 drop_timepoints: int = 50,
                 progress_bar: bool = True,
                 diagonal: int = 0,
                 fisher_z: bool = False,
                 tril: bool = False):

        super().__init__(time_series, diagonal, fisher_z, tril)

        self.N_estimates = self.T
        self.dfc = np.full((self.P,self.P, self.N_estimates), np.nan)
        self.method = method
        self.TR = TR
        self.fmin = fmin
        self.fmax = fmax
        self.n_scales = n_scales
        self.drop_scales = drop_scales
        self.drop_timepoints = drop_timepoints
        self.progress_bar = progress_bar
        self.iwc = None

    def estimate(self):
        """
        Calculate instantaneous wavelet coherence.

        Returns
        -------
        np.ndarray
            Dynamic functional connectivity as a PxPxN array.
        """
        # Time series dimensions
        P = self.time_series.shape[1]
        T = self.time_series.shape[0]

        # Initial parameters
        dt = self.TR # TR of the data
        s0 = 1 / self.fmax # Smallest scale of the wavelet
        J = self.n_scales - 1 # Scales range from s0 up to s0 * 2**(J * dj), which gives a total of (J + 1) scales
        dj = np.log2(self.fmax / self.fmin) / J  # Spacing between discrete scales to achieve the desired frequency range
        mother_wavelet = Morlet(6) # Morlet wavelet with omega_0 = 6
        self.iwc = np.full((P, P, J+1, T), np.nan) # resulting wavelet coherence matrices

        W_list = []
        S_list = []
        for i in range(P):
            # Calculate continuous wavelet transform
            y_normal = (self.time_series[:,i] - self.time_series[:,i].mean()) / self.time_series[:,i].std()
            W, s, _, _, _, _ = cwt(y_normal, dt, dj=dj, s0=s0, J=J, wavelet=mother_wavelet)
            scales = np.ones([1, y_normal.size]) * s[:, None]
            S = mother_wavelet.smooth(np.abs(W) ** 2 / scales, dt, dj, s) # Smooth the wavelet spectra

            W_list.append(W)
            S_list.append(S)

        y_normal = (self.time_series[:,0] - self.time_series[:,0].mean()) / self.time_series[:,0].std()
        _, s1, _, _, _, _ = cwt(y_normal, dt, dj=dj, s0=s0, J=J, wavelet=mother_wavelet)
        scales = np.ones([1, y_normal.size]) * s1[:, None]

        total_pairs = P * (P + 1) // 2
        pair_iter = ((i, j) for i in range(P) for j in range(i, P))
        for i, j in tqdm(pair_iter, total=total_pairs, disable=not self.progress_bar, desc="Wavelet coherence", dynamic_ncols=True):
            W1 = W_list[i]; W2 = W_list[j]
            S1 = S_list[i]; S2 = S_list[j]

            # Calculate wavelet coherence
            W12 = W1 * np.conj(W2)
            S12 = mother_wavelet.smooth(W12 / scales, dt, dj, s1)
            WCT = np.abs(S12) ** 2 / (S1 * S2)
            self.iwc[i, j, :, :] = WCT
            self.iwc[j, i, :, :] = WCT

            if self.method == "weighted":
                CWP = np.abs(W1 * np.conj(W2))
                CWP = CWP[self.drop_scales:-self.drop_scales,
                        self.drop_timepoints:-self.drop_timepoints]
                WCT2 = WCT[self.drop_scales:-self.drop_scales,
                        self.drop_timepoints:-self.drop_timepoints]
                cross_power = CWP / np.sum(CWP, axis=0)

                vals = 1 - (cross_power * WCT2).sum(axis=0)
                self.dfc[i, j, self.drop_timepoints:-self.drop_timepoints] = vals
                self.dfc[j, i, self.drop_timepoints:-self.drop_timepoints] = vals
            else:
                raise NotImplementedError("Other methods not yet implemented")

        # Get rid of empty time points
        self.dfc = self.dfc[:,:, self.drop_timepoints:-self.drop_timepoints]
        self.dfc = self.postproc()

        return self.dfc

class DCC(ConnectivityMethod):
    """
    Dynamic Conditional Correlation (DCC) as described by Lindquist et al. (2014).

    Parameters
    ----------
    time_series : np.ndarray
        The input time series data.
    num_cores : int, optional
        Number of CPU cores to use for parallel processing. Default is 16.
    standardizeData : bool, optional
        Whether to standardize the time series data. Default is True.
    diagonal : int, optional
        Value to set on the diagonal of connectivity matrices. Default is 0.
    fisher_z : bool, optional
        Whether to apply Fisher z-transformation. Default is False.
    tril : bool, optional
        Whether to return only the lower triangle of the matrices. Default is False.

    References
    ----------
    Lindquist, M. A., Xu, Y., Nebel, M. B., & Caffo, B. S. (2014). Evaluating dynamic bivariate correlations
    in resting-state fMRI: a comparison study and a new approach. NeuroImage, 101, 531-546.
    https://doi.org/10.1016/j.neuroimage.2014.06.052
    """
    name = "CONT Dynamic Conditional Correlation"

    def __init__(self,
                 time_series: np.ndarray,
                 num_cores: int = 16,
                 standardizeData: bool = True,
                 progress_bar: bool = True,
                 diagonal: int = 0,
                 fisher_z: bool = False,
                 tril: bool = False):

        super().__init__(time_series, diagonal, fisher_z, tril)

        self.N_estimates = self.T
        self.dfc = np.full((self.P,self.P, self.N_estimates), np.nan)
        self.standardizeData = standardizeData
        self.progress_bar = progress_bar
        self.num_cores = num_cores

        self.H = None # dynamic conditional covariance tensor
        self.Theta = None # GARCH(1,1) parameters
        self.X = None # DCC parameters

    def _loglikelihood_garch11(self, theta, data):
        """
        Calculates a real number proportional to the negative log-likelihood of the GARCH(1,1) model

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
        epsilon : n-by-1 np.ndarray
            standardized residual vector
        d : 1-by-n np.ndarray
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
        """
        Calculates the time-varying conditional correlation matrices from standardized returns

        Parameters
        ----------
        epsilon : T-by-N matrix
            standardized residual time series
        theta : 1-by-2 vector
            dcc parameter vector

        Returns
        -------
        R : N-by-N-by-T np.ndarray
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
        """
        Calculates the correlation component of the log-likelihood (eq. 29)

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
        T, _ = epsilon.shape
        S2 = np.cov(epsilon, rowvar=False)
        SS = S2 * (1 - sum(theta))
        Q = S2.copy()
        R = np.diag(1/np.sqrt(np.diag(Q))) @ Q @ np.diag(1/np.sqrt(np.diag(Q)))

        output = 0
        for t in range(T):
            output += np.log(det(R)) + epsilon[t, :] @ inv(R) @ epsilon[t, :]
            temp = epsilon[t, :] * np.sqrt(np.diag(Q))
            Q = SS + theta[0] * np.outer(temp, temp) + theta[1] * Q
            R = np.diag(1/np.sqrt(np.diag(Q))) @ Q @ np.diag(1/np.sqrt(np.diag(Q)))

        return output

    def _compute_garch(self, n):
        """
        Fit a univariate GARCH(1,1) model to a time series and calculate standardized residuals and conditional volatilities.

        Parameters
        ----------
        n : int
            Index of the time series to be processed.

        Returns
        -------
        tuple
            - np.ndarray : Fitted GARCH(1,1) parameters.
            - np.ndarray : Standardized residuals.
            - np.ndarray : Estimated conditional volatilities.
        """
        ts_n = np.ascontiguousarray(self.time_series[:, n])
        constraints = {'type': 'ineq', 'fun': lambda x: np.array([1 - x[0] - x[1] - x[2], x[0], x[1], x[2]])}
        bounds = ((0, 1), (0, 1), (0, 1))
        theta0 = (0.25, 0.25, 0.25)
        res = minimize(self._loglikelihood_garch11, theta0, args=(ts_n,), constraints=constraints, bounds=bounds)

        ep, d = self._rToEpsilon(ts_n, res.x)
        return res.x, ep, d

    def estimate(self):
        """
        DCC algorithm

        Parameters
        ----------
        theta : T-by-N matrix
            fMRI time series data

        Returns
        -------
        R : N*N*T np.ndarray
            estimated dynamic conditional correlation tensor
        """
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
        if getattr(self, "num_cores", 1) == 1:
            it = tqdm(range(N), desc="DCC GARCH (1 core)") if getattr(self, "progress_bar", True) else range(N)
            results = [self._compute_garch(n) for n in it]
        else:
            cm = tqdm_joblib(total=N, desc=f"DCC GARCH ({self.num_cores} cores)") if getattr(self, "progress_bar", True) else None
            gen = (delayed(self._compute_garch)(n) for n in range(N))
            if cm:
                with cm:
                    results = Parallel(n_jobs=self.num_cores, prefer="processes", batch_size=1)(gen)
            else:
                results = Parallel(n_jobs=self.num_cores, prefer="processes", batch_size=1)(gen)

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
            except Exception as e:
                print(f"Exception: {e}")
                print(f"Attempt {attempt + 1} failed, trying new random initial values")
                x0 = (random.uniform(0.25, 0.5), random.uniform(0.25, 0.5))
                if attempt == 4:
                    print("All attempts failed.")
        X = res.x

        # Time-varying conditional correlation matrices
        R = self._epsilonToR(epsilon, X)

        # Time-varying conditional covariance matrices
        for t in tqdm(range(T), disable=not self.progress_bar, desc="Conditional covariance matrices", dynamic_ncols=True):
            H[:,:,t] = np.diag(np.sqrt(D[t,:])) @ R[:,:,t] @ np.diag(np.sqrt(D[t,:]))
        
        self.dfc = R 
        self.dfc = self.postproc()

        self.H = H
        self.Theta = Theta
        self.X = X

        return self.dfc

class EdgeConnectivity(ConnectivityMethod):
    """
    Edge-centric connectivity method.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series data.
    method : string, optional
        The specific connectivity to calculate. Default is "eTS".
            - eTS: returns the edge time series (edges x time)
            - eFC: returns the edge functional connectivity (edges x edges x time).
    standardizeData : bool, optional
        Whether to standardize the time series data. Default is True.
    vlim : float, optional
        Limit for plotting in the GUI (not used in the method itself). Default is 3.

    References
    ----------
    Faskowitz, J., Esfahlani, F. Z., Jo, Y., Sporns, O., & Betzel, R. F. (2020). Edge-centric functional
    network representations of human cerebral cortex reveal overlapping system-level architecture.
    Nature neuroscience, 23(12), 1644–1654. DOI: https://doi.org/10.1038/s41593-020-00719-y
    """
    name = "CONT Edge-centric Connectivity"

    def __init__(self,
                 time_series: np.ndarray,
                 method: Literal["eTS", "eFC"] = "eTS",
                 standardizeData: bool = True,
                 labels: list = None,
                 vlim: float = 3):

        super().__init__(time_series, 0, False, False)
        self.method = method
        self.standardizeData = standardizeData
        self.labels = labels
        self.edge_labels = None
        self.u = None # Row indices of the upper triangle of the connectivity matrix
        self.v = None # Column indices of the upper triangle of the connectivity matrix

    def estimate(self):
        """
        Calculate edge-centric connectivity (eTS or eFC).

        Returns
        -------
        np.ndarray
            Dynamic functional connectivity depending on the method.
                For eTS: Time x Edge array.
                For eFC: Edge x Edge x Time array.
        """
        z = zscore(self.time_series, axis=0, ddof=1) if self.standardizeData else self.time_series
        self.u, self.v = np.triu_indices(self.time_series.shape[1], k=1)
        print(self.u, self.v)
        a = np.multiply(z[:, self.u], z[:, self.v]) # edge time series

        # If labels are provided we can create edge names
        if isinstance(self.labels, list):
            self.edge_labels = [f"{self.labels[i]}–{self.labels[j]}" for i, j in zip(self.u, self.v)]

        if self.method == "eTS":
            return a

        b = a.T @ a # Inner product
        c = np.sqrt(np.diag(b)) # Square root of the diagonal elements (variance) to normalize
        d = np.outer(c, c) # Normalization matrix
        eFC = b / d # Element-wise division to get the correlation matrix

        eFC = self.postproc()
        self.dfc = eFC

        return self.dfc


"""
SECTION: State based dFC methods
"""
class SlidingWindowClustering(ConnectivityMethod):
    """
    Siding window clustering (SWC) state-based connectivity (2-level clustering).

    References
    ----------
    Torabi, M., Mitsis, G. D., & Poline, J. B. (2024). On the variability of 
    dynamic functional connectivity assessment methods. GigaScience, 13, giae009.
    https://doi.org/10.1093/gigascience/giae009

    Parameters
    ----------
    time_series : list or np.ndarray
        The input time series data.
    n_states : int, optional
        Number of states for the method. Default is 5.
    subject_clusters : int, optional
        Number of clusters for the first level clustering. Default is 5.
    windowsize : int, optional
        Size of the sliding window. Default is 29.
    shape : str, optional
        Shape of the window. Default is "gaussian".
    std : float, optional
        Standard deviation for gaussian window. Default is 10.
    stepsize : int, optional
        Step size for the sliding window. Default is 15.
    """
    name = "STATE Sliding Window Clustering"

    def __init__(self,
                 time_series: Union[np.ndarray, list],
                 n_states: int = 5,
                 subject_clusters: int = 5,
                 windowsize: int = 29,
                 shape: Literal["rectangular", "gaussian", "hamming"] = "gaussian",
                 std: float = 10,
                 stepsize: int = 15,
                 random_state: int | None = None,
                 n_init: int = 50,
                 progress_bar: bool = True):

        super().__init__(time_series, 0, False, False)
        self.n_states = n_states
        self.windowsize = windowsize
        self.shape = shape
        self.std = std
        self.stepsize = stepsize
        self.random_state = random_state
        self.n_init = n_init
        self.progress_bar = progress_bar
        self.subject_clusters = subject_clusters
        self.N_estimates = (self.T - self.windowsize) // self.stepsize + 1

    def vec2mat(self, F, N):
        T = F.shape[0]
        C = np.zeros((T, N, N), dtype=F.dtype)
        iu = np.triu_indices(N, k=1)
        C[:, iu[0], iu[1]] = F # Assign vectorized values to upper triangles
        C = C + np.transpose(C, (0, 2, 1)) + np.eye(N) # Symmetrize and set main diagonal
        return C

    def mat2vec(self, C_t):
        if C_t.ndim == 2:
            # 2D square matrix
            return C_t[np.triu_indices_from(C_t, k=1)]
        elif C_t.ndim == 3:
            # 3D array, C_t is a stack of square matrices.
            idx = np.triu_indices_from(C_t[0], k=1)
            return C_t[:, idx[0], idx[1]]
        else:
            raise ValueError("Input must be a 2D or 3D array.")

    def estimate(self):
        """
        Estimate state-based connectivity

        Returns
        -------
        np.ndarray
            State time course (n_subjects x T)
        np.ndarray
            Connectivity states (P x P x n_states)
        """
        FCS_1st_level = None
        SW_dFC = None

        for i in tqdm(range(self.n_subjects), disable=not self.progress_bar, desc="Sliding Window Clustering", dynamic_ncols=True):
            # Sliding window
            subject_ts = self.time_series3D[i, :, :]
            sw = SlidingWindow(time_series=subject_ts, windowsize=self.windowsize, stepsize=self.stepsize, shape=self.shape, std=self.std, diagonal=1)
            dfc = sw.estimate()
            dfc = np.moveaxis(dfc, -1, 0)
            F = self.mat2vec(dfc)

            # First level clustering
            kmeans_ = KMeans(n_clusters=self.subject_clusters, random_state=self.random_state, n_init=self.n_init).fit(F)
            F_cent = kmeans_.cluster_centers_

            FCS = self.vec2mat(F_cent, N=self.P)

            if FCS_1st_level is None:
                FCS_1st_level = FCS
            else:
                FCS_1st_level = np.concatenate((FCS_1st_level, FCS), axis=0)

            if SW_dFC is None:
                SW_dFC = dfc
            else:
                SW_dFC = np.concatenate((SW_dFC, dfc), axis=0)

        # Second level clustering
        F = self.mat2vec(FCS_1st_level)
        kmeans_ = KMeans(n_clusters=self.n_states, n_init=self.n_init, random_state=self.random_state).fit(F)
        F_cent = kmeans_.cluster_centers_

        self.states = self.vec2mat(F_cent, N=self.P)
        self.states = self.states.transpose(1, 2, 0)

        self.state_tc = kmeans_.predict(self.mat2vec(SW_dFC))
        self.state_tc = self.state_tc.reshape(self.n_subjects, len(self.state_tc)//self.n_subjects)

        return self.state_tc, self.states

class KSVD(ConnectivityMethod):
    """
    Windowless state-based connectivity based on K-SVD.

    References
    ----------
    Torabi, M., Mitsis, G. D., & Poline, J. B. (2024). On the variability of 
    dynamic functional connectivity assessment methods. GigaScience, 13, giae009.
    https://doi.org/10.1093/gigascience/giae009

    Rubinstein, R., Zibulevsky, M., & Elad, M. (2008). Efficient implementation of the 
    K-SVD algorithm using batch orthogonal matching pursuit. Cs Technion, 40(8), 1-15.

    Parameters
    ----------
    time_series : list or np.ndarray
        The input time series data.
    n_states : int, optional
        Number of states for the method. Default is 5.
    """
    name = "STATE K-SVD"

    def __init__(self,
                 time_series: Union[np.ndarray, list],
                 n_states: int = 5):

        super().__init__(time_series, 0, False, False)
        self.n_states = n_states
        self.N_estimates = self.n_subjects * self.T
        
        self.state_tc = np.zeros(self.N_estimates)
        self.states = np.full((self.P,self.P, self.n_states), np.nan)

    def estimate(self):
        """
        Estimate state-based connectivity

        Returns
        -------
        np.ndarray
            State time course (n_subjects x T)
        np.ndarray
            Connectivity states (P x P x n_states)
        """
        # Estimate states
        aksvd = ApproximateKSVD(n_components=self.n_states, transform_n_nonzero_coefs=1)
        dictionary = np.asarray(aksvd.fit(self.time_series).components_)
        gamma = aksvd.transform(self.time_series)

        # Build state matrices: v_k v_k^T for each atom k (P x P x K)
        self.states = np.einsum('kp,kq->pqk', dictionary, dictionary)

        # Assign each time sample to a state and reshape to (n_subjects, T)
        idx = np.argmax(np.abs(gamma), axis=1)  # (N,)
        self.state_tc = idx.reshape(self.n_subjects, self.T)

        return self.state_tc, self.states

class CoactivationPatterns(ConnectivityMethod):
    """
    Co-activation patterns (state-based connectivity).

    References
    ----------
    Torabi, M., Mitsis, G. D., & Poline, J. B. (2024). On the variability of 
    dynamic functional connectivity assessment methods. GigaScience, 13, giae009.
    https://doi.org/10.1093/gigascience/giae009

    Parameters
    ----------
    time_series : list or np.ndarray
        The input time series data.
    n_states : int, optional
        Number of states for the method. Default is 5.
    subject_clusters : int, optional
        Number of subject clusters. Default is 5.
    """
    name = "STATE Co-activation Patterns"

    def __init__(self,
                 time_series: Union[np.ndarray, list],
                 n_states: int = 5,
                 subject_clusters: int = 5,
                 random_state: int | None = None,
                 n_init: int = 50,
                 progress_bar: bool = True):

        super().__init__(time_series, 0, False, False)
        
        self.N_estimates = self.T * time_series.shape[-1] if isinstance(time_series, np.ndarray) else self.T * len(time_series)
        self.n_states = n_states
        self.subject_clusters = subject_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.progress_bar = progress_bar
       
    def cluster_ts(self, act, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, n_init=self.n_init, random_state=self.random_state).fit(act)
        centroids = kmeans.cluster_centers_

        return centroids, kmeans

    def estimate(self):
        """
        Estimate state-based connectivity

        Returns
        -------
        np.ndarray
            State time course (n_subjects x T)
        np.ndarray
            Connectivity states (P x P x n_states)
        """
        center_1st_level = None
        for subject in tqdm(range(self.n_subjects), disable=not self.progress_bar, desc="CAP", dynamic_ncols=True):
            ts = self.time_series3D[subject, :,:]

            if ts.shape[0] < self.subject_clusters:
                print(f"Number of subject-level clusters changed to {ts.shape[0]} as they cannot be more than time samples.")
                self.subject_clusters = ts.shape[0]

            centroids, _ = self.cluster_ts(act=ts, n_clusters=self.subject_clusters)
            
            if center_1st_level is None:
                center_1st_level = centroids
            else:
                center_1st_level = np.concatenate((center_1st_level, centroids), axis=0)
        
        group_centroids, kmeans= self.cluster_ts(act=center_1st_level, n_clusters=self.n_states)
        
        self.states = np.full((self.P,self.P, self.n_states), np.nan)
        for i, group_centroid in enumerate(group_centroids):
            self.states[:,:,i] = np.multiply(group_centroid[:,np.newaxis], group_centroid[np.newaxis,:])

        self.state_tc = kmeans.predict(self.time_series)
        self.state_tc = self.state_tc.reshape(self.n_subjects, self.T)

        return self.state_tc, self.states

class ContinuousHMM(ConnectivityMethod):
    """
    Continuous hidden markov model (state-based connectivity).

    References
    ----------
    Torabi, M., Mitsis, G. D., & Poline, J. B. (2024). On the variability of 
    dynamic functional connectivity assessment methods. GigaScience, 13, giae009.
    https://doi.org/10.1093/gigascience/giae009

    Parameters
    ----------
    time_series : list or np.ndarray
        The input time series data.
    n_states : int, optional
        Number of states for the method. Default is 5.
    hmm_iter : int, optional
        Number of iterations for the HMM. Default is 20.
    """
    name = "STATE Continuous Hidden Markov Model"

    def __init__(self,
                 time_series: Union[np.ndarray, list],
                 n_states: int = 5,
                 hmm_iter: int = 20,
                 progress_bar: bool = True):

        super().__init__(time_series, 0, False, False)

        self.N_estimates = self.T * time_series.shape[-1] if isinstance(time_series, np.ndarray) else self.T * len(time_series)
        self.n_states = n_states
        self.hmm_iter = hmm_iter
        self.progress_bar = progress_bar

    def estimate(self):
        """
        Estimate state-based connectivity

        Returns
        -------
        np.ndarray
            State time course (n_subjects x T)
        np.ndarray
            Connectivity states (P x P x n_states)
        """
        models, scores = [], []
        for _ in tqdm(range(self.hmm_iter), disable=not self.progress_bar, desc="Continuous HMM", dynamic_ncols=True):
            model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full")
            model.fit(self.time_series)
            models.append(model)

            score = model.score(self.time_series)
            scores.append(score)

        hmm_model = models[np.argmax(scores)]
        self.states = hmm_model.covars_
        self.states = self.states.transpose(1, 2, 0)

        self.state_tc = hmm_model.predict(self.time_series)
        self.state_tc = self.state_tc.reshape(self.n_subjects, self.T)

        return self.state_tc, self.states

class DiscreteHMM(ConnectivityMethod):
    """
    Discrete hidden markov model (state-based connectivity).

    References
    ----------
    Torabi, M., Mitsis, G. D., & Poline, J. B. (2024). On the variability of 
    dynamic functional connectivity assessment methods. GigaScience, 13, giae009.
    https://doi.org/10.1093/gigascience/giae009

    Parameters
    ----------
    time_series : list or np.ndarray
        The input time series data.
    n_states : int, optional
        Number of states for the method. Default is 5.
    state_ratio : float, optional
        Ratio of states to use for clustering. Default is 3/5.
    subject_clusters : int, optional
        Number of subject clusters. Default is 5.
    windowsize : int, optional
        Size of the sliding window. Default is 29.
    shape : str, optional
        Shape of the window. Default is "gaussian".
    std : float, optional
        Standard deviation for gaussian window. Default is 10.
    stepsize : float, optional
        Step size for the sliding window. Default is 15.
    hmm_iter : int, optional
        Number of iterations for the HMM. Default is 20.
    """
    name = "STATE Discrete Hidden Markov Model"

    def __init__(self,
                 time_series: Union[np.ndarray, list],
                 n_states: int = 5,
                 state_ratio: float = 3/5,
                 subject_clusters: int = 5,
                 windowsize: int = 29,
                 shape: Literal["rectangular", "gaussian", "hamming"] = "gaussian",
                 std: float = 10,
                 stepsize: int = 15,
                 hmm_iter: int = 20,
                 progress_bar: bool = True):

        super().__init__(time_series, 0, False, False)
        self.time_series = self.time_series3D
        self.n_states = n_states
        self.state_ratio = state_ratio
        self.subject_clusters = subject_clusters
        self.windowsize = windowsize
        self.shape = shape
        self.std = std
        self.stepsize = stepsize
        self.hmm_iter = hmm_iter
        self.progress_bar = progress_bar
        self.N_estimates = (self.T - self.windowsize) // self.stepsize + 1

    def estimate(self):
        """
        Estimate state-based connectivity

        Returns
        -------
        np.ndarray
            State time course (n_subjects x T)
        np.ndarray
            Connectivity states (P x P x n_states)
        """
        # Run sliding window clustering
        n_cluster_states = int(self.n_states * self.state_ratio)
        state_tc, states = SlidingWindowClustering(self.time_series,
                                                   n_states=n_cluster_states,
                                                   subject_clusters=self.subject_clusters,
                                                   windowsize=self.windowsize,
                                                   shape=self.shape,
                                                   stepsize=self.stepsize).estimate()
        
        states = states.transpose(2, 0, 1)
        SWC_dFC = states[state_tc.flatten()]
        state_tc = state_tc.reshape(-1, 1)

        # Fit the categorical HMM
        models, scores = [], []
        for i in tqdm(range(self.hmm_iter), disable=not self.progress_bar, desc="Discrete HMM", dynamic_ncols=True):
            model = hmm.CategoricalHMM(n_components=self.n_states)
            model.fit(state_tc)
            models.append(model)
            
            score = model.score(state_tc)
            scores.append(score)

        # Select the best model and get the states/connectivity estimates
        hmm_model = models[np.argmax(scores)]
        self.state_tc = hmm_model.predict(state_tc)

        self.states = np.full((self.P, self.P, self.n_states), np.nan)
        for i in range(self.n_states):
            ids = np.array([int(state == i) for state in self.state_tc])
            self.states[:,:,i] = np.average(SWC_dFC, weights=ids, axis=0)

        self.state_tc = self.state_tc.reshape(self.n_subjects, self.N_estimates)

        return self.state_tc, self.states


"""
SECTION: Static FC methods
"""
class Static_Pearson(ConnectivityMethod):
    """
    Static functional connectivity method using Pearson correlation.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series data.
    cov_estimator : str, optional
        Method to estimate covariance. Default is Ledoit-Wolf shrinkage.
    diagonal : int, optional
        Value to set on the diagonal of connectivity matrices. Default is 0.
    fisher_z : bool, optional
        Whether to apply Fisher z-transformation. Default is False.
    tril : bool, optional
        Whether to return only the lower triangle of the matrices. Default is False.
    """
    name = "STATIC Pearson Correlation"

    def __init__(self,
                 time_series: np.ndarray,
                 cov_estimator: Literal[None, "LedoitWolf"] = "LedoitWolf",
                 diagonal: int = 0,
                 fisher_z: bool = False,
                 tril: bool = False):

        self.fc = None
        self.cov_estimator = cov_estimator
        super().__init__(time_series, diagonal, fisher_z, tril)

    def estimate(self):
        """
        Estimate the functional connectivity.

        Returns
        -------
        np.ndarray
            Static functional connectivity matrix.
        """
        if self.cov_estimator == "LedoitWolf":
            C = LedoitWolf().fit(self.time_series).covariance_
        else:
            C = np.cov(self.time_series.T)
        
        D = np.diag(1/np.sqrt(np.diag(C)))
        self.fc = D @ C @ D
    
        self.fc = self.postproc()
        return self.fc

class Static_Partial(ConnectivityMethod):
    """
    Static functional connectivity method using partial correlation.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series data.
    cov_estimator : str, optional
        Method to estimate covariance. If "LedoitWolf", it uses Ledoit-Wolf shrinkage. Default is LedoitWolf.
    diagonal : int, optional
        Value to set on the diagonal of connectivity matrices. Default is 0.
    fisher_z : bool, optional
        Whether to apply Fisher z-transformation. Default is False.
    tril : bool, optional
        Whether to return only the lower triangle of the matrices. Default is False.
    """
    name = "STATIC Partial Correlation"

    def __init__(self,
                 time_series: np.ndarray,
                 cov_estimator: Literal["LedoitWolf", None] = "LedoitWolf",
                 diagonal: int = 0,
                 fisher_z: bool = False,
                 tril: bool = False):

        self.fc = None
        self.cov_estimator = cov_estimator
        super().__init__(time_series, diagonal, fisher_z, tril)

    def estimate(self):
        """
        Estimate the functional connectivity.

        Returns
        -------
        np.ndarray
            Static functional connectivity matrix.
        """
        if self.cov_estimator == "LedoitWolf":
            C = LedoitWolf().fit(self.time_series).covariance_
        else:
            C = np.cov(self.time_series.T)
        
        P = np.linalg.inv(C)
        D = np.diag(1/np.sqrt(np.diag(P)))
        self.fc = -D @ P @ D

        self.fc = self.postproc()
        return self.fc

class Static_Mutual_Info(ConnectivityMethod):
    """
    Static functional connectivity method using mutual information.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series data.
    num_bins : int, optional
        Number of bins to use for the mutual information calculation. Default is 10.
    diagonal : int, optional
        Value to set on the diagonal of connectivity matrices. Default is 0.
    fisher_z : bool, optional
        Whether to apply Fisher z-transformation. Default is False.
    tril : bool, optional
        Whether to return only the lower triangle of the matrices. Default is False.
    """
    name = "STATIC Mutual Information"

    def __init__(self,
                 time_series: np.ndarray,
                 num_bins: int = 10,
                 diagonal: int = 0,
                 fisher_z: bool = False,
                 tril: bool = False):
        
        self.fc = None
        self.num_bins = num_bins
        super().__init__(time_series, diagonal, fisher_z, tril)

    def estimate(self):
        """
        Estimate the functional connectivity.

        Returns
        -------
        np.ndarray
            Static functional connectivity matrix.
        """
        if self.num_bins is None:
            raise ValueError("Number of bins must be specified for mutual information method")

        binned_data = np.zeros_like(self.time_series, dtype=int)

        # Determine the bin edges and bin the data for each time series
        for i in range(self.P):
            bin_edges = np.histogram_bin_edges(self.time_series[:, i], bins=self.num_bins)
            binned_data[:, i] = np.digitize(self.time_series[:, i], bins=bin_edges, right=False)

        self.fc = np.zeros((self.P, self.P))

        for i in range(self.P):
            for j in range(i + 1, self.P):
                mi = mutual_info_score(binned_data[:, i], binned_data[:, j])
                self.fc[i, j] = mi
                self.fc[j, i] = mi

        self.fc = self.postproc()
        return self.fc

class Static_Covariance(ConnectivityMethod):
    name = "STATIC Covariance"

    def __init__(self,
                 time_series: np.ndarray,
                 cov_estimator: Literal["LedoitWolf", None] = "LedoitWolf",
                 diagonal: int = 0,
                 fisher_z: bool = False,
                 tril: bool = False):

        self.fc = None
        self.cov_estimator = cov_estimator
        super().__init__(time_series, diagonal, fisher_z, tril)

    def estimate(self):
        """
        Estimate the functional connectivity.

        Returns
        -------
        np.ndarray
            Static functional connectivity matrix.
        """
        if self.cov_estimator == "LedoitWolf":
            self.fc = LedoitWolf().fit(self.time_series).covariance_
        else:
            self.fc = np.cov(self.time_series.T)
        
        self.fc = self.postproc()
        return self.fc
