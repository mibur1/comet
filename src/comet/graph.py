import bct
import warnings
import numpy as np
import scipy.sparse
from numba import njit
from typing import Literal

# Ignore warnings from bctpy about NaNs in centrality calculations
warnings.filterwarnings("ignore", message=r"invalid value encountered in divide",
                        category=RuntimeWarning, module=r".*bct\.algorithms\.centrality")

"""
SECTION: Graph processing functions
 - General functions to preprocess connectivity/adjacency matrices
   before graph analysis.
"""
def handle_negative_weights(G: np.ndarray,
                            type: Literal["absolute", "discard"] = "absolute",
                            copy: bool = True) -> np.ndarray:
    '''
    Handle negative weights in a connectivity/adjacency matrix

    Connectivity methods can produce negative estimates, which can be handled in different ways before graph analysis.

    Parameters
    ----------
    G : np.ndarray
        2D (PxP) or 3D (PxPxT) adjacency/connectivity matrix

    type : string, optional
        type of handling, can be *absolute* or *discard*
        default is *absolute*

    copy : bool, optional
        if True, a copy of W is returned, otherwise W is modified in place
        default is True

    Returns
    -------
    G : PxP np.ndarray
        adjacency/connectivity matrix with only positive weights
    '''

    if copy:
        G = G.copy()

    if type == "absolute":
        G = np.abs(G)
    elif type == "discard":
        G[G < 0] = 0
    else:
        raise NotImplementedError("Options are: *absolute* or *discard*")
    return G

def threshold(G: np.ndarray,
              type: Literal["density", "absolute"] = "density",
              threshold: float = None,
              density: float = 0.2,
              copy: bool = True) -> np.ndarray:
    '''
    Thresholding of connectivity/adjacency matrix

    Performs absolute or density-based thresholding

    Parameters
    ----------
    G : np.ndarray
        2D (PxP) or 3D (PxPxT) adjacency/connectivity matrix

    type : string, optional
        type of thresholding, can be *absolute* or *density*
        default is *absolute*

    threshold : float, optional
        threshold value for absolute thresholding
        default is None

    density : float, optional
        density value for density-based thresholding, has to be between 0 and 1 (keep x% of strongest connections)
        default is 0.2 (20%)

    copy : bool, optional
        if True, a copy of W is returned, otherwise W is modified in place
        default is True

    Returns
    -------
    G : PxP np.ndarray
        thresholded adjacency/connectivity matrix

    Notes
    -----
    The implemented for density based thresholding always keeps the exact same number of connections. If multiple edges have the same weight,
    the included edges are chosen "randomly" (based on their order in the sorted indices). This is identical to the behaviour in the BCT implementation.
    '''

    if copy:
        G = G.copy()

    if type == "absolute":
        G[G < threshold] = 0
    elif type == "density":
        if density is None or not (0 <= density <= 1):
            raise ValueError("Density must be between 0 and 1.")

        def _threshold(A):
            if not np.allclose(A, A.T):
                raise ValueError("Matrix is not symmetrical")

            A = A.copy()
            A[np.tril_indices(len(A))] = 0
            triu = np.triu_indices_from(A, k=1)
            sorted_idx = np.argsort(A[triu])[::-1]
            cutoff = int(np.round(len(sorted_idx) * density + 1e-10))
            mask = np.zeros_like(A, dtype=bool)
            mask[triu[0][sorted_idx[:cutoff]], triu[1][sorted_idx[:cutoff]]] = True
            A[~mask] = 0
            A = A + A.T
            return A

        if G.ndim == 2:
            return _threshold(G)
        elif G.ndim == 3:
            return np.stack([_threshold(G[:, :, t]) for t in range(G.shape[2])], axis=2)
        else:
            raise ValueError("G must be a 2D or 3D array.")
    else:
        raise NotImplementedError("Thresholding must be of type *absolute* or *density*")
    return G

def binarise(G: np.ndarray,
             copy: bool = True) -> np.ndarray:
    '''
    Binarise connectivity/adjacency matrix

    Parameters
    ----------
    G : np.ndarray
        2D (PxP) or 3D (PxPxT) adjacency/connectivity matrix

    copy : bool, optional
        if True, a copy of W is returned, otherwise W is modified in place
        default is True

    Returns
    -------
    G : PxP np.ndarray
        binarised adjacency/connectivity matrix
    '''

    if copy:
        G = G.copy()

    G[G != 0] = 1
    return G

def normalise(G: np.ndarray,
              copy: bool = True) -> np.ndarray:
    '''
    Normalise connectivity/adjacency matrix

    Parameters
    ----------
    G : np.ndarray
        2D (PxP) or 3D (PxPxT) adjacency/connectivity matrix

    copy : bool, optional
        if True, a copy of W is returned, otherwise W is modified in place
        default is True

    Returns
    -------
    G : PxP np.ndarray
        normalised adjacency/connectivity matrix
    '''

    if copy:
        G = G.copy()

    if G.ndim == 2:
        max_val = np.max(np.abs(G))
        if max_val == 0:
            raise ValueError("Error: Matrix contains only zeros")
        G /= max_val

    elif G.ndim == 3:
        for t in range(G.shape[2]):
            max_val = np.max(np.abs(G[:, :, t]))
            if max_val == 0:
                raise ValueError(f"Error: Matrix at time index {t} contains only zeros")
            G[:, :, t] /= max_val
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

    return G

def invert(G: np.ndarray,
           copy: bool = True) -> np.ndarray:
    '''
    Invert connectivity/adjacency matrix

    Element wise inversion W such that each value W[i,j] will be 1 / W[i,j] (internode strengths internode distances)

    Parameters
    ----------
    G : np.ndarray
        2D (PxP) or 3D (PxPxT) adjacency/connectivity matrix

    copy : bool, optional
        if True, a copy of W is returned, otherwise W is modified in place
        default is True

    Returns
    -------
    G : PxP np.ndarray
        element wise inverted adjacency/connectivity matrix
    '''

    if copy:
        G = G.copy()

    G[G == 0] = np.inf
    return 1 / G

def logtransform(G: np.ndarray,
                 epsilon: float = 1e-10,
                 copy: bool = True) -> np.ndarray:
    '''
    Log transform of connectivity/adjacency matrix

    Element wise log transform of W such that each value W[i,j] will be -log(W[i,j]

    Parameters
    ----------
    G : np.ndarray
        2D (PxP) or 3D (PxPxT) adjacency/connectivity matrix

    epsilon : float, optional
        clipping value for numeric stability,
        default is 1e-10

    copy : bool, optional
        if True, a copy of W is returned, otherwise W is modified in place
        default is True

    Returns
    -------
    G : PxP np.ndarray
        element wise log transformed adjacency/connectivity matrix
    '''

    if copy:
        G = G.copy()

    if (G > 1).any() or (G <= 0).any():
        raise ValueError("All connections must be in the range (0, 1] to apply logtransform.")

    G_safe = np.clip(G, a_min=epsilon, a_max=None) # clip very small values for numeric stability
    return -np.log(G_safe)

def symmetrise(G: np.ndarray,
               copy: bool = True) -> np.ndarray:
    '''
    Symmetrise connectivity/adjacency matrix

    Symmetrise G such that each value G[i,j] will be G[j,i].

    Parameters
    ----------
    G : np.ndarray
        2D (PxP) or 3D (PxPxT) adjacency/connectivity matrix

    copy : bool, optional
        if True, a copy of W is returned, otherwise W is modified in place
        default is True

    Returns
    -------
    G : PxP np.ndarray
        symmetrised adjacency/connectivity matrix
    '''

    if copy:
        G = G.copy()

    def _symmetrise(A):
        """Symmetrise a single 2D matrix."""
        is_binary = np.all(np.isclose(A, 0) | np.isclose(A, 1))
        if is_binary:
            return np.logical_or(A, A.T).astype(float)
        else:
            return 0.5 * (A + A.T)

    if G.ndim == 2:
        return _symmetrise(G)
    elif G.ndim == 3:
        return np.stack([_symmetrise(G[:, :, t]) for t in range(G.shape[2])], axis=2)
    else:
        raise ValueError("Input must be a 2D or 3D array.")

def randomise(G: np.ndarray,
              copy: bool = True) -> np.ndarray:
    '''
    Randomly rewire edges of an adjacency/connectivity matrix

    Implemented as in https://github.com/rkdan/small_world_propensity

    Parameters
    ----------
    G : np.ndarray
        2D (PxP) or 3D (PxPxT) adjacency/connectivity matrix

    Returns
    -------
    G_rand : PxP np.ndarray
        randomised adjacency/connectivity matrix
    '''

    if copy:
        G = G.copy()

    def _randomise(A):
        """Randomisation for a single 2D matrix."""
        n = A.shape[0]
        mask = np.triu(np.ones((n, n)), 1).astype(bool)
        i, j = np.where(mask)
        edges = A[i, j]
        np.random.shuffle(edges)
        A_rand = np.zeros_like(A)
        A_rand[i, j] = edges
        A_rand[j, i] = edges  # ensure symmetry
        np.fill_diagonal(A_rand, np.diag(A))  # preserve diagonal
        return A_rand

    if G.ndim == 2:
        return _randomise(G)
    elif G.ndim == 3:
        return np.stack([_randomise(G[:, :, t]) for t in range(G.shape[2])], axis=2)
    else:
        raise ValueError("Input must be a 2D or 3D array.")

def regular_matrix(G: np.ndarray, r: int) -> np.ndarray:
    '''
    Create a regular matrix.

    Adapted from https://github.com/rkdan/small_world_propensity

    Parameters
    ----------
    G : np.ndarray
        2D (PxP) or 3D (PxPxT) adjacency/connectivity matrix.
    r : int
        Average effective radius of the network.

    Returns
    -------
    M : np.ndarray
        Regularised adjacency matrix (same shape as input).
    '''
    def _regularise(G2D, r):
        """Regularisation for a single 2D matrix."""
        n = G2D.shape[0]
        G_upper = np.triu(G2D)  # Keep only the upper triangular part
        B = np.sort(G_upper.flatten(order="F"))[::-1]  # Flatten and sort including zeros

        # Calculate padding and reshape B to match the second function
        num_els = np.ceil(len(B) / (2 * n))
        num_zeros = int(2 * n * num_els - n * n)
        B_extended = np.concatenate((B, np.zeros(num_zeros)))
        B_matrix = B_extended.reshape((n, -1), order="F")

        M = np.zeros((n, n))
        for i in range(n):
            for z in range(r):
                a = np.random.randint(0, n)
                # Adjust the condition for selecting a non-zero weight
                while (B_matrix[a, z] == 0 and z != r - 1) or \
                      (B_matrix[a, z] == 0 and z == r - 1 and np.any(B_matrix[:, r - 1] != 0)):
                    a = np.random.randint(0, n)

                y_coord = (i + z + 1) % n
                M[i, y_coord] = B_matrix[a, z]
                M[y_coord, i] = B_matrix[a, z]
                B_matrix[a, z] = 0  # Remove the used weight

        return M

    if G.ndim == 2:
        return _regularise(G, r)
    elif G.ndim == 3:
        return np.stack([_regularise(G[:, :, t], r) for t in range(G.shape[2])], axis=2)
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

def postproc(G: np.ndarray,
             diag: float = 0,
             copy: bool = True) -> np.ndarray:
    '''
    Postprocessing of connectivity/adjacency matrix

    Ensures G is symmetric, sets diagonal to diag, removes NaNs and infinities, and ensures exact binarity

    Parameters
    ----------
    G : np.ndarray
        2D (PxP) or 3D (PxPxT) adjacency/connectivity matrix

    diag : int, optional
        set diagonal to this value
        default is 0

    copy : bool, optional
        if True, a copy of G is returned, otherwise G is modified in place
        default is True

    Returns
    -------
    G : PxP np.ndarray
        processed adjacency/connectivity matrix
    '''
    if copy:
        G = G.copy()

    def _postproc(A):
        """Postprocessing for a single 2D matrix."""
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.allclose(A, A.T):
            raise ValueError("Error: Matrix is not symmetrical")
        np.fill_diagonal(A, diag)
        return np.round(A, decimals=6)

    if G.ndim == 2:
        return _postproc(G)
    elif G.ndim == 3:
        return np.stack([_postproc(G[:, :, t]) for t in range(G.shape[2])], axis=2)
    else:
        raise ValueError("G must be a 2D or 3D array.")


"""
SECTION: Graph analysis functions
 - Comet specfic graph analysis functions
 - Functions which rely on path length calculations are compiled
   to machine code using numba for improved performance
"""
def avg_shortest_path(G: np.ndarray,
                      include_diagonal: bool = False,
                      include_infinite: bool = False) -> np.ndarray:
    '''
    Compute the average shortest path length for a 2D or 3D connectivity matrix.

    For binary matrices, the connection length matrix is identical to the connectivity matrix.
    For weighted matrices, the connection length matrix can be obtained via `invert()`.

    Parameters
    ----------
    G : np.ndarray
        2D (NxN) or 3D (NxN x T) undirected connectivity matrix (binary or weighted).

    include_diagonal : bool, optional
        If False, diagonal values are ignored when computing the average.
        Default is False.

    include_infinite : bool, optional
        If False, infinite path lengths are ignored when computing the average.
        Default is False.

    Returns
    -------
    np.ndarray
        Average shortest path length(s). If 2D, a single float is returned; if 3D, a 1D array
        of average path lengths is returned, one for each time slice.
    '''
    def _avg_shortest_path(D):
        """Average shortest path for a single 2D matrix."""
        if np.isinf(D).any():
            warnings.warn("The graph is not fully connected; infinite path lengths were set to NaN.")
        if not include_diagonal:
            np.fill_diagonal(D, np.nan)
        if not include_infinite:
            D[np.isinf(D)] = np.nan
        Dv = D[~np.isnan(D)]
        return np.mean(Dv) if len(Dv) > 0 else np.nan

    is_binary = np.all(np.logical_or(np.isclose(G, 0), np.isclose(G, 1)))

    if G.ndim == 2:
        D = distance_bin(G) if is_binary else distance_wei(G)
        return _avg_shortest_path(D)
    
    elif G.ndim == 3:
        avg_paths = []
        for t in range(G.shape[2]):
            D = distance_bin(G[:, :, t]) if is_binary else distance_wei(G[:, :, t])
            avg_paths.append(_avg_shortest_path(D))
        return np.array(avg_paths)
    
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

def transitivity_und(G: np.ndarray) -> np.ndarray:
    '''
    Compute transitivity for undirected networks (binary and weighted),
    adapted from the bctpy implementation: https://github.com/aestrivex/bctpy

    Transitivity is the ratio of triangles to triplets in the network and is
    a classical version of the clustering coefficient.

    Parameters
    ----------
    G : np.ndarray
        2D (NxN) or 3D (NxN x T) undirected connectivity matrix (binary or weighted).

    Returns
    -------
    np.ndarray
        Transitivity scalar(s). If 2D, a single float is returned; if 3D, a 1D array
        of transitivity values is returned, one for each time slice.
    '''

    def _transitivity(A):
        """Transitivity for a single 2D matrix."""
        is_binary = np.all(np.logical_or(np.isclose(A, 0), np.isclose(A, 1)))

        if is_binary:
            tri3 = np.trace(A @ (A @ A))
            tri2 = np.sum(A @ A) - np.trace(A @ A)
            return tri3 / tri2 if tri2 > 0 else 0
        else:
            K = np.sum(A > 0, axis=1)
            ws = np.cbrt(A)
            cyc3 = np.diag(ws @ (ws @ ws))
            denominator = np.sum(K * (K - 1))
            return np.sum(cyc3) / denominator if denominator > 0 else 0

    if G.ndim == 2:
        return _transitivity(G)
    elif G.ndim == 3:
        return np.array([_transitivity(G[:, :, t]) for t in range(G.shape[2])])
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

def avg_clustering_onella(G: np.ndarray) -> np.ndarray:
    '''
    Compute the average clustering coefficient as described by Onnela et al. (2005).

    Parameters
    ----------
    G : np.ndarray
        2D (NxN) or 3D (NxN x T) undirected connectivity matrix (binary or weighted).

    Returns
    -------
    np.ndarray
        Average clustering coefficient. If 2D, a single float is returned; 
        if 3D, a 1D array of clustering coefficients is returned, one for each time slice.

    References
    ----------
    Onnela, J. P., Saramäki, J., Kertész, J., & Kaski, K. (2005). Intensity and
    coherence of motifs in weighted complex networks. Physical Review E, 71(6), 065103.
    DOI: https://doi.org/10.1103/PhysRevE.71.065103
    '''

    def _clustering(A):
        """Average clustering for a single 2D matrix."""
        K = np.count_nonzero(A, axis=1)
        if np.max(A) > 0:
            G2 = A / np.max(A)  # Normalise
        else:
            G2 = A
        cyc3 = np.diagonal(np.linalg.matrix_power(G2 ** (1/3), 3))
        
        # Prevent division by zero
        valid = K > 1
        C = np.zeros_like(K, dtype=float)
        C[valid] = cyc3[valid] / (K[valid] * (K[valid] - 1))

        return np.nanmean(C)  # Average over non-zero entries

    if G.ndim == 2:
        return _clustering(G)
    elif G.ndim == 3:
        return np.array([_clustering(G[:, :, t]) for t in range(G.shape[2])])
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

def efficiency(G: np.ndarray,
               local: bool = False) -> np.ndarray:
    '''
    Efficiency for binary and weighted networks (global and local)

    Optimized version of the bctpy implelementation by Roan LaPlante (https://github.com/aestrivex/bctpy)

    Global efficiency is the average of inverse shortest path length, and is inversely related to the characteristic path length.
    Local efficiency is the global efficiency computed on the neighborhood of the node, and is related to the clustering coefficient.

    Parameters
    ----------
    G : np.ndarray
        2D (NxN) or 3D (NxNxT) undirected connectivity matrix (binary or weighted).

    local : bool, optional
        if True, local efficiency is computed. Default is False (global efficiency)

    Returns
    -------
    E (global) : float
        global efficiency, if local is False

    E (local) : Nx1 np.ndarray
        local efficiency, if local is True


    References
    ----------
    Latora, V., & Marchiori, M. (2001). Efficient behavior of small-world networks.
    Physical review letters, 87(19), 198701. DOI: https://doi.org/10.1103/PhysRevLett.87.198701

    Onnela, J. P., Saramäki, J., Kertész, J., & Kaski, K. (2004). Intensity and coherence of motifs in weighted c
    omplex networks. Physical Review E, 71(6), 065103. DOI: https://doi.org/10.1103/PhysRevE.71.065103

    Fagiolo, G. (2007). Clustering in complex directed networks. Physical Review E, 76(2), 026107.
    DOI: https://doi.org/10.1103/PhysRevE.76.026107

    Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: uses and interpretations.
    Neuroimage, 52(3), 1059-1069. DOI: https://doi.org/10.1016/j.neuroimage.2009.10.003

    Wang, Y., Ghumare, E., Vandenberghe, R., & Dupont, P. (2017). Comparison of different generalizations of clustering coefficient
    and local efficiency for weighted undirected graphs. Neural computation, 29(2), 313-331. DOI: https://doi.org/10.1162/NECO_a_00914
    '''

    def _efficiency(G2D, local):
        """Efficiency for a single 2D matrix."""
        n = len(G2D)
        is_binary = np.all(np.logical_or(np.isclose(G2D, 0), np.isclose(G2D, 1)))

        # Efficiency for binary networks
        if is_binary:
            if local:
                E = np.zeros((n,))
                for u in range(n):
                    V, = np.where(np.logical_or(G2D[u, :], G2D[:, u].T))
                    e = distance_bin(G2D[np.ix_(V, V)], inv=True)
                    se = e + e.T
                    sa = G2D[u, V] + G2D[V, u].T
                    numer = np.sum(np.outer(sa.T, sa) * se) / 2
                    if numer != 0:
                        denom = np.sum(sa) ** 2 - np.sum(sa * sa)
                        E[u] = numer / denom
            else:
                e = distance_bin(G2D, inv=True)
                E = np.sum(e) / (n * n - n)

        # Efficiency for weighted networks
        else:
            Gl = invert(G2D, copy=True)
            A = (G2D != 0).astype(int)
            if local:
                E = np.zeros((n,))
                for u in range(n):
                    V, = np.where(np.logical_or(G2D[u, :], G2D[:, u].T))
                    sw = np.cbrt(G2D[u, V]) + np.cbrt(G2D[V, u].T)
                    e = distance_wei(np.cbrt(Gl)[np.ix_(V, V)], inv=True)
                    se = e + e.T
                    numer = np.sum(np.outer(sw.T, sw) * se) / 2
                    if numer != 0:
                        sa = A[u, V] + A[V, u].T
                        denom = np.sum(sa) ** 2 - np.sum(sa * sa)
                        E[u] = numer / denom
            else:
                e = distance_wei(Gl, inv=True)
                E = np.sum(e) / (n * n - n)

        return E

    if G.ndim == 2:
        return _efficiency(G, local)
    elif G.ndim == 3:
        if local:
            return np.stack([_efficiency(G[:, :, t], local) for t in range(G.shape[2])], axis=1)
        else:
            return np.array([_efficiency(G[:, :, t], local) for t in range(G.shape[2])])
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

def small_world_sigma(G: np.ndarray,
                      nrand: int = 10) -> np.ndarray:
    '''
    Compute the small-worldness sigma for undirected networks (binary or weighted).

    Small-worldness sigma is calculated as the ratio of the clustering coefficient
    and the characteristic path length of the real network to the average clustering 
    coefficient and characteristic path length of the random networks.

    Parameters
    ----------
    G : np.ndarray
        2D (PxP) or 3D (PxPxT) undirected adjacency/connectivity matrix.

    nrand : int, optional
        Number of random networks to generate (and average over).
        Default is 10.

    Returns
    -------
    float or np.ndarray
        Small-worldness sigma. If 2D, a single float is returned.
        If 3D, a 1D array of sigma values is returned, one for each time slice.

    Notes
    -----
    This implementation of small worldness relies on matrix operations and is 
    *drastically* faster than the Networkx implementation. However, it uses a 
    different approach for rewiring edges, so the results may differ.
    '''

    def _small_world_sigma(G2D, nrand):
        """Small-worldness sigma for a single 2D matrix."""
        randMetrics = {"C": [], "L": []}
        
        for _ in range(nrand):
            Gr = randomise(G2D)
            randMetrics["C"].append(transitivity_und(Gr))
            randMetrics["L"].append(avg_shortest_path(Gr))

        C = transitivity_und(G2D)
        L = avg_shortest_path(G2D)

        Cr = np.mean(randMetrics["C"])
        Lr = np.mean(randMetrics["L"])

        return (C / Cr) / (L / Lr)

    if G.ndim == 2:
        return _small_world_sigma(G, nrand)
    elif G.ndim == 3:
        return np.array([_small_world_sigma(G[:, :, t], nrand) for t in range(G.shape[2])])
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

def small_world_propensity(G: np.ndarray) -> np.ndarray:
    '''
    Small world propensity calculation for undirected and symmetric networks.
    Clustering is calculated using the approach of Onnela et al. (2005).

    Based on the MATLAB and Python implementations by Eric Bridgeford, Sarah F. Muldoon, and Ryan Daniels:
    https://kk1995.github.io/BauerLab/BauerLab/MATLAB/lib/+mouse/+graph/smallWorldPropensity.html
    https://github.com/rkdan/small_world_propensity

    Parameters
    ----------
    F : PxP np.ndarray
        undirected and symmetric adjacency/connectivity matrix

    Returns
    -------
    SWP : float
        small world propensity of the matrix
    delta_C : float
        fractional deviation from the expected culstering coefficient of a random network
    delta_L : float
        fractional deviation from the expected path length of a random network
    alpha : float
        angle between delta_C and delta_L
    delta : float
        scaled version of alpha in the range of [-1,1]

    References
    ----------
    Muldoon, S., Bridgeford, E. & Bassett, D. Small-World Propensity and Weighted Brain Networks.
    Sci Rep 6, 22057 (2016). DOI: https://doi.org/10.1038/srep22057

    Onnela, J. P., Saramäki, J., Kertész, J., & Kaski, K. (2005). Intensity and
    coherence of motifs in weighted complex networks. Physical Review E, 71(6), 065103.
    DOI: https://doi.org/10.1103/PhysRevE.71.065103
    '''
    def _small_world_propensity(G2D):
        if not np.allclose(G2D, G2D.T):
            raise ValueError("Error: Matrix is not symmetrical")

        n = G2D.shape[0]  # Number of nodes

        # Compute the average degree of the unweighted network (approximate radius)
        num_connections = np.count_nonzero(G2D)
        avg_deg_unw = num_connections / n
        avg_rad_unw = avg_deg_unw / 2
        avg_rad_eff = np.ceil(avg_rad_unw).astype(int)
        # Compute the regular and random matrix for the network
        G2D_reg = regular_matrix(G2D, avg_rad_eff)
        G2D_rand = randomise(G2D)

        # Path length calculations for the network, ignore divide by zero warnings
        with np.errstate(divide='ignore'):
            G2D_reg_inv = np.divide(1.0, G2D_reg)
            G2D_rand_inv = np.divide(1.0, G2D_rand)
            G2D_inv = np.divide(1.0, G2D)

        reg_path = avg_shortest_path(G2D_reg_inv)
        rand_path = avg_shortest_path(G2D_rand_inv)
        net_path = avg_shortest_path(G2D_inv)

        # Compute the normalized difference in path lengths
        A = max(net_path - rand_path, 0)  # Ensure A is non-negative
        diff_path = A / (reg_path - rand_path) if (reg_path != float('inf') and rand_path != float('inf') and net_path != float('inf')) else 1
        diff_path = min(diff_path, 1)  # Ensure diff_path does not exceed 1

        # Compute all clustering calculations for the network
        reg_clus = avg_clustering_onella(G2D_reg)
        rand_clus = avg_clustering_onella(G2D_rand)
        net_clus = avg_clustering_onella(G2D)

        B = max(reg_clus - net_clus, 0)
        diff_clus = B / (reg_clus - rand_clus) if (not np.isnan(reg_clus) and not np.isnan(rand_clus) and not np.isnan(net_clus)) else 1
        diff_clus = min(diff_clus, 1)

        # Assuming diff_path is calculated elsewhere
        SWP = 1 - (np.sqrt(diff_clus**2 + diff_path**2) / np.sqrt(2))
        delta_C = diff_clus
        delta_L = diff_path

        # We ignore divide by zero warnings here as arctan(inf) is pi/2
        with np.errstate(divide='ignore'):
            alpha = np.arctan(delta_L / delta_C)
            delta = (4 * alpha / np.pi) - 1

        return SWP, delta_C, delta_L, alpha, delta

    if G.ndim == 2:
        return _small_world_propensity(G)
    elif G.ndim == 3:
        results = [np.array(x) for x in zip(*[_small_world_propensity(G[:, :, t]) for t in range(G.shape[2])])]
        return tuple(results)
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

def matching_ind_und(G: np.ndarray) -> np.ndarray:
    '''
    Matching index for undirected networks

    Based on the MATLAB implementation by Stuart Oldham:
    https://github.com/StuartJO/FasterMatchingIndex

    Matching index is a measure of similarity between two nodes' connectivity profiles
    (excluding their mutual connection, should it exist). Weighted matrices will be binarised.

    Parameters
    ----------
    G : np.ndarray
        2D (PxP) or 3D (PxP x T) undirected adjacency/connectivity matrix.

    Returns
    -------
    M : PxP np.ndarray
        matching index matrix

    References
    ----------
    Oldham, S., Fulcher, B. D., Aquino, K., Arnatkevičiūtė, A., Paquola, C.,
    Shishegar, R., & Fornito, A. (2022). Modeling spatial, developmental,
    physiological, and topological constraints on human brain connectivity.
    Science advances, 8(22), eabm6127. DOI: https://doi.org/10.1126/sciadv.abm6127

    Betzel, R. F., Avena-Koenigsberger, A., Goñi, J., He, Y., De Reus, M. A.,
    Griffa, A., ... & Sporns, O. (2016).
    Generative models of the human connectome. Neuroimage, 124, 1054-1064.
    DOI: https://doi.org/10.1016/j.neuroimage.2015.09.041

    Notes
    -----
    Important note: As of Jan 2024 there is a bug in the bctpy version of this function
    (ncon2 = c1 * CIJ should instead be ncon2 = CIJ * use)

    This bug is irrelevant here due to the opimized implementation.
    '''
    if G.ndim == 2:
            return _matching_ind(G)
    elif G.ndim == 3:
        return np.stack([_matching_ind(G[:, :, t]) for t in range(G.shape[2])], axis=2)
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

@njit(fastmath=True, cache=True)
def distance_wei(G: np.ndarray, inv: bool = False) -> np.ndarray:
    '''
    (Inverse) distance matrix for weighted networks with significantly
    improved performance due to numba compilation.

    Parameters
    ----------
    G : PxP np.ndarray
        undireted weighted adjacency/connectivity matrix

    inv : bool, optional
        if True, the element wise inverse of the distance matrux is returned. Default is True

    Returns
    -------
    D : PxP np.ndarray
        (inverse) distance matrix

    Notes
    -----
    Algorithm: Modified Dijkstra's algorithm
    '''

    n = len(G)
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)

    for u in range(n):
        # distance permanence (true is temporary)
        S = np.ones((n,), dtype=np.bool_)
        G1 = G.copy()
        V = np.array([u], dtype=np.int64)
        while True:
            S[V] = 0  # distance u->V is now permanent
            G1[:, V] = 0  # no in-edges as already shortest

            for v in V:
                W = np.where(G1[v, :])[0]  # neighbors of smallest nodes
                max_len = n
                td = np.empty((2, max_len))
                len_W = len(W)
                td[0, :len_W] = D[u, W]
                td[1, :len_W] = D[u, v] + G1[v, W]
                for idx in range(len_W):
                    D[u, W[idx]] = min(td[0, idx], td[1, idx])

            if D[u, S].size == 0:  # all nodes reached
                break
            minD = np.min(D[u, S])
            if np.isinf(minD):  # some nodes cannot be reached
                break
            V = np.where(D[u, :] == minD)[0]

    if inv:
        np.fill_diagonal(D, 1)
        D = 1 / D
        np.fill_diagonal(D, 0)

    return D

@njit(fastmath=True, cache=True)
def distance_bin(G: np.ndarray, inv: bool = False) -> np.ndarray:
    '''
    Distance matrix calculation for binary networks with significantly
    improved performance due to numba compilation.

    Parameters
    ----------
    G : PxP np.ndarray
        undireted weighted adjacency/connectivity matrix

    Returns
    -------
    D : PxP np.ndarray
        (inverse) distance matrix

    Notes
    -----
    Algorithm: Matrix multiplication to find paths, faster than original Dijkstra's algorithm
    '''

    D = np.eye(len(G))
    n = 1
    nPATH = G.copy()
    L = (nPATH != 0)

    while np.any(L):
        D += n * L
        n += 1
        nPATH = np.dot(nPATH, G)
        L = (nPATH != 0) * (D == 0)

    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if not D[i, j]:
                D[i, j] = np.inf

    if inv:
        D = 1 / D

    np.fill_diagonal(D, 0)

    return D

@njit(fastmath=True, cache=True)
def _matching_ind(G2D: np.ndarray) -> np.ndarray:
    "Compute the matching index for a single 2D adjacency matrix."
    G2D = (G2D > 0).astype(np.float64) # binarise the adjacency matrix
    n = G2D.shape[0]

    nei = np.dot(G2D, G2D)
    deg = np.sum(G2D, axis=1)
    degsum = deg[:, np.newaxis] + deg

    denominator = np.where((degsum <= 2) & (nei != 1), 1.0, degsum - 2 * G2D)
    M = np.where(denominator != 0, (nei * 2) / denominator, 0.0)

    for i in range(n):
        M[i, i] = 0.0

    return M


"""
SECTION: bctpy wrapper functions
 - The wrapper functions implement type hinting which is required for the GUI
 - For the scripting API, users can also directly use the bctpy functions
"""
def backbone_wu(G: np.ndarray,
                avgdeg: int = 0,
                verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    '''
    Wrapper for `bct.backbone_wu()` from the Brain Connectivity Toolbox.
    
    The network backbone contains the dominant connections in the network
    and may be used to aid network visualization. This function computes
    the backbone of a given weighted and undirected connectivity matrix G,
    using a minimum-spanning-tree based algorithm.

    Parameters
    ----------
    G : np.ndarray
        2D (NxN) or 3D (NxN x T) weighted undirected connectivity matrix.
    avgdeg : int, optional
        Desired average degree of the backbone. Default is 0.
    verbose : bool, optional
        If True, prints out edges while building the spanning tree.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Gtree : NxN (or NxN x T) np.ndarray
            Connection matrix of the minimum spanning tree of G.
        Gclus : NxN (or NxN x T) np.ndarray
            Connection matrix of the minimum spanning tree plus strongest
            connections up to the specified average degree.
    '''

    if G.ndim == 2:
        # Single timepoint, just call bct
        return bct.backbone_wu(G, avgdeg, verbose)

    elif G.ndim == 3:
        # Multiple timepoints, process each slice independently
        n, _, t = G.shape
        Gtree_stack = np.zeros((n, n, t))
        Gclus_stack = np.zeros((n, n, t))

        for i in range(t):
            if verbose:
                print(f"Processing time slice {i + 1} of {t}")
            Gtree_stack[:, :, i], Gclus_stack[:, :, i] = bct.backbone_wu(G[:, :, i], avgdeg, verbose)

        return Gtree_stack, Gclus_stack
    
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

def betweenness(G: np.ndarray) -> np.ndarray:
    '''
    This is a wrapper function for the `betweenness_bin()` and `betweenness_wei()` functions
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    Node betweenness centrality is the fraction of all shortest paths in
    the network that contain a given node. Nodes with high values of
    betweenness centrality participate in a large number of shortest paths.

    Parameters
    ----------
    G : np.ndarray
        2D (NxN) or 3D (NxN x T) binary/weighted directed/undirected connectivity matrix.

    Returns
    -------
    np.ndarray
        Node betweenness centrality vector. If G is 2D, the output is Nx1.
        If G is 3D, the output is NxT.

    Notes
    -----
    - For binary graphs, betweenness is normalized as BC/[(N-1)(N-2)].
    - For weighted graphs, it is based on connection lengths and similarly normalized.
    - If the input is 3D, each time slice is computed independently.
    '''
    is_binary = np.all(np.logical_or(np.isclose(G, 0), np.isclose(G, 1)))

    if G.ndim == 2:
        # Single timepoint, just call bct
        return bct.betweenness_bin(G) if is_binary else bct.betweenness_wei(G)

    elif G.ndim == 3:
        # Multiple timepoints, process each slice independently
        n, _, t = G.shape
        BC_stack = np.zeros((n, t))

        for i in range(t):
            BC_stack[:, i] = bct.betweenness_bin(G[:, :, i]) if is_binary else bct.betweenness_wei(G[:, :, i])

        return BC_stack
    
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

def clustering_coef(G: np.ndarray) -> np.ndarray:
    '''
    This is a wrapper function for the `clustering_coef_*()` functions
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    The binary clustering coefficient is the fraction of triangles around a node
    (equiv. the fraction of nodes neighbors that are neighbors of each other).

    The weighted clustering coefficient is the average "intensity" of triangles
    around a node.

    Parameters
    ----------
    G : np.ndarray
        2D (NxN) or 3D (NxN x T) weighted undirected connectivity matrix.

    Returns
    -------
    np.ndarray
        Clustering coefficient vector. If G is 2D, the output is Nx1.
        If G is 3D, the output is NxT.
    '''
    is_binary = np.all(np.logical_or(np.isclose(G, 0), np.isclose(G, 1)))

    if G.ndim == 2:
        # Single timepoint, just call bct
        return bct.clustering_coef_bu(G) if is_binary else bct.clustering_coef_wu(G)

    elif G.ndim == 3:
        # Multiple timepoints, process each slice independently
        n, _, t = G.shape
        C_stack = np.zeros((n, t))

        for i in range(t):
            if is_binary:
                C_stack[:, i] = bct.clustering_coef_bu(G[:, :, i])
            else:
                C_stack[:, i] = bct.clustering_coef_wu(G[:, :, i])

        return C_stack
    
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

def degrees_und(G: np.ndarray) -> np.ndarray:
    '''
    This is a wrapper function for the `degrees_und()` function
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    Node degree is the number of links connected to the node.

    Parameters
    ----------
    G : np.ndarray
        2D (NxN) or 3D (NxN x T) undirected binary/weighted connectivity matrix.

    Returns
    -------
    np.ndarray
        Node degree vector. If G is 2D, the output is Nx1.
        If G is 3D, the output is NxT.

    Notes
    -----
    - If the graph is 3D (NxN x T), each slice is processed independently.
    - Weight information is discarded.
    '''

    if G.ndim == 2:
        # Single timepoint, just call bct
        return bct.degrees_und(G)

    elif G.ndim == 3:
        # Multiple timepoints, process each slice independently
        n, _, t = G.shape
        deg_stack = np.zeros((n, t))

        for i in range(t):
            deg_stack[:, i] = bct.degrees_und(G[:, :, i])

        return deg_stack
    
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

def density_und(G: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    This is a wrapper function for the `density_und()` function
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    Density is the fraction of present connections to possible connections.

    Parameters
    ----------
    G : np.ndarray
        2D (NxN) or 3D (NxN x T) undirected weighted/binary connectivity matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - `kden`: Density of the graph. If G is 2D, returns a float, if G is 3D, returns a 1D array.
        - `N`: Number of vertices. If G is 2D, returns an integer, if G is 3D, returns a 1D array.
        - `k`: Number of edges. If G is 2D, returns an integer, if G is 3D, returns a 1D array.

    Notes
    -----
    - Assumes G is undirected and has no self-connections.
    - Weight information is discarded.
    - If G is 3D (NxN x T), each slice is processed independently.
    '''
    if G.ndim == 2:
        # Single timepoint, just call bct
        return bct.density_und(G)

    elif G.ndim == 3:
        # Multiple timepoints, process each slice independently
        t = G.shape[2]
        kden_stack = np.zeros(t)
        N_stack = np.zeros(t, dtype=int)
        k_stack = np.zeros(t, dtype=int)

        for i in range(t):
            kden_stack[i], N_stack[i], k_stack[i] = bct.density_und(G[:, :, i])

        return kden_stack, N_stack, k_stack
    
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

def eigenvector_centrality_und(G: np.ndarray) -> np.ndarray:
    '''
    This is a wrapper function for the `eigenvector_centrality_und()` function
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    Eigenvector centrality is a self-referential measure of centrality:
    nodes have high eigenvector centrality if they connect to other nodes
    that have high eigenvector centrality. The eigenvector centrality of
    node i is equivalent to the ith element in the eigenvector
    corresponding to the largest eigenvalue of the adjacency matrix.

    Parameters
    ----------
    G : np.ndarray
        2D (NxN) or 3D (NxN x T) binary/weighted undirected adjacency matrix.

    Returns
    ----------
    np.ndarray
        Eigenvector associated with the largest eigenvalue of the matrix.
        If G is 2D, the output is Nx1.
        If G is 3D, the output is NxT.
    '''

    if G.ndim == 2:
        # Single timepoint, just call bct
        return bct.eigenvector_centrality_und(G)

    elif G.ndim == 3:
        # Multiple timepoints, process each slice independently
        n, _, t = G.shape
        eig_centrality_stack = np.zeros((n, t))

        for i in range(t):
            eig_centrality_stack[:, i] = bct.eigenvector_centrality_und(G[:, :, i])

        return eig_centrality_stack
    
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

def pagerank_centrality(G: np.ndarray,
                        d: float = 0.85) -> np.ndarray:
    '''
    This is a wrapper function for the `pagerank_centrality()` function
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    The PageRank centrality is a variant of eigenvector centrality. This
    function computes the PageRank centrality of each vertex in a graph.

    Parameters
    ----------
    G : np.ndarray
        2D (NxN) or 3D (NxN x T) adjacency matrix.

    d : float, optional
        Damping factor (default is 0.85). Specifies the fraction of the time
        that a random walker will transition to one of its current state's neighbors.

    Returns
    -------
    np.ndarray
        Vectors of page rankings.
        If G is 2D, the output is Nx1.
        If G is 3D, the output is NxT.

    Notes
    -----
    - This function does not currently support the `falff` parameter from bctpy.
    - If G is 3D, each time slice is processed independently.
    '''

    if G.ndim == 2:
        # Single timepoint, just call bct
        return bct.pagerank_centrality(G, d, None)

    elif G.ndim == 3:
        # Multiple timepoints, process each slice independently
        n, _, t = G.shape
        pagerank_stack = np.zeros((n, t))

        for i in range(t):
            pagerank_stack[:, i] = bct.pagerank_centrality(G[:, :, i], d, None)

        return pagerank_stack
    
    else:
        raise ValueError("Input must be a 2D or 3D matrix.")
    
def participation_coef(G: np.ndarray,
                       ci: Literal["louvain"] = "louvain",
                       degree: Literal["undirected", "in", "out"] = "undirected") -> np.ndarray:
    '''
    This is a wrapper function for the `participation_coef` and `participation_coef_sparse`
    functions of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    The participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.

    Parameters
    ----------
    G : np.ndarray or scipy.sparse.csr_matrix
        2D (NxN) or 3D (NxN x T) binary/weighted directed/undirected connectivity matrix.

    ci : str or array-like
        Community detection method. If "louvain", uses bct.community_louvain.
        Otherwise, `ci` itself is treated as an array of community labels.

    degree : str, optional
        Flag to describe nature of graph:
        - 'undirected': For undirected graphs
        - 'in': Uses the in-degree
        - 'out': Uses the out-degree
        Default is "undirected".

    Returns
    -------
    np.ndarray
        Participation coefficient vector. 
        If G is 2D, the output is Nx1.
        If G is 3D, the output is NxT.
    '''

    def _compute_communities(G2D):
        if isinstance(ci, str) and ci.lower() == "louvain":
            communities, _ = bct.community_louvain(G2D)
        else:
            communities = np.asarray(ci)
        return communities

    if G.ndim == 2:
        # Single timepoint, compute communities and participation coefficient
        communities = _compute_communities(G)
        if isinstance(G, scipy.sparse.csr_matrix):
            return bct.participation_coef_sparse(G, communities, degree)
        else:
            return bct.participation_coef(G, communities, degree)

    elif G.ndim == 3:
        # Multiple timepoints, process each slice independently
        n, _, t = G.shape
        P_stack = np.zeros((n, t))

        for i in range(t):
            communities = _compute_communities(G[:, :, i])
            if isinstance(G[:, :, i], scipy.sparse.csr_matrix):
                P_stack[:, i] = bct.participation_coef_sparse(G[:, :, i], communities, degree)
            else:
                P_stack[:, i] = bct.participation_coef(G[:, :, i], communities, degree)

        return P_stack

    else:
        raise ValueError("Input must be a 2D or 3D matrix.")

def participation_coef_sign(G: np.ndarray,
                            ci: Literal["louvain"] = "louvain") -> tuple[np.ndarray, np.ndarray]:
    '''
    This is a wrapper function for the `participation_coef_sign()` function
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    The participation coefficient is a measure of diversity of intermodular
    connections of individual nodes, considering both positive and negative weights.

    Parameters
    ----------
    G : np.ndarray
        2D (NxN) or 3D (NxN x T) undirected connectivity matrix with positive and negative weights.

    ci : str or array-like
        Community detection method. If "louvain", uses bct.community_louvain.
        Otherwise, `ci` itself is treated as an array of community labels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - `Ppos`: Participation coefficient from positive weights.
        - `Pneg`: Participation coefficient from negative weights.
        
        If G is 2D, each is Nx1. If G is 3D, each is NxT.
    '''

    def _compute_communities(G2D):
        if isinstance(ci, str) and ci.lower() == "louvain":
            communities, _ = bct.community_louvain(G2D)
        else:
            communities = np.asarray(ci)
        return communities

    if G.ndim == 2:
        # Single timepoint, compute communities and participation coefficients
        communities = _compute_communities(G)
        Ppos, Pneg = bct.participation_coef_sign(G, communities)
        return Ppos, Pneg

    elif G.ndim == 3:
        # Multiple timepoints, process each slice independently
        n, _, t = G.shape
        Ppos_stack = np.zeros((n, t))
        Pneg_stack = np.zeros((n, t))

        for i in range(t):
            communities = _compute_communities(G[:, :, i])
            Ppos, Pneg = bct.participation_coef_sign(G[:, :, i], communities)
            Ppos_stack[:, i] = Ppos
            Pneg_stack[:, i] = Pneg

        return Ppos_stack, Pneg_stack

    else:
        raise ValueError("Input must be a 2D or 3D matrix.")
