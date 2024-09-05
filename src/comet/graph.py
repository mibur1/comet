import bct
import numpy as np
import scipy.sparse
from numba import jit
from typing import Literal

"""
SECTION: Graph processing functions
 - General functions to preprocess connectivity/adjacency matrices
   before graph analysis.
"""
def handle_negative_weights(W: np.ndarray,
                            type: Literal["absolute", "discard"] = "absolute",
                            copy: bool = True) -> np.ndarray:
    '''
    Handle negative weights in a connectivity/adjacency matrix

    Connectivity methods can produce negative estimates, which can be handled in different ways before graph analysis.

    Parameters
    ----------
    W : PxP np.ndarray
        adjacency/connectivity matrix

    type : string, optional
        type of handling, can be *absolute* or *discard*
        default is *absolute*

    copy : bool, optional
        if True, a copy of W is returned, otherwise W is modified in place
        default is True

    Returns
    -------
    W : PxP np.ndarray
        adjacency/connectivity matrix with only positive weights
    '''

    if copy:
        W = W.copy()

    if type == "absolute":
        W = np.abs(W)
    elif type == "discard":
        W[W < 0] = 0
    else:
        raise NotImplementedError("Options are: *absolute* or *discard*")
    return W

def threshold(W: np.ndarray,
              type: Literal["density", "absolute"] = "density",
              threshold: float = None,
              density: float = None,
              copy: bool = True) -> np.ndarray:
    '''
    Thresholding of connectivity/adjacency matrix

    Performs absolute or density-based thresholding

    Parameters
    ----------
    W : PxP np.ndarray
        adjacency/connectivity matrix

    type : string, optional
        type of thresholding, can be *absolute* or *density*
        default is *absolute*

    threshold : float, optional
        threshold value for absolute thresholding
        default is None

    density : float, optional
        density value for density-based thresholding, has to be between 0 and 1 (keep x% of strongest connections)
        default is None

    copy : bool, optional
        if True, a copy of W is returned, otherwise W is modified in place
        default is True

    Returns
    -------
    W : PxP np.ndarray
        thresholded adjacency/connectivity matrix

    Notes
    -----
    The implemented for density based thresholding always keeps the exact same number of connections. If multiple edges have the same weight,
    the included edges are chosen "randomly" (based on their order in the sorted indices). This is identical to the behaviour in the BCT implementation.
    '''

    if copy:
        W = W.copy()

    if type == "absolute":
        W[W < threshold] = 0
    elif type == "density":
        if not density >=0 and density <= 1:
            raise ValueError("Error: Density must be between 0 and 1")
        if not np.allclose(W, W.T):
            raise ValueError("Error: Matrix is not symmetrical")

        W[np.tril_indices(len(W))] = 0 # set lower triangle to zero
        triu_indices = np.triu_indices_from(W, k=1) # get upper triangle indices
        sorted_indices = np.argsort(W[triu_indices])[::-1] # sort upper triangle by indices
        cutoff_idx = int(np.round((len(sorted_indices) * density) + 1e-10)) # find cutoff index, add small constant to round .5 to 1
        keep_mask = np.zeros_like(W, dtype=bool)
        keep_mask[triu_indices[0][sorted_indices[:cutoff_idx]], triu_indices[1][sorted_indices[:cutoff_idx]]] = True # set values larger than cutoff to True
        W[~keep_mask] = 0
        W = W + W.T # restore symmetry
    else:
        raise NotImplementedError("Thresholding must be of type *absolute* or *density*")
    return W

def binarise(W: np.ndarray,
             copy: bool = True) -> np.ndarray:
    '''
    Binarise connectivity/adjacency matrix

    Parameters
    ----------
    W : PxP np.ndarray
        adjacency/connectivity matrix

    copy : bool, optional
        if True, a copy of W is returned, otherwise W is modified in place
        default is True

    Returns
    -------
    W : PxP np.ndarray
        binarised adjacency/connectivity matrix
    '''

    if copy:
        W = W.copy()

    W[W != 0] = 1
    return W

def normalise(W: np.ndarray,
              copy: bool = True) -> np.ndarray:
    '''
    Normalise connectivity/adjacency matrix

    Parameters
    ----------
    W : PxP np.ndarray
        adjacency/connectivity matrix

    copy : bool, optional
        if True, a copy of W is returned, otherwise W is modified in place
        default is True

    Returns
    -------
    W : PxP np.ndarray
        normalised adjacency/connectivity matrix
    '''

    if copy:
        W = W.copy()

    if not np.max(np.abs(W)) > 0:
        raise ValueError("Error: Matrix contains only zeros")

    W /= np.max(np.abs(W))
    return W

def invert(W: np.ndarray,
           copy: bool = True) -> np.ndarray:
    '''
    Invert connectivity/adjacency matrix

    Element wise inversion W such that each value W[i,j] will be 1 / W[i,j] (internode strengths internode distances)

    Parameters
    ----------
    W : PxP np.ndarray
        adjacency/connectivity matrix

    copy : bool, optional
        if True, a copy of W is returned, otherwise W is modified in place
        default is True

    Returns
    -------
    W : PxP np.ndarray
        element wise inverted adjacency/connectivity matrix
    '''

    if copy:
        W = W.copy()

    W_safe = np.where(W == 0, np.inf, W)
    W = 1 / W_safe
    return W

def logtransform(W: np.ndarray,
                 epsilon: float = 1e-10,
                 copy: bool = True) -> np.ndarray:
    '''
    Log transform of connectivity/adjacency matrix

    Element wise log transform of W such that each value W[i,j] will be -log(W[i,j]

    Parameters
    ----------
    W : PxP np.ndarray
        adjacency/connectivity matrix

    epsilon : float, optional
        clipping value for numeric stability,
        default is 1e-10

    copy : bool, optional
        if True, a copy of W is returned, otherwise W is modified in place
        default is True

    Returns
    -------
    W : PxP np.ndarray
        element wise log transformed adjacency/connectivity matrix
    '''

    if copy:
        W = W.copy()

    if np.logical_or(W > 1, W <= 0).any():
        raise ValueError("Connections must be between (0,1] to use logtransform")
    W_safe = np.clip(W, a_min=epsilon, a_max=None) # clip very small values for numeric stability
    W = -np.log(W_safe)
    return W

def symmetrise(W: np.ndarray,
               copy: bool = True) -> np.ndarray:
    '''
    Symmetrise connectivity/adjacency matrix

    Symmetrise W such that each value W[i,j] will be W[j,i].

    Parameters
    ----------
    W : PxP np.ndarray
        adjacency/connectivity matrix

    copy : bool, optional
        if True, a copy of W is returned, otherwise W is modified in place
        default is True

    Returns
    -------
    W : PxP np.ndarray
        symmetrised adjacency/connectivity matrix
    '''

    if copy:
        W = W.copy()

    is_binary = np.all(np.logical_or(np.isclose(W, 0), np.isclose(W, 1)))

    if is_binary:
        W = np.logical_or(W, W.T).astype(float)
    else:
        W_mean = (np.triu(W, k=1) + np.tril(W, k=-1)) / 2
        W = W_mean + W_mean.T + np.diag(np.diag(W))

    return W

def randomise(G: np.ndarray,
              copy: bool = True) -> np.ndarray:
    '''
    Randomly rewire edges of an adjacency/connectivity matrix

    Implemented as in https://github.com/rkdan/small_world_propensity

    Parameters
    ----------
    G : PxP np.ndarray
        adjacency/connectivity matrix

    Returns
    -------
    G_rand : PxP np.ndarray
        randomised adjacency/connectivity matrix
    '''

    if copy:
        G = G.copy()

    num_nodes = G.shape[0]
    G_rand = np.zeros((num_nodes, num_nodes))
    mask = np.triu(np.ones((num_nodes, num_nodes)), 1)

    grab_indices = np.column_stack(np.nonzero(mask.T)) # Find the indices where mask > 0

    orig_edges = G[grab_indices[:, 0], grab_indices[:, 1]]
    num_edges = len(orig_edges)
    rand_index = np.random.choice(num_edges, num_edges, replace=False)
    randomized_edges = orig_edges[rand_index]
    edge = 0
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            G_rand[i, j] = randomized_edges[edge]
            G_rand[j, i] = randomized_edges[edge]
            edge += 1

    return G_rand

def regular_matrix(G: np.ndarray, r: int) -> np.ndarray:
    '''
    Create a regular matrix

    Adapted from https://github.com/rkdan/small_world_propensity

    Parameters
    ----------
    G : PxP np.ndarray
        adjacency/connectivity matrix
    r : int
        average effective radius of the network

    Returns
    -------
    M : PxP np.ndarray
        adjacency matric of the regularised matrix
    '''

    n = G.shape[0]
    G_upper = np.triu(G)  # Keep only the upper triangular part
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

def postproc(W: np.ndarray,
             diag: float = 0,
             copy: bool = True) -> np.ndarray:
    '''
    Postprocessing of connectivity/adjacency matrix

    Ensures W is symmetric, sets diagonal to diag, removes NaNs and infinities, and ensures exact binarity

    Parameters
    ----------
    W : PxP np.ndarray
        adjacency/connectivity matrix

    diag : int, optional
        set diagonal to this value
        default is 0

    copy : bool, optional
        if True, a copy of W is returned, otherwise W is modified in place
        default is True

    Returns
    -------
    W : PxP np.ndarray
        processed adjacency/connectivity matrix
    '''
    if copy:
        W = W.copy()

    if not np.allclose(W, W.T):
        raise ValueError("Error: Matrix is not symmetrical")

    np.fill_diagonal(W, diag)
    np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    W = np.round(W, decimals=6) # This should ensure exact binarity if floating point inaccuracies occur
    return W


"""
SECTION: Graph analysis functions
 - Comet specfic graph analysis functions
 - Functions which rely on path length calculations are compiled
   to machine code using numba for improved performance
"""
def avg_shortest_path(G: np.ndarray,
                      include_diagonal: bool = False,
                      include_infinite: bool = False) -> float:
    '''
    Average shortest path length calculated from a connection length matrix.

    For binary matrices the connection length matrix is identical to the connectiivty matrix,
    for weighted connectivity matrices it can be obtained though e.g. graph.invert().

    Parameters
    ----------
    G : NxN np.ndarray
        undirected connection matrix (binary or weighted)

    Returns
    -------
    D : float
        average shortest path length
    '''

    is_binary = np.all(np.logical_or(np.isclose(G, 0), np.isclose(G, 1)))
    D = distance_bin(G) if is_binary else distance_wei(G)

    if np.isinf(D).any():
        import warnings
        issue = "The graph is not fully connected and infinite path lenghts were set to NaN"
        warnings.warn(issue)
    if not include_diagonal:
        np.fill_diagonal(D, np.nan)
    if not include_infinite:
        D[np.isinf(D)] = np.nan

    Dv = D[~np.isnan(D)]
    return np.mean(Dv)

def transitivity_und(A: np.ndarray) -> np.ndarray:
    '''
    Transitivity for undirected networks (binary and weighted), adapted from
    the bctpy implementation: https://github.com/aestrivex/bctpy

    Transitivity is the ratio of triangles to triplets in the network and is
    a classical version of the clustering coefficient.

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected connection matrix

    Returns
    -------
    T : float
        transitivity scalar
    '''

    is_binary = np.all(np.logical_or(np.isclose(A, 0), np.isclose(A, 1)))

    if is_binary:
        tri3 = np.trace(np.dot(A, np.dot(A, A)))
        tri2 = np.sum(np.dot(A, A)) - np.trace(np.dot(A, A))
        return tri3 / tri2
    else:
        K = np.sum(np.logical_not(A == 0), axis=1)
        ws = np.cbrt(A)
        cyc3 = np.diag(np.dot(ws, np.dot(ws, ws)))
        return np.sum(cyc3, axis=0) / np.sum(K * (K - 1), axis=0)

def avg_clustering_onella(W: np.ndarray) -> np.ndarray:
    '''
    Average clustering coefficient as described by Onnela et al. (2005) and as implemented in
    https://kk1995.github.io/BauerLab/BauerLab/MATLAB/lib/+mouse/+graph/smallWorldPropensity.html

    Parameters
    ----------
    W : NxN np.ndarray
        binary or weighted, undirected connection matrix

    Returns
    -------
    C : float
        average clustering coefficient

    References
    ----------
    Onnela, J. P., Saramäki, J., Kertész, J., & Kaski, K. (2005). Intensity and
    coherence of motifs in weighted complex networks. Physical Review E, 71(6), 065103.
    DOI: https://doi.org/10.1103/PhysRevE.71.065103
    '''

    K = np.count_nonzero(W, axis=1) # count all non-zero values as in the MATLAB implementation
    W2 = W / W.max()
    cyc3 = np.diagonal(np.linalg.matrix_power(W2 ** (1/3), 3))
    K = np.where(cyc3 == 0, np.inf, K)
    C = cyc3 / (K * K-1)

    return C.mean()

def efficiency(G: np.ndarray,
               local: bool = False) -> np.ndarray:
    '''
    Efficiency for binary and weighted networks (global and local)

    Optimized version of the bctpy implelementation by Roan LaPlante (https://github.com/aestrivex/bctpy)

    Global efficiency is the average of inverse shortest path length, and is inversely related to the characteristic path length.
    Local efficiency is the global efficiency computed on the neighborhood of the node, and is related to the clustering coefficient.

    Parameters
    ----------
    Gw : PxP np.ndarray
        adjacency/connectivity matrix (binary or weighted, directed or undirected)

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

    n = len(G)
    is_binary = np.all(np.logical_or(np.isclose(G, 0), np.isclose(G, 1)))

    # Efficiency for binary networks
    if is_binary:
        # Local efficiency
        if local:
            E = np.zeros((n,))

            for u in range(n):
                # find pairs of neighbors
                V, = np.where(np.logical_or(G[u, :], G[u, :].T))
                # inverse distance matrix
                e = distance_bin(G[np.ix_(V, V)], inv=True)
                # symmetrized inverse distance matrix
                se = e + e.T

                # symmetrized adjacency vector
                sa = G[u, V] + G[V, u].T
                numer = np.sum(np.outer(sa.T, sa) * se) / 2
                if numer != 0:
                    denom = np.sum(sa)**2 - np.sum(sa * sa)
                    E[u] = numer / denom

        # Global efficiency
        else:
            e = distance_bin(G, inv=True)
            E = np.sum(e) / (n * n - n)

    # Efficiency for weighted networks
    else:
        n = len(G)
        Gl = invert(G, copy=True) # invert to get connection length matrix
        A = np.array((G != 0), dtype=int)

        # Local efficiency as described by Wang et al 2016
        if local:
            E = np.zeros((n,))
            for u in range(n):
                V, = np.where(np.logical_or(G[u, :], G[:, u].T))
                sw = np.cbrt(G[u, V]) + np.cbrt(G[V, u].T)
                e = distance_wei(np.cbrt(Gl)[np.ix_(V, V)], inv=True)
                se = e+e.T

                numer = np.sum(np.outer(sw.T, sw) * se) / 2
                if numer != 0:
                    # symmetrized adjacency vector
                    sa = A[u, V] + A[V, u].T
                    denom = np.sum(sa)**2 - np.sum(sa * sa)

                    E[u] = numer / denom  # local efficiency

        # Global efficiency
        else:
            e = distance_wei(Gl, inv=True)
            E = np.sum(e) / (n * n - n)

    return E

def small_world_sigma(G: np.ndarray,
                      nrand: int = 10) -> np.ndarray:
    '''
    Small-worldness sigma for undirected networks (binary or weighted)

    Small worldness sigma is calculated as the ratio of the clustering coefficient and the characteristic path length
    of the real network to the average clustering coefficient and characteristic path length of the random networks.

    Parameters
    ----------
    G : PxP np.ndarray
        undireted adjacency/connectivity matrix

    nrand : int, optional
        number of random networks to generate (and average over). Default is 10.

    Returns
    -------
    sigms : float
        small-worldness sigma

    Notes
    -----
    This implementation of small worldness relies on matrix operations and is *drastically* faster than the
    Networkx implementation. However, it uses a different approch for rewiring edges, so the results will differ.

    It automatically detects if the input matrix is binary or weighted.
    '''

    randMetrics = {"C": [], "L": []}
    for _ in range(nrand):
        Gr = randomise(G)
        randMetrics["C"].append(transitivity_und(Gr))
        randMetrics["L"].append(avg_shortest_path(Gr))

    C = transitivity_und(G)
    L = avg_shortest_path(G)

    Cr = np.mean(randMetrics["C"])
    Lr = np.mean(randMetrics["L"])

    sigma = (C / Cr) / (L / Lr)
    return sigma

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

    if not np.allclose(G, G.T):
        raise ValueError("Error: Matrix is not symmetrical")

    n = G.shape[0]  # Number of nodes

    # Compute the average degree of the unweighted network (approximate radius)
    num_connections = np.count_nonzero(G)
    avg_deg_unw = num_connections / n
    avg_rad_unw = avg_deg_unw / 2
    avg_rad_eff = np.ceil(avg_rad_unw).astype(int)
    # Compute the regular and random matrix for the network
    G_reg = regular_matrix(G, avg_rad_eff)
    G_rand = randomise(G)

    # Path length calculations for the network, ignore divide by zero warnings
    with np.errstate(divide='ignore'):
            G_reg_inv = np.divide(1.0, G_reg)
            G_rand_inv = np.divide(1.0, G_rand)
            G_inv = np.divide(1.0, G)

    reg_path = avg_shortest_path(G_reg_inv)
    rand_path = avg_shortest_path(G_rand_inv)
    net_path = avg_shortest_path(G_inv)

    # Compute the normalized difference in path lengths
    A = max(net_path - rand_path, 0)  # Ensure A is non-negative
    diff_path = A / (reg_path - rand_path) if (reg_path != float('inf') and rand_path != float('inf') and net_path != float('inf')) else 1
    diff_path = min(diff_path, 1)  # Ensure diff_path does not exceed 1

    # Compute all clustering calculations for the network
    reg_clus = avg_clustering_onella(G_reg)
    rand_clus = avg_clustering_onella(G_rand)
    net_clus = avg_clustering_onella(G)

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

@jit(nopython=True)
def matching_ind_und(G: np.ndarray) -> np.ndarray:
    '''
    Matching index for undirected networks

    Based on the MATLAB implementation by Stuart Oldham:
    https://github.com/StuartJO/FasterMatchingIndex

    Matching index is a measure of similarity between two nodes' connectivity profiles
    (excluding their mutual connection, should it exist). Weighted matrices will be binarised.

    Parameters
    ----------
    W : PxP np.ndarray
        undirected adjacency/connectivity matrix, will be binarised

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

    G = (G > 0).astype(np.float64) # binarise the adjacency matrix
    n = G.shape[0]

    nei = np.dot(G, G)
    deg = np.sum(G, axis=1)
    degsum = deg[:, np.newaxis] + deg

    denominator = np.where((degsum <= 2) & (nei != 1), 1.0, degsum - 2 * G)
    M = np.where(denominator != 0, (nei * 2) / denominator, 0.0)

    for i in range(n):
        M[i, i] = 0.0

    return M

@jit(nopython=True)
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

@jit(nopython=True)
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


"""
SECTION: bctpy wrapper functions
 - The wrapper functions implement type hinting which is required for the GUI
 - For the scripting API, users can also directly use the bctpy functions
"""
def backbone_wu(CIJ: np.ndarray,
                avgdeg: int = 0,
                verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    '''
    This is a wrapper function for the backbone_wu() function
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    The network backbone contains the dominant connections in the network
    and may be used to aid network visualization. This function computes
    the backbone of a given weighted and undirected connection matrix CIJ,
    using a minimum-spanning-tree based algorithm.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        weighted undirected connection matrix
    avgdeg : float
        desired average degree of backbone
    verbose : bool
        print out edges whilst building spanning tree. Default False.

    Returns
    -------
    CIJtree : NxN np.ndarray
        connection matrix of the minimum spanning tree of CIJ
    CIJclus : NxN np.ndarray
        connection matrix of the minimum spanning tree plus strongest
        connections up to some average degree 'avgdeg'. Identical to CIJtree
        if the degree requirement is already met.

    Notes
    -----
    Nodes with zero strength are discarded.

    CIJclus will have a total average degree exactly equal to
        (or very close to) 'avgdeg'.

    'avgdeg' backfill is handled slightly differently than in Hagmann et al. (2008)
    '''

    return bct.backbone_wu(CIJ, avgdeg, verbose)

def betweenness(G: np.ndarray) -> np.ndarray:
    '''
    This is a wrapper function for the betweenness_*() functions
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    Node betweenness centrality is the fraction of all shortest paths in
    the network that contain a given node. Nodes with high values of
    betweenness centrality participate in a large number of shortest paths.

    Parameters
    ----------
    G : NxN np.ndarray
        binary/weighted directed/undirected connection matrix

    Returns
    -------
    BC : Nx1 np.ndarray
        node betweenness centrality vector

    Notes
    -----
    Binary:
    Betweenness centrality may be normalised to the range [0,1] as
    BC/[(N-1)(N-2)], where N is the number of nodes in the network.

    Weighted:
    The input matrix must be a connection-length matrix, typically
    obtained via a mapping from weight to length. For instance, in a
    weighted correlation network higher correlations are more naturally
    interpreted as shorter distances and the input matrix should
    consequently be some inverse of the connectivity matrix.
    Betweenness centrality may be normalised to the range [0,1] as
    BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    '''

    is_binary = np.all(np.logical_or(np.isclose(G, 0), np.isclose(G, 1)))
    res = bct.betweenness_bin(G) if is_binary else bct.betweenness_wei(G)
    return res

def clustering_coef(W: np.ndarray) -> np.ndarray:
    '''
    This is a wrapper function for the clustering_coef_*() functions
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    The binary clustering coefficient is the fraction of triangles around a node
    (equiv. the fraction of nodes neighbors that are neighbors of each other).

    The weighted clustering coefficient is the average "intensity" of triangles
    around a node.

    Parameters
    ----------
    W : NxN np.ndarray
        weighted undirected connection matrix

    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector
    '''

    is_binary = np.all(np.logical_or(np.isclose(W, 0), np.isclose(W, 1)))
    res = bct.clustering_coef_bu(W) if is_binary else bct.clustering_coef_wu(W)
    return res

def degrees_und(CIJ: np.ndarray) -> np.ndarray:
    '''
    This is a wrapper function for the degrees_und() function
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    Node degree is the number of links connected to the node.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        undirected binary/weighted connection matrix

    Returns
    -------
    deg : Nx1 np.ndarray
        node degree

    Notes
    -----
    Weight information is discarded.
    '''

    return bct.degrees_und(CIJ)

def density_und(CIJ: np.ndarray) -> tuple[float, int, int]:
    '''
    This is a wrapper function for the density_und() function
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    Density is the fraction of present connections to possible connections.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        directed weighted/binary connection matrix

    Returns
    -------
    kden : float
        density
    N : int
        number of vertices
    k : int
        number of edges

    Notes
    -----
    Assumes CIJ is directed and has no self-connections.
    Weight information is discarded.
    '''

    return bct.density_und(CIJ)

def eigenvector_centrality_und(CIJ: np.ndarray) -> np.ndarray:
    '''
    This is a wrapper function for the eigenvector_centrality_*() functions
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    Eigenector centrality is a self-referential measure of centrality:
    nodes have high eigenvector centrality if they connect to other nodes
    that have high eigenvector centrality. The eigenvector centrality of
    node i is equivalent to the ith element in the eigenvector
    corresponding to the largest eigenvalue of the adjacency matrix.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        Binary/weighted undirected adjacency matrix

    Returns
    ----------
    v : Nx1 np.ndarray
        Eigenvector associated with the largest eigenvalue of the matrix
    '''

    return bct.eigenvector_centrality_und(CIJ)

def gateway_coef_sign(W: np.ndarray,
                      ci: Literal["louvain"] = "louvain",
                      centrality_type: Literal["degree", "betweenness"] = "degree", ) \
                                        -> tuple[np.ndarray, np.ndarray]:
    '''
    This is a wrapper function for the gateway_coef_sign() function
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    The gateway coefficient is a variant of participation coefficient.
    It is weighted by how critical the connections are to intermodular
    connectivity (e.g. if a node is the only connection between its
    module and another module, it will have a higher gateway coefficient,
    unlike participation coefficient).

    Parameters
    ----------
    W : NxN np.ndarray
        undirected signed connection matrix
    ci : Nx1 np.ndarray
        community affiliation vector
    centrality_type : enum
        'degree' - uses the weighted degree (i.e, node strength)
        'betweenness' - uses the betweenness centrality

    Returns
    -------
    Gpos : N x nr_mod np.ndarray
        gateway coefficient for positive weights
    Gneg : N x nr_mod np.ndarray
        gateway coefficient for negative weights

    References
    ----------
    Vargas ER, Wahl LM, Eur Phys J B (2014) 87:1-10
    '''

    ci, _ = bct.community_louvain(W)
    return bct.gateway_coef_sign(W, ci, centrality_type)

def pagerank_centrality(A: np.ndarray,
                        d: float = 0.85,
                        falff: float = None) -> np.ndarray:
    '''
    This is a wrapper function for the pagerank_centrality() function
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    The PageRank centrality is a variant of eigenvector centrality. This
    function computes the PageRank centrality of each vertex in a graph.

    Formally, PageRank is defined as the stationary distribution achieved
    by instantiating a Markov chain on a graph. The PageRank centrality of
    a given vertex, then, is proportional to the number of steps (or amount
    of time) spent at that vertex as a result of such a process.

    The PageRank index gets modified by the addition of a damping factor,
    d. In terms of a Markov chain, the damping factor specifies the
    fraction of the time that a random walker will transition to one of its
    current state's neighbors. The remaining fraction of the time the
    walker is restarted at a random vertex. A common value for the damping
    factor is d = 0.85.

    Parameters
    ----------
    A : NxN np.narray
        adjacency matrix
    d : float
        damping factor (see description)
    falff : Nx1 np.ndarray | None
        Initial page rank probability, non-negative values. Default value is
        None. If not specified, a naive bayesian prior is used.

    Returns
    -------
    r : Nx1 np.ndarray
        vectors of page rankings

    Notes
    -----
    Note: The algorithm will work well for smaller matrices (number of
    nodes around 1000 or less)
    '''

    return bct.pagerank_centrality(A, d, falff)

def participation_coef(W: np.ndarray,
                       ci: Literal["louvain"] = "louvain",
                       degree: Literal["undirected"] = "undirected") -> np.ndarray:
    '''
    This is a wrapper function for the participation_coef functions
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    The participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.

    Parameters
    ----------
    W : NxN np.ndarray or scipy.sparse.csr_matrix
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.ndarray
        community affiliation vector (just for the GUI, will always use bct.community_louvain())
    degree : str
        Flag to describe nature of graph 'undirected': For undirected graphs
                                         'in': Uses the in-degree
                                         'out': Uses the out-degree

    Returns
    -------
    P : Nx1 np.ndarray
        participation coefficient
    '''

    ci, _ = bct.community_louvain(W)
    res = bct.participation_coef_sparse(W, ci, degree) if isinstance(W, scipy.sparse.csr_matrix) else bct.participation_coef(W, ci, degree)
    return res

def participation_coef_sign(W: np.ndarray,
                            ci: Literal["louvain"] = "louvain",) -> tuple[np.ndarray, np.ndarray]:
    '''
    This is a wrapper function for the participation_coef_sign() function
    of the bctpy toolbox: https://github.com/aestrivex/bctpy.

    The participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected connection matrix with positive and negative weights
    ci : Nx1 np.ndarray
        community affiliation vector (just for the GUI, will always use bct.community_louvain())

    Returns
    -------
    Ppos : Nx1 np.ndarray
        participation coefficient from positive weights
    Pneg : Nx1 np.ndarray
        participation coefficient from negative weights
    '''

    ci, _ = bct.community_louvain(W)
    res = bct.participation_coef_sign(W, ci)
    return res

"""
def rich_club(CIJ: np.ndarray,
                 weighted: bool=True,
                 klevel: int = None) -> np.ndarray:
    res = bct.rich_club_wu(CIJ, klevel) if weighted else bct.rich_club_bu(CIJ, klevel)
    print("rich club", type(res))
    print(res)
    label = "Rich club coefficient vectors (weighted)" if weighted else "Rich club coefficient vectors (binary)"
    return res, label"""
