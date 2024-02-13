import numpy as np
from numba import jit

def handle_negative_weights(W, type="absolute", copy=True):
    '''Handle negative weights in a connectivity/adjacency matrix

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

def threshold(W, type="absolute", threshold=None, density=None, copy=True):
    '''Thresholding of connectivity/adjacency matrix
    
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
        assert density > 0 and density < 1, "Error: Density must be between 0 and 1"
        assert np.allclose(W, W.T), "Error: Matrix is not symmetric"
        
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

def binarise(W, copy=True):
    '''Binarise connectivity/adjacency matrix

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

def normalise(W, copy=True):
    '''Normalise connectivity/adjacency matrix

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

    assert np.max(np.abs(W)) > 0, "Error: Matrix contains only zeros"
    W /= np.max(np.abs(W))
    return W

def invert(W, copy=True):
    '''Invert connectivity/adjacency matrix

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

def logtransform(W, epsilon=1e-10, copy=True):
    '''Log transform of connectivity/adjacency matrix

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

def postproc(W, diag=0, copy=True):
    '''Postprocessing of connectivity/adjacency matrix
    
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

    assert np.allclose(W, W.T), "Error: Matrix is not symmetrical"
    np.fill_diagonal(W, diag)
    np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    W = np.round(W, decimals=5) # This should ensure exact binarity if floating point inaccuracies occur
    return W

def efficiency_wei(Gw, local=False):
    '''Efficiency for weighted networks
    
    Based on the bctpy implelementation by Roan LaPlante: https://github.com/aestrivex/bctpy
    Global efficiency is the average of inverse shortest path length, and is inversely related to the characteristic path length.
    Local efficiency is the global efficiency computed on the neighborhood of the node, and is related to the clustering coefficient.
    
    Parameters
    ----------
    Gw : PxP np.ndarray
        unidireted weighted adjacency/connectivity matrix

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

    Notes
    -----
    Algorithm: Modified Dijkstra's algorithm
    '''
    @jit(nopython=True)
    def distance_inv_wei(G):
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

        np.fill_diagonal(D, 1)
        D = 1 / D
        np.fill_diagonal(D, 0)
        return D

    n = len(Gw)
    Gl = invert(Gw, copy=True)  # connection length matrix
    A = np.array((Gw != 0), dtype=int)
   
    #local efficiency algorithm described by Wang et al 2016, recommended
    if local:
        E = np.zeros((n,))
        for u in range(n):
            V, = np.where(np.logical_or(Gw[u, :], Gw[:, u].T))
            sw = np.cbrt(Gw[u, V]) + np.cbrt(Gw[V, u].T)
            e = distance_inv_wei(np.cbrt(Gl)[np.ix_(V, V)])
            se = e+e.T
            
            numer = np.sum(np.outer(sw.T, sw) * se) / 2
            if numer != 0:
                # symmetrized adjacency vector
                sa = A[u, V] + A[V, u].T
                denom = np.sum(sa)**2 - np.sum(sa * sa)
                # print numer,denom
                E[u] = numer / denom  # local efficiency
    else:
        e = distance_inv_wei(Gl)
        E = np.sum(e) / (n * n - n)

    return E

def efficiency_bin(G, local=False):    
    '''Efficiency for binary networks
    
    Based on the bctpy implelementation by Roan LaPlante: https://github.com/aestrivex/bctpy
    Global efficiency is the average of inverse shortest path length, and is inversely related to the characteristic path length.
    Local efficiency is the global efficiency computed on the neighborhood of the node, and is related to the clustering coefficient.
    
    Parameters
    ----------
    G : PxP np.ndarray
        unidireted binary adjacency/connectivity matrix

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

    Fagiolo, G. (2007). Clustering in complex directed networks. Physical Review E, 76(2), 026107.
    DOI: https://doi.org/10.1103/PhysRevE.76.026107

    Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: uses and interpretations. 
    Neuroimage, 52(3), 1059-1069. DOI: https://doi.org/10.1016/j.neuroimage.2009.10.003

    Notes
    -----
    Algorithm: Matrix multiplication to find paths, faster than original Dijkstra's algorithm
    '''
    @jit(nopython=True)
    def distance_inv(g):
        D = np.eye(len(g))
        n = 1
        nPATH = g.copy()
        L = (nPATH != 0)

        while np.any(L):
            D += n * L
            n += 1
            nPATH = np.dot(nPATH, g)
            L = (nPATH != 0) * (D == 0)
        
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if not D[i, j]:
                    D[i, j] = np.inf

        D = 1 / D
        np.fill_diagonal(D, 0)
        return D

    G = binarise(G)
    n = len(G)  # number of nodes
    if local:
        E = np.zeros((n,))  # local efficiency

        for u in range(n):
            # find pairs of neighbors
            V, = np.where(np.logical_or(G[u, :], G[u, :].T))
            # inverse distance matrix
            e = distance_inv(G[np.ix_(V, V)])
            # symmetrized inverse distance matrix
            se = e + e.T

            # symmetrized adjacency vector
            sa = G[u, V] + G[V, u].T
            numer = np.sum(np.outer(sa.T, sa) * se) / 2
            if numer != 0:
                denom = np.sum(sa)**2 - np.sum(sa * sa)
                E[u] = numer / denom  # local efficiency
    else:
        e = distance_inv(G)
        E = np.sum(e) / (n * n - n)
    
    return E

@jit(nopython=True)
def matching_ind_und(G):
    '''Matching index for undirected networks
    
    Based on the MATLAB implementation by Stuart Oldham: https://github.com/StuartJO/FasterMatchingIndex
    Matching index is a measure of similarity between two nodes' connectivity profiles (excluding their mutual connection, should it exist).

    Parameters
    ----------
    W : PxP np.ndarray
        unidireted adjacency/connectivity matrix

    Returns
    -------
    M : PxP np.ndarray
        matching index matrix
   
    References
    ----------
    Oldham, S., Fulcher, B. D., Aquino, K., Arnatkevičiūtė, A., Paquola, C., Shishegar, R., & Fornito, A. (2022). Modeling spatial, developmental, 
    physiological, and topological constraints on human brain connectivity. Science advances, 8(22), eabm6127. DOI: https://doi.org/10.1126/sciadv.abm6127 

    Betzel, R. F., Avena-Koenigsberger, A., Goñi, J., He, Y., De Reus, M. A., Griffa, A., ... & Sporns, O. (2016). 
    Generative models of the human connectome. Neuroimage, 124, 1054-1064.DOI: https://doi.org/10.1016/j.neuroimage.2015.09.041

    Notes
    -----
    Important note: As of Jan 2024 there is a bug in the bctpy version of this function (ncon2 = c1 * CIJ should be ncon2 = CIJ * use).
    This bug is fixed/irrelevant here due to the more efficient implementation using matrix operations and numba.
    '''
    G = (G > 0).astype(np.float64)
    n = G.shape[0]
    nei = np.dot(G, G)
    deg = np.sum(G, axis=1)
    degsum = deg[:, np.newaxis] + deg
    denominator = np.where((degsum <= 2) & (nei != 1), 1.0, degsum - 2 * A)
    M = np.where(denominator != 0, (nei * 2) / denominator, 0.0)
    for i in range(n):
        M[i, i] = 0.0
    return M

def small_worldness(G):
    pass





