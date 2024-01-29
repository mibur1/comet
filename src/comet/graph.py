import bct
import numpy as np
from numba import jit

def efficiency_wei(Gw, local=True):
    """
    The local efficiency is the global efficiency computed on the
    neighborhood of the node, and is related to the clustering coefficient.
    """

    def invert(W, copy=True):
        if copy:
            W = W.copy()
        E = np.where(W)
        W[E] = 1. / W[E]
        return W
    
    def cuberoot(x):
        return np.sign(x) * np.abs(x)**(1 / 3)

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
            sw = cuberoot(Gw[u, V]) + cuberoot(Gw[V, u].T)
            e = distance_inv_wei(cuberoot(Gl)[np.ix_(V, V)])
            se = e+e.T
            
            numer = np.sum(np.outer(sw.T, sw) * se) / 2
            if numer != 0:
                # symmetrized adjacency vector
                sa = A[u, V] + A[V, u].T
                denom = np.sum(sa)**2 - np.sum(sa * sa)
                # print numer,denom
                E[u] = numer / denom  # local efficiency
    else:
        raise ValueError("Removed global efficiency as it is not used")

    return E

def efficiency_bin(G, local=True):
    """
    The local efficiency is the global efficiency computed on the
    neighborhood of the node, and is related to the clustering coefficient.
    """

    def binarise(W, copy=True):
        if copy:
            W = W.copy()
        W[W != 0] = 1
        return W

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
        raise ValueError("Removed global efficiency as it is not used")
    
    return E

def efficiency(W, binarise=False):
    return efficiency_bin(W, local=True) if binarise else efficiency_wei(W, local=True)

def participation(W):
    ci, q = bct.community_louvain(W)
    return bct.participation_coef(W, ci, degree="undirected")

def clustering(W, binarise=False):
    return  bct.clustering_coef_bu(W) if binarise else bct.clustering_coef_wu(W)

def compute_graph_measures(t, features, index, density, binarise):
    W = np.asarray(features[t, :, :]).copy()  # Create a copy of the data
    np.fill_diagonal(W, 0)
    W = np.abs(W) # only absolute values
    
    # Density based thresholding
    sorted_weights = np.sort(W[np.triu_indices_from(W, k=1)])[::-1] # sort in descending order (excluding the diagonal)
    threshold_idx = int(len(sorted_weights) * density) # find index for resulting density
    threshold_value = sorted_weights[threshold_idx] # get the threshold value
    W = np.where(W >= threshold_value, W, 0) # apply threshold

    # Calculate and return graph measures
    return {
        "participation": participation(W),
        "clustering": clustering(W, binarise),
        "efficiency": efficiency(W, binarise),
        "index": index[t]
    }