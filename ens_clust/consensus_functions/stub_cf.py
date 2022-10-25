import numpy as np
from numpy import ndarray


def consensus_function(L: ndarray, k_out: int) -> ndarray:
    """
    Given the label matrix of an ensemble with N points and M base clusterings, 
    compute a consensus clustering with k_out clusters.
    Args:
        L (ndarray of shape (N, M) and dtype int or np.int64): Representation of label matrix:
                    L[i][j]==l iff i-th point belongs to l-th cluster in j-th base clustering
                    with i in range(N) and j in range(M). If L contains k distinct values, these must be the ints
                    in range(k). Arbitrary values are not allowed.
        k_out (int): desired number of clusters in consensus_clustering
    Returns:
        consensus_clustering (ndarray of shape (N,) and dtype int or np.int64): consensus_clustering[i]==l iff 
        i-th point belongs to l-th cluster in consensus_clustering. consensus_clustering may contain
        k_actual distinct values, with  0 < k_actual <= k_out. These values must be the ints in range(k_actual). 
        Arbitrary values are not allowd.
    """
    N, M = L.shape
    consensus_clustering = np.zeros(shape=(N, ), dtype=int)
    # Actual consensus_clustering Computation HERE
    return consensus_clustering
