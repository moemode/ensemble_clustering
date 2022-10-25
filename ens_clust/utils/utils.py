import numpy as np
from numpy import ndarray
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csr import csr_matrix
from scipy.stats import entropy
from sklearn.preprocessing import normalize


def partition_entropy(y: ndarray) -> float:
    """
    Calculate entropy of label vector of a single clustering
    Args:
        y (ndarray): label vector of clustering / partition
    Returns:
        float : calculated entropy
    """
    y = y.reshape(1, -1)
    N = y.shape[1]
    _, cluster_counts = np.unique(y, return_counts=True)
    cluster_frequencies = cluster_counts / N
    return entropy(cluster_frequencies)


def partition_entropies(Y: ndarray) -> np.ndarray:
    """
    Calculates entropies of an ensemble.
    Args:
        Y (ndarray of shape (N, H)): ensemble of clusterings
    Returns:
        np.ndarray of shape (H,): One entropy for each of the H clusterings / partitions.
    """
    _, H = Y.shape[0], Y.shape[1]
    entropies = np.empty((H, ))
    for j in range(H):
        entropies[j] = partition_entropy(Y[:, j])
    return entropies


def max_bipartite_matching_ba(ba0: ndarray, ba1: ndarray):
    """Has the advantage of working for soft clusterings represented by their association matrices

    Args:
        ba0 (ndarray): [description]
        ba1 (ndarray): [description]

    Returns:
        [type]: [description]
    """
    cluster_coassoc_matrix = ba0.T.dot(ba1).toarray()
    return linear_sum_assignment(cluster_coassoc_matrix, maximize=True)


def permutation_bipartite_matching(association_matrix_0,
                                   association_matrix_1,
                                   return_permutation_matrix=False):
    """The association matrices represent partitions i.e. clusterings. These do not have to be
    hard so the entries of the matrices can be reals in [0, 1]. Binary association matrices
    containing only 0/1 entries are covered by this.
    Args:
        association_matrix_0 (ndarray of shape (N, k0)): A matrix representing the association of points with clusters. 
        There are k0 clusters. association_matrix_0[i][j] = association of point i with cluster j in clustering 0.
        association_matrix_1 (ndarray of shape (N, k1)): A matrix representing the association of points with clusters. 
        There are k1 clusters. association_matrix_1[i][j] = association of point i with cluster j in clustering 1
    """
    k_ref, k_U = association_matrix_0.shape[1], association_matrix_1.shape[1]
    # ref_indices[i] corresponds to U_indices[i]
    ref_indices, U_indices = max_bipartite_matching_ba(association_matrix_0,
                                                       association_matrix_1)
    # construct a matrix W s.t. columns of association_matrix_1 are swapped when multiplied by W
    # according to the correspondence of ref_indices with U_indices
    W = csr_matrix((np.repeat(1, ref_indices.size), (U_indices, ref_indices)),
                   shape=(k_U, k_ref))
    return W


def cluster_distribution(association_matrix: ndarray) -> ndarray:
    """[summary]

    Args:
        association_matrix (ndarray of shape(N, k)): represents a soft clustering, i.e. values are in [0, 1] and
        rows sum to one.
    Returns:
        ndarray of shape (k,): frequency with which each of the k clusters is assigned
    """
    if type(association_matrix) is not np.ndarray:
        association_matrix = association_matrix.toarray()
    N = association_matrix.shape[0]
    return association_matrix.sum(axis=0) / N


def normalize_columns(a: ndarray) -> ndarray:
    return normalize(a, norm="l1", axis=0)


def nunique(a: ndarray, axis=0) -> int:
    return (np.diff(np.sort(a, axis=axis), axis=axis) != 0).sum(axis=axis) + 1


def min_k_partition_indices(ens: np.ndarray, k: int) -> np.ndarray:
    """
    Determine which clusterings / partitions in ensemble have at least k clusters.
    Args:
        ens (np.ndarray of shape (N, H)): ensemble
        k (int): minimum number of clusters
    Returns:
        np.ndarray of shape (t,): column indices into ens, of the t clusters / partitions with at least k clusters
    """
    return (nunique(ens) >= k).nonzero()[0]
