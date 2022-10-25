import numpy as np
from numpy import ndarray


def generate_random_ensemble(N: int, H, k_max: int) -> ndarray:
    """
    Generates a numpy array representing an ensemble, cluster labels might not be consecutive.
    Each column of the array represents a clustering of the dataset of n points. 
    Args:
        N: number of rows of returned array, number of points in dataset
        H: number of columns of returned array, number of clusterigns in ensemble
        k_max ([type]): integers in resulting array are from 0..k_max-1
    Returns:
        ndarray of shape (N, H): represents the ensemble
    """
    return np.random.randint(k_max, size=(N, H))


def generate_random_ensemble_cluster_counts(N: int, H: int,
                                            cluster_counts) -> ndarray:
    """
    Generates a numpy array result of shape (N, H), cluster labels might not be consecutive. 
    Every point is assigned to a random cluster from 0..k_max-1.
    Each column of the array represents a clustering of the dataset of n points. 
    Args:
        N: number of rows of returned array, number of points in dataset
        H: number of columns of returned array, number of clusterings in ensemble
        cluster_counts (iterable of length H): cluster_counts[i] is desired number of clusters in i-th clustering
    Returns:            
        ndarray of shape (N, H): represents the ensemble
    """
    result = np.empty((N, H)).astype("int64")
    for (clustering_index, cluster_count) in enumerate(cluster_counts):
        result[:, clustering_index] = np.random.randint(0,
                                                        cluster_count,
                                                        size=(N))
    return result
