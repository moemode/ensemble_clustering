import math
import random
import numpy as np
from numpy import ndarray
from ens_clust.generation.base_clusterer import single_kmeans
from typing import Callable


def fixed_k_ensemble(instances: ndarray,
                     ensemble_k: int,
                     nr_partitions: int,
                     clustering_function: Callable = single_kmeans) -> ndarray:
    """
    Create an ensemble of nr_partitions clustering.
    Each clusterings results from applying the clustering_function with k
    set to ensemble_k to instances.
    Args:
        instances (ndarray of shape (N, d)): dataset
        ensemble_k (int): k for clustering_function
        nr_partitions (int): number of partitions/clusterings
        clustering_function (Callable, optional): produces nr_partitions
        Defaults to single_kmeans.
    Returns:
        ndarray of shape (N, nr_partitions): fixed-k ensemble
    """
    ens = np.empty((instances.shape[0], nr_partitions), dtype=int)
    for j in range(nr_partitions):
        ens[:, j] = clustering_function(instances, ensemble_k)
    return ens


def k_range_ensemble(instances: ndarray,
                     ensemble_k_range,
                     nr_partitions: ndarray,
                     clustering_function: Callable = single_kmeans):
    """
    Create an ensemble of nr_partitions clustering.
    Each clusterings results from applying the clustering_function with k
    set to a value in ensemble_k_range to instances.
    Args:
        instances (ndarray of shape (N, d)): dataset
        ensemble_k_range: range of values for k of clustering_function
        nr_partitions (int): number of partitions/clusterings
        clustering_function (Callable, optional): produces nr_partitions.
            Defaults to single_kmeans.
    Returns:
        ndarray of shape (N, nr_partitions): ensemble
    """
    ens = np.empty((instances.shape[0], nr_partitions), dtype=int)
    for j in range(nr_partitions):
        ens[:, j] = clustering_function(instances,
                                        np.random.choice(ensemble_k_range))
    return ens


def huang2020_ensemble(instances: ndarray, true_k: int, nr_partitions=20):
    """
    Creates ensembles like in the experiments of Huang, Dong, Chang-Dong Wang,
    Hongxing Peng, Jianhuang Lai, and Chee-Keong Kwoh
    “Enhanced Ensemble Clustering via Fast Propagation
    of Cluster-Wise Similarities.” https://doi.org/10.1109/TSMC.2018.2876202.
    k is in true_k...min(100, sqrt(N)) in each of nr_partitions clusters.
    Args:
        instances (ndarray of shape (N, d)): dataset
        true_k (int): true / actual number of clusters in dataset
        nr_partitions (int, optional): [description]. Defaults to 20.
    Raises:
        ValueError: if true_k > 100 or true_k > sqrt(N)
    Returns:
        ndarray of shape (N, nr_partitions): ensemble
    """
    N = instances.shape[0]
    upper_bound = min(round(math.sqrt(N)), 100) + 1
    ensemble_k_range = np.arange(true_k, upper_bound)
    if (ensemble_k_range.size == 0):
        raise ValueError("The range of possible k's for ensemble generation is\
                         empty for these instances combined with true_k")
    return k_range_ensemble(instances, ensemble_k_range, nr_partitions)


def random_feature_selection_ensemble(
        instances: ndarray,
        features_per_partition: int,
        ensemble_k: int,
        nr_partitions: int = 20,
        clustering_function: Callable = single_kmeans):
    """
    Creates ensemble by clustering randomly selected attributes/dimensions
    of the points in instances.
    Args:
        instances (ndarray of shape (N, d)): dataset
        features_per_partition (int): number of attributes/dimensions to
            randomly select for each point in instances before clustering
        ensemble_k: k of clustering_function
        nr_partitions (int, optional): Defaults to 20.
        clustering_function (Callable, optional): [description].
        Defaults to single_kmeans.

    Returns:
        ndarray of shape (N, nr_partitions): [description]
    """
    N, d = instances.shape
    dimension_indices = range(d)
    ens = np.empty((N, nr_partitions), dtype=int)
    for j in range(nr_partitions):
        selected_dimensions = random.sample(dimension_indices,
                                            features_per_partition)
        ens[:, j] = clustering_function(instances[:, selected_dimensions],
                                        np.random.choice(ensemble_k))
    return ens
