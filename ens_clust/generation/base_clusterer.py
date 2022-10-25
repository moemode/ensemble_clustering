from typing import Callable, List
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from numpy import ndarray


def k_free(function: Callable) -> Callable:
    """
    Decorator to mark a clustering function if it does NOT need k.
    Args:
        function (Callable): clustering function
    Returns:
        Callable: decorated clustering function
    """
    function.k_free = True
    return function


def is_k_free(clustering_function: Callable):
    """
    Check whether a function has been decorated with k_free
    Args:
        clustering_function (Callable)
    Returns:
        bool: true iff function is k_free
    """
    return hasattr(clustering_function, 'k_free')


def single_random_init_kmeans(instances: ndarray, k: int):
    """
    Does one run of randomly initialized K-means with k clusters on instances.
    Args:
        instances (ndarray of shape (N, d)): dataset 
        k (int): number of clusters

    Returns:
        ndarray of shape (N,): labels of clustering / partition
    """
    return KMeans(n_clusters=k, init="random", n_init=1).fit_predict(instances)


def single_kmeans(instances: ndarray, k: int):
    """
    Does one run of K-means++ with k clusters on instances.
    Args:
        instances (ndarray of shape (N, d)): dataset 
        k (int): number of clusters

    Returns:
        ndarray of shape (N,): labels of clustering / partition
    """
    return KMeans(n_clusters=k, n_init=1).fit_predict(instances)


def sklearn_default_kmeans(instances, k):
    return KMeans(n_clusters=k).fit_predict(instances)


def single_linkage(instances, k):
    return AgglomerativeClustering(linkage="single",
                                   n_clusters=k).fit_predict(instances)


def average_linkage(instances, k):
    return AgglomerativeClustering(linkage="average",
                                   n_clusters=k).fit_predict(instances)


def complete_linkage(instances, k):
    return AgglomerativeClustering(linkage="complete",
                                   n_clusters=k).fit_predict(instances)


@k_free
def optics_default(instances):
    return OPTICS().fit_predict(instances)


clustering_function_name = {
    single_kmeans: "Single SK KM",
    single_random_init_kmeans: "Rand KM",
    sklearn_default_kmeans: "SK KM",
    optics_default: "OPTICS",
    single_linkage: "SL",
    complete_linkage: "CL",
    average_linkage: "AL"
}
clustering_functions: List[Callable] = [
    sklearn_default_kmeans, single_kmeans, single_linkage, complete_linkage,
    average_linkage, optics_default
]
