from sklearn.cluster import KMeans
from numpy import ndarray
from ens_clust.datastructures.datastructures import adjusted_ba


def adjusted_ba_kmeans(Y: ndarray,
                       k_out: int,
                       n_init: int = 3,
                       random_state=None,
                       verbose=False):
    """
    Given the label matrix Y of an ensemble with N points and M base clusterings,
    compute consensus_clustering with k_out clusters.
    The consensus clustering is the result of K-means with k_out clusters applied to
    a normalized version of the binary association matrix of the ensemble.
    Implements QMI from A. Topchy, A. Jain, W. Punch. 
    “Clustering Ensembles: Models of Consensus and Weak Partitions”.(doi: 10.1109/TPAMI.2005.237)
    Args:
        Y (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
        k_out (int): desired number of clusters in consensus clustering
        n_init (int): Number of time the K-means algorithm will be run with different centroid seeds on the adjusted BA. 
        The final results will be the best output of n_init consecutive runs in terms of SSE.
        random_state (number): Determines random number generation for centroid initialization in K-means.
        verbose (bool): iff true, return the adjusted binary association matrix
    Returns:
        ndarray of shape (N,): consensus clustering
    """
    adjusted_ba_matrix = adjusted_ba(Y)
    labels = KMeans(n_clusters=k_out, n_init=n_init,
                    random_state=random_state).fit_predict(adjusted_ba_matrix)
    return [labels] + [adjusted_ba_matrix] if verbose else labels


qmi = adjusted_ba_kmeans
