import operator
from functools import reduce

import numpy as np
from scipy import sparse


def membership_matrix(cluster_run):
    # Author: Gregory Giecold for the GC Yuan Lab
    # Affiliation: Harvard University
    # Contact: g.giecold@gmail.com, ggiecold@jimmy.harvard.edu
    """For a label vector represented by cluster_run, constructs the binary 
        membership indicator matrix. Such matrices, when concatenated, contribute 
        to the adjacency matrix for a hypergraph representation of an 
        ensemble of clusterings.

    Parameters
    ----------
    cluster_run : array of shape (n_samples,) or (n_samples, 1)

    Returns
    -------
    A binary matrix such that m[i,j]==1 iff cluster_run[i]=j in compressed sparse row form.
    """

    cluster_run = np.asanyarray(cluster_run)

    if reduce(operator.mul, cluster_run.shape, 1) != max(cluster_run.shape):
        raise ValueError(
            "\nERROR: Cluster_Ensembles: create_membership_matrix: "
            "problem in dimensions of the cluster label vector "
            "under consideration.")
    else:
        cluster_run = cluster_run.reshape(cluster_run.size)

        cluster_ids = np.unique(
            np.compress(np.isfinite(cluster_run), cluster_run))

        indices = np.empty(0, dtype=np.int32)
        indptr = np.zeros(1, dtype=np.int32)

        for elt in cluster_ids:
            indices = np.append(indices, np.where(cluster_run == elt)[0])
            indptr = np.append(indptr, indices.size)

        data = np.ones(indices.size, dtype=int)
        return sparse.csr_matrix((data, indices, indptr),
                                 shape=(cluster_ids.size, cluster_run.size)).T


def binary_association_matrix(partition_label_vectors):
    # Author: Gregory Giecold for the GC Yuan Lab
    # Affiliation: Harvard University
    # Contact: g.giecold@gmail.com, ggiecold@jimmy.harvard.edu
    """[summary]

    Args:
        partition_label_vectors (ndarray (N, H)): [description]
    Returns:
        [type]: [description]
    """
    N, H = partition_label_vectors.shape[0], partition_label_vectors.shape[1]
    partition_label_vectors = partition_label_vectors.T
    ba = membership_matrix(partition_label_vectors[0])
    for i in range(1, H):
        ba = sparse.hstack(
            [ba, membership_matrix(partition_label_vectors[i])], format='csr')
    return ba


def adjusted_ba_from_ba(ba):
    """The adjusted BA is called TMB by Iam-On and Boongoen. 
    QMI consensus clustering consists of computing it and running KMeans on it.
    Topchy described QMI in 2004. We implement QMI under the name adjusted_ba_kmeans.
    Args:
        ba (ndarray of shape (N, R)): A matrix representing a cluster ensemble whoose partitions have R clusters cumulatively. 
        It contains only 0 or 1 entries. ba[i][j]==1 iff point i is in the j-th cluster of an ensemble,
        i in 0..N-1, j in 0..R-1
    Returns:
        adjusted_ba [ndarray of shape (N, R)]: adjusted_ba[i][j] = ba[i][j]/p(j)
        where p(j) is the frequency of assigning cluster label j to a point in the partition of cluster j
    """
    return ba - (np.sum(ba, axis=0) / ba.shape[0]).reshape((1, ba.shape[1]))


def adjusted_ba(Y):
    """See adjusted_ba_from_ba
    Args:
        Y (ndarray (N, H)): [description]
    Returns:
        See adjusted_ba_from_ba
    """
    return adjusted_ba_from_ba(binary_association_matrix(Y))


def point_coassocation_matrix(partition_label_vectors):
    """[summary]

    Args:
        partition_label_vectors (ndarray (N, H)): [description]

    Returns:
        [type]: [description]
    """
    H = partition_label_vectors.shape[1]
    ba = binary_association_matrix(partition_label_vectors)
    return ba.dot(ba.T) / H


def cluster_coassociation_matrix(partition_pair):
    """[summary]
    Args:
        partition_pair (ndarray of shape(N,2)): 
        partition_pair[:, 0] and partition_pair[:, 1] each represent a hard assignment of N points to a number of clusters. 
        The cluster labels must be in range 0..k-1 if there a k clusters in a partition.
    Returns:
        ndarray of shape(clusters_count(partition_pair[:, 0]), cluster_count(partition_pair[:, 1])): 
        cluster_coassociation_matrix[i][j]=count where count is the number of times that a point was
        assigned to cluster i in partition0 and to cluster j in partition1. i in 0..cluster_count0,
        j in 0..cluster_count1
    """
    return membership_matrix(partition_pair[:, 0]).T.dot(
        membership_matrix(partition_pair[:, 1]))
