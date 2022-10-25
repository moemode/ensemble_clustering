import numpy as np
from numpy import ndarray
from ens_clust.datastructures.datastructures import binary_association_matrix
from ens_clust.utils.utils import normalize_columns, partition_entropies

# This simplified, inefficient implementation is used to ensure the more efficient one behaves equivalently


def ada_cvote_aggregate_simplified(Y: ndarray) -> ndarray:
    """Implemented according to pseudo-code in Ayad, Hanan G., and Mohamed S. Kamel. 
    “On Voting-Based Consensus of Cluster Ensembles.” Pattern Recognition 43, no. 5 (May 2010): 1943–53. 
    https://doi.org/10.1016/j.patcog.2009.11.012. (page 1947)
    In contrast to full ada_cvote, no clustering is extracted. Only the final reference is obtained.
    Args:
        Y (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
    Returns:
       ndarray of shape (N, k0): the reference after aggragation. It is a oft clustering of N points into k0 clusters,
       where k0 is the number of clusters in reference. 
    """
    # N is #instances, H is #partitions in ensemble
    N, _ = Y.shape[0], Y.shape[1]
    entropies = partition_entropies(Y)
    # sorting negated values increasingly is the same as sorting original values in decreasing order
    decreasing_entropy_indices = np.argsort(-entropies)
    max_entropy_index = decreasing_entropy_indices[0]
    moving_reference_partition = binary_association_matrix(
        Y[:, max_entropy_index].reshape(N, 1)).toarray()
    for (count, j) in enumerate(decreasing_entropy_indices[1:]):
        current_partition = binary_association_matrix(Y[:, j].reshape(
            N, 1)).toarray()
        current_partition_normalized = normalize_columns(current_partition)
        current_projection = current_partition_normalized.T.dot(
            moving_reference_partition)
        current_partition_projected = current_partition.dot(current_projection)
        # Always: fraction_current + fraction_reference = 1
        # count moves from 0 to H-2, fraction_current from 1/2 to 1/H, fraction_reference from 1/2 to H-1/H
        fraction_current, fraction_reference = (1 /
                                                (count + 2)), ((count + 1) /
                                                               (count + 2))
        moving_reference_partition = fraction_reference * moving_reference_partition + fraction_current * current_partition_projected
    return moving_reference_partition
