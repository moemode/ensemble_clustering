import random
from enum import Enum, auto
import numpy as np
from numpy import ndarray
from information_bottleneck.information_bottleneck_algorithms.aIB_class import aIB
from ens_clust.datastructures.datastructures import binary_association_matrix
from ens_clust.utils.utils import min_k_partition_indices, normalize_columns, partition_entropies, \
    permutation_bipartite_matching


class VoteMode(Enum):
    CVOTE = auto()
    BVOTE = auto()


def ada_cvote(Y: ndarray, k: int) -> ndarray:
    """
    Args:
        Y (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
        k (int): desired number of clusters in consensus clustering
        Algorithm is desribed in Ayad, Hanan. “Voting-Based Consensus of Data Partitions,” 2008. 
        http://hdl.handle.net/10012/3934.
    Returns:
        ndarray of shape (N, ): consensus clustering
    """
    return ada_vote(Y, k, VoteMode.CVOTE)


def ada_bvote(Y: ndarray, k: int) -> ndarray:
    """
    Args:
        Y (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
        k (int): desired number of clusters in consensus clustering
        Algorithm is desribed in Ayad, Hanan. “Voting-Based Consensus of Data Partitions,” 2008. 
        http://hdl.handle.net/10012/3934.
    Returns:
        ndarray of shape (N,): consensus clustering
    """
    return ada_vote(Y, k, VoteMode.BVOTE)


def ada_vote(Y: ndarray, k: int, mode: VoteMode) -> ndarray:
    """
    Implements Cumulative Voting by Ayad.
    Algorithm is desribed in Ayad, Hanan. “Voting-Based Consensus of Data Partitions,” 2008. 
    http://hdl.handle.net/10012/3934.
    Args:
        Y (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
        k (int): desired number of clusters in consensus clustering
        mode (VoteMode): iff CVOTE solve LCP by least squares else if BVOTE solve by bipartite matching
    Raises:
        ValueError: if no partition / clustering in ensemble has at least k clusters
    Returns:
        ndarray of shape (N,): consensus clustering
    """
    # Step 1: Reference Selection & Aggregation
    # Aggregation means Relabeling and incorporating into reference
    result, _ = ada_vote_aggregate(Y, k, mode, False, False)
    # Step 2: Extraction of consensus clustering
    _, k_aggregated = result.shape
    if k_aggregated == k:
        return vote_extract_map(result)
    if k_aggregated > k:
        IB_algo = aIB(p_x_y_=result.T.toarray(), card_T_=k)
        IB_algo.run_IB_algo()
        p_t_given_y, p_x_given_t, p_t = IB_algo.get_results()
        # p_t has counts how often a value of t appears, convert them to actual frequencies/probabilities
        real_p_t = (p_t / p_t.sum()).reshape(k, 1)
        # p_x_and_t has shape (k, N)
        p_x_and_t = p_x_given_t * real_p_t
        # find max for every point i.e. column, set axis=0
        return vote_extract_map(p_x_and_t, axis=0)
    raise ValueError("k_aggregated (" + str(k_aggregated) + ") < k (" +
                     str(k) + "). This should never happen.")


def ada_vote_aggregate(Y,
                       reference_k_min: int,
                       mode: VoteMode = VoteMode.CVOTE,
                       return_mse=False,
                       return_intermediate=False):
    """
    Adaptively select reference and aggregate other partitions / clusterings into it.
    Args:
        Y (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
        reference_k_min (int): minimum number of clusters reference must have.
        mode (VoteMode): iff CVOTE solve LCP by least squares else if BVOTE solve by bipartite matching
        return_mse (bool, optional): see vote_aggregate. Defaults to Falase.
        return_intermediate (bool, optional): see vote_aggregate. Defaults to False.

    Returns:
        ndarray of shape (N, k0): Soft clustering of N points into k0 clusters, where k0
            is number of clusters in reference. 
    """
    # sorting negated values increasingly is the same as sorting original values decreasingly
    decreasing_entropy_indices = list(np.argsort(-partition_entropies(Y)))
    reference_index = ada_select_reference(Y, reference_k_min,
                                           decreasing_entropy_indices)
    decreasing_entropy_indices.remove(reference_index)
    """
    by having the reference_index as first index in the aggregation order,
    the first iteration of vote_aggregate's main loop relabels 
    the highest suitable entropy partition according to itself.
    This hacky no-op allows us to use the same vote implementation for ada-cvote as for bvote.
    """
    decreasing_entropy_indices.insert(0, reference_index)
    return vote_aggregate(Y, mode, reference_index, decreasing_entropy_indices,
                          return_mse, return_intermediate)


def ada_select_reference(Y: np.ndarray, reference_k_min: int,
                         decreasing_entropy_indices: np.ndarray) -> int:
    """
    Selects clustering / partition with maximum entropy out of clusterings with more than reference_k_min 
    clusters as reference.
    Args:
        Y (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
        reference_k_min (int): minimum number of clusters reference must have.
        decreasing_entropy_indices (np.ndarray): column indices into Y, 
            correponding to clusterings/partitions in ensemble ordered by decreasing clustering entropy
    Raises:
        ValueError: if no clustering in Y has at least reference_k_min clusters
    Returns:
        int: index of chosen reference
    """
    eligible_partition_indices = min_k_partition_indices(Y, reference_k_min)
    if (eligible_partition_indices.size == 0):
        raise ValueError("You called ada_vote with k=" + str(reference_k_min) +
                         ".\nBut \
            there is no clustering with at least " + str(reference_k_min) +
                         " clusters in the \
            provided ensemble.")
    # go through partitions in order of decreasing entropy and return first that has at least k clusters
    # i.e. of partitions with enough clusters return the one with largest entropy
    for partition_index in decreasing_entropy_indices:
        if partition_index in eligible_partition_indices:
            return partition_index


def vote_aggregate(Y: ndarray,
                   mode: VoteMode,
                   reference_index: int = None,
                   aggregation_order=None,
                   return_mse=False,
                   return_intermediate_results=False):
    """
    Aggregate, i.e. relabel and average partitions of ensemble into reference.
    Args:
        Y (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
        mode (VoteMode): iff CVOTE solve LCP by least squares else if BVOTE solve by bipartite matching
        reference_index (int, optional): reference is Y[:, reference_index]. Defaults to random.
        aggregation_order ([type], optional): Order in which clusterigns / partitions are relabeled and averaged into 
            reference. Defaults to random.
        return_mse (bool, optional): Return sum of MSE between reference and all relabeled clusterings / partitions. 
            Defaults to False.
        return_intermediate_results (bool, optional): iff true, return relabeled partitions and reference in 
            intermediate steps for testing/debugging. Defaults to False.
    Raises:
        ValueError: if mode != CVOTE and mode != BVOTE
        ValueError: if reference_index is not valid column index of Y
    Returns:
        ndarray of shape (N, k0): Soft clustering of N points into k0 clusters, where k0
        is number of clusters in reference. 
    """
    # b in the paper is H here. It stands for total number of partitions in ensemble
    N, H = Y.shape[0], Y.shape[1]
    if (reference_index is None):
        reference_index = random.randint(0, H - 1)
    elif (reference_index < 0 or reference_index > H - 1):
        raise ValueError("The provided reference index: " +
                         str(reference_index) + "is out of bounds")
    if (mode != VoteMode.CVOTE and mode != VoteMode.BVOTE):
        raise ValueError("The provided mode is not supported")
    # if no aggregation order is provided a random one is generated. The final result depends on the order.
    if (aggregation_order is None):
        partition_indices = list(range(H))
        random.shuffle(partition_indices)
        aggregation_order = partition_indices
    # work with BA of refernce initially, after first round reference is soft clustering
    reference = binary_association_matrix(Y[:, reference_index].reshape(N, 1))
    Vs = []
    intermediate_results = []
    for (i, partition_index) in enumerate(aggregation_order):
        # U^i in paper, holds the next partition to be aggregated with the reference
        U_current = binary_association_matrix(Y[:,
                                                partition_index].reshape(N, 1))
        if (mode == VoteMode.CVOTE):
            # least squares for relabeling
            W_current = normalize_columns(U_current).T.dot(reference)
        elif (mode == VoteMode.BVOTE):
            # bipartite matching for relabeling
            W_current = permutation_bipartite_matching(reference, U_current)
        V_current = U_current.dot(W_current)
        Vs.append(V_current)
        reference = (i / (i + 1)) * reference + V_current / (i + 1)
        if return_intermediate_results:
            intermediate_results.append((W_current, V_current, reference))
    # end of calculation, calculate optional return values
    optional_return_values = []
    mse = 0
    if (return_mse):
        for V in Vs:
            mse += (reference - V).power(2).sum() / N
        optional_return_values.append(mse)
    if return_intermediate_results:
        optional_return_values.append(intermediate_results)
    return (reference, optional_return_values)


def vote_extract_map(reference: ndarray, axis: int = 1) -> ndarray:
    """
    Converst soft clustering to hard clustering by assigning points to
    maximally associated cluster.
    Args:
        reference (ndarray of shape (N, k0)): A soft assignment of N points into k0 clusters
        axis (int, optional): argmax axis. Defaults to 1.

    Returns:
        ndarray of shape (N,): with entries in 0..k0-1. Hard clustering of N points to
        cluster it is most strongly associated with in reference
    """
    return np.asarray(reference.argmax(axis=axis)).flatten()
