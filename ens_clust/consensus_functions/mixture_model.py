import numpy as np
from numpy import ndarray
from typing import Tuple


def MM(Y: ndarray, k_out: int, iterations=100, attempts=2) -> ndarray:
    """
    Given the label matrix Y of an ensemble with N points and H base clusterings,
    compute consensus_clustering with k_out clusters.
    Run MM_single attempts times and return 'best' clustering.
    Implementation of Topchy et. al. “A Mixture Model for Clustering Ensembles”. https://doi.org/10.1137/1.9781611972740.35.
    Args:
        Y (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
        k_out (int): desired number of clusters in consenus clustering
        iterations (int, optional): Maximum number of iterations of EM algorithm. Defaults to 100.
        attempts (int, optional): Number of times the EM algorithm is started with random initialization. Defaults to 2.
    Raises:
        ValueError: if Y (ensemble) contains more than 300 clusterings
        ValueError: if div by zero occured because of numerical issues that occur for large number of clusterings
    Returns:
        ndarray of shape (N,): consensus clustering with highest log likelihood obtained in attempts runs
    """
    _, H = Y.shape[0], Y.shape[1]
    # if there are more than 300 partitions there is no hope of escaping
    if (H > 300):
        raise ValueError(
            "Running MM with more than 300 partitions in the ensemble will\
            inevitably lead to numerical problems, preventing a result")
    failed_attempts = 0
    # likelihoods of MM_single attempts are negative
    best_likelihood = np.NINF
    best_result = ()
    for i in range(attempts):
        try:
            current_result = (MM_single(Y, k_out, iterations))
            # the 4-th entry contains the likelihoods over the full run, get the likelihood of the last iteration
            current_final_likelihood = current_result[3][-1]
            if (current_final_likelihood > best_likelihood):
                best_result = current_result
        except ValueError:
            failed_attempts = failed_attempts + 1
    if attempts == failed_attempts:
        raise ValueError("All MM iterations failed due to numerical issues.\n\
        The number of partitions (" + str(H) + ") is likely too large")
    assignment, _, _, _, _ = best_result
    return assignment


def MM_single(Y: ndarray, k_out: int, iterations: int = 100):
    """
    Given the label matrix Y of an ensemble with N points and H base clusterings,
    compute consensus_clustering with k_out clusters.
    Implementation of Topchy et. al. “A Mixture Model for Clustering Ensembles”.
    https://doi.org/10.1137/1.9781611972740.35.
    See BA p. 34, 35
    In contrast to MM_iterative this implementation avoids looping over the N points.
    Args:
        Y (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
        k_out (int): desired number of clusters in consenus clustering
        iterations (int, optional): maximum iterations of EM algorithm if no convergence beforehand. Defaults to 100.
    Returns:
        ndarray of shape (N,): consensus clustering, additional values for debugging
    """
    # use M as alias for k_out because paper does
    M = k_out
    N, H = Y.shape[0], Y.shape[1]
    # k_in_max is largest number of clusters appearing in an input partition
    k_in_max = Y.max() + 1
    parameters = []
    # initialization, Ez is initialized randomly and normalized
    Ez = np.random.uniform(size=(N, M))
    Ez = Ez / Ez.sum(axis=1)[:, np.newaxis]
    # alpha and outcome_probabilities are calculated based on the randomly initialized Ez
    alpha = Ez.sum(axis=0) / Ez.sum()
    outcome_probabilities = init_outcome_probabilities(M, H, Y, Ez, k_in_max)
    # initial parameters are saved to feed to MM_iterative for reproduction of results
    initial_parameters = copy_parameters(alpha, outcome_probabilities, Ez)
    parameters.append(initial_parameters)
    likelihoods = np.zeros(iterations)
    likelihoods[0] = np.sum(np.log(Ez.sum(axis=1)))
    # precompute for every cluster label a HxN point membership matrix
    points_for_partitions = np.empty((k_in_max, *(Y.T.shape)), dtype=np.bool)
    for k in range(k_in_max):
        points_for_partitions[k] = (Y == k).T
    # START of Algorithm: E and M steps iterated in loop
    for it in range(iterations):
        # 1. E-Step: compute Ez ndarray of shape (N, M)
        for m in range(M):
            Ez[:, m] = alpha[m] * select_outcome_probabilities(
                outcome_probabilities[m], Y).prod(axis=1)
        # Sum up logs of rowsums for all rows. This is the log likelihood. Can be used as internal quality measure.
        rowsums = Ez.sum(axis=1)
        if 0 in rowsums:
            raise ValueError("The number of partitions (H=" + str(H) +
                             ") is too large,\
                which leads to numerical issues")
        likelihoods[it] = (np.sum(np.log(rowsums)))
        Ez = Ez / Ez.sum(axis=1)[:, np.newaxis]
        if (it > 0 and likelihoods[it - 1] == likelihoods[it]):
            break
        # 2. M-Step: update alpha
        alpha = Ez.sum(axis=0) / Ez.sum()
        # 3. M-step: update outcome_probabilities
        for k in range(k_in_max):
            for j in range(H):
                outcome_probabilities[:, j,
                                      k] = Ez[points_for_partitions[k, j]].sum(
                                          axis=0)
        for m in range(M):
            outcome_probabilities[m] = outcome_probabilities[m] / Ez[:,
                                                                     m].sum()
        parameters.append(copy_parameters(alpha, outcome_probabilities, Ez))
    # get ndarray of shape (N,) assigning a label from 0..M-1 to each point
    assignment = Ez.argmax(axis=1)
    return assignment, (Ez, alpha,
                        outcome_probabilities), parameters, likelihoods[:it +
                                                                        1], it


def copy_parameters(alpha: ndarray, outcome_probabilities: ndarray,
                    Ez: ndarray) -> Tuple:
    """
    Copy of intermediate results in MM_single
    Args:
        alpha (ndarray)
        outcome_probabilities (ndarray)
        Ez (ndarray)
    Returns:
        Tuple: containing copies of the input parameters
    """
    return (np.copy(alpha), np.copy(outcome_probabilities), np.copy(Ez))


def init_outcome_probabilities(M: int, H: int, Y: ndarray, Ez: ndarray,
                               k_in_max: int):
    """
    Initialize outcome probabilities based on random assignment of points to mixture components.
    Args:
        M (int): number of mixture components. It equals the number of desired clusters in consensus clustering.
        H (int): number of partitions / clusterings in ensemble
        Y (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
        Ez (ndarray of shape (N, M)): assignment of N points to M (=k_out) mixture components
        k_in_max (int): largest number of clusters that any clustering in ensemble has
    Returns:
        ndarray of shape (M, H, k_in_max): stores outcome probabilities of M multinomials.
            A multinomial has one outcome probability for each clustering in the ensemble.
            Example: The outcome probability of the m-th multinomial for the 3rd clustering in the
            2nd base clustering is in outcome_probabilities[m][2][3]
    """
    outcome_probabilities = np.empty((M, H, k_in_max))
    for k in range(k_in_max):
        points_for_partitions = (Y == k).T
        for j in range(H):
            outcome_probabilities[:, j,
                                  k] = Ez[points_for_partitions[j]].sum(axis=0)
    for m in range(M):
        outcome_probabilities[m] = outcome_probabilities[m] / Ez[:, m].sum()
    return outcome_probabilities


def MM_iterative(Y: ndarray,
                 cluster_counts: ndarray,
                 k_out: int,
                 init: Tuple,
                 iterations=100):
    """
    Loop-based implementation of Topchy et. al. “A Mixture Model for Clustering Ensembles”.
    https://doi.org/10.1137/1.9781611972740.35. See BA p. 34, 35.
    Used to compare against MM_single which avoids looping over all N points.
    Args:
        Y (ndarray of shape (N, H)): label matrix, Y[i][j]=k iff i-th point belongs to k-th cluster in j-th partition
        cluster_counts (ndarray): [description]
        k_out (int): desired number of clusters in consenus clustering
        init (Tuple): contains initial alpha, outcome_probabilities, Ez. Used to compare deterministically
            against MM_single.
        iterations (int, optional): maximum iterations of EM algorithm if no convergence beforehand. Defaults to 100.
    Returns:
        ndarray of shape (N,): consensus clustering, additional values for debugging
    """
    alpha, outcome_probabilities, _ = init
    M = k_out
    N, H = Y.shape[0], Y.shape[1]
    Ez = np.empty((N, M))
    for it in range(iterations):
        # 1. E-Step: compute Ez
        for i in range(N):
            for m in range(M):
                p = alpha[m]
                for j in range(H):
                    p *= outcome_probabilities[m][j][Y[i, j]]
                Ez[i][m] = p
        Ez = Ez / Ez.sum(axis=1)[:, np.newaxis]
        # 2. M-Step: compute alpha
        Ez_tot = Ez.sum()
        for m in range(M):
            s = 0
            for i in range(N):
                s += Ez[i][m]
            alpha[m] = s / Ez_tot
        # 3. M-Step: compute output probabilities
        for m in range(M):
            for j in range(H):
                for k in range(cluster_counts[j]):
                    s = 0
                    for i in range(N):
                        s += Ez[i][m] if Y[i][j] == k else 0
                    d = 0
                    for i in range(N):
                        for c in range(cluster_counts[j]):
                            d += Ez[i][m] if Y[i][j] == c else 0
                    outcome_probabilities[m][j][k] = s / d
    assignment = Ez.argmax(axis=1)
    return assignment, (Ez, alpha, outcome_probabilities)


def select_outcome_probabilities(outcome_probabilities_single, Y):
    """
    Returns probabilities an ndarray of shape (H,N) with
    probabilities[i] = outcome_probabilities[i][y[i]] for i in range(H) 
    Args:
        outcome_probabilities (ndarray of shape (H, k)): [description]
        Y (ndarray of shape (N, H)): stacked column label vectors
    """
    return np.take_along_axis(outcome_probabilities_single.T, Y, axis=0)
