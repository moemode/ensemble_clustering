import copy
import unittest

import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans

from ens_clust.consensus_functions.mixture_model import MM, MM_single, MM_iterative, select_outcome_probabilities
from ens_clust.generation import generate_random_ensemble_cluster_counts


class TestMM(unittest.TestCase):
    outcome_probabilities_0 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],
                                        [10, 11, 12]])
    outcome_probabilities_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],
                                        [10, 11, 12]])
    outcome_probabilities = np.array(
        [outcome_probabilities_0, outcome_probabilities_1])

    def test_select_single_outcome_probabilities(self):
        Y = np.array([[1, 0, 2, 1]])
        correct_selection = np.array([[2, 4, 9, 11]])
        res = select_outcome_probabilities(self.outcome_probabilities[0], Y)
        np.testing.assert_array_equal(res, correct_selection)

    def test_select_single_outcome_probabilities2(self):
        Y = np.array([[1, 0, 2, 1], [0, 1, 1, 2]])
        correct_selection = np.array([[2, 4, 9, 11], [1, 5, 8, 12]])
        res = select_outcome_probabilities(self.outcome_probabilities[0], Y)
        np.testing.assert_array_equal(res, correct_selection)

    def test_EM(self):
        Y = np.array([[1, 3, 0, 2], [2, 0, 0, 3]])
        cluster_counts = np.array([3, 4, 1, 4])
        k_out = 2
        MM_single(Y, k_out)

    def test_EM2(self):
        N, H = (3, 2)
        Y = np.array([[0, 2], [1, 0], [0, 1]])
        cluster_counts = np.array([2, 3])
        k_out = 2
        vassignment, (
            vEz, valpha, voutcome_probabilities
        ), parameters, likelihoods, needed_iterations = MM_single(Y, k_out)
        print(likelihoods)
        initial_alpha, initial_outcome_probs, initial_Ez = copy.deepcopy(
            parameters[0])
        # Check that initial parameters used by MM are properly normalized
        self.assertAlmostEqual(initial_alpha.sum(), 1)
        for m in range(k_out):
            for j in range(H):
                self.assertAlmostEqual(initial_outcome_probs[m][j].sum(), 1)
        for i in range(N):
            self.assertAlmostEqual(initial_Ez[i].sum(), 1)

        assignment, (Ez, alpha, outcome_probabilities) = MM_iterative(
            Y, cluster_counts, k_out, parameters[0])
        #Check that results of the vectorized and loop-based implementation agree
        np.testing.assert_almost_equal(Ez, vEz)
        np.testing.assert_almost_equal(alpha, valpha)
        np.testing.assert_almost_equal(outcome_probabilities,
                                       voutcome_probabilities)
        np.testing.assert_equal(vassignment, assignment)

    def test_MM_loop_vs_vec(self):
        #for i in range(100):
        for i in range(3):
            N, H = 100, 10
            max_cluster_count = 6
            cluster_counts = np.random.randint(2, max_cluster_count, size=H)
            Y = generate_random_ensemble_cluster_counts(N, H, cluster_counts)
            #sorting negated values increasingly is the same as sorting original values in decreasing order
            vassignment, (
                vEz, valpha, voutcome_probabilities
            ), parameters, likelihoods, needed_iterations = MM_single(Y, 3)
            assignment, (Ez, alpha, outcome_probabilities) = MM_iterative(
                Y, cluster_counts, 3, parameters[0])
            np.testing.assert_equal(vassignment, assignment)
            #Check that results of the vectorized and loop-based implementation agree
            #np.testing.assert_almost_equal(Ez, vEz)
            #np.testing.assert_almost_equal(alpha, valpha)
            #np.testing.assert_almost_equal(outcome_probabilities, voutcome_probabilities)

    def test_EM3(self):
        N = 1000
        seed = 170
        X, y = datasets.make_moons(n_samples=N,
                                   shuffle=True,
                                   noise=0.02,
                                   random_state=seed)
        H = 100
        ens = np.empty((N, H), dtype=int)
        k_ens = 10
        for j in range(H):
            ens[:, j] = KMeans(n_clusters=k_ens, init="random").fit_predict(X)
        vassignment, (
            vEz, valpha, voutcome_probabilities
        ), parameters, likelihoods, needed_iterations = MM_single(ens, k_out=2)

    def test_EM_many_attempts(self):
        N = 1000
        seed = 170
        X, y = datasets.make_moons(n_samples=N,
                                   shuffle=True,
                                   noise=0.02,
                                   random_state=seed)
        H = 100
        ens = np.empty((N, H), dtype=int)
        k_ens = 10
        for j in range(H):
            ens[:, j] = KMeans(n_clusters=k_ens, init="random").fit_predict(X)
        assignment = MM(ens, k_out=2, attempts=4)


if __name__ == '__main__':
    unittest.main()
