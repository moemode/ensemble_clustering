import unittest
import numpy as np

from ens_clust.datastructures.datastructures import adjusted_ba_from_ba, binary_association_matrix, \
    cluster_coassociation_matrix


class TestDs(unittest.TestCase):

    def test_ba(self):
        ba = binary_association_matrix(
            np.array([[0, 0, 1, 1, 2, 3, 4, 2, 3, 4]]).T).toarray()
        desired_ba = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0,
                                                 0], [0, 1, 0, 0, 0],
                               [0, 1, 0, 0, 0], [0, 0, 1, 0,
                                                 0], [0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 1], [0, 0, 1, 0, 0],
                               [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
        np.testing.assert_array_equal(ba, desired_ba)

    def test_select_single_outcome_probabilities(self):
        # 4 points and 2 partitions in ensemble the first has k=2, the second k=3, total #clusters = 5
        Y = np.array([[1, 1], [0, 2], [0, 0], [1, 2]])
        correct_ba = np.array([[0, 1, 0, 1, 0], [1, 0, 0, 0, 1],
                               [1, 0, 1, 0, 0], [0, 1, 0, 0, 1]])
        res = binary_association_matrix(Y).toarray()
        np.testing.assert_array_equal(res, correct_ba)

    def test_cluster_coassociation_matrix(self):
        """Test cluster contingency matrix generation on example of label assignment of 
        Boongoen and Iam-On, “Cluster Ensembles.” p.6. Beware that we use zero based indexing, while
        they use 1-based indexing. Also the contingency matrix in the paper is not the correct one for
        the label assignment in Fig. 2
        """
        correct_cluster_coassoc_matrix = np.array([[1, 1], [1, 1], [0, 1]])
        Y0 = np.array([[0, 0, 1, 1, 2]]).T
        Y1 = np.array([[1, 2, 1, 2, 2]]).T
        Y = np.hstack((Y0, Y1))
        result = cluster_coassociation_matrix(Y).toarray()
        np.testing.assert_array_equal(result, correct_cluster_coassoc_matrix)

    def test_adjusted_ba_from_ba(self):
        """Compare the computed adjusted BA matrix against desired_adjusted_ba, the correct one calculated by hand.
        The ensemble is in Y.
        """
        Y_0 = np.array([[0, 0, 1, 1, 2]])
        Y_1 = np.array([[0, 1, 0, 1, 1]])
        Y = np.hstack((Y_0.T, Y_1.T))
        desired_adjusted_ba = np.array([[0.6, -0.4, -0.2, 0.6, -0.6],
                                        [0.6, -0.4, -0.2, -0.4, 0.4],
                                        [-0.4, 0.6, -0.2, 0.6, -0.6],
                                        [-0.4, 0.6, -0.2, -0.4, 0.4],
                                        [-0.4, -0.4, 0.8, -0.4, 0.4]])
        adjusted_ba_result = adjusted_ba_from_ba(binary_association_matrix(Y))
        np.testing.assert_array_equal(adjusted_ba_result, desired_adjusted_ba)


if __name__ == '__main__':
    unittest.main()
