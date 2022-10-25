import unittest
import numpy as np

from ens_clust.datastructures import binary_association_matrix
from ens_clust.utils.utils import permutation_bipartite_matching


class BipartiteMatchingTest(unittest.TestCase):

    U_0 = np.array([0, 0, 1, 1, 2, 3, 4, 2, 3, 4])
    U_1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    def test_bipartite_matching(self):
        U_0 = binary_association_matrix(
            np.array([[0, 0, 0, 1, 2, 2, 4, 2, 3, 4]]).T)
        U_1 = binary_association_matrix(
            np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]).T)
        #the 0 label of U_1 should be mapped to 0 of U_0 and the 1 label of U1 should be labeled to the 2 label of U1
        desired_W = np.array([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]])
        desired_relabeled_U_1 = np.vstack(
            (np.repeat([[1, 0, 0, 0, 0]], 4,
                       axis=0), np.repeat([[0, 0, 1, 0, 0]], 6, axis=0)))
        """ The following code fragment is used in the BVOTE variant of vote
        """
        W = permutation_bipartite_matching(U_0,
                                           U_1,
                                           return_permutation_matrix=True)
        relabeled_U_1 = U_1.dot(W)
        np.testing.assert_array_equal(W.toarray(), desired_W)
        np.testing.assert_array_equal(relabeled_U_1.toarray(),
                                      desired_relabeled_U_1)

    def test_bipartite_matching2(self):
        U_0 = binary_association_matrix(
            np.array([[0, 0, 0, 1, 2, 3, 3, 2, 3, 4]]).T)
        U_1 = binary_association_matrix(
            np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]).T)
        #the 0 label of U_1 should be mapped to 0 of U_0 and the 1 label of U1 should be labeled to the 3 label of U1
        desired_W = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]])
        #desired_relabeled_U_1 = binary_association_matrix(np.array([[0, 0, 0, 0, 3, 3, 3, 3, 3, 3]]).T).toarray()
        desired_relabeled_U_1 = np.vstack(
            (np.repeat([[1, 0, 0, 0, 0]], 4,
                       axis=0), np.repeat([[0, 0, 0, 1, 0]], 6, axis=0)))
        """ The following code fragment is used in the BVOTE variant of vote
        """
        W = permutation_bipartite_matching(U_0,
                                           U_1,
                                           return_permutation_matrix=True)
        relabeled_U_1 = U_1.dot(W)
        np.testing.assert_array_equal(W.toarray(), desired_W)
        np.testing.assert_array_equal(relabeled_U_1.toarray(),
                                      desired_relabeled_U_1)


if __name__ == '__main__':
    unittest.main()
