import unittest

import numpy as np

from ens_clust.datastructures import binary_association_matrix
from ens_clust.utils.utils import cluster_distribution


class UtilsTest(unittest.TestCase):

    def test_cluster_distribution0(self):
        U_0 = np.array([0, 0, 1, 1, 2, 3, 4, 2, 3, 4]).reshape(10, 1)
        desired_distribution = np.array([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5])
        computed_distribution = cluster_distribution(
            binary_association_matrix(U_0))
        np.testing.assert_almost_equal(computed_distribution,
                                       desired_distribution)

    def test_cluster_distribution0_toarray(self):
        U_0 = np.array([0, 0, 1, 1, 2, 3, 4, 2, 3, 4]).reshape(10, 1)
        desired_distribution = np.array([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5])
        computed_distribution = cluster_distribution(
            binary_association_matrix(U_0).toarray())
        np.testing.assert_almost_equal(computed_distribution,
                                       desired_distribution)

    def test_cluster_distribution1(self):
        U_1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).reshape(10, 1)
        desired_distribution = np.array([2 / 5, 3 / 5])
        computed_distribution = cluster_distribution(
            binary_association_matrix(U_1))
        np.testing.assert_almost_equal(computed_distribution,
                                       desired_distribution)

    def test_cluster_distribution1_toarray(self):
        U_1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).reshape(10, 1)
        desired_distribution = np.array([2 / 5, 3 / 5])
        computed_distribution = cluster_distribution(
            binary_association_matrix(U_1).toarray())
        np.testing.assert_almost_equal(computed_distribution,
                                       desired_distribution)


if __name__ == '__main__':
    unittest.main()
