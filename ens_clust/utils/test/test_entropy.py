import unittest

import numpy as np

from ens_clust.utils.utils import partition_entropies, partition_entropy


class EntropyTest(unittest.TestCase):

    U_0 = np.array([0, 0, 1, 1, 2, 3, 4, 2, 3, 4])
    U_1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    def test_H(self):
        Y_0 = np.array([[1, 2, 3, 4]]).T
        Y_1 = np.array([[1, 1, 1, 1]]).T
        Y = np.hstack((Y_0, Y_1))
        H = partition_entropies(Y)
        np.testing.assert_almost_equal(H[0], partition_entropy(Y_0))
        np.testing.assert_almost_equal(H[1], partition_entropy(Y_1))
        print(partition_entropies(Y))


if __name__ == '__main__':
    unittest.main()
