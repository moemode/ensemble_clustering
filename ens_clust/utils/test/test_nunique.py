import unittest

import numpy as np

from ens_clust.generation import generate_random_ensemble_cluster_counts
from ens_clust.utils.utils import nunique


class UtilsTest(unittest.TestCase):

    def test_nunique(self):
        actual_cluster_counts = [7, 3, 5, 10, 2]
        ens = generate_random_ensemble_cluster_counts(100, 5,
                                                      actual_cluster_counts)
        computed_cluster_counts = nunique(ens)
        np.testing.assert_array_equal(actual_cluster_counts,
                                      computed_cluster_counts)


if __name__ == '__main__':
    unittest.main()
