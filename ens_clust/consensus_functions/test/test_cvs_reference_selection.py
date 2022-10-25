import unittest

import numpy as np
from numpy.testing import assert_array_equal

from ens_clust.consensus_functions.cvs import ada_select_reference
from ens_clust.generation import generate_random_ensemble_cluster_counts
from ens_clust.utils.utils import min_k_partition_indices


class CVSReferenceSelectionTest(unittest.TestCase):

    #min_k_partition_indices is used by ada_select_reference
    def test_eligible_partition_indices(self):
        cluster_counts = [7, 3, 5, 10, 2]
        ens = generate_random_ensemble_cluster_counts(100, 5, cluster_counts)
        #all partitions have at least k=2
        mink_2_indices = np.arange(5)
        #only partititons 0, 2, 3 have at least k=4
        mink_4_indices = np.array([0, 2, 3])
        mink_10_indices = np.array([3])
        min_k_11_indices = np.array([])
        min_k_indices = [(2, mink_2_indices), (4, mink_4_indices),
                         (10, mink_10_indices), (11, min_k_11_indices)]
        for (mink, correct_indices) in min_k_indices:
            assert_array_equal(correct_indices,
                               min_k_partition_indices(ens, mink))

        #min_k_partition_indices is used by ada_select_reference
    def test_reference_selection(self):
        cluster_counts = [7, 3, 5, 10, 2]
        ens = generate_random_ensemble_cluster_counts(100, 5, cluster_counts)
        #assume 5-th partition has highest entropy, third second-highest...
        made_up_decreasing_entropy_indices = [4, 2, 3, 0, 1]
        mink_2_indices = np.arange(5)
        #only partititons 0, 2, 3 have at least k=4
        mink_4_indices = np.array([0, 2, 3])
        mink_10_indices = np.array([3])
        legal_min_k_indices = [(2, mink_2_indices, 4), (4, mink_4_indices, 2),
                               (10, mink_10_indices, 3)]
        for (min_k, mink_indices, highest_entropy_idx) in legal_min_k_indices:
            assert_array_equal(
                highest_entropy_idx,
                ada_select_reference(ens, min_k,
                                     made_up_decreasing_entropy_indices))
        # There are no partitions with at least 11 clusters, so no reference can be selected
        self.assertRaises(ValueError, ada_select_reference, ens, 11,
                          made_up_decreasing_entropy_indices)


if __name__ == '__main__':
    unittest.main()
