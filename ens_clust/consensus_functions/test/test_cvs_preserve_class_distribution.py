import unittest

import numpy as np

from ens_clust.consensus_functions.cvs import ada_vote_aggregate
from ens_clust.consensus_functions.cvs2 import ada_cvote_aggregate_simplified
from ens_clust.datastructures.datastructures import binary_association_matrix
from ens_clust.generation import generate_random_ensemble_cluster_counts
from ens_clust.utils.utils import cluster_distribution, partition_entropies


class CVSClassDistributionTest(unittest.TestCase):

    def test_is_class_dist_preserved_ada_simplified(self):
        """On page 63 of Ayad, Hanan. “Voting-Based Consensus of Data Partitions,” 2008. http://hdl.handle.net/10012/3934.
        Ayad claims that the class distribution of the final result of the cvote consensus function is equal to the class
        distribution of the reference it begins with. She gives a proof of this. Additionally I verify this claim here empirically.
        H random clusterings of N points are generated. Every clustering can have from 2 to max_cluster_count classes/clusters.
        This is decided randomly. We verify this for multiple runs and find that it holds indeed because the test passes.
        This test is for the ada_cvote_simplified implementation.
        """
        for i in range(10):
            N, H = 100, 5
            max_cluster_count = 6
            cluster_counts = np.random.randint(2, max_cluster_count, size=H)
            Y = generate_random_ensemble_cluster_counts(N, H, cluster_counts)
            entropies = partition_entropies(Y)
            #sorting negated values increasingly is the same as sorting original values in decreasing order
            decreasing_entropy_indices = np.argsort(-entropies)
            max_entropy_index = decreasing_entropy_indices[0]
            moving_reference_partition = binary_association_matrix(
                Y[:, max_entropy_index].reshape(N, 1)).toarray()
            #print(moving_reference_partition)
            reference_cluster_dist = cluster_distribution(
                moving_reference_partition)
            #ada_result = ada_cvote(Y, 3)[0].toarray()
            ada_simplified_result = ada_cvote_aggregate_simplified(Y)
            result_dist = cluster_distribution(ada_simplified_result)
            np.testing.assert_array_almost_equal(reference_cluster_dist,
                                                 result_dist)

    def test_is_class_dist_preserved_ada(self):
        """On page 63 of Ayad, Hanan. “Voting-Based Consensus of Data Partitions,” 2008. http://hdl.handle.net/10012/3934.
        Ayad claims that the class distribution of the final result of the cvote consensus function is equal to the class
        distribution of the reference it begins with. She gives a proof of this. Additionally I verify this claim here empirically.
        H random clusterings of N points are generated. Every clustering can have from 2 to max_cluster_count classes/clusters.
        This is decided randomly. We verify this for multiple runs and find that it holds indeed because the test passes.
        This test is for the ada_cvote implementation.
        """
        #for i in range(100):
        for i in range(10):
            N, H = 100, 5
            max_cluster_count = 6
            cluster_counts = np.random.randint(2, max_cluster_count, size=H)
            Y = generate_random_ensemble_cluster_counts(N, H, cluster_counts)
            entropies = partition_entropies(Y)
            #sorting negated values increasingly is the same as sorting original values in decreasing order
            decreasing_entropy_indices = np.argsort(-entropies)
            max_entropy_index = decreasing_entropy_indices[0]
            moving_reference_partition = binary_association_matrix(
                Y[:, max_entropy_index].reshape(N, 1)).toarray()
            #print(moving_reference_partition)
            reference_cluster_dist = cluster_distribution(
                moving_reference_partition)
            ada_result = ada_vote_aggregate(Y, 0)[0].toarray()
            result_dist = cluster_distribution(ada_result)
            np.testing.assert_array_almost_equal(reference_cluster_dist,
                                                 result_dist)


if __name__ == '__main__':
    unittest.main()
