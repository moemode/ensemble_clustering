import unittest

import numpy as np

from ens_clust.consensus_functions.cvs import ada_vote_aggregate
from ens_clust.consensus_functions.cvs2 import ada_cvote_aggregate_simplified
from ens_clust.generation import generate_random_ensemble


class CompareCVSTest(unittest.TestCase):

    def test_compare_ada_cvotes(self):
        for i in range(10):
            Y = generate_random_ensemble(10, 3, 3)
            ada_result = ada_vote_aggregate(Y, 0)[0].toarray()
            ada_simplified_result = ada_cvote_aggregate_simplified(Y)
            np.testing.assert_array_almost_equal(ada_result,
                                                 ada_simplified_result)


if __name__ == '__main__':
    unittest.main()
