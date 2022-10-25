from typing import Callable, Iterable, List

from .adjusted_ba_kmeans import adjusted_ba_kmeans as qmi
from .cvs import ada_bvote, ada_cvote
from .ivc import iterative_voting_consensus
from .mixture_model import MM

all_consensus_functions: List[Callable] = [
    qmi, iterative_voting_consensus, ada_cvote, ada_bvote, MM
]
all_consensus_functions_aml: Iterable[Callable] = all_consensus_functions
all_consensus_functions_no_vote: Iterable[Callable] = [
    qmi, iterative_voting_consensus, MM
]
all_all_consensus_functions: Iterable[Callable] = [
    qmi, iterative_voting_consensus, ada_cvote, MM
]
all_no_vote: Iterable[Callable] = [qmi, iterative_voting_consensus, MM]
all_no_vote_no_ivc: Iterable[Callable] = [qmi, MM]

consensus_functions_display_names = {
    qmi.__name__: "QMI",
    iterative_voting_consensus.__name__: "IVC",
    ada_cvote.__name__: "A-CV",
    ada_bvote.__name__: "A-BV",
    MM.__name__: "MM",
}

consensus_functions_display_names_to_function_names = {
    "QMI": qmi.__name__,
    "IVC": iterative_voting_consensus.__name__,
    "ADA-CVOTE": ada_cvote.__name__,
    "ADA-BVOTE": ada_bvote.__name__,
    "MM": MM.__name__,
}
