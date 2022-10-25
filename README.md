# ens_clust

ens_clust provides the following consensus functions which work on
an ensemble of clusterings.
| Consensus Function | File |
| ---      |  ------  |
| MM  |  [mixture_model.py](/ens_clust/consensus_functions/mixture_model.py)|
| Cumulative Voting: A-CV | [cvs.py](/ens_clust/consensus_functions/cvs.py)|
| Cumulative Voting: A-BV | [cvs.py](/ens_clust/consensus_functions/cvs.py)|
| Iterative Voting Consensus  | [ivc.py](/ens_clust/consensus_functions/ivc.py) |
| QMI |  [adjusted_ba_kmeans.py](/ens_clust/consensus_functions/adjusted_ba_kmeans.py) |

ens_clust contains [multiple, simple ensemble generation strategies](ens_clust/generation/ensemble_generation.py).

## Installation

### Install build-essential and miniconda

1. `sudo apt-get install build-essential`
2. Follow <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>
to install Miniconda.
We will create a Python 3.6 environment but you can install
conda with a higher Python version (tested with Python 3.9).

### Get the code and dependencies

```
git clone https://github.com/moemode/ensemble_clustering
cd ensemble_clustering
# Create and activate conda environment with needed dependencies
conda env create -f environment.yml
conda activate ensemble_clustering
```

### Add ens_clust to environment

To install ens_clust, we create a file in the site packages folder of the ensemble_clustering
environment.
This has the advantage that you can change the code without reinstalling the package.
Edit the file `~/miniconda3/envs/ensemble_clustering/lib/python3.6/site-packages/conda.pth`  
Write into it the absolute path to the ensemble_clustering folder.
These must be absolute paths (do not use ~), for example

```
/home/#username/ensemble_clustering
```

### Dependency: Install ib_base

ib_base is unfortunately neither on PyPi nor available as conda package.
We need to download it manually to a separate folder.

```
git clone https://collaborating.tuhh.de/cip3725/ib_base.git
cd ib_base
python setup.py install
cd ..
```

## Tests

Run the following command in the ensemble_clustering folder to run all tests  
`python -m unittest discover`
