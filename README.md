# MCA_Loc

MCA_Loc is a PyTorch-based model for multi-label protein subcellular localization. It uses precomputed ProtTrans residue embeddings and an improved Light Attention encoder to predict 12 subcellular locations.

This repository includes model code, five fold-specific checkpoints, a test dataset, and an evaluation script.

## Model Overview

The encoder combines dilated depthwise separable convolutions, residual connections, and attention-based pooling. Attention-weighted and max-pooled features are concatenated into a protein representation, followed by a multi-label classification head.

## Repository Structure

```text
.
├── README.md
├── requirement.txt                    # Pinned pip dependencies
└── MCA_Loc/
    ├── environment.yml                # Conda environment
    ├── Data/
    │   ├── test_iprdata.csv           # Dataset S5
    │   └── test.csv                   # Dataset S3
    ├── model/
    │   ├── MCA_Loc_iprdata_fold*.pt   # Five model checkpoints on Dataset S4
    │   └── MCA_Loc_fold*.pt           # Five model checkpoints on Dataset S2
    └── script/
        ├── Encoder.py                 # Encoder modules
        ├── MCA_model.py               # Model and dataset definitions
        └── MCA_inference.py           # Inference script
```

## Installation

Key versions from `environment.yml` and the original OpenFold installation:

| Dependency | Version |
| --- | --- |
| Python | 3.8.16 |
| PyTorch | 1.13.1+cu117 |
| NumPy | 1.24.4 |
| pandas | 2.0.3 |
| scikit-learn | 1.3.2 |
| SciPy | 1.10.1 |
| Transformers | 4.30.1 |
| OpenFold | 1.0.0 |
| Biopython | 1.81 |

The original environment uses Python 3.8.16 and PyTorch 1.13.1 with CUDA 11.7. The environment file was exported on Linux; use Linux or WSL2, remove its machine-specific `prefix` entry, and adjust platform-specific dependencies or package indexes as needed.

From the repository root, run:

```bash
conda env create -f MCA_Loc/environment.yml
conda activate MCA_Loc
```

[`requirement.txt`](requirement.txt) contains the pip dependencies from the YAML file with their original version pins, plus OpenFold 1.0.0. Python and Conda-managed system packages are excluded. OpenFold was installed from local source in the original environment; install the matching source checkout with `python -m pip install -e /path/to/openfold` before installing the remaining dependencies:

```bash
python -m pip install -r requirement.txt
```

CUDA-specific PyTorch/PyG wheels and packages such as `dllogger` may require additional package sources. The original OpenFold source commit is not specified; its version pin alone does not identify that checkout. Installation compatibility has not been verified.

## Usage

Precomputed ProtTrans features are required and are not included in this repository. Update `test_path` and `outpath` in `MCA_Loc/script/MCA_test.py` to match your local paths. The `outpath` value must point to the parent directory of `ProtTrans/` and end with `/`.

From the repository root, run:

```bash
cd MCA_Loc/script
python MCA_inference.py
```

The script loads the five checkpoints from `MCA_Loc/model/`. Training and feature extraction scripts are not included in the current release.
