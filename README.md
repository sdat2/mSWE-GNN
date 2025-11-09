# Simon's Branch of the mSWE-GNN Repository, for using the same models for storm surge emulation

## Introduction

We created additional training data through forcing the ADCIRC model with storm surge scenarios from the ADFORCE python package using data from IBTraCS.

Look at the `generate_training_data.py` script for how we generated the training data:
<https://github.com/sdat2/PotentialHeight/blob/main/adforce/generate_training_data.py>

We have adapted the pytorch geometric based mSWE-GNN code to work with the ADFORCE data structure and to create models that can emulate storm surge events over the North West Atlantic/ Carribean Mesh.

## Environment setup

Using `micromamba` and flexible yaml settings for a robust pure-cpu environment:

```bash
micromamba create -n mswegnn -f env.yml
```

If you want to run on JASMIN GPU nodes, use:

```bash
micromamba create -n mswegnn-gpu -f env_jas_gpu.yml
```

If you are installing the python environment through a different method instead, to locally install the `mswegnn` package, run:

```bash
pip install -e .
```

## ADFORCE pipeline file structure

- `adforce_main.py`: Main script to run the Adforce/SWE-GNN training pipeline.
- `conf/`: Configuration files for hyperparameters and settings.
- `archer2.slurm`: SLURM job script for running on the Archer2 supercomputer.
- `mswegnn/`: All the main code turned into a python package for easier management.
    - `database/`: Data handling and preprocessing modules. (not used).
    - `models/`: Model definitions and architectures.
        - `adforce_gnn.py`: graph neural network layers.
        - `adforce_models.py`: full model architectures.
        - `adforce_helpers.py`: helper functions for models.
    - `training/`: Training routines and loss functions.
        - `adforce_train.py`: training loop and evaluation.
        - `adforce_loss.py`: loss functions.
    - `utils/`: Utility functions for various tasks.
        - `adforce_dataset.py`: dataset loading and batching.
        - `adforce_scaling.py`: data normalization and scaling.
        - `adforce_animate.py`: animate adforce inputs and outputs using dataloader.
- `jasmin.slurm`: SLURM job script for running on the JASMIN supercomputer.
- `env_jas_gpu.yml`: Micromamba environment file for JASMIN GPU nodes.
- `env.yml`: Micromamba environment file for local CPU usage.
- `requirements.txt`: Old original Python package dependencies (doesn't correctly install on most machines).
- `setup.py`: Setup file for locally installing the `mswegnn` package.

## Run instructions
To run the training pipeline, use the following command:

```bash
python -m adforce_main

python -m adforce_main model_params.model_type=MonolithicMLP

python -m adforce_main model_params.model_type=MLP

```

# Old README content:
# mSWE-GNN (Repository for paper "Multi-scale hydraulic graph neural networks for flood modelling")
(Version 1.1 - Nov. 28th, 2024)

![Architecture](Architecture.png)

## Overview

For reproducing the paper's results, explore **plot_results.ipynb**

For training the model run **main.py**

For training and exploring the model, run **main.ipynb**

For testing a model, run **test_model.py**

Both **main.py** and **main.ipynb** use a **config.yaml** as reference configuration file.

The repository is divided in the following folders:

* **database:** Creation of hydrodynamic simulations (**D-Hydro simulations.ipynb**) [requires the license and installation of "D-HYDRO Suite 1D2D"] and conversion of the NETCDF output files into PyTorch Geometric-friendly data (**create_dataset.ipynb**).
Also contains the output of the hydrodynamic simulations (**raw_datasets**: for downloading the datasets go to <https://doi.org/10.5281/zenodo.13326595>). This is converted into Pickle files that are then stored and separated into training and testing datasets in **datasets**.

* **models:**  Deep learning models developed for surrogating the hydraulic one: contains a base class with common inputs and functions and one for the SWE-GNN and mSWE-GNN models.

* **results:** Contains results and trained models of the mSWE-GNN Pareto front, used for the paper's results.

* **training:** Contains loss and training functions.

* **utils:** Contains Python functions for loading, creating and scaling the dataset. There are also other miscellaneous functions and visualization functions.

## Environment setup

The required libraries are in requirements.txt.