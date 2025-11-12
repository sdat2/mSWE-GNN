"""
Main training script for the mSWE-GNN Adforce pipeline.

This script ties together all the new components:
1.  Loads the configuration from Hydra via @hydra.main.
    Config is injected into the main() function as 'cfg'.
2.  Finds and splits NetCDF files.
    --- REFACTOR ---
    a.  Supports two modes: random split (default) or explicit split
        by providing .txt files in `cfg.data_params.split_files`.
    b.  Manual holdout (e.g., Katrina) is now defined in config.
    ---
3.  Calculates scaling statistics (mean/std) from the training files.
4.  Uses 'AdforceLazyDataset' to create train/val datasets
    (which now apply the scaling).
5.  Calculates model dimensions based on the 'cfg.features' config.
6.  Instantiates the correct model ('GNNModelAdforce', 'MonolithicMLPModel',
    or 'PointwiseMLPModel') using the calculated dimensions.
7.  Uses the 'DataModule' and 'LightningTrainer' from adforce_train.py to run
    the training loop.
8.  Includes ModelCheckpoint callback for saving best/last models.
9.  Allows resuming from a checkpoint specified in the config.
10. Integrates WandbLogger for experiment tracking.
"""

import glob
import os
import lightning as L
from sklearn.model_selection import train_test_split
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import xarray as xr
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import wandb
from typing import List
import warnings

from mswegnn.utils.adforce_dataset import AdforceLazyDataset, _load_static_data_from_ds
from mswegnn.utils.load import (
    read_config,
)
from mswegnn.models.adforce_models import (
    GNNModelAdforce,
    PointwiseMLPModel,
    MonolithicMLPModel,
)
from mswegnn.training.adforce_train import LightningTrainer, DataModule
from mswegnn.utils.adforce_scaling import compute_and_save_adforce_stats
from mswegnn.utils.adforce_dataset import AdforceLazyDataset


def _load_files_from_split(split_file_path: str, data_dir: str) -> List[str]:
    """
    Loads a YAML file containing a list of basenames, joins them
    with the data_dir, and returns a list of full file paths.

    Args:
        split_file_path (str): Path to the .yaml split file (e.g., "conf/train.yaml").
        data_dir (str): Absolute path to the directory containing the .nc files.

    Returns:
        List[str]: A list of absolute paths to the .nc files for this split.
    """
    abs_split_path = hydra.utils.to_absolute_path(split_file_path)
    if not os.path.exists(abs_split_path):
        raise FileNotFoundError(
            f"Split file not found. Expected at: {abs_split_path}\n"
            f"Check 'data_params.split_files' in your config."
        )

    print(f"Loading split from: {abs_split_path}")
    with open(abs_split_path, "r") as f:
        # Use OmegaConf to load the simple list from YAML
        basenames = OmegaConf.load(f)

        # --- THIS IS THE FIX ---
        # We must check against 'ListConfig', not Python's 'list'
        if not isinstance(basenames, ListConfig):
            raise ValueError(
                f"Split file {abs_split_path} did not contain a valid YAML list."
            )
        # --- END OF FIX ---

    # Create full paths (e.g., /path/to/data_dir/262_GILBERT_1988.nc)
    # We also convert the OmegaConf ListConfig to a standard Python list here
    full_paths = [os.path.join(data_dir, str(basename)) for basename in basenames]

    # Verify that files exist
    if full_paths:
        if not os.path.exists(full_paths[0]):
            print(f"Warning: File {full_paths[0]} not found.")
            print(f"       (Basename: {basenames[0]}, Data Dir: {data_dir})")
            print("       Please check 'machine.data_dir' and split file paths.")

    return full_paths


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training loop initiated by Hydra.
    """

    # --- 1. Setup & Config Resolution ---
    # Resolve all interpolations (e.g., ${paths.raw_dir}) *FIRST*.
    # This is critical for all other paths to work.
    try:
        OmegaConf.resolve(cfg)
    except Exception as e:
        print(f"Error resolving config: {e}")
        print(
            "Check for missing variables or syntax errors in config.yaml and machine/*.yaml"
        )
        return

    # Now that config is resolved, we can safely access values
    L.seed_everything(cfg.models.seed)

    # Get the current working directory (Hydra's output folder)
    hydra_run_dir = os.getcwd()
    print(f"Hydra run directory: {hydra_run_dir}")
    print("--- Resolved Config ---")
    print(OmegaConf.to_yaml(cfg))  # Print the resolved config for debugging
    print("-----------------------")

    # --- 2. Weights & Biases Setup ---
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=cfg.wandb.entity,
        # mode=cfg.wandb.mode,
        save_dir=cfg.machine.wandb_dir,
    )
    # --- BUG FIX: .watch() call moved to after model is created ---

    # Log the *resolved* config to W&B
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # --- 3. Checkpointing Setup ---
    checkpoint_dir = cfg.machine.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{cfg.model_params.model_type}-{{epoch}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    last_model_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{cfg.model_params.model_type}-last-{{epoch:02d}}",
            save_last=True,
        )

    all_callbacks = [best_model_callback, last_model_callback]

    # --- [ THE SUGGESTED ADDITION ] ---
    # Save a copy of the config in the checkpoint directory
    # This makes inference *much* easier later.
    config_save_path = os.path.join(checkpoint_dir, "config.yaml")
    try:
        # Save the *resolved* config
        with open(config_save_path, "w") as f:
            OmegaConf.save(cfg, f)  # Save the resolved DictConfig
        print(f"Saved resolved config to {config_save_path}")
    except Exception as e:
        print(f"Warning: Could not save config copy. Error: {e}")
    # --- [ END OF ADDITION ] ---

    # --- 3.5. Load Data File Lists ---
    data_dir = hydra.utils.to_absolute_path(cfg.machine.data_dir)
    print(f"Loading data splits relative to: {data_dir}")

    if not cfg.data_params.get("split_files"):
        raise ValueError(
            "Configuration must provide 'data_params.split_files' (e.g., train, val paths)"
        )

    split_files_cfg = cfg.data_params.split_files

    train_files = _load_files_from_split(split_files_cfg.train, data_dir)
    val_files = _load_files_from_split(split_files_cfg.val, data_dir)

    print(f"Found {len(train_files)} train files and {len(val_files)} val files.")

    if not train_files:
        raise ValueError(
            "No training files were loaded. Check 'data_params.split_files.train'."
        )
    if not val_files:
        print(
            "Warning: No validation files were loaded. Check 'data_params.split_files.val'."
        )

    # --- 4. Calculate or Load Scaling Statistics ---
    features_cfg = cfg.features
    if cfg.data_params.compute_scaling:
        print("Computing scaling stats from training files...")
        # compute_and_save_adforce_stats(
        #     nc_files=train_files,  # <-- Use the loaded file list
        #     output_stats_path=cfg.data_params.scaling_stats_path,
        #     features_cfg=features_cfg,
        # )
        compute_and_save_adforce_stats(
                train_files, 
                cfg.data_params.scaling_stats_path,
                features_cfg,  # <-- NEW
            )
        print("Scaling stats computed and saved.")
    else:
        print(f"Loading scaling stats from {cfg.data_params.scaling_stats_path}")
        if not os.path.exists(cfg.data_params.scaling_stats_path):
            raise FileNotFoundError(
                f"scaling_stats_path not found: {cfg.data_params.scaling_stats_path}"
                "\nSet data_params.compute_scaling=True to generate it."
            )

    # --- 5. Setup Datasets & DataLoaders ---
    print("Setting up datasets...")
    # --- Train Dataset ---
    train_dataset = AdforceLazyDataset(
        root=cfg.data_params.train_root_path,
        nc_files=train_files,  # <-- Use the loaded file list
        previous_t=cfg.model_params.previous_t,
        scaling_stats_path=cfg.data_params.scaling_stats_path,
        features_cfg=features_cfg,
    )

    # --- Validation Dataset ---
    val_dataset = AdforceLazyDataset(
        root=cfg.data_params.val_root_path,
        nc_files=val_files,  # <-- Use the loaded file list
        previous_t=cfg.model_params.previous_t,
        scaling_stats_path=cfg.data_params.scaling_stats_path,
        features_cfg=features_cfg,
    )



    # 4. Create "lazy" datasets
    # print("Initializing training dataset (this may run .process()...)")
    # # --- REFACTOR: Pass features_cfg to dataset ---
    # train_dataset = AdforceLazyDataset(
    #         root=train_root,
    #         nc_files=train_files,
    #         previous_t=p_t,
    #         scaling_stats_path=train_stats_path,
    #         features_cfg=cfg.features  # <-- PASS THE FEATURES CONFIG
    #     )

    # print("Initializing validation dataset (this may run .process()...)")
    #     # --- REFACTOR: Pass features_cfg to dataset ---
    # val_dataset = AdforceLazyDataset(
    #         root=val_root,
    #         nc_files=val_files,
    #         previous_t=p_t,
    #         scaling_stats_path=train_stats_path,  # <-- PASS THE *TRAIN* STATS
    #         features_cfg=cfg.features  # <-- PASS THE FEATURES CONFIG
    #     )

    # 5. Instantiate Lightning DataModule
    data_module = DataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=cfg.trainer_options.batch_size,
            num_workers=cfg.data_params.num_workers,
        )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")


    # --- 6. Initialize Model ---
    print("Initializing model...")
    model_cfg_dict = dict(cfg.models)  # Make a copy
    models = dict(cfg.model_params)
    model_type = models.pop("model_type")

    # --- Dynamically calculate model dimensions from config ---
    p_t = models["previous_t"]
    num_static_node_features = len(features_cfg.static)
    num_dynamic_node_features = len(features_cfg.forcing)

    # The state can include derived features, so we count them all
    num_current_state_features = len(features_cfg.state)
    if features_cfg.get("derived_state"):
        num_current_state_features += len(features_cfg.derived_state)

    num_node_features = (
        num_static_node_features
        + (num_dynamic_node_features * p_t)
        + num_current_state_features
    )
    num_edge_features = len(features_cfg.edge)

    # Model predicts the delta for the state (which includes derived)
    num_output_features = num_current_state_features

    print(f"Model dimensions calculated from config:")
    print(f"  num_node_features: {num_node_features}")
    print(f"  num_edge_features: {num_edge_features}")
    print(f"  num_output_features: {num_output_features}")

    # Instantiate the underlying model
    # GNNModelAdforce,
    # PointwiseMLPModel,
    # MonolithicMLPModel,

    if model_type == "GNN":
        model = GNNModelAdforce(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_output_features=num_output_features,
            num_static_features=num_static_node_features,  # Pass the calculated count
            **model_cfg_dict,  # This dict includes previous_t
        )

    elif model_type == "MLP":
        model = PointwiseMLPModel(
            num_node_features=num_node_features,
            num_output_features=num_output_features,
            **model_cfg_dict,
        )
    elif model_type == "MonolithicMLP":
        n_nodes_fixed = int(train_dataset.total_nodes)
        if n_nodes_fixed is None:
            raise ValueError(
                "Could not determine n_nodes from train_dataset.total_nodes"
            )
        print(f"Found fixed n_nodes from dataset: {n_nodes_fixed}")
        model = MonolithicMLPModel(
            n_nodes=n_nodes_fixed,
            num_node_features=num_node_features,
            num_output_features=num_output_features,
            **model_cfg_dict,
        )
    else:
        raise ValueError(
            f"Unknown model_type in config: {model_type}. Must be 'GNN', 'MLP', or 'MonolithicMLP'."
        )

    # --- 7. Initialize Lightning Trainer ---
    pl_trainer = LightningTrainer(
        model=model, lr_info=cfg.lr_info, trainer_options=cfg.trainer_options
    )

    # --- 8. Setup Trainer ---

    # --- BUG FIX: Moved .watch() call to *after* model is initialized ---
    # Now we can watch the model, since it exists.
    # We watch pl_trainer.model to get the raw torch model.
    print("Setting up W&B model watch...")
    try:
        wandb_logger.watch(pl_trainer.model, log="all", log_freq=100)
    except Exception as e:
        print(f"Warning: wandb_logger.watch() failed: {e}")
    # --- END BUG FIX ---

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        max_epochs=cfg.trainer_options.max_epochs,
        logger=wandb_logger,
        callbacks=[*all_callbacks, lr_monitor],
        accelerator=cfg.machine.accelerator,
        devices=cfg.machine.devices,
        precision=cfg.trainer_options.precision,
        log_every_n_steps=100,
    )

    # --- 9. Start Training ---
    print("Starting training...")
    trainer.fit(pl_trainer, datamodule=data_module)

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
