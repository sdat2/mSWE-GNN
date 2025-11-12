"""
Main training script for the mSWE-GNN Adforce pipeline.

This script ties together all the new components:
1.  Loads the configuration from Hydra via @hydra.main.
    Config is injected into the main() function as 'cfg'.
2.  Finds and splits NetCDF files.
3.  Calculates scaling statistics (mean/std) from the training files.
4.  Uses 'AdforceLazyDataset' to create train/val datasets
    (which now apply the scaling).
    --- REFACTOR ---
    a.  Passes the 'cfg.features' object to the dataset.
    b.  The dataset is now responsible for loading/assembling features
        based on the config.
    ---
5.  Calculates model dimensions based on the 'cfg.features' config.
    (This removes all hard-coded feature counts).
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
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch
import xarray as xr
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

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


# --- REFACTOR: All hard-coded feature constants removed ---
# The feature dimensions are now calculated from the config.


def print_tensor_size_mb(tensor_dict):
    """Helper to print the size of tensors in a dictionary."""
    total_size = 0
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor):
            size = v.element_size() * v.nelement()
            total_size += size
            print(f"  - Static tensor '{k}': {size / (1024**2):.2f} MB")
    print(f"  --- TOTAL STATIC DATA SIZE: {total_size / (1024**2):.2f} MB ---")


torch.set_float32_matmul_precision("medium") # try get higher performance with Tensor Cores


@hydra.main(
    config_path="conf", config_name="config", version_base=None
)
def main(cfg: DictConfig):
    """
    Main function to run the data loading and training pipeline.
    """
    # 1. Load Configuration
    print("--- Configuration (from Hydra) ---")
    print(OmegaConf.to_yaml(cfg))
    print("----------------------------------")
    print(f"Hydra working directory: {os.getcwd()}")

    # Get resume path from machine config
    resume_checkpoint_path = cfg.machine.resume_checkpoint_path
    if resume_checkpoint_path:
        resume_checkpoint_path = hydra.utils.to_absolute_path(resume_checkpoint_path)

    # --- REFACTOR: Get p_t from model_params ---
    p_t = cfg.model_params.previous_t

    # HYDRA: Resolve data_dir relative to original CWD
    data_dir = hydra.utils.to_absolute_path(cfg.machine.data_dir)
    processed_dir = hydra.utils.to_absolute_path(cfg.machine.processed_dir)

    # 2. Find and split data files
    print(f"Searching for NetCDF files in {data_dir}...")
    all_nc_files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    if not all_nc_files:
        print(f"ERROR: No *.nc files found in {data_dir}.")
        print("Please check the 'data_dir' path in your config file.")
        return
    print(f"Found {len(all_nc_files)} total simulation files.")

    # --- MODIFICATION START: 3-way split with manual file reservation ---
    manual_test_file_name = "152_KATRINA_2005.nc"
    katrina_path = None
    for f_path in all_nc_files:
        if f_path.endswith(manual_test_file_name):
            katrina_path = f_path
            break

    if katrina_path is None:
        raise FileNotFoundError(
            f"Could not find required test file {manual_test_file_name} in {data_dir}. "
            f"Please ensure this file exists."
        )
    else:
        print(f"Manually reserving file for test set: {manual_test_file_name}")

    remaining_files = [f for f in all_nc_files if f != katrina_path]

    try:
        held_out_test_size = cfg.data_params.held_out_test_size
    except Exception:
        print("\n" + "=" * 50)
        print("ERROR: 'data_params.held_out_test_size' not found in config file.")
        print("Please add 'held_out_test_size: <ratio>' (e.g., 0.1) to")
        print("the 'data_params' section in your YAML config.")
        print("=" * 50 + "\n")
        raise

    validation_size = cfg.data_params.test_size
    random_state = cfg.data_params.random_state

    if held_out_test_size + validation_size >= 1.0:
        raise ValueError(
            f"Sum of held_out_test_size ({held_out_test_size}) and "
            f"validation_size ({validation_size}) must be less than 1.0"
        )

    train_val_files, additional_test_files = train_test_split(
        remaining_files,
        test_size=held_out_test_size,
        random_state=random_state,
    )

    test_files = [katrina_path] + additional_test_files

    val_split_ratio = validation_size / (1.0 - held_out_test_size)

    train_files, val_files = train_test_split(
        train_val_files,
        test_size=val_split_ratio,
        random_state=random_state,
    )

    print(
        f"Training on {len(train_files)} files, validating on {len(val_files)} files."
    )
    print(
        f"Held-out test set: {len(test_files)} files (including {manual_test_file_name})."
    )

    test_file_list_path = os.path.join(os.getcwd(), "held_out_test_files.txt")
    try:
        with open(test_file_list_path, "w") as f:
            f.write(
                "# This is the held-out test set, it was NOT used for training or validation.\n"
            )
            for item in test_files:
                f.write(f"{os.path.basename(item)}\n")
        print(f"Saved list of held-out test file names to: {test_file_list_path}")
    except Exception as e:
        print(f"Warning: Could not save test file list: {e}")
    # --- MODIFICATION END ---


    print(f"Loading shared static data from: {train_files[0]}...")
    static_data_cpu = {}
    try:
        with xr.open_dataset(train_files[0]) as ds:
            # --- REFACTOR: _load_static_data_from_ds will need to be updated
            # to accept cfg.features to load the correct vars
            static_data_cpu = _load_static_data_from_ds(ds)
        print("Shared static data loaded to CPU. Measuring size...")

        print_tensor_size_mb(static_data_cpu)

    except Exception as e:
        raise IOError(f"Failed to load static data from {train_files[0]}: {e}")

    try:
        # --- NEW 3: Compute scaling stats if they don't exist ---
        train_root = os.path.join(processed_dir, "train_new")
        train_stats_path = os.path.join(train_root, "scaling_stats.yaml")
        os.makedirs(train_root, exist_ok=True)

        if not os.path.exists(train_stats_path):
            print(f"Scaling stats file not found at {train_stats_path}.")
            print("Calculating stats from training files... This may take a while.")
            # --- MODIFIED: Pass the features config to the scaling function ---
            compute_and_save_adforce_stats(
                train_files, 
                train_stats_path, 
                cfg.features
            )
            # --- END MODIFICATION ---
            print(f"Stats saved to {train_stats_path}.")
        else:
            print(f"Found existing scaling stats: {train_stats_path}")
        # --- END NEW BLOCK ---

        # 4. Create "lazy" datasets
        print("Initializing training dataset (this may run .process()...)")
        # --- REFACTOR: Pass features_cfg to dataset ---
        train_dataset = AdforceLazyDataset(
            root=train_root,
            nc_files=train_files,
            previous_t=p_t,
            scaling_stats_path=train_stats_path,
            features_cfg=cfg.features  # <-- PASS THE FEATURES CONFIG
        )

        print("Initializing validation dataset (this may run .process()...)")
        # --- REFACTOR: Pass features_cfg to dataset ---
        val_dataset = AdforceLazyDataset(
            root=os.path.join(processed_dir, "val"),
            nc_files=val_files,
            previous_t=p_t,
            scaling_stats_path=train_stats_path,  # <-- PASS THE *TRAIN* STATS
            features_cfg=cfg.features  # <-- PASS THE FEATURES CONFIG
        )

        # 5. Instantiate Lightning DataModule
        data_module = DataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=cfg.trainer_options.batch_size,
            num_workers=cfg.data_params.num_workers,
        )

        # --- REFACTOR: 6. Calculate Model Dimensions FROM CONFIG ---
        
        # +1 for 'node_type' which is auto-added by the dataset
        num_static_features = len(cfg.features.static) + 1
        num_forcing_features = len(cfg.features.forcing)
        num_state_features = len(cfg.features.state) + len(cfg.features.derived_state)
        num_output_features = len(cfg.features.targets)
        
        # This count is now also from the config
        num_edge_features = len(cfg.features.edge)

        # Total input features: static + (forcing * p_t) + state
        num_node_features = (
            num_static_features
            + (num_forcing_features * p_t)
            + num_state_features
        )

        print("-" * 30)
        print(f"Model dimensions calculated from config:")
        print(f"  Static Features: {num_static_features} ({len(cfg.features.static)} from config + 1 'node_type')")
        print(f"  Edge Features: {num_edge_features} ({cfg.features.edge})")
        print(f"  Forcing Features (per step): {num_forcing_features} (x{p_t} steps)")
        print(f"  State Features: {num_state_features} ({len(cfg.features.state)} state + {len(cfg.features.derived_state)} derived)")
        print(f"  --> TOTAL INPUT FEATURES: {num_node_features}")
        print(f"  Output Features: {num_output_features} ({cfg.features.targets})")
        print("-" * 30)
        # --- END REFACTOR ---

        # 7. Instantiate the Model
        model_type = cfg.model_params.model_type
        print(f"Instantiating model type: {model_type}...")

        model_cfg_dict = OmegaConf.to_container(cfg.model_params, resolve=True)

        # --- REFACTOR: Use calculated dimensions ---
        if model_type == "GNN":
            model = GNNModelAdforce(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                num_output_features=num_output_features,
                num_static_features=num_static_features, # Pass the calculated count
                **model_cfg_dict, # This dict includes previous_t
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
                f"Unknown model_type in config: {model_type}. Must be 'GNN', 'SWEGNN', 'MLP', or 'MonolithicMLP'."
            )
        # --- END REFACTOR ---

        # 8. Instantiate the LightningTrainer (the module)
        lightning_model = LightningTrainer(
            model=model,
            lr_info=cfg.lr_info,
            trainer_options=cfg.trainer_options
        )

        # 8.a: Configure Checkpointing
        print(f"Configuring checkpoints for {model_type}...")
        checkpoint_dir = cfg.machine.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        best_model_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{model_type}-best-{{epoch:02d}}-{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
        )

        last_model_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{model_type}-last-{{epoch:02d}}",
            save_last=True,
        )

        all_callbacks = [best_model_callback, last_model_callback]

        # W&B: Configure Logger
        print("Initializing Weights & Biases Logger...")
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            log_model=cfg.wandb.log_model,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )
        wandb_logger.watch(model, log="all", log_freq=100)

        # 9. Instantiate the Lightning Trainer (the runner)
        lt_cfg_dict = OmegaConf.to_container(cfg.lightning_trainer, resolve=True)

        trainer = L.Trainer(
            **lt_cfg_dict,
            callbacks=all_callbacks,
            logger=wandb_logger,
        )

        # 10. Start Training
        print("Starting training... 🚀")

        ckpt_path_to_use = None
        if resume_checkpoint_path:
            if os.path.exists(resume_checkpoint_path):
                print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
                ckpt_path_to_use = resume_checkpoint_path
            else:
                print(
                    f"WARNING: Checkpoint path specified in config not found: {resume_checkpoint_path}"
                )
                print("Starting a new training run.")
        else:
            print("Starting a new training run (no checkpoint specified).")

        trainer.fit(
            lightning_model,
            datamodule=data_module,
            ckpt_path=ckpt_path_to_use,
        )

        print("Training complete. ✅")
        print(f"Best model checkpoint saved to: {best_model_callback.best_model_path}")

        # W&B: Finish the run
        wandb.finish()

    except Exception as e:
        print(f"\nAn error occurred during dataset initialization or training: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("File handles (if any) will be closed by the DataModule's teardown hook.")


if __name__ == "__main__":
    main()