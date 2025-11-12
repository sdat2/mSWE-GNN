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
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch
import xarray as xr
import hydra
from omegaconf import DictConfig, OmegaConf
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


def _load_split_file(split_file_path: str, data_dir: str) -> List[str]:
    """
    Reads a .txt file containing a list of basenames and
    returns a list of full paths.
    """
    # Resolve the path to the .txt file (e.g., conf/splits/train.txt)
    # This makes it relative to the original working directory
    abs_split_path = hydra.utils.to_absolute_path(split_file_path)
    
    if not os.path.exists(abs_split_path):
        raise FileNotFoundError(
            f"Split file not found. Expected at: {abs_split_path}\n"
            f"Check 'data_params.split_files' in your config."
        )

    print(f"Loading split from: {abs_split_path}")
    with open(abs_split_path, 'r') as f:
        # Read basenames (e.g., "262_GILBERT_1988.nc")
        basenames = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
    # Create full paths (e.g., /path/to/data_dir/262_GILBERT_1988.nc)
    full_paths = [os.path.join(data_dir, basename) for basename in basenames]
    
    return full_paths


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

    # --- REFACTOR: Check for explicit split files vs. random split ---
    
    if cfg.data_params.split_files.train:
        # --- MODE 2: Load splits from explicit files ---
        print("Using explicit split files defined in config.")
        
        train_files = _load_split_file(cfg.data_params.split_files.train, data_dir)
        val_files = _load_split_file(cfg.data_params.split_files.val, data_dir)
        test_files = _load_split_file(cfg.data_params.split_files.test, data_dir)
        
        # Verify that the loaded files are a subset of all files
        all_found_files = set([os.path.basename(f) for f in all_nc_files])
        all_split_files = set(
            [os.path.basename(f) for f in train_files + val_files + test_files]
        )
        
        missing_files = all_split_files - all_found_files
        if missing_files:
            warnings.warn(
                f"WARNING: The following {len(missing_files)} files from your split lists "
                f"were not found in data_dir ({data_dir}): {missing_files}"
            )
        
        # Check for overlaps
        train_val_overlap = set(train_files) & set(val_files)
        train_test_overlap = set(train_files) & set(test_files)
        val_test_overlap = set(val_files) & set(test_files)

        if train_val_overlap or train_test_overlap or val_test_overlap:
            warnings.warn("WARNING: Overlap detected between train/val/test split files!")
            if train_val_overlap: print(f"  Train-Val overlap: {train_val_overlap}")
            if train_test_overlap: print(f"  Train-Test overlap: {train_test_overlap}")
            if val_test_overlap: print(f"  Val-Test overlap: {val_test_overlap}")

    else:
        # --- MODE 1: Use random train_test_split (original logic) ---
        print("Using random split based on 'data_params' config.")
        
        # 1. Manually reserve the required test file
        manual_test_file_name = cfg.data_params.manual_test_holdout
        katrina_path = None
        if manual_test_file_name:
            for f_path in all_nc_files:
                if f_path.endswith(manual_test_file_name):
                    katrina_path = f_path
                    break

            if katrina_path is None:
                raise FileNotFoundError(
                    f"Could not find required test file {manual_test_file_name} in {data_dir}. "
                    f"Please check 'data_params.manual_test_holdout' in your config."
                )
            else:
                print(f"Manually reserving file for test set: {manual_test_file_name}")
            
            # 2. Get all other files available for splitting
            remaining_files = [f for f in all_nc_files if f != katrina_path]
        else:
            print("No 'manual_test_holdout' specified. Splitting all files.")
            remaining_files = all_nc_files
            katrina_path = None # Ensure it's None

        # 3. Define split ratios from config
        try:
            held_out_test_size = cfg.data_params.held_out_test_size
        except Exception:
            # (Original error handling for missing config param)
            print("\n" + "=" * 50)
            print("ERROR: 'data_params.held_out_test_size' not found in config file.")
            print("=" * 50 + "\n")
            raise

        validation_size = cfg.data_params.test_size
        random_state = cfg.data_params.random_state

        if held_out_test_size + validation_size >= 1.0:
            raise ValueError(
                f"Sum of held_out_test_size ({held_out_test_size}) and "
                f"validation_size ({validation_size}) must be less than 1.0"
            )

        # 4. Split 'remaining_files' into (train+val) and (additional_test)
        train_val_files, additional_test_files = train_test_split(
            remaining_files,
            test_size=held_out_test_size,
            random_state=random_state,
        )

        # 5. Create the final, held-out test set
        test_files = additional_test_files
        if katrina_path:
            test_files = [katrina_path] + additional_test_files

        # 6. Split 'train_val_files' into train and val
        val_split_ratio = validation_size / (1.0 - held_out_test_size)
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=val_split_ratio,
            random_state=random_state,
        )

    # --- END REFACTOR ---

    # Ensure no files were lost
    if not train_files:
        raise ValueError("No training files were found. Check your data_dir or split logic.")
    if not val_files:
        warnings.warn("No validation files were found. Check your data_dir or split logic.")
    if not test_files:
        warnings.warn("No test files were found. Check your data_dir or split logic.")


    print(
        f"Training on {len(train_files)} files, validating on {len(val_files)} files."
    )
    print(
        f"Held-out test set: {len(test_files)} files."
    )

    # 7. (Optional) Save the split file lists to the Hydra run directory
    try:
        with open(os.path.join(os.getcwd(), "train_files.txt"), "w") as f:
            f.write("# Training files\n")
            f.write("\n".join([os.path.basename(item) for item in train_files]))
        
        with open(os.path.join(os.getcwd(), "val_files.txt"), "w") as f:
            f.write("# Validation files\n")
            f.write("\n".join([os.path.basename(item) for item in val_files]))
            
        with open(os.path.join(os.getcwd(), "test_files.txt"), "w") as f:
            f.write("# Test files\n")
            f.write("\n".join([os.path.basename(item) for item in test_files]))
            
        print(f"Saved lists of train/val/test file names to: {os.getcwd()}")
    except Exception as e:
        print(f"Warning: Could not save split file lists: {e}")
    # --- END SPLIT LOGIC ---


    print(f"Loading shared static data from: {train_files[0]}...")
    static_data_cpu = {}
    try:
        with xr.open_dataset(train_files[0]) as ds:
            # --- REFACTOR (BUG FIX): Pass the feature lists from the config ---
            static_data_cpu = _load_static_data_from_ds(
                ds,
                cfg.features.static,
                cfg.features.edge
            )
        print("Shared static data loaded to CPU. Measuring size...")

        print_tensor_size_mb(static_data_cpu)

    except Exception as e:
        try:
            print("Could have been unlucky")
            with xr.open_dataset(train_files[1]) as ds:
                # --- REFACTOR (BUG FIX): Pass the feature lists from the config ---
                static_data_cpu = _load_static_data_from_ds(
                    ds,
                    cfg.features.static,
                    cfg.features.edge
                )
            print("Shared static data loaded to CPU. Measuring size...")

            print_tensor_size_mb(static_data_cpu)
        except Exception as ee:
            raise IOError(f"Failed to load static data from {train_files[0]} and {train_files[1]}: {e}, {ee}")

    try:
        # --- NEW 3: Compute scaling stats if they don't exist ---
        train_root = os.path.join(processed_dir, "train_flex")
        train_stats_path = os.path.join(train_root, "scaling_stats.yaml")
        os.makedirs(train_root, exist_ok=True)

        if not os.path.exists(train_stats_path):
            print(f"Scaling stats file not found at {train_stats_path}.")
            print("Calculating stats from training files... This may take a while.")
            # --- MODIFIED: Pass the features config to the scaling function ---
            compute_and_save_adforce_stats(
                train_files, 
                train_stats_path, 
                cfg.features  # <-- NEW
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