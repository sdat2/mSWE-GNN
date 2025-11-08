"""
Main training script for the mSWE-GNN Adforce pipeline.

This script ties together all the new components:
1.  Loads the 'adforce_config.yaml' for all hyperparameters.
    --- HYDRA MODIFICATION ---
    a.  Uses Hydra for configuration management via @hydra.main.
    b.  Config is injected into the main() function as 'cfg'.
    ---
2.  Finds and splits NetCDF files.
3.  Calculates scaling statistics (mean/std) from the training files.
4.  Uses 'AdforceLazyDataset' to create train/val datasets
    (which now apply the scaling).
5.  Calculates model dimensions based on the dataset's known structure.
6.  Instantiates the correct model ('GNNModelAdforce', 'MonoliticMLPModel' or 'PointwiseMLPModel').
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
from lightning.pytorch.loggers import WandbLogger  # <-- W&B: Added
import torch
import xarray as xr
import hydra  # <-- HYDRA: Added
from omegaconf import DictConfig, OmegaConf  # <-- HYDRA: Added
import wandb  # <-- W&B: Added

from mswegnn.utils.adforce_dataset import AdforceLazyDataset, _load_static_data_from_ds
from mswegnn.utils.load import (
    read_config,
)  # <-- HYDRA: This is no longer used, but kept for reference
from mswegnn.models.adforce_models import (
    GNNModelAdforce,
    PointwiseMLPModel,
    MonolithicMLPModel
)
from mswegnn.training.adforce_train import LightningTrainer, DataModule
from mswegnn.utils.adforce_scaling import compute_and_save_adforce_stats


# --- Constants based on adforce_dataset.py ---
#
# 5 static node features: (DEM, slopex, slopey, area, node_type)
NUM_STATIC_NODE_FEATURES: int = 5
# 3 dynamic node features: (WX, WY, P)
NUM_DYNAMIC_NODE_FEATURES: int = 3
# 3 current state features: (WD(t), VX(t), VY(t))
NUM_CURRENT_STATE_FEATURES: int = 3
# 2 static edge features: (face_distance, edge_slope)
NUM_STATIC_EDGE_FEATURES: int = 2
# 3 target variables: (WD, VX, VY)
NUM_OUTPUT_FEATURES: int = 3

# --- Configuration ---
# CONFIG_PATH: str = "adforce_config.yaml" # <-- HYDRA: No longer needed


def print_tensor_size_mb(tensor_dict):
    """Helper to print the size of tensors in a dictionary."""
    total_size = 0
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor):
            size = v.element_size() * v.nelement()
            total_size += size
            print(f"  - Static tensor '{k}': {size / (1024**2):.2f} MB")
    print(f"  --- TOTAL STATIC DATA SIZE: {total_size / (1024**2):.2f} MB ---")


@hydra.main(
    config_path="conf", config_name="config", version_base=None
)  # <-- HYDRA: Added decorator
def main(cfg: DictConfig):  # <-- HYDRA: Config injected
    """
    Main function to run the data loading and training pipeline.
    """
    # 1. Load Configuration
    # --- HYDRA: Configuration is now injected as 'cfg' ---
    print("--- Configuration (from Hydra) ---")
    print(OmegaConf.to_yaml(cfg))
    print("----------------------------------")
    print(f"Hydra working directory: {os.getcwd()}")
    # ---

    # --- HYDRA: Old config loading removed ---
    # print(f"Loading configuration from {CONFIG_PATH}...")
    # try:
    #     config = read_config(CONFIG_PATH)
    # except FileNotFoundError:
    #     print(f"ERROR: Configuration file not found at {CONFIG_PATH}")
    #     print("Please create 'adforce_config.yaml' based on the example provided.")
    #     return
    # except Exception as e:
    #     print(f"Error reading config file: {e}")
    #     return
    #
    # data_cfg = config.get("data_params", {})
    # model_cfg = config.get("model_params", {})
    # trainer_cfg = config.get("trainer_options", {})
    # lr_cfg = config.get("lr_info", {})
    # lt_cfg = config.get("lightning_trainer", {})
    # --- END HYDRA MODIFICATION ---

    # --- HYDRA: Access params from cfg object ---
    # --- MODIFICATION: Pop the checkpoint path from lt_cfg ---
    # Get the path, defaulting to None if not specified.
    # By using .pop(), we *remove* it from the lt_cfg dictionary,
    # so it won't be incorrectly passed to the L.Trainer constructor.
    # resume_checkpoint_path = lt_cfg.pop("start_from_checkpoint_path", None) # <-- HYDRA: Old way

    # HYDRA: Get resume path from machine config
    resume_checkpoint_path = cfg.machine.resume_checkpoint_path
    if resume_checkpoint_path:
        # Resolve path relative to original directory
        resume_checkpoint_path = hydra.utils.to_absolute_path(resume_checkpoint_path)

    # --- END MODIFICATION ---

    p_t = cfg.data_params.previous_t  # data_cfg.get("previous_t", 1)

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
    # all_nc_files = all_nc_files[:10]  # just for quick testing

    train_files, val_files = train_test_split(
        all_nc_files,
        test_size=cfg.data_params.test_size,  # data_cfg.get("test_size", 0.2),
        random_state=cfg.data_params.random_state,  # data_cfg.get("random_state", 42),
    )
    print(
        f"Training on {len(train_files)} files, validating on {len(val_files)} files."
    )

    print(f"Loading shared static data from: {train_files[0]}...")
    # We load from the first *training* file. Assumes mesh is identical.
    static_data_cpu = {}
    try:
        with xr.open_dataset(train_files[0]) as ds:
            static_data_cpu = _load_static_data_from_ds(ds)
        print("Shared static data loaded to CPU. Measuring size...")

        print_tensor_size_mb(static_data_cpu)

    except Exception as e:
        raise IOError(f"Failed to load static data from {train_files[0]}: {e}")

    try:
        # --- NEW 3: Compute scaling stats if they don't exist ---
        # Stats are saved in the 'train' processed directory
        train_root = os.path.join(
            processed_dir, "train"
        )  # <-- HYDRA: Use processed_dir
        train_stats_path = os.path.join(train_root, "scaling_stats.yaml")

        # Ensure the processed directory exists
        os.makedirs(train_root, exist_ok=True)

        if not os.path.exists(train_stats_path):
            print(f"Scaling stats file not found at {train_stats_path}.")
            print("Calculating stats from training files... This may take a while.")
            # This function computes stats and saves them to the path
            compute_and_save_adforce_stats(train_files, train_stats_path)
            print(f"Stats saved to {train_stats_path}.")
        else:
            print(f"Found existing scaling stats: {train_stats_path}")
        # --- END NEW BLOCK ---

        # 4. Create "lazy" datasets
        print("Initializing training dataset (this may run .process()...)")
        train_dataset = AdforceLazyDataset(
            root=train_root,  # Use same root dir
            nc_files=train_files,
            previous_t=p_t,
            scaling_stats_path=train_stats_path,  # <-- PASS THE PATH
        )

        print("Initializing validation dataset (this may run .process()...)")
        val_dataset = AdforceLazyDataset(
            root=os.path.join(processed_dir, "val"),  # <-- HYDRA: Use processed_dir
            nc_files=val_files,
            previous_t=p_t,
            scaling_stats_path=train_stats_path,  # <-- PASS THE *TRAIN* STATS
        )

        # 5. Instantiate Lightning DataModule
        # This handles DataLoaders and auto-closes file handles via setup/teardown
        data_module = DataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=cfg.trainer_options.batch_size,  # trainer_cfg.get("batch_size", 32),
            num_workers=cfg.data_params.num_workers,  # data_cfg.get("num_workers", 4),
        )

        # 6. Calculate Model Dimensions
        num_node_features = (
            NUM_STATIC_NODE_FEATURES
            + (NUM_DYNAMIC_NODE_FEATURES * p_t)
            + NUM_CURRENT_STATE_FEATURES
        )
        num_edge_features = NUM_STATIC_EDGE_FEATURES
        num_output_features = NUM_OUTPUT_FEATURES

        print("-" * 30)
        print(f"Model dimensions calculated:")
        print(
            f"  Input Node Features: {num_node_features} (5 static + {NUM_DYNAMIC_NODE_FEATURES} forcing * {p_t} steps + {NUM_CURRENT_STATE_FEATURES} state)"
        )
        print(f"  Input Edge Features: {num_edge_features}")
        print(f"  Output Features: {num_output_features}")
        print("-" * 30)

        # 7. Instantiate the Model
        model_type = cfg.model_params.model_type  # model_cfg.get("model_type", "GNN")
        print(f"Instantiating model type: {model_type}...")

        # --- HYDRA: Convert model_params group to a dict for **kwargs
        model_cfg_dict = OmegaConf.to_container(cfg.model_params, resolve=True)
        # ---

        # --- MODIFIED: Added 'MLP' option ---
        # Pass all model_params, the constructors will pick what they need
        if model_type == "GNN":
            model = GNNModelAdforce(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                previous_t=p_t,
                num_output_features=num_output_features,
                num_static_features=NUM_STATIC_NODE_FEATURES,
                **model_cfg_dict,  # **model_cfg,
            )
        elif model_type == "MLP":
            model = PointwiseMLPModel(
                num_node_features=num_node_features,
                num_output_features=num_output_features,
                **model_cfg_dict,  # **model_cfg,
            )
        elif model_type == "MonolithicMLP":
            n_nodes_fixed = int(train_dataset.total_nodes)
            if n_nodes_fixed is None:
                raise ValueError("Could not determine n_nodes from train_dataset.total_nodes")
            print(f"Found fixed n_nodes from dataset: {n_nodes_fixed}")
            model = MonolithicMLPModel(
                n_nodes=n_nodes_fixed,  # <-- ADDED THIS
                num_node_features=num_node_features,
                num_output_features=num_output_features,
                **model_cfg_dict,  # **model_cfg,
            )
        else:
            raise ValueError(
                f"Unknown model_type in config: {model_type}. Must be 'GNN', 'MSGNN', or 'MLP'."
            )
        # --- END MODIFICATION ---

        # 8. Instantiate the LightningTrainer (the module)
        lightning_model = LightningTrainer(
            model=model,
            lr_info=cfg.lr_info,  # lr_cfg,
            trainer_options=cfg.trainer_options,  # trainer_cfg
        )

        # --- NEW 8.a: Configure Checkpointing ---
        print(f"Configuring checkpoints for {model_type}...")

        # Define a directory based on the model type
        # --- HYDRA: Use path from config, which defaults to hydra's run dir ---
        checkpoint_dir = (
            cfg.machine.checkpoint_dir
        )  # os.path.join("checkpoints", model_type)
        os.makedirs(checkpoint_dir, exist_ok=True)  # HYDRA: Ensure it exists
        # ---

        # Create a callback to save the best model based on 'val_loss'
        best_model_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{model_type}-best-{{epoch:02d}}-{{val_loss:.4f}}",
            monitor="val_loss",  # Metric to monitor (from validation_step)
            mode="min",  # 'min' because we want the lowest val_loss
            save_top_k=1,  # Save only the single best model
            verbose=True,
        )

        # Create a callback to save the last checkpoint (for resuming)
        last_model_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{model_type}-last-{{epoch:02d}}",
            save_last=True,  # Saves a 'last.ckpt' file
        )

        all_callbacks = [best_model_callback, last_model_callback]
        # --- END NEW BLOCK ---

        # --- W&B: Configure Logger ---
        print("Initializing Weights & Biases Logger...")
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            log_model=cfg.wandb.log_model,
            # Save the flattened config to W&B
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )
        wandb_logger.watch(model, log="all", log_freq=100)
        # --- END W&B BLOCK ---

        # 9. Instantiate the Lightning Trainer (the runner)
        # You can add loggers, callbacks, etc., here
        # e.g., from lightning.pytorch.loggers import TensorBoardLogger
        # logger = TensorBoardLogger("tb_logs", name="mswe-gnn-adforce")

        # --- HYDRA: Convert lightning_trainer config to dict ---
        # **lt_cfg now only contains valid L.Trainer arguments
        # because 'start_from_checkpoint_path' was popped out earlier.
        lt_cfg_dict = OmegaConf.to_container(cfg.lightning_trainer, resolve=True)
        # ---

        # --- MODIFIED: Pass the callbacks to the Trainer ---
        trainer = L.Trainer(
            **lt_cfg_dict,  # **lt_cfg,
            callbacks=all_callbacks,
            logger=wandb_logger,  # <-- W&B: Pass logger
        )
        # --- END MODIFICATION ---

        # 10. Start Training
        print("Starting training... 🚀")

        # --- MODIFICATION: Check if resume_checkpoint_path from config exists ---
        ckpt_path_to_use = None
        if resume_checkpoint_path:  # This variable was set at the top
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
        # --- END MODIFICATION ---

        # --- MODIFIED: Pass 'ckpt_path_to_use' to trainer.fit() ---
        trainer.fit(
            lightning_model,
            datamodule=data_module,
            ckpt_path=ckpt_path_to_use,  # This will be None if starting new
        )
        # --- END MODIFICATION ---

        print("Training complete. ✅")
        print(f"Best model checkpoint saved to: {best_model_callback.best_model_path}")

        # --- W&B: Finish the run ---
        wandb.finish()
        # ---

    except Exception as e:
        print(f"\nAn error occurred during dataset initialization or training: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Note: The L.Trainer.fit() call will automatically call
        # data_module.teardown(), which closes the file handles.
        #
        print("File handles (if any) will be closed by the DataModule's teardown hook.")


if __name__ == "__main__":
    # python -m adforce_main
    main()
