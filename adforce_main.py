"""
Main training script for the mSWE-GNN Adforce pipeline.

This script ties together all the new components:
1.  Loads the 'adforce_config.yaml' for all hyperparameters.
2.  Finds and splits NetCDF files.
3.  Calculates scaling statistics (mean/std) from the training files.
4.  Uses 'AdforceLazyDataset' to create train/val datasets
    (which now apply the scaling).
5.  Calculates model dimensions based on the dataset's known structure.
6.  Instantiates the correct model ('GNNModel_new', 'MSGNNModel_new', or 'MLPModel_new').
7.  Uses the 'DataModule' and 'LightningTrainer' from adforce_train.py to run
    the training loop.
"""

import glob
import os
import lightning as L
from sklearn.model_selection import train_test_split

# Imports from our mSWE-GNN project
from mswegnn.utils.adforce_dataset import AdforceLazyDataset
from mswegnn.utils.load import read_config

# --- MODIFIED: Added MLPModel_new ---
from mswegnn.models.adforce_models import GNNModel_new, MSGNNModel_new, MLPModel_new
from mswegnn.training.adforce_train import LightningTrainer, DataModule

# --- ADDED: Import for scaling stats calculator ---
from mswegnn.utils.adforce_scaling import compute_and_save_adforce_stats


# --- Constants based on adforce_dataset.py ---
#
# 5 static node features: (DEM, slopex, slopey, area, node_type)
NUM_STATIC_NODE_FEATURES = 5
# 3 dynamic node features: (WX, WY, P)
NUM_DYNAMIC_NODE_FEATURES = 3
# 2 static edge features: (face_distance, edge_slope)
NUM_STATIC_EDGE_FEATURES = 2
# 3 target variables: (WD, VX, VY)
NUM_OUTPUT_FEATURES = 3

# --- Configuration ---
CONFIG_PATH = "adforce_config.yaml"


def main():
    """
    Main function to run the data loading and training pipeline.
    """
    # 1. Load Configuration
    print(f"Loading configuration from {CONFIG_PATH}...")
    try:
        config = read_config(CONFIG_PATH)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {CONFIG_PATH}")
        print("Please create 'adforce_config.yaml' based on the example provided.")
        return
    except Exception as e:
        print(f"Error reading config file: {e}")
        return

    data_cfg = config.get("data_params", {})
    model_cfg = config.get("model_params", {})
    trainer_cfg = config.get("trainer_options", {})
    lr_cfg = config.get("lr_info", {})
    lt_cfg = config.get("lightning_trainer", {})

    p_t = data_cfg.get("previous_t", 1)
    data_dir = data_cfg.get("data_dir", "data/")

    # 2. Find and split data files
    print(f"Searching for NetCDF files in {data_dir}...")
    all_nc_files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    if not all_nc_files:
        print(f"ERROR: No *.nc files found in {data_dir}.")
        print("Please check the 'data_dir' path in your config file.")
        return
    print(f"Found {len(all_nc_files)} total simulation files.")
    all_nc_files = all_nc_files[:10]  # just for quick testing

    train_files, val_files = train_test_split(
        all_nc_files,
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
    )
    print(
        f"Training on {len(train_files)} files, validating on {len(val_files)} files."
    )

    try:
        # --- NEW 3: Compute scaling stats if they don't exist ---
        # Stats are saved in the 'train' processed directory
        train_root = "data_processed/train"
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
            root="data_processed/val",
            nc_files=val_files,
            previous_t=p_t,
            scaling_stats_path=train_stats_path,  # <-- PASS THE *TRAIN* STATS
        )

        # 5. Instantiate Lightning DataModule
        # This handles DataLoaders and auto-closes file handles via setup/teardown
        data_module = DataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=trainer_cfg.get("batch_size", 32),
            num_workers=data_cfg.get("num_workers", 4),
        )

        # 6. Calculate Model Dimensions
        num_node_features = NUM_STATIC_NODE_FEATURES + (NUM_DYNAMIC_NODE_FEATURES * p_t)
        num_edge_features = NUM_STATIC_EDGE_FEATURES
        num_output_features = NUM_OUTPUT_FEATURES

        print("-" * 30)
        print(f"Model dimensions calculated:")
        print(
            f"  Input Node Features: {num_node_features} (5 static + {NUM_DYNAMIC_NODE_FEATURES} dynamic * {p_t} steps)"
        )
        print(f"  Input Edge Features: {num_edge_features}")
        print(f"  Output Features: {num_output_features}")
        print("-" * 30)

        # 7. Instantiate the Model
        model_type = model_cfg.get("model_type", "GNN")
        print(f"Instantiating model type: {model_type}...")

        # --- MODIFIED: Added 'MLP' option ---
        # Pass all model_params, the constructors will pick what they need
        if model_type == "GNN":
            model = GNNModel_new(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                previous_t=p_t,
                num_output_features=num_output_features,
                num_static_features=NUM_STATIC_NODE_FEATURES,
                **model_cfg,
            )
        elif model_type == "MSGNN":
            model = MSGNNModel_new(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                previous_t=p_t,
                num_output_features=num_output_features,
                num_static_features=NUM_STATIC_NODE_FEATURES,
                **model_cfg,
            )
        elif model_type == "MLP":
            model = MLPModel_new(
                num_node_features=num_node_features,
                num_output_features=num_output_features,
                **model_cfg,  # Passes hid_features, mlp_layers, etc.
            )
        else:
            raise ValueError(
                f"Unknown model_type in config: {model_type}. Must be 'GNN', 'MSGNN', or 'MLP'."
            )
        # --- END MODIFICATION ---

        # 8. Instantiate the LightningTrainer (the module)
        lightning_model = LightningTrainer(
            model=model, lr_info=lr_cfg, trainer_options=trainer_cfg
        )

        # 9. Instantiate the Lightning Trainer (the runner)
        # You can add loggers, callbacks, etc., here
        # e.g., from lightning.pytorch.loggers import TensorBoardLogger
        # logger = TensorBoardLogger("tb_logs", name="mswe-gnn-adforce")
        trainer = L.Trainer(
            **lt_cfg  # Passes max_epochs, accelerator, devices, etc.
            # logger=logger,
        )

        # 10. Start Training
        print("Starting training... 🚀")
        trainer.fit(lightning_model, datamodule=data_module)

        print("Training complete. ✅")

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
