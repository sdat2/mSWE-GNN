# Libraries
import torch
import wandb
import time
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import glob
import os
from sklearn.model_selection import train_test_split

# --- New Imports ---
from mswegnn.utils.adforce_dataset import AdforceLazyDataset # Your new data loader
from mswegnn.training.train import LightningTrainer, DataModule # Updated PL modules

# --- Old/Removed Imports ---
# from utils.dataset import create_model_dataset, to_temporal_dataset
# from utils.dataset import get_temporal_test_dataset_parameters
# from utils.visualization import PlotRollout
# from utils.miscellaneous import get_numerical_times, get_speed_up, SpatialAnalysis
# from training.train import CurriculumLearning

# --- Kept Imports ---
from mswegnn.utils.load import read_config
from mswegnn.utils.miscellaneous import get_model, fix_dict_in_config

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

def main(config):
    L.seed_everything(config.models['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_parameters = config.dataset_parameters
    temporal_dataset_parameters = config.temporal_dataset_parameters

    # --- New Data Loading Pipeline ---
    print("--- Starting Data Loading ---")
    data_dir = dataset_parameters.get('dataset_folder', 'data/processed')

    # Use glob to find all NetCDF files
    try:
        all_nc_files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
        if not all_nc_files:
            raise FileNotFoundError(f"No .nc files found in directory: {data_dir}")
        print(f"Found {len(all_nc_files)} total simulation files.")
    except Exception as e:
        print(f"Error finding data files: {e}")
        return

    # Split the *file list* into train, validation, and test sets
    # e.g., 80% train+val, 20% test
    val_prcnt = dataset_parameters.get('val_prcnt', 0.2)
    test_prcnt = dataset_parameters.get('test_prcnt', 0.2) # Add new config param

    # Adjust validation percent to be from the training set, not total
    # e.g., test=0.2 -> 80% left. val=0.25 of 80% -> 20% of total
    train_val_prcnt = 1.0 - test_prcnt
    val_prcnt_adjusted = val_prcnt / train_val_prcnt # e.g., 0.25 = 0.2 / 0.8

    train_val_files, test_files = train_test_split(
        all_nc_files,
        test_size=test_prcnt,
        random_state=dataset_parameters.get('seed', 42)
    )

    train_files, val_files = train_test_split(
        train_val_files,
        test_size=val_prcnt_adjusted,
        random_state=dataset_parameters.get('seed', 42)
    )

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")

    # Get temporal parameters
    previous_t = temporal_dataset_parameters.get('previous_t', 1)

    # NOTE: rollout_steps is REMOVED from the lazy loader.
    # The loader is assumed to *always* load 1-step-ahead (t+1).
    # We must modify AdforceLazyDataset to accept previous_t.

    train_dataset, val_dataset, test_dataset = None, None, None
    try:
        train_dataset = AdforceLazyDataset(
            root="data_processed/train",
            nc_files=train_files
            # TODO: You MUST update AdforceLazyDataset to accept/use previous_t
            # previous_t=previous_t
        )

        val_dataset = AdforceLazyDataset(
            root="data_processed/val",
            nc_files=val_files
            # previous_t=previous_t
        )

        test_dataset = AdforceLazyDataset(
            root="data_processed/test",
            nc_files=test_files
            # previous_t=previous_t
        )

        print(f"Total training samples:   {len(train_dataset)}")
        print(f"Total validation samples: {len(val_dataset)}")
        print(f"Total test samples:       {len(test_dataset)}")

        # Get feature counts from the *first sample*
        # This forces the .get() method to run once
        sample_data = train_dataset[0]
        num_node_features = sample_data.x.size(-1)
        num_edge_features = sample_data.edge_attr.size(-1)

        print(f"Number of node features: {num_node_features}")
        print(f"Number of edge features: {num_edge_features}")

        # --- Model Instantiation (Unchanged) ---
        model_parameters = config.models
        model_type = model_parameters.pop('model_type')

        if model_type == 'MSGNN':
            # This logic might need checking.
            # Where does num_scales come from? Assuming it's in config.
            if 'num_scales' not in model_parameters:
                print("Warning: model_type is MSGNN but 'num_scales' not in config.models")
                # Need a way to get this. Forcing to 1.
                model_parameters['num_scales'] = 1

        model = get_model(model_type)(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            previous_t=previous_t, # Pass previous_t to model
            device=device,
            **model_parameters
        ).to(device)

        # --- Trainer Setup ---
        trainer_options = config.trainer_options
        lr_info = config.lr_info

        # We no longer pass temporal_test_dataset_parameters
        plmodule = LightningTrainer(model, lr_info, trainer_options)

        pldatamodule = DataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=trainer_options['batch_size'],
            num_workers=trainer_options.get('num_workers', 4)
        )

        total_parameters = sum(p.numel() for p in model.parameters())
        wandb.log({"total parameters": total_parameters})

        # --- Training ---

        # Define callbacks (CurriculumLearning removed)
        checkpoint_callback = ModelCheckpoint(
            dirpath='lightning_logs/models',
            monitor="val_loss",
            mode='min',
            save_top_k=1
        )

        # Early stopping now monitors 'val_loss' since 'val_CSI_005' is gone
        early_stopping = EarlyStopping(
            'val_loss',
            mode='min',
            patience=trainer_options.get('patience', 100)
        )

        wandb_logger.watch(model, log="all", log_graph=False)

        # Load trained model
        plmodule_kwargs = {
            'model': model,
            'lr_info': lr_info,
            'trainer_options': trainer_options
        }

        if 'saved_model' in config:
            model = plmodule.load_from_checkpoint(
                config['saved_model'],
                map_location=device,
                **plmodule_kwargs
            )

        # Define trainer
        trainer = L.Trainer(
            accelerator="auto",
            devices='auto',
            max_epochs=trainer_options['max_epochs'],
            gradient_clip_val=1,
            precision='16-mixed',
            enable_progress_bar=True, # Changed to True
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping] # Removed curriculum_callback
        )

        # Train and get trained model
        print("--- Starting Training ---")
        trainer.fit(plmodule, pldatamodule)

        # Load the best model checkpoint
        print("--- Loading Best Model for Validation/Testing ---")
        plmodule = plmodule.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            map_location=device,
            **plmodule_kwargs
        )

        # Validate with trained model
        print("--- Running Validation ---")
        trainer.validate(plmodule, pldatamodule)

        # Test with trained model
        print("--- Running Testing ---")
        trainer.test(plmodule, pldatamodule)

        # --- All Rollout/Spatial/Plotting/Speedup logic is REMOVED ---
        # This logic was dependent on the old data structure and
        # full-rollout validation, which we are no longer doing.
        # You would need to write a new, separate script for
        # full-rollout inference.

        print('Training and testing finished!')

    except Exception as e:
        print(f"\n--- An error occurred ---")
        print(f"Error: {e}")
    finally:
        # Clean up file handles
        print("--- Closing file handles ---")
        if pldatamodule:
            pldatamodule.teardown()


if __name__ == '__main__':
    # Read configuration file with parameters
    cfg = read_config('config.yaml') # Assumes new config is named 'config.yaml'

    wandb_logger = WandbLogger(
        log_model=True,
        # mode='disabled',
        config=cfg
    )

    fix_dict_in_config(wandb)
    config = wandb.config

    main(config)
