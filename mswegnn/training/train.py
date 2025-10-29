# Libraries
import numpy as np
import torch
import torch.optim as optim
import lightning as L
from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch

from mswegnn.training.loss import loss_function
# Removed imports: get_rollout_loss, get_CSI, use_prediction, apply_boundary_condition

# Define constants for the new data format
# Input features: 5 static + N*3 dynamic
# (DEM, slopex, slopey, area, node_type) + (WX, WY, P) * previous_t
NUM_OUTPUT_FEATURES = 3 # WD, VX, VY


class LightningTrainer(L.LightningModule):
    def __init__(self, model, lr_info, trainer_options):
        """
        Args:
            model (nn.Module): The GNN model.
            lr_info (dict): Learning rate configuration.
            trainer_options (dict): Trainer configuration.
        """
        super().__init__()
        self.model = model

        # Store hparams
        self.lr_info = lr_info
        self.learning_rate = lr_info['learning_rate']
        self.batch_size = trainer_options['batch_size']
        self.only_where_water = trainer_options['only_where_water']
        self.velocity_scaler = trainer_options['velocity_scaler']
        self.type_loss = trainer_options['type_loss']

        # Validate loss type
        assert self.type_loss in ['RMSE','MAE'], "loss_type must be either 'RMSE' or 'MAE'"

        # Note: We no longer store curriculum_epoch, conservation, or temporal_test_dataset_parameters
        # The number of dynamic_vars is handled by the model's `previous_t`

    def training_step(self, batch, batch_idx):
        """
        Performs a single 1-step-ahead training step.
        Rollout loops and curriculum learning are removed.
        """
        # Model prediction (1-step)
        # We no longer apply boundary conditions as they aren't loaded
        preds = self.model(batch) # shape [N, 3]

        # Calculate loss
        # batch.y is shape [N, 3] from AdforceLazyDataset
        # We remove the conservation term as BCs are not provided
        loss = loss_function(
            preds,
            batch.y,
            batch,
            conservation_target=None, # Removed BC-dependent term
            type_loss=self.type_loss,
            only_where_water=self.only_where_water,
            velocity_scaler=self.velocity_scaler
        )

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr_info['learning_rate'],
            weight_decay=self.lr_info['weight_decay']
        )

        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.lr_info['step_size'],
            gamma=self.lr_info['gamma']
        )
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        """
        Performs a single 1-step-ahead validation step.
        Full rollouts are no longer calculated here.
        """
        # Model prediction (1-step)
        preds = self.model(batch) # shape [N, 3]

        # Calculate loss
        val_loss = loss_function(
            preds,
            batch.y,
            batch,
            conservation_target=None,
            type_loss=self.type_loss,
            only_where_water=self.only_where_water,
            velocity_scaler=self.velocity_scaler
        )

        # Log 1-step validation loss
        # We no longer log CSI, as that requires a full rollout
        self.log("val_loss", val_loss, prog_bar=True, batch_size=batch.num_graphs)

    def test_step(self, batch, batch_idx):
        """
        Performs a single 1-step-ahead test step.
        """
        # Model prediction (1-step)
        preds = self.model(batch) # shape [N, 3]

        # Calculate loss
        test_loss = loss_function(
            preds,
            batch.y,
            batch,
            conservation_target=None,
            type_loss=self.type_loss,
            only_where_water=self.only_where_water,
            velocity_scaler=self.velocity_scaler
        )

        self.log("test_loss", test_loss, batch_size=batch.num_graphs)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Predicts a single 1-step-ahead batch.
        """
        preds = self.model(batch)
        # De-batch the predictions
        return [preds[batch.ptr[i]:batch.ptr[i+1]]
                for i in range(batch.num_graphs)]


class DataModule(L.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset=None,
                 batch_size: int = 8, num_workers: int = 4):
        """
        Args:
            train_dataset (Dataset): Training dataset (AdforceLazyDataset).
            val_dataset (Dataset): Validation dataset (AdforceLazyDataset).
            test_dataset (Dataset, optional): Test dataset (AdforceLazyDataset).
            batch_size (int, optional): Batch size.
            num_workers (int, optional): Number of workers for DataLoader.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Ensure datasets are not None
        if self.train_dataset is None or self.val_dataset is None:
            raise ValueError("train_dataset and val_dataset must be provided.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True
            )
        return None

    def setup(self, stage=None):
        # This hook is called on every process in DDP.
        # We don't need to do anything here since datasets are passed in __init__
        pass

    def teardown(self, stage=None):
        # Close file handles on all datasets
        if hasattr(self.train_dataset, 'close'):
            self.train_dataset.close()
        if hasattr(self.val_dataset, 'close'):
            self.val_dataset.close()
        if self.test_dataset and hasattr(self.test_dataset, 'close'):
            self.test_dataset.close()

# Removed CurriculumLearning and CurriculumBatchSizeFinder classes
