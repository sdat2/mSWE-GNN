"""
Autoregressive Rollout Utility for ADForce GNN Models.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import xarray as xr
from tqdm import tqdm
import torch
import lightning as L
from mswegnn.utils.adforce_dataset import AdforceLazyDataset


def load_static_data(
    nc_file_path: str, dataset: AdforceLazyDataset, features_cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads static coordinates and DEM data.

    Args:
        nc_file_path (str): Path to the NetCDF file.
        dataset (AdforceLazyDataset): The initialized dataset instance.
        features_cfg (Dict[str, Any]): The 'features' block from the config.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: x_coords, y_coords, dem
    """
    print("Loading static data (coordinates and DEM)...")
    try:
        with xr.open_dataset(nc_file_path) as ds:
            x_coords = ds["x"].values
            y_coords = ds["y"].values
    except Exception as e:
        print(f"Failed to read file coordinates from {nc_file_path}: {e}")
        raise

    # --- FIX: Refactored to read 'DEM' directly from dataset.static_data ---
    # The AdforceLazyDataset now stores static features individually in this dict.
    try:
        if "DEM" not in dataset.static_data:
            raise KeyError(
                f"'DEM' not found in dataset.static_data. Available keys: {list(dataset.static_data.keys())}"
            )

        # DEM in static_data is typically kept RAW (unscaled) in the dataset class
        dem = dataset.static_data["DEM"].cpu().numpy()

    except Exception as e:
        print(f"Failed to get DEM from dataset.static_data.")
        print(
            f"Config expected 'DEM' in features_cfg.static: {list(features_cfg.static)}"
        )
        raise e
    # --- END FIX ---

    return x_coords, y_coords, dem


@torch.no_grad()
def perform_rollout(
    model: L.LightningModule,
    dataset: AdforceLazyDataset,
    device: torch.device,
    features_cfg: Dict[str, Any],
    scaling_stats: Dict[str, Any],
    rollout_horizon: int = 1,
) -> List[np.ndarray]:
    """
    Performs an autoregressive rollout based on the specified horizon.

    This function assumes the model predicts the *SCALED DELTA* (change in state).
    It automatically handles scaling if the dataset provides raw unscaled data.

    Args:
        model (L.LightningModule): The trained Lightning model (on device).
        dataset (AdforceLazyDataset): The dataset for a single simulation.
        device (torch.device): The torch device (e.g., 'cuda' or 'cpu').
        features_cfg (Dict[str, Any]): The 'features' block from the config.
        scaling_stats (Dict[str, Any]): The loaded scaling_stats.yaml dict.
        rollout_horizon (int, optional): The rollout strategy.
            - (N = -1): "Full Rollout". Runs one long simulation from t=0.
            - (N > 0): "Fixed Horizon". Runs a new N-step simulation for each frame.

    Returns:
        List[np.ndarray]: A list of unscaled predicted states for each frame.
    """
    model.eval()

    # 1. Determine if we need to manually scale inputs
    # If dataset.apply_scaling is False (due to error or config), we must scale manually.
    manual_scaling_needed = not getattr(dataset, "apply_scaling", False)
    if manual_scaling_needed:
        print("WARNING: Dataset is unscaled. Performing MANUAL SCALING inside rollout.")

    # 2. Load ALL scaling stats to device
    try:
        # State stats
        y_mean = torch.tensor(scaling_stats["y_mean"], dtype=torch.float32).to(device)
        y_std = (
            torch.tensor(scaling_stats["y_std"], dtype=torch.float32)
            .to(device)
            .clamp(min=1e-6)
        )
        y_delta_mean = torch.tensor(
            scaling_stats["y_delta_mean"], dtype=torch.float32
        ).to(device)
        y_delta_std = (
            torch.tensor(scaling_stats["y_delta_std"], dtype=torch.float32)
            .to(device)
            .clamp(min=1e-6)
        )

        # Inputs stats (needed if manual_scaling_needed is True)
        x_static_mean = torch.tensor(
            scaling_stats["x_static_mean"], dtype=torch.float32
        ).to(device)
        x_static_std = (
            torch.tensor(scaling_stats["x_static_std"], dtype=torch.float32)
            .to(device)
            .clamp(min=1e-6)
        )
        x_dyn_mean = torch.tensor(
            scaling_stats["x_dynamic_mean"], dtype=torch.float32
        ).to(device)
        x_dyn_std = (
            torch.tensor(scaling_stats["x_dynamic_std"], dtype=torch.float32)
            .to(device)
            .clamp(min=1e-6)
        )

        # Edge stats
        if "edge_mean" in scaling_stats:
            edge_mean = torch.tensor(
                scaling_stats["edge_mean"], dtype=torch.float32
            ).to(device)
            edge_std = (
                torch.tensor(scaling_stats["edge_std"], dtype=torch.float32)
                .to(device)
                .clamp(min=1e-6)
            )

            # Apply edge scaling to dataset IN-PLACE if needed (since edges are static)
            if manual_scaling_needed and "static_edge_attr" in dataset.static_data:
                # Check if we already scaled it in a previous call (hacky check)
                # Ideally, dataset should handle this, but we do it here to save the run.
                # We assume if manual_scaling_needed is True, edges are raw.
                raw_edges = dataset.static_data["static_edge_attr"].to(device)
                dataset.static_data["static_edge_attr"] = (
                    raw_edges - edge_mean
                ) / edge_std
                # Update the cached CPU version too to avoid re-transfer issues if dataset.get re-reads
                # (Though dataset.get uses the dict reference, so modifying the dict is enough if shared)

    except (KeyError, TypeError) as e:
        print(f"Error: Scaling stats dict is missing keys or invalid: {e}")
        raise e

    # --- HACK: Need DEM on device for derived features ---
    if "DEM" in dataset.static_data:
        dem_gpu = dataset.static_data["DEM"].to(device)
    else:
        print(
            "Warning: 'DEM' missing from static data. Derived features using DEM will fail."
        )
        dem_gpu = None

    predictions_list = []

    # --- Feature Counts & Index Slicing ---
    num_static = len(features_cfg.static) + 1  # +1 for node_type
    num_forcing = len(features_cfg.forcing)
    num_state = len(features_cfg.state)
    num_derived = len(features_cfg.derived_state)
    p_t = dataset.previous_t
    num_total_state = num_state + num_derived

    # The 'x' tensor structure: [static | forcing | state]
    # Indices:
    static_start, static_end = 0, num_static
    forcing_start, forcing_end = static_end, static_end + (num_forcing * p_t)
    state_block_start, state_block_end = forcing_end, forcing_end + num_total_state

    # Sub-slice for the base state (the part we predict/update)
    state_base_start = state_block_start
    state_base_end = state_block_start + num_state

    # --- Helper to ensure a batch is SCALED ---
    def get_scaled_batch(batch_idx):
        batch = dataset.get(batch_idx).to(device)

        if not manual_scaling_needed:
            return batch  # Already scaled by dataset

        # Perform manual scaling on batch.x
        x = batch.x.clone()

        # 1. Scale Static
        x[:, static_start:static_end] = (
            x[:, static_start:static_end] - x_static_mean
        ) / x_static_std

        # 2. Scale Forcing (Broadcast over p_t)
        # x_dyn_mean/std is shape [F]. We need to tile it p_t times.
        x_dyn_mean_broadcast = x_dyn_mean.repeat(p_t)
        x_dyn_std_broadcast = x_dyn_std.repeat(p_t)
        x[:, forcing_start:forcing_end] = (
            x[:, forcing_start:forcing_end] - x_dyn_mean_broadcast
        ) / x_dyn_std_broadcast

        # 3. Scale State
        x[:, state_block_start:state_block_end] = (
            x[:, state_block_start:state_block_end] - y_mean
        ) / y_std

        batch.x = x
        return batch

    # --- BRANCH 1: FULL ROLLOUT ---
    if rollout_horizon == -1:
        print("Starting full, free-running rollout (predicting deltas)...")

        # 1. Get Initial Batch (Scaled)
        # Note: We use get_scaled_batch to ensure 'current_full_state_scaled' is valid model input.
        current_batch = get_scaled_batch(0)

        # 2. Extract Initial State
        # a) SCALED full state (for model input)
        current_full_state_scaled = current_batch.x[
            :, state_block_start:state_block_end
        ].clone()

        # b) RAW base state (for physics update)
        # Since we just ensured current_batch is scaled, we MUST unscale it to get raw physics state.
        current_y_t_raw = (
            current_batch.x[:, state_base_start:state_base_end].clone()
            * y_std[:num_state]
        ) + y_mean[:num_state]

        for idx in tqdm(range(len(dataset)), desc="Full Rollout"):
            # 1. Get Ground Truth Forcing (Scaled)
            gt_batch = get_scaled_batch(idx)

            # 2. Create Prediction Input
            pred_input_batch = gt_batch.clone()

            # 3. Overwrite state with PREVIOUS PREDICTION
            pred_input_batch.x[:, state_block_start:state_block_end] = (
                current_full_state_scaled
            )

            # 4. Predict Scaled Delta
            pred_scaled_delta = model.model(pred_input_batch)

            # 5. Unscale Delta
            pred_raw_delta = (pred_scaled_delta * y_delta_std) + y_delta_mean

            # 6. Apply Delta to RAW state
            next_y_t_raw = current_y_t_raw + pred_raw_delta

            # 7. Store Result
            predictions_list.append(next_y_t_raw.cpu().numpy())

            # 8. Prepare for next step: Derived Features & Scaling

            # a) Calculate Derived Features (in RAW space)
            y_t_dict_gpu = {
                var: next_y_t_raw[:, i]
                for i, var in enumerate(list(features_cfg.state))
            }
            derived_list = []

            for derived_spec in features_cfg.derived_state:
                arg_data = []
                for arg_name in derived_spec["args"]:
                    if arg_name in y_t_dict_gpu:
                        arg_data.append(y_t_dict_gpu[arg_name])
                    elif arg_name == "DEM":
                        arg_data.append(dem_gpu)
                    else:
                        raise ValueError(f"Unknown arg '{arg_name}'")

                if derived_spec["op"] == "add":
                    val = arg_data[0] + arg_data[1]
                elif derived_spec["op"] == "subtract":
                    val = arg_data[0] - arg_data[1]
                elif derived_spec["op"] == "magnitude":
                    val = torch.sqrt(arg_data[0] ** 2 + arg_data[1] ** 2)
                else:
                    raise ValueError(f"Unknown op {derived_spec['op']}")
                derived_list.append(val.unsqueeze(1))

            # b) Concatenate to Full Raw State
            if derived_list:
                full_state_raw = torch.cat([next_y_t_raw] + derived_list, dim=1)
            else:
                full_state_raw = next_y_t_raw

            # c) Scale Full State for next input
            current_full_state_scaled = (full_state_raw - y_mean) / y_std

            # d) Update Raw State for next iteration
            current_y_t_raw = next_y_t_raw

    # --- BRANCH 2: FIXED HORIZON ---
    else:
        print(f"Starting {rollout_horizon}-step fixed-horizon rollout...")

        for idx in tqdm(range(len(dataset)), desc=f"{rollout_horizon}-Step"):
            start_idx = max(0, idx - rollout_horizon + 1)
            steps_to_run = idx - start_idx + 1

            # Initialize from Ground Truth at start_idx
            gt_batch_start = get_scaled_batch(start_idx)

            current_full_state_scaled = gt_batch_start.x[
                :, state_block_start:state_block_end
            ].clone()
            current_y_t_raw = (
                gt_batch_start.x[:, state_base_start:state_base_end].clone()
                * y_std[:num_state]
            ) + y_mean[:num_state]

            # Mini-Rollout
            for k in range(steps_to_run):
                forcing_idx = start_idx + k
                if forcing_idx >= len(dataset):
                    break

                gt_forcing_batch = get_scaled_batch(forcing_idx)

                pred_input = gt_forcing_batch.clone()
                pred_input.x[:, state_block_start:state_block_end] = (
                    current_full_state_scaled
                )

                pred_delta = model.model(pred_input)
                pred_raw_delta = (pred_delta * y_delta_std) + y_delta_mean
                next_y_t_raw = current_y_t_raw + pred_raw_delta

                # Derived & Rescale
                y_t_dict_gpu = {
                    var: next_y_t_raw[:, i]
                    for i, var in enumerate(list(features_cfg.state))
                }
                derived_list = []
                for derived_spec in features_cfg.derived_state:
                    arg_data = []
                    for arg_name in derived_spec["args"]:
                        if arg_name in y_t_dict_gpu:
                            arg_data.append(y_t_dict_gpu[arg_name])
                        elif arg_name == "DEM":
                            arg_data.append(dem_gpu)
                        else:
                            raise ValueError(f"Unknown arg '{arg_name}'")

                    if derived_spec["op"] == "add":
                        val = arg_data[0] + arg_data[1]
                    elif derived_spec["op"] == "subtract":
                        val = arg_data[0] - arg_data[1]
                    elif derived_spec["op"] == "magnitude":
                        val = torch.sqrt(arg_data[0] ** 2 + arg_data[1] ** 2)
                    derived_list.append(val.unsqueeze(1))

                if derived_list:
                    full_state_raw = torch.cat([next_y_t_raw] + derived_list, dim=1)
                else:
                    full_state_raw = next_y_t_raw

                current_full_state_scaled = (full_state_raw - y_mean) / y_std
                current_y_t_raw = next_y_t_raw

            predictions_list.append(current_y_t_raw.cpu().numpy())

    print("Rollout complete.")
    return predictions_list
