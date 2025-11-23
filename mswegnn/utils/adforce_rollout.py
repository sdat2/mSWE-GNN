"""
Autoregressive Rollout Utility for ADForce GNN Models.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
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

    Args:
        model (L.LightningModule): The trained Lightning model (on device).
        dataset (AdforceLazyDataset): The dataset for a single simulation.
        device (torch.device): The torch device (e.g., 'cuda' or 'cpu').
        features_cfg (Dict[str, Any]): The 'features' block from the config.
        scaling_stats (Dict[str, Any]): The loaded scaling_stats.yaml dict.
        rollout_horizon (int, optional): The rollout strategy.
            - (N = -1): "Full Rollout". Runs one long simulation from t=0.
              predictions_list[k] is the k-step-ahead prediction.
            - (N > 0): "Fixed Horizon". Runs a new N-step simulation for
              each frame. predictions_list[k] is the N-step-ahead
              prediction starting from ground truth at t=(k-N+1).
              The first N-1 frames are k-step-ahead predictions.

    Returns:
        List[np.ndarray]: A list of unscaled predicted states (matching the
                          order of features_cfg.state) for each frame.
    """
    model.eval()  # Set model to evaluation mode

    # --- Get all necessary scaling stats from the dict ---
    try:
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
    except (KeyError, TypeError) as e:
        print(f"Error: Scaling stats dict is missing keys or invalid: {e}")
        raise e

    # --- HACK: Need DEM on device for derived features ---
    # A more robust solution would pass all static features needed.
    dem_gpu = dataset.static_data["DEM"].to(device)

    predictions_list = []

    # --- Get feature counts from config ---
    num_static_features = len(features_cfg.static) + 1  # +1 for node type
    num_forcing_features = len(features_cfg.forcing)
    num_state_features = len(features_cfg.state)
    num_derived_features = len(features_cfg.derived_state)
    p_t = dataset.previous_t
    num_total_state_features = num_state_features + num_derived_features

    # --- FIX: Corrected state index calculation ---
    # The 'x' tensor is structured: [static, forcing, state (base + derived)]

    # Start of the *full* state block (base + derived)
    state_block_start_idx = num_static_features + (num_forcing_features * p_t)
    state_block_end_idx = state_block_start_idx + num_total_state_features

    # The slice for the *base state* (which we predict and replace)
    # is the first num_state_features of this block.
    state_base_start_idx = state_block_start_idx
    state_base_end_idx = state_block_start_idx + num_state_features
    # --- END FIX ---

    # --- BRANCH 1: FULL, FREE-RUNNING ROLLOUT ---
    if rollout_horizon == -1:
        print("Starting full, free-running rollout (predicting deltas)...")

        # --- 1. Get the *initial state* from frame 0 ---
        current_batch = dataset.get(0).to(device)

        # Get the SCALED *full* state (base + derived) from the initial batch
        current_full_state_scaled = current_batch.x[
            :, state_block_start_idx:state_block_end_idx
        ].clone()

        # Get the RAW *base state* for applying deltas
        # We un-scale just the base state part of the input vector
        current_y_t_raw = (
            current_batch.x[:, state_base_start_idx:state_base_end_idx].clone()
            * y_std[:num_state_features]
        ) + y_mean[:num_state_features]

        # --- AUTOREGRESSIVE LOOP: SCALING & DERIVED FEATURES ---
        # The core logic here maintains two states:
        # 1. `current_y_t_raw`: The UNCALED base state (e.g., [WD, VX, VY]).
        #    This is used for physically-correct operations.
        # 2. `current_full_state_scaled`: The SCALED full state (e.g.,
        #    [scaled_WD, ..., scaled_SSH]). This is the input for the model.
        #
        # The process is:
        # 1. Model predicts `pred_scaled_delta` from `current_full_state_scaled`.
        # 2. `pred_scaled_delta` is un-scaled -> `pred_raw_delta`.
        # 3. `pred_raw_delta` is added to `current_y_t_raw` -> `next_y_t_raw`.
        # 4. Derived features (e.g., SSH) are calculated in UNCALED space
        #    using `next_y_t_raw` and `dem_gpu`.
        # 5. The new `full_state_tensor_raw` (base + derived) is built.
        # 6. This full tensor is SCALED -> `current_full_state_scaled` for the
        #    next iteration.
        # ---

        for idx in tqdm(range(len(dataset)), desc="Full Rollout"):
            # 1. Get the *ground truth batch* for this step's *forcing*
            gt_batch = dataset.get(idx).to(device)

            # 2. Create the *prediction input*
            pred_input_batch = gt_batch.clone()

            # 3. ...but replace the *full state* with our *predicted* state
            pred_input_batch.x[:, state_block_start_idx:state_block_end_idx] = (
                current_full_state_scaled
            )

            # 4. Run the model to predict the *scaled delta*
            pred_scaled_delta = model.model(pred_input_batch)

            # 5. Un-scale the predicted delta
            pred_raw_delta = (pred_scaled_delta * y_delta_std) + y_delta_mean

            # 6. Apply the delta to get the next *base state*
            next_y_t_raw = current_y_t_raw + pred_raw_delta

            # 7. Store the *unscaled predicted base state* (for plotting)
            predictions_list.append(next_y_t_raw.cpu().numpy())

            # 8. Prepare for the *next* loop iteration

            # --- 8a. Re-calculate derived features (in UNCALED space) ---

            # 1. We have `next_y_t_raw` (base state) [N, 3] (UNSCALED)
            y_t_dict_gpu = {
                var: next_y_t_raw[:, i]
                for i, var in enumerate(list(features_cfg.state))
            }

            # 2. Build derived features list
            derived_state_features_list = []
            for derived_spec in features_cfg.derived_state:
                arg_data = []
                for arg_name in derived_spec["args"]:
                    if arg_name in y_t_dict_gpu:
                        arg_data.append(y_t_dict_gpu[arg_name])  # (UNSCALED)
                    elif arg_name == "DEM":  # HACK: hard-coding static features
                        arg_data.append(dem_gpu)  # (UNSCALED)
                    else:
                        raise ValueError(
                            f"Rollout: Unknown arg '{arg_name}' for derived feature '{derived_spec['name']}'"
                        )

                # Perform operation in UNCALED space
                if derived_spec["op"] == "add":
                    derived_feat = arg_data[0] + arg_data[1]
                elif derived_spec["op"] == "subtract":
                    derived_feat = arg_data[0] - arg_data[1]
                elif derived_spec["op"] == "magnitude":
                    derived_feat = torch.sqrt(arg_data[0] ** 2 + arg_data[1] ** 2)
                else:
                    raise ValueError(f"Rollout: Unknown op '{derived_spec['op']}'")

                derived_state_features_list.append(derived_feat.unsqueeze(1))

            # 3. Build *full raw state*
            if derived_state_features_list:
                full_state_tensor_raw = torch.cat(
                    [next_y_t_raw] + derived_state_features_list, dim=1
                )
            else:
                full_state_tensor_raw = next_y_t_raw

            # 4. Scale the *full raw state* to be the next input
            # Note: y_mean/y_std must have shape (num_state + num_derived)
            current_full_state_scaled = (full_state_tensor_raw - y_mean) / y_std

            # 5. Update the raw state for the *next* loop's delta calculation
            current_y_t_raw = next_y_t_raw

            # --- End derived feature logic ---

    # --- BRANCH 2: FIXED-HORIZON ROLLOUT ---
    else:
        print(
            f"Starting {rollout_horizon}-step fixed-horizon rollout (predicting deltas)..."
        )

        # Loop for each frame we want to generate
        for idx in tqdm(range(len(dataset)), desc=f"{rollout_horizon}-Step Rollout"):

            # 1. Determine the *start* of this mini-rollout
            start_idx = max(0, idx - rollout_horizon + 1)

            # 2. Determine how many steps to run
            steps_to_run = idx - start_idx + 1

            # 3. Get the *ground truth* state at the *start* of the mini-rollout
            gt_batch_start = dataset.get(start_idx).to(device)

            # This is the SCALED full state [scaled_WD, ..., scaled_SSH]
            current_full_state_scaled = gt_batch_start.x[
                :, state_block_start_idx:state_block_end_idx
            ].clone()

            # This is the UNCALED base state [WD, VX, VY]
            current_y_t_raw = (
                gt_batch_start.x[:, state_base_start_idx:state_base_end_idx].clone()
                * y_std[:num_state_features]
            ) + y_mean[:num_state_features]

            # 4. Run the inner mini-rollout loop
            # This loop is identical to the one in Branch 1, just for fewer steps
            for k in range(steps_to_run):
                # Get the *forcing data* for step 'k' of this rollout
                forcing_batch_idx = start_idx + k

                if forcing_batch_idx >= len(dataset):
                    break  # Should not happen if logic is correct, but safe_guard

                gt_forcing_batch = dataset.get(forcing_batch_idx).to(device)

                # Create input, but swap in our predicted state
                pred_input_batch = gt_forcing_batch.clone()
                pred_input_batch.x[:, state_block_start_idx:state_block_end_idx] = (
                    current_full_state_scaled
                )

                # 1. Predict scaled delta
                pred_scaled_delta = model.model(pred_input_batch)
                # 2. Unscale delta
                pred_raw_delta = (pred_scaled_delta * y_delta_std) + y_delta_mean
                # 3. Apply to unscaled base state
                next_y_t_raw = current_y_t_raw + pred_raw_delta

                # --- 4. Re-compute derived features (in UNCALED space) ---
                y_t_dict_gpu = {
                    var: next_y_t_raw[:, i]
                    for i, var in enumerate(list(features_cfg.state))
                }
                derived_state_features_list = []
                for derived_spec in features_cfg.derived_state:
                    arg_data = []
                    for arg_name in derived_spec["args"]:
                        if arg_name in y_t_dict_gpu:
                            arg_data.append(y_t_dict_gpu[arg_name])
                        elif arg_name == "DEM":
                            arg_data.append(dem_gpu)
                        else:
                            raise ValueError(
                                f"Rollout: Unknown arg '{arg_name}' for derived feature '{derived_spec['name']}'"
                            )

                    if derived_spec["op"] == "add":
                        derived_feat = arg_data[0] + arg_data[1]
                    elif derived_spec["op"] == "subtract":
                        derived_feat = arg_data[0] - arg_data[1]
                    elif derived_spec["op"] == "magnitude":
                        derived_feat = torch.sqrt(arg_data[0] ** 2 + arg_data[1] ** 2)
                    else:
                        raise ValueError(f"Rollout: Unknown op '{derived_spec['op']}'")
                    derived_state_features_list.append(derived_feat.unsqueeze(1))

                # 5. Build new full UNCALED state
                if derived_state_features_list:
                    full_state_tensor_raw = torch.cat(
                        [next_y_t_raw] + derived_state_features_list, dim=1
                    )
                else:
                    full_state_tensor_raw = next_y_t_raw

                # 6. Re-scale full state for next model input
                current_full_state_scaled = (full_state_tensor_raw - y_mean) / y_std
                # 7. Update unscaled base state for next delta
                current_y_t_raw = next_y_t_raw
                # --- End derived features ---

            # 5. After the inner loop, 'current_y_t_raw' holds the
            # final N-step-ahead prediction. Store it.
            predictions_list.append(current_y_t_raw.cpu().numpy())

    print("Rollout complete.")
    return predictions_list
