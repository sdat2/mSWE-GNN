"""
Prediction animation function for the AdforceLazyDataset.

This script loads a trained mSWE-GNN model and runs an
autoregressive rollout (prediction) for a full simulation event.
It is driven by a config file and a model checkpoint.

This version supports two rollout modes via ROLLOUT_HORIZON:
1. (N > 0): "Fixed Horizon" mode. Each frame 'k' in the animation
   shows the result of an N-step prediction that *started* at
   frame 'k - N + 1'. This is computationally intensive as it
   re-runs the rollout for every frame.
2. (N = -1): "Full Rollout" mode. Runs a single, free-running
   simulation from t=0. Each frame 'k' shows the result of a
   'k'-step-long prediction.

Example Usage:
# Create a unique output directory
export RUN_ID=my_katrina_run_01
export RUN_DIR=/work/scratch-pw3/sithom/animation_runs/$RUN_ID
mkdir -p $RUN_DIR

python -m mswegnn.utils.adforce_predict_animate \
    -c /path/to/your/config.yaml \
    -ckpt /path/to/your/model.ckpt \
    -nc /path/to/your/152_KATRINA_2005.nc \
    -r -1 \
    -o $RUN_DIR

"""

import os
import shutil
import glob
import warnings
import argparse
import yaml
from typing import List, Tuple, Dict, Any
import numpy as np
import xarray as xr
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio.v3 as iio
from omegaconf import OmegaConf


# --- IMPORTS ---
from sithom.plot import plot_defaults, label_subplots
import lightning as L
from mswegnn.training.adforce_train import AdforceLightningModule
from mswegnn.utils.adforce_dataset import AdforceLazyDataset
from mswegnn.utils.adforce_misc import model_from_cfg_and_checkpoint


# --- END IMPORTS ---

# Try to import cmocean
try:
    import cmocean
except ImportError:
    print(
        "Error: 'cmocean' library not found. Please install it:"
        "\n  pip install cmocean"
    )
    exit()

# Suppress Matplotlib/Numpy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning)


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


@torch.no_grad()  # We don't need gradients for inference
def perform_rollout(
    model: L.LightningModule,
    dataset: AdforceLazyDataset,
    device: torch.device,
    features_cfg: Dict[str, Any],
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

    # --- Get all necessary scaling stats from the dataset ---
    y_mean = dataset.y_mean.to(device)
    y_std = dataset.y_std.to(device)
    y_delta_mean = dataset.y_delta_mean.to(device)
    y_delta_std = dataset.y_delta_std.to(device)

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

            # 7. Store the *unscaled predicted base state*
            predictions_list.append(next_y_t_raw.cpu().numpy())

            # 8. Prepare for the *next* loop iteration

            # --- 8a. Re-calculate derived features ---

            # 1. We have `next_y_t_raw` (base state) [N, 3]

            # 2. Get component features from `next_y_t_raw`
            y_t_dict_gpu = {
                var: next_y_t_raw[:, i]
                for i, var in enumerate(list(features_cfg.state))
            }

            # 3. Build derived features list
            derived_state_features_list = []
            for derived_spec in features_cfg.derived_state:
                arg_data = []
                for arg_name in derived_spec["args"]:
                    if arg_name in y_t_dict_gpu:
                        arg_data.append(y_t_dict_gpu[arg_name])
                    elif arg_name == "DEM":  # HACK: hard-coding static features
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

            # 4. Build *full raw state*
            if derived_state_features_list:
                full_state_tensor_raw = torch.cat(
                    [next_y_t_raw] + derived_state_features_list, dim=1
                )
            else:
                full_state_tensor_raw = next_y_t_raw

            # 5. Scale the *full raw state* to be the next input
            # Note: y_mean/y_std must have shape (num_state + num_derived)
            current_full_state_scaled = (full_state_tensor_raw - y_mean) / y_std
            current_y_t_raw = next_y_t_raw  # for the next loop's delta

            # --- End derived feature logic ---

    # --- BRANCH 2: FIXED-HORIZON ROLLOUT ---
    else:
        print(
            f"Starting {rollout_horizon}-step fixed-horizon rollout (predicting deltas)..."
        )

        # Loop for each frame we want to generate
        for idx in tqdm(range(len(dataset)), desc="Fixed-Horizon Rollout"):

            # 1. Determine the *start* of this mini-rollout
            start_idx = max(0, idx - rollout_horizon + 1)

            # 2. Determine how many steps to run
            steps_to_run = idx - start_idx + 1

            # 3. Get the *ground truth* state at the *start* of the mini-rollout
            gt_batch_start = dataset.get(start_idx).to(device)

            current_full_state_scaled = gt_batch_start.x[
                :, state_block_start_idx:state_block_end_idx
            ].clone()

            current_y_t_raw = (
                gt_batch_start.x[:, state_base_start_idx:state_base_end_idx].clone()
                * y_std[:num_state_features]
            ) + y_mean[:num_state_features]

            # 4. Run the inner mini-rollout loop
            for k in range(steps_to_run):
                # Get the *forcing data* for step 'k' of this rollout
                forcing_batch_idx = start_idx + k

                if forcing_batch_idx >= len(dataset):
                    break

                gt_forcing_batch = dataset.get(forcing_batch_idx).to(device)

                pred_input_batch = gt_forcing_batch.clone()
                pred_input_batch.x[:, state_block_start_idx:state_block_end_idx] = (
                    current_full_state_scaled
                )

                pred_scaled_delta = model.model(pred_input_batch)
                pred_raw_delta = (pred_scaled_delta * y_delta_std) + y_delta_mean

                # Update the state for the next inner-loop step
                next_y_t_raw = current_y_t_raw + pred_raw_delta

                # --- Re-compute derived features ---
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

                if derived_state_features_list:
                    full_state_tensor_raw = torch.cat(
                        [next_y_t_raw] + derived_state_features_list, dim=1
                    )
                else:
                    full_state_tensor_raw = next_y_t_raw

                current_full_state_scaled = (full_state_tensor_raw - y_mean) / y_std
                current_y_t_raw = next_y_t_raw
                # --- End derived features ---

            # 5. After the inner loop, 'current_y_t_raw' holds the
            # final N-step-ahead prediction. Store it.
            predictions_list.append(current_y_t_raw.cpu().numpy())

    print("Rollout complete.")
    return predictions_list


def get_frame_data(
    dataset: AdforceLazyDataset,
    idx: int,
    dem: np.ndarray,
    prediction_state: np.ndarray,
    features_cfg: Dict[str, Any],
    plot_idx_map: Dict[str, Dict[str, int]],
) -> Dict[str, np.ndarray]:
    """
    Retrieves and processes all 6 variables for a single animation frame.
    Uses ground-truth inputs (P, WX, WY) but predicted outputs.

    Args:
        dataset (AdforceLazyDataset): The initialized dataset.
        idx (int): The frame index to get.
        dem (np.ndarray): The DEM data.
        prediction_state (np.ndarray): The UNCALED state y(t+1) from rollout.
        features_cfg (Dict[str, Any]): The 'features' block from the config.
        plot_idx_map (Dict[str, Dict[str, int]]): A map to find plot variables
            (P, WX, WY, WD, VX, VY) in the config-ordered lists.

    Returns:
        Dict[str, np.ndarray]: A dictionary holding the 6 plotting variables.
    """
    data = dataset.get(idx)
    p_t = dataset.previous_t
    x_data = data.x.cpu()

    # --- 1. Extract and Un-scale Inputs (P, WX, WY) ---
    num_static = len(features_cfg.static) + 1  # +1 for node_type
    num_forcing = len(features_cfg.forcing)

    forcing_start_idx = num_static
    forcing_end_idx = num_static + (num_forcing * p_t)

    # Get the *last* forcing step available in this batch
    # (which corresponds to the state we are plotting)
    last_forcing_step_start = forcing_end_idx - num_forcing
    last_forcing_step_scaled = x_data[:, last_forcing_step_start:forcing_end_idx]

    if dataset.apply_scaling:
        # We need the mean/std for a *single* step, not broadcasted
        # This assumes the order in x_dyn_mean matches features_cfg.forcing
        mean = dataset.x_dyn_mean_broadcast.cpu()[:num_forcing]
        std = dataset.x_dyn_std_broadcast.cpu()[:num_forcing]
        inputs_unscaled = (last_forcing_step_scaled * std) + mean
    else:
        inputs_unscaled = last_forcing_step_scaled

    # Find P, WX, WY dynamically
    idx_map_forcing = plot_idx_map["forcing"]
    wx_data = inputs_unscaled[:, idx_map_forcing["WX"]].numpy()
    wy_data = inputs_unscaled[:, idx_map_forcing["WY"]].numpy()
    p_data = inputs_unscaled[:, idx_map_forcing["P"]].numpy()

    # --- 2. Extract Outputs (WD, VX, VY) from the prediction ---
    outputs = prediction_state  # Use the passed-in unscaled state

    # Find WD, VX, VY dynamically
    idx_map_state = plot_idx_map["state"]
    wd_data = outputs[:, idx_map_state["WD"]]
    vx_data = outputs[:, idx_map_state["VX"]]
    vy_data = outputs[:, idx_map_state["VY"]]

    # --- 3. Calculate SSH ---
    # This assumes 'SSH' is a derived feature and not in the state vector
    ssh_data = wd_data + dem

    return {
        "P": p_data,
        "WX": wx_data,
        "WY": wy_data,
        "SSH": ssh_data,
        "VX": vx_data,
        "VY": vy_data,
    }


# --- REUSE: New helper function to centralize index map creation ---
def _create_plot_index_map(features_cfg: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Creates a mapping from plot variable names to their index in the config lists."""
    try:
        forcing_vars_list = list(features_cfg.forcing)
        state_vars_list = list(features_cfg.state)

        plot_idx_map = {
            "forcing": {
                "P": forcing_vars_list.index("P"),
                "WX": forcing_vars_list.index("WX"),
                "WY": forcing_vars_list.index("WY"),
            },
            "state": {
                "WD": state_vars_list.index("WD"),
                "VX": state_vars_list.index("VX"),
                "VY": state_vars_list.index("VY"),
            },
        }
        return plot_idx_map
    except ValueError as e:
        print(f"Error: A required plotting variable is missing from config.features.")
        print(f"Needed: P, WX, WY in features.forcing")
        print(f"Needed: WD, VX, VY in features.state")
        raise e


# --- END REUSE ---


def calculate_global_climits(
    dataset: AdforceLazyDataset, dem: np.ndarray, features_cfg: Dict[str, Any]
) -> Dict[str, Tuple[float, float]]:
    """
    Calculates global vmin/vmax by iterating through the *GROUND TRUTH* dataset.

    Args:
        dataset (AdforceLazyDataset): The initialized dataset.
        dem (np.ndarray): The DEM data.
        features_cfg (Dict[str, Any]): The 'features' block from the config.

    Returns:
        Dict[str, Tuple[float, float]]: Global color limits for plot variables.
    """
    print(
        f"Calculating global color limits from GROUND TRUTH data ({len(dataset)} frames)..."
    )

    plot_order = ["P", "WX", "WY", "SSH", "VX", "VY"]
    diverging_vars = ["WX", "WY", "SSH", "VX", "VY"]

    p2_vals = {key: [] for key in plot_order}
    p98_vals = {key: [] for key in plot_order}

    # --- REUSE: Call the new helper function ---
    plot_idx_map = _create_plot_index_map(features_cfg)
    idx_map_forcing = plot_idx_map["forcing"]
    idx_map_state = plot_idx_map["state"]
    # --- END REUSE ---

    # --- Get feature counts for slicing ---
    p_t = dataset.previous_t
    num_static = len(features_cfg.static) + 1  # +1 for node_type
    num_forcing = len(features_cfg.forcing)
    forcing_end_idx = num_static + (num_forcing * p_t)
    last_forcing_step_start = forcing_end_idx - num_forcing

    for idx in tqdm(range(len(dataset)), desc="Scanning data"):
        data_gt = dataset.get(idx)

        # data.y_unscaled is y_tplus1_raw (the base state)
        outputs_gt = data_gt.y_unscaled.cpu().numpy()

        wd_gt = outputs_gt[:, idx_map_state["WD"]]
        vx_gt = outputs_gt[:, idx_map_state["VX"]]
        vy_gt = outputs_gt[:, idx_map_state["VY"]]
        ssh_gt = wd_gt + dem

        # Get input data
        x_data = data_gt.x.cpu()
        last_forcing_step_scaled = x_data[:, last_forcing_step_start:forcing_end_idx]

        if dataset.apply_scaling:
            mean = dataset.x_dyn_mean_broadcast.cpu()[:num_forcing]
            std = dataset.x_dyn_std_broadcast.cpu()[:num_forcing]
            inputs_unscaled = (last_forcing_step_scaled * std) + mean
        else:
            inputs_unscaled = last_forcing_step_scaled

        wx_gt = inputs_unscaled[:, idx_map_forcing["WX"]].numpy()
        wy_gt = inputs_unscaled[:, idx_map_forcing["WY"]].numpy()
        p_gt = inputs_unscaled[:, idx_map_forcing["P"]].numpy()

        data_dict = {
            "P": p_gt,
            "WX": wx_gt,
            "WY": wy_gt,
            "SSH": ssh_gt,
            "VX": vx_gt,
            "VY": vy_gt,
        }

        for key, data in data_dict.items():
            if data.size > 0:
                p2_vals[key].append(np.nanpercentile(data, 1))
                p98_vals[key].append(np.nanpercentile(data, 99))

    climits = {}
    for key in plot_order:
        if not p2_vals[key]:
            climits[key] = (0.0, 1.0)
            continue

        global_p2 = np.nanmin(p2_vals[key])
        global_p98 = np.nanmax(p98_vals[key])

        if key in diverging_vars:
            v_abs = np.nanmax([np.abs(global_p2), np.abs(global_p98)])
            if v_abs == 0:
                v_abs = 0.1
            climits[key] = (-v_abs, v_abs)
        else:
            if global_p2 == global_p98:
                global_p98 += 0.1
            climits[key] = (global_p2, global_p98)

    print("Global color limits calculated from ground truth.")
    for key, (vmin, vmax) in climits.items():
        print(f"  {key}: ({vmin:.2f}, {vmax:.2f})")

    return climits


def plot_single_frame(
    idx: int,
    total_frames: int,
    dataset: AdforceLazyDataset,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    dem: np.ndarray,
    climits: Dict[str, Tuple[float, float]],
    frame_path: str,
    prediction_state: np.ndarray,
    features_cfg: Dict[str, Any],
    plot_idx_map: Dict[str, Dict[str, int]],
):
    """
    Creates, plots, and saves a *single* frame from scratch.

    Args:
        idx (int): Frame index.
        total_frames (int): Total frames for title.
        dataset (AdforceLazyDataset): The dataset.
        x_coords (np.ndarray): Node x-coordinates.
        y_coords (np.ndarray): Node y-coordinates.
        dem (np.ndarray): DEM data.
        climits (Dict[str, Tuple[float, float]]): Color limits.
        frame_path (str): Path to save the PNG file.
        prediction_state (np.ndarray): The unscaled predicted state.
        features_cfg (Dict[str, Any]): The 'features' block from the config.
        plot_idx_map (Dict[str, Dict[str, int]]): Map to find plot variables.
    """

    data_dict = get_frame_data(
        dataset, idx, dem, prediction_state, features_cfg, plot_idx_map
    )

    try:
        nc_path, t_start = dataset.index_map[idx]
        t_plot_idx = t_start + dataset.previous_t
        with xr.open_dataset(nc_path, cache=True) as ds:
            timestamp = ds.time[t_plot_idx].values
            title = (
                np.datetime_as_string(timestamp, unit="s").replace("T", " ")[:-6]
                + ":00"
            )
    except Exception as e:
        if idx == 0:
            print(
                f"Warning: Could not read timestamp. Falling back to index. Error: {e}"
            )
        title = f"Dataset Index: {idx} / {total_frames - 1}"

    fig, axs = plt.subplots(2, 3, figsize=(6 * 1.2, 4 * 1.2), sharex=True, sharey=True)

    titles = [
        ["P [m]", "WX [m s$^{-1}$]", "WY [m s$^{-1}$]"],
        ["SSH [m]", "VX [m s$^{-1}$]", "VY [m s$^{-1}$]"],
    ]
    keys = [["P", "WX", "WY"], ["SSH", "VX", "VY"]]
    cmaps = [
        [cmocean.cm.thermal, cmocean.cm.balance, cmocean.cm.balance],
        [cmocean.cm.balance, cmocean.cm.balance, cmocean.cm.balance],
    ]

    for i in range(2):
        for j in range(3):
            ax = axs[i, j]
            key = keys[i][j]
            data = data_dict[key]
            vmin, vmax = climits[key]

            scat = ax.scatter(
                x_coords,
                y_coords,
                c=data,
                cmap=cmaps[i][j],
                s=0.2,
                marker=".",
                vmin=vmin,
                vmax=vmax,
            )
            fig.colorbar(scat, ax=ax)
            ax.set_title(titles[i][j])
            ax.set_aspect("equal")

            if i == 1:
                ax.set_xlabel("Longitude [$^{\circ}$E]")
            if j == 0:
                ax.set_ylabel("Latitude [$^{\circ}$N]")

            if x_coords.size > 0 and y_coords.size > 0:
                ax.set_xlim(np.nanmin(x_coords), np.nanmax(x_coords))
                ax.set_ylim(np.nanmin(y_coords), np.nanmax(y_coords))

    fig.suptitle(title, y=0.92)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    label_subplots(axs)
    fig.savefig(frame_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compile_gif_from_frames(
    frame_dir: str, output_gif_path: str, fps: int, images: List[np.ndarray]
):
    """Compiles a list of image arrays into a GIF."""
    if not images:
        print("No images found for GIF compilation.")
        return
    print(f"Compiling {len(images)} frames into {output_gif_path}...")
    iio.imwrite(output_gif_path, images, fps=fps, loop=0)
    print("GIF compilation complete.")


def compile_video_from_frames(
    frame_dir: str, output_video_path: str, fps: int, images: List[np.ndarray]
):
    """Compiles a list of image arrays into an MP4 video."""
    if not images:
        print("No images found for Video compilation.")
        return
    print(f"Compiling {len(images)} frames into {output_video_path}...")
    try:
        iio.imwrite(
            output_video_path,
            images,
            fps=fps,
            codec="libx264",
            quality=9,
            pixelformat="yuv420p",
        )
        print("Video compilation complete.")
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"Video compilation failed: {e}")
        print("This *likely* means the 'imageio-ffmpeg' plugin is not installed.")
        print("Please install it and try again:")
        print("  pip install imageio[ffmpeg]")
        print("-------------")


if __name__ == "__main__":
    # --- 1. CONFIGURE ARGPARSE ---
    parser = argparse.ArgumentParser(
        description="Run mSWE-GNN prediction rollout and generate animations."
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=True,
        help="Path to the config.yaml file used for training.",
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the .ckpt model checkpoint file to use for inference.",
    )
    parser.add_argument(
        "-nc",
        "--netcdf_file",
        type=str,
        required=True,
        help="Path to the single .nc file to animate (e.g., '152_KATRINA_2005.nc').",
    )
    # --- FIX: Changed -o to be required and serve as the base path ---
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="UNIQUE base directory to save all outputs (cache, frames, video).",
    )
    # --predict_root is now derived from -o
    parser.add_argument(
        "-r",
        "--rollout_horizon",
        type=int,
        default=-1,
        help="Rollout strategy. -1 for 'Full Rollout', N > 0 for 'Fixed N-step Horizon'.",
    )
    args = parser.parse_args()

    # --- 2. LOAD CONFIG AND FEATURES ---
    print(f"Loading config from {args.config_path}...")
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found at {args.config_path}")
        exit()
    cfg = OmegaConf.load(args.config_path)
    features_cfg = cfg.features

    # --- 3. CONFIGURE OUTPUTS (Now based on -o) ---
    ROLLOUT_HORIZON = args.rollout_horizon
    rollout_type_str = "full" if ROLLOUT_HORIZON == -1 else f"{ROLLOUT_HORIZON}step"
    anim_fps = 10

    # --- NEW: Define all paths based on the required output_dir ---
    base_output_dir = args.output_dir
    output_gif = os.path.join(
        base_output_dir, f"adforce_6panel_PREDICTION_{rollout_type_str}.gif"
    )
    output_video = os.path.join(
        base_output_dir, f"adforce_6panel_PREDICTION_{rollout_type_str}.mp4"
    )

    # Cache and frames are now subdirectories
    predict_root = os.path.join(base_output_dir, "dataset_cache")
    frame_dir = os.path.join(base_output_dir, "animation_frames")
    # --- END NEW PATHS ---

    # --- 4. SETUP DEVICE ---
    plot_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check for file existence
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        exit()
    if not os.path.exists(args.netcdf_file):
        print(f"Error: NetCDF file not found at {args.netcdf_file}")
        exit()
    if not os.path.exists(cfg.data_params.scaling_stats_path):
        print(
            f"Error: Scaling stats file not found at {cfg.data_params.scaling_stats_path}"
        )
        print(
            "(This path is read from your config.yaml: data_params.scaling_stats_path)"
        )
        exit()

    # --- 5. INITIALIZE DATASET (Using new paths) ---
    print(f"Initializing dataset for {args.netcdf_file}...")
    print(f"Dataset cache will be in: {predict_root}")
    print(f"Animation frames will be in: {frame_dir}")

    previous_t = cfg.model_params.previous_t
    scaling_stats_path = cfg.data_params.scaling_stats_path

    # --- NEW: Clean up *inside* the unique directory ---
    # This deletes old cache/frames if you re-run with the *same* -o path
    shutil.rmtree(frame_dir, ignore_errors=True)
    shutil.rmtree(predict_root, ignore_errors=True)

    # Create the base and frame directories
    os.makedirs(frame_dir, exist_ok=True)
    # AdforceLazyDataset will create the predict_root

    try:
        dataset = AdforceLazyDataset(
            root=predict_root,  # <-- Use the new cache path
            nc_files=[args.netcdf_file],
            previous_t=previous_t,
            scaling_stats_path=scaling_stats_path,
            features_cfg=features_cfg,
        )
    except Exception as e:
        print(f"Failed to initialize AdforceLazyDataset: {e}")
        exit()

    if len(dataset) == 0:
        print("Dataset is empty. Check time steps and p_t.")
        exit()

    total_frames = len(dataset)
    print(f"Dataset loaded. Total samples to predict: {total_frames}")

    # --- 6. CONFIGURE AND LOAD MODEL (Config-Driven) ---
    print(f"Loading model from {args.checkpoint_path}...")

    try:
        lightning_model = model_from_cfg_and_checkpoint(
            cfg,
            args.checkpoint_path,
        )
        lightning_model.to(device)  # --- Don't forget to move model to device! ---
    except Exception as e:
        print(f"Error: Failed to instantiate model structure: {e}")
        print(
            "\nCheck if your config.models block is compatible with the model's __init__."
        )
        exit()

    # --- 7. LOAD STATIC DATA & CLIMITS ---
    x_coords, y_coords, dem = load_static_data(args.netcdf_file, dataset, features_cfg)
    climits = calculate_global_climits(dataset, dem, features_cfg)

    # --- 8. PERFORM ROLLOUT ---
    all_predictions = perform_rollout(
        lightning_model,
        dataset,
        device,
        features_cfg,
        rollout_horizon=ROLLOUT_HORIZON,
    )

    if len(all_predictions) != total_frames:
        print(
            f"Error: Rollout returned {len(all_predictions)} frames, expected {total_frames}"
        )
        exit()

    # --- 9. RENDER FRAMES ---
    print("Rendering predicted frames...")
    frame_files = []

    # --- REUSE: Create the dynamic index maps ONCE ---
    plot_idx_map = _create_plot_index_map(features_cfg)
    # --- END REUSE ---

    for idx in tqdm(range(total_frames), desc="Rendering frames"):
        frame_path = os.path.join(frame_dir, f"frame_{idx:05d}.png")
        frame_files.append(frame_path)

        prediction_for_this_frame = all_predictions[idx]

        plot_single_frame(
            idx,
            total_frames,
            dataset,
            x_coords,
            y_coords,
            dem,
            climits,
            frame_path,
            prediction_state=prediction_for_this_frame,
            features_cfg=features_cfg,
            plot_idx_map=plot_idx_map,
        )

    # --- 10. COMPILE & CLEANUP ---
    images = []
    if output_gif or output_video:
        for frame_file in tqdm(frame_files, desc="Reading frames into memory"):
            images.append(iio.imread(frame_file))

    if output_gif:
        compile_gif_from_frames(frame_dir, output_gif, anim_fps, images)

    if output_video:
        compile_video_from_frames(frame_dir, output_video, anim_fps, images)

    # Clean up temporary frame directory
    # shutil.rmtree(frame_dir, ignore_errors=True)
    print(f"Temporary frames saved in {os.path.abspath(frame_dir)}")

    print(f"\nPrediction animation complete.")
    if output_gif:
        print(f"GIF saved to {os.path.abspath(output_gif)}")
    if output_video:
        print(f"Video saved to {os.path.abspath(output_video)}")
