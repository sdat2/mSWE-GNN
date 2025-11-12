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
python -m mswegnn.utils.adforce_predict_animate \
    -c /path/to/your/config.yaml \
    -ckpt /path/to/your/model.ckpt \
    -nc /path/to/your/152_KATRINA_2005.nc \
    -r 3
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

# --- IMPORTS ---
import lightning as L
from mswegnn.training.adforce_train import AdforceLightningModule
from mswegnn.models.adforce_models import (
    GNNModelAdforce,
    PointwiseMLPModel,
    MonolithicMLPModel,
)
from mswegnn.utils.adforce_dataset import AdforceLazyDataset
from mswegnn.utils.load import read_config
from mswegnn.utils.miscellaneous import get_model
from sithom.plot import plot_defaults, label_subplots

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
        features_cfg (Dict[str, Any]): The 'features' block from the config,
                                       used to find 'DEM'.

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

    try:
        # DEM is found dynamically from the config
        static_vars_list = list(features_cfg.static)
        dem_idx = static_vars_list.index("DEM")

        dem = dataset.static_data["static_node_features"][:, dem_idx].cpu().numpy()
    except Exception as e:
        print(f"Failed to get DEM from dataset.static_data.")
        print(f"Could not find 'DEM' in features_cfg.static: {static_vars_list}")
        raise e

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

    predictions_list = []

    # --- Get feature counts from config ---
    num_static_features = len(features_cfg.static)
    num_forcing_features = len(features_cfg.forcing)
    num_state_features = len(features_cfg.state)
    p_t = dataset.previous_t

    # Calculate state indices in the x tensor
    # x tensor structure: [static, forcing (p_t * N_forcing), state]
    state_start_idx = num_static_features + (num_forcing_features * p_t)
    state_end_idx = state_start_idx + num_state_features

    # --- BRANCH 1: FULL, FREE-RUNNING ROLLOUT ---
    if rollout_horizon == -1:
        print("Starting full, free-running rollout (predicting deltas)...")

        # --- 1. Get the *initial state* from frame 0 ---
        current_batch = dataset.get(0).to(device)
        # Find state vector dynamically
        current_y_t_scaled = current_batch.x[
            :, state_start_idx:state_end_idx
        ].clone()  # State y(t)
        current_y_t_raw = (current_y_t_scaled * y_std) + y_mean

        for idx in tqdm(range(len(dataset)), desc="Full Rollout"):
            # 1. Get the *ground truth batch* for this step's *forcing*
            gt_batch = dataset.get(idx).to(device)

            # 2. Create the *prediction input*
            pred_input_batch = gt_batch.clone()

            # 3. ...but replace the state with our *predicted* state
            pred_input_batch.x[:, state_start_idx:state_end_idx] = current_y_t_scaled

            # 4. Run the model to predict the *scaled delta*
            pred_scaled_delta = model.model(pred_input_batch)

            # 5. Un-scale the predicted delta
            pred_raw_delta = (pred_scaled_delta * y_delta_std) + y_delta_mean

            # 6. Apply the delta to get the next state
            next_y_t_raw = current_y_t_raw + pred_raw_delta

            # 7. Store the *unscaled predicted state*
            predictions_list.append(next_y_t_raw.cpu().numpy())

            # 8. Prepare for the *next* loop iteration
            current_y_t_raw = next_y_t_raw
            current_y_t_scaled = (current_y_t_raw - y_mean) / y_std

    # --- BRANCH 2: FIXED-HORIZON ROLLOUT ---
    else:
        print(
            f"Starting {rollout_horizon}-step fixed-horizon rollout (predicting deltas)..."
        )

        # Loop for each frame we want to generate
        for idx in tqdm(range(len(dataset)), desc="Fixed-Horizon Rollout"):

            # 1. Determine the *start* of this mini-rollout
            # For frame 0, start_idx=0. For frame 1, start_idx=0. ...
            # For frame 3 (n=3), start_idx=1.
            start_idx = max(0, idx - rollout_horizon + 1)

            # 2. Determine how many steps to run
            # For frame 0, steps=1. For frame 1, steps=2. ...
            # For frame 2 (n=3), steps=3. For frame 3 (n=3), steps=3.
            steps_to_run = idx - start_idx + 1

            # 3. Get the *ground truth* state at the *start* of the mini-rollout
            gt_batch_start = dataset.get(start_idx).to(device)
            current_y_t_scaled = gt_batch_start.x[
                :, state_start_idx:state_end_idx
            ].clone()
            current_y_t_raw = (current_y_t_scaled * y_std) + y_mean

            # 4. Run the inner mini-rollout loop
            for k in range(steps_to_run):
                # Get the *forcing data* for step 'k' of this rollout
                forcing_batch_idx = start_idx + k

                # This should not happen if steps_to_run is correct, but as a safeguard:
                if forcing_batch_idx >= len(dataset):
                    break

                gt_forcing_batch = dataset.get(forcing_batch_idx).to(device)

                pred_input_batch = gt_forcing_batch.clone()
                pred_input_batch.x[:, state_start_idx:state_end_idx] = (
                    current_y_t_scaled
                )

                pred_scaled_delta = model.model(pred_input_batch)
                pred_raw_delta = (pred_scaled_delta * y_delta_std) + y_delta_mean

                # Update the state for the next inner-loop step
                current_y_t_raw = current_y_t_raw + pred_raw_delta
                current_y_t_scaled = (current_y_t_raw - y_mean) / y_std

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
    num_static = len(features_cfg.static)
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
    # If 'SSH' *is* in the state vector, this logic needs to adapt
    ssh_data = wd_data + dem

    return {
        "P": p_data,
        "WX": wx_data,
        "WY": wy_data,
        "SSH": ssh_data,
        "VX": vx_data,
        "VY": vy_data,
    }


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

    # --- Create dynamic index maps ---
    try:
        forcing_vars_list = list(features_cfg.forcing)
        state_vars_list = list(features_cfg.state)

        idx_map_forcing = {
            "P": forcing_vars_list.index("P"),
            "WX": forcing_vars_list.index("WX"),
            "WY": forcing_vars_list.index("WY"),
        }
        idx_map_state = {
            "WD": state_vars_list.index("WD"),
            "VX": state_vars_list.index("VX"),
            "VY": state_vars_list.index("VY"),
        }
    except ValueError as e:
        print(f"Error: A required plotting variable is missing from config.features.")
        print(f"Needed: P, WX, WY in features.forcing")
        print(f"Needed: WD, VX, VY in features.state")
        raise e

    # --- Get feature counts for slicing ---
    p_t = dataset.previous_t
    num_static = len(features_cfg.static)
    num_forcing = len(features_cfg.forcing)
    forcing_end_idx = num_static + (num_forcing * p_t)
    last_forcing_step_start = forcing_end_idx - num_forcing

    for idx in tqdm(range(len(dataset)), desc="Scanning data"):
        data_gt = dataset.get(idx)

        # data.y_unscaled is y_tplus1_raw (the full state)
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

    cfg = read_config(args.config_path)
    features_cfg = cfg.features

    # --- 3. CONFIGURE OUTPUTS ---
    ROLLOUT_HORIZON = args.rollout_horizon
    rollout_type_str = "full" if ROLLOUT_HORIZON == -1 else f"{ROLLOUT_HORIZON}step"
    output_gif = f"adforce_6panel_PREDICTION_{rollout_type_str}.gif"
    output_video = f"adforce_6panel_PREDICTION_{rollout_type_str}.mp4"
    anim_fps = 10

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

    # --- 5. INITIALIZE DATASET (Config-Driven) ---
    print(f"Initializing dataset for {args.netcdf_file}...")

    # Get required params from config
    # TODO: 'predict_root' is still hard-coded from the original script.
    # Consider moving this to data_params.predict_root in your config.
    predict_root = "/Volumes/s/tcpips/mSWE-GNN/data_processed/predict_katrina"
    previous_t = cfg.model_params.previous_t
    scaling_stats_path = cfg.data_params.scaling_stats_path

    frame_dir = f"./animation_frames_temp_PREDICT_{rollout_type_str}"
    shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir, exist_ok=True)

    try:
        dataset = AdforceLazyDataset(
            root=predict_root,
            nc_files=[args.netcdf_file],
            previous_t=previous_t,
            scaling_stats_path=scaling_stats_path,
            features_cfg=features_cfg,  # <-- The new required argument
        )
    except Exception as e:
        print(f"Failed to initialize AdforceLazyDataset: {e}")
        # This will now catch errors if your stats file doesn't match
        # your config, thanks to the sanity check you added!
        exit()

    if len(dataset) == 0:
        print("Dataset is empty. Check time steps and p_t.")
        exit()

    total_frames = len(dataset)
    print(f"Dataset loaded. Total samples to predict: {total_frames}")

    # --- 6. CONFIGURE AND LOAD MODEL (Config-Driven) ---
    print(f"Loading model from {args.checkpoint_path}...")

    # --- A. Dynamically calculate model dimensions from config ---
    num_static_node_features = len(features_cfg.static)
    num_dynamic_node_features = len(features_cfg.forcing)
    # The state can include derived features, so we count them all
    num_current_state_features = len(features_cfg.state)
    if features_cfg.get("derived_state"):
        num_current_state_features += len(features_cfg.derived_state)

    num_node_features = (
        num_static_node_features
        + (num_dynamic_node_features * previous_t)
        + num_current_state_features
    )
    num_edge_features = len(features_cfg.edge)

    # Model predicts the delta for the state (which includes derived)
    num_output_features = num_current_state_features

    print(f"Model dimensions calculated from config:")
    print(f"  num_node_features: {num_node_features}")
    print(f"  num_edge_features: {num_edge_features}")
    print(f"  num_output_features: {num_output_features}")

    # --- B. Get model parameters from config ---
    model_parameters = dict(cfg.models)  # Make a copy
    model_type = model_parameters.pop("model_type")

    # Handle GNNModelAdforce-specific params that were in config
    # but not in 'models' block
    model_parameters["num_static_features"] = num_static_node_features

    # --- C. Instantiate the underlying model ---
    try:
        model_to_load = get_model(model_type)(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            previous_t=previous_t,
            num_output_features=num_output_features,
            **model_parameters,
        )
        print(f"Instantiated {model_type} model structure.")

    except Exception as e:
        print(f"Error: Failed to instantiate model structure: {e}")
        print(
            "\nCheck if your config.models block is compatible with the model's __init__."
        )
        print(f"Params passed: {model_parameters}")
        exit()

    # --- D. Load the AdforceLightningModule from checkpoint ---
    try:
        lightning_model = AdforceLightningModule.load_from_checkpoint(
            args.checkpoint_path,
            map_location=device,
            model=model_to_load,
            lr_info=cfg.lr_info,  # <-- From config
            trainer_options=cfg.trainer_options,  # <-- From config
        )
        lightning_model.to(device)
        lightning_model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model checkpoint: {e}")
        print("\nThis often happens if the model architecture in your config")
        print("does not match the architecture *saved* in the checkpoint.")
        exit()

    # --- 7. LOAD STATIC DATA & CLIMITS ---
    x_coords, y_coords, dem = load_static_data(args.netcdf_file, dataset, features_cfg)
    climits = calculate_global_climits(dataset, dem, features_cfg)

    # --- 8. PERFORM ROLLOUT ---
    all_predictions = perform_rollout(
        lightning_model,
        dataset,
        device,
        features_cfg,  # <-- New argument
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

    # Create the dynamic index maps ONCE
    # This relies on the lists in calculate_global_climits
    # being the source of truth for plotting.
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
    except ValueError as e:
        print(f"Error: A required plotting variable is missing from config.features.")
        print(f"Needed: P, WX, WY in features.forcing")
        print(f"Needed: WD, VX, VY in features.state")
        exit()

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
            features_cfg=features_cfg,  # <-- New argument
            plot_idx_map=plot_idx_map,  # <-- New argument
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
