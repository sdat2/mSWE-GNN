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

This script can run multiple rollouts at once by passing a list
of horizons to the -r flag.

*** NEW ***
This script now ALSO saves the raw numerical predictions for each
rollout horizon into a NetCDF file (e.g., 'adforce_PREDICTION_full.nc'),
including all coordinates (time, x, y, DEM) for easy analysis.

Example Usage:
# Create a unique output directory
export RUN_ID=my_katrina_run_01
export RUN_DIR=/work/scratch-pw3/sithom/animation_runs/$RUN_ID
mkdir -p $RUN_DIR

# Run a full rollout (-1) and two fixed-horizon rollouts (12, 24)
python -m mswegnn.utils.adforce_predict_animate \
    -c /path/to/your/config.yaml \
    -ckpt /path/to/your/model.ckpt \
    -nc /path/to/your/152_KATRINA_2005.nc \
    -r -1 12 24 \
    -o $RUN_DIR \
    -s /optional/override/path/to/scaling_stats.yaml

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
from mswegnn.utils.adforce_rollout import perform_rollout, load_static_data
import cmocean


# Suppress Matplotlib/Numpy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_frame_data(
    dataset: AdforceLazyDataset,
    idx: int,
    dem: np.ndarray,
    prediction_state: np.ndarray,
    features_cfg: Dict[str, Any],
    plot_idx_map: Dict[str, Dict[str, int]],
    scaling_stats: Dict[str, Any],  # <-- ADDED
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
        scaling_stats (Dict[str, Any]): The loaded scaling_stats.yaml dict.

    Returns:
        Dict[str, np.ndarray]: A dictionary holding the 6 plotting variables.
    """
    # Get the ground-truth data batch, which contains the *forcing* we need
    data = dataset.get(idx)
    p_t = dataset.previous_t
    x_data = data.x.cpu()  # x_data is SCALED

    # --- 1. Extract and Un-scale Inputs (P, WX, WY) ---
    num_static = len(features_cfg.static) + 1  # +1 for node_type
    num_forcing = len(features_cfg.forcing)

    forcing_start_idx = num_static
    forcing_end_idx = num_static + (num_forcing * p_t)

    # Get the *last* forcing step available in this batch
    # (which corresponds to the state we are plotting)
    last_forcing_step_start = forcing_end_idx - num_forcing
    last_forcing_step_scaled = x_data[:, last_forcing_step_start:forcing_end_idx]

    # --- Unscale forcing data using the stats dict ---
    try:
        # Load stats to CPU (as data is on CPU)
        x_dyn_mean = torch.tensor(scaling_stats["x_dynamic_mean"], dtype=torch.float32)
        x_dyn_std = torch.tensor(
            scaling_stats["x_dynamic_std"], dtype=torch.float32
        ).clamp(min=1e-6)
    except (KeyError, TypeError) as e:
        print(f"Error: Scaling stats dict missing x_dynamic keys: {e}")
        raise e

    # We need the mean/std for a *single* step.
    # This assumes the order in x_dyn_mean matches features_cfg.forcing
    mean = x_dyn_mean[:num_forcing]
    std = x_dyn_std[:num_forcing]
    inputs_unscaled = (last_forcing_step_scaled * std) + mean

    # Find P, WX, WY dynamically using the pre-computed index map
    idx_map_forcing = plot_idx_map["forcing"]
    wx_data = inputs_unscaled[:, idx_map_forcing["WX"]].numpy()
    wy_data = inputs_unscaled[:, idx_map_forcing["WY"]].numpy()
    p_data = inputs_unscaled[:, idx_map_forcing["P"]].numpy()

    # --- 2. Extract Outputs (WD, VX, VY) from the prediction ---
    outputs = prediction_state  # Use the passed-in UNCALED state from the rollout

    # Find WD, VX, VY dynamically
    idx_map_state = plot_idx_map["state"]
    wd_data = outputs[:, idx_map_state["WD"]]
    vx_data = outputs[:, idx_map_state["VX"]]
    vy_data = outputs[:, idx_map_state["VY"]]

    # --- 3. Calculate SSH ---
    # This assumes 'SSH' is a derived feature and not in the state vector
    # We add UNCALED WD to UNCALED DEM
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
        # Ensure we are using simple lists, not OmegaConf lists
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
    dataset: AdforceLazyDataset,
    dem: np.ndarray,
    features_cfg: Dict[str, Any],
    scaling_stats: Dict[str, Any],  # <-- ADDED
) -> Dict[str, Tuple[float, float]]:
    """
    Calculates global vmin/vmax by iterating through the *GROUND TRUTH* dataset.

    Args:
        dataset (AdforceLazyDataset): The initialized dataset.
        dem (np.ndarray): The DEM data.
        features_cfg (Dict[str, Any]): The 'features' block from the config.
        scaling_stats (Dict[str, Any]): The loaded scaling_stats.yaml dict.

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

    # --- Load scaling stats from dict ---
    try:
        x_dyn_mean = torch.tensor(scaling_stats["x_dynamic_mean"], dtype=torch.float32)
        x_dyn_std = torch.tensor(
            scaling_stats["x_dynamic_std"], dtype=torch.float32
        ).clamp(min=1e-6)
    except (KeyError, TypeError) as e:
        print(f"Error: Scaling stats dict missing x_dynamic keys: {e}")
        raise e

    # Get single-step stats (on CPU)
    mean = x_dyn_mean[:num_forcing]
    std = x_dyn_std[:num_forcing]

    for idx in tqdm(range(len(dataset)), desc="Scanning data"):
        # Get ground truth data for this step
        data_gt = dataset.get(idx)

        # data.y_unscaled is y_tplus1_raw (the base state), UNCALED
        outputs_gt = data_gt.y_unscaled.cpu().numpy()

        wd_gt = outputs_gt[:, idx_map_state["WD"]]
        vx_gt = outputs_gt[:, idx_map_state["VX"]]
        vy_gt = outputs_gt[:, idx_map_state["VY"]]
        ssh_gt = wd_gt + dem  # (UNSCALED + UNCALED)

        # Get input data (SCALED)
        x_data = data_gt.x.cpu()
        last_forcing_step_scaled = x_data[:, last_forcing_step_start:forcing_end_idx]

        # --- Unscale forcing data using the stats dict ---
        inputs_unscaled = (last_forcing_step_scaled * std) + mean

        wx_gt = inputs_unscaled[:, idx_map_forcing["WX"]].numpy()
        wy_gt = inputs_unscaled[:, idx_map_forcing["WY"]].numpy()
        p_gt = inputs_unscaled[:, idx_map_forcing["P"]].numpy()

        # Collate all UNCALED ground truth data
        data_dict = {
            "P": p_gt,
            "WX": wx_gt,
            "WY": wy_gt,
            "SSH": ssh_gt,
            "VX": vx_gt,
            "VY": vy_gt,
        }

        # Store percentiles
        for key, data in data_dict.items():
            if data.size > 0:
                p2_vals[key].append(np.nanpercentile(data, 1))
                p98_vals[key].append(np.nanpercentile(data, 99))

    # Calculate global limits from all stored percentiles
    climits = {}
    for key in plot_order:
        if not p2_vals[key]:  # Handle empty data
            climits[key] = (0.0, 1.0)
            continue

        global_p2 = np.nanmin(p2_vals[key])
        global_p98 = np.nanmax(p98_vals[key])

        if key in diverging_vars:
            # Center diverging maps at 0
            v_abs = np.nanmax([np.abs(global_p2), np.abs(global_p98)])
            if v_abs == 0:
                v_abs = 0.1
            climits[key] = (-v_abs, v_abs)
        else:
            # Standard min/max for sequential maps
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
    scaling_stats: Dict[str, Any],  # <-- ADDED
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
        scaling_stats (Dict[str, Any]): The loaded scaling_stats.yaml dict.
    """

    # 1. Get all 6 plot variables (UNSCALED)
    data_dict = get_frame_data(
        dataset,
        idx,
        dem,
        prediction_state,  # This is the UNCALED predicted state
        features_cfg,
        plot_idx_map,
        scaling_stats,  # <-- Pass stats
    )

    # 2. Get timestamp for title
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

    # 3. Create 2x3 plot
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
            data = data_dict[key]  # Use UNCALED data
            vmin, vmax = climits[key]  # Use global UNCALED limits

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

    # 4. Save figure
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
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="UNIQUE base directory to save all outputs (cache, frames, video).",
    )
    # --- START: MODIFIED ARGUMENT ---
    parser.add_argument(
        "-r",
        "--rollout_horizon",
        type=int,
        nargs="+",  # <-- Accept one or more inputs as a list
        default=[-1],
        help="One or more rollout strategies. -1 for 'Full Rollout', N > 0 for 'Fixed N-step Horizon'. E.g., -r -1 12 24",
    )
    # --- END: MODIFIED ARGUMENT ---
    parser.add_argument(
        "-s",
        "--scaling_stats_path",
        type=str,
        help="Path to scaling stats. (Optional; read from config if not provided)",
        default=None,
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
    # --- MODIFIED: Get list of horizons ---
    rollout_horizons_list = args.rollout_horizon
    anim_fps = 10
    base_output_dir = args.output_dir

    # Dataset cache path is shared for all rollouts, defined once
    predict_root = os.path.join(base_output_dir, "dataset_cache")
    # --- END MODIFIED ---

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

    # --- 5. LOAD SCALING STATS (UPDATED BLOCK) ---
    # Get path from -s argument, fall back to config if not provided
    scaling_stats_path = (
        args.scaling_stats_path
        if args.scaling_stats_path is not None
        else cfg.data_params.scaling_stats_path
    )

    print(f"Loading scaling stats from: {scaling_stats_path}")
    if args.scaling_stats_path is not None:
        print("(Using path from -s/--scaling_stats_path argument)")
    else:
        print("(Using path from config.yaml: data_params.scaling_stats_path)")

    if not os.path.exists(scaling_stats_path):
        print(
            f"FATAL: Scaling stats file not found at the specified path: {scaling_stats_path}"
        )
        exit()
    try:
        with open(scaling_stats_path, "r") as f:
            scaling_stats = yaml.safe_load(f)
        print("Scaling stats loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load or parse {scaling_stats_path}: {e}")
        exit()
    # --- END 5. ---

    # --- 6. INITIALIZE DATASET (Run Once) ---
    # We only need to build the dataset and its index map once.
    print(f"Initializing dataset for {args.netcdf_file}...")
    print(f"Dataset cache will be in: {predict_root}")

    previous_t = cfg.model_params.previous_t

    # --- NEW: Clean up *dataset cache* once ---
    shutil.rmtree(predict_root, ignore_errors=True)
    # AdforceLazyDataset will create the predict_root

    try:
        dataset = AdforceLazyDataset(
            root=predict_root,  # <-- Use the new cache path
            nc_files=[args.netcdf_file],
            previous_t=previous_t,
            scaling_stats_path=scaling_stats_path,  # <-- Pass the determined path
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

    # --- 7. CONFIGURE AND LOAD MODEL (Run Once) ---
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

    # --- 8. LOAD STATIC DATA & CLIMITS (Run Once) ---
    # These are the same for all rollouts, so we do it once.
    x_coords, y_coords, dem = load_static_data(args.netcdf_file, dataset, features_cfg)
    climits = calculate_global_climits(
        dataset, dem, features_cfg, scaling_stats
    )  # <-- Pass stats

    # --- Create the dynamic plot variable index maps (Run Once) ---
    plot_idx_map = _create_plot_index_map(features_cfg)

    # --- NEW: Load time coordinates (Run Once) ---
    # We need this to save in the output NetCDF
    print(f"Loading time coordinates from {args.netcdf_file}...")
    try:
        with xr.open_dataset(args.netcdf_file) as ds:
            # The predictions start at frame `previous_t` and go for `total_frames`
            time_slice = slice(previous_t, previous_t + total_frames)
            time_coords = ds["time"].isel(time=time_slice).values

        if len(time_coords) != total_frames:
            print(
                f"Warning: Time coordinate length ({len(time_coords)}) does not match total frames ({total_frames}). Using simple index."
            )
            time_coords = np.arange(total_frames)
    except Exception as e:
        print(
            f"Warning: Could not load time coordinates. Using simple index. Error: {e}"
        )
        time_coords = np.arange(total_frames)
    # --- END NEW ---

    # --- 9. START MAIN ROLLOUT LOOP ---
    # This loop will run once for each horizon provided (e.g., -r -1 12 24)
    print(
        f"\nFound {len(rollout_horizons_list)} rollout(s) to run: {rollout_horizons_list}"
    )

    for rollout_horizon in rollout_horizons_list:

        # === STARTING ROLLOUT FOR HORIZON: {rollout_horizon} ===

        # --- 9a. Define unique names for this horizon ---
        # This creates names like "full" or "12step"
        rollout_type_str = "full" if rollout_horizon == -1 else f"{rollout_horizon}step"
        print(
            f"\n--- Starting Rollout: {rollout_type_str} (Horizon={rollout_horizon}) ---"
        )

        # Create dynamic output paths based on the rollout horizon
        output_gif = os.path.join(
            base_output_dir, f"adforce_6panel_PREDICTION_{rollout_type_str}.gif"
        )
        output_video = os.path.join(
            base_output_dir, f"adforce_6panel_PREDICTION_{rollout_type_str}.mp4"
        )
        frame_dir = os.path.join(
            base_output_dir, f"animation_frames_{rollout_type_str}"
        )

        print(f"Animation frames will be in: {frame_dir}")

        # Clean up *frame dir* for this specific rollout
        shutil.rmtree(frame_dir, ignore_errors=True)
        os.makedirs(frame_dir, exist_ok=True)

        # --- 9b. PERFORM ROLLOUT ---
        all_predictions = perform_rollout(
            lightning_model,
            dataset,
            device,
            features_cfg,
            scaling_stats,  # <-- Pass stats
            rollout_horizon=rollout_horizon,  # <-- Pass the *current* horizon
        )

        if len(all_predictions) != total_frames:
            print(
                f"Error: Rollout returned {len(all_predictions)} frames, expected {total_frames}"
            )
            continue  # Skip to the next horizon

        # --- 9c. NEW: SAVE PREDICTIONS TO NETCDF ---
        print(f"Saving raw predictions to NetCDF for '{rollout_type_str}'...")

        # Define the path for the prediction .nc file
        output_pred_nc = os.path.join(
            base_output_dir, f"adforce_PREDICTION_{rollout_type_str}.nc"
        )

        try:
            # Stack the list of arrays into one big array
            # Shape becomes: (total_frames, num_nodes, num_state_features)
            pred_data_np = np.stack(all_predictions, axis=0)

            # Get the names of the predicted variables (e.g., ['WD', 'VX', 'VY'])
            var_coords = list(features_cfg.state)

            # Create a dictionary for the xarray.Dataset
            # This maps variable names to their data, with correct dims
            data_vars = {
                var_coords[i]: (("time", "num_nodes"), pred_data_np[:, :, i])
                for i in range(len(var_coords))
            }

            # Create the Dataset
            pred_ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": time_coords,  # Use the loaded time coordinates
                    "num_nodes": np.arange(pred_data_np.shape[1]),
                    "x": ("num_nodes", x_coords),  # Add static coords
                    "y": ("num_nodes", y_coords),
                    "DEM": ("num_nodes", dem),
                },
            )

            # Save to file
            pred_ds.to_netcdf(output_pred_nc)
            print(f"Prediction data saved to {os.path.abspath(output_pred_nc)}")

        except Exception as e:
            print(f"--- WARNING: Failed to save prediction NetCDF ---")
            print(f"Error: {e}")
        # --- END NEW SECTION ---

        # --- 9d. RENDER FRAMES ---
        print(f"Rendering {total_frames} predicted frames for '{rollout_type_str}'...")
        frame_files = []

        # Iterate and render one frame at a time
        for idx in tqdm(
            range(total_frames), desc=f"Rendering frames ({rollout_type_str})"
        ):
            # Use the dynamic frame_dir path
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
                frame_path,  # Pass the unique frame path
                prediction_state=prediction_for_this_frame,
                features_cfg=features_cfg,
                plot_idx_map=plot_idx_map,
                scaling_stats=scaling_stats,  # <-- Pass stats
            )

        # --- 9e. COMPILE & CLEANUP ---
        images = []
        if output_gif or output_video:
            # Read all saved frames from the unique frame_dir
            for frame_file in tqdm(
                frame_files, desc=f"Reading frames ({rollout_type_str})"
            ):
                images.append(iio.imread(frame_file))

        if output_gif:
            # Save to the unique .gif path
            compile_gif_from_frames(frame_dir, output_gif, anim_fps, images)

        if output_video:
            # Save to the unique .mp4 path
            compile_video_from_frames(frame_dir, output_video, anim_fps, images)

        # We keep the frames for inspection, just print the path
        print(f"Temporary frames saved in {os.path.abspath(frame_dir)}")

        print(f"\nPrediction animation for '{rollout_type_str}' complete.")
        if output_gif:
            print(f"GIF saved to {os.path.abspath(output_gif)}")
        if output_video:
            print(f"Video saved to {os.path.abspath(output_video)}")

        # === FINISHED ROLLOUT FOR HORIZON: {rollout_horizon} ===

    print("\nAll requested rollouts are complete.")
