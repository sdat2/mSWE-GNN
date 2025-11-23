"""
Time Series Analysis Script for mSWE-GNN Rollouts.

This script loads a trained model and a dataset, then runs
multiple rollout scenarios (e.g., full free-running and 3-step horizon).

It identifies a specific node in the mesh (e.g., closest to
New Orleans) and plots a time series comparison of the
Sea Surface Height (SSH) for:
1. Ground Truth
2. Full Rollout (horizon = -1)
3. N-Step Horizon (e.g., horizon = 3)

--- REFACTOR ---
This script is now config-driven and command-line operated.
It relies on the 'config.yaml' file saved in the checkpoint
directory to load the model and dataset with the correct feature
configuration. It reuses the rollout and data-loading functions
from the utility scripts.

Example Usage:
python -m mswegnn.utils.adforce_predict_timeseries \
    -ckpt /path/to/model/GNN-best.ckpt \
    -nc /path/to/data/152_KATRINA_2005.nc \
    -o /path/to/outputs/katrina_ssh_plot.pdf \
    --lon -90.0715 \
    --lat 29.9511 \
    --horizon 3
"""

import os
import shutil
import glob
import warnings
import argparse
from typing import List, Tuple, Dict
import numpy as np
import xarray as xr
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.dates as mdates
import lightning as L
from mswegnn.utils.adforce_dataset import AdforceLazyDataset
from sithom.plot import plot_defaults
from omegaconf import OmegaConf
from mswegnn.utils.adforce_misc import model_from_cfg_and_checkpoint
from mswegnn.utils.adforce_rollout import perform_rollout, load_static_data


# Suppress Matplotlib/Numpy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning)


plot_defaults()


def find_closest_node(
    x_coords: np.ndarray, y_coords: np.ndarray, target_lon: float, target_lat: float
) -> int:
    """
    Finds the index of the node closest to the target coordinates.

    Args:
        x_coords (np.ndarray): Array of node longitudes.
        y_coords (np.ndarray): Array of node latitudes.
        target_lon (float): Target longitude.
        target_lat (float): Target latitude.

    Returns:
        int: The index of the closest node.

    Doctest:
    >>> import numpy as np
    >>> x = np.array([-90.0, -89.0, -88.0])
    >>> y = np.array([29.0, 30.0, 31.0])
    >>> find_closest_node(x, y, -88.1, 30.9)
    Searching for closest node to (-88.1, 30.9)...
    Found closest node at index: 2
      -> Coords: (-88.0000, 31.0000)
    2
    """
    print(f"Searching for closest node to ({target_lon}, {target_lat})...")
    # Calculate squared Euclidean distance
    dist_sq = (x_coords - target_lon) ** 2 + (y_coords - target_lat) ** 2
    node_index = np.argmin(dist_sq)

    print(f"Found closest node at index: {node_index}")
    print(f"  -> Coords: ({x_coords[node_index]:.4f}, {y_coords[node_index]:.4f})")
    return int(node_index)


def get_time_axis(dataset: AdforceLazyDataset) -> List[np.datetime64]:
    """
    Extracts the list of datetimes for the plot's x-axis.

    Args:
        dataset (AdforceLazyDataset): The initialized dataset.

    Returns:
        List[np.datetime64]: A list of datetime objects for each frame.
    """
    print("Extracting time axis from dataset...")
    time_axis = []
    for idx in tqdm(range(len(dataset)), desc="Reading timestamps"):
        try:
            nc_path, t_start = dataset.index_map[idx]
            t_plot_idx = t_start + dataset.previous_t
            with xr.open_dataset(nc_path, cache=True) as ds:
                timestamp = ds.time[t_plot_idx].values
                time_axis.append(timestamp)
        except Exception as e:
            print(f"Warning: Could not read timestamp for index {idx}. Error: {e}")
            time_axis.append(np.datetime64("NaT"))
    return time_axis


def extract_ssh_timeseries(
    all_predictions: List[np.ndarray], node_index: int, dem_at_node: float
) -> np.ndarray:
    """
    Extracts the SSH time series for a single node from a list of predictions.

    Args:
        all_predictions (List[np.ndarray]): The output from perform_rollout.
                                            Each item is [N_nodes, N_state_features].
        node_index (int): The index of the node to extract.
        dem_at_node (float): The DEM value at that node.

    Returns:
        np.ndarray: A 1D array of the SSH time series.
    """
    ssh_series = []
    # We assume 'WD' is the first feature (index 0) in the state.
    # This is a strong assumption but consistent with the original script.
    # A safer way would be to get the WD index from features_cfg.state.
    wd_feature_index = 0  # Assuming 'WD' is the first target variable

    for pred_state in all_predictions:
        # pred_state shape is [N_nodes, N_state_features] (e.g., WD, VX, VY)
        wd_at_node = pred_state[node_index, wd_feature_index]
        ssh_at_node = wd_at_node + dem_at_node
        ssh_series.append(ssh_at_node)
    return np.array(ssh_series)


def extract_ground_truth_ssh(
    dataset: AdforceLazyDataset, node_index: int, dem_at_node: float
) -> np.ndarray:
    """
    Extracts the ground truth SSH time series for a single node.

    Args:
        dataset (AdforceLazyDataset): The initialized dataset.
        node_index (int): The index of the node to extract.
        dem_at_node (float): The DEM value at that node.

    Returns:
        np.ndarray: A 1D array of the ground truth SSH time series.
    """
    print("Extracting ground truth SSH time series...")
    ssh_series = []
    # We assume 'WD' is the first feature (index 0) in the *unscaled* target.
    # This is consistent with AdforceLazyDataset's `get()` method.
    wd_feature_index = 0  # Assuming 'WD' is the first target variable

    for idx in tqdm(range(len(dataset)), desc="Reading Ground Truth"):
        # data.y_unscaled is the unscaled state [WD, VX, VY] at t+1
        data = dataset.get(idx)
        wd_at_node = data.y_unscaled.cpu().numpy()[node_index, wd_feature_index]
        ssh_at_node = wd_at_node + dem_at_node
        ssh_series.append(ssh_at_node)
    return np.array(ssh_series)


def plot_comparison_timeseries(
    time_axis: List,
    node_index: int,
    target_coords: Tuple[float, float],
    gt_ssh: np.ndarray,
    full_rollout_ssh: np.ndarray,
    n_step_ssh: np.ndarray,
    n_step_val: int,
    output_pdf_path: str,  # <-- [NEW] Argument for output path
):
    """
    Plots the three SSH time series on a single graph and saves it.

    Args:
        time_axis (List): List of datetime objects.
        node_index (int): The plotted node's index.
        target_coords (Tuple[float, float]): The (lon, lat) of the target.
        gt_ssh (np.ndarray): Ground truth SSH time series.
        full_rollout_ssh (np.ndarray): Full rollout SSH time series.
        n_step_ssh (np.ndarray): N-step horizon SSH time series.
        n_step_val (int): The 'N' value for the N-step plot (e.g., 3).
        output_pdf_path (str): The full path to save the output PDF file.
    """
    print("Plotting comparison graph...")
    plot_defaults()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Plot the data
    ax.plot(time_axis, gt_ssh, label="Ground Truth", color="black", linewidth=2)
    ax.plot(
        time_axis,
        full_rollout_ssh,
        label="Full Rollout (horizon = -1)",
        color="red",
        linestyle="--",
        alpha=0.9,
    )
    ax.plot(
        time_axis,
        n_step_ssh,
        label=f"{n_step_val}-Step Horizon",
        color="blue",
        linestyle=":",
        alpha=0.9,
    )

    # Format the x-axis for datetimes
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.xticks(rotation=90)  # Rotate x-axis labels

    ax.set_xlabel("Date & Time (UTC)")
    ax.set_ylabel("Sea Surface Height (SSH) [m]")
    ax.set_title(
        f"SSH Time Series Comparison near ({target_coords[0]:.4f}, {target_coords[1]:.4f}) (Node {node_index})"
    )
    ax.legend()
    ax.grid(True, which="major", linestyle="--", alpha=0.5)

    # Save the figure
    # --- [UPDATED] Save to the specified output path ---
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_pdf_path, dpi=300, bbox_inches="tight")
    print(f"Graph saved to {os.path.abspath(output_pdf_path)}")
    plt.close(fig)


if __name__ == "__main__":
    # --- 1. [NEW] CONFIGURE ARGPARSE ---
    parser = argparse.ArgumentParser(
        description="Run mSWE-GNN time series comparison for a specific node."
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the .ckpt model checkpoint file.",
    )
    parser.add_argument(
        "-nc",
        "--netcdf_file",
        type=str,
        required=True,
        help="Path to the single .nc file to analyze (e.g., '152_KATRINA_2005.nc').",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output PDF plot (e.g., 'katrina_ssh_new_orleans.pdf').",
    )
    parser.add_argument(
        "--lon",
        type=float,
        required=True,
        help="Target longitude for time series (e.g., -90.0715 for New Orleans).",
    )
    parser.add_argument(
        "--lat",
        type=float,
        required=True,
        help="Target latitude for time series (e.g., 29.9511 for New Orleans).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="N-step horizon to compare against (default: 3).",
    )
    args = parser.parse_args()

    # --- 2. SETUP DEVICE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. [NEW] LOAD CONFIGURATION ---
    print(f"Loading config from checkpoint directory...")
    config_path = os.path.join(os.path.dirname(args.checkpoint_path), "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.yaml not found at: {config_path}\n"
            f"This script relies on the 'config.yaml' file saved by 'adforce_main.py'."
        )

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)  # Resolve any interpolations
    features_cfg = cfg.features  # Get the features block

    # --- [NEW] Get paths and params from config ---
    scaling_stats_file = cfg.data_params.scaling_stats_path
    previous_time_steps = cfg.model_params.previous_t

    print("--- Script Configuration ---")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  NetCDF File: {args.netcdf_file}")
    print(f"  Output PDF: {args.output_file}")
    print(f"  Scaling Stats: {scaling_stats_file}")
    print(f"  Target Coords: ({args.lon}, {args.lat})")
    print(f"  N-Step Horizon: {args.horizon}")
    print("----------------------------")

    # --- 4. [UPDATED] INITIALIZE DATASET ---
    print(f"Initializing dataset for {args.netcdf_file}...")

    # Create a unique root for this script's dataset cache
    predict_root_dir = os.path.join(
        os.path.dirname(args.output_file), "timeseries_cache"
    )

    try:
        dataset = AdforceLazyDataset(
            root=predict_root_dir,
            nc_files=[args.netcdf_file],
            previous_t=previous_time_steps,
            scaling_stats_path=scaling_stats_file,
            features_cfg=features_cfg,  # <-- THE CRITICAL ADDITION
        )
    except Exception as e:
        print(f"Failed to initialize AdforceLazyDataset: {e}")
        exit()

    total_frames = len(dataset)
    print(f"Dataset loaded. Total samples: {total_frames}")

    # --- 5. [UPDATED] LOAD MODEL ---
    print(f"Loading model from {args.checkpoint_path}...")
    try:
        lightning_model = model_from_cfg_and_checkpoint(cfg, args.checkpoint_path)
        lightning_model.to(device)
        lightning_model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model checkpoint: {e}")
        exit()

    # --- 6. [UPDATED] FIND NODE & EXTRACT STATIC DATA ---
    x_coords, y_coords, dem = load_static_data(
        args.netcdf_file, dataset, features_cfg=features_cfg
    )
    node_index = find_closest_node(x_coords, y_coords, args.lon, args.lat)
    dem_at_node = dem[node_index]
    target_coords_found = (x_coords[node_index], y_coords[node_index])

    # --- 7. EXTRACT TIME AXIS ---
    time_axis = get_time_axis(dataset)

    # --- 8. EXTRACT GROUND TRUTH ---
    gt_ssh = extract_ground_truth_ssh(dataset, node_index, dem_at_node)

    # --- 9. [UPDATED] RUN FULL ROLLOUT ---
    print("Running full rollout (horizon = -1)...")
    preds_full = perform_rollout(
        lightning_model,
        dataset,
        device,
        features_cfg=features_cfg,
        rollout_horizon=-1,  # -1 for full rollout
    )
    full_rollout_ssh = extract_ssh_timeseries(preds_full, node_index, dem_at_node)

    # --- 10. [UPDATED] RUN N-STEP ROLLOUT ---
    print(f"Running N-Step rollout (horizon = {args.horizon})...")
    preds_n_step = perform_rollout(
        lightning_model,
        dataset,
        device,
        features_cfg=features_cfg,
        rollout_horizon=args.horizon,
    )
    n_step_ssh = extract_ssh_timeseries(preds_n_step, node_index, dem_at_node)

    # --- 11. [UPDATED] PLOT RESULTS ---
    plot_comparison_timeseries(
        time_axis,
        node_index,
        target_coords_found,
        gt_ssh,
        full_rollout_ssh,
        n_step_ssh,
        n_step_val=args.horizon,
        output_pdf_path=args.output_file,  # <-- Pass the output path
    )

    print("\nTime series analysis complete.")
