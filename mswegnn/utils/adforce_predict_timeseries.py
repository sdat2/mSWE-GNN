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
"""

import os
import shutil
import glob
import warnings
from typing import List, Tuple, Dict
import numpy as np
import xarray as xr
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.dates as mdates

# --- Imports from your project ---
import lightning as L
from mswegnn.training.adforce_train import AdforceLightningModule
from mswegnn.models.adforce_models import (
    GNNModelAdforce,
    PointwiseMLPModel,
    MonolithicMLPModel,
)
from mswegnn.utils.adforce_dataset import AdforceLazyDataset
from sithom.plot import plot_defaults

# --- Import the functions from the animation script ---
# We assume this script is in the same directory (mswegnn/utils)
# If not, you may need to adjust the import path.
try:
    from mswegnn.utils.adforce_predict_animate import (
        perform_rollout,
        load_static_data,
    )
except ImportError:
    print("Error: Could not import from adforce_predict_animate.py.")
    print("Please ensure both scripts are in the 'mswegnn/utils' directory.")
    exit()


# Suppress Matplotlib/Numpy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning)


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
        node_index (int): The index of the node to extract.
        dem_at_node (float): The DEM value at that node.

    Returns:
        np.ndarray: A 1D array of the SSH time series.
    """
    ssh_series = []
    for pred_state in all_predictions:
        # pred_state shape is [N_nodes, 3] (WD, VX, VY)
        wd_at_node = pred_state[node_index, 0]
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
    for idx in tqdm(range(len(dataset)), desc="Reading Ground Truth"):
        # data.y_unscaled is the unscaled state [WD, VX, VY] at t+1
        data = dataset.get(idx)
        wd_at_node = data.y_unscaled.cpu().numpy()[node_index, 0]
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
    """
    print("Plotting comparison graph...")
    plot_defaults()

    fig, ax = plt.subplots(1, 1)

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
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    # they need to be rotated to 90 degrees to fit
    # plt.gcf().autofmt_xdate() # Auto-rotate dates
    # plt.gcf().autofmt_xdate() # Auto-rotate dates

    ax.set_xlabel("Date & Time (UTC)")
    ax.set_ylabel("Sea Surface Height (SSH) [m]")
    # ax.set_title(f"SSH Time Series Comparison near New Orleans\nNode Index: {node_index} (Coords: {target_coords[0]:.4f}, {target_coords[1]:.4f})")
    ax.legend()
    ax.grid(True, which="major", linestyle="--", alpha=0.5)

    # Save the figure
    output_filename = f"ssh_timeseries_comparison_node_{node_index}.pdf"
    plt.tight_layout()
    fig.savefig(output_filename, dpi=300)
    print(f"Graph saved to {os.path.abspath(output_filename)}")
    plt.close(fig)


if __name__ == "__main__":
    # python -m mswegnn.utils.adforce_predict_timeseries
    # --- 1. CONFIGURE YOUR PATHS HERE ---
    checkpoint_path = (
        "/Volumes/s/tcpips/mSWE-GNN/checkpoints/GNN-best-epoch=37-val_loss=0.5033.ckpt"
    )
    root_directory = "/Volumes/s/tcpips/swegnn_5sec/"
    netcdf_file = os.path.join(root_directory, "152_KATRINA_2005.nc")
    scaling_stats_file = (
        "/Volumes/s/tcpips/mSWE-GNN/data_processed/train/scaling_stats.yaml"
    )

    # --- 2. CONFIGURE ROLLOUT ---
    previous_time_steps = 1

    # Define the target location
    TARGET_LON = -90.0715  # New Orleans Lon
    TARGET_LAT = 29.9511  # New Orleans Lat

    # Define which N-step horizon to test (besides the full rollout)
    N_STEP_HORIZON = 3

    # --- 3. CONFIGURE MODEL PARAMETERS ---
    # (These must match the loaded checkpoint)
    model_type = "GNN"
    model_params = {
        "model_type": "GNN",
        "type_gnn": "SWEGNN",
        "hid_features": 64,
        "mlp_layers": 2,
        "K": 3,
        "normalize": True,
        "gnn_activation": "tanh",
        "edge_mlp": True,
        "with_gradient": True,
    }
    mock_lr_info = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "step_size": 20,
        "gamma": 0.5,
    }
    mock_trainer_options = {
        "batch_size": 4,
        "only_where_water": True,
        "velocity_scaler": 5.0,
        "type_loss": "RMSE",
    }

    # --- 4. SETUP DEVICE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 5. INITIALIZE DATASET ---
    print(f"Initializing dataset for {netcdf_file}...")
    try:
        predict_root = "/Volumes/s/tcpips/mSWE-GNN/data_processed/predict_katrina"
        dataset = AdforceLazyDataset(
            root=predict_root,
            nc_files=[netcdf_file],
            previous_t=previous_time_steps,
            scaling_stats_path=scaling_stats_file,
        )
    except Exception as e:
        print(f"Failed to initialize AdforceLazyDataset: {e}")
        exit()

    total_frames = len(dataset)
    print(f"Dataset loaded. Total samples: {total_frames}")

    # --- 6. LOAD MODEL ---
    print(f"Loading model from {checkpoint_path}...")

    p_t = previous_time_steps
    NUM_STATIC_NODE_FEATURES = 5
    NUM_DYNAMIC_NODE_FEATURES = 3
    NUM_CURRENT_STATE_FEATURES = 3
    NUM_STATIC_EDGE_FEATURES = 2
    NUM_OUTPUT_FEATURES = 3

    num_node_features = (
        NUM_STATIC_NODE_FEATURES
        + (NUM_DYNAMIC_NODE_FEATURES * p_t)
        + NUM_CURRENT_STATE_FEATURES
    )
    num_edge_features = NUM_STATIC_EDGE_FEATURES
    num_output_features = NUM_OUTPUT_FEATURES

    try:
        model_to_load = GNNModelAdforce(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            previous_t=p_t,
            num_output_features=num_output_features,
            num_static_features=NUM_STATIC_NODE_FEATURES,
            **model_params,
        )
        lightning_model = AdforceLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            model=model_to_load,
            lr_info=mock_lr_info,
            trainer_options=mock_trainer_options,
        )
        lightning_model.to(device)
        lightning_model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model checkpoint: {e}")
        exit()

    # --- 7. FIND NODE & EXTRACT STATIC DATA ---
    x_coords, y_coords, dem = load_static_data(netcdf_file, dataset)
    node_index = find_closest_node(x_coords, y_coords, TARGET_LON, TARGET_LAT)
    dem_at_node = dem[node_index]
    target_coords_found = (x_coords[node_index], y_coords[node_index])

    # --- 8. EXTRACT TIME AXIS ---
    time_axis = get_time_axis(dataset)

    # --- 9. EXTRACT GROUND TRUTH ---
    gt_ssh = extract_ground_truth_ssh(dataset, node_index, dem_at_node)

    # --- 10. RUN FULL ROLLOUT ---
    preds_full = perform_rollout(
        lightning_model, dataset, device, rollout_horizon=-1  # -1 for full rollout
    )
    full_rollout_ssh = extract_ssh_timeseries(preds_full, node_index, dem_at_node)

    # --- 11. RUN N-STEP ROLLOUT ---
    preds_n_step = perform_rollout(
        lightning_model, dataset, device, rollout_horizon=N_STEP_HORIZON
    )
    n_step_ssh = extract_ssh_timeseries(preds_n_step, node_index, dem_at_node)

    # --- 12. PLOT RESULTS ---
    plot_comparison_timeseries(
        time_axis,
        node_index,
        target_coords_found,
        gt_ssh,
        full_rollout_ssh,
        n_step_ssh,
        n_step_val=N_STEP_HORIZON,
    )

    print("\nTime series analysis complete.")
