"""
Prediction animation function for the AdforceLazyDataset.

This script loads a trained mSWE-GNN model and runs an
autoregressive rollout (prediction) for a full simulation event.

It then creates a 6-panel animation of the *predicted* simulation
by rendering each frame as a PNG. It compiles these frames
into *both* a high-quality MP4 (for video) and a quantized GIF
(for markdown/previews).

The input forcings (P, WX, WY) are taken from the ground truth
dataset at each step, while the state (WD, VX, VY) is predicted
by the model.
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
import imageio.v3 as iio

# --- NEW IMPORTS ---
import lightning as L
from mswegnn.training.adforce_train import LightningTrainer
from mswegnn.models.adforce_models import (
    GNNModelAdforce,
    PointwiseMLPModel,
    MonolithicMLPModel,
)
# --- END NEW IMPORTS ---

from mswegnn.utils.adforce_dataset import AdforceLazyDataset
from sithom.plot import plot_defaults, label_subplots

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
    nc_file_path: str, dataset: AdforceLazyDataset
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads static coordinates and DEM data.
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
        # DEM is the 0-th column of the static_node_features
        dem = dataset.static_data["static_node_features"][:, 0].cpu().numpy()
    except Exception as e:
        print(f"Failed to get DEM from dataset.static_data: {e}")
        raise

    return x_coords, y_coords, dem


@torch.no_grad()  # We don't need gradients for inference
def perform_rollout(
    model: L.LightningModule, 
    dataset: AdforceLazyDataset, 
    device: torch.device
) -> List[np.ndarray]:
    """
    Performs an autoregressive rollout using the model for all steps in the dataset.
    
    Args:
        model: The trained Lightning model (on the correct device).
        dataset: The AdforceLazyDataset for a *single* simulation (e.g., Katrina).
        device: The torch device (e.g., 'cuda' or 'cpu').

    Returns:
        A list of numpy arrays, where each array is the unscaled predicted
        output (WD, VX, VY) for a single time step.
    """
    print("Starting autoregressive rollout...")
    model.eval()  # Set model to evaluation mode
    
    # Get unscaling parameters from the dataset
    mean = dataset.y_mean_broadcast.to(device)
    std = dataset.y_std_broadcast.to(device)

    predictions_list = []
    
    # Get the very first data sample
    # This contains the initial state (t) and forcings
    current_batch = dataset.get(0).to(device)

    for idx in tqdm(range(len(dataset)), desc="Rollout prediction"):
        # 1. Run the model on the current state (from the previous loop)
        # model.model contains the actual GNN
        pred_y_scaled = model.model(current_batch) 
        
        # 2. Un-scale the prediction
        pred_y_unscaled = (pred_y_scaled * std) + mean
        
        # 3. Store the unscaled prediction (for plotting)
        predictions_list.append(pred_y_unscaled.cpu().numpy())

        # 4. Prepare the input for the *next* time step
        if idx < len(dataset) - 1:
            # Get the ground truth data for the *next* step (idx + 1)
            # We need this to get the *correct forcing data* for the next step
            next_batch_truth = dataset.get(idx + 1).to(device)
            
            # Create the next input batch
            next_input_batch = next_batch_truth.clone()
            
            # --- This is the key autoregressive step ---
            # Replace the "current state" features of the next input
            # with the prediction we just made.
            # The last 3 features are the current state [WD, VX, VY]
            # (See adforce_main.py: NUM_CURRENT_STATE_FEATURES = 3)
            next_input_batch.x[:, -3:] = pred_y_scaled # Use scaled for next input
            
            # Update the current_batch for the next loop iteration
            current_batch = next_input_batch
            
    print("Rollout complete.")
    return predictions_list


def get_frame_data(
    dataset: AdforceLazyDataset, 
    idx: int, 
    dem: np.ndarray,
    prediction_state: np.ndarray  # <-- MODIFIED: Added prediction arg
) -> Dict[str, np.ndarray]:
    """
    Retrieves and processes all 6 variables for a single animation frame.
    Uses ground-truth inputs (P, WX, WY) but predicted outputs.
    """
    # We still get the data to extract the *input forcings*
    data = dataset.get(idx)
    p_t = dataset.previous_t
    x_data = data.x.cpu()

    # --- 1. Extract and Un-scale Inputs (P, WX, WY) ---
    forcing_start_idx = 5
    forcing_end_idx = 5 + (3 * p_t)
    last_forcing_step_scaled = x_data[:, forcing_end_idx - 3 : forcing_end_idx]

    if dataset.apply_scaling:
        if not hasattr(dataset, "x_dyn_mean_broadcast"):
            inputs_unscaled = last_forcing_step_scaled
        else:
            mean = dataset.x_dyn_mean_broadcast[-3:].cpu()
            std = dataset.x_dyn_std_broadcast[-3:].cpu()
            inputs_unscaled = (last_forcing_step_scaled * std) + mean
    else:
        inputs_unscaled = last_forcing_step_scaled
    wx_data = inputs_unscaled[:, 0].numpy()
    wy_data = inputs_unscaled[:, 1].numpy()
    p_data = inputs_unscaled[:, 2].numpy()

    # --- 2. Extract Outputs (WD, VX, VY) from the prediction ---
    outputs = prediction_state # Use the passed-in prediction
    wd_data = outputs[:, 0]
    vx_data = outputs[:, 1]
    vy_data = outputs[:, 2]

    # --- 3. Calculate SSH ---
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
    dataset: AdforceLazyDataset, dem: np.ndarray
) -> Dict[str, Tuple[float, float]]:
    """
    Calculates global vmin/vmax by iterating through the entire dataset.
    
    NOTE: This still reads the *GROUND TRUTH* data from the dataset
    to ensure the color limits are consistent and comparable.
    """
    print(f"Calculating global color limits from GROUND TRUTH data ({len(dataset)} frames)...")
    
    plot_order = ["P", "WX", "WY", "SSH", "VX", "VY"]
    diverging_vars = ["WX", "WY", "SSH", "VX", "VY"]
    
    p2_vals = {key: [] for key in plot_order}
    p98_vals = {key: [] for key in plot_order}

    for idx in tqdm(range(len(dataset)), desc="Scanning data"):
        # Get ground truth data for climits
        data_gt = dataset.get(idx)
        outputs_gt = data_gt.y_unscaled.cpu().numpy()
        wd_gt = outputs_gt[:, 0]
        vx_gt = outputs_gt[:, 1]
        vy_gt = outputs_gt[:, 2]
        ssh_gt = wd_gt + dem
        
        # Get input data (same as get_frame_data)
        p_t = dataset.previous_t
        x_data = data_gt.x.cpu()
        forcing_start_idx = 5
        forcing_end_idx = 5 + (3 * p_t)
        last_forcing_step_scaled = x_data[:, forcing_end_idx - 3 : forcing_end_idx]
        if dataset.apply_scaling:
            if not hasattr(dataset, "x_dyn_mean_broadcast"):
                inputs_unscaled = last_forcing_step_scaled
            else:
                mean = dataset.x_dyn_mean_broadcast[-3:].cpu()
                std = dataset.x_dyn_std_broadcast[-3:].cpu()
                inputs_unscaled = (last_forcing_step_scaled * std) + mean
        else:
            inputs_unscaled = last_forcing_step_scaled
        wx_gt = inputs_unscaled[:, 0].numpy()
        wy_gt = inputs_unscaled[:, 1].numpy()
        p_gt = inputs_unscaled[:, 2].numpy()

        data_dict = {
            "P": p_gt, "WX": wx_gt, "WY": wy_gt,
            "SSH": ssh_gt, "VX": vx_gt, "VY": vy_gt,
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
            if v_abs == 0: v_abs = 0.1 
            climits[key] = (-v_abs, v_abs)
        else:
            if global_p2 == global_p98: global_p98 += 0.1
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
    prediction_state: np.ndarray # <-- MODIFIED: Added prediction arg
):
    """
    Creates, plots, and saves a *single* frame from scratch.
    """
    
    data_dict = get_frame_data(dataset, idx, dem, prediction_state) 

    try:
        nc_path, t_start = dataset.index_map[idx]
        t_plot_idx = t_start + dataset.previous_t 
        with xr.open_dataset(nc_path, cache=True) as ds:
            timestamp = ds.time[t_plot_idx].values
            title = np.datetime_as_string(timestamp, unit='s').replace('T', ' ')[:-6] + ":00"
    except Exception as e:
        if idx == 0: 
            print(f"Warning: Could not read timestamp. Falling back to index. Error: {e}")
        title = f"Dataset Index: {idx} / {total_frames - 1}"

    fig, axs = plt.subplots(
        2, 3, figsize=(6*1.2, 4*1.2), 
        sharex=True, sharey=True
    )

    titles = [
        ["P [m]", "WX [m s$^{-1}$]", "WY [m s$^{-1}$]"],
        ["SSH [m]", "VX [m s$^{-1}$]", "VY [m s$^{-1}$]"],
    ]
    keys = [
        ["P", "WX", "WY"],
        ["SSH", "VX", "VY"]
    ]
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
                vmax=vmax
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
    fig.savefig(frame_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compile_gif_from_frames(
    frame_dir: str, 
    output_gif_path: str, 
    fps: int,
    images: List[np.ndarray]
):
    if not images:
        print("No images found for GIF compilation.")
        return
    print(f"Compiling {len(images)} frames into {output_gif_path}...")
    iio.imwrite(output_gif_path, images, fps=fps, loop=0)
    print("GIF compilation complete.")


def compile_video_from_frames(
    frame_dir: str, 
    output_video_path: str, 
    fps: int,
    images: List[np.ndarray]
):
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
            pixelformat="yuv420p"
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
    # --- 1. CONFIGURE YOUR PATHS HERE (UPDATED) ---
    
    # Path to your *saved model checkpoint*
    checkpoint_path = "/Volumes/s/tcpips/mSWE-GNN/checkpoints/GNN/GNN-best-epoch=53-val_loss=0.4979.ckpt"
    
    # This should be the directory containing your NetCDF file
    root_directory = "/Volumes/s/tcpips/swegnn_5sec/"
    
    # This is the single .nc file you want to animate (e.g., Katrina)
    netcdf_file = os.path.join(root_directory, "152_KATRINA_2005.nc")
    
    # This must be the *training* stats file your model was trained with
    scaling_stats_file = "/Volumes/s/tcpips/mSWE-GNN/data_processed/train/scaling_stats.yaml"
    
    # This must match the `previous_t` used to train your model
    previous_time_steps = 2
    
    # Define the output paths
    output_gif = "adforce_6panel_PREDICTION.gif"
    output_video = "adforce_6panel_PREDICTION.mp4"
    
    # Frames per second for the final animations
    anim_fps = 10
    # --- END CONFIGURATION ---


    # --- 2. SETUP DEVICE ---
    plot_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        print("Please update the 'checkpoint_path' variable in this script.")
        exit()
    if not os.path.exists(netcdf_file):
        print(f"Error: NetCDF file not found at {netcdf_file}")
        exit()
    if not os.path.exists(scaling_stats_file):
        print(f"Error: Scaling stats file not found at {scaling_stats_file}")
        exit()

    # --- 3. INITIALIZE DATASET (UPDATED) ---
    print(f"Initializing dataset for {netcdf_file}...")
    frame_dir = "./animation_frames_temp_PREDICT"
    shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir, exist_ok=True)

    try:
        # This is a temporary directory for the .pt files for *this specific file*
        # Using a path consistent with your scaling_stats_file path
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
        
    if len(dataset) == 0:
        print("Dataset is empty. Check time steps and p_t.")
        exit()
    
    total_frames = len(dataset)
    print(f"Dataset loaded. Total samples to predict: {total_frames}")


    # --- 4. CONFIGURE AND LOAD MODEL ---
    print(f"Loading model from {checkpoint_path}...")

    # --- A. !!! CONFIGURE YOUR MODEL PARAMS HERE !!! ---
    # These MUST match the 'model_params' section of the
    # config.yaml file used for the training run.
    # --- (Example for GNNModelAdforce) ---
    model_type = "GNN"
    model_params = {
        'model_type': 'GNN',
        'hidden_dim': 128,        # <--- !! CHECK THIS
        'num_layers': 5,          # <--- !! CHECK THIS
        'mp_passes': 15,          # <--- !! CHECK THIS
        'use_model_residual': True, # <--- !! CHECK THIS
        'use_edge_attr': True,    # <--- !! CHECK THIS
        'hid_features': 64,        # <--- !! CHECK THIS
        'mlp_layers': 2,          # <--- !! CHECK THIS
    }
    # --- (Example for PointwiseMLPModel) ---
    # model_type = "MLP"
    # model_params = {
    #     'model_type': 'MLP',
    #     'hidden_dim': 128,
    #     'num_layers': 3,
    #     'use_model_residual': True,
    # }
    # -------------------------------------

    # --- B. Define MOCK configs for lr_info and trainer_options ---
    # These are required to initialize the LightningTrainer class,
    # but their values are not used during inference/rollout.
    mock_lr_info = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'step_size': 10,
        'gamma': 0.5
    }
    mock_trainer_options = {
        'batch_size': 32, # Value doesn't matter here
        'only_where_water': False,
        'velocity_scaler': 1.0,
        'type_loss': 'RMSE'
    }
    
    # --- C. Calculate model dimensions (copied from adforce_main.py) ---
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

    # --- D. Instantiate the underlying model ---
    try:
        if model_type == "GNN":
            model_to_load = GNNModelAdforce(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                previous_t=p_t,
                num_output_features=num_output_features,
                num_static_features=NUM_STATIC_NODE_FEATURES,
                **model_params,
            )
        elif model_type == "MLP":
            model_to_load = PointwiseMLPModel(
                num_node_features=num_node_features,
                num_output_features=num_output_features,
                **model_params,
            )
        elif model_type == "MonolithicMLP":
            n_nodes_fixed = int(dataset.total_nodes)
            if n_nodes_fixed is None:
                raise ValueError("Could not determine n_nodes from dataset")
            print(f"Found fixed n_nodes from dataset: {n_nodes_fixed}")
            model_to_load = MonolithicMLPModel(
                n_nodes=n_nodes_fixed,
                num_node_features=num_node_features,
                num_output_features=num_output_features,
                **model_params,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        print(f"Instantiated {model_type} model structure.")
        
    except Exception as e:
        print(f"Error: Failed to instantiate model structure: {e}")
        print("Please check your 'model_params' in this script.")
        exit()

    # --- E. Load the LightningTrainer from checkpoint ---
    try:
        # Pass the instantiated model and mock configs
        lightning_model = LightningTrainer.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            # --- Pass missing __init__ args ---
            model=model_to_load,
            lr_info=mock_lr_info,
            trainer_options=mock_trainer_options
        )
        lightning_model.to(device)
        lightning_model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model checkpoint: {e}")
        print("This may be due to a mismatch between your model_params and the saved weights.")
        exit()


    # --- 5. LOAD STATIC DATA & CLIMITS ---
    x_coords, y_coords, dem = load_static_data(netcdf_file, dataset)
    climits = calculate_global_climits(dataset, dem)

    # --- 6. PERFORM ROLLOUT ---
    all_predictions = perform_rollout(lightning_model, dataset, device)
    
    if len(all_predictions) != total_frames:
        print(f"Error: Rollout returned {len(all_predictions)} frames, expected {total_frames}")
        exit()

    # --- 7. RENDER FRAMES ---
    print("Rendering predicted frames...")
    frame_files = []
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
            prediction_state=prediction_for_this_frame 
        )
    
    # --- 8. COMPILE & CLEANUP ---
    images = []
    if output_gif or output_video:
        for frame_file in tqdm(frame_files, desc="Reading frames into memory"):
            images.append(iio.imread(frame_file))

    if output_gif:
        compile_gif_from_frames(frame_dir, output_gif, anim_fps, images)

    if output_video:
        compile_video_from_frames(frame_dir, output_video, anim_fps, images)
    
    # Clean up temp frames
    #try:
    #    shutil.rmtree(frame_dir)
    #    print(f"Cleaned up temporary directory: {frame_dir}")
    #except Exception as e:
    #    print(f"Warning: Failed to clean up {frame_dir}. Error: {e}")

    print(f"\nPrediction animation complete.")
    if output_gif:
        print(f"GIF saved to {os.path.abspath(output_gif)}")
    if output_video:
        print(f"Video saved to {os.path.abspath(output_video)}")
