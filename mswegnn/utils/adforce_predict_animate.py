"""
Prediction animation function for the AdforceLazyDataset.

This script loads a trained mSWE-GNN model and runs an
autoregressive rollout (prediction) for a full simulation event.

This version supports two rollout modes via ROLLOUT_HORIZON:
1. (N > 0): "Fixed Horizon" mode. Each frame 'k' in the animation
   shows the result of an N-step prediction that *started* at
   frame 'k - N + 1'. This is computationally intensive as it
   re-runs the rollout for every frame.
2. (N = -1): "Full Rollout" mode. Runs a single, free-running
   simulation from t=0. Each frame 'k' shows the result of a
   'k'-step-long prediction.
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

# --- IMPORTS ---
import lightning as L
from mswegnn.training.adforce_train import LightningTrainer
from mswegnn.models.adforce_models import (
    GNNModelAdforce,
    PointwiseMLPModel,
    MonolithicMLPModel,
)
from mswegnn.utils.adforce_dataset import AdforceLazyDataset
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
    device: torch.device,
    rollout_horizon: int = 1,
) -> List[np.ndarray]:
    """
    Performs an autoregressive rollout based on the specified horizon.

    --- UPDATED ---
    This function assumes the model predicts the *SCALED NEXT STATE*.
    It now correctly prepends the initial t=0 ground truth state to
    the list, so the returned list is [y_0_GT, y_1_pred, y_2_pred, ...].
    This ensures the animation starts at the correct initial condition.

    Args:
        model (L.LightningModule): The trained Lightning model (on device).
        dataset (AdforceLazyDataset): The dataset for a single simulation.
        device (torch.device): The torch device (e.g., 'cuda' or 'cpu').
        rollout_horizon (int, optional): The rollout strategy.

    Returns:
        List[np.ndarray]: A list of unscaled predicted states
                          [y_0_GT, y_1_pred, y_2_pred, ...].
    """
    model.eval()  # Set model to evaluation mode

    # --- Get all necessary scaling stats from the dataset ---
    if not dataset.apply_scaling:
        raise ValueError("This script assumes scaling is applied to the dataset.")
        
    y_mean = dataset.y_mean.to(device)
    y_std = dataset.y_std.to(device)
    # y_delta_mean and y_delta_std are no longer needed

    predictions_list = []

    # --- 1. Get the *initial state* from frame 0 ---
    current_batch = dataset.get(0).to(device)
    current_y_t_scaled = current_batch.x[:, -3:].clone()  # State y(t=0)
    current_y_t_raw = (current_y_t_scaled * y_std) + y_mean
    
    # --- MODIFICATION: Add initial state as Frame 0 ---
    predictions_list.append(current_y_t_raw.cpu().numpy())

    # --- BRANCH 1: FULL, FREE-RUNNING ROLLOUT ---
    if rollout_horizon == -1:
        print("Starting full, free-running rollout (predicting next state)...")
        
        # --- MODIFICATION: Loop for N-1 predictions (since we have t=0) ---
        # dataset.get(idx) gets forcing for t=idx, to predict t=idx+1
        for idx in tqdm(range(len(dataset) - 1), desc="Full Rollout"):
            # 1. Get the *ground truth batch* for this step's *forcing*
            gt_batch = dataset.get(idx).to(device) # Gets forcing_idx

            # 2. Create the *prediction input*
            pred_input_batch = gt_batch.clone()

            # 3. ...but replace the state with our *predicted* state
            # On idx=0, this uses current_y_t_scaled (y_0_GT)
            pred_input_batch.x[:, -3:] = current_y_t_scaled

            # 4. Run the model to predict the *scaled next state*
            pred_scaled_y_tplus1 = model.model(pred_input_batch) # Predicts y_1_pred

            # 5. Un-scale the predicted state
            pred_raw_y_tplus1 = (pred_scaled_y_tplus1 * y_std) + y_mean

            # 6. Store the *unscaled predicted state*
            predictions_list.append(pred_raw_y_tplus1.cpu().numpy()) # Appends y_1_pred

            # 7. Prepare for the *next* loop iteration (efficient update)
            current_y_t_scaled = pred_scaled_y_tplus1 # Becomes y_1_pred for next loop

    # --- BRANCH 2: FIXED-HORIZON ROLLOUT ---
    else:
        print(f"Starting {rollout_horizon}-step fixed-horizon rollout (predicting next state)...")

        # Loop for each frame we want to generate (frame 1 to N-1)
        # We already have frame 0
        for idx in tqdm(range(1, len(dataset)), desc="Fixed-Horizon Rollout"):
            
            # 1. Determine the start of the GT state
            start_idx = max(0, idx - rollout_horizon + 1)
            # 2. Determine how many prediction steps to take
            steps_to_take = idx - start_idx
            
            # 3. Get the GT state at the start
            gt_batch_start = dataset.get(start_idx).to(device)
            current_y_t_scaled = gt_batch_start.x[:, -3:].clone() # State y(t_start)

            # 4. Run the inner mini-rollout loop
            for k in range(steps_to_take):
                # We need forcing from t_start, t_start+1, ..., idx-1
                forcing_batch_idx = start_idx + k
                if forcing_batch_idx >= len(dataset):
                    break
                
                gt_forcing_batch = dataset.get(forcing_batch_idx).to(device)
                pred_input_batch = gt_forcing_batch.clone()
                pred_input_batch.x[:, -3:] = current_y_t_scaled

                pred_scaled_y_tplus1 = model.model(pred_input_batch)
                current_y_t_scaled = pred_scaled_y_tplus1 # Update for next step
            
            # 5. After the loop, current_y_t_scaled is the final prediction
            final_pred_raw = (current_y_t_scaled * y_std) + y_mean
            predictions_list.append(final_pred_raw.cpu().numpy())

    print("Rollout complete.")
    return predictions_list


def get_frame_data(
    dataset: AdforceLazyDataset, 
    idx: int, 
    dem: np.ndarray,
    prediction_state: np.ndarray  # <-- This is the UNCALED state y(t+1)
) -> Dict[str, np.ndarray]:
    """
    Retrieves and processes all 6 variables for a single animation frame.
    Uses ground-truth inputs (P, WX, WY) but predicted outputs.
    """
    data = dataset.get(idx)
    p_t = dataset.previous_t
    x_data = data.x.cpu()

    # --- 1. Extract and Un-scale Inputs (P, WX, WY) ---
    forcing_start_idx = 5
    forcing_end_idx = 5 + (3 * p_t)
    # Get the *last* forcing step available in this batch
    # (which corresponds to the state we are plotting)
    last_forcing_step_scaled = x_data[:, forcing_end_idx - 3 : forcing_end_idx] 

    if dataset.apply_scaling:
        # We need the mean/std for a *single* step, not broadcasted
        mean = dataset.x_dyn_mean_broadcast.cpu()[:3] 
        std = dataset.x_dyn_std_broadcast.cpu()[:3]
        inputs_unscaled = (last_forcing_step_scaled * std) + mean
    else:
        inputs_unscaled = last_forcing_step_scaled
        
    # Forcing order from _get_forcing_slice is ["WX", "WY", "P"]
    wx_data = inputs_unscaled[:, 0].numpy()
    wy_data = inputs_unscaled[:, 1].numpy()
    p_data = inputs_unscaled[:, 2].numpy()

    # --- 2. Extract Outputs (WD, VX, VY) from the prediction ---
    outputs = prediction_state # Use the passed-in unscaled state
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
    Calculates global vmin/vmax by iterating through the *GROUND TRUTH* dataset.
    """
    print(f"Calculating global color limits from GROUND TRUTH data ({len(dataset)} frames)...")
    
    plot_order = ["P", "WX", "WY", "SSH", "VX", "VY"]
    diverging_vars = ["WX", "WY", "SSH", "VX", "VY"]
    
    p2_vals = {key: [] for key in plot_order}
    p98_vals = {key: [] for key in plot_order}

    for idx in tqdm(range(len(dataset)), desc="Scanning data"):
        data_gt = dataset.get(idx)
        
        # data.y_unscaled is y_tplus1_raw (the full state)
        outputs_gt = data_gt.y_unscaled.cpu().numpy()
        
        wd_gt = outputs_gt[:, 0]
        vx_gt = outputs_gt[:, 1]
        vy_gt = outputs_gt[:, 2]
        ssh_gt = wd_gt + dem
        
        # Get input data
        p_t = dataset.previous_t
        x_data = data_gt.x.cpu()
        forcing_start_idx = 5
        forcing_end_idx = 5 + (3 * p_t)
        last_forcing_step_scaled = x_data[:, forcing_end_idx - 3 : forcing_end_idx] # Fixed index
        
        if dataset.apply_scaling:
            mean = dataset.x_dyn_mean_broadcast.cpu()[:3]
            std = dataset.x_dyn_std_broadcast.cpu()[:3]
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
    prediction_state: np.ndarray
):
    """
    Creates, plots, and saves a *single* frame from scratch.
    
    --- UPDATED ---
    This function now correctly fetches the timestamp for the
    corresponding input state, fixing the off-by-one error.
    """
    
    data_dict = get_frame_data(dataset, idx, dem, prediction_state) 

    try:
        nc_path, t_start = dataset.index_map[idx]
        
        # --- MODIFICATION: Get timestamp for the *input state* ---
        # t_start is the *start* of the input window.
        # The *state* we are plotting corresponds to the *end* of that window,
        # which is the state at (t_start + previous_t - 1).
        t_plot_idx = t_start + dataset.previous_t - 1
        # For idx=0: t_start=0, p_t=1 -> t_plot_idx=0. Correct.
        # For idx=1: t_start=1, p_t=1 -> t_plot_idx=1. Correct.
        # --- END MODIFICATION ---

        with xr.open_dataset(nc_path, cache=True) as ds:
            timestamp = ds.time[t_plot_idx].values
            # More robust timestamp formatting
            title_str = np.datetime_as_string(timestamp, unit='s').replace('T', ' ')
            if '.' in title_str:
                 title_str = title_str.split('.')[0] # Remove fractional seconds
            title = title_str
            
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
    # python -m mswegnn.utils.adforce_predict_animate
    # --- 1. CONFIGURE YOUR PATHS HERE ---
    
    # Path to your *saved model checkpoint*
    # checkpoint_path = "/Volumes/s/tcpips/mSWE-GNN/checkpoints/GNN-best-epoch=37-val_loss=0.5033.ckpt"
    checkpoint_path = "/home/users/sithom/mSWE-GNN/checkpoints/GNN-best-epoch=08-val_loss=0.1434.ckpt"
     
    # This should be the directory containing your NetCDF file
    # root_directory = "/Volumes/s/tcpips/swegnn_5sec/"
    root_directory = "/home/users/sithom/swegnn_5sec/"
    
    # This is the single .nc file you want to animate (e.g., Katrina)
    netcdf_file = os.path.join(root_directory, "152_KATRINA_2005.nc")
    
    # This must be the *training* stats file your model was trained with
    # scaling_stats_file = "/Volumes/s/tcpips/mSWE-GNN/data_processed/train/scaling_stats.yaml"
    scaling_stats_file = "/home/users/sithom/mSWE-GNN/data_processed/train/scaling_stats.yaml"
    
    # --- 2. CONFIGURE ROLLOUT ---
    
    # This must match the `previous_t` used to train your model
    # (Based on your logs, this should be 1)
    previous_time_steps = 1
    
    # --- THIS IS THE NEW PARAMETER ---
    # What kind of rollout to perform?
    # -1 = "Full Rollout": Free-running simulation from t=0.
    #  1 = "Fixed 1-step": Every frame is a 1-step-ahead prediction.
    #  3 = "Fixed 3-step": Every frame (after first 2) is a 3-step-ahead prediction.
    ROLLOUT_HORIZON = -1 
    
    # --- 3. CONFIGURE OUTPUTS ---
    rollout_type_str = "full" if ROLLOUT_HORIZON == -1 else f"{ROLLOUT_HORIZON}step"
    output_gif = f"adforce_6panel_PREDICTION_{rollout_type_str}.gif"
    output_video = f"adforce_6panel_PREDICTION_{rollout_type_str}.mp4"
    anim_fps = 10
    
    # --- END CONFIGURATION ---


    # --- 4. SETUP DEVICE ---
    plot_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        exit()
    if not os.path.exists(netcdf_file):
        print(f"Error: NetCDF file not found at {netcdf_file}")
        exit()
    if not os.path.exists(scaling_stats_file):
        print(f"Error: Scaling stats file not found at {scaling_stats_file}")
        exit()

    # --- 5. INITIALIZE DATASET ---
    print(f"Initializing dataset for {netcdf_file}...")
    frame_dir = f"./animation_frames_temp_PREDICT_{rollout_type_str}"
    shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir, exist_ok=True)

    try:
        # predict_root = "/Volumes/s/tcpips/mSWE-GNN/data_processed/predict_katrina"
        predict_root = "/home/users/sithom/mSWE-GNN/data_processed/predict_katrina"
        
        dataset = AdforceLazyDataset(
            root=predict_root, 
            nc_files=[netcdf_file],
            previous_t=previous_time_steps, # Should be 1
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

    # --- 6. CONFIGURE AND LOAD MODEL ---
    print(f"Loading model from {checkpoint_path}...")

    # --- A. Model parameters from your (working) log ---
    model_type = "GNN"
    model_params = {
        'model_type': 'GNN',
        'type_gnn': 'SWEGNN',
        'hid_features': 64,
        'mlp_layers': 2,
        'K': 3,
        'normalize': True,
        'gnn_activation': "tanh",
        'edge_mlp': True,
        'with_gradient': True,
    }

    # --- B. Mock configs from your (working) log ---
    mock_lr_info = {
        'learning_rate': 1e-3, 'weight_decay': 1e-4,
        'step_size': 20, 'gamma': 0.5
    }
    mock_trainer_options = {
        'batch_size': 4, 'only_where_water': True,
        'velocity_scaler': 5.0, 'type_loss': 'RMSE'
    }
    
    # --- C. Calculate model dimensions ---
    p_t = previous_time_steps # Should be 1
    NUM_STATIC_NODE_FEATURES = 5
    NUM_DYNAMIC_NODE_FEATURES = 3
    NUM_CURRENT_STATE_FEATURES = 3
    NUM_STATIC_EDGE_FEATURES = 2
    NUM_OUTPUT_FEATURES = 3
    
    # This should be (5 + (3*1) + 3) = 11
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
        else:
             raise ValueError(f"Model type {model_type} not configured in this script")
            
        print(f"Instantiated {model_type} model structure.")
        
    except Exception as e:
        print(f"Error: Failed to instantiate model structure: {e}")
        exit()

    # --- E. Load the LightningTrainer from checkpoint ---
    try:
        lightning_model = LightningTrainer.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            model=model_to_load,
            lr_info=mock_lr_info,
            trainer_options=mock_trainer_options
        )
        lightning_model.to(device)
        lightning_model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model checkpoint: {e}")
        exit()


    # --- 7. LOAD STATIC DATA & CLIMITS ---
    x_coords, y_coords, dem = load_static_data(netcdf_file, dataset)
    climits = calculate_global_climits(dataset, dem)

    # --- 8. PERFORM ROLLOUT ---
    all_predictions = perform_rollout(
        lightning_model, 
        dataset, 
        device,
        rollout_horizon=ROLLOUT_HORIZON
    )
    
    # --- MODIFICATION: Check total_frames vs all_predictions length ---
    # all_predictions now has length total_frames (y_0 to y_{N-1})
    # total_frames has length N (from len(dataset))
    # This check is now correct.
    if len(all_predictions) != total_frames:
        print(f"Error: Rollout returned {len(all_predictions)} frames, expected {total_frames}")
        exit()

    # --- 9. RENDER FRAMES ---
    print("Rendering predicted frames...")
    frame_files = []
    # This loop is now correct: idx=0 will get all_predictions[0] (y_0_GT)
    # and plot_single_frame(idx=0) will get timestamp for t=0.
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
    
    # --- 10. COMPILE & CLEANUP ---
    images = []
    if output_gif or output_video:
        for frame_file in tqdm(frame_files, desc="Reading frames into memory"):
            images.append(iio.imread(frame_file))

    if output_gif:
        compile_gif_from_frames(frame_dir, output_gif, anim_fps, images)

    if output_video:
        compile_video_from_frames(frame_dir, output_video, anim_fps, images)

    print(f"\nPrediction animation complete.")
    if output_gif:
        print(f"GIF saved to {os.path.abspath(output_gif)}")
    if output_video:
        print(f"Video saved to {os.path.abspath(output_video)}")