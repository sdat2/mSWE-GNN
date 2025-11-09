"""
Improved animation function for the AdforceLazyDataset.

This script creates a 6-panel animation of a full simulation event
by rendering each frame as a PNG. It then compiles these frames
into *both* a high-quality MP4 (for video) and a quantized GIF
(for markdown/previews).

---
V9 (Timestamp Titles):
- All user preferences from V8 are retained.
- The `plot_single_frame` function now reads the 'time'
  coordinate from the source NetCDF file for each frame.
- The `suptitle` is set to the actual datetime of the
  plotted data (e.g., "2005-08-28 12:00:00").
- A try/except block provides a fallback to the index
  number if the time lookup fails.
---
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


def get_frame_data(
    dataset: AdforceLazyDataset, idx: int, dem: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Retrieves and processes all 6 variables for a single animation frame.
    """
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

    # Forcing order from _get_forcing_slice is ["WX", "WY", "P"]
    wx_data = inputs_unscaled[:, 0].numpy()
    wy_data = inputs_unscaled[:, 1].numpy()
    p_data = inputs_unscaled[:, 2].numpy()

    # --- 2. Extract Outputs (WD, VX, VY) ---
    # `data.y_unscaled` holds the unscaled target state at t+1
    # Order from _get_target_slice is ["WD", "VX", "VY"]
    outputs = data.y_unscaled.cpu().numpy()
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
    """
    print(f"Calculating global color limits for {len(dataset)} frames...")
    
    plot_order = ["P", "WX", "WY", "SSH", "VX", "VY"]
    diverging_vars = ["WX", "WY", "SSH", "VX", "VY"]
    
    # Store the 2nd and 98th percentiles for each frame
    p2_vals = {key: [] for key in plot_order}
    p98_vals = {key: [] for key in plot_order}

    for idx in tqdm(range(len(dataset)), desc="Scanning data"):
        data_dict = get_frame_data(dataset, idx, dem)
        for key, data in data_dict.items():
            if data.size > 0:
                p2_vals[key].append(np.nanpercentile(data, 1))
                p98_vals[key].append(np.nanpercentile(data, 99))

    climits = {}
    for key in plot_order:
        if not p2_vals[key]: # Handle case of no valid data
             climits[key] = (0.0, 1.0)
             continue

        global_p2 = np.nanmin(p2_vals[key])
        global_p98 = np.nanmax(p98_vals[key])

        if key in diverging_vars:
            # Center on zero
            v_abs = np.nanmax([np.abs(global_p2), np.abs(global_p98)])
            if v_abs == 0: v_abs = 0.1 # Avoid vmin=vmax
            climits[key] = (-v_abs, v_abs)
        else:
            # Sequential (Pressure)
            if global_p2 == global_p98: global_p98 += 0.1
            climits[key] = (global_p2, global_p98)
    
    print("Global color limits calculated.")
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
):
    """
    Creates, plots, and saves a *single* frame from scratch.
    (Incorporates user's plotting preferences)
    """
    
    # 1. Get data for this specific frame
    data_dict = get_frame_data(dataset, idx, dem)

    # 2. Get timestamp for title
    try:
        nc_path, t_start = dataset.index_map[idx]
        # The plotted data is the target, which is at t_start + previous_t
        t_plot_idx = t_start + dataset.previous_t 
        with xr.open_dataset(nc_path, cache=True) as ds: # Use cache
            timestamp = ds.time[t_plot_idx].values
            # Format to 'YYYY-MM-DD HH:MM:SS'
            title = np.datetime_as_string(timestamp, unit='s').replace('T', ' ')[:-6] + ":00"
    except Exception as e:
        # Fallback title if time lookup fails
        if idx == 0: # Only warn once to avoid spam
            print(f"Warning: Could not read timestamp. Falling back to index. Error: {e}")
        title = f"Dataset Index: {idx} / {total_frames - 1}"

    # 3. Create a new figure and axes for this frame
    fig, axs = plt.subplots(
        2, 3, figsize=(6*1.2, 4*1.2), 
        sharex=True, sharey=True
    )

    # 4. Define titles, keys, and colormaps
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
    
    # 5. Plot all 6 subplots
    for i in range(2):
        for j in range(3):
            ax = axs[i, j]
            key = keys[i][j]
            data = data_dict[key]
            vmin, vmax = climits[key]
            
            # Plot data *directly* into the new axes
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

            # Add colorbar without label
            fig.colorbar(scat, ax=ax)
            
            ax.set_title(titles[i][j])
            ax.set_aspect("equal")
            
            if i == 1:
                ax.set_xlabel("Longitude [$^{\circ}$E]")
            if j == 0:
                ax.set_ylabel("Latitude [$^{\circ}$N]")
            
            # Fix axes limits to prevent plot jitter
            if x_coords.size > 0 and y_coords.size > 0:
                ax.set_xlim(np.nanmin(x_coords), np.nanmax(x_coords))
                ax.set_ylim(np.nanmin(y_coords), np.nanmax(y_coords))

    # 6. Add title and save
    fig.suptitle(title, y=0.92) # Use new timestamp title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    label_subplots(axs)
    fig.savefig(frame_path, dpi=150, bbox_inches='tight')
    
    # 7. --- VITAL --- Close the figure to free memory
    plt.close(fig)


def compile_gif_from_frames(
    frame_dir: str, 
    output_gif_path: str, 
    fps: int,
    images: List[np.ndarray] # Accepts pre-loaded images
):
    """
    Uses imageio.v3.imwrite to compile all PNGs into a single GIF.
    """
    if not images:
        print("No images found for GIF compilation.")
        return

    print(f"Compiling {len(images)} frames into {output_gif_path}...")
    
    # Pass the list of *images (arrays)* to imwrite
    iio.imwrite(output_gif_path, images, fps=fps, loop=0)
    
    print("GIF compilation complete.")


def compile_video_from_frames(
    frame_dir: str, 
    output_video_path: str, 
    fps: int,
    images: List[np.ndarray] # Accepts pre-loaded images
):
    """
    Uses imageio.v3.imwrite to compile all PNGs into a high-quality MP4.
    """
    if not images:
        print("No images found for Video compilation.")
        return

    print(f"Compiling {len(images)} frames into {output_video_path}...")
    
    # Use 'libx264' codec for high-quality MP4
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


def create_animation_from_frames(
    root_dir: str,
    nc_file_path: str,
    p_t: int,
    scaling_stats_path: str = None,
    output_gif_path: str = None, # Can be None
    output_video_path: str = None, # Can be None
    fps: int = 10,
):
    """
    Main function to run the full animation process.
    """
    plot_defaults()
    frame_dir = "./animation_frames_temp"

    # 1. Clean up and create temporary frame directory
    shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir, exist_ok=True)
    print(f"Temporary frame directory created at {frame_dir}")

    # 2. Initialize Dataset
    try:
        dataset = AdforceLazyDataset(
            root=root_dir,
            nc_files=[nc_file_path],
            previous_t=p_t,
            scaling_stats_path=scaling_stats_path,
        )
    except Exception as e:
        print(f"Failed to initialize AdforceLazyDataset: {e}")
        return

    if len(dataset) == 0:
        print("Dataset is empty. Check time steps and p_t.")
        return
    
    total_frames = len(dataset)
    print(f"Dataset loaded. Total samples: {total_frames}")

    # 3. Load static data (coords, DEM)
    x_coords, y_coords, dem = load_static_data(nc_file_path, dataset)

    # 4. Pre-calculate global color limits
    climits = calculate_global_climits(dataset, dem)

    # 5. Render all frames
    print("Rendering frames (recreating plot for each frame)...")
    frame_files = [] # Store file paths
    for idx in tqdm(range(total_frames), desc="Rendering frames"):
        frame_path = os.path.join(frame_dir, f"frame_{idx:05d}.png")
        frame_files.append(frame_path)
        
        # Call the stateless plotting function
        plot_single_frame(
            idx,
            total_frames,
            dataset,
            x_coords,
            y_coords,
            dem,
            climits,
            frame_path
        )
    
    # 6. Read images back into memory *once*
    images = []
    if output_gif_path or output_video_path:
        for frame_file in tqdm(frame_files, desc="Reading frames into memory"):
            images.append(iio.imread(frame_file))

    # 7. Compile GIF
    if output_gif_path:
        compile_gif_from_frames(frame_dir, output_gif_path, fps, images)

    # 8. Compile Video
    if output_video_path:
        compile_video_from_frames(frame_dir, output_video_path, fps, images)

    # 9. Clean up (Commented out per user request)
    #try:
    #    shutil.rmtree(frame_dir)
    #    print(f"Cleaned up temporary directory: {frame_dir}")
    #except Exception as e:
    #    print(f"Warning: Failed to clean up {frame_dir}. Error: {e}")


if __name__ == "__main__":
    # --- CONFIGURE YOUR PATHS HERE ---
    
    # This should be the directory containing your NetCDF file
    # and the 'scaling_stats.yaml' file.
    root_directory = "/Volumes/s/tcpips/swegnn_5sec/"
    
    # This is the single .nc file you want to animate
    netcdf_file = os.path.join(root_directory, "152_KATRINA_2005.nc")
    
    # This is the path to your scaling stats.
    # Set to None if you are not using scaling.
    scaling_stats_file = os.path.join(root_directory, "scaling_stats.yaml")
    
    # This must match the `previous_t` used to create the 'processed'
    # directory for the AdforceLazyDataset.
    previous_time_steps = 2
    
    # Define the output paths
    # Set to None to skip creation
    output_gif = "adforce_6panel_animation.gif"
    output_video = "adforce_6panel_animation.mp4"
    
    # Frames per second for the final animations
    anim_fps = 10
    # --- END CONFIGURATION ---

    if not os.path.exists(netcdf_file):
        print(f"Error: NetCDF file not found at {netcdf_file}")
    else:
        create_animation_from_frames(
            root_directory,
            netcdf_file,
            previous_time_steps,
            scaling_stats_path=scaling_stats_file,
            output_gif_path=output_gif,
            output_video_path=output_video,
            fps=anim_fps,
        )
        print(f"\nAnimation test complete.")
        if output_gif:
            print(f"GIF saved to {os.path.abspath(output_gif)}")
        if output_video:
            print(f"Video saved to {os.path.abspath(output_video)}")