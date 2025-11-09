"""
Improved animation function for the AdforceLazyDataset.

This script creates a 6-panel animation of a full simulation event
by rendering each frame as a PNG and then compiling them into a GIF
using `imageio`.

---
V5 (Final Fix):
- Corrected the `compile_gif_from_frames` function.
- The `imageio.v3.imwrite` function (for GIFs) expects a
  *list of image arrays (numpy)*, not a *list of filenames*.
- This version now correctly reads all the saved PNGs back
  into a list (`images = [iio.imread(f) for f in ...]`)
  and passes that list of arrays to `iio.imwrite`.
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
from sithom.plot import plot_defaults

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

    This function un-scales inputs and calculates SSH from outputs.
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

    This is a slow but necessary step to ensure fixed color scales.
    It uses robust percentiles (2nd, 98th) to handle outliers.
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
                p2_vals[key].append(np.nanpercentile(data, 2))
                p98_vals[key].append(np.nanpercentile(data, 98))

    climits = {}
    for key in plot_order:
        if not p2_vals[key]: # Handle case of no valid data
             climits[key] = (0.0, 1.0)
             continue

        global_p2 = np.nanmin(p2_vals[key])
        global_p98 = np.nanmax(p98_vals[key])

        if key in diverging_vars:
            # Center on zero using the largest absolute percentile
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

    This function is called in a loop and is designed to be stateless.
    It creates a new figure and closes it to prevent memory leaks.
    """
    
    # 1. Get data for this specific frame
    data_dict = get_frame_data(dataset, idx, dem)

    # 2. Create a new figure and axes for this frame
    fig, axs = plt.subplots(
        2, 3, figsize=(9, 5), 
        sharex=True, sharey=True
    )

    titles = [
        ["Pressure (P) [m]", "X-Wind (WX) [m s$^{-1}$]", "Y-Wind (WY) [m s$^{-1}$]"],
        ["Storm Surge Height (SSH) [m]", "X-Velocity (VX) [m s$^{-1}$]", "Y-Velocity (VY) [m s$^{-1}$]"],
    ]
    keys = [
        ["P", "WX", "WY"],
        ["SSH", "VX", "VY"]
    ]
    cmaps = [
        [cmocean.cm.thermal, cmocean.cm.balance, cmocean.cm.balance],
        [cmocean.cm.balance, cmocean.cm.balance, cmocean.cm.balance],
    ]
    
    # 3. Plot all 6 subplots
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
                s=1,
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

    # 4. Add title and save
    fig.suptitle(f"Dataset Index: {idx} / {total_frames - 1}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(frame_path, dpi=150, bbox_inches='tight')
    
    # 5. --- VITAL --- Close the figure to free memory
    plt.close(fig)


# --- *** CORRECTED FUNCTION *** ---
def compile_gif_from_frames(
    frame_dir: str, 
    output_gif_path: str, 
    fps: int
):
    """
    Uses imageio.v3.imwrite to compile all PNGs into a single GIF.
    """
    # Find all frame files and sort them
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    if not frame_files:
        print(f"Error: No frames found in {frame_dir}")
        return

    print(f"Compiling {len(frame_files)} frames into {output_gif_path}...")
    
    # --- *** FIX *** ---
    # We must read the images into a list of numpy arrays first.
    # `iio.imwrite` (with the pillow backend) expects image data,
    # not filenames.
    images = []
    for frame_file in tqdm(frame_files, desc="Reading frames for GIF"):
        images.append(iio.imread(frame_file))

    # Now, pass the list of *images (arrays)* to imwrite
    iio.imwrite(output_gif_path, images, fps=fps, loop=0)
    # --- *** END FIX *** ---
    
    print("GIF compilation complete.")


def create_animation_from_frames(
    root_dir: str,
    nc_file_path: str,
    p_t: int,
    scaling_stats_path: str = None,
    output_gif_path: str = "adforce_6panel_animation.gif",
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
    
    for idx in tqdm(range(total_frames), desc="Rendering frames"):
        frame_path = os.path.join(frame_dir, f"frame_{idx:05d}.png")
        
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
    
    # 6. Compile GIF
    compile_gif_from_frames(frame_dir, output_gif_path, fps)

    # 7. Clean up
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
    
    # Define the output GIF path
    output_gif = "adforce_6panel_animation_robust.gif"
    
    # Frames per second for the final GIF
    gif_fps = 10
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
            fps=gif_fps,
        )
        print(f"\nAnimation test complete. GIF saved to {os.path.abspath(output_gif)}")