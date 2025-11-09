"""
Improved animation function for the AdforceLazyDataset.

This script creates a 6-panel animation of a full simulation event,
displaying both the model inputs (forcing) and the corresponding
ground-truth outputs (state).

It uses the `AdforceLazyDataset` to access the data for each time step.

The 6 panels are:
[[Pressure (P),     Wind X (WX),  Wind Y (WY)],
 [Sea Surface (SSH), Velocity X (VX), Velocity Y (VY)]]

*** MODIFIED to use zero-centered diverging colormaps ('cmo.diff')
    for velocity, wind, and SSH. ***
"""

import os
import warnings
from typing import List, Tuple, Dict
import numpy as np
import xarray as xr
import torch
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.collections import PathCollection
from mswegnn.utils.adforce_dataset import AdforceLazyDataset
from sithom.plot import plot_defaults
from sithom.time import timeit

# --- NEW IMPORT ---
try:
    import cmocean
except ImportError:
    print(
        "Error: 'cmocean' library not found. Please install it:"
        "\n  pip install cmocean"
    )
    exit()

# Suppress Matplotlib warnings for dynamic color limits
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def load_static_data(
    nc_file_path: str, dataset: AdforceLazyDataset
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads all static (non-time-series) data required for plotting.

    This includes the node coordinates (x, y) from the original NetCDF file
    and the Digital Elevation Model (DEM) from the dataset's cached static
    data.

    Args:
        nc_file_path (str): Path to the NetCDF file to get coordinates from.
        dataset (AdforceLazyDataset): The initialized dataset object, which
            holds the cached static features (including DEM).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - x_coords: Node x-coordinates.
            - y_coords: Node y-coordinates.
            - dem: DEM (bathymetry) at each node.
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
        # [dem, slopex, slopey, area, node_type]
        dem = dataset.static_data["static_node_features"][:, 0].cpu().numpy()
    except Exception as e:
        print(f"Failed to get DEM from dataset.static_data: {e}")
        raise

    return x_coords, y_coords, dem


def setup_animation_plots(
    fig: plt.Figure, x_coords: np.ndarray, y_coords: np.ndarray
) -> Tuple[np.ndarray, List[PathCollection], List[plt.colorbar]]:
    """
    Creates the 2x3 subplot grid and initializes scatter plots.

    Args:
        fig (plt.Figure): The main Matplotlib figure object.
        x_coords (np.ndarray): Node x-coordinates.
        y_coords (np.ndarray): Node y-coordinates.

    Returns:
        Tuple[np.ndarray, List[PathCollection], List[plt.colorbar]]:
            - axs: A (2, 3) numpy array of Matplotlib Axes objects.
            - scats: A list of 6 PathCollection (scatter) objects.
            - cbs: A list of 6 Colorbar objects.
    """
    axs = fig.subplots(2, 3, sharex=True, sharey=True)
    scats = []
    cbs = []

    # Panel titles and colorbar labels
    # Row 0: Inputs (Forcing)
    # Row 1: Outputs (State)
    titles = [
        ["Pressure (P) [m]", "X-Wind (WX) [m/s]", "Y-Wind (WY) [m/s]"],
        ["Sea Surface (SSH) [m]", "X-Velocity (VX) [m/s]", "Y-Velocity (VY) [m/s]"],
    ]
    
    # --- MODIFIED CMAPS ---
    # Use 'cmo.thermal' for sequential (Pressure)
    # Use 'cmo.diff' for diverging (all others)
    cmaps = [
        [cmocean.cm.thermal, cmocean.cm.diff, cmocean.cm.diff],
        [cmocean.cm.diff, cmocean.cm.diff, cmocean.cm.diff],
    ]
    # --- END MODIFICATION ---

    for i in range(2):
        for j in range(3):
            ax = axs[i, j]
            scat = ax.scatter(
                x_coords, y_coords, c=[], cmap=cmaps[i][j], s=1, vmin=0, vmax=1
            )
            cb = fig.colorbar(scat, ax=ax, label=titles[i][j])
            
            ax.set_title(titles[i][j].split(" [")[0]) # Title without units
            ax.set_aspect("equal")
            if i == 1:
                ax.set_xlabel("X Coordinate")
            if j == 0:
                ax.set_ylabel("Y Coordinate")
                
            scats.append(scat)
            cbs.append(cb)

    return axs, scats, cbs


def get_frame_data(
    dataset: AdforceLazyDataset, idx: int, dem: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Retrieves and processes all 6 variables for a single animation frame.

    This function:
    1.  Calls `dataset.get(idx)` to get the PyG Data object.
    2.  Extracts the *scaled* input forcing from `data.x`.
    3.  Un-scales the forcing data using the dataset's cached stats.
    4.  Extracts the *unscaled* output state from `data.y_unscaled`.
    5.  Calculates Sea Surface Height (SSH) = Water Depth (WD) + DEM.

    Args:
        dataset (AdforceLazyDataset): The initialized dataset.
        idx (int): The frame index to retrieve.
        dem (np.ndarray): The DEM array (for SSH calculation).

    Returns:
        Dict[str, np.ndarray]: A dictionary mapping variable names
        ('P', 'WX', 'WY', 'SSH', 'VX', 'VY') to their data arrays.
    """
    data = dataset.get(idx)
    p_t = dataset.previous_t
    x_data = data.x.cpu()

    # --- 1. Extract and Un-scale Inputs (P, WX, WY) ---
    # The `data.x` tensor is structured as:
    # [static (5), forcing (3*p_t), state_y_t (3)]
    
    # We want the *last* time step of the input forcing
    forcing_start_idx = 5
    forcing_end_idx = 5 + (3 * p_t)
    last_forcing_step_scaled = x_data[:, forcing_end_idx - 3 : forcing_end_idx]

    if dataset.apply_scaling:
        if not hasattr(dataset, "x_dyn_mean_broadcast"):
            raise ValueError(
                "Dataset scaling is on but stats (e.g., 'x_dyn_mean_broadcast') "
                "were not found. Check scaling_stats.yaml path."
            )
        # Get the mean/std for a single forcing step (last 3 elements)
        mean = dataset.x_dyn_mean_broadcast[-3:].cpu()
        std = dataset.x_dyn_std_broadcast[-3:].cpu()
        
        # Unscale: (scaled * std) + mean
        inputs_unscaled = (last_forcing_step_scaled * std) + mean
    else:
        # Data was never scaled
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


@timeit
def create_animation(
    root_dir: str,
    nc_file_path: str,
    p_t: int,
    scaling_stats_path: str = None,
    output_gif_path: str = "adforce_6panel_animation.gif",
):
    """
    Runs the full 6-panel animation test and saves the result as a GIF.

    Args:
        root_dir (str): Root directory of the dataset (for 'processed' folder).
        nc_file_path (str): Path to the single NetCDF file to animate.
        p_t (int): Number of previous time steps (window size).
        scaling_stats_path (str, optional): Path to 'scaling_stats.yaml'.
            This is REQUIRED if the model was trained with scaling,
            to correctly un-scale the inputs for visualization.
        output_gif_path (str, optional): Path to save the output GIF.
    """
    plot_defaults()
    print("Starting 6-panel animation test...")

    # 1. Initialize Dataset
    try:
        dataset = AdforceLazyDataset(
            root=root_dir,
            nc_files=[nc_file_path],
            previous_t=p_t,
            scaling_stats_path=scaling_stats_path,  # Pass in the stats path
        )
    except Exception as e:
        print(f"Failed to initialize AdforceLazyDataset: {e}")
        if "Window mismatch" in str(e):
            print(
                f"Please delete the 'processed' directory inside '{root_dir}' and try again."
            )
        return

    if len(dataset) == 0:
        print("Dataset is empty. Check time steps and p_t.")
        return

    print(f"Dataset loaded. Total samples: {len(dataset)}")
    if not dataset.apply_scaling:
        print("WARNING: Running without scaling stats. Input plots will be unscaled.")

    # 2. Load static data (coords, DEM)
    x_coords, y_coords, dem = load_static_data(nc_file_path, dataset)

    # 3. Set up Matplotlib Animation
    fig, axs = plt.subplots(
        2, 3, figsize=(18, 10), sharex=True, sharey=True
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

    axs_flat, scats, cbs = setup_animation_plots(fig, x_coords, y_coords)
    
    # Define the order of data to match the plot layout
    plot_order = ["P", "WX", "WY", "SSH", "VX", "VY"]
    
    # --- NEW: Define which variables are diverging ---
    diverging_vars = ["WX", "WY", "SSH", "VX", "VY"]

    def init():
        """Initializes the animation."""
        for scat in scats:
            scat.set_array(np.array([]))
        return scats

    # --- MODIFIED UPDATE FUNCTION ---
    def update(frame_index):
        """Updates one frame of the animation."""
        try:
            # Get all 6 data arrays
            data_dict = get_frame_data(dataset, frame_index, dem)
            
            # Update each of the 6 plots
            for i, key in enumerate(plot_order):
                data = data_dict[key]
                scat = scats[i]
                
                # Set the data
                scat.set_array(data)
                
                # Dynamically update color limits
                with warnings.catch_warnings():
                    # Suppress warnings if data is all NaN
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    
                    if key in diverging_vars:
                        # --- New Logic: Center on zero ---
                        # Use 98th percentile of absolute value for robust range
                        v_abs = np.nanpercentile(np.abs(data), 98)
                        vmin = -v_abs
                        vmax = v_abs
                    else:
                        # --- Old Logic: (for Pressure) ---
                        vmin = np.nanpercentile(data, 2)
                        vmax = np.nanpercentile(data, 98)
                
                # Handle edge case where vmin == vmax (e.g., all zeros)
                if vmin == vmax:
                    # Add a small epsilon to prevent error
                    vmin -= 0.1
                    vmax += 0.1
                
                scat.set_clim(vmin=vmin, vmax=vmax)

            fig.suptitle(f"Dataset Index: {frame_index} / {len(dataset) - 1}")
            
            # Print progress to console
            if frame_index % 20 == 0:
                print(f"Processing frame {frame_index}...")

            return scats

        except Exception as e:
            print(f"Error updating frame {frame_index}: {e}")
            return scats
    # --- END MODIFIED UPDATE FUNCTION ---

    # 4. Create and Save Animation
    interval_ms = 100  # 100ms per frame = 10 FPS
    fps = 1000 / interval_ms
    print(f"Creating animation ({len(dataset)} frames, {fps} FPS)...")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(dataset),
        init_func=init,
        blit=True,
        interval=interval_ms,
        repeat=False,
    )

    print(f"Saving animation to {output_gif_path} (this may take a while)...")
    try:
        # Use the 'pillow' writer to create the GIF
        ani.save(output_gif_path, writer="pillow", fps=fps)
        print(f"Successfully saved animation to {output_gif_path}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
        print("Please ensure 'Pillow' is installed (`pip install Pillow`)")
    finally:
        # Close the plot to free memory
        plt.close(fig)


if __name__ == "__main__":
    # --- CONFIGURE YOUR PATHS HERE ---
    # python -m mswegnn.utils.adforce_animate

    
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
    output_gif = "adforce_6panel_animation.gif"
    # --- END CONFIGURATION ---

    if not os.path.exists(netcdf_file):
        print(f"Error: NetCDF file not found at {netcdf_file}")
        print("Please update the 'root_directory' and 'netcdf_file' variables.")
    elif scaling_stats_file and not os.path.exists(scaling_stats_file):
        print(f"Warning: Scaling stats file not found at {scaling_stats_file}")
        print("Inputs will be plotted as-is (which may be scaled).")
        create_animation(
            root_directory,
            netcdf_file,
            previous_time_steps,
            scaling_stats_path=None,
            output_gif_path=output_gif,
        )
    else:
        create_animation(
            root_directory,
            netcdf_file,
            previous_time_steps,
            scaling_stats_path=scaling_stats_file,
            output_gif_path=output_gif,
        )

    print(f"Animation test complete. GIF saved to {os.path.abspath(output_gif)}")