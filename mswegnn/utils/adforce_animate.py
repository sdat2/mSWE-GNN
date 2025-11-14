"""
Refactored animation function for mSWE-GNN.

This script creates a 6-panel animation of a simulation event
by rendering each frame as a PNG. It then compiles these frames
into *both* a high-quality MP4 and a GIF.

This version is config-driven. It reads a config.yaml to
determine how to load or derive variables (e.g., SSH).

Example usage:
python mswegnn/utils/adforce_animate.py \
    -c conf/config.yaml \
    -f ../SurgeNetTestPH/new_orleans_2100_3.nc \
    -v P WX WY SSH VX VY \
    -u "m" "m s$^{-1}$" "m s$^{-1}$" "m" "m s$^{-1}$" "m s$^{-1}$" \
    --diverging-vars WX WY VX VY SSH \
    --cmap-seq cmo.thermal \
    --cmap-div cmo.balance \
    -o test_ani.mp4 \
    --gif

python mswegnn/utils/adforce_animate.py \
    -c conf/config.yaml \
    -f ../swegnn_5sec/152_KATRINA_2005.nc \
    -v P WX WY SSH VX VY \
    -u "m" "m s$^{-1}$" "m s$^{-1}$" "m" "m s$^{-1}$" "m s$^{-1}$" \
    --diverging-vars WX WY VX VY SSH \
    --cmap-seq cmo.thermal \
    --cmap-div cmo.balance \
    -o train_ani.mp4 \ 
    --gif

"""
import os
import shutil
import warnings
import argparse
import doctest
from typing import List, Tuple, Dict
import numpy as np
import xarray as xr
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio.v3 as iio
from omegaconf import DictConfig, OmegaConf

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


# --- NEW, CONFIG-DRIVEN HELPER FUNCTIONS ---

def _perform_op(op: str, args: List[torch.Tensor]) -> torch.Tensor:
    """
    Performs an operation (add, subtract, magnitude) on a list of tensors,
    handling broadcasting (e.g., [time, nodes] + [nodes]).

    Doctest:
    >>> t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) # [time, nodes]
    >>> t2 = torch.tensor([10.0, 20.0])              # [nodes]
    >>> _perform_op("add", [t1, t2])
    tensor([[11., 22.],
            [13., 24.]])
    >>> _perform_op("magnitude", [torch.tensor([3., 12.]), torch.tensor([4., 5.])])
    tensor([ 5., 13.])
    """
    if op == "add":
        result = args[0] + args[1]
    elif op == "subtract":
        result = args[0] - args[1]
    elif op == "magnitude":
        result = torch.sqrt(args[0]**2 + args[1]**2)
    else:
        raise NotImplementedError(f"Operation '{op}' not implemented for animation.")
    
    return torch.nan_to_num(result, nan=0.0)


def get_animation_data(
    ds: xr.Dataset, 
    features_cfg: DictConfig, 
    var_name: str,
    loaded_cache: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Recursively loads or derives a variable for animation from a config.
    """
    if var_name in loaded_cache:
        return loaded_cache[var_name]

    if var_name in features_cfg.static:
        data = torch.tensor(ds[var_name].values, dtype=torch.float)
    elif var_name in features_cfg.state or \
         var_name in features_cfg.forcing or \
         var_name in features_cfg.targets:
        data = torch.tensor(ds[var_name].values, dtype=torch.float)
    elif hasattr(features_cfg, 'derived_state'):
        for derived_spec in features_cfg.derived_state:
            if derived_spec["name"] == var_name:
                arg_data = [
                    get_animation_data(ds, features_cfg, arg, loaded_cache)
                    for arg in derived_spec["args"]
                ]
                data = _perform_op(derived_spec["op"], arg_data)
                loaded_cache[var_name] = data
                return torch.nan_to_num(data, nan=0.0)
        else:
            raise ValueError(f"Var '{var_name}' not found in file or derived config.")
    else:
        raise ValueError(f"Var '{var_name}' not in config and no derived_state found.")

    data = torch.nan_to_num(data, nan=0.0)
    loaded_cache[var_name] = data
    return data


# --- ORIGINAL PLOTTING FUNCTIONS (Modified to be config-driven) ---

def calculate_global_climits(
    data_dict: Dict[str, torch.Tensor],
    diverging_vars: List[str]
) -> Dict[str, Tuple[float, float]]:
    """
    Calculates global vmin/vmax by operating on the pre-loaded
    data dictionary. Keeps original diverging/sequential logic.
    """
    print("Calculating global color limits...")

    climits = {}
    for key, data_tensor in data_dict.items():
        data = data_tensor.numpy()
        
        if data.size == 0:
            climits[key] = (0.0, 1.0)
            continue

        p2 = np.nanpercentile(data, 1)
        p98 = np.nanpercentile(data, 99)

        if key in diverging_vars:
            # Center on zero (original logic)
            v_abs = np.nanmax([np.abs(p2), np.abs(p98)])
            if v_abs == 0:
                v_abs = 0.1
            climits[key] = (-v_abs, v_abs)
        else:
            # Sequential (original logic)
            if p2 == p98:
                p98 += 0.1
            climits[key] = (p2, p98)

    print("Global color limits calculated.")
    for key, (vmin, vmax) in climits.items():
        print(f"  {key}: ({vmin:.2f}, {vmax:.2f})")

    return climits


def plot_single_frame(
    t_idx: int,
    all_timestamps: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    data_for_frame: Dict[str, np.ndarray],
    climits: Dict[str, Tuple[float, float]],
    colormaps: Dict[str, plt.cm.ScalarMappable],
    titles: Dict[str, str],
    frame_path: str,
):
    """
    Creates, plots, and saves a *single* frame from scratch.
    (This is your original plotting function, with a new signature.
     It no longer loads data, it just plots.)
    """

    # 1. Get timestamp for title (original logic)
    try:
        timestamp = all_timestamps[t_idx]
        title = (
            np.datetime_as_string(timestamp, unit="s").replace("T", " ")[:-6]
            + ":00"
        )
    except Exception as e:
        title = f"Frame Index: {t_idx}"

    # 2. Create a new figure and axes for this frame (original logic)
    fig, axs = plt.subplots(2, 3, figsize=(6 * 1.2, 4 * 1.2), sharex=True, sharey=True)

    # 3. Define keys
    keys = [["P", "WX", "WY"], ["SSH", "VX", "VY"]]

    # 5. Plot all 6 subplots (original logic)
    for i in range(2):
        for j in range(3):
            ax = axs[i, j]
            key = keys[i][j]
            data = data_for_frame[key]
            vmin, vmax = climits[key]
            cmap_to_use = colormaps[key]
            title_str = titles[key]

            # Plot data *directly* (original logic)
            scat = ax.scatter(
                x_coords,
                y_coords,
                c=data,
                cmap=cmap_to_use,
                s=0.2,
                marker=".",
                vmin=vmin,
                vmax=vmax,
            )

            fig.colorbar(scat, ax=ax)
            ax.set_title(title_str)
            ax.set_aspect("equal")

            if i == 1:
                ax.set_xlabel("Longitude [$^{\circ}$E]")
            if j == 0:
                ax.set_ylabel("Latitude [$^{\circ}$N]")

            if x_coords.size > 0 and y_coords.size > 0:
                ax.set_xlim(np.nanmin(x_coords), np.nanmax(x_coords))
                ax.set_ylim(np.nanmin(y_coords), np.nanmax(y_coords))

    # 6. Add title and save (original logic)
    fig.suptitle(title, y=0.92)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    label_subplots(axs)
    fig.savefig(frame_path, dpi=150, bbox_inches="tight")

    # 7. --- VITAL --- Close the figure to free memory (original logic)
    plt.close(fig)


def compile_gif_from_frames(
    output_gif_path: str,
    fps: int,
    images: List[np.ndarray],  # Accepts pre-loaded images
):
    """
    Uses imageio.v3.imwrite to compile all PNGs into a single GIF.
    (Original function)
    """
    if not images:
        print("No images found for GIF compilation.")
        return
    print(f"Compiling {len(images)} frames into {output_gif_path}...")
    iio.imwrite(output_gif_path, images, fps=fps, loop=0)
    print("GIF compilation complete.")


def compile_video_from_frames(
    output_video_path: str,
    fps: int,
    images: List[np.ndarray],  # Accepts pre-loaded images
):
    """
    Uses imageio.v3.imwrite to compile all PNGs into a high-quality MP4.
    (Original function)
    """
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
        print(f"\n--- ERROR ---: Video compilation failed: {e}")
        print("  pip install imageio[ffmpeg]")


def create_animation_from_frames(
    config_path: str,
    nc_file_path: str,
    var_names: List[str],
    var_units: List[str],
    diverging_vars: List[str],
    cmap_seq_name: str,
    cmap_div_name: str,
    output_video_path: str,
    output_gif_path: str, # Can be None
    fps: int,
):
    """
    Main function to run the full, config-driven animation process.
    """
    plot_defaults()
    frame_dir = "./animation_frames_temp"

    shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir, exist_ok=True)

    # 1. Load Config and Dataset
    print(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    features_cfg = cfg.features
    
    print(f"Loading data from: {nc_file_path}")
    ds = xr.open_dataset(nc_file_path)

    # 2. Load coordinates and timestamps
    x_coords = ds["x"].values
    y_coords = ds["y"].values
    # The 'time' coord is needed for the titles
    all_timestamps = ds.time.values
    num_timesteps = ds.sizes["time"]
    print(f"Data loaded. Found {num_timesteps} timesteps.")

    # 3. Load all data *once* using config-driven helpers
    data_cache = {}
    data_to_plot_torch = {}
    print(f"Loading variables: {var_names}")
    for var in var_names:
        data_to_plot_torch[var] = get_animation_data(
            ds, features_cfg, var, data_cache
        )
        print(f"  - Loaded '{var}' with shape {data_to_plot_torch[var].shape}")

    # 4. Pre-calculate global color limits (uses original logic)
    climits = calculate_global_climits(data_to_plot_torch, diverging_vars)

    # 5. Prepare colormaps and titles
    cmap_seq = plt.get_cmap(cmap_seq_name)
    cmap_div = plt.get_cmap(cmap_div_name)
    
    colormaps = {}
    titles = {}
    for var, unit in zip(var_names, var_units):
        titles[var] = f"{var} [" + unit + "]"
        if var in diverging_vars:
            colormaps[var] = cmap_div
        else:
            colormaps[var] = cmap_seq

    # 6. Render all frames
    print("Rendering frames...")
    frame_files = []
    for t_idx in tqdm(range(num_timesteps), desc="Rendering frames"):
        frame_path = os.path.join(frame_dir, f"frame_{t_idx:05d}.png")
        frame_files.append(frame_path)

        # Build the data dict for this frame
        data_for_frame = {}
        for var in var_names:
            tensor = data_to_plot_torch[var]
            if tensor.dim() == 1: # Static data (e.g., DEM)
                data_for_frame[var] = tensor.numpy()
            else: # Time-series data (e.g., WD)
                data_for_frame[var] = tensor[t_idx, :].numpy()

        # Call the original, stateless plotting function
        plot_single_frame(
            t_idx,
            all_timestamps,
            x_coords,
            y_coords,
            data_for_frame,
            climits,
            colormaps,
            titles,
            frame_path
        )

    # 7. Read images back into memory *once*
    images = []
    if output_gif_path or output_video_path:
        for frame_file in tqdm(frame_files, desc="Reading frames into memory"):
            images.append(iio.imread(frame_file))

    # 8. Compile GIF
    if output_gif_path:
        compile_gif_from_frames(output_gif_path, fps, images)

    # 9. Compile Video
    if output_video_path:
        compile_video_from_frames(output_video_path, fps, images)

    # 10. Clean up
    try:
       shutil.rmtree(frame_dir)
       print(f"Cleaned up temporary directory: {frame_dir}")
    except Exception as e:
       print(f"Warning: Failed to clean up {frame_dir}. Error: {e}")
    
    ds.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate mSWE-GNN simulation data (config-driven).")
    
    # --- File/Var Arguments ---
    parser.add_argument("-c", "--config-path", required=True, help="Path to the main config.yaml")
    parser.add_argument("-f", "--nc-file", required=True, help="Path to the .nc simulation file")
    parser.add_argument(
        "-v", "--vars", 
        nargs=6, 
        required=True, 
        metavar="VAR",
        default=["P", "WX", "WY", "SSH", "VX", "VY"],
        help="List of EXACTLY 6 variables to animate."
    )
    parser.add_argument(
        "-u", "--units",
        nargs=6,
        required=True,
        metavar="UNIT",
        default=["m", "m s$^{-1}$", "m s$^{-1}$", "m", "m s$^{-1}$", "m s$^{-1}$"],
        help="List of 6 units (LaTeX format), in same order as --vars."
    )
    
    # --- Plotting Arguments ---
    parser.add_argument(
        "--cmap-seq", 
        default="cmocean.cm.thermal", 
        help="Colormap for sequential data (default: cmocean.cm.thermal)"
    )
    parser.add_argument(
        "--cmap-div", 
        default="cmocean.cm.balance", 
        help="Colormap for diverging data (default: cmocean.cm.balance)"
    )
    parser.add_argument(
        "--diverging-vars",
        nargs='*',
        default=["WX", "WY", "SSH", "VX", "VY"],
        help="List of variable names to treat as diverging."
    )

    # --- Output Arguments ---
    parser.add_argument("-o", "--output-video", default="adforce_6panel_animation.mp4", help="Output video file")
    parser.add_argument("--gif", action="store_true", help="Also generate a .gif version.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument(
        "--test", action="store_true", help="Run doctests and exit."
    )

    args = parser.parse_args()

    if args.test:
        print("Running doctests...")
        doctest.testmod(verbose=True)
        exit()

    if not os.path.exists(args.nc_file):
        print(f"Error: NetCDF file not found at {args.nc_file}")
    elif not os.path.exists(args.config_path):
        print(f"Error: Config file not found at {args.config_path}")
    else:
        # Check for cmocean in cmap names
        if 'cmocean' in args.cmap_seq:
            args.cmap_seq = getattr(cmocean.cm, args.cmap_seq.split('.')[-1])
        if 'cmocean' in args.cmap_div:
            args.cmap_div = getattr(cmocean.cm, args.cmap_div.split('.')[-1])

        output_gif_path = None
        if args.gif:
            output_gif_path = os.path.splitext(args.output_video)[0] + ".gif"

        print(args)
        broken_unit_str = "m s{-1}$"
        correct_unit_str = "m s$^{-1}$"
        fixed_units = [u.replace(broken_unit_str, correct_unit_str) for u in args.units]
        args.units = fixed_units

        create_animation_from_frames(
            config_path=args.config_path,
            nc_file_path=args.nc_file,
            var_names=args.vars,
            var_units=args.units,
            diverging_vars=args.diverging_vars,
            cmap_seq_name=args.cmap_seq,
            cmap_div_name=args.cmap_div,
            output_video_path=args.output_video,
            output_gif_path=output_gif_path,
            fps=args.fps,
        )
        
        print(f"\nAnimation test complete.")
        if output_gif_path:
            print(f"GIF saved to {os.path.abspath(output_gif_path)}")
        if args.output_video:
            print(f"Video saved to {os.path.abspath(args.output_video)}")