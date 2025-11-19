"""
Diagnostic animation function for mSWE-GNN (Delta Visualization).

This script generates a 12-panel animation (4 rows x 3 cols) to diagnose
model performance and instability. Instead of an autoregressive rollout,
it performs 'Teacher Forcing' (1-step prediction) for every frame.

Layout:
  Row 1: Forcing Inputs (P, WX, WY)
  Row 2: Current State (SSH, VX, VY)
  Row 3: TRUE Update/Delta (dSSH, dVX, dVY)
  Row 4: PREDICTED Update/Delta (dSSH_hat, dVX_hat, dVY_hat)

Usage:
    python -m mswegnn.utils.adforce_delta_animate \
        -c config.yaml \
        -ckpt model.ckpt \
        -nc simulation.nc \
        -o output_dir
"""

import os
import shutil
import warnings
import argparse
import yaml
import numpy as np
import xarray as xr
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio.v3 as iio
from omegaconf import OmegaConf
from typing import List, Tuple, Dict, Any

# --- IMPORTS ---
from sithom.plot import plot_defaults, label_subplots
import lightning as L
from mswegnn.utils.adforce_dataset import AdforceLazyDataset
from mswegnn.utils.adforce_misc import model_from_cfg_and_checkpoint

# Try to import cmocean
try:
    import cmocean
except ImportError:
    print("Error: 'cmocean' library not found. Please pip install cmocean")
    exit()

# Suppress Warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_static_data(
    dataset: AdforceLazyDataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads static coordinates (x, y) and DEM from the dataset cache."""
    try:
        # Retrieve path from the first entry in the index map
        nc_path = dataset.index_map[0][0]

        with xr.open_dataset(nc_path) as ds:
            x_coords = ds["x"].values
            y_coords = ds["y"].values

        # DEM should be in static_data if it's a feature, otherwise read from file
        if "DEM" in dataset.static_data:
            dem = dataset.static_data["DEM"].cpu().numpy()
        else:
            # Fallback if DEM isn't a feature
            with xr.open_dataset(nc_path) as ds:
                dem = ds["DEM"].values

    except Exception as e:
        print(f"Error loading static data: {e}")
        raise e

    return x_coords, y_coords, dem


def _get_index_map(features_cfg: Any) -> Dict[str, Dict[str, int]]:
    """Creates a mapping from variable names to their indices in feature lists."""
    try:
        # Convert ListConfig to standard list to avoid indexing errors
        forcing_vars = list(features_cfg.forcing)
        state_vars = list(features_cfg.state)
        target_vars = list(features_cfg.targets)

        return {
            "forcing": {v: i for i, v in enumerate(forcing_vars)},
            "state": {v: i for i, v in enumerate(state_vars)},
            "target": {v: i for i, v in enumerate(target_vars)},
        }
    except Exception as e:
        print(f"Error creating index map. Ensure config has forcing/state/targets. {e}")
        raise e


def get_delta_frame_data(
    dataset: AdforceLazyDataset,
    model: L.LightningModule,
    idx: int,
    dem: np.ndarray,
    idx_map: Dict[str, Dict[str, int]],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Retrieves Forcing, State, True Delta, and Predicted Delta for a single frame.
    """
    # 1. Get Batch
    batch = dataset.get(idx).to(device)

    # 2. Get Stats for Unscaling (Loaded in dataset)
    x_dyn_mean = dataset.x_dyn_mean_broadcast.to(device)  # [Previous_T * V_forcing]
    x_dyn_std = dataset.x_dyn_std_broadcast.to(device)

    y_mean = dataset.y_mean.to(device)  # [V_state]
    y_std = dataset.y_std.to(device)

    y_delta_mean = dataset.y_delta_mean.to(device)  # [V_target]
    y_delta_std = dataset.y_delta_std.to(device)

    # --- ROW 1: FORCING (P, WX, WY) ---
    # Extract the *last* time step of forcing from the flattened input
    num_forcing = len(idx_map["forcing"])
    # The batch.x structure: [Static, Forcing, State]

    num_static = dataset.features_cfg.static.__len__() + 1  # +1 for node_type
    forcing_start = num_static
    forcing_end = num_static + (dataset.previous_t * num_forcing)

    forcing_seq_scaled = batch.x[:, forcing_start:forcing_end]
    forcing_seq_unscaled = (forcing_seq_scaled * x_dyn_std) + x_dyn_mean

    # Take the last step (t)
    forcing_t_unscaled = forcing_seq_unscaled[:, -num_forcing:]

    r1_data = {}
    for var in ["P", "WX", "WY"]:
        col_idx = idx_map["forcing"][var]
        r1_data[var] = forcing_t_unscaled[:, col_idx].cpu().numpy()

    # --- ROW 2: CURRENT STATE (SSH, VX, VY) ---
    # State is at the end of x
    state_scaled = batch.x[:, forcing_end:]
    state_unscaled = (state_scaled * y_std) + y_mean

    # Extract WD, VX, VY
    wd = state_unscaled[:, idx_map["state"]["WD"]].cpu().numpy()
    vx = state_unscaled[:, idx_map["state"]["VX"]].cpu().numpy()
    vy = state_unscaled[:, idx_map["state"]["VY"]].cpu().numpy()

    # Derived SSH
    ssh = wd + dem

    r2_data = {"SSH": ssh, "VX": vx, "VY": vy}

    # --- ROW 3: TRUE DELTA (dSSH, dVX, dVY) ---
    # batch.y is the SCALED delta
    true_delta_scaled = batch.y
    true_delta_unscaled = (true_delta_scaled * y_delta_std) + y_delta_mean

    d_wd = true_delta_unscaled[:, idx_map["target"]["WD"]].cpu().numpy()
    d_vx = true_delta_unscaled[:, idx_map["target"]["VX"]].cpu().numpy()
    d_vy = true_delta_unscaled[:, idx_map["target"]["VY"]].cpu().numpy()

    # Delta SSH = Delta WD (since DEM is constant)
    r3_data = {"dSSH": d_wd, "dVX": d_vx, "dVY": d_vy}

    # --- ROW 4: PREDICTED DELTA (pred_dSSH, pred_dVX, pred_dVY) ---
    with torch.no_grad():
        pred_delta_scaled = model.model(batch)

    pred_delta_unscaled = (pred_delta_scaled * y_delta_std) + y_delta_mean

    p_d_wd = pred_delta_unscaled[:, idx_map["target"]["WD"]].cpu().numpy()
    p_d_vx = pred_delta_unscaled[:, idx_map["target"]["VX"]].cpu().numpy()
    p_d_vy = pred_delta_unscaled[:, idx_map["target"]["VY"]].cpu().numpy()

    r4_data = {"pdSSH": p_d_wd, "pdVX": p_d_vx, "pdVY": p_d_vy}

    return {**r1_data, **r2_data, **r3_data, **r4_data}


def calculate_global_limits(
    dataset: AdforceLazyDataset, idx_map: Dict[str, Dict[str, int]], dem: np.ndarray
) -> Dict[str, Tuple[float, float]]:
    """
    Scans the dataset to find global 1st/99th percentiles.
    """
    print("Calculating global color limits (scanning dataset)...")

    # Accumulators
    vars_to_scan = ["P", "WX", "WY", "SSH", "VX", "VY", "dSSH", "dVX", "dVY"]
    values = {v: [] for v in vars_to_scan}

    # For unscaling (we do this manually on CPU to avoid moving everything to GPU)
    y_delta_mean = dataset.y_delta_mean.numpy()
    y_delta_std = dataset.y_delta_std.numpy()

    # We use a stride to speed this up if dataset is huge
    stride = max(1, len(dataset) // 200)

    for idx in tqdm(range(0, len(dataset), stride), desc="Scanning"):
        data = dataset.get(idx)

        # --- TRUE DELTAS ---
        # Unscale batch.y (deltas)
        delta_raw = (data.y.numpy() * y_delta_std) + y_delta_mean

        d_wd = delta_raw[:, idx_map["target"]["WD"]]
        d_vx = delta_raw[:, idx_map["target"]["VX"]]
        d_vy = delta_raw[:, idx_map["target"]["VY"]]

        values["dSSH"].append(np.nanpercentile(d_wd, [1, 99]))
        values["dVX"].append(np.nanpercentile(d_vx, [1, 99]))
        values["dVY"].append(np.nanpercentile(d_vy, [1, 99]))

        # --- ABSOLUTE STATE (for SSH) ---
        # batch.y_unscaled is y(t+1) state. We can use this for stats.
        state_next = data.y_unscaled.numpy()  # [WD, VX, VY] (unscaled)
        wd = state_next[:, idx_map["state"]["WD"]]
        ssh = wd + dem
        vx = state_next[:, idx_map["state"]["VX"]]
        vy = state_next[:, idx_map["state"]["VY"]]

        values["SSH"].append(np.nanpercentile(ssh, [1, 99]))
        values["VX"].append(np.nanpercentile(vx, [1, 99]))
        values["VY"].append(np.nanpercentile(vy, [1, 99]))

        # --- FORCING ---
        x_dyn_mean = dataset.x_dyn_mean_broadcast.numpy()
        x_dyn_std = dataset.x_dyn_std_broadcast.numpy()
        num_forcing = len(idx_map["forcing"])
        num_static = dataset.features_cfg.static.__len__() + 1

        # Slice x
        f_start = num_static
        f_end = num_static + (dataset.previous_t * num_forcing)
        forcing_scaled = data.x[:, f_start:f_end].numpy()
        forcing_unscaled = (forcing_scaled * x_dyn_std) + x_dyn_mean

        # Last step
        f_t = forcing_unscaled[:, -num_forcing:]
        values["P"].append(np.nanpercentile(f_t[:, idx_map["forcing"]["P"]], [1, 99]))
        values["WX"].append(np.nanpercentile(f_t[:, idx_map["forcing"]["WX"]], [1, 99]))
        values["WY"].append(np.nanpercentile(f_t[:, idx_map["forcing"]["WY"]], [1, 99]))

    # Aggregation
    limits = {}
    diverging_vars = ["WX", "WY", "SSH", "VX", "VY", "dSSH", "dVX", "dVY"]

    for v in vars_to_scan:
        vals = np.array(values[v])
        # Min of 1st percentiles, Max of 99th percentiles
        g_min = np.nanmin(vals[:, 0])
        g_max = np.nanmax(vals[:, 1])

        if v in diverging_vars:
            abs_max = max(abs(g_min), abs(g_max))
            if abs_max == 0:
                abs_max = 0.1
            limits[v] = (-abs_max, abs_max)
        else:
            if g_min == g_max:
                g_max += 0.1
            limits[v] = (g_min, g_max)

    print("Global Limits:")
    for k, v in limits.items():
        print(f"  {k}: {v}")

    return limits


def plot_delta_frame(
    idx: int,
    data_dict: Dict[str, np.ndarray],
    climits: Dict[str, Tuple[float, float]],
    coords: Tuple[np.ndarray, np.ndarray],
    timestamp: str,
    save_path: str,
):
    """Renders the 4x3 grid."""
    x_coords, y_coords = coords

    # Width = 6 * 1.2 = 7.2
    # Height = (4 * 1.2) * 2 = 9.6
    fig, axs = plt.subplots(4, 3, figsize=(7.2, 9.6), sharex=True, sharey=True)

    # Matrix Layout
    rows = [
        ["P", "WX", "WY"],
        ["SSH", "VX", "VY"],
        ["dSSH", "dVX", "dVY"],
        ["pdSSH", "pdVX", "pdVY"],
    ]

    titles = [
        ["P [m]", "WX [m s$^{-1}$]", "WY [m s$^{-1}$]"],
        ["SSH [m]", "VX [m s$^{-1}$]", "VY [m s$^{-1}$]"],
        [r"$\Delta$ SSH [m]", r"$\Delta$ VX [m s$^{-1}$]", r"$\Delta$ VY [m s$^{-1}$]"],
        [
            r"$\Delta$ $\hat{\text{SSH}}$ [m]",
            r"$\Delta$ $\hat{\text{VX}}$ [m s$^{-1}$]",
            r"$\Delta$ $\hat{\text{VY}}$ [m s$^{-1}$]",
        ],
    ]

    cmaps = [
        [cmocean.cm.thermal, cmocean.cm.balance, cmocean.cm.balance],  # Row 1
        [cmocean.cm.balance, cmocean.cm.balance, cmocean.cm.balance],  # Row 2
        [cmocean.cm.balance, cmocean.cm.balance, cmocean.cm.balance],  # Row 3
        [cmocean.cm.balance, cmocean.cm.balance, cmocean.cm.balance],  # Row 4
    ]

    for i in range(4):
        for j in range(3):
            key = rows[i][j]
            ax = axs[i, j]

            # Key for limits:
            # For Row 4 (Pred), use Row 3 (True) limits key
            limit_key = key
            if i == 3:
                limit_key = rows[2][j]  # Map 'pdSSH' -> 'dSSH'

            vmin, vmax = climits[limit_key]

            scat = ax.scatter(
                x_coords,
                y_coords,
                c=data_dict[key],
                s=0.5,
                cmap=cmaps[i][j],
                vmin=vmin,
                vmax=vmax,
                marker=".",
            )

            ax.set_aspect("equal")

            ax.set_xlim(np.min(x_coords), np.max(x_coords))
            ax.set_ylim(np.min(y_coords), np.max(y_coords))
            ax.set_title(titles[i][j])

            # Add colorbar
            cbar = fig.colorbar(scat, ax=ax, shrink=0.8)
            # cbar.ax.tick_params(labelsize=8)

            if i == 3:
                ax.set_xlabel("Longitude [$^{\circ}$E]")
            if j == 0:
                ax.set_ylabel("Latitude [$^{\circ}$N]")

    # Lower the title position slightly to sit closer to the subplots
    fig.suptitle(timestamp, y=0.92, fontsize=12)

    label_subplots(axs)

    # Use tight_layout with bbox_inches='tight' in savefig to control whitespace
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # bbox_inches="tight" removes excess whitespace
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("-ckpt", "--checkpoint_path", type=str, required=True)
    parser.add_argument("-nc", "--netcdf_file", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-s", "--scaling_stats_path", type=str, default=None)
    args = parser.parse_args()

    # 1. Load Config
    cfg = OmegaConf.load(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_defaults()

    # 2. Stats Path
    stats_path = args.scaling_stats_path or cfg.data_params.scaling_stats_path
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    # 3. Init Dataset (Clean cache to ensure correctness)
    cache_dir = os.path.join(args.output_dir, "dataset_cache_delta")
    shutil.rmtree(cache_dir, ignore_errors=True)

    print("Initializing dataset...")
    dataset = AdforceLazyDataset(
        root=cache_dir,
        nc_files=[args.netcdf_file],
        previous_t=cfg.model_params.previous_t,
        features_cfg=cfg.features,
        scaling_stats_path=stats_path,
    )

    # 4. Load Model
    print("Loading model...")
    model = model_from_cfg_and_checkpoint(cfg, args.checkpoint_path)
    model.to(device)
    model.eval()

    # 5. Load Static & Limits
    x_coords, y_coords, dem = load_static_data(dataset)
    idx_map = _get_index_map(cfg.features)
    climits = calculate_global_limits(dataset, idx_map, dem)

    # 6. Prepare Output
    frame_dir = os.path.join(args.output_dir, "delta_frames")
    os.makedirs(frame_dir, exist_ok=True)

    # 7. Main Loop
    images = []
    frames_to_plot = list(range(len(dataset)))

    print(f"Generating {len(frames_to_plot)} frames...")

    for idx in tqdm(frames_to_plot):
        # Get Time
        nc_path, t_start = dataset.index_map[idx]
        # The "Current State" is at t_start + previous_t - 1
        # The "Delta" is from that time to the next.
        # We label it with the "Current State" time.
        t_current_idx = t_start + cfg.model_params.previous_t - 1

        try:
            with xr.open_dataset(nc_path) as ds:
                ts = ds.time[t_current_idx].values
                t_str = np.datetime_as_string(ts, unit="s").replace("T", " ")
        except:
            t_str = f"Index {idx}"

        # Get Data
        data_dict = get_delta_frame_data(dataset, model, idx, dem, idx_map, device)

        # Plot
        fpath = os.path.join(frame_dir, f"delta_{idx:05d}.png")
        plot_delta_frame(idx, data_dict, climits, (x_coords, y_coords), t_str, fpath)
        images.append(iio.imread(fpath))

    # 8. Compile
    out_gif = os.path.join(args.output_dir, "adforce_diagnostic_delta.gif")
    print(f"Saving GIF to {out_gif}...")
    iio.imwrite(out_gif, images, fps=8, loop=0)

    # Optional MP4
    try:
        out_mp4 = os.path.join(args.output_dir, "adforce_diagnostic_delta.mp4")
        iio.imwrite(out_mp4, images, fps=8, codec="libx264")
        print(f"Saved MP4 to {out_mp4}")
    except:
        pass

    print("Done.")
