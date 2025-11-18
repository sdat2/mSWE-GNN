"""
python -m mswegnn.utils.adforce_evaluate_models \
    --results_dir /home/users/sithom/my_results \
    --data_dir /home/users/sithom/swegnn_5sec \
    --conf_dir /home/users/sithom/mSWE-GNN/conf \
    --output comparison

"""

import os
import glob
import re
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from mswegnn.utils.adforce_misc import model_from_cfg_and_checkpoint
from mswegnn.utils.adforce_dataset import AdforceLazyDataset


def find_best_checkpoint(checkpoint_dir: str) -> str:
    """Finds the .ckpt file with the lowest validation loss."""
    if not os.path.isdir(checkpoint_dir):
        return None

    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not ckpt_files:
        return None

    # Regex finds 'val_loss=0.1234' in filenames like 'GNN-epoch=99-val_loss=0.3644.ckpt'
    best_ckpt = None
    min_loss = float("inf")

    for ckpt in ckpt_files:
        match = re.search(r"val_loss=([0-9]+\.[0-9]+)", ckpt)
        if match:
            loss = float(match.group(1))
            if loss < min_loss:
                min_loss = loss
                best_ckpt = ckpt

    return best_ckpt


def get_run_paths(run_dir: str) -> dict:
    """
    Validates a run directory and returns paths to critical files.
    Returns None if the run is incomplete or invalid.
    """
    paths = {}

    # 1. Checkpoints
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    paths["ckpt"] = find_best_checkpoint(ckpt_dir)

    # 2. Config (try checkpoints dir first, then root)
    cfg_path_ckpt = os.path.join(ckpt_dir, "config.yaml")
    cfg_path_root = os.path.join(run_dir, "config.yaml")

    if os.path.exists(cfg_path_ckpt):
        paths["config"] = cfg_path_ckpt
    elif os.path.exists(cfg_path_root):
        paths["config"] = cfg_path_root
    else:
        paths["config"] = None

    # 3. Scaling Stats (The Robust Fix)
    # We look for it in run_dir/processed/scaling_stats.yaml
    stats_path = os.path.join(run_dir, "processed", "scaling_stats.yaml")
    if os.path.exists(stats_path):
        paths["stats"] = stats_path
    else:
        paths["stats"] = None

    # Validation: Do we have everything?
    if all(paths.values()):
        return paths
    else:
        return None


def load_file_list(list_path: str, data_root: str) -> list[str]:
    """Loads list of .nc filenames and prepends the local data root."""
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"Could not find split file: {list_path}")

    with open(list_path, "r") as f:
        filenames = yaml.safe_load(f)
    return [os.path.join(data_root, fname) for fname in filenames]


def get_ssh_delta_rmse(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, target_idx: int
) -> float:
    """Calculates unscaled RMSE for a specific target index (SSH)."""
    model.eval()
    squared_errors = []

    # Extract unscaling parameters for the target variable
    ds = loader.dataset
    y_delta_mean = ds.y_delta_mean[target_idx].to(device)
    y_delta_std = ds.y_delta_std[target_idx].to(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)

            # Predict
            out_scaled = model(batch)

            # Unscale Prediction
            pred_delta_scaled = out_scaled[:, target_idx]
            pred_delta_raw = (pred_delta_scaled * y_delta_std) + y_delta_mean

            # Unscale Ground Truth
            true_delta_scaled = batch.y[:, target_idx]
            true_delta_raw = (true_delta_scaled * y_delta_std) + y_delta_mean

            # Error
            se = (pred_delta_raw - true_delta_raw) ** 2
            squared_errors.append(se.cpu().numpy())

    if not squared_errors:
        return float("nan")

    rmse = np.sqrt(np.mean(np.concatenate(squared_errors)))
    return float(rmse)


def evaluate_run(run_name: str, run_paths: dict, data_root: str, conf_dir: str) -> dict:
    """Loads model and evaluates on all splits."""

    # Load Config
    cfg = OmegaConf.load(run_paths["config"])

    # Check if model predicts WD (SSH)
    try:
        wd_idx = list(cfg.features.targets).index("WD")
    except ValueError:
        print(f"Skipping {run_name}: Target 'WD' not found in config.")
        return None

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = model_from_cfg_and_checkpoint(cfg, run_paths["ckpt"]).to(device)
    except Exception as e:
        print(f"Failed to load model {run_name}: {e}")
        return None

    results = {
        "Model": run_name,
        "Checkpoint": os.path.basename(run_paths["ckpt"]),
    }

    # Define splits using the provided conf_dir
    split_files = {
        "train": os.path.join(conf_dir, "train.yaml"),
        "val": os.path.join(conf_dir, "val.yaml"),
        "test": os.path.join(conf_dir, "test.yaml"),
    }

    for split, list_file in split_files.items():
        try:
            nc_files = load_file_list(list_file, data_root)

            # Initialize Dataset with the RUN-SPECIFIC stats file
            # We use a cache folder inside data_root to avoid clutter
            cache_dir = os.path.join(data_root, f"processed_{split}_cache")
            os.makedirs(cache_dir, exist_ok=True)

            ds = AdforceLazyDataset(
                root=cache_dir,
                nc_files=nc_files,
                previous_t=cfg.model_params.previous_t,
                features_cfg=cfg.features,
                scaling_stats_path=run_paths["stats"],  # <--- CRITICAL: Local path
            )

            # Use larger batch size for inference if possible
            loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

            print(f"[{run_name}] {split.capitalize()} set... ", end="", flush=True)
            rmse = get_ssh_delta_rmse(model, loader, device, wd_idx)
            print(f"RMSE: {rmse:.4f}")
            results[f"{split.capitalize()} RMSE"] = rmse

        except FileNotFoundError:
            print(f"\nWarning: Could not find file list for {split} at {list_file}")
            results[f"{split.capitalize()} RMSE"] = np.nan
        except Exception as e:
            print(f"\nError evaluating {split} for {run_name}: {e}")
            results[f"{split.capitalize()} RMSE"] = np.nan

    return results


if __name__ == "__main__":
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(
        description="Evaluate mSWE-GNN models (SSH Delta RMSE) across multiple runs."
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        required=True,
        help="Path to the base directory containing model run subfolders (e.g., /path/to/my_results)",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing raw .nc files (e.g., /path/to/swegnn_5sec)",
    )
    parser.add_argument(
        "-c",
        "--conf_dir",
        type=str,
        default="conf",
        help="Path to directory containing train.yaml, val.yaml, test.yaml (default: 'conf')",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="ssh_rmse_results",
        help="Base filename for output tables (default: 'ssh_rmse_results')",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        exit(1)
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        exit(1)
    if not os.path.exists(args.conf_dir):
        print(f"Error: Configuration directory not found: {args.conf_dir}")
        exit(1)

    run_dirs = sorted(glob.glob(os.path.join(args.results_dir, "*")))
    all_results = []

    print(f"Scanning {len(run_dirs)} directories in {args.results_dir}...\n")

    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue

        run_name = os.path.basename(run_dir)

        # 1. Validate directory structure
        paths = get_run_paths(run_dir)
        if not paths:
            # Silently skip invalid folders (e.g., logs, empty dirs)
            continue

        # 2. Evaluate
        print(f"=== Evaluating: {run_name} ===")
        print(f"  Ckpt: {os.path.basename(paths['ckpt'])}")

        res = evaluate_run(run_name, paths, args.data_dir, args.conf_dir)
        if res:
            all_results.append(res)
            print("")  # Newline spacing

    if not all_results:
        print("No valid results found.")
        exit(0)

    # Output Formatting
    df = pd.DataFrame(all_results)

    # Sort by Validation RMSE if available
    sort_col = "Val RMSE" if "Val RMSE" in df.columns else df.columns[-1]
    df = df.sort_values(sort_col)

    print("\n### Results Summary")
    print(df.to_markdown(index=False, floatfmt=".4f"))

    # Save LaTeX
    latex = df.to_latex(
        index=False,
        float_format="%.4f",
        caption="RMSE of SSH Delta predictions (meters) across splits.",
        label="tab:ssh_results",
        escape=False,
    )
    # Format Latex for academic papers (booktabs style)
    latex = (
        latex.replace("\\toprule", "\\topline")
        .replace("\\midrule", "\\midline")
        .replace("\\bottomrule", "\\bottomline")
    )

    tex_file = f"{args.output}.tex"
    with open(tex_file, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {tex_file}")

    # Save Markdown
    md_file = f"{args.output}.md"
    with open(md_file, "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
    print(f"Markdown table saved to {md_file}")
