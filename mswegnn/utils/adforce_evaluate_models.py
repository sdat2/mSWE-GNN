"""
Evaluation script for mSWE-GNN Adforce models.

This script iterates through a directory of model run results, identifies the
best checkpoint for each run (based on validation loss), and evaluates the
model's performance on Train, Validation, Test, and optional Extreme splits.

It employs an efficient "Online Accumulator" pattern to calculate multiple
metrics (e.g., SSH RMSE, Velocity Vector RMSE) in a single pass through the
dataloader, minimizing expensive I/O and GPU transfer overhead.

Usage:
    python -m mswegnn.utils.adforce_evaluate_models \
        --results_dir /home/users/sithom/my_results \
        --data_dir /home/users/sithom/swegnn_5sec \
        --conf_dir /home/users/sithom/mSWE-GNN/conf \
        --extreme_dir /home/users/sithom/SurgeNetTestPH \
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
from omegaconf import OmegaConf, DictConfig
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from mswegnn.utils.adforce_misc import model_from_cfg_and_checkpoint
from mswegnn.utils.adforce_dataset import AdforceLazyDataset


# -----------------------------------------------------------------------------
# Metric Classes
# -----------------------------------------------------------------------------


class OnlineMetric:
    """
    Base class for online metric calculation (running statistics).

    This class defines the interface for metrics that accumulate statistics
    batch-by-batch to compute a final result without storing all predictions.

    Args:
        name (str): The name of the metric (e.g., "SSH_RMSE").
        device (torch.device): The device (CPU/GPU) where tensors are stored.
    """

    def __init__(self, name: str, device: torch.device):
        self.name = name
        self.device = device
        self.reset()

    def reset(self):
        """Resets the internal accumulators to zero."""
        pass

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Updates the metric with a new batch of predictions and targets.

        Args:
            preds (torch.Tensor): Unscaled predictions in physical units.
            targets (torch.Tensor): Unscaled ground truth in physical units.
        """
        raise NotImplementedError

    def compute(self):
        """
        Computes the final metric value based on accumulated statistics.

        Returns:
            float: The computed metric value.
        """
        raise NotImplementedError


class RMSEMetric(OnlineMetric):
    """
    Computes the Root Mean Squared Error (RMSE) for scalar variables.

    Args:
        name (str): The name of the metric.
        device (torch.device): The device to use for tensor operations.

    Example:
        >>> import torch
        >>> metric = RMSEMetric("TestRMSE", torch.device("cpu"))
        >>> # Batch 1: Preds=[2, 2], True=[0, 0] -> Diff=[2, 2], Sq=[4, 4]
        >>> metric.update(torch.tensor([2.0, 2.0]), torch.tensor([0.0, 0.0]))
        >>> # Batch 2: Preds=[3], True=[3] -> Diff=[0], Sq=[0]
        >>> metric.update(torch.tensor([3.0]), torch.tensor([3.0]))
        >>> # Total Sq=8, Count=3, MSE=8/3=2.666..., RMSE=1.63299
        >>> val = metric.compute()
        >>> abs(val - 1.63299) < 1e-4
        True
    """

    def __init__(self, name: str, device: torch.device):
        super().__init__(name, device)
        self.sum_squared_error = torch.tensor(0.0, device=device)
        self.count = torch.tensor(0, device=device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Updates the running sum of squared errors."""
        # preds/targets shape: [Batch] or [Batch, 1]
        diff = preds - targets
        self.sum_squared_error += torch.sum(diff**2)
        self.count += diff.numel()

    def compute(self) -> float:
        """Returns the scalar RMSE."""
        if self.count == 0:
            return float("nan")
        mse = self.sum_squared_error / self.count
        return torch.sqrt(mse).item()


class VectorMagnitudeRMSEMetric(OnlineMetric):
    """
    Computes the RMSE of the Euclidean distance vector between prediction and truth.

    This is used for vector quantities like velocity. It computes:
    RMSE = sqrt( mean( ||v_pred - v_true||^2 ) )
         = sqrt( mean( (vx_p - vx_t)^2 + (vy_p - vy_t)^2 ) )

    Args:
        name (str): The name of the metric.
        device (torch.device): The device to use for tensor operations.

    Example:
        >>> import torch
        >>> metric = VectorMagnitudeRMSEMetric("VelRMSE", torch.device("cpu"))
        >>> # Batch 1: 2 vectors.
        >>> # Vec1: Pred=[1, 1], True=[0, 0]. ErrVec=[1, 1]. SqErr = 1^2+1^2 = 2.
        >>> # Vec2: Pred=[2, 0], True=[2, 2]. ErrVec=[0, -2]. SqErr = 0^2+(-2)^2 = 4.
        >>> preds = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
        >>> targets = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
        >>> metric.update(preds, targets)
        >>> # Total Sq Error = 2 + 4 = 6. Total Samples = 2.
        >>> # MSE = 6 / 2 = 3. RMSE = sqrt(3) ~= 1.732
        >>> val = metric.compute()
        >>> abs(val - 1.73205) < 1e-4
        True
    """

    def __init__(self, name: str, device: torch.device):
        super().__init__(name, device)
        self.sum_squared_error = torch.tensor(0.0, device=device)
        self.count = torch.tensor(0, device=device)

    def update(self, preds_vec: torch.Tensor, targets_vec: torch.Tensor):
        """
        Updates the metric using vector inputs.

        Args:
            preds_vec (torch.Tensor): Shape [Batch, Components].
            targets_vec (torch.Tensor): Shape [Batch, Components].
        """
        diff = preds_vec - targets_vec  # [Batch, Components]

        # Squared Euclidean magnitude of the error vector per sample
        # Sum over component dim (dim=1)
        squared_error_per_node = torch.sum(diff**2, dim=1)  # [Batch]

        self.sum_squared_error += torch.sum(squared_error_per_node)
        self.count += squared_error_per_node.numel()

    def compute(self) -> float:
        """Returns the vector RMSE."""
        if self.count == 0:
            return float("nan")
        mse = self.sum_squared_error / self.count
        return torch.sqrt(mse).item()


# -----------------------------------------------------------------------------
# Evaluator Engine
# -----------------------------------------------------------------------------


class ModelEvaluator:
    """
    Orchestrates the evaluation loop for a single model.

    This class handles data loading, unscaling, and distributing data to
    various metrics in a single pass. It is robust to feature permutations
    in the configuration file.

    Args:
        model (torch.nn.Module): The loaded PyTorch model.
        loader (DataLoader): The PyG DataLoader for the dataset.
        device (torch.device): The computing device (CPU or GPU).
        cfg (DictConfig): The model configuration object.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        cfg: DictConfig,
    ):
        self.model = model
        self.loader = loader
        self.device = device
        self.cfg = cfg
        self.metrics = []
        self.target_map = self._build_target_map()

        # --- Configure Metrics Dynamically ---

        # 1. SSH RMSE (Target: WD)
        if "WD" in self.target_map:
            self.metrics.append(
                {
                    "metric": RMSEMetric("SSH_RMSE", device),
                    "indices": [self.target_map["WD"]],
                }
            )

        # 2. Velocity Vector RMSE (Targets: VX, VY)
        if "VX" in self.target_map and "VY" in self.target_map:
            self.metrics.append(
                {
                    "metric": VectorMagnitudeRMSEMetric("Vel_RMSE", device),
                    "indices": [self.target_map["VX"], self.target_map["VY"]],
                }
            )

    def _build_target_map(self) -> dict:
        """
        Creates a dictionary mapping target names to their indices in the output.

        Returns:
            dict: Mapping of target name to index (e.g., {'WD': 0, 'VX': 1}).
        """
        # Cast omegaconf list to standard list to be safe
        targets = list(self.cfg.features.targets)
        return {name: i for i, name in enumerate(targets)}

    def run(self) -> dict:
        """
        Executes the evaluation loop over the DataLoader.

        Returns:
            dict: A dictionary of computed results (e.g., {"SSH_RMSE": 0.12}).
        """
        self.model.eval()

        # Pre-fetch Unscaling params to device
        ds = self.loader.dataset

        # Handle cases where scaling might be disabled or different
        if hasattr(ds, "y_delta_mean"):
            y_mean = ds.y_delta_mean.to(self.device)
            y_std = ds.y_delta_std.to(self.device)
        else:
            # Fallback if dataset doesn't have stats loaded
            num_targets = len(self.cfg.features.targets)
            y_mean = torch.zeros(num_targets, device=self.device)
            y_std = torch.ones(num_targets, device=self.device)

        with torch.no_grad():
            for batch in tqdm(self.loader, desc="Evaluating", leave=False):
                batch = batch.to(self.device)

                # 1. Forward Pass
                if hasattr(self.model, "model"):
                    out_scaled = self.model.model(batch)
                else:
                    out_scaled = self.model(batch)

                # 2. Unscale Everything Once (Vectorized on GPU)
                # Predicted Raw Delta
                pred_delta_raw = (out_scaled * y_std) + y_mean

                # True Raw Delta (Ground Truth)
                true_delta_raw = (batch.y * y_std) + y_mean

                # 3. Update All Metrics
                for item in self.metrics:
                    metric = item["metric"]
                    indices = item["indices"]

                    # Slice specific features based on config indices
                    # shape: [Batch, len(indices)]
                    p_slice = pred_delta_raw[:, indices]
                    t_slice = true_delta_raw[:, indices]

                    # If single dimension metric (RMSE), squeeze to [Batch]
                    if len(indices) == 1:
                        p_slice = p_slice.squeeze(-1)
                        t_slice = t_slice.squeeze(-1)

                    metric.update(p_slice, t_slice)

        # 4. Compute Final Results
        results = {}
        for item in self.metrics:
            metric = item["metric"]
            results[metric.name] = metric.compute()

        return results


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def find_best_checkpoint(checkpoint_dir: str) -> str:
    """
    Finds the checkpoint file with the lowest validation loss.

    Args:
        checkpoint_dir (str): Path to the directory containing .ckpt files.

    Returns:
        str: Full path to the best checkpoint, or None if not found.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not ckpt_files:
        return None
    best_ckpt = None
    min_loss = float("inf")
    for ckpt in ckpt_files:
        match = re.search(r"val_loss=([0-9]+\.[0-9]+)", ckpt)
        if match:
            try:
                loss = float(match.group(1))
                if loss < min_loss:
                    min_loss = loss
                    best_ckpt = ckpt
            except ValueError:
                continue
    return best_ckpt


def get_run_paths(run_dir: str) -> dict:
    """
    Retrieves paths for checkpoint, config, and stats for a given run.

    Args:
        run_dir (str): The root directory of the model run.

    Returns:
        dict: Keys 'ckpt', 'config', 'stats' with file paths, or None if invalid.
    """
    paths = {}
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    paths["ckpt"] = find_best_checkpoint(ckpt_dir)

    cfg_path_ckpt = os.path.join(ckpt_dir, "config.yaml")
    cfg_path_root = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cfg_path_ckpt):
        paths["config"] = cfg_path_ckpt
    elif os.path.exists(cfg_path_root):
        paths["config"] = cfg_path_root
    else:
        paths["config"] = None

    stats_path = os.path.join(run_dir, "processed", "scaling_stats.yaml")
    if os.path.exists(stats_path):
        paths["stats"] = stats_path
    else:
        paths["stats"] = None

    if all(paths.values()):
        return paths
    else:
        return None


def load_file_list(list_path: str, data_root: str) -> list:
    """
    Loads a list of filenames from a YAML file and prepends the data root.

    Args:
        list_path (str): Path to the YAML file with the list.
        data_root (str): Directory to prepend to filenames.

    Returns:
        list: List of full file paths.

    Raises:
        FileNotFoundError: If list_path does not exist.
    """
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"Could not find split file: {list_path}")
    with open(list_path, "r") as f:
        filenames = yaml.safe_load(f)
    return [os.path.join(data_root, fname) for fname in filenames]


def evaluate_run(
    run_name: str,
    run_paths: dict,
    data_root: str,
    conf_dir: str,
    extreme_dir: str = None,
) -> dict:
    """
    Evaluates a single model run across multiple data splits.

    Args:
        run_name (str): Identifier for the run.
        run_paths (dict): Paths to ckpt, config, and stats.
        data_root (str): Base directory for raw .nc files.
        conf_dir (str): Directory containing split YAMLs (train.yaml, etc.).
        extreme_dir (str, optional): Path to extreme test set directory.

    Returns:
        dict: Dictionary of evaluation results (RMSEs) for the run.
    """

    # Load Config
    cfg = OmegaConf.load(run_paths["config"])

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = model_from_cfg_and_checkpoint(cfg, run_paths["ckpt"]).to(device)
    except Exception as e:
        print(f"Failed to load model {run_name}: {e}")
        return None

    results = {
        "Model": run_name,
    }

    # --- DYNAMIC BATCH SIZE SELECTION ---
    batch_size = 32
    if cfg.model_params.model_type == "GNN" and cfg.models.type_gnn == "SWEGNN":
        batch_size = cfg.trainer_options.get("batch_size", 8)
        print(
            f"  Note: Using conservative batch size {batch_size} for memory-intensive SWEGNN."
        )

    # Define tasks
    eval_tasks = []
    for split in ["train", "val", "test"]:
        eval_tasks.append(
            {
                "name": split,
                "type": "yaml",
                "path": os.path.join(conf_dir, f"{split}.yaml"),
                "root": data_root,
            }
        )

    if extreme_dir:
        eval_tasks.append(
            {"name": "extreme", "type": "dir", "path": extreme_dir, "root": extreme_dir}
        )

    for task in eval_tasks:
        split_name = task["name"].capitalize()
        try:
            if task["type"] == "yaml":
                nc_files = load_file_list(task["path"], task["root"])
            elif task["type"] == "dir":
                nc_files = sorted(
                    glob.glob(os.path.join(task["path"], "**", "*.nc"), recursive=True)
                )
                if not nc_files:
                    print(f"Warning: No .nc files found in {task['path']}")
                    # Fill NaNs
                    results[f"{split_name} SSH_RMSE"] = np.nan
                    continue

            cache_dir = os.path.join(task["root"], f"processed_{task['name']}_cache")
            os.makedirs(cache_dir, exist_ok=True)

            ds = AdforceLazyDataset(
                root=cache_dir,
                nc_files=nc_files,
                previous_t=cfg.model_params.previous_t,
                features_cfg=cfg.features,
                scaling_stats_path=run_paths["stats"],
            )

            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True if len(nc_files) > 8 else False,
            )

            print(
                f"[{run_name}] {split_name} set ({len(nc_files)} files)... ",
                end="",
                flush=True,
            )

            # --- USE NEW EVALUATOR ENGINE ---
            evaluator = ModelEvaluator(model, loader, device, cfg)
            metrics_results = evaluator.run()

            # Flatten results: e.g. "Train SSH_RMSE", "Train Vel_RMSE"
            for metric_name, val in metrics_results.items():
                results[f"{split_name} {metric_name}"] = val

            # Summary String
            summary = ", ".join([f"{k}: {v:.4f}" for k, v in metrics_results.items()])
            print(f"Done. [{summary}]")

        except FileNotFoundError:
            print(f"\nWarning: Could not find file list/dir for {task['name']}")
        except Exception as e:
            if isinstance(e, torch.cuda.OutOfMemoryError) and batch_size <= 4:
                print(f"\nFATAL ERROR: OOM occurred. Cannot continue.")
                raise e
            print(f"\nError evaluating {task['name']} for {run_name}: {e}")

    return results


if __name__ == "__main__":
    # python -m mswegnn.utils.adforce_evaluate_models
    parser = argparse.ArgumentParser(
        description="Evaluate mSWE-GNN models (SSH & Velocity RMSE)."
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        required=True,
        help="Path to results/models dir",
    )
    parser.add_argument(
        "-d", "--data_dir", type=str, required=True, help="Path to raw data dir"
    )
    parser.add_argument(
        "-c",
        "--conf_dir",
        type=str,
        default="conf",
        help="Path to conf dir (train.yaml etc)",
    )
    parser.add_argument(
        "-e", "--extreme_dir", type=str, default=None, help="Optional extreme test dir"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="evaluation_results",
        help="Output filename base",
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        exit(1)

    run_dirs = sorted(glob.glob(os.path.join(args.results_dir, "*")))
    all_results = []

    print(f"Scanning {len(run_dirs)} directories in {args.results_dir}...\n")

    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue

        run_name = os.path.basename(run_dir)
        paths = get_run_paths(run_dir)
        if not paths:
            continue

        print(f"=== Evaluating: {run_name} ===")
        try:
            res = evaluate_run(
                run_name, paths, args.data_dir, args.conf_dir, args.extreme_dir
            )
            if res:
                all_results.append(res)
                print("")
        except torch.cuda.OutOfMemoryError as e:
            print(f"\nSkipping remaining evaluations due to persistent OOM error.")
            break

    if not all_results:
        print("No valid results found.")
        exit(0)

    df = pd.DataFrame(all_results)

    # Sort by Validation SSH RMSE if available
    sort_col = "Val SSH_RMSE" if "Val SSH_RMSE" in df.columns else df.columns[-1]
    df = df.sort_values(sort_col)

    print("\n### Results Summary")
    print(df.to_string(index=False, float_format="%.4f"))

    # LaTeX
    latex = df.to_latex(
        index=False,
        float_format="%.4f",
        caption="RMSE of predictions across splits.",
        label="tab:results",
        escape=False,
    )
    with open(f"{args.output}.tex", "w") as f:
        f.write(latex)

    # Markdown
    with open(f"{args.output}.md", "w") as f:
        try:
            f.write(df.to_markdown(index=False, floatfmt=".4f"))
        except ImportError:
            f.write(df.to_string(index=False, float_format="%.4f"))

    print(f"\nSaved results to {args.output}.tex and {args.output}.md")
