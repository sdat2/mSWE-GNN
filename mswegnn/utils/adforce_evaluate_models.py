"""
Evaluation script for mSWE-GNN Adforce models.

This script iterates through a directory of model run results, identifies the
best checkpoint for each run (based on validation loss), and evaluates the
model's performance on Train, Validation, Test, and optional Extreme splits.

It employs an efficient "Online Accumulator" pattern to calculate multiple
metrics (e.g., SSH RMSE, Velocity Vector RMSE, Bias, NSE, MaxError)
in a single pass through the dataloader, minimizing expensive I/O and GPU overhead.

The output is organized into a directory containing separate tables (.tex and .md)
for each metric type, allowing for specific comparisons (e.g., "Which model has
the best tail behavior?" vs "Which model has the best average accuracy?").

Usage:
    python -m mswegnn.utils.adforce_evaluate_models \
        --results_dir /home/users/sithom/my_results \
        --data_dir /home/users/sithom/swegnn_5sec \
        --conf_dir /home/users/sithom/mSWE-GNN/conf \
        --extreme_dir /home/users/sithom/SurgeNetTestPH \
        --output results_tables_v1

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
from mswegnn.utils.adforce_loop import (
    load_file_list,
    get_run_paths,
    find_best_checkpoint,
)


# -----------------------------------------------------------------------------
# Metric Classes
# -----------------------------------------------------------------------------


class OnlineMetric:
    """
    Base class for online metric calculation (running statistics).

    This class defines the interface for metrics that accumulate statistics
    batch-by-batch to compute a final result without storing all predictions.

    Attributes:
        name (str): The name of the metric (e.g., "SSH_RMSE").
        device (torch.device): The device (CPU/GPU) where tensors are stored.
    """

    def __init__(self, name: str, device: torch.device):
        """
        Initializes the OnlineMetric.

        Args:
            name (str): The identifier for the metric.
            device (torch.device): The device to store accumulators on.
        """
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

    def compute(self) -> float:
        """
        Computes the final metric value based on accumulated statistics.

        Returns:
            float: The computed metric value.
        """
        raise NotImplementedError


class RMSEMetric(OnlineMetric):
    """
    Computes the Root Mean Squared Error (RMSE) for scalar variables.

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

    Example:
        >>> import torch
        >>> metric = VectorMagnitudeRMSEMetric("VelRMSE", torch.device("cpu"))
        >>> # Batch 1: 2 vectors.
        >>> # Vec1: Pred=[1, 1], True=[0, 0]. ErrVec=[1, 1]. SqErr = 2.
        >>> # Vec2: Pred=[2, 0], True=[2, 2]. ErrVec=[0, -2]. SqErr = 4.
        >>> preds = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
        >>> targets = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
        >>> metric.update(preds, targets)
        >>> # Total Sq Error = 6. Total Samples = 2. RMSE = sqrt(3) ~= 1.732
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
        diff = preds_vec - targets_vec
        # Squared Euclidean magnitude of the error vector per sample
        # Sum over component dim (dim=1), then sum over batch
        self.sum_squared_error += torch.sum(torch.sum(diff**2, dim=1))
        self.count += diff.shape[0]

    def compute(self) -> float:
        """Returns the vector RMSE."""
        if self.count == 0:
            return float("nan")
        mse = self.sum_squared_error / self.count
        return torch.sqrt(mse).item()


class BiasMetric(OnlineMetric):
    """
    Computes Mean Signed Error (Bias) for scalar variables.

    Negative values indicate systematic under-prediction (damping).
    Positive values indicate systematic over-prediction.

    Example:
        >>> import torch
        >>> m = BiasMetric("Bias", torch.device("cpu"))
        >>> m.update(torch.tensor([1.0]), torch.tensor([2.0])) # Error -1
        >>> m.compute()
        -1.0
    """

    def __init__(self, name: str, device: torch.device):
        super().__init__(name, device)
        self.sum_error = torch.tensor(0.0, device=device)
        self.count = torch.tensor(0, device=device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Updates the running sum of signed errors."""
        diff = preds - targets
        self.sum_error += torch.sum(diff)
        self.count += diff.numel()

    def compute(self) -> float:
        """Returns the Mean Signed Error."""
        if self.count == 0:
            return float("nan")
        return (self.sum_error / self.count).item()


class VectorMagnitudeBiasMetric(OnlineMetric):
    """
    Computes the Bias of the Magnitude (Speed Bias).

    This metric determines if the model generally predicts faster or slower
    water than reality, regardless of direction.

    Formula: Bias = Mean( ||pred|| - ||target|| )
    """

    def __init__(self, name: str, device: torch.device):
        super().__init__(name, device)
        self.sum_error = torch.tensor(0.0, device=device)
        self.count = torch.tensor(0, device=device)

    def update(self, preds_vec: torch.Tensor, targets_vec: torch.Tensor):
        """
        Updates bias based on vector norms.

        Args:
            preds_vec (torch.Tensor): [Batch, Components]
            targets_vec (torch.Tensor): [Batch, Components]
        """
        norm_p = torch.norm(preds_vec, dim=1)
        norm_t = torch.norm(targets_vec, dim=1)
        self.sum_error += torch.sum(norm_p - norm_t)
        self.count += norm_p.numel()

    def compute(self) -> float:
        """Returns the Speed Bias."""
        if self.count == 0:
            return float("nan")
        return (self.sum_error / self.count).item()


class MaxErrorMetric(OnlineMetric):
    """
    Tracks the single largest absolute error (or vector error magnitude)
    observed in the entire dataset.

    This is crucial for identifying catastrophic failures or instabilities
    that might be masked by a low average RMSE.
    """

    def __init__(self, name: str, device: torch.device):
        super().__init__(name, device)
        self.max_error = torch.tensor(0.0, device=device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Updates the max error seen so far."""
        if preds.dim() > 1 and preds.shape[1] > 1:
            # Vector magnitude error
            error = torch.norm(preds - targets, dim=1)
        else:
            error = torch.abs(preds - targets).flatten()

        batch_max = torch.max(error)
        if batch_max > self.max_error:
            self.max_error = batch_max

    def compute(self) -> float:
        """Returns the maximum error encountered."""
        return self.max_error.item()


class NSEMetric(OnlineMetric):
    """
    Nash-Sutcliffe Efficiency (NSE).

    Formula: NSE = 1 - (MSE / Variance)

    Interpretation:
        * NSE = 1: Perfect model.
        * NSE = 0: Predictive power equal to the mean of the target.
        * NSE < 0: Worse than the mean.

    Uses float64 precision for internal accumulators to prevent numerical instability.
    """

    def __init__(self, name: str, device: torch.device):
        super().__init__(name, device)
        # Numerator: Sum of Squared Errors
        self.sum_sq_error = torch.tensor(0.0, device=device, dtype=torch.float64)
        # Denominator: Target Statistics for Variance
        self.sum_target = torch.tensor(0.0, device=device, dtype=torch.float64)
        self.sum_sq_target = torch.tensor(0.0, device=device, dtype=torch.float64)
        self.count = torch.tensor(0, device=device, dtype=torch.float64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Updates accumulators for MSE and Target Variance."""
        if preds.dim() > 1 and preds.shape[1] > 1:
            # Vector: Sum of squared errors of components
            diff = preds - targets
            self.sum_sq_error += torch.sum(diff**2)
            # Target Variance is calculated over all components flattened
            t_flat = targets.flatten().to(dtype=torch.float64)
        else:
            # Scalar
            diff = preds - targets
            self.sum_sq_error += torch.sum(diff**2)
            t_flat = targets.flatten().to(dtype=torch.float64)

        self.sum_target += torch.sum(t_flat)
        self.sum_sq_target += torch.sum(t_flat**2)
        self.count += t_flat.numel()

    def compute(self) -> float:
        """Returns the NSE value."""
        if self.count == 0:
            return float("nan")

        # Variance = E[x^2] - (E[x])^2
        variance = (self.sum_sq_target / self.count) - (
            self.sum_target / self.count
        ) ** 2
        mse = self.sum_sq_error / self.count

        if variance <= 1e-9:
            # Target is a flat line; any error is infinite penalty relative to variance
            return float("-inf")

        return (1.0 - (mse / variance)).item()


class TargetStdMetric(OnlineMetric):
    """
    Computes the Standard Deviation of the target variable (ground truth).

    This provides a baseline for the 'difficulty' of the dataset. High standard
    deviation in the target deltas implies a more volatile system.
    """

    def __init__(self, name: str, device: torch.device):
        super().__init__(name, device)
        self.sum_x = torch.tensor(0.0, device=device, dtype=torch.float64)
        self.sum_sq_x = torch.tensor(0.0, device=device, dtype=torch.float64)
        self.count = torch.tensor(0, device=device, dtype=torch.float64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Updates stats based on targets only (preds are ignored)."""
        if targets.dim() > 1 and targets.shape[1] > 1:
            # For vectors, compute std of magnitudes
            vals = torch.linalg.norm(targets, dim=1)
        else:
            vals = targets.flatten()

        vals_64 = vals.to(dtype=torch.float64)
        self.sum_x += torch.sum(vals_64)
        self.sum_sq_x += torch.sum(vals_64**2)
        self.count += vals.numel()

    def compute(self) -> float:
        """Returns the standard deviation."""
        if self.count == 0:
            return float("nan")

        mean = self.sum_x / self.count
        variance = (self.sum_sq_x / self.count) - (mean**2)

        # Clip negative values due to float precision errors
        if variance < 0:
            variance = torch.tensor(0.0, device=self.device, dtype=torch.float64)

        return torch.sqrt(variance).item()


# -----------------------------------------------------------------------------
# Evaluator Engine
# -----------------------------------------------------------------------------


class ModelEvaluator:
    """
    Orchestrates the evaluation loop for a single model.

    This class handles data loading, unscaling, and distributing data to
    various metrics in a single pass. It is robust to feature permutations
    in the configuration file by using a dynamic target map.

    Attributes:
        model (torch.nn.Module): The loaded PyTorch model.
        loader (DataLoader): The PyG DataLoader for the dataset.
        device (torch.device): The computing device (CPU or GPU).
        cfg (DictConfig): The model configuration object.
        metrics (list): A list of dictionaries containing metric objects and indices.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        cfg: DictConfig,
    ):
        """
        Initializes the ModelEvaluator.

        Args:
            model (torch.nn.Module): The loaded PyTorch model.
            loader (DataLoader): The PyG DataLoader for the dataset.
            device (torch.device): The computing device (CPU or GPU).
            cfg (DictConfig): The model configuration object.
        """
        self.model = model
        self.loader = loader
        self.device = device
        self.cfg = cfg
        self.metrics = []
        self.target_map = self._build_target_map()

        # --- 1. SSH Metrics (Target: WD) ---
        if "WD" in self.target_map:
            idx = [self.target_map["WD"]]
            self.metrics.extend(
                [
                    {"metric": RMSEMetric("SSH_RMSE", device), "indices": idx},
                    {"metric": BiasMetric("SSH_Bias", device), "indices": idx},
                    {"metric": MaxErrorMetric("SSH_MaxErr", device), "indices": idx},
                    {"metric": NSEMetric("SSH_NSE", device), "indices": idx},
                    {"metric": TargetStdMetric("SSH_SD", device), "indices": idx},
                ]
            )

        # --- 2. Velocity Metrics (Targets: VX, VY) ---
        if "VX" in self.target_map and "VY" in self.target_map:
            idx = [self.target_map["VX"], self.target_map["VY"]]
            self.metrics.extend(
                [
                    {
                        "metric": VectorMagnitudeRMSEMetric("Vel_RMSE", device),
                        "indices": idx,
                    },
                    {
                        "metric": VectorMagnitudeBiasMetric("Vel_Bias", device),
                        "indices": idx,
                    },
                    {"metric": MaxErrorMetric("Vel_MaxErr", device), "indices": idx},
                    {"metric": NSEMetric("Vel_NSE", device), "indices": idx},
                    {"metric": TargetStdMetric("Vel_SD", device), "indices": idx},
                ]
            )

    def _build_target_map(self) -> dict:
        """
        Creates a dictionary mapping target names to their indices in the output.

        Returns:
            dict: Mapping of target name to index (e.g., {'WD': 0, 'VX': 1}).
        """
        targets = list(self.cfg.features.targets)
        return {name: i for i, name in enumerate(targets)}

    def run(self) -> dict:
        """
        Executes the evaluation loop.

        Returns:
            dict: A dictionary of computed results (e.g., {"SSH_RMSE": 0.12}).
        """
        self.model.eval()
        ds = self.loader.dataset

        # Unscaling parameters
        if hasattr(ds, "y_delta_mean"):
            y_mean = ds.y_delta_mean.to(self.device)
            y_std = ds.y_delta_std.to(self.device)
        else:
            num_targets = len(self.cfg.features.targets)
            y_mean = torch.zeros(num_targets, device=self.device)
            y_std = torch.ones(num_targets, device=self.device)

        with torch.no_grad():
            for batch in tqdm(self.loader, desc="Evaluating", leave=False):
                batch = batch.to(self.device)

                if hasattr(self.model, "model"):
                    out_scaled = self.model.model(batch)
                else:
                    out_scaled = self.model(batch)

                # Unscale: Raw Delta = (ScaledOutput * Std) + Mean
                pred_delta_raw = (out_scaled * y_std) + y_mean
                true_delta_raw = (batch.y * y_std) + y_mean

                # Update Metrics ("Piggybacking" on the GPU tensors)
                for item in self.metrics:
                    metric = item["metric"]
                    indices = item["indices"]

                    p_slice = pred_delta_raw[:, indices]
                    t_slice = true_delta_raw[:, indices]

                    if len(indices) == 1:
                        p_slice = p_slice.squeeze(-1)
                        t_slice = t_slice.squeeze(-1)

                    metric.update(p_slice, t_slice)

        results = {}
        for item in self.metrics:
            metric = item["metric"]
            results[metric.name] = metric.compute()

        return results


def evaluate_run(
    run_name: str, run_paths: dict, data_root: str, conf_dir: str, extreme_dir: str
) -> dict:
    """
    Evaluates a single model run across all defined splits.

    Args:
        run_name (str): The name of the run.
        run_paths (dict): Dictionary of paths (ckpt, config, stats).
        data_root (str): Path to raw data directory.
        conf_dir (str): Path to configuration directory.
        extreme_dir (str): Path to extreme test directory.

    Returns:
        dict: A dictionary containing evaluation metrics for all splits.
    """
    cfg = OmegaConf.load(run_paths["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = model_from_cfg_and_checkpoint(cfg, run_paths["ckpt"]).to(device)
    except Exception as e:
        print(f"Failed load {run_name}: {e}")
        return None

    results = {"Model": run_name}

    # Adjust batch size for memory-intensive models
    batch_size = 32
    if cfg.model_params.model_type == "GNN" and cfg.models.type_gnn == "SWEGNN":
        batch_size = cfg.trainer_options.get("batch_size", 8)
        print(f"  Note: Using conservative batch size {batch_size} for SWEGNN.")

    # Define Evaluation Tasks
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
            {
                "name": "extreme",
                "type": "dir",
                "path": extreme_dir,
                "root": extreme_dir,
            }
        )

    for task in eval_tasks:
        split_name = task["name"].capitalize()
        try:
            # Determine file list
            if task["type"] == "yaml":
                nc_files = load_file_list(task["path"], task["root"])
            elif task["type"] == "dir":
                nc_files = sorted(
                    glob.glob(os.path.join(task["path"], "**", "*.nc"), recursive=True)
                )
                if not nc_files:
                    continue

            # Setup caching and dataset
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
                persistent_workers=(len(nc_files) > 8),
            )

            print(
                f"[{run_name}] {split_name} ({len(nc_files)} files)... ",
                end="",
                flush=True,
            )

            # Run One-Pass Evaluator
            evaluator = ModelEvaluator(model, loader, device, cfg)
            metrics_results = evaluator.run()

            # Flatten results: e.g., "Train SSH_RMSE": 0.02
            for k, v in metrics_results.items():
                results[f"{split_name} {k}"] = v

            # Concise summary for console
            summary = []
            for k, v in metrics_results.items():
                if "RMSE" in k:
                    summary.append(f"{k}: {v:.4f}")
            print(f"Done. [{', '.join(summary)}]")

        except Exception as e:
            if isinstance(e, torch.cuda.OutOfMemoryError) and batch_size <= 4:
                raise e
            print(f"\nError {task['name']}: {e}")

    return results


def save_metric_tables(df: pd.DataFrame, output_dir: str) -> None:
    """
    Pivots the master dataframe to create separate tables for each metric type,
    saving them as both .tex and .md files in the output directory.

    Args:
        df (pd.DataFrame): The master results dataframe containing all metrics.
        output_dir (str): Directory where tables will be saved.
    """
    # Identify all unique metrics (e.g., 'SSH_RMSE', 'Vel_Bias')
    # Suffix extraction: "Train SSH_RMSE" -> "SSH_RMSE"
    metric_types = set()
    splits = ["Train", "Val", "Test", "Extreme"]

    for col in df.columns:
        if col == "Model":
            continue
        for split in splits:
            if col.startswith(split + " "):
                metric_name = col[len(split) + 1 :]
                metric_types.add(metric_name)

    print(f"\nGenerating tables for {len(metric_types)} metrics in '{output_dir}'...")

    for metric in sorted(metric_types):
        # Create a sub-dataframe for this metric
        table_data = {"Model": df["Model"]}
        valid_splits = []

        for split in splits:
            col_name = f"{split} {metric}"
            if col_name in df.columns:
                table_data[split] = df[col_name]
                valid_splits.append(split)

        sub_df = pd.DataFrame(table_data)
        if sub_df.shape[1] <= 1:
            continue  # Skip if no data cols found

        # Sort by Validation Score if possible
        # Descending for NSE (higher is better), Ascending for Errors
        sort_split = "Val" if "Val" in valid_splits else valid_splits[-1]
        ascending = True
        if "NSE" in metric:
            ascending = False

        sub_df = sub_df.sort_values(sort_split, ascending=ascending)

        # Save files
        base_name = os.path.join(output_dir, metric)

        # Markdown
        with open(f"{base_name}.md", "w") as f:
            try:
                f.write(sub_df.to_markdown(index=False, floatfmt=".4f"))
            except ImportError:
                f.write(sub_df.to_string(index=False, float_format="%.4f"))

        # LaTeX
        caption = f"{metric.replace('_', ' ')} comparison across splits."
        with open(f"{base_name}.tex", "w") as f:
            f.write(
                sub_df.to_latex(
                    index=False,
                    float_format="%.4f",
                    caption=caption,
                    label=f"tab:{metric.lower()}",
                    escape=False,
                )
            )


if __name__ == "__main__":
    # To run doctests: python -m mswegnn.utils.adforce_evaluate_models
    import doctest

    doctest.testmod()

    parser = argparse.ArgumentParser(description="Evaluate mSWE-GNN models.")
    parser.add_argument(
        "-r", "--results_dir", required=True, help="Directory containing model runs"
    )
    parser.add_argument(
        "-d", "--data_dir", required=True, help="Directory containing raw data"
    )
    parser.add_argument(
        "-c", "--conf_dir", default="conf", help="Directory with split YAMLs"
    )
    parser.add_argument(
        "-e", "--extreme_dir", default=None, help="Directory for extreme test set"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="evaluation_results",
        help="Directory for output tables",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

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
        except torch.cuda.OutOfMemoryError:
            print(f"Skipping remaining evaluations due to persistent OOM error.")
            break

    if not all_results:
        print("No valid results found.")
        exit(0)

    df = pd.DataFrame(all_results)
    save_metric_tables(df, args.output)
    print(f"Done. All metric tables saved to {args.output}/")
