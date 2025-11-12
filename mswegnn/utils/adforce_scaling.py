"""
Script to compute and save scaling statistics (mean, std)
for the Adforce pipeline.

--- REFACTOR ---
This script is now config-driven. It requires a 'features_cfg'
object (from config.yaml) to determine which features to
calculate statistics for.

It computes and saves stats for:
1.  x_static: (features_cfg.static)
2.  x_dynamic: (features_cfg.forcing)
3.  y: (features_cfg.targets + features_cfg.derived_state)
4.  y_delta: (deltas of features_cfg.targets)

This version uses a memory-efficient online algorithm (Welford's)
to calculate stats in a single pass and explicitly handles NaNs
by converting them to 0.0 before aggregation.
"""

import os
import yaml
import warnings
from typing import List, Dict
import xarray as xr
import torch
from tqdm import tqdm
from omegaconf import DictConfig


class StatsAggregator:
    """
    Computes mean and variance in a single pass using Welford's
    online algorithm, adapted for batches. This is memory-efficient.

    Uses float64 for intermediate calculations to ensure numerical stability.

    Doctest:
    >>> import torch
    >>> # Test with two batches
    >>> batch1 = torch.tensor([[1.0, 10.0], [2.0, 20.0]], dtype=torch.float32)
    >>> batch2 = torch.tensor([[3.0, 30.0], [4.0, 40.0]], dtype=torch.float32)
    >>>
    >>> aggregator = StatsAggregator()
    >>> aggregator.update(batch1)
    >>> aggregator.update(batch2)
    >>>
    >>> mean, std = aggregator.finalize()
    >>>
    >>> # Mean of [1, 2, 3, 4] is 2.5
    >>> # Mean of [10, 20, 30, 40] is 25.0
    >>> print(f"Mean: {mean}")
    Mean: tensor([ 2.5000, 25.0000])
    >>>
    >>> # Std of [1, 2, 3, 4] is 1.29099... (sample std, ddof=1)
    >>> # Std of [10, 20, 30, 40] is 12.9099... (sample std, ddof=1)
    >>> print(f"Std: {std}")
    Std: tensor([ 1.2910, 12.9099])
    """

    def __init__(self, dtype=torch.float64):
        """
        Initialize the aggregator.
        Args:
            dtype: The torch dtype to use for intermediate calculations.
                   float64 is recommended for numerical stability.
        """
        self.count = 0
        self.mean = None
        self.M2 = None
        self.dtype = dtype

    def update(self, data: torch.Tensor):
        """
        Update statistics with a new batch of data.
        Args:
            data (torch.Tensor): A tensor of shape [N, F],
                                 where N is batch size and F is feature dim.
        """
        if data.shape[0] == 0:
            return

        # Flatten data to [N, F] if it's [N, T, F] or similar
        if data.dim() > 2:
            data = data.reshape(-1, data.shape[-1])

        if data.dim() == 1:
            data = data.unsqueeze(-1)  # Ensure 2D

        # Move to high precision for stable calculation
        data = data.to(self.dtype)

        if self.mean is None:
            # Initialize on first batch
            self.mean = torch.zeros(data.shape[1], dtype=self.dtype)
            self.M2 = torch.zeros(data.shape[1], dtype=self.dtype)

        # Welford's algorithm for batches
        n1 = self.count
        n2 = data.shape[0]
        self.count = n1 + n2
        n = self.count

        # Mean of the new batch
        batch_mean = data.mean(dim=0)

        # Difference between old mean and new batch mean
        delta = batch_mean - self.mean

        # Update running mean
        self.mean += delta * (n2 / n)

        # Update M2 (sum of squared differences from the mean)
        batch_M2 = ((data - batch_mean) ** 2).sum(dim=0)
        self.M2 += batch_M2 + (delta**2) * n1 * n2 / n

    def finalize(self):
        """
        Finalizes the statistics, returning mean and sample standard deviation.

        Returns:
            (torch.Tensor, torch.Tensor): (mean, std) in float32
        """
        if self.count < 2:
            warnings.warn(
                f"Finalizing stats with count={self.count} (< 2 samples). Std will be 1.0."
            )
            mean = (
                self.mean
                if self.mean is not None
                else torch.tensor(0.0, dtype=self.dtype)
            )
            std = torch.tensor(1.0, dtype=self.dtype)
        else:
            mean = self.mean
            # Sample variance (ddof=1)
            variance = self.M2 / (self.count - 1)
            std = torch.sqrt(variance).clamp(min=1e-6)  # Avoid div by zero

        # Return in standard float32
        return mean.to(torch.float32), std.to(torch.float32)


def _load_tensor_from_ds(ds: xr.Dataset, var_list: List[str]) -> torch.Tensor:
    """
    Loads variables from xarray dataset and stacks them into a tensor.

    This function is now aware of (time, num_nodes) dimensions and
    will return a tensor in (num_nodes, time, features) or (num_nodes, features)
    format.
    """
    tensors = []

    # --- BUG FIX: Determine if loading time-series from VARS, not DATASET ---
    is_time_series = False
    if var_list:  # If there are variables to load
        # Check the dimensions of the *first* variable in the list
        is_time_series = "time" in ds[var_list[0]].dims
    # --- END BUG FIX ---

    for var in var_list:
        if var not in ds:
            raise ValueError(f"Scaling: Variable '{var}' not found in dataset.")
        # Load as tensor, ensuring float32
        tensors.append(torch.tensor(ds[var].values, dtype=torch.float32))

    if not tensors:
        num_nodes = ds.sizes.get("num_nodes", 0)
        if is_time_series:  # This check is now correct
            num_time = ds.sizes.get("time", 0)
            return torch.empty((num_nodes, num_time, 0), dtype=torch.float32)
        else:
            return torch.empty((num_nodes, 0), dtype=torch.float32)

    # Stack along the *last* dimension to create the feature channel
    # Input tensors are [T, N] (from NetCDF dump) or [N] for static
    stacked_tensor = torch.stack(tensors, dim=-1)  # Shape [T, N, F] or [N, F]

    # If it's time-series, permute to [N, T, F]
    if is_time_series:
        # Check if first dim is time, second is nodes
        if (
            stacked_tensor.dim() == 3
            and stacked_tensor.shape[0] == ds.sizes["time"]
            and stacked_tensor.shape[1] == ds.sizes["num_nodes"]
        ):
            stacked_tensor = stacked_tensor.permute(1, 0, 2)  # [T, N, F] -> [N, T, F]
        elif stacked_tensor.dim() == 2 and stacked_tensor.shape[0] == ds.sizes["time"]:
            # This handles the case where num_nodes=1
            stacked_tensor = stacked_tensor.permute(1, 0).unsqueeze(
                0
            )  # [T, F] -> [F, T] -> [1, T, F]
        elif (
            stacked_tensor.dim() == 3
            and stacked_tensor.shape[0] == ds.sizes["num_nodes"]
        ):
            # Already in [N, T, F] format
            pass
        else:
            # Raise an error for unhandled dimension order
            raise ValueError(
                f"Unexpected tensor shape {stacked_tensor.shape} "
                f"for time-series data with dims time={ds.sizes.get('time')}, "
                f"num_nodes={ds.sizes.get('num_nodes')}"
            )
    else:
        # --- BUG FIX: This is static data, shape is [N, F]. We're done. ---
        # It should not raise an error here.
        if (
            stacked_tensor.dim() != 2
            or stacked_tensor.shape[0] != ds.sizes["num_nodes"]
        ):
            raise ValueError(
                f"Unexpected tensor shape {stacked_tensor.shape} "
                f"for static data with num_nodes={ds.sizes.get('num_nodes')}"
            )
        pass  # Shape is [N, F], which is correct.

    return stacked_tensor


def _get_derived_state(
    y_t_dict: Dict[str, torch.Tensor],
    static_data_dict: Dict[str, torch.Tensor],
    derived_state_specs: List[DictConfig],
) -> torch.Tensor:
    """Calculates derived state features."""
    derived_features_list = []
    # Find a sample tensor to get the num_nodes dim
    num_nodes = 0
    if y_t_dict:
        # y_t_dict tensors are [N] or [N, T]
        num_nodes = y_t_dict[list(y_t_dict.keys())[0]].shape[0]
    elif static_data_dict:
        # static_data_dict tensors are [N]
        num_nodes = static_data_dict[list(static_data_dict.keys())[0]].shape[0]

    if num_nodes == 0:
        # Can't derive anything if there's no data
        return torch.empty((0, 0), dtype=torch.float32)

    for derived_spec in derived_state_specs:
        arg_data = []
        for arg_name in derived_spec["args"]:
            if arg_name in y_t_dict:
                arg_data.append(y_t_dict[arg_name])
            elif arg_name in static_data_dict:
                arg_data.append(static_data_dict[arg_name])
            else:
                raise ValueError(
                    f"Scaling: Unknown arg '{arg_name}' for derived feature '{derived_spec['name']}'"
                )

        # Perform operation
        if derived_spec["op"] == "subtract":
            derived_feat = arg_data[0] - arg_data[1]
        elif derived_spec["op"] == "magnitude":
            derived_feat = torch.sqrt(arg_data[0] ** 2 + arg_data[1] ** 2)
        elif derived_spec["op"] == "add":
            derived_feat = arg_data[0] + arg_data[1]
        else:
            raise ValueError(f"Scaling: Unknown op '{derived_spec['op']}'")

        derived_features_list.append(derived_feat.unsqueeze(-1))  # Add feature dim

    if not derived_features_list:
        # Check if we are time-series
        is_time_series = False
        if y_t_dict:
            is_time_series = y_t_dict[list(y_t_dict.keys())[0]].dim() > 1

        if is_time_series:
            num_time = y_t_dict[list(y_t_dict.keys())[0]].shape[1]
            return torch.empty((num_nodes, num_time, 0), dtype=torch.float32)
        else:
            return torch.empty((num_nodes, 0), dtype=torch.float32)

    return torch.cat(derived_features_list, dim=-1)


def compute_and_save_adforce_stats(
    nc_files: List[str], save_path: str, features_cfg: DictConfig
):
    """
    Computes and saves scaling statistics based on the feature config.

    Args:
        nc_files (List[str]): List of training NetCDF file paths.
        save_path (str): Path to save the 'scaling_stats.yaml'.
        features_cfg (DictConfig): The 'features' block from config.yaml.
    """

    # --- 1. Get feature lists from config ---
    static_node_vars = list(features_cfg.static)
    forcing_vars = list(features_cfg.forcing)
    state_vars = list(features_cfg.targets)
    target_vars = list(features_cfg.targets)
    derived_state_specs = list(features_cfg.derived_state)

    if set(state_vars) != set(target_vars):
        warnings.warn(
            f"Scaling: 'features.targets' ({state_vars}) and "
            f"'features.targets' ({target_vars}) do not match. "
            "Delta stats will be computed for targets, "
            "but this may be an error in your config."
        )

    # --- 2. Initialize Aggregators ---
    print("Initializing statistics aggregators...")
    stats_aggs = {
        "x_static": StatsAggregator(),
        "x_dynamic": StatsAggregator(),
        "y": StatsAggregator(),
        "y_delta": StatsAggregator(),
    }

    static_data_dict_cpu = {}

    # --- 3. Compute Static Stats (from first file only) ---
    if not nc_files:
        raise ValueError("No NetCDF files provided to calculate stats.")

    print(f"Calculating static stats from: {nc_files[0]}...")
    try:
        with xr.open_dataset(nc_files[0]) as ds:
            # static_node_data is [N, F_static]
            static_node_data = _load_tensor_from_ds(ds, static_node_vars)

            # --- NAN FIX ---
            static_node_data.nan_to_num_(nan=0.0)
            # --- END FIX ---

            stats_aggs["x_static"].update(static_node_data)

            # Store static features needed for derived calculations
            all_derived_args = set()
            for spec in derived_state_specs:
                all_derived_args.update(spec["args"])

            for var in all_derived_args:
                if var in ds and "time" not in ds[var].dims:
                    # Load as [N] tensor
                    tensor_data = torch.tensor(ds[var].values, dtype=torch.float32)
                    # --- NAN FIX ---
                    tensor_data.nan_to_num_(nan=0.0)
                    # --- END FIX ---
                    static_data_dict_cpu[var] = tensor_data

    except Exception as e:
        raise IOError(f"Failed to load static data from {nc_files[0]}: {e}")

    # --- 4. Loop files for dynamic, state, and delta stats ---
    print(f"Calculating dynamic stats from {len(nc_files)} training files...")
    for nc_path in tqdm(nc_files, desc="Processing files for stats"):
        try:
            # .load() pulls all data into memory for this *one* file
            with xr.open_dataset(nc_path).load() as ds:

                # a. Forcing stats
                if forcing_vars:
                    # forcing_data = [N, T, F_forcing]
                    forcing_data = _load_tensor_from_ds(ds, forcing_vars)
                    # --- NAN FIX ---
                    forcing_data.nan_to_num_(nan=0.0)
                    # --- END FIX ---
                    stats_aggs["x_dynamic"].update(forcing_data)

                # b. State and Delta stats
                if state_vars:
                    # state_data = [N, T, F_state]
                    state_data = _load_tensor_from_ds(ds, state_vars)
                    # --- NAN FIX ---
                    state_data.nan_to_num_(nan=0.0)
                    # --- END FIX ---

                    # target_data = [N, T, F_target]
                    target_data = _load_tensor_from_ds(ds, target_vars)
                    # --- NAN FIX ---
                    target_data.nan_to_num_(nan=0.0)
                    # --- END FIX ---

                    # Compute deltas for all steps at once: y(t) - y(t-1)
                    # deltas = [N, T-1, F_target]
                    deltas = target_data[:, 1:, :] - target_data[:, :-1, :]
                    # No nan_to_num needed, inputs are clean
                    stats_aggs["y_delta"].update(deltas)

                    # Compute derived features for all steps
                    # y_t_dict = { 'WD': [N, T], ... }
                    y_t_dict = {
                        var: state_data[:, :, i] for i, var in enumerate(state_vars)
                    }

                    # derived_y_t = [N, T, F_derived]
                    derived_y_t = _get_derived_state(
                        y_t_dict,
                        # Unsqueeze static data to broadcast along time dim [N] -> [N, 1]
                        {k: v.unsqueeze(1) for k, v in static_data_dict_cpu.items()},
                        derived_state_specs,
                    )

                    # full_state_tensor = [N, T, F_state + F_derived]
                    full_state_tensor = torch.cat([state_data, derived_y_t], dim=-1)
                    # No nan_to_num needed, inputs are clean
                    stats_aggs["y"].update(full_state_tensor)

        except Exception as e:
            warnings.warn(f"Failed to process file {nc_path}: {e}. Skipping.")
            continue

    # --- 5. Finalize and Save Stats ---
    print("Finalizing statistics...")
    final_stats = {}

    mean, std = stats_aggs["x_static"].finalize()
    final_stats["x_static_mean"] = mean.tolist()
    final_stats["x_static_std"] = std.tolist()

    mean, std = stats_aggs["x_dynamic"].finalize()
    final_stats["x_dynamic_mean"] = mean.tolist()
    final_stats["x_dynamic_std"] = std.tolist()

    mean, std = stats_aggs["y"].finalize()
    final_stats["y_mean"] = mean.tolist()
    final_stats["y_std"] = std.tolist()

    mean, std = stats_aggs["y_delta"].finalize()
    final_stats["y_delta_mean"] = mean.tolist()
    final_stats["y_delta_std"] = std.tolist()

    print(f"Saving scaling stats to {save_path}...")
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            yaml.dump(final_stats, f, default_flow_style=False)
        print("Stats saved successfully.")

        print("\n--- STATS SUMMARY (MEANS) ---")
        print(f"x_static_mean ({len(final_stats['x_static_mean'])} features):")
        print(f"  {final_stats['x_static_mean']}")
        print(f"x_dynamic_mean ({len(final_stats['x_dynamic_mean'])} features):")
        print(f"  {final_stats['x_dynamic_mean']}")
        print(f"y_mean ({len(final_stats['y_mean'])} features):")
        print(f"  {final_stats['y_mean']}")
        print(f"y_delta_mean ({len(final_stats['y_delta_mean'])} features):")
        print(f"  {final_stats['y_delta_mean']}")
        print("-----------------------------\n")

    except Exception as e:
        raise IOError(f"Failed to save stats file to {save_path}: {e}")


if __name__ == "__main__":
    """
    Run doctests for this module.

    From the command line (e.g., from the root sdat2/mswe-gnn/mSWE-GNN-sdat2/ dir):
    python -m mswegnn.utils.adforce_scaling
    """
    import doctest

    doctest.testmod(verbose=True)
    print("Doctests for mswegnn.utils.adforce_scaling complete.")
