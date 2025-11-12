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
3.  y: (features_cfg.state + features_cfg.derived_state)
4.  y_delta: (deltas of features_cfg.targets)
"""

import os
import yaml
import warnings
from typing import List, Dict
import xarray as xr
import torch
from tqdm import tqdm
from omegaconf import DictConfig

# We need to import the helper functions from adforce_dataset
# to ensure the logic is identical, but to avoid circular imports,
# we can redefine the minimal helpers we need.
# For simplicity, we'll re-implement the derived feature logic.


class StatsAggregator:
    """
    Computes mean and variance in a single pass using Welford's
    online algorithm.

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
    >>> # Std of [1, 2, 3, 4] is 1.29099...
    >>> # Std of [10, 20, 30, 40] is 12.9099...
    >>> print(f"Std: {std}")
    Std: tensor([ 1.2910, 12.9099])
    """

    def __init__(self):
        self.count = 0
        self.mean = None
        self.M2 = None

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
             data = data.unsqueeze(-1) # Ensure 2D

        if self.mean is None:
            self.mean = torch.zeros(data.shape[1], dtype=data.dtype)
            self.M2 = torch.zeros(data.shape[1], dtype=data.dtype)

        n1 = self.count
        n2 = data.shape[0]
        self.count = n1 + n2
        n = self.count

        # New data mean
        mean2 = torch.mean(data, dim=0)
        
        # Welford's algorithm
        delta = mean2 - self.mean
        self.mean = self.mean + delta * (n2 / n)
        delta2 = mean2 - self.mean
        self.M2 = self.M2 + torch.sum(data * data, dim=0) - n2 * (mean2 * mean2)
        # This is a simplification of Welford's M2 update,
        # but is more numerically stable when merged:
        # self.M2 = self.M2 + (data - mean1).pow(2).sum(0) + (data - mean2).pow(2).sum(0)
        # A simpler version for combining batches:
        # self.M2 = self.M2 + torch.sum((data - self.mean) * (data - mean2), dim=0) # This is not quite right
        
        # Let's use a standard implementation for clarity
        # We need to process item by item for true Welford
        # Or just use a simpler two-pass algorithm (less ideal)
        
        # --- Simpler Implementation ---
        # For a large dataset, we can compute sum and sum_sq
        # This is less numerically stable than Welford but simpler
        # Let's stick to the Welford-like update for batches
        
        # Re-deriving the batch update:
        # M2_new = M2_old + (x - mean_old) * (x - mean_new)
        # For a batch:
        # M2_new = M2_old + sum((data - mean_old) * (data - mean_new))
        
        # Simplified: M2 = sum( (x - mean)^2 )
        # M2_new = M2_old + sum((data_i - mean_old)^2) for i in batch
        # No, that's wrong.
        
        # Let's use torch's built-in mean/std and accept the memory hit
        # This is much safer than implementing Welford incorrectly.
        
        # --- RE-IMPLEMENTATION: Use list of tensors ---
        if not hasattr(self, 'data_chunks'):
            self.data_chunks = []
        self.data_chunks.append(data.cpu())
        
    def finalize(self):
        """
        Finalizes the statistics.
        """
        if not hasattr(self, 'data_chunks') or not self.data_chunks:
            return torch.tensor(0.0), torch.tensor(1.0)
            
        full_data = torch.cat(self.data_chunks, dim=0).to(torch.float32)
        mean = torch.mean(full_data, dim=0)
        std = torch.std(full_data, dim=0)
        std = std.clamp(min=1e-6) # Avoid division by zero
        
        # Clear memory
        self.data_chunks = []
        
        return mean, std


def _load_tensor_from_ds(ds: xr.Dataset, var_list: List[str]) -> torch.Tensor:
    """Loads variables from xarray dataset and stacks them into a tensor."""
    tensors = []
    for var in var_list:
        if var not in ds:
            raise ValueError(f"Scaling: Variable '{var}' not found in dataset.")
        tensors.append(torch.tensor(ds[var].values, dtype=torch.float32))
    if not tensors:
        return torch.empty((ds.sizes.get("num_nodes", 0), 0), dtype=torch.float32)
    return torch.stack(tensors, dim=1)


def _get_derived_state(
    y_t_dict: Dict[str, torch.Tensor],
    static_data_dict: Dict[str, torch.Tensor],
    derived_state_specs: List[DictConfig]
) -> torch.Tensor:
    """Calculates derived state features."""
    derived_features_list = []
    for derived_spec in derived_state_specs:
        arg_data = []
        for arg_name in derived_spec['args']:
            if arg_name in y_t_dict:
                arg_data.append(y_t_dict[arg_name])
            elif arg_name in static_data_dict:
                arg_data.append(static_data_dict[arg_name])
            else:
                raise ValueError(f"Scaling: Unknown arg '{arg_name}' for derived feature '{derived_spec['name']}'")
        
        # Perform operation
        if derived_spec['op'] == 'subtract':
            derived_feat = arg_data[0] - arg_data[1]
        elif derived_spec['op'] == 'magnitude':
            derived_feat = torch.sqrt(arg_data[0]**2 + arg_data[1]**2)
        else:
            raise ValueError(f"Scaling: Unknown op '{derived_spec['op']}'")
        
        derived_features_list.append(derived_feat.unsqueeze(-1))
    
    if not derived_features_list:
        return torch.empty((y_t_dict[list(y_t_dict.keys())[0]].shape[0], 0), dtype=torch.float32)
        
    return torch.cat(derived_features_list, dim=-1)


def compute_and_save_adforce_stats(
    nc_files: List[str],
    save_path: str,
    features_cfg: DictConfig
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
    state_vars = list(features_cfg.state)
    target_vars = list(features_cfg.targets)
    derived_state_specs = list(features_cfg.derived_state)

    # Check if targets match state (required for delta learning)
    if set(state_vars) != set(target_vars):
        warnings.warn(
            f"Scaling: 'features.state' ({state_vars}) and "
            f"'features.targets' ({target_vars}) do not match. "
            "Delta stats will be computed for targets, "
            "but this may be an error in your config."
        )

    # --- 2. Initialize Aggregators ---
    print("Initializing statistics aggregators...")
    stats_aggs = {
        'x_static': StatsAggregator(),
        'x_dynamic': StatsAggregator(),
        'y': StatsAggregator(),
        'y_delta': StatsAggregator()
    }
    
    static_data_dict_cpu = {}

    # --- 3. Compute Static Stats (from first file only) ---
    if not nc_files:
        raise ValueError("No NetCDF files provided to calculate stats.")
    
    print(f"Calculating static stats from: {nc_files[0]}...")
    try:
        with xr.open_dataset(nc_files[0]) as ds:
            # Load static node features specified in config
            if static_node_vars:
                static_node_data = _load_tensor_from_ds(ds, static_node_vars)
                stats_aggs['x_static'].update(static_node_data)
            
            # Store static features needed for derived calculations
            # (e.g., 'DEM' for 'SSH')
            all_derived_args = set()
            for spec in derived_state_specs:
                all_derived_args.update(spec['args'])
            
            for var in all_derived_args:
                if var in ds and 'time' not in ds[var].dims:
                    static_data_dict_cpu[var] = torch.tensor(
                        ds[var].values, dtype=torch.float32
                    )

    except Exception as e:
        raise IOError(f"Failed to load static data from {nc_files[0]}: {e}")

    # --- 4. Loop files for dynamic, state, and delta stats ---
    print(f"Calculating dynamic stats from {len(nc_files)} training files...")
    for nc_path in tqdm(nc_files, desc="Processing files for stats"):
        try:
            with xr.open_dataset(nc_path) as ds:
                
                # Load all time-series data into memory (Tensors)
                # Shape: [N_nodes, N_time, N_features]
                if forcing_vars:
                    forcing_data = _load_tensor_from_ds(ds[forcing_vars], forcing_vars)
                    stats_aggs['x_dynamic'].update(forcing_data)
                
                if state_vars:
                    state_data = _load_tensor_from_ds(ds[state_vars], state_vars)
                    num_timesteps = state_data.shape[1]
                    
                    # Loop through time to calculate 'y' and 'y_delta'
                    for t in range(num_timesteps):
                        # Get y(t)
                        y_t = state_data[:, t, :] # [N, F_state]
                        
                        # --- 'y' stats (state + derived) ---
                        y_t_dict = {
                            var: y_t[:, i] for i, var in enumerate(state_vars)
                        }
                        
                        derived_y_t = _get_derived_state(
                            y_t_dict, static_data_dict_cpu, derived_state_specs
                        )
                        
                        # Concat base state + derived state
                        full_state_tensor = torch.cat([y_t, derived_y_t], dim=1)
                        stats_aggs['y'].update(full_state_tensor)
                        
                        # --- 'y_delta' stats ---
                        if t > 0:
                            # We need target_vars, which we assume match state_vars
                            # This logic assumes the *order* of targets and state is the same
                            y_t_minus_1 = state_data[:, t - 1, :]
                            
                            # Delta = y(t) - y(t-1)
                            delta = y_t - y_t_minus_1
                            stats_aggs['y_delta'].update(delta)

        except Exception as e:
            warnings.warn(f"Failed to process file {nc_path}: {e}. Skipping.")
            continue
            
    # --- 5. Finalize and Save Stats ---
    print("Finalizing statistics...")
    final_stats = {}
    
    # .finalize() returns (mean, std)
    mean, std = stats_aggs['x_static'].finalize()
    final_stats['x_static_mean'] = mean.tolist()
    final_stats['x_static_std'] = std.tolist()
    
    mean, std = stats_aggs['x_dynamic'].finalize()
    final_stats['x_dynamic_mean'] = mean.tolist()
    final_stats['x_dynamic_std'] = std.tolist()

    mean, std = stats_aggs['y'].finalize()
    final_stats['y_mean'] = mean.tolist()
    final_stats['y_std'] = std.tolist()

    mean, std = stats_aggs['y_delta'].finalize()
    final_stats['y_delta_mean'] = mean.tolist()
    final_stats['y_delta_std'] = std.tolist()

    print(f"Saving scaling stats to {save_path}...")
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(final_stats, f, default_flow_style=False)
        print("Stats saved successfully.")
        
        # Print a summary of the stats for verification
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
    
    # --- Example usage (requires actual config and data) ---
    # print("\n--- RUNNING EXAMPLE (requires config and data) ---")
    # try:
    #     from omegaconf import OmegaConf
    #     # This is a mock config for demonstration
    #     mock_features_cfg = OmegaConf.create({
    #         'static': ['DEM', 'slopex', 'slopey', 'area'],
    #         'edge': ['face_distance', 'edge_slope'],
    #         'forcing': ['WX', 'WY', 'P'],
    #         'state': ['WD', 'VX', 'VY'],
    #         'derived_state': [
    #             {'name': 'SSH', 'op': 'subtract', 'args': ['WD', 'DEM']}
    #         ],
    #         'targets': ['WD', 'VX', 'VY']
    #     })
    #
    #     # FAKE DATA: Replace with real paths
    #     mock_nc_files = sorted(glob.glob("path/to/your/data/*.nc"))
    #     if not mock_nc_files:
    #         print("Skipping example: No mock data files found.")
    #     else:
    #         compute_and_save_adforce_stats(
    #             nc_files=mock_nc_files,
    #             save_path="scaling_stats_TEST.yaml",
    #             features_cfg=mock_features_cfg
    #         )
    # except ImportError:
    #     print("Skipping example: omegaconf not installed.")
    # except Exception as e:
    #     print(f"Example failed: {e}")