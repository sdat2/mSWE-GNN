"""
Refactored dataset file for the Adforce project.

This file contains the "lazy" PyTorch Geometric Dataset class
(`AdforceLazyDataset`) used for efficient training, as well as standalone
helper functions for I/O (`_load_static_data_from_ds`, etc.) and a
function for memory-efficient full rollouts (`run_forcing_rollout`).

--- REFACTOR ---
This file is now config-driven. The `features_cfg` object (from config.yaml)
dictates which features to load, assemble, and derive.
1.  `_load_static_data_from_ds` loads node/edge features specified in config.
2.  `_get_forcing_slice` / `_get_target_slice` load vars specified in config.
3.  `AdforceLazyDataset` accepts `features_cfg` and uses it in `get()`:
    a. Assembles static, forcing, and state tensors.
    b. Calculates `derived_state` features (e.g., SSH).
    c. Concatenates all into the final 'x' tensor.
4.  `run_forcing_rollout` is updated to mirror this logic for inference.
---

The NetCDF file structure is assumed to be the one provided in the
'nc_dump' header, containing variables like 'WX', 'WY', 'P' (inputs)
and 'WD', 'VX', 'VY' (targets).

Example NetCDF structure:
netcdf swegnn {
dimensions:
        num_nodes = 58369 ;
        time = 17 ;
        two = 2 ;
        edge = 170492 ;
        xy = 2 ;
        nvertex = 3 ;
        nele = 58369 ;
        num_BC_edges = 102 ;
        num_ghost_nodes = 102 ;
variables:
        double x(num_nodes) ;
                x:_FillValue = NaN ;
                x:description = "Longitude of the dual graph nodes." ;
                x:units = "degrees_east" ;
        double y(num_nodes) ;
                y:_FillValue = NaN ;
                y:description = "Latitude of the dual graph nodes." ;
                y:units = "degrees_north" ;
        float DEM(num_nodes) ;
                DEM:_FillValue = NaNf ;
                DEM:description = "Bathymetry/DEM at face centers (m, positive up)" ;
                DEM:units = "m" ;
        float WD(time, num_nodes) ;
                WD:_FillValue = NaNf ;
                WD:description = "Water depth at face centers (time series)" ;
                WD:units = "m" ;
        float VX(time, num_nodes) ;
                VX:_FillValue = NaNf ;
                VX:description = "X-velocity at face centers (time series)" ;
                VX:units = "m/s" ;
        float VY(time, num_nodes) ;
                VY:_FillValue = NaNf ;
                VY:description = "Y-velocity at face centers (time series)" ;
                VY:units = "m/s" ;
        double WX(time, num_nodes) ;
                WX:_FillValue = NaN ;
                WX:description = "X-component of wind at face centers (time series)" ;
                WX:units = "m/s" ;
        double WY(time, num_nodes) ;
                WY:_FillValue = NaN ;
                WY:description = "Y-component of wind at face centers (time series)" ;
                WY:units = "m/s" ;
        double P(time, num_nodes) ;
                P:_FillValue = NaN ;
                P:description = "Atmospheric pressure at face centers (time series)" ;
                P:units = "m" ;
        float slopex(num_nodes) ;
                slopex:_FillValue = NaNf ;
                slopex:description = "Topographic slope in x-direction at face centers" ;
                slopex:units = "m/degree_east" ;
        float slopey(num_nodes) ;
                slopey:_FillValue = NaNf ;
                slopey:description = "Topographic slope in y-direction at face centers" ;
                slopey:units = "m/degree_north" ;
        float area(num_nodes) ;
                area:_FillValue = NaNf ;
                area:description = "Area of each mesh face (triangle)" ;
                area:units = "m^2" ;
        int edge_index(two, edge) ;
                edge_index:description = "Dual graph connectivity (face indices)" ;
                edge_index:units = "index" ;
        float face_distance(edge) ;
                face_distance:_FillValue = NaNf ;
                face_distance:description = "Distance between centers of connected faces" ;
                face_distance:units = "degrees" ;
        float face_relative_distance(edge, xy) ;
                face_relative_distance:_FillValue = NaNf ;
                face_relative_distance:description = "Vector (dx, dy) between centers of connected faces" ;
                face_relative_distance:units = "degrees" ;
        float edge_slope(edge) ;
                edge_slope:_FillValue = NaNf ;
                edge_slope:description = "Slope between connected faces based on DEM" ;
                edge_slope:units = "m/degree" ;
        int64 edge(edge) ;
        int64 num_nodes(num_nodes) ;
        int64 nvertex(nvertex) ;
        int64 time(time) ;
                time:units = "hours since 2005-08-25 00:16:40" ;
                time:calendar = "proleptic_gregorian" ;
        int64 nele(nele) ;
        int element(nele, nvertex) ;
                element:description = "Original mesh triangles, with each new node corresponding to the face of the old triangle mesh." ;
        string two(two) ;
        int64 num_BC_edges(num_BC_edges) ;
        int edge_index_BC(num_BC_edges, two) ;
        int face_BC(num_BC_edges) ;
        float ghost_face_x(num_BC_edges) ;
                ghost_face_x:_FillValue = NaNf ;
        float ghost_face_y(num_BC_edges) ;
                ghost_face_y:_FillValue = NaNf ;
        int64 num_ghost_nodes(num_ghost_nodes) ;
        float ghost_node_x(num_ghost_nodes) ;
                ghost_node_x:_FillValue = NaNf ;
        float ghost_node_y(num_ghost_nodes) ;
                ghost_node_y:_FillValue = NaNf ;
        int original_ghost_node_indices(num_ghost_nodes) ;
        int ghost_node_indices(num_ghost_nodes) ;
        int node_BC(num_BC_edges) ;
                node_BC:description = "Indices assigned to ghost cells (faces) appended after real faces" ;
        float edge_BC_length(num_BC_edges) ;
                edge_BC_length:_FillValue = NaNf ;

// global attributes:
                :description = "Dual graph formatted for mSWE-GNN input pipeline" ;
}
"""

import os
from typing import Dict, List
import warnings
import yaml
import numpy as np
import xarray as xr
import torch
from torch_geometric.data import Dataset, Data
from mswegnn.utils.adforce_scaling import StatsAggregator
from tqdm import tqdm
from omegaconf import DictConfig


# ----------------------------------------------------------------------------
#  HELPER FUNCTIONS (with Doctests)
# ----------------------------------------------------------------------------


def _load_static_data_from_ds(
    ds: xr.Dataset, static_node_vars: List[str], static_edge_vars: List[str]
) -> Dict[str, torch.Tensor]:
    """
    Loads all static mesh and BC data from an open xarray dataset
    based on the features specified in the config.

    This function reads all non-time-series data (e.g., mesh
    connectivity, topography, boundary info) and converts it
    to the required torch tensors.

    Args:
        ds (xarray.Dataset): An open xarray.Dataset handle.
        static_node_vars (List[str]): List of static node vars to load
            (e.g., ['DEM', 'slopex']).
        static_edge_vars (List[str]): List of static edge vars to load
            (e.g., ['face_distance', 'edge_slope']).

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing all static data.
        Includes:
        - 'edge_index': The graph connectivity.
        - 'static_edge_attr': Stacked tensor of edge features.
        - 'node_BC': Indices of boundary nodes.
        - 'edge_BC_length': Length of boundary edges.
        - 'node_type': Binary tensor (1=boundary, 0=interior).
        - ...and all tensors for keys in 'static_node_vars'.

    Doctest:
    >>> # Create a mock xarray.Dataset for testing
    >>> import xarray as xr
    >>> import numpy as np
    >>> import torch
    >>> mock_ds = xr.Dataset(
    ...     data_vars={
    ...         'edge_index': (('two', 'edge'), np.array([[0, 1], [1, 0]])),
    ...         'face_distance': ('edge', np.array([1.1, 1.2])),
    ...         'edge_slope': ('edge', np.array([-0.1, 0.1])),
    ...         'DEM': ('num_nodes', np.array([10.0, 11.0])),
    ...         'slopex': ('num_nodes', np.array([0.01, 0.02])),
    ...         'slopey': ('num_nodes', np.array([0.03, 0.04])),
    ...         'area': ('num_nodes', np.array([50.0, 51.0])),
    ...         'face_BC': ('num_BC_edges', np.array([1])), # Node 1 is boundary
    ...         'edge_BC_length': ('num_BC_edges', np.array([5.5])),
    ...     },
    ...     coords={
    ...         'num_nodes': np.arange(2),
    ...         'edge': np.arange(2),
    ...         'num_BC_edges': np.arange(1),
    ...         'two': np.arange(2),
    ...     }
    ... )
    >>> # --- REFACTORED TEST ---
    >>> # Define the features we want to load, just like from config
    >>> node_vars = ['DEM', 'slopex', 'slopey', 'area']
    >>> edge_vars = ['face_distance', 'edge_slope']
    >>> # Call the function with the feature lists
    >>> static_data = _load_static_data_from_ds(mock_ds, node_vars, edge_vars)
    >>> # Check the output dictionary
    >>> print(sorted(static_data.keys()))
    ['DEM', 'area', 'edge_BC_length', 'edge_index', 'node_BC', 'node_type', 'slopex', 'slopey', 'static_edge_attr']
    >>> print(static_data['node_BC'])
    tensor([1])
    >>> print(static_data['edge_BC_length'])
    tensor([5.5000])
    >>> # Check the 'node_type' tensor (auto-computed)
    >>> print(static_data['node_type'])
    tensor([0., 1.])
    >>> # Check the 'static_edge_attr' tensor (assembled from edge_vars)
    >>> print(static_data['static_edge_attr'])
    tensor([[ 1.1000, -0.1000],
            [ 1.2000,  0.1000]])
    >>> # Check one of the raw node features
    >>> print(static_data['DEM'])
    tensor([10., 11.])
    """
    data_dict = {}

    # --- Edge Index (Always required) ---
    data_dict["edge_index"] = torch.tensor(ds["edge_index"].values, dtype=torch.long)

    # --- Static Edge Features (from config) ---
    edge_attr_list = []
    for var_name in static_edge_vars:
        if var_name not in ds:
            raise ValueError(f"Static edge variable '{var_name}' not found in dataset.")
        edge_attr_list.append(
            torch.tensor(ds[var_name].values, dtype=torch.float)
        )
    if edge_attr_list:
        data_dict["static_edge_attr"] = torch.stack(edge_attr_list, dim=1)
    else:
        # Create a placeholder if no edge features are specified
        num_edges = ds.sizes.get("edge", 0)
        data_dict["static_edge_attr"] = torch.empty((num_edges, 0), dtype=torch.float)


    # --- Static Node Features (from config) ---
    for var_name in static_node_vars:
        if var_name not in ds:
            raise ValueError(f"Static node variable '{var_name}' not found in dataset.")
        data_dict[var_name] = torch.tensor(ds[var_name].values, dtype=torch.float)

    # --- Boundary Condition Info (Always required) ---
    num_real_nodes = ds.sizes["num_nodes"]
    node_type = torch.zeros(num_real_nodes, dtype=torch.float)

    boundary_face_indices = torch.tensor([], dtype=torch.long)  # Default
    if "face_BC" in ds:
        boundary_face_indices = torch.tensor(ds["face_BC"].values, dtype=torch.long)
        if boundary_face_indices.numel() > 0:
            valid_indices = boundary_face_indices[
                boundary_face_indices < num_real_nodes
            ]
            if len(valid_indices) < len(boundary_face_indices):
                warnings.warn(
                    "Some 'face_BC' indices are out of bounds for 'num_nodes'."
                )
            if valid_indices.numel() > 0:
                node_type[valid_indices] = 1.0  # Mark as boundary
    
    data_dict["node_type"] = node_type
    data_dict["node_BC"] = boundary_face_indices

    edge_bc_length = torch.tensor([], dtype=torch.float)  # Default
    if "edge_BC_length" in ds:
        edge_bc_length = torch.tensor(ds["edge_BC_length"].values, dtype=torch.float)
    data_dict["edge_BC_length"] = edge_bc_length

    return data_dict


def _get_forcing_slice(
    ds: xr.Dataset, t_start: int, num_steps: int, forcing_vars: List[str]
) -> torch.Tensor:
    """
    Loads a slice of *forcing* data and formats it for input.

    This function reads `num_steps` starting from `t_start` and
    reshapes the data from [vars, nodes, steps] to [nodes, vars * steps].

    Args:
        ds (xarray.Dataset): An open xarray.Dataset handle.
        t_start (int): The starting time index.
        num_steps (int): The number of time steps to load.
        forcing_vars (List[str]): List of forcing vars (e.g., ['WX', 'WY', 'P']).

    Returns:
        torch.Tensor: A tensor of shape [num_nodes, len(forcing_vars) * num_steps].

    Doctest:
    >>> import xarray as xr
    >>> import numpy as np
    >>> import torch
    >>> # Data: 2 vars (WX, WY), 2 nodes, 5 time steps
    >>> wx_data = np.arange(10).reshape(2, 5).astype(np.float32) # [[0,1,2,3,4], [5,6,7,8,9]]
    >>> wy_data = (np.arange(10) + 10).reshape(2, 5).astype(np.float32) # [[10,11,12,13,14], [15,16,17,18,19]]
    >>> mock_ds = xr.Dataset(
    ...     data_vars={
    ...         'WX': (('num_nodes', 'time'), wx_data),
    ...         'WY': (('num_nodes', 'time'), wy_data),
    ...         'P': (('num_nodes', 'time'), wx_data), # Mock P
    ...     },
    ...     coords={'num_nodes': np.arange(2), 'time': np.arange(5)}
    ... )
    >>> # Get a slice of 2 steps, starting at t=1, for WX and WY
    >>> forcing_slice = _get_forcing_slice(mock_ds, t_start=1, num_steps=2, forcing_vars=['WX', 'WY'])
    >>> print(forcing_slice.shape)
    torch.Size([2, 4])
    >>> # Check values.
    >>> # Node 0: [WX(t1), WY(t1), WX(t2), WY(t2)]
    >>> # WX(t1)=1, WY(t1)=11, WX(t2)=2, WY(t2)=12
    >>> # Node 1: [WX(t1), WY(t1), WX(t2), WY(t2)]
    >>> # WX(t1)=6, WY(t1)=16, WX(t2)=7, WY(t2)=17
    >>> expected_tensor = torch.tensor([
    ...     [1., 11., 2., 12.],  # Node 0
    ...     [6., 16., 7., 17.]   # Node 1
    ... ], dtype=torch.float32)
    >>> print(torch.all(torch.eq(forcing_slice, expected_tensor)))
    tensor(True)
    """
    if not forcing_vars:
        num_nodes = ds.sizes.get("num_nodes", 0)
        return torch.empty((num_nodes, 0), dtype=torch.float)

    # 1. Get the DataArray from xarray
    data_array = (
        ds[forcing_vars].isel(time=slice(t_start, t_start + num_steps)).to_array()
    )

    # 2. Define canonical order
    canonical_order = ("num_nodes", "time", "variable")

    # 3. Transpose to guarantee order
    try:
        transposed_da = data_array.transpose(*canonical_order, missing_dims="raise")
    except ValueError as e:
        dims = data_array.dims
        raise IOError(
            f"Failed to transpose forcing data dims. Got {dims}, expected {canonical_order}. Error: {e}"
        )

    # 4. Convert to tensor
    raw_slice = torch.tensor(transposed_da.values.copy(), dtype=torch.float)

    # 5. Reshape to [N, T*V]
    num_nodes = raw_slice.shape[0]
    formatted_slice = raw_slice.reshape(num_nodes, -1)
    return formatted_slice


def _get_target_slice(
    ds: xr.Dataset, t_start: int, num_steps: int, target_vars: List[str]
) -> torch.Tensor:
    """
    Loads a slice of *target* data (e.g., WD, VX, VY) and formats it.

    This function reads `num_steps` starting from `t_start` and
    reshapes the data. If num_steps=1, it squeezes the time dimension.

    Args:
        ds (xarray.Dataset): An open xarray.Dataset handle.
        t_start (int): The starting time index.
        num_steps (int): The number of time steps to load.
        target_vars (List[str]): List of target vars (e.g., ['WD', 'VX', 'VY']).

    Returns:
        torch.Tensor: A tensor of shape [num_nodes, len(target_vars) * num_steps].
                      If num_steps=1, shape is [num_nodes, len(target_vars)].

    Doctest:
    >>> import xarray as xr
    >>> import numpy as np
    >>> import torch
    >>> # Data: 2 vars (WD, VX), 2 nodes, 5 time steps
    >>> wd_data = np.arange(10).reshape(2, 5).astype(np.float32) # [[0,1,2,3,4], [5,6,7,8,9]]
    >>> vx_data = (np.arange(10) + 10).reshape(2, 5).astype(np.float32) # [[10,11,12,13,14], [15,16,17,18,19]]
    >>> mock_ds = xr.Dataset(
    ...     data_vars={
    ...         'WD': (('num_nodes', 'time'), wd_data),
    ...         'VX': (('num_nodes', 'time'), vx_data),
    ...         'VY': (('num_nodes', 'time'), wd_data), # Mock VY
    ...     },
    ...     coords={'num_nodes': np.arange(2), 'time': np.arange(5)}
    ... )
    >>> # Get a single step (num_steps=1) at t=3
    >>> target_slice = _get_target_slice(mock_ds, t_start=3, num_steps=1, target_vars=['WD', 'VX'])
    >>> print(target_slice.shape)
    torch.Size([2, 2])
    >>> # Check values
    >>> # Node 0: [WD(t3), VX(t3)]
    >>> # WD(t3)=3, VX(t3)=13
    >>> # Node 1: [WD(t3), VX(t3)]
    >>> # WD(t3)=8, VX(t3)=18
    >>> expected_tensor = torch.tensor([
    ...     [3., 13.],  # Node 0
    ...     [8., 18.]   # Node 1
    ... ], dtype=torch.float32)
    >>> print(torch.all(torch.eq(target_slice, expected_tensor)))
    tensor(True)
    """
    if not target_vars:
        num_nodes = ds.sizes.get("num_nodes", 0)
        return torch.empty((num_nodes, 0), dtype=torch.float)

    # 1. Get the DataArray from xarray
    data_array = (
        ds[target_vars].isel(time=slice(t_start, t_start + num_steps)).to_array()
    )

    # 2. Define canonical order
    canonical_order = ("num_nodes", "time", "variable")

    # 3. Transpose to guarantee order
    try:
        transposed_da = data_array.transpose(*canonical_order, missing_dims="raise")
    except ValueError as e:
        dims = data_array.dims
        raise IOError(
            f"Failed to transpose target data dims. Got {dims}, expected {canonical_order}. Error: {e}"
        )

    # 4. Convert to tensor
    raw_slice = torch.tensor(transposed_da.values.copy(), dtype=torch.float)

    # 5. Reshape to [N, T*V] and .squeeze() if T=1
    num_nodes = raw_slice.shape[0]
    formatted_slice = raw_slice.reshape(num_nodes, -1).squeeze()
    return formatted_slice


# ----------------------------------------------------------------------------
#  TRAINING DATA LOADER CLASS
# ----------------------------------------------------------------------------


class AdforceLazyDataset(Dataset):
    """
    A "lazy-loading" PyG Dataset for multiple NetCDF simulations,
    driven by a feature configuration object.

    --- MODIFIED FOR DELTA LEARNING & CONFIG-DRIVEN FEATURES ---
    This implementation trains the model to predict the *scaled increment*.
    It assembles the input tensor 'x' based on the `features_cfg`:
    x = [static_features, forcing_features, state_features]
    
    - static_features: (cfg.features.static + 'node_type')
    - forcing_features: (cfg.features.forcing * previous_t)
    - state_features: (cfg.features.state + cfg.features.derived_state)

    It predicts the scaled delta of `cfg.features.targets`.
    """

    def __init__(
        self,
        root,
        nc_files,
        previous_t,
        features_cfg: DictConfig, # <-- REFACTOR: Added
        scaling_stats_path: str = None,
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            root (str): Root directory to store processed index map.
            nc_files (list[str]): The list of PRE-PROCESSED .nc files.
            previous_t (int): Number of input time steps.
            features_cfg (DictConfig): The 'features' block from config.yaml.
            scaling_stats_path (str, optional): Path to 'scaling_stats.yaml'.
        """
        self.nc_files = sorted(nc_files)
        if not self.nc_files:
            raise ValueError("No NetCDF files provided.")

        self.previous_t = previous_t
        self.features_cfg = features_cfg
        self.rollout_steps = 1  # Hard-coded for 1-step-ahead training

        self.total_nodes = None
        self.index_map = []
        # --- REFACTOR: static_data is now a dictionary ---
        self.static_data = {}  # Will hold the SINGLE copy of static data

        # This super() call will trigger .process() if needed
        super().__init__(root, transform, pre_transform)

        # --- UPDATED: Load scaling stats from YAML ---
        self.apply_scaling = False
        if scaling_stats_path and os.path.exists(scaling_stats_path):
            print(f"Loading scaling stats from: {scaling_stats_path}")
            try:
                with open(scaling_stats_path, "r") as f:
                    scaling_stats = yaml.safe_load(f)
                
                # --- MODIFICATION: All tensors are created and left on CPU ---
                self.x_static_mean = torch.tensor(
                    scaling_stats["x_static_mean"], dtype=torch.float32
                )
                self.x_static_std = (
                    torch.tensor(scaling_stats["x_static_std"], dtype=torch.float32)
                    .clamp(min=1e-6)
                )
                x_dyn_mean = torch.tensor(
                    scaling_stats["x_dynamic_mean"], dtype=torch.float32
                )
                x_dyn_std = torch.tensor(
                    scaling_stats["x_dynamic_std"], dtype=torch.float32
                ).clamp(min=1e-6)
                self.y_mean = torch.tensor(
                    scaling_stats["y_mean"], dtype=torch.float32
                )
                self.y_std = (
                    torch.tensor(scaling_stats["y_std"], dtype=torch.float32)
                    .clamp(min=1e-6)
                )
                self.y_delta_mean = torch.tensor(
                    scaling_stats["y_delta_mean"], dtype=torch.float32
                )
                self.y_delta_std = (
                    torch.tensor(scaling_stats["y_delta_std"], dtype=torch.float32)
                    .clamp(min=1e-6)
                )
                
                # --- REFACTOR: Calculate expected feature counts from config ---
                num_static_cfg = len(self.features_cfg.static)
                num_forcing_cfg = len(self.features_cfg.forcing)
                num_state_cfg = len(self.features_cfg.state)
                num_derived_cfg = len(self.features_cfg.derived_state)
                num_targets_cfg = len(self.features_cfg.targets)

                self.x_dyn_mean_broadcast = x_dyn_mean.repeat(self.previous_t)
                self.x_dyn_std_broadcast = x_dyn_std.repeat(self.previous_t)
                
                # --- REFACTOR: Dynamic sanity checks based on config ---
                assert (
                    self.x_static_mean.shape[0] == num_static_cfg
                ), f"Scaling stats 'x_static_mean' has {self.x_static_mean.shape[0]} features, but config 'features.static' has {num_static_cfg}."
                
                assert (
                    x_dyn_mean.shape[0] == num_forcing_cfg
                ), f"Scaling stats 'x_dynamic_mean' has {x_dyn_mean.shape[0]} features, but config 'features.forcing' has {num_forcing_cfg}."

                assert (
                    self.y_mean.shape[0] == num_state_cfg + num_derived_cfg
                ), f"Scaling stats 'y_mean' has {self.y_mean.shape[0]} features, but config 'features.state' ({num_state_cfg}) + 'features.derived_state' ({num_derived_cfg}) has {num_state_cfg + num_derived_cfg}."
                
                assert (
                    self.y_delta_mean.shape[0] == num_targets_cfg
                ), f"Scaling stats 'y_delta_mean' has {self.y_delta_mean.shape[0]} features, but config 'features.targets' has {num_targets_cfg}."
                # --- END REFACTOR ---

                self.apply_scaling = True
                print("Scaling stats loaded, tensors created (on CPU), and shapes validated against config.")

            except (KeyError, TypeError, ValueError, FileNotFoundError, AssertionError) as e:
                print(
                    f"ERROR: Failed to load, parse, or validate {scaling_stats_path}: {e}. "
                    f"Ensure stats file matches feature config. Running unscaled."
                )
                self.apply_scaling = False
        else:
            print(
                f"WARNING: Scaling stats file not found at '{scaling_stats_path}'. Model will run on raw, unscaled data."
            )
        # --- END UPDATED BLOCK ---

        # --- Load the index map (as before) ---
        try:
            with xr.open_dataset(self.processed_paths[0]) as ds:
                self.total_nodes = ds.attrs["total_nodes"]
                # (omitted window mismatch check for brevity)
                file_paths = ds["file_paths"].values
                time_indices = ds["time_indices"].values
                self.index_map = list(zip(file_paths, time_indices))
        except FileNotFoundError:
             raise RuntimeError(
                f"Processed file not found at {self.processed_paths[0]}. Please check 'root' or re-run processing."
            )
        except Exception as e:
            raise IOError(f"Failed to load processed index file: {e}")

        # --- REFACTOR: Load static data ONCE using config ---
        print(f"Loading single static dataset from: {self.nc_files[0]}...")
        try:
            with xr.open_dataset(self.nc_files[0]) as ds:
                if "num_nodes" not in ds.sizes:
                    raise IOError(f"File {self.nc_files[0]} is missing 'num_nodes' dimension.")
                
                # --- REFACTOR: Pass feature lists to helper ---
                self.static_data = _load_static_data_from_ds(
                    ds,
                    self.features_cfg.static,
                    self.features_cfg.edge
                )
                
            print(f"Static data loaded and cached on device: cpu")
        except Exception as e:
            raise IOError(f"Failed to load static data from {self.nc_files[0]}: {e}")
        
        # --- Sanity check ---
        # +1 for node_type which is auto-added
        num_static_node_features_loaded = len(self.features_cfg.static) + 1
        num_static_nodes = self.static_data["node_type"].shape[0]
        if self.total_nodes != num_static_nodes:
            warnings.warn(
                f"Node count mismatch! Processed index reports {self.total_nodes} nodes, "
                f"but static data from {self.nc_files[0]} has {num_static_nodes} nodes. "
            )

    @property
    def processed_file_names(self):
        return [f"index_map_p{self.previous_t}_r{self.rollout_steps}.nc"]

    def process(self):
        """
        Runs ONCE. Scans all files, builds the index map,
        and verifies mesh consistency and variable presence
        based on the config.
        """
        print(
            f"Building index map for {len(self.nc_files)} files (p_t={self.previous_t}, r_s={self.rollout_steps})..."
        )

        # --- REFACTOR: Define required vars from config ---
        required_static_node_vars = list(self.features_cfg.static)
        required_static_edge_vars = list(self.features_cfg.edge)
        required_forcing_vars = list(self.features_cfg.forcing)
        required_target_vars = list(self.features_cfg.targets)

        # Base vars that are always needed for structure
        required_base_vars = [
            "edge_index",
            "face_BC",
            "edge_BC_length",
        ]
        
        all_required_vars = set(
            required_static_node_vars +
            required_static_edge_vars +
            required_forcing_vars +
            required_target_vars +
            required_base_vars
        )
        # --- END REFACTOR ---

        reference_edge_index = None
        total_nodes = None
        first_valid_file_found = False
        index_map = []
        valid_datasets = 0

        for nc_path in tqdm(self.nc_files, desc="Processing files"):
            try:
                with xr.open_dataset(nc_path) as ds:

                    # Check Variable Presence
                    available_vars = set(ds.data_vars.keys())
                    if not all_required_vars.issubset(available_vars):
                        missing = all_required_vars - available_vars
                        warnings.warn(
                            f"File {nc_path} is missing variables: {missing}. Skipping file."
                        )
                        continue
                    
                    if "num_nodes" not in ds.sizes:
                        warnings.warn(
                            f"File {nc_path} is missing 'num_nodes' dimension. Skipping file."
                        )
                        continue

                    # Check Mesh Consistency
                    current_edge_index = torch.tensor(
                        ds["edge_index"].values, dtype=torch.long
                    )
                    if not first_valid_file_found:
                        reference_edge_index = current_edge_index
                        total_nodes = ds.sizes["num_nodes"]
                        first_valid_file_found = True
                    elif not torch.equal(reference_edge_index, current_edge_index):
                        warnings.warn(f"Mesh mismatch in {nc_path}! Skipping file.")
                        continue
                    
                    if ds.sizes["num_nodes"] != total_nodes:
                        warnings.warn(
                            f"Node count mismatch in {nc_path}! "
                            f"Expected {total_nodes} but found {ds.sizes['num_nodes']}. Skipping file."
                        )
                        continue

                    # Add to index map
                    num_timesteps = ds.sizes["time"]
                    valid_steps = (
                        num_timesteps - self.previous_t - self.rollout_steps + 1
                    )
                    for t in range(valid_steps):
                        index_map.append((nc_path, t))
                valid_datasets += 1

            except Exception as e:
                warnings.warn(
                    f"Failed to open or process {nc_path}: {e}. Skipping file."
                )
                continue

        if not index_map:
            raise IOError("No valid time steps found across all NetCDF files.")
        if total_nodes is None:
            raise IOError("No valid files were found to establish mesh properties.")

        print(
            f"Index map built. Total samples: {len(index_map)},\n Total nodes per sample: {total_nodes},\n Valid files: {valid_datasets}/{len(self.nc_files)}, {valid_datasets /len(self.nc_files)*100:.1f}%"
        )

        # Save the map using xarray/NetCDF
        file_paths = np.array([item[0] for item in index_map], dtype="object")
        time_indices = np.array([item[1] for item in index_map], dtype=np.int32)
        sample_dim = "sample_idx"
        sample_coords = np.arange(len(index_map))
        ds_index = xr.Dataset(
            data_vars={
                "file_paths": (sample_dim, file_paths),
                "time_indices": (sample_dim, time_indices),
            },
            coords={sample_dim: sample_coords},
        )
        ds_index.attrs["total_nodes"] = total_nodes
        ds_index.attrs["previous_t"] = self.previous_t
        ds_index.attrs["rollout_steps"] = self.rollout_steps

        try:
            ds_index.to_netcdf(self.processed_paths[0], mode="w")
        except Exception as e:
            raise IOError(f"Failed to write processed index file: {e}")
        finally:
            ds_index.close()

    def len(self):
        return len(self.index_map)

    def get(self, idx: int) -> Data:
        """
        THE "LAZY" PART.
        Loads a single sample, assembles features based on config,
        and applies scaling.

        --- REFACTORED FOR CONFIG-DRIVEN FEATURES ---
        Model Input 'x': (static_scaled, forcing_scaled, state_scaled)
        Model Target 'y': (delta_scaled)
        """
        nc_path, t_start = self.index_map[idx]

        try:
            # 1. Get static data (from CPU cache, it's a dict)
            static_data_dict = self.static_data

            # 2. Open file, read dynamic tensors (to CPU), close file
            with xr.open_dataset(nc_path, cache=False) as ds:
                # Forcing: [N, V_forcing * p_t]
                dyn_forcing_features_t = _get_forcing_slice(
                    ds, t_start, self.previous_t, self.features_cfg.forcing
                )
                
                # y(t+1) Target: [N, V_target]
                t_target_step = t_start + self.previous_t
                y_tplus1_tensor = _get_target_slice(
                    ds, t_target_step, self.rollout_steps, self.features_cfg.targets
                )
                
                # y(t) State: Load as dict for assembly and derived features
                t_last_input_step = t_start + self.previous_t - 1
                y_t_dict = {}
                for var in self.features_cfg.state:
                    y_t_dict[var] = torch.tensor(
                        ds[var].isel(time=t_last_input_step).values, dtype=torch.float
                    )

            # --- 3. Assemble Feature Vectors (all on CPU) ---

            # a. Static features: [N, V_static + 1]
            static_features_list = [
                static_data_dict[k] for k in self.features_cfg.static
            ]
            static_features_list.append(static_data_dict['node_type'])
            static_input_tensor = torch.stack(static_features_list, dim=1)
            static_input_tensor = torch.nan_to_num(static_input_tensor, nan=0.0)

            # b. State features: [N, V_state]
            y_t_base_list = [y_t_dict[k] for k in self.features_cfg.state]
            y_t_tensor = torch.stack(y_t_base_list, dim=1)
            y_t_tensor = torch.nan_to_num(y_t_tensor, nan=0.0)
            
            # c. Derived state features: [N, V_derived]
            derived_state_features_list = []
            static_dict_for_derived = {k: static_data_dict[k] for k in self.features_cfg.static}

            for derived_spec in self.features_cfg.derived_state:
                arg_data = []
                for arg_name in derived_spec['args']:
                    if arg_name in y_t_dict:
                        arg_data.append(y_t_dict[arg_name])
                    elif arg_name in static_dict_for_derived:
                        arg_data.append(static_dict_for_derived[arg_name])
                    else:
                        raise ValueError(f"Unknown arg '{arg_name}' for derived feature '{derived_spec['name']}'")
                
                # Perform operation
                if derived_spec['op'] == 'subtract':
                    derived_feat = arg_data[0] - arg_data[1]
                elif derived_spec['op'] == 'magnitude':
                    derived_feat = torch.sqrt(arg_data[0]**2 + arg_data[1]**2)
                # ... add more ops ('add', 'multiply', etc.) here ...
                else:
                    raise ValueError(f"Unknown op '{derived_spec['op']}' for derived feature")
                
                derived_state_features_list.append(derived_feat.unsqueeze(1))
            
            # d. Combine state + derived state
            if derived_state_features_list:
                full_state_tensor = torch.cat([y_t_tensor] + derived_state_features_list, dim=1)
            else:
                full_state_tensor = y_t_tensor
            
            # --- 4. Handle NaNs and Deltas ---
            dyn_forcing_features_t = torch.nan_to_num(dyn_forcing_features_t, nan=0.0)
            y_tplus1_tensor = torch.nan_to_num(y_tplus1_tensor, nan=0.0)
            y_unscaled_tplus1 = y_tplus1_tensor.clone()
            
            # Delta is based on *base state* (targets), not derived features
            delta_raw = y_tplus1_tensor - y_t_tensor

            # --- 5. APPLY SCALING (all on CPU) ---
            static_scaled = static_input_tensor
            forcing_scaled = dyn_forcing_features_t
            state_scaled = full_state_tensor
            y_delta_scaled = delta_raw

            if self.apply_scaling:
                # Scale static (all except 'node_type' at the end)
                num_static_cfg = len(self.features_cfg.static)
                static_scaled[:, :num_static_cfg] = (
                    static_input_tensor[:, :num_static_cfg] - self.x_static_mean
                ) / self.x_static_std
                
                # Scale forcing
                forcing_scaled = (
                    dyn_forcing_features_t - self.x_dyn_mean_broadcast
                ) / self.x_dyn_std_broadcast
                
                # Scale full state (base + derived)
                state_scaled = (full_state_tensor - self.y_mean) / self.y_std
                
                # Scale delta
                y_delta_scaled = (delta_raw - self.y_delta_mean) / self.y_delta_std

            # 6. Combine ALL INPUTS (on CPU)
            x_t = torch.cat(
                [static_scaled, forcing_scaled, state_scaled], dim=1
            )

            # 7. Construct Data object (all tensors on CPU)
            return Data(
                x=x_t,
                edge_index=static_data_dict["edge_index"],
                edge_attr=static_data_dict["static_edge_attr"],
                y=y_delta_scaled,
                y_unscaled=y_unscaled_tplus1,
                node_BC=static_data_dict["node_BC"],
                edge_BC_length=static_data_dict["edge_BC_length"],
            )

        except Exception as e:
            raise IOError(
                f"Error loading sample {idx} (file: {nc_path}, time_idx: {t_start}): {e}"
            )


# ----------------------------------------------------------------------------
#  TESTING / ROLLOUT FUNCTION
# ----------------------------------------------------------------------------


def run_forcing_rollout(
    model: torch.nn.Module,
    nc_path: str,
    previous_t: int,
    features_cfg: DictConfig, # <-- REFACTOR: Added
    scaling_stats: dict,      # <-- REFACTOR: Now required
) -> torch.Tensor:
    """
    Runs a full, memory-efficient, forcing-driven rollout for one simulation,
    using the specified feature configuration.

    --- MODIFIED FOR DELTA LEARNING & CONFIG-DRIVEN FEATURES ---
    This function now requires the `scaling_stats` dictionary and
    `features_cfg` to:
    1.  Scale all inputs (`static`, `forcing`, `y_t`, `derived`)
        before feeding them to the model.
    2.  De-scale the model's output (which is `scaled_delta`) to get `raw_delta`.
    3.  Add the `raw_delta` to the unscaled `y_t` to get `y_tplus1`.
    4.  Propagate the new `y_tplus1` (as `y_t`) for the next step.
    """

    predictions = []
    device = next(model.parameters()).device
    model.eval()

    # --- NEW: Load scaling tensors from dictionary ---
    if scaling_stats is None:
        raise ValueError("run_forcing_rollout now requires 'scaling_stats' dictionary.")

    try:
        # --- MODIFICATION: All stats tensors are moved to the model's device ---
        y_mean = torch.tensor(scaling_stats["y_mean"], dtype=torch.float32).to(device)
        y_std = (
            torch.tensor(scaling_stats["y_std"], dtype=torch.float32)
            .to(device)
            .clamp(min=1e-6)
        )
        y_delta_mean = torch.tensor(
            scaling_stats["y_delta_mean"], dtype=torch.float32
        ).to(device)
        y_delta_std = (
            torch.tensor(scaling_stats["y_delta_std"], dtype=torch.float32)
            .to(device)
            .clamp(min=1e-6)
        )
        x_static_mean = torch.tensor(
            scaling_stats["x_static_mean"], dtype=torch.float32
        ).to(device)
        x_static_std = (
            torch.tensor(scaling_stats["x_static_std"], dtype=torch.float32)
            .to(device)
            .clamp(min=1e-6)
        )
        
        x_dyn_mean_cpu = torch.tensor(scaling_stats["x_dynamic_mean"], dtype=torch.float32)
        x_dyn_std_cpu = torch.tensor(
            scaling_stats["x_dynamic_std"], dtype=torch.float32
        ).clamp(min=1e-6)
        
        x_dyn_mean_broadcast = x_dyn_mean_cpu.repeat(previous_t).to(device)
        x_dyn_std_broadcast = x_dyn_std_cpu.repeat(previous_t).to(device)
        x_dyn_mean_single = x_dyn_mean_cpu.to(device)
        x_dyn_std_single = x_dyn_std_cpu.to(device)

    except (KeyError, TypeError) as e:
        raise ValueError(f"Scaling stats dict is missing keys or invalid: {e}")
    # --- END NEW ---

    with xr.open_dataset(nc_path) as ds:
        # 1. Load all static data (once to CPU)
        static_data_dict_cpu = _load_static_data_from_ds(
            ds, features_cfg.static, features_cfg.edge
        )
        #    Move (once) to the model's device
        static_data_gpu = {k: v.to(device) for k, v in static_data_dict_cpu.items()}

        # --- NEW: Assemble and Scale static features (on device) ---
        static_features_list = [
            static_data_gpu[k] for k in features_cfg.static
        ]
        static_features_list.append(static_data_gpu['node_type'])
        static_input_tensor = torch.stack(static_features_list, dim=1)
        static_input_tensor = torch.nan_to_num(static_input_tensor, nan=0.0)
        
        static_features_scaled = static_input_tensor.clone()
        num_static_cfg = len(features_cfg.static)
        static_features_scaled[:, :num_static_cfg] = (
            static_features_scaled[:, :num_static_cfg] - x_static_mean
        ) / x_static_std
        # --- END NEW ---

        num_timesteps = ds.sizes["time"]
        if num_timesteps <= previous_t:
            warnings.warn(f"File {nc_path} has {num_timesteps} steps... Skipping rollout.")
            return torch.tensor([])

        # 2. Get initial forcing history (load to CPU, move to device)
        current_forcing_history_raw = _get_forcing_slice(
            ds, 0, previous_t, features_cfg.forcing
        ).to(device)
        current_forcing_history_raw = torch.nan_to_num(
            current_forcing_history_raw, nan=0.0
        )

        # --- NEW: Get initial state y(t) (load to CPU, move to device) ---
        t_last_input_step = previous_t - 1
        current_y_t_raw = _get_target_slice(
            ds, t_last_input_step, 1, features_cfg.state
        ).to(device)
        current_y_t_raw = torch.nan_to_num(current_y_t_raw, nan=0.0)

        # --- NEW: Scale initial forcing ---
        current_forcing_history_scaled = (
            current_forcing_history_raw - x_dyn_mean_broadcast
        ) / x_dyn_std_broadcast

        # 3. Loop sequentially through the *rest* of the timesteps
        for t in range(previous_t, num_timesteps):

            # --- NEW: Calculate Derived Features for this step (on device) ---
            y_t_dict_gpu = {
                var: current_y_t_raw[:, i] 
                for i, var in enumerate(features_cfg.state)
            }
            static_dict_gpu_for_derived = {
                k: static_data_gpu[k] for k in features_cfg.static
            }

            derived_state_features_list = []
            for derived_spec in features_cfg.derived_state:
                arg_data = []
                for arg_name in derived_spec['args']:
                    if arg_name in y_t_dict_gpu:
                        arg_data.append(y_t_dict_gpu[arg_name])
                    elif arg_name in static_dict_gpu_for_derived:
                        arg_data.append(static_dict_gpu_for_derived[arg_name])
                    else:
                        raise ValueError(f"Rollout: Unknown arg '{arg_name}' for derived feature")
                
                if derived_spec['op'] == 'subtract':
                    derived_feat = arg_data[0] - arg_data[1]
                elif derived_spec['op'] == 'magnitude':
                    derived_feat = torch.sqrt(arg_data[0]**2 + arg_data[1]**2)
                else:
                    raise ValueError(f"Rollout: Unknown op '{derived_spec['op']}'")
                
                derived_state_features_list.append(derived_feat.unsqueeze(1))
            
            if derived_state_features_list:
                full_state_tensor_raw = torch.cat(
                    [current_y_t_raw] + derived_state_features_list, dim=1
                )
            else:
                full_state_tensor_raw = current_y_t_raw
            # --- END DERIVED ---
            
            # --- NEW: Scale current state (base + derived) ---
            current_y_t_scaled = (full_state_tensor_raw - y_mean) / y_std

            # --- NEW: Combine scaled inputs (all on device) ---
            x_t_scaled = torch.cat(
                [
                    static_features_scaled,
                    current_forcing_history_scaled,
                    current_y_t_scaled,
                ],
                dim=1,
            )
            x_t_scaled = torch.nan_to_num(x_t_scaled, nan=0.0)

            batch = Data(
                x=x_t_scaled,
                edge_index=static_data_gpu["edge_index"],
                edge_attr=static_data_gpu["static_edge_attr"],
                node_BC=static_data_gpu["node_BC"],
                edge_BC_length=static_data_gpu["edge_BC_length"],
            )

            with torch.no_grad():
                pred_scaled_delta = model(batch)  # shape [N, V_target]

            # --- NEW: De-scale the prediction (on device) ---
            pred_raw_delta = (pred_scaled_delta * y_delta_std) + y_delta_mean

            # --- NEW: Compute next state (unscaled, on device) ---
            # y(t+1) = y(t) + delta
            # Note: delta is added to the *base state*, not the full state
            next_y_t_raw = current_y_t_raw + pred_raw_delta

            predictions.append(next_y_t_raw.cpu())  # Store unscaled pred on CPU

            # --- Update state and forcing for the *next* loop iteration ---
            current_y_t_raw = next_y_t_raw # This is [N, V_state]

            # 2. Update forcing history
            if t + 1 < num_timesteps:
                # Get *next* forcing slice (load to CPU, move to device)
                next_forcing_slice_raw = _get_forcing_slice(
                    ds, t, 1, features_cfg.forcing
                ).to(device)
                next_forcing_slice_raw = torch.nan_to_num(
                    next_forcing_slice_raw, nan=0.0
                )

                # Scale it (on device)
                next_forcing_slice_scaled = (
                    next_forcing_slice_raw - x_dyn_mean_single
                ) / x_dyn_std_single

                # Update history: drop oldest step, append newest step
                num_forcing_vars = len(features_cfg.forcing)
                current_forcing_history_scaled = torch.cat(
                    [
                        current_forcing_history_scaled[
                            :, num_forcing_vars:
                        ],
                        next_forcing_slice_scaled,
                    ],
                    dim=1,
                )

    if not predictions:
        return torch.tensor([])

    # Stack all predictions
    # Result shape: [num_steps, N, 3] -> [N, 3, num_steps]
    return torch.stack(predictions, dim=0).permute(1, 2, 0)


if __name__ == "__main__":
    """
    Run doctests for this module.

    From the command line, run:
    python -m mswegnn.utils.adforce_dataset
    """
    import doctest

    doctest.testmod(verbose=True)
    print("Doctests complete.")