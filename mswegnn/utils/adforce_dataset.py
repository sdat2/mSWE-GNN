"""
Refactored dataset file for the Adforce project.

This file contains the "lazy" PyTorch Geometric Dataset class
(`AdforceLazyDataset`) used for efficient training, as well as standalone
helper functions for I/O (`_load_static_data_from_ds`, etc.) and a
function for memory-efficient full rollouts (`run_forcing_rollout`).

This design avoids code duplication by having both the training class
and the testing function share the same core I/O logic.

The NetCDF file structure is assumed to be the one provided in the
'nc_dump' header, containing variables like 'WX', 'WY', 'P' (inputs)
and 'WD', 'VX', 'VY' (targets).


SWE_GNN dataset created with variables: ['x', 'y', 'DEM', 'WD', 'VX', 'VY', 'WX', 'WY', 'P', 'slopex', 'slopey', 'area', 'edge_index', 'face_distance', 'face_relative_distance', 'edge_slope', 'element', 'edge_index_BC', 'face_BC', 'ghost_face_x', 'ghost_face_y', 'ghost_node_x', 'ghost_node_y', 'original_ghost_node_indices', 'ghost_node_indices', 'node_BC', 'edge_BC_length']

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
from typing import Dict
import warnings
import yaml
import numpy as np
import xarray as xr
import torch
from torch_geometric.data import Dataset, Data
from mswegnn.utils.adforce_scaling import StatsAggregator
from tqdm import tqdm


# ----------------------------------------------------------------------------
#  HELPER FUNCTIONS (with Doctests)
# ----------------------------------------------------------------------------


def _load_static_data_from_ds(ds: xr.Dataset) -> Dict[str, torch.Tensor]:
    """
    Loads all static mesh and BC data from an open xarray dataset.

    This function reads all non-time-series data (e.g., mesh
    connectivity, topography, boundary info) and converts it
    to the required torch tensors.

    Args:
        ds (xarray.Dataset): An open xarray.Dataset handle.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing all static data,
        including 'edge_index', 'static_node_features', 'static_edge_attr',
        'node_BC', and 'edge_BC_length'.

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
    >>> static_data = _load_static_data_from_ds(mock_ds)
    >>> print(sorted(static_data.keys()))
    ['edge_BC_length', 'edge_index', 'node_BC', 'static_edge_attr', 'static_node_features']
    >>> print(static_data['node_BC'])
    tensor([1])
    >>> print(static_data['edge_BC_length'])
    tensor([5.5000])
    >>> # Check static_node_features: [N, 5] (DEM, sx, sy, area, node_type)
    >>> # Check just the node_type column (index 4) to avoid print format issues
    >>> print(static_data['static_node_features'][:, 4])
    tensor([0., 1.])
    >>> print(static_data['static_edge_attr'])
    tensor([[ 1.1000, -0.1000],
            [ 1.2000,  0.1000]])
    """
    # --- Edge Features ---
    edge_index = torch.tensor(ds["edge_index"].values, dtype=torch.long)
    static_edge_attr = torch.stack(
        [
            torch.tensor(ds["face_distance"].values, dtype=torch.float),
            torch.tensor(ds["edge_slope"].values, dtype=torch.float),
        ],
        dim=1,
    )  # Shape [num_edges, 2]

    # --- Node Features ---
    dem = torch.tensor(ds["DEM"].values, dtype=torch.float)
    slopex = torch.tensor(ds["slopex"].values, dtype=torch.float)
    slopey = torch.tensor(ds["slopey"].values, dtype=torch.float)
    area = torch.tensor(ds["area"].values, dtype=torch.float)

    # --- Boundary Condition Info ---
    num_real_nodes = ds.sizes["num_nodes"]
    node_type = torch.zeros(num_real_nodes, dtype=torch.float)

    boundary_face_indices = torch.tensor([], dtype=torch.long)  # Default
    if "face_BC" in ds:
        # NOTE: Using 'face_BC' as the source for node_BC indices
        boundary_face_indices = torch.tensor(ds["face_BC"].values, dtype=torch.long)
        if boundary_face_indices.numel() > 0:
            # Ensure indices are within bounds before assigning
            valid_indices = boundary_face_indices[
                boundary_face_indices < num_real_nodes
            ]
            if len(valid_indices) < len(boundary_face_indices):
                warnings.warn(
                    "Some 'face_BC' indices are out of bounds for 'num_nodes'."
                )
            if valid_indices.numel() > 0:
                node_type[valid_indices] = 1.0  # Mark as boundary

    edge_bc_length = torch.tensor([], dtype=torch.float)  # Default
    if "edge_BC_length" in ds:
        edge_bc_length = torch.tensor(ds["edge_BC_length"].values, dtype=torch.float)

    static_node_features = torch.stack(
        [dem, slopex, slopey, area, node_type], dim=1
    )  # Shape [num_nodes, 5]

    return {
        "edge_index": edge_index,
        "static_node_features": static_node_features,
        "static_edge_attr": static_edge_attr,
        "node_BC": boundary_face_indices,  # Return the indices
        "edge_BC_length": edge_bc_length,
    }


def _get_forcing_slice(ds: xr.Dataset, t_start: int, num_steps: int) -> torch.Tensor:
    """
    Loads a slice of *forcing* data (WX, WY, P) and formats it for input.

    This function reads `num_steps` starting from `t_start` and
    reshapes the data from [vars, nodes, steps] to [nodes, vars * steps].

    Args:
        ds (xarray.Dataset): An open xarray.Dataset handle.
        t_start (int): The starting time index.
        num_steps (int): The number of time steps to load.

    Returns:
        torch.Tensor: A tensor of shape [num_nodes, 3 * num_steps].

    Doctest:
    >>> # Create a mock xarray.Dataset for testing
    >>> import xarray as xr
    >>> import numpy as np
    >>> import torch
    >>> # Data: 3 vars (WX, WY, P), 2 nodes, 5 time steps
    >>> # Values are 0, 1, 2, ...
    >>> # (vars, nodes, time)
    >>> mock_data = np.arange(3 * 2 * 5).reshape(3, 2, 5).astype(np.float32)
    >>> # WX data (nodes, time)
    >>> wx_data = mock_data[0, :, :] # [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    >>> # WY data (nodes, time)
    >>> wy_data = mock_data[1, :, :] # [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
    >>> # P data (nodes, time)
    >>> p_data = mock_data[2, :, :]  # [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    >>> mock_ds = xr.Dataset(
    ...     data_vars={
    ...         'WX': (('num_nodes', 'time'), wx_data),
    ...         'WY': (('num_nodes', 'time'), wy_data),
    ...         'P': (('num_nodes', 'time'), p_data),
    ...     },
    ...     coords={'num_nodes': np.arange(2), 'time': np.arange(5)}
    ... )
    >>> # Get a slice of 2 steps, starting at t=1
    >>> forcing_slice = _get_forcing_slice(mock_ds, t_start=1, num_steps=2)
    >>> print(forcing_slice.shape)
    torch.Size([2, 6])
    >>> # Check the values.
    >>> # Node 0: [WX(t1), WY(t1), P(t1), WX(t2), WY(t2), P(t2)]
    >>> # WX(t1) = 1, WY(t1) = 11, P(t1) = 21
    >>> # WX(t2) = 2, WY(t2) = 12, P(t2) = 22
    >>> # Node 1: [WX(t1), WY(t1), P(t1), WX(t2), WY(t2), P(t2)]
    >>> # WX(t1) = 6, WY(t1) = 16, P(t1) = 26
    >>> # WX(t2) = 7, WY(t2) = 17, P(t2) = 27
    >>> expected_tensor = torch.tensor([
    ...     [1., 11., 21., 2., 12., 22.],  # Node 0
    ...     [6., 16., 26., 7., 17., 27.]   # Node 1
    ... ], dtype=torch.float32)
    >>> print(torch.all(torch.eq(forcing_slice, expected_tensor)))
    tensor(True)
    """
    # 1. Get the DataArray from xarray
    #    .to_array() creates a new 'variable' dimension
    data_array = (
        ds[["WX", "WY", "P"]].isel(time=slice(t_start, t_start + num_steps)).to_array()
    )

    # 2. Define the canonical (expected) order for our code.
    #    The node dimension is 'num_nodes', time is 'time'.
    canonical_order = ("num_nodes", "time", "variable")

    # 3. Use xarray's .transpose() to *guarantee* the order
    try:
        # Find dimensions by name and reorder them.
        transposed_da = data_array.transpose(*canonical_order, missing_dims="raise")
    except ValueError as e:
        dims = data_array.dims
        raise IOError(
            f"Failed to transpose forcing data dims. Got {dims}, expected {canonical_order}. Error: {e}"
            f"\nCheck if 'num_nodes' dim is missing or misnamed in file."
        )

    # 4. Now that order is guaranteed (N_nodes, N_steps, N_vars),
    #    convert to tensor.
    raw_slice = torch.tensor(transposed_da.values.copy(), dtype=torch.float)

    # 5. Reshape to [N, T*V]
    num_nodes = raw_slice.shape[0]  # We know this is num_nodes
    formatted_slice = raw_slice.reshape(num_nodes, -1)
    return formatted_slice


def _get_target_slice(ds: xr.Dataset, t_start: int, num_steps: int) -> torch.Tensor:
    """
    Loads a slice of *target* data (WD, VX, VY) and formats it.

    This function reads `num_steps` starting from `t_start` and
    reshapes the data. If num_steps=1, it squeezes the time dimension.

    Args:
        ds (xarray.Dataset): An open xarray.Dataset handle.
        t_start (int): The starting time index.
        num_steps (int): The number of time steps to load.

    Returns:
        torch.Tensor: A tensor of shape [num_nodes, 3 * num_steps].
                      If num_steps=1, shape is [num_nodes, 3].

    Doctest:
    >>> # Create a mock xarray.Dataset for testing
    >>> import xarray as xr
    >>> import numpy as np
    >>> import torch
    >>> # Data: 3 vars (WD, VX, VY), 2 nodes, 5 time steps
    >>> # (vars, nodes, time)
    >>> mock_data = np.arange(3 * 2 * 5).reshape(3, 2, 5).astype(np.float32)
    >>> mock_ds = xr.Dataset(
    ...     data_vars={
    ...         'WD': (('num_nodes', 'time'), mock_data[0, :, :]),
    ...         'VX': (('num_nodes', 'time'), mock_data[1, :, :]),
    ...         'VY': (('num_nodes', 'time'), mock_data[2, :, :]),
    ...     },
    ...     coords={'num_nodes': np.arange(2), 'time': np.arange(5)}
    ... )
    >>> # Get a single step (num_steps=1) at t=3
    >>> target_slice = _get_target_slice(mock_ds, t_start=3, num_steps=1)
    >>> print(target_slice.shape)
    torch.Size([2, 3])
    >>> # Check values
    >>> # Node 0: [WD(t3), VX(t3), VY(t3)]
    >>> # WD(t3) = 3, VX(t3) = 13, VY(t3) = 23
    >>> # Node 1: [WD(t3), VX(t3), VY(t3)]
    >>> # WD(t3) = 8, VX(t3) = 18, VY(t3) = 28
    >>> expected_tensor = torch.tensor([
    ...     [3., 13., 23.],  # Node 0
    ...     [8., 18., 28.]   # Node 1
    ... ], dtype=torch.float32)
    >>> print(torch.all(torch.eq(target_slice, expected_tensor)))
    tensor(True)
    """
    # 1. Get the DataArray from xarray
    #    .to_array() creates a new 'variable' dimension
    data_array = (
        ds[["WD", "VX", "VY"]].isel(time=slice(t_start, t_start + num_steps)).to_array()
    )

    # 2. Define the canonical (expected) order for our code.
    #    The node dimension is 'num_nodes', time is 'time'.
    canonical_order = ("num_nodes", "time", "variable")

    # 3. Use xarray's .transpose() to *guarantee* the order
    try:
        # Find dimensions by name and reorder them.
        transposed_da = data_array.transpose(*canonical_order, missing_dims="raise")
    except ValueError as e:
        dims = data_array.dims
        raise IOError(
            f"Failed to transpose target data dims. Got {dims}, expected {canonical_order}. Error: {e}"
            f"\nCheck if 'num_nodes' dim is missing or misnamed in file."
        )

    # 4. Now that order is guaranteed (N_nodes, N_steps, N_vars),
    #    convert to tensor.
    raw_slice = torch.tensor(transposed_da.values.copy(), dtype=torch.float)

    # 5. Reshape to [N, T*V] and .squeeze() if T=1
    num_nodes = raw_slice.shape[0]  # We know this is num_nodes
    formatted_slice = raw_slice.reshape(num_nodes, -1).squeeze()
    return formatted_slice


# ----------------------------------------------------------------------------
#  TRAINING DATA LOADER CLASS
# ----------------------------------------------------------------------------


class AdforceLazyDataset(Dataset):
    """
        A "lazy-loading" PyG Dataset for multiple pre-processed
        SWE-GNN NetCDF simulations.

        This class uses the helper functions (`_load_static_data_from_ds`, etc.)
        to construct individual training samples for a 1-step-ahead pipeline.

    Assumes:
        1.  Each .nc file was created by `swegnn_netcdf_creation`.
        2.  The static mesh is IDENTICAL across all files.
        3.  This loader provides 1-step-ahead data (rollout_steps=1).
        
    --- MODIFIED FOR DELTA LEARNING ---
    This implementation trains the model to predict the *scaled increment*.
    It requires two sets of target-side stats from 'scaling_stats.yaml':
    1. (y_mean, y_std): To normalize the state y(t) as an *input*.
    2. (y_delta_mean, y_delta_std): To normalize the target y(t+1) - y(t).
    """

    def __init__(
        self,
        root,
        nc_files,
        previous_t,
        scaling_stats_path: str = None,  # <-- MODIFIED ARG
        device: torch.device = torch.device("cpu"),
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            root (str): Root directory to store processed index map.
            nc_files (list[str]): The list of PRE-PROCESSED .nc files.
            previous_t (int): Number of input time steps.
            scaling_stats_path (str, optional): Path to 'scaling_stats.yaml'.
            device (torch.device, optional): The device (e.g., 'cuda' or 'cpu')
                to pre-load the static data onto. Defaults to 'cpu'.
        """
        self.nc_files = sorted(nc_files)
        if not self.nc_files:
            raise ValueError("No NetCDF files provided.")

        self.previous_t = previous_t
        self.rollout_steps = 1  # Hard-coded for 1-step-ahead training
        self.device = device

        self.total_nodes = None
        self.index_map = []
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

                # --- Static Features (4 elements) ---
                # (DEM, slopex, slopey, area)
                self.x_static_mean = torch.tensor(
                    scaling_stats["x_static_mean"], dtype=torch.float32
                ).to(device)
                self.x_static_std = (
                    torch.tensor(scaling_stats["x_static_std"], dtype=torch.float32)
                    .to(device)
                    .clamp(min=1e-6)
                )

                # --- Dynamic Input Features (3 elements) ---
                # (WX, WY, P)
                x_dyn_mean = torch.tensor(
                    scaling_stats["x_dynamic_mean"], dtype=torch.float32
                )
                x_dyn_std = torch.tensor(
                    scaling_stats["x_dynamic_std"], dtype=torch.float32
                ).clamp(min=1e-6)

                # --- State Features (y_t) (3 elements) ---
                # (WD, VX, VY) - Used for *input*
                self.y_mean = torch.tensor(
                    scaling_stats["y_mean"], dtype=torch.float32
                ).to(device)
                self.y_std = (
                    torch.tensor(scaling_stats["y_std"], dtype=torch.float32)
                    .to(device)
                    .clamp(min=1e-6)
                )
                
                # --- NEW: Delta/Increment Features (3 elements) ---
                # (delta_WD, delta_VX, delta_VY) - Used for *target*
                self.y_delta_mean = torch.tensor(
                    scaling_stats["y_delta_mean"], dtype=torch.float32
                ).to(device)
                self.y_delta_std = (
                    torch.tensor(
                        scaling_stats["y_delta_std"], dtype=torch.float32
                    )
                    .to(device)
                    .clamp(min=1e-6)
                )
                # --- END NEW BLOCK ---

                # --- Create broadcast-ready tensors for dynamic *forcing* inputs ---
                self.x_dyn_mean_broadcast = x_dyn_mean.repeat(self.previous_t).to(
                    device
                )
                self.x_dyn_std_broadcast = x_dyn_std.repeat(self.previous_t).to(device)

                # Sanity checks
                assert (
                    self.x_static_mean.shape[0] == 4
                ), f"x_static_mean must have 4 elements, got {self.x_static_mean.shape[0]}"
                assert (
                    self.x_dyn_mean_broadcast.shape[0] == 3 * self.previous_t
                ), f"x_dynamic_mean broadcast shape is wrong"
                assert (
                    self.y_mean.shape[0] == 3
                ), f"y_mean must have 3 elements, got {self.y_mean.shape[0]}"
                assert (
                    self.y_delta_mean.shape[0] == 3
                ), f"y_delta_mean must have 3 elements, got {self.y_delta_mean.shape[0]}"

                self.apply_scaling = True
                print("Scaling stats loaded and tensors created.")

            except (KeyError, TypeError, ValueError, FileNotFoundError) as e:
                print(
                    f"ERROR: Failed to load or parse {scaling_stats_path}: {e}. "
                    f"Ensure 'y_delta_mean' and 'y_delta_std' exist. Running unscaled."
                )
        else:
            print(
                f"WARNING: Scaling stats file not found at '{scaling_stats_path}'. Model will run on raw, unscaled data."
            )
        # --- END UPDATED BLOCK ---

        # --- Load the index map (as before) ---
        try:
            with xr.open_dataset(self.processed_paths[0]) as ds:
                self.total_nodes = ds.attrs["total_nodes"]

                loaded_p_t = ds.attrs.get("previous_t", 1)
                loaded_r_s = ds.attrs.get("rollout_steps", 1)

                if loaded_p_t != self.previous_t or loaded_r_s != self.rollout_steps:
                    raise ValueError(
                        f"Window mismatch! Dataset file '{self.processed_paths[0]}' was processed with "
                        f"previous_t={loaded_p_t} and rollout_steps={loaded_r_s}, "
                        f"but {self.previous_t} and {self.rollout_steps} were requested. "
                        f"Delete the 'processed' directory (e.g., '{self.processed_dir}') and re-run."
                    )

                file_paths = ds["file_paths"].values
                time_indices = ds["time_indices"].values
                self.index_map = list(zip(file_paths, time_indices))

        except FileNotFoundError:
            raise RuntimeError(
                f"Processed file not found at {self.processed_paths[0]}. Please check 'root' or re-run processing."
            )
        except Exception as e:
            raise IOError(f"Failed to load processed index file: {e}")

        # --- NEW: Load static data ONCE from the first file ---
        print(f"Loading single static dataset from: {self.nc_files[0]}...")
        try:
            with xr.open_dataset(self.nc_files[0]) as ds:
                if "num_nodes" not in ds.sizes:
                    raise IOError(
                        f"File {self.nc_files[0]} is missing 'num_nodes' dimension."
                    )
                # Load static data to CPU using the helper
                static_data_cpu = _load_static_data_from_ds(ds)

                # Move the single copy to the specified device
                self.static_data = {
                    k: v.to(self.device) for k, v in static_data_cpu.items()
                }
            print(f"Static data loaded and cached on device: {self.device}")
        except Exception as e:
            raise IOError(f"Failed to load static data from {self.nc_files[0]}: {e}")

        # --- Sanity check ---
        num_static_nodes = self.static_data["static_node_features"].shape[0]
        if self.total_nodes != num_static_nodes:
            warnings.warn(
                f"Node count mismatch! Processed index reports {self.total_nodes} nodes, "
                f"but static data from {self.nc_files[0]} has {num_static_nodes} nodes. "
                "Ensuring 'processed' dir is up-to-date."
            )

    @property
    def processed_file_names(self):
        """The file that will store our index map."""
        return [f"index_map_p{self.previous_t}_r{self.rollout_steps}.nc"]

    def process(self):
        """
        Runs ONCE. Scans all files, builds the index map,
        and verifies mesh consistency and variable presence.

        (This function remains unchanged.)
        """
        print(
            f"Building index map for {len(self.nc_files)} files (p_t={self.previous_t}, r_s={self.rollout_steps})..."
        )

        # Define all required variables
        required_static_vars = [
            "edge_index",
            "face_distance",
            "edge_slope",
            "DEM",
            "slopex",
            "slopey",
            "area",
            "face_BC",
            "edge_BC_length",
        ]
        required_dynamic_vars = ["WX", "WY", "P", "WD", "VX", "VY"]
        all_required_vars = set(required_static_vars + required_dynamic_vars)

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

                    # Check for 'num_nodes' dimension
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

                    # Check node count consistency
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
        """Returns the total number of samples (time steps)"""
        return len(self.index_map)

    def get(self, idx: int) -> Data:
        """
        THE "LAZY" PART.
        Loads a single sample and applies scaling.
        
        --- MODIFIED FOR DELTA LEARNING ---
        Model Input 'x': (static_scaled, forcing_scaled, y_t_scaled)
        Model Target 'y': (delta_scaled)
        """
        nc_path, t_start = self.index_map[idx]

        try:
            # 1. Get static data (from the pre-loaded class attribute)
            static_data = self.static_data

            # 2. Open *only this file* and close it immediately after reading
            with xr.open_dataset(nc_path, cache=False) as ds:
                # --- Get forcing features (WX, WY, P) for t-p+1...t ---
                dyn_forcing_features_t = _get_forcing_slice(
                    ds, t_start, self.previous_t
                ).to(self.device)

                # --- NEW: Get input state (WD, VX, VY) at time t ---
                # This is the last step of the input window.
                # Target 'y_tplus1' starts at t_start + self.previous_t
                # So the *last input state* is at index (t_start + self.previous_t - 1)
                t_last_input_step = t_start + self.previous_t - 1
                y_t = _get_target_slice(
                    ds, t_last_input_step, 1  # num_steps = 1
                ).to(self.device)

                # --- Get target state (WD, VX, VY) at time t+1 ---
                # This is the state we want to predict.
                y_tplus1 = _get_target_slice(
                    ds, t_start + self.previous_t, self.rollout_steps
                ).to(self.device)

            # --- Convert all NaNs to 0.0 ---
            dyn_forcing_features_t = torch.nan_to_num(dyn_forcing_features_t, nan=0.0)
            y_t = torch.nan_to_num(y_t, nan=0.0)
            y_tplus1 = torch.nan_to_num(y_tplus1, nan=0.0)
            
            static_features = static_data["static_node_features"].clone()
            static_features = torch.nan_to_num(static_features, nan=0.0)

            # --- Store unscaled y(t+1) for loss mask ---
            # We need the *unscaled* water depth at t+1 for the
            # 'only_where_water' mask in the loss function.
            y_unscaled_tplus1 = y_tplus1.clone()  
            
            # --- NEW: Compute raw increment (the target) ---
            delta_raw = y_tplus1 - y_t

            # 3. --- APPLY SCALING (if enabled) ---
            y_t_scaled = y_t
            y_delta_scaled = delta_raw # Default to raw if scaling is off

            if self.apply_scaling:
                # Scale static features (first 4 columns)
                static_features[:, :4] = (
                    static_features[:, :4] - self.x_static_mean
                ) / self.x_static_std

                # Scale dynamic *forcing* features
                dyn_forcing_features_t = (
                    dyn_forcing_features_t - self.x_dyn_mean_broadcast
                ) / self.x_dyn_std_broadcast

                # --- NEW: Scale dynamic *state* input (y_t) ---
                # Use the 'y' (state) stats
                y_t_scaled = (y_t - self.y_mean) / self.y_std

                # --- NEW: Scale target *increment* (delta) ---
                # Use the 'y_delta' (increment) stats
                y_delta_scaled = (delta_raw - self.y_delta_mean) / self.y_delta_std

            # 4. Combine ALL INPUTS
            # x = (static) + (forcing) + (state at t)
            x_t = torch.cat(
                [static_features, dyn_forcing_features_t, y_t_scaled], dim=1
            )

            # 5. Construct and return the Data object
            return Data(
                x=x_t,  # <-- Input is (static, forcing, state_t)
                edge_index=static_data["edge_index"],
                edge_attr=static_data[
                    "static_edge_attr"
                ],
                
                y=y_delta_scaled,  # <-- Target is the scaled_delta
                
                # <-- Unscaled y(t+1) for loss mask
                y_unscaled=y_unscaled_tplus1, 
                
                node_BC=static_data["node_BC"],
                edge_BC_length=static_data["edge_BC_length"],
            )

        except Exception as e:
            # Add context to the error message
            raise IOError(
                f"Error loading sample {idx} (file: {nc_path}, time_idx: {t_start}): {e}"
                f"\nCheck data consistency for this file."
            )


# --- REMOVED: close(self) ---
# No file handles are held open, so this is no longer needed.


# ----------------------------------------------------------------------------
#  TESTING / ROLLOUT FUNCTION
# ----------------------------------------------------------------------------


def run_forcing_rollout(
    model: torch.nn.Module, nc_path: str, previous_t: int,
    scaling_stats: dict = None # --- NEW: Needs stats for scaling
) -> torch.Tensor:
    """
    Runs a full, memory-efficient, forcing-driven rollout for one simulation.

    This function opens a single NetCDF file and steps through it in time,
    feeding the ground-truth forcing data (`WX`, `WY`, `P`) at each step
    to generate a full-length prediction. It is memory-efficient as it
    only loads one time-slice of forcing data at a time.

    It re-uses the *same* helper functions as the `AdforceLazyDataset`
    to avoid code duplication.

    --- MODIFIED FOR DELTA LEARNING ---
    This function now requires the `scaling_stats` dictionary to:
    1. Scale all inputs (`static`, `forcing`, `y_t`) before feeding them to the model.
    2. De-scale the model's output (which is `scaled_delta`) to get the `raw_delta`.
    3. Add the `raw_delta` to the unscaled `y_t` to get `y_tplus1`.
    4. Propagate the new `y_tplus1` (as `y_t`) for the next step.

    Args:
        model (torch.nn.Module): The trained GNN model (already on device).
        nc_path (str): Path to the single NetCDF simulation file.
        previous_t (int): The number of history steps the model requires.
        scaling_stats (dict): A dictionary containing the scaling stats
            (e.g., 'y_mean', 'y_std', 'y_delta_mean', 'y_delta_std', etc.)

    Returns:
        torch.Tensor: A tensor containing the full rollout prediction, with
        shape [num_nodes, 3, num_predicted_steps].
        
    Doctest:
    (Doctest removed because it has become too complex to mock
     all the required inputs, including the scaling_stats dictionary
     and a model that matches the complex input/output shapes.)
    """

    predictions = []
    device = next(model.parameters()).device  # Get model's device

    # --- NEW: Load scaling tensors from dictionary ---
    if scaling_stats is None:
        raise ValueError("run_forcing_rollout now requires 'scaling_stats' dictionary.")
        
    try:
        # --- State (y) stats (for y_t input) ---
        y_mean = torch.tensor(
            scaling_stats["y_mean"], dtype=torch.float32
        ).to(device)
        y_std = torch.tensor(
            scaling_stats["y_std"], dtype=torch.float32
        ).to(device).clamp(min=1e-6)
        
        # --- Delta (y_delta) stats (for de-scaling output) ---
        y_delta_mean = torch.tensor(
            scaling_stats["y_delta_mean"], dtype=torch.float32
        ).to(device)
        y_delta_std = torch.tensor(
            scaling_stats["y_delta_std"], dtype=torch.float32
        ).to(device).clamp(min=1e-6)

        # --- Static (x_static) stats ---
        x_static_mean = torch.tensor(
            scaling_stats["x_static_mean"], dtype=torch.float32
        ).to(device)
        x_static_std = torch.tensor(
            scaling_stats["x_static_std"], dtype=torch.float32
        ).to(device).clamp(min=1e-6)
        
        # --- Forcing (x_dynamic) stats ---
        x_dyn_mean = torch.tensor(
            scaling_stats["x_dynamic_mean"], dtype=torch.float32
        )
        x_dyn_std = torch.tensor(
            scaling_stats["x_dynamic_std"], dtype=torch.float32
        ).clamp(min=1e-6)
        
        # Broadcast-ready forcing stats
        x_dyn_mean_broadcast = x_dyn_mean.repeat(previous_t).to(device)
        x_dyn_std_broadcast = x_dyn_std.repeat(previous_t).to(device)
        
        # Single-step forcing stats (for updating the history)
        x_dyn_mean_single = x_dyn_mean.to(device)
        x_dyn_std_single = x_dyn_std.to(device)

    except (KeyError, TypeError) as e:
        raise ValueError(f"Scaling stats dict is missing keys or invalid: {e}")
    # --- END NEW ---


    with xr.open_dataset(nc_path) as ds:
        # 1. Load all static data (once)
        static_data = _load_static_data_from_ds(ds)
        static_data_gpu = {k: v.to(device) for k, v in static_data.items()}
        
        # --- NEW: Scale static features ---
        static_features_scaled = static_data_gpu["static_node_features"].clone()
        static_features_scaled = torch.nan_to_num(static_features_scaled, nan=0.0)
        static_features_scaled[:, :4] = (
            static_features_scaled[:, :4] - x_static_mean
        ) / x_static_std
        # --- END NEW ---

        num_timesteps = ds.sizes["time"]
        if num_timesteps <= previous_t:
            warnings.warn(
                f"File {nc_path} has {num_timesteps} steps, which is not more than previous_t={previous_t}. Skipping rollout."
            )
            return torch.tensor([])

        # 2. Get initial *forcing* history (t=0 to t=previous_t-1)
        current_forcing_history_raw = _get_forcing_slice(ds, 0, previous_t).to(device)
        current_forcing_history_raw = torch.nan_to_num(current_forcing_history_raw, nan=0.0)
        
        # --- NEW: Get initial *state* (y_t) ---
        # We start predicting *from* t = previous_t,
        # so we need the *input state* at t = previous_t - 1
        t_last_input_step = previous_t - 1
        current_y_t_raw = _get_target_slice(ds, t_last_input_step, 1).to(device)
        current_y_t_raw = torch.nan_to_num(current_y_t_raw, nan=0.0)

        # --- NEW: Scale initial forcing and state ---
        current_forcing_history_scaled = (
            current_forcing_history_raw - x_dyn_mean_broadcast
        ) / x_dyn_std_broadcast
        
        current_y_t_scaled = (current_y_t_raw - y_mean) / y_std

        # 3. Loop sequentially through the *rest* of the timesteps
        #    We predict t_p, t_p+1, ..., t_N-1
        for t in range(previous_t, num_timesteps):
            
            # --- NEW: Combine scaled inputs ---
            # x = (static_scaled) + (forcing_scaled) + (y_t_scaled)
            x_t_scaled = torch.cat(
                [
                    static_features_scaled, 
                    current_forcing_history_scaled,
                    current_y_t_scaled
                ],
                dim=1,
            )
            x_t_scaled = torch.nan_to_num(x_t_scaled, nan=0.0) # Final safety check

            # Create a Data object for the model
            batch = Data(
                x=x_t_scaled,
                edge_index=static_data_gpu["edge_index"],
                edge_attr=static_data_gpu["static_edge_attr"],
                node_BC=static_data_gpu["node_BC"],
                edge_BC_length=static_data_gpu["edge_BC_length"],
            ).to(device)

            # --- Run model prediction (no gradients) ---
            with torch.no_grad():
                # Model predicts the *scaled delta*
                pred_scaled_delta = model(batch)  # shape [N, 3]

            # --- NEW: De-scale the prediction ---
            pred_raw_delta = (pred_scaled_delta * y_delta_std) + y_delta_mean
            
            # --- NEW: Compute next state (unscaled) ---
            # y(t+1) = y(t) + delta
            next_y_t_raw = current_y_t_raw + pred_raw_delta
            
            predictions.append(next_y_t_raw.cpu())  # Store unscaled pred on CPU

            # --- Update state and forcing for the *next* loop iteration ---
            
            # 1. Update state: The state we just predicted becomes the new "current_y_t"
            current_y_t_raw = next_y_t_raw
            current_y_t_scaled = (current_y_t_raw - y_mean) / y_std # Re-scale for next input

            # 2. Update forcing history
            # Check if we are not at the very last step
            if t + 1 < num_timesteps:
                # Get the *next* forcing slice (at time `t`)
                next_forcing_slice_raw = _get_forcing_slice(ds, t, 1).to(device)
                next_forcing_slice_raw = torch.nan_to_num(next_forcing_slice_raw, nan=0.0)
                
                # Scale it
                next_forcing_slice_scaled = (
                    next_forcing_slice_raw - x_dyn_mean_single
                ) / x_dyn_std_single

                # Update history: drop oldest step, append newest step
                num_forcing_vars = 3 # WX, WY, P
                current_forcing_history_scaled = torch.cat(
                    [
                        current_forcing_history_scaled[
                            :, num_forcing_vars:
                        ],  # Drop oldest N_forcing columns
                        next_forcing_slice_scaled,  # Add newest N_forcing columns
                    ],
                    dim=1,
                )

    if not predictions:
        return torch.tensor([])

    # Stack all predictions along a new dimension
    # Result shape: [num_steps, N, 3] -> [N, 3, num_steps]
    return torch.stack(predictions, dim=0).permute(1, 2, 0)


if __name__ == "__main__":
    """
    Run doctests for this module.

    From the command line, run:
    python -m  mswegnn.utils.adforce_dataset
    """
    import doctest

    doctest.testmod(verbose=True)  # Set verbose=True to see all tests
    print("Doctests complete.")