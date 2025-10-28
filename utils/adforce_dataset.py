"""
New file, e.g., adforce_dataset.py
This version uses the standard "lazy-loading" Dataset class.
"""
import torch
import xarray as xr
import glob
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import os
import pickle
import warnings


class AdforceLazyDataset(Dataset):
    """
    A "lazy-loading" PyG Dataset for multiple NetCDF simulations.

    It works by:
    1.  In process(), it scans all NetCDF files to build an "index map".
        This map links a global index (0 to N-1) to a specific
        (file_path, time_index_in_file) tuple.
    2.  In get(idx), it looks up the file and time, opens the NetCDF,
        and constructs the single Data object on the fly.
    """
    def __init__(self, root, nc_files, transform=None, pre_transform=None):
        """
        Args:
            root (str): Root directory to store processed index map.
            nc_files (list[str]): The list of NetCDF files for THIS split.
        """
        self.nc_files = sorted(nc_files)

        # --- Caches for static data and open file handles ---
        # We'll populate these on the first call to get()
        self._static_data = None
        self._file_handles = {} # Keep files open for speed

        super().__init__(root, transform, pre_transform)

        # Load the index map created by process()
        with open(self.processed_paths[0], 'rb') as f:
            self.index_map = pickle.load(f)

    @property
    def processed_file_names(self):
        """The file that will store our index map."""
        return ['index_map.pkl']

    def process(self):
        """
        Runs ONCE. Scans all files, verifies mesh consistency,
        and builds the index map.
        """
        print(f"Building index map for {len(self.nc_files)} files...")

        # 1. Load static data from the FIRST file for comparison
        static_data = self._load_static_data(self.nc_files[0])

        index_map = []
        global_idx = 0

        for nc_path in tqdm(self.nc_files, desc="Processing files"):
            with xr.open_dataset(nc_path) as ds:

                # --- Sanity Check ---
                # Verify the mesh is identical
                if not torch.equal(static_data['edge_index'],
                                   torch.tensor(ds['edges'].values, dtype=torch.long)):
                    warnings.warn(f"Mesh mismatch in {nc_path}! Skipping file.")
                    continue

                num_timesteps = ds.dims['time']

                # We can't get a target for the last time step
                for t in range(num_timesteps - 1):
                    index_map.append((nc_path, t))
                    global_idx += 1

        if not index_map:
            raise IOError("No valid time steps found across all NetCDF files.")

        print(f"Index map built. Total samples: {len(index_map)}")

        # Save the map
        with open(self.processed_paths[0], 'wb') as f:
            pickle.dump(index_map, f)

    def _load_static_data(self, nc_path):
        """Helper to load mesh/static data from a file."""
        with xr.open_dataset(nc_path) as ds_static:
            edge_index = torch.tensor(ds_static['edges'].values, dtype=torch.long)
            bathymetry = torch.tensor(ds_static['z'].values, dtype=torch.float)
            edge_length = torch.tensor(ds_static['edge_length'].values, dtype=torch.float)
            edge_width = torch.tensor(ds_static['width'].values, dtype=torch.float)

            nodes_n1 = ds_static['edges'].values[0]
            nodes_n2 = ds_static['edges'].values[1]
            depth_n1 = torch.tensor(ds_static['z'].sel(node=nodes_n1).values, dtype=torch.float)
            depth_n2 = torch.tensor(ds_static['z'].sel(node=nodes_n2).values, dtype=torch.float)
            mean_depth = 0.5 * (depth_n1 + depth_n2)
            cross_section = (edge_width * mean_depth).unsqueeze(1) # [n_edges, 1]

            static_edge_attr = torch.cat([
                edge_length.unsqueeze(1),
                cross_section
            ], dim=1) # [n_edges, 2]

            # ! Implement your boundary logic here !
            node_type = torch.zeros(ds_static.dims['node'], dtype=torch.float)

            static_node_features = torch.stack([
                bathymetry,
                node_type
            ], dim=1) # [n_nodes, 2]

            return {
                'edge_index': edge_index,
                'static_node_features': static_node_features,
                'static_edge_attr': static_edge_attr
            }

    def len(self):
        """Returns the total number of samples (time steps)"""
        return len(self.index_map)

    def get(self, idx):
        """
        THE "LAZY" PART.
        Loads a single (t, t+1) sample from disk.
        """

        # 1. Load static data if not already cached
        if self._static_data is None:
            self._static_data = self._load_static_data(self.nc_files[0])

        # 2. Find which file and time index this sample corresponds to
        nc_path, t = self.index_map[idx]

        # 3. Open the file (or get from cache)
        if nc_path not in self._file_handles:
            self._file_handles[nc_path] = xr.open_dataset(nc_path)
        ds = self._file_handles[nc_path]

        # 4. Get dynamic features for time t (inputs)
        dyn_node_features_t = torch.tensor(
            ds[['windx', 'windy', 'pressure']].isel(time=t).to_array().values.T,
            dtype=torch.float
        )

        # 5. Get dynamic features for time t+1 (targets)
        y_tplus1 = torch.tensor(
            ds[['water_level', 'velocity_u', 'velocity_v']].isel(time=t+1).to_array().values.T,
            dtype=torch.float
        )

        # 6. Combine static and dynamic inputs
        x_t = torch.cat([
            self._static_data['static_node_features'],
            dyn_node_features_t
        ], dim=1)

        # 7. Construct and return the Data object
        return Data(x=x_t,
                    edge_index=self._static_data['edge_index'],
                    edge_attr=self._static_data['static_edge_attr'],
                    y=y_tplus1)

    def close(self):
        """Call this to close all open NetCDF file handles."""
        for handle in self._file_handles.values():
            handle.close()
        self._file_handles = {}
