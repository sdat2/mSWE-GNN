# mswegnn/models/adforce_models.py
from typing import Union, Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Data, Batch

# Import from our new refactored files
from .adforce_base import AdforceBaseModel
from .adforce_helpers import make_mlp
from .adforce_processors import GNN_Adforce, SWEGNN_Adforce


class MonolithicMLPModel(nn.Module):
    """
    A "monolithic" MLP baseline that flattens the *entire* graph's
    node features into a single vector.

    Args:
        n_nodes (int): The *fixed* number of nodes in the graph.
                       This model cannot handle variable-sized graphs.
        num_node_features (int): Number of input features *per node*.
        num_output_features (int): Number of output features *per node*.
        hid_features (int): Width of the hidden layers.
        mlp_layers (int): Number of *hidden* layers.
        mlp_activation (str): Activation function to use (e.g., 'relu').
        **kwargs: Catches unused arguments from the config.

    Doctest:
    >>> import torch
    >>> from torch_geometric.data import Data
    >>>
    >>> # 1. Define test parameters
    >>> N_NODES_FIXED = 10
    >>> NUM_IN_FEATURES = 17
    >>> NUM_OUT_FEATURES = 3
    >>>
    >>> # 2. Create a single mock data object
    >>> mock_x = torch.rand(N_NODES_FIXED, NUM_IN_FEATURES)
    >>> mock_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    >>> data = Data(x=mock_x, edge_index=mock_edge_index)
    >>>
    >>> # 3. Instantiate the model
    >>> model = MonolithicMLPModel(
    ...     n_nodes=N_NODES_FIXED,
    ...     num_node_features=NUM_IN_FEATURES,
    ...     num_output_features=NUM_OUT_FEATURES,
    ...     hid_features=8,
    ...     mlp_layers=2,
    ...     mlp_activation='relu'
    ... )
    MonolithicMLPModel initialized:
      Fixed N_nodes: 10
      Node In Feat: 17
      Node Out Feat: 3
      --> Flat Input: 170
      --> Flat Output: 30
      Hidden Dim: 8 | Layers: 2
    >>>
    >>> # 4. Run forward pass on the single data object
    >>> out = model(data)
    >>>
    >>> # 5. Check output shape. B=1, N=10, F_out=3
    >>> print(f"Output shape: {out.shape}")
    Output shape: torch.Size([10, 3])
    """

    def __init__(
        self,
        n_nodes: int,
        num_node_features: int,
        num_output_features: int,
        hid_features: int,
        mlp_layers: int,
        mlp_activation: str,
        **kwargs,  # Catch unused args
    ):
        super().__init__()

        if not isinstance(n_nodes, int) or n_nodes <= 0:
            raise ValueError(f"n_nodes must be a positive integer. Got: {n_nodes}")

        self.n_nodes = n_nodes
        self.in_features_per_node = num_node_features
        self.out_features_per_node = num_output_features

        self.flat_input_dim = self.n_nodes * self.in_features_per_node
        self.flat_output_dim = self.n_nodes * self.out_features_per_node

        if mlp_activation.lower() == "relu":
            activation = nn.ReLU()
        elif mlp_activation.lower() == "gelu":
            activation = nn.GELU()
        elif mlp_activation.lower() == "tanh":
            activation = nn.Tanh()
        elif mlp_activation.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif mlp_activation.lower() == "prelu":
            activation = nn.PReLU()
        else:
            raise ValueError(f"Unknown activation: {mlp_activation}")

        layers = []
        layers.append(nn.Linear(self.flat_input_dim, hid_features))
        layers.append(activation)
        for _ in range(mlp_layers - 1):
            layers.append(nn.Linear(hid_features, hid_features))
            layers.append(activation)
        layers.append(nn.Linear(hid_features, self.flat_output_dim))
        self.network = nn.Sequential(*layers)

        print(
            f"MonolithicMLPModel initialized:\n"
            f"  Fixed N_nodes: {self.n_nodes}\n"
            f"  Node In Feat: {self.in_features_per_node}\n"
            f"  Node Out Feat: {self.out_features_per_node}\n"
            f"  --> Flat Input: {self.flat_input_dim}\n"
            f"  --> Flat Output: {self.flat_output_dim}\n"
            f"  Hidden Dim: {hid_features} | Layers: {mlp_layers}"
        )

    def forward(self, batch: Union[Data, Batch]) -> torch.Tensor:
        x = batch.x
        batch_size = getattr(batch, "num_graphs", 1)

        if x.shape[0] != batch_size * self.n_nodes:
            raise ValueError(
                f"Input shape mismatch for MonolithicMLP.\n"
                f"Model configured for n_nodes={self.n_nodes} (total nodes expected in batch: {batch_size * self.n_nodes}).\n"
                f"Received x shape {x.shape} (total nodes: {x.shape[0]}).\n"
                f"This model requires all graphs in the dataset to have *exactly* {self.n_nodes} nodes."
            )

        x_reshaped = x.view(batch_size, self.n_nodes, self.in_features_per_node)
        x_flat = x_reshaped.view(batch_size, -1)
        out_flat = self.network(x_flat)
        out_reshaped = out_flat.view(
            batch_size, self.n_nodes, self.out_features_per_node
        )
        out = out_reshaped.view(-1, self.out_features_per_node)
        return out


class PointwiseMLPModel(nn.Module):
    """
    A standalone MLP model wrapper that applies an MLP to each node.

    Args:
        num_node_features (int): Total number of features in the input `x` tensor.
        num_output_features (int): Number of output features to predict.
        **mlp_kwargs: Additional keyword arguments passed to the MLP constructor.

    Doctest:
    >>> import torch
    >>> from torch_geometric.data import Data
    >>>
    >>> # 1. Define test parameters
    >>> N_NODES = 10
    >>> # 5 static + (3 forcing * 3 steps) + 3 state = 17
    >>> NUM_IN_FEATURES = 17
    >>> NUM_OUT_FEATURES = 3 # (delta WD, delta VX, delta VY)
    >>>
    >>> # 2. Create mock data object (as a batch)
    >>> mock_x = torch.rand(N_NODES, NUM_IN_FEATURES)
    >>> mock_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    >>> batch = Data(x=mock_x, edge_index=mock_edge_index)
    >>>
    >>> # 3. Define model kwargs
    >>> mlp_kwargs = {
    ...     'hid_features': 16,
    ...     'mlp_layers': 3,
    ...     'mlp_activation': 'relu'
    ... }
    >>>
    >>> # 4. Instantiate the model
    >>> model = PointwiseMLPModel(
    ...     num_node_features=NUM_IN_FEATURES,
    ...     num_output_features=NUM_OUT_FEATURES,
    ...     **mlp_kwargs
    ... )
    PointwiseMLPModel initialized: 17 input features, 3 output features.
    >>>
    >>> # 5. Run forward pass
    >>> out = model(batch)
    >>>
    >>> # 6. Check output shape
    >>> print(f"Output shape: {out.shape}")
    Output shape: torch.Size([10, 3])
    """

    def __init__(
        self,
        num_node_features,
        num_output_features,
        **mlp_kwargs,
    ):
        super().__init__()
        self.in_features = num_node_features
        self.out_features = num_output_features

        # Map 'mlp_activation' (from config) to 'activation' (for make_mlp)
        activation_type = mlp_kwargs.pop("mlp_activation", "relu")
        
        # Map 'mlp_layers' (from config) to 'n_layers' (for make_mlp)
        n_layers = mlp_kwargs.pop("mlp_layers", 2)

        # Remove keys that GNN/MSGNN models use
        mlp_kwargs.pop("model_type", None)
        mlp_kwargs.pop("previous_t", None)
        mlp_kwargs.pop("num_static_features", None)
        mlp_kwargs.pop("num_edge_features", None)
        mlp_kwargs.pop("num_scales", None)
        mlp_kwargs.pop("learned_pooling", None)
        mlp_kwargs.pop("skip_connections", None)
        mlp_kwargs.pop("gnn_activation", None)
        mlp_kwargs.pop("type_gnn", None)

        self.mlp = make_mlp(
            input_size=self.in_features,
            output_size=self.out_features,
            activation=activation_type,
            n_layers=n_layers,
            **mlp_kwargs,  # Pass remaining (e.g., hid_features)
        )

        if "hid_features" in mlp_kwargs:
            print(
                f"PointwiseMLPModel initialized: {self.in_features} input features, "
                f"{self.out_features} output features."
            )

    def forward(self, batch):
        x = batch.x
        out = self.mlp(x)
        return out


class GNNModelAdforce(nn.Module):
    """
    The main GNNModel wrapper for the Adforce pipeline.

    This is the class your training script should import and use.

    Args:
        num_node_features (int): Total number of features in the input `x` tensor.
        num_edge_features (int): Total number of features in the `edge_attr` tensor.
        previous_t (int): Number of history steps included in the input.
        num_output_features (int): Number of output features to predict (Must be 3).
        num_static_features (int, optional): Number of static features at the
            start of the `x` tensor. Defaults to 5.
        **kwargs: Additional keyword arguments.

    Doctest:
    >>> import torch
    >>> from torch_geometric.data import Data
    >>>
    >>> # 1. Define test parameters
    >>> N_NODES = 10
    >>> PREVIOUS_T = 3
    >>> NUM_STATIC_FEATURES = 5  # (DEM, sx, sy, area, type)
    >>> NUM_OUTPUT_FEATURES = 3  # (delta WD, delta VX, delta VY)
    >>> NUM_DYNAMIC_FORCING_FEATURES = 3 # (WX, WY, P)
    >>> NUM_DYNAMIC_STATE_FEATURES = 3 # (WD, VX, VY)
    >>>
    >>> # 2. Calculate total input features
    >>> # 5 static + (3 forcing * 3 steps) + 3 state = 17
    >>> num_node_features = NUM_STATIC_FEATURES + (NUM_DYNAMIC_FORCING_FEATURES * PREVIOUS_T) + NUM_DYNAMIC_STATE_FEATURES
    >>> print(f"Calculated num_node_features: {num_node_features}")
    Calculated num_node_features: 17
    >>>
    >>> # 3. Create mock data object (as a batch)
    >>> mock_x = torch.rand(N_NODES, num_node_features)
    >>> mock_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    >>> batch = Data(x=mock_x, edge_index=mock_edge_index)
    >>>
    >>> # 4. Define model kwargs
    >>> gnn_kwargs = {
    ...     'hid_features': 16,
    ...     'mlp_layers': 2,
    ...     'type_gnn': 'GCN',
    ...     'learned_residuals': False # Arg for BaseFloodModel
    ... }
    >>>
    >>> # 5. Instantiate the model
    >>> model = GNNModelAdforce(
    ...     num_node_features=num_node_features,
    ...     num_edge_features=2, # Mock value, not used by GCN
    ...     previous_t=PREVIOUS_T,
    ...     num_output_features=NUM_OUTPUT_FEATURES,
    ...     num_static_features=NUM_STATIC_FEATURES,
    ...     **gnn_kwargs
    ... )
    GNNModelAdforce initialized: 5 static, 9 forcing (x3), 3 state features. Total input: 17. Output: 3.
    GNNModelAdforce building internal GNN of type: GCN
    >>>
    >>> # 6. Run forward pass
    >>> out = model(batch)
    >>>
    >>> # 7. Check output shape
    >>> print(f"Output shape: {out.shape}")
    Output shape: torch.Size([10, 3])
    """

    def __init__(
        self,
        num_node_features,
        num_edge_features,
        previous_t,
        num_output_features,  # This is 3
        num_static_features=5,
        **kwargs,  # Catches all other config args
    ):

        # # --- Split kwargs for AdforceBaseModel ---
        # base_model_kwargs = {}
        # base_keys = [
        #     "learned_residuals",
        #     "seed",
        #     "residuals_base",
        #     "residual_init",
        #     "device",
        # ]
        # for k in list(kwargs.keys()):
        #     if k in base_keys:
        #         base_model_kwargs[k] = kwargs.pop(k)

        # base_model_kwargs["previous_t"] = previous_t
        # base_model_kwargs["num_output_vars"] = num_output_features

        # # Call the parent __init__ (from base.py)
        # super().__init__(**base_model_kwargs)
        # commented out base model init for now
        super().__init__()
        # # --- END ---

        self.previous_t = previous_t
        self.num_output_features = num_output_features
        self.num_static_features = num_static_features

        # --- Feature calculation ---
        self.num_dynamic_forcing_features_per_step = 3
        self.num_dynamic_state_features = 3
        num_forcing_features = (
            self.num_dynamic_forcing_features_per_step * self.previous_t
        )
        self.dynamic_vars = num_node_features - self.num_static_features
        expected_dynamic_vars = num_forcing_features + self.num_dynamic_state_features

        if self.dynamic_vars != expected_dynamic_vars:
            raise ValueError(
                f"Feature mismatch! Total features {num_node_features} - "
                f"static features {self.num_static_features} = {self.dynamic_vars} dynamic features. "
                f"But expected {expected_dynamic_vars} (forcing={num_forcing_features} + state={self.num_dynamic_state_features}). "
                f"Check num_static_features in your config."
            )

        if "hid_features" in kwargs:
            print(
                f"GNNModelAdforce initialized: {self.num_static_features} static, "
                f"{num_forcing_features} forcing (x{self.previous_t}), "
                f"{self.num_dynamic_state_features} state features. "
                f"Total input: {num_node_features}. Output: {self.num_output_features}."
            )

        self.in_features = num_node_features  # Total features

        # --- GNN Switch Logic ---
        self.type_GNN = kwargs.pop("type_gnn", "GCN").upper()
        print(f"GNNModelAdforce building internal GNN of type: {self.type_GNN}")

        if self.type_GNN == "SWEGNN":
            # Build the SWEGNN "Inner Box" from processors.py
            self.gnn = SWEGNN_Adforce(
                in_features_static=self.num_static_features,
                in_features_dynamic=self.dynamic_vars,
                in_features_edge=num_edge_features,
                num_output_features=self.num_output_features,
                **kwargs,  # Pass all remaining GNN args
            )
        else:
            # Build the standard GNN "Inner Box" from processors.py
            self.gnn = GNN_Adforce(
                in_features=self.in_features,
                num_output_features=self.num_output_features,
                type_gnn=self.type_GNN,
                **kwargs,  # Pass all remaining GNN args
            )

    def forward(self, batch):
        x = batch.x
        x0_input = x.clone()

        static_features = x[:, : self.num_static_features]
        dynamic_features = x[:, self.num_static_features :]

        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        # 1. Get the "delta" prediction from the inner GNN
        # out_delta = self.gnn(
        #     static_features, dynamic_features, edge_index, edge_attr, batch=batch
        # )
        out = self.gnn(
             static_features, dynamic_features, edge_index, edge_attr, batch=batch
        )

        # 2. Add residual connection
        # out = out_delta + self._add_residual_connection(x0_input)

        # 3. Apply activation (matches old gnn.py logic)
        # out = torch.relu(out)

        # 4. Apply masking
        # out = self._mask_small_WD(out, epsilon=0.0001)

        out = out.reshape(-1, self.num_output_features)
        return out


if __name__ == "__main__":
    """
    Run doctests for this module.

    From the command line (e.g., from the root sdat2/mswe-gnn/mSWE-GNN-sdat2/ dir):
    python -m mswegnn.models.models
    """
    import doctest
    # We need to import the test dependencies for the doctests to run
    from torch_geometric.data import Data
    
    doctest.testmod(verbose=True)
    print("Doctests for mswegnn.models.models complete.")