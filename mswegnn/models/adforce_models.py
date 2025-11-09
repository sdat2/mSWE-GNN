# mswegnn/models/adforce_models.py
# (This is a new file, replacing mswegnn/models/models.py)
from typing import Union, Optional
import torch
from torch import Tensor
import torch.nn as nn

# from torch_geometric.data.batch import Batch
from torch_geometric.data import Data, Batch  # <-- Import Data for doctest
from mswegnn.models.adforce_gnn import GNN_Adforce, SWEGNN, MLP


class MonolithicMLPModel(nn.Module):
    """
    A "monolithic" MLP baseline that flattens the *entire* graph's
    node features into a single vector.

    This model is *not* graph-structured, has no weight sharing
    between nodes, and its parameters scale linearly with the
    number of nodes (N_nodes).

    It is fundamentally non-extensible to graphs of different sizes.
    It serves as a baseline to test data efficiency against a
    brute-force function approximator.

    Args:
        n_nodes (int): The *fixed* number of nodes in the graph.
                       This model cannot handle variable-sized graphs.
        num_node_features (int): Number of input features *per node*.
        num_output_features (int): Number of output features *per node*.
        hid_features (int): Width of the hidden layers.
        mlp_layers (int): Number of *hidden* layers.
        mlp_activation (str): Activation function to use (e.g., 'relu').
        **kwargs: Catches unused arguments from the config (like
                  'model_type', 'previous_t') so initialization doesn't fail.

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
    >>> # This data object *must* have N_NODES_FIXED nodes
    >>> mock_x = torch.rand(N_NODES_FIXED, NUM_IN_FEATURES)
    >>> mock_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    >>> data = Data(x=mock_x, edge_index=mock_edge_index)
    >>> # data.num_graphs will be 1
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
    >>> # Expected shape is [B * N, F_out] = [10, 3]
    >>> print(f"Output shape: {out.shape}")
    Output shape: torch.Size([10, 3])
    >>>
    >>> # 6. Test failure case (graph with wrong number of nodes)
    >>> mock_x_wrong_size = torch.rand(N_NODES_FIXED + 1, NUM_IN_FEATURES)
    >>> data_wrong = Data(x=mock_x_wrong_size, edge_index=mock_edge_index)
    >>>
    >>> try:
    ...     model(data_wrong)
    ... except ValueError as e:
    ...     print(f"Caught expected error: Input shape mismatch...")
    Caught expected error: Input shape mismatch...
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

        # Calculate the dimensions of the flattened vectors
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
        # Input layer
        layers.append(nn.Linear(self.flat_input_dim, hid_features))
        layers.append(activation)

        # Hidden layers
        for _ in range(mlp_layers - 1):
            layers.append(nn.Linear(hid_features, hid_features))
            layers.append(activation)

        # Output layer
        layers.append(nn.Linear(hid_features, self.flat_output_dim))

        self.network = nn.Sequential(*layers)

        # This print is helpful for verifying config
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
        """
        Forward pass.
        Assumes batch contains one or more graphs, each with *exactly*
        self.n_nodes.

        Args:
            batch (torch_geometric.data.Batch or torch_geometric.data.Data):
                The input batch object.
                Assumes batch.x is [B * N, F_in] where N=self.n_nodes.

        Returns:
            torch.Tensor: The model predictions, shape [B * N, F_out].
        """
        # x shape is [B * N, F_in], where N = self.n_nodes
        x = batch.x

        # 1. Get batch size (B).
        # This works for both a single Data object (B=1) and a Batch (B > 1).
        batch_size = getattr(batch, "num_graphs", 1)

        # 2. Validate input shape
        # Check that the total nodes in x matches B * N
        if x.shape[0] != batch_size * self.n_nodes:
            raise ValueError(
                f"Input shape mismatch for MonolithicMLP.\n"
                f"Model configured for n_nodes={self.n_nodes} (total nodes expected in batch: {batch_size * self.n_nodes}).\n"
                f"Received x shape {x.shape} (total nodes: {x.shape[0]}).\n"
                f"This model requires all graphs in the dataset to have *exactly* {self.n_nodes} nodes."
            )

        # 3. Reshape and Flatten
        # [B * N, F_in] -> [B, N, F_in]
        x_reshaped = x.view(batch_size, self.n_nodes, self.in_features_per_node)

        # [B, N, F_in] -> [B, N * F_in]
        x_flat = x_reshaped.view(batch_size, -1)

        # 4. Pass through MLP
        # [B, N * F_in] -> [B, N * F_out]
        out_flat = self.network(x_flat)

        # 5. Reshape output
        # [B, N * F_out] -> [B, N, F_out]
        out_reshaped = out_flat.view(
            batch_size, self.n_nodes, self.out_features_per_node
        )

        # 6. Reshape back to PyG format for the loss function
        # [B, N, F_out] -> [B * N, F_out]
        out = out_reshaped.view(-1, self.out_features_per_node)

        return out


class PointwiseMLPModel(nn.Module):
    """
    A standalone MLP model wrapper.

    This model applies a simple MLP to the node features and ignores all
    graph/edge information (e.g., edge_index, edge_attr). It serves as a
    baseline to compare against graph-based models.

    Args:
        num_node_features (int): Total number of features in the input `x` tensor.
        num_output_features (int): Number of output features to predict.
        **mlp_kwargs: Additional keyword arguments passed to the MLP constructor
            (e.g., hid_features, mlp_layers). Expects 'mlp_activation'
            for activation type.

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
    >>> # Note: edge_index is ignored by this model, but part of the batch
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

        # --- FIX ---
        # Get the 'mlp_activation' value from the kwargs, default to 'relu'
        # (matching the default in the MLP helper class).
        # This maps 'mlp_activation' (from config) to 'activation' (for MLP class).
        activation_type = mlp_kwargs.pop("mlp_activation", "relu")

        # Remove keys that GNN/MSGNN models use but the standalone MLP doesn't
        # to avoid passing them to the MLP constructor.
        mlp_kwargs.pop("model_type", None)  # <-- THIS IS THE NEW LINE
        mlp_kwargs.pop("previous_t", None)
        mlp_kwargs.pop("num_static_features", None)
        mlp_kwargs.pop("num_edge_features", None)
        mlp_kwargs.pop("num_scales", None)
        mlp_kwargs.pop("learned_pooling", None)
        mlp_kwargs.pop("skip_connections", None)
        mlp_kwargs.pop("gnn_activation", None)  # Used by GNN, not MLP
        mlp_kwargs.pop("type_gnn", None)  # Used by GNN, not MLP

        # Instantiate the MLP from adforce_gnn.py
        self.mlp = MLP(
            in_features=self.in_features,
            out_features=self.out_features,
            activation=activation_type,  # <-- Pass with the correct key 'activation'
            **mlp_kwargs,  # Pass remaining (hid_features, mlp_layers)
        )

        if "hid_features" in mlp_kwargs:  # Avoid printing during doctest
            print(
                f"PointwiseMLPModel initialized: {self.in_features} input features, "
                f"{self.out_features} output features."
            )

    def forward(self, batch):
        """
        Forward pass. Ignores all graph structure.

        Args:
            batch (torch_geometric.data.Batch): The input batch object.

        Returns:
            torch.Tensor: The model predictions, shape [num_nodes, num_output_features].
        """
        # x shape is [num_nodes_in_batch, num_node_features]
        x = batch.x

        # Apply the MLP directly to the node features
        out = self.mlp(x)  # shape [num_nodes_in_batch, num_output_features]

        return out


class GNNModelAdforce(nn.Module):
    """
    Refactored GNNModel wrapper.

    --- MODIFIED FOR DELTA LEARNING ---
    This model's __init__ method is updated to validate the new input structure:
    x = (static features, forcing features, current_state features)
    e.g., 5 static + (3 forcing * p_t steps) + 3 state

    Args:
        num_node_features (int): Total number of features in the input `x` tensor.
        num_edge_features (int): Total number of features in the `edge_attr` tensor.
        previous_t (int): Number of history steps included in the input.
        num_output_features (int): Number of output features to predict.
        num_static_features (int, optional): Number of static features at the
            start of the `x` tensor. Defaults to 5.
        **gnn_kwargs: Additional keyword arguments passed to the GNN_Adforce constructor.

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
    ...     'type_gnn': 'GCN'
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
    >>>
    >>> # 6. Run forward pass
    >>> out = model(batch)
    >>>
    >>> # 7. Check output shape
    >>> print(f"Output shape: {out.shape}")
    Output shape: torch.Size([10, 3])
    >>>
    >>> # 8. Test error case: mismatched features
    >>> try:
    ...     model_fail = GNNModelAdforce(
    ...         num_node_features=16, # 16 is not 5 + (3*3) + 3
    ...         num_edge_features=2,
    ...         previous_t=PREVIOUS_T,
    ...         num_output_features=NUM_OUTPUT_FEATURES,
    ...         num_static_features=NUM_STATIC_FEATURES,
    ...         **gnn_kwargs
    ...     )
    ... except ValueError as e:
    ...     print(f"Caught expected error: {e}")
    Caught expected error: Feature mismatch! Total features 16 - static features 5 = 11 dynamic features. But expected 12 (forcing=9 + state=3). Check num_static_features in your config and the num_node_features calculation in adforce_main.py.
    """

    def __init__(
        self,
        num_node_features,
        num_edge_features,
        previous_t,
        num_output_features,
        num_static_features=5,  # From adforce_dataset (DEM, sx, sy, area, type)
        **gnn_kwargs,
    ):
        super().__init__()

        self.previous_t = previous_t
        self.num_output_features = num_output_features
        self.num_static_features = num_static_features

        # --- NEW: Explicit feature calculation for validation ---
        # 3 dynamic *forcing* vars (WX, WY, P)
        self.num_dynamic_forcing_features_per_step = 3
        # 3 dynamic *state* vars (WD, VX, VY) - this is the new part
        self.num_dynamic_state_features = 3

        # Calculate expected feature counts
        num_forcing_features = (
            self.num_dynamic_forcing_features_per_step * self.previous_t
        )

        # This is the total number of "dynamic" features, i.e.,
        # everything *except* the static features.
        self.dynamic_vars = num_node_features - self.num_static_features

        # Calculate what the expected dynamic vars should be
        expected_dynamic_vars = num_forcing_features + self.num_dynamic_state_features

        if self.dynamic_vars != expected_dynamic_vars:
            raise ValueError(
                f"Feature mismatch! Total features {num_node_features} - "
                f"static features {self.num_static_features} = {self.dynamic_vars} dynamic features. "
                f"But expected {expected_dynamic_vars} (forcing={num_forcing_features} + state={self.num_dynamic_state_features}). "
                f"Check num_static_features in your config and the num_node_features calculation in adforce_main.py."
            )

        # This print is useful for debugging and for the doctest
        if "hid_features" in gnn_kwargs:
            # Avoid printing during doctest's error check
            print(
                f"GNNModelAdforce initialized: {self.num_static_features} static, "
                f"{num_forcing_features} forcing (x{self.previous_t}), "
                f"{self.num_dynamic_state_features} state features. "
                f"Total input: {num_node_features}. Output: {self.num_output_features}."
            )
        # --- END NEW ---

        self.in_features = num_node_features

        self.gnn = GNN_Adforce(
            in_features=self.in_features,
            num_output_features=self.num_output_features,  # <-- Pass new arg
            **gnn_kwargs,
        )

    def forward(self, batch):
        """
        The GNN_Adforce class's forward just concatenates static and dynamic,
        so we feed all features (forcing + state) as 'dynamic_features'.
        """
        x = batch.x

        static_features = x[:, : self.num_static_features]
        dynamic_features = x[:, self.num_static_features :]

        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        out = self.gnn(
            static_features, dynamic_features, edge_index, edge_attr, batch=batch
        )

        out = out.reshape(-1, self.num_output_features)

        return out

# In mswegnn/models/adforce_models.py

class SWEGNN_Adforce(nn.Module):
    """
    Wrapper for the SWEGNN layer to match the Adforce pipeline.

    This class replicates the encoder-processor-decoder structure
    from the original gnn.py. It takes raw static and dynamic features,
    encodes them, processes them with SWEGNN, and decodes them.

    Args:
        in_features_static (int): Number of raw static node features.
        in_features_dynamic (int): Number of raw dynamic node features
            (forcing + state).
        in_features_edge (int): Number of raw edge features.
        hid_features (int): The hidden dimension for encoders, GNN,
            and decoder.
        num_output_features (int): The final number of output features
            (e.g., 3 for delta_WD, delta_VX, delta_VY).
        mlp_layers (int): Number of layers to use in the encoder/decoder MLPs.
        mlp_activation (str, optional): Activation for MLPs. Defaults to "prelu".
        gnn_activation (str, optional): Activation *after* the GNN layer.
            Defaults to "tanh".
        **gnn_kwargs: Additional arguments passed to the SWEGNN layer,
            such as `K`, `normalize`, `with_gradient`, `edge_mlp`, etc.

    Doctest:
    >>> import torch
    >>> from torch_geometric.data import Data
    >>>
    >>> # 1. Define test parameters
    >>> N_NODES = 10
    >>> IN_FEAT_STATIC = 5
    >>> IN_FEAT_DYNAMIC = 12  # (3 forcing * 3 steps) + 3 state
    >>> IN_FEAT_EDGE = 2
    >>> NUM_OUTPUT = 3
    >>> HID_FEAT = 16
    >>>
    >>> # 2. Create mock tensors
    >>> mock_static_x = torch.rand(N_NODES, IN_FEAT_STATIC)
    >>> mock_dynamic_x = torch.rand(N_NODES, IN_FEAT_DYNAMIC)
    >>> mock_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    >>> mock_edge_attr = torch.rand(mock_edge_index.shape[1], IN_FEAT_EDGE)
    >>>
    >>> # 3. Instantiate the model
    >>> # We pass SWEGNN-specific args like K and edge_mlp via kwargs
    >>> model = SWEGNN_Adforce(
    ...     in_features_static=IN_FEAT_STATIC,
    ...     in_features_dynamic=IN_FEAT_DYNAMIC,
    ...     in_features_edge=IN_FEAT_EDGE,
    ...     hid_features=HID_FEAT,
    ...     num_output_features=NUM_OUTPUT,
    ...     mlp_layers=2,
    ...     mlp_activation='relu',
    ...     gnn_activation='tanh',
    ...     K=2,
    ...     edge_mlp=True,
    ...     normalize=True,
    ...     with_gradient=True
    ... )
    >>>
    >>> # 4. Run forward pass
    >>> out = model(
    ...     static_features=mock_static_x,
    ...     dynamic_features=mock_dynamic_x,
    ...     edge_index=mock_edge_index,
    ...     edge_attr=mock_edge_attr
    ... )
    >>>
    >>> # 5. Check output shape
    >>> print(f"Output shape: {out.shape}")
    Output shape: torch.Size([10, 3])
    """
    def __init__(
        self,
        in_features_static: int,
        in_features_dynamic: int,
        in_features_edge: int,
        hid_features: int,
        num_output_features: int,
        mlp_layers: int,
        mlp_activation: str = "prelu",
        gnn_activation: str = "tanh",
        type_gnn: str = "SWEGNN",     # Catches 'type_gnn' from config
        **gnn_kwargs,                 # 'model_type' will land in here
    ):
        super().__init__()
        
        self.hid_features = hid_features

        # 1. Encoders
        self.static_node_encoder = MLP(
            in_features=in_features_static,
            out_features=hid_features,
            hid_features=hid_features,
            mlp_layers=mlp_layers,
            activation=mlp_activation,
        )
        self.dynamic_node_encoder = MLP(
            in_features=in_features_dynamic,
            out_features=hid_features,
            hid_features=hid_features,
            mlp_layers=mlp_layers,
            activation=mlp_activation,
        )
        
        # 2. Optional Edge Encoder
        self.edge_mlp_flag = gnn_kwargs.get("edge_mlp", True)
        self.num_edge_features_for_gnn = in_features_edge
        
        if self.edge_mlp_flag:
            self.num_edge_features_for_gnn = hid_features
            self.edge_encoder = MLP(
                in_features=in_features_edge,
                out_features=hid_features,
                hid_features=hid_features,
                mlp_layers=mlp_layers,
                activation=mlp_activation,
            )

        # --- FIX: Remove keys from kwargs that SWEGNN does not need ---
        # 'model_type' is used by adforce_main.py for logic, not the model.
        # 'type_gnn' is used by GNNModelAdforce, but not SWEGNN_Adforce.
        gnn_kwargs.pop('model_type', None)
        gnn_kwargs.pop('type_gnn', None) # Pop this too just in case
        # --- END FIX ---

        # 3. GNN Processor (The SWEGNN layer itself)
        self.gnn = SWEGNN(
            static_node_features=hid_features,
            dynamic_node_features=hid_features,
            edge_features=self.num_edge_features_for_gnn,
            mlp_layers=mlp_layers,
            activation=mlp_activation,
            **gnn_kwargs # Now 'model_type' is removed
        )

        # 4. GNN Activation
        if gnn_activation == "relu":
            self.gnn_activation = nn.ReLU()
        elif gnn_activation == "prelu":
            self.gnn_activation = nn.PReLU()
        elif gnn_activation == "tanh":
            self.gnn_activation = nn.Tanh()
        else:
            self.gnn_activation = nn.Identity() # No activation
            
        # 5. Decoder
        self.decoder = MLP(
            in_features=hid_features,
            out_features=num_output_features,
            hid_features=hid_features,
            mlp_layers=mlp_layers,
            activation=mlp_activation,
        )
    
    def forward(
        self, 
        static_features: Tensor, 
        dynamic_features: Tensor, 
        edge_index: Tensor, 
        edge_attr: Optional[Tensor], 
        **kwargs # Catches the 'batch' argument
    ) -> Tensor:
        
        # 1. Encode Nodes
        x_s = self.static_node_encoder(static_features)
        x_d = self.dynamic_node_encoder(dynamic_features)
        
        # 2. Encode Edges
        e_attr_for_gnn = edge_attr
        if self.edge_mlp_flag and edge_attr is not None:
            e_attr_for_gnn = self.edge_encoder(edge_attr)
        
        # 3. Process
        x = self.gnn(
            x_s, x_d, edge_index, edge_features=e_attr_for_gnn
        )
        
        # 4. GNN Activation
        x = self.gnn_activation(x)
            
        # 5. Decode
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    """
    Run doctests for this module.

    From the command line, run:
    python -m  mswegnn.models.adforce_models
    """
    import doctest

    doctest.testmod(verbose=True)
    print("Doctests complete.")
