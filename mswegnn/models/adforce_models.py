# models/models_new.py
# (This is a new file, replacing models/models.py)

import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data  # <-- Import Data for doctest
from mswegnn.models.adforce_gnn import GNN_new, MSGNN_new, MLP


class MLPModel_new(nn.Module):
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
    >>> NUM_IN_FEATURES = 14 # (5 static + 3 dynamic * 3 steps)
    >>> NUM_OUT_FEATURES = 3 # (WD, VX, VY)
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
    >>> model = MLPModel_new(
    ...     num_node_features=NUM_IN_FEATURES,
    ...     num_output_features=NUM_OUT_FEATURES,
    ...     **mlp_kwargs
    ... )
    MLPModel_new initialized: 14 input features, 3 output features.
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
        # --- END FIX ---

        # Remove keys that GNN/MSGNN models use but the standalone MLP doesn't
        # to avoid passing them to the MLP constructor.
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
                f"MLPModel_new initialized: {self.in_features} input features, "
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


class GNNModel_new(nn.Module):
    """
    Refactored GNNModel wrapper.

    This model no longer hardcodes the number of static or dynamic features.
    It infers them from `num_node_features` and `previous_t`.
    It also passes the new `num_output_features` to the GNN.

    Args:
        num_node_features (int): Total number of features in the input `x` tensor.
        num_edge_features (int): Total number of features in the `edge_attr` tensor.
        previous_t (int): Number of history steps included in the input.
        num_output_features (int): Number of output features to predict.
        num_static_features (int, optional): Number of static features at the
            start of the `x` tensor. Defaults to 5.
        **gnn_kwargs: Additional keyword arguments passed to the GNN_new constructor.

    Doctest:
    >>> import torch
    >>> from torch_geometric.data import Data
    >>>
    >>> # 1. Define test parameters
    >>> N_NODES = 10
    >>> PREVIOUS_T = 3
    >>> NUM_STATIC_FEATURES = 5  # (DEM, sx, sy, area, type)
    >>> NUM_OUTPUT_FEATURES = 3  # (WD, VX, VY)
    >>> NUM_DYNAMIC_IN_FEATURES = 3 # (WX, WY, P)
    >>>
    >>> # 2. Calculate total input features (5 static + 3 dynamic * 3 steps)
    >>> num_node_features = NUM_STATIC_FEATURES + (NUM_DYNAMIC_IN_FEATURES * PREVIOUS_T)
    >>> print(f"Calculated num_node_features: {num_node_features}")
    Calculated num_node_features: 14
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
    >>> model = GNNModel_new(
    ...     num_node_features=num_node_features,
    ...     num_edge_features=2, # Mock value, not used by GCN
    ...     previous_t=PREVIOUS_T,
    ...     num_output_features=NUM_OUTPUT_FEATURES,
    ...     num_static_features=NUM_STATIC_FEATURES,
    ...     **gnn_kwargs
    ... )
    GNNModel_new initialized: 5 static features, 3 dynamic input features (x3 steps), 3 output features.
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
    ...     model_fail = GNNModel_new(
    ...         num_node_features=13, # 13 is not 5 + (3*N)
    ...         num_edge_features=2,
    ...         previous_t=PREVIOUS_T,
    ...         num_output_features=NUM_OUTPUT_FEATURES,
    ...         **gnn_kwargs
    ...     )
    ... except ValueError as e:
    ...     print(f"Caught expected error: {e}")
    Caught expected error: Dynamic features (8) are not evenly divisible by previous_t (3). Check num_static_features.
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

        # --- CHANGED: Dynamic feature calculation ---
        # No more 'self.NUM_WATER_VARS = 2'
        self.num_static_features = num_static_features

        # Calculate dynamic features based on total node features
        self.dynamic_vars = num_node_features - self.num_static_features

        if self.dynamic_vars < 0:
            raise ValueError(
                "num_node_features cannot be less than num_static_features."
            )

        if self.previous_t <= 0:
            raise ValueError("previous_t must be greater than 0.")

        if self.dynamic_vars % self.previous_t != 0:
            raise ValueError(
                f"Dynamic features ({self.dynamic_vars}) are not evenly "
                f"divisible by previous_t ({self.previous_t}). "
                f"Check num_static_features."
            )

        self.num_dynamic_in_features = self.dynamic_vars // self.previous_t

        # This print is useful for debugging and for the doctest
        if "hid_features" in gnn_kwargs:  # Avoid printing during doctest's error check
            print(
                f"GNNModel_new initialized: {self.num_static_features} static features, "
                f"{self.num_dynamic_in_features} dynamic input features (x{self.previous_t} steps), "
                f"{self.num_output_features} output features."
            )
        # --- END CHANGE ---

        self.in_features = num_node_features

        self.gnn = GNN_new(
            in_features=self.in_features,
            num_output_features=self.num_output_features,  # <-- Pass new arg
            **gnn_kwargs,
        )

    def forward(self, batch):
        x = batch.x

        # --- CHANGED: Use self.num_static_features to split ---
        static_features = x[:, : self.num_static_features]
        dynamic_features = x[:, self.num_static_features :]
        # --- END CHANGE ---

        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        out = self.gnn(
            static_features, dynamic_features, edge_index, edge_attr, batch=batch
        )

        # --- CHANGED: Use self.num_output_features to reshape ---
        out = out.reshape(-1, self.num_output_features)
        # --- END CHANGE ---

        return out


class MSGNNModel_new(GNNModel_new):
    """
    Refactored MSGNNModel wrapper.

    Inherits the new dynamic feature calculations from GNNModel_new.
    Passes `num_output_features` to the `MSGNN_new` constructor.

    NOTE: This class is too complex to doctest effectively, as it
    requires a mock Batch object with `node_ptr`, `edge_ptr`, etc.
    It is best tested with a dedicated pytest file.
    """

    def __init__(
        self,
        num_node_features,
        num_edge_features,
        previous_t,
        num_output_features,
        num_static_features=5,
        **gnn_kwargs,
    ):

        # Call GNNModel_new's __init__ but *without* instantiating self.gnn
        super(GNNModel_new, self).__init__()  # Use super(GNNModel_new, ...)

        self.previous_t = previous_t
        self.num_output_features = num_output_features

        # --- CHANGED: Dynamic feature calculation (copied from parent) ---
        self.num_static_features = num_static_features
        self.dynamic_vars = num_node_features - self.num_static_features

        if self.dynamic_vars < 0:
            raise ValueError(
                "num_node_features cannot be less than num_static_features."
            )

        if self.previous_t <= 0:
            raise ValueError("previous_t must be greater than 0.")

        if self.dynamic_vars % self.previous_t != 0:
            raise ValueError(
                f"Dynamic features ({self.dynamic_vars}) not divisible by previous_t."
            )

        self.num_dynamic_in_features = self.dynamic_vars // self.previous_t

        # This print is useful for debugging
        if (
            "hid_features" in gnn_kwargs
        ):  # Avoid printing during potential doctest error checks
            print(
                f"MSGNNModel_new initialized: {self.num_static_features} static features, "
                f"{self.num_dynamic_in_features} dynamic input features (x{self.previous_t} steps), "
                f"{self.num_output_features} output features."
            )
        # --- END CHANGE ---

        self.in_features = num_node_features

        # --- CHANGED: Instantiate MSGNN_new ---
        self.gnn = MSGNN_new(
            in_features=self.in_features,
            num_output_features=self.num_output_features,  # <-- Pass new arg
            **gnn_kwargs,
        )

    def forward(self, batch):
        # This forward is identical to the parent, but is needed
        # to ensure the correct self.gnn (MSGNN_new) is called.
        x = batch.x
        static_features = x[:, : self.num_static_features]
        dynamic_features = x[:, self.num_static_features :]
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        # This will call the MSGNN_new forward method
        out = self.gnn(
            static_features, dynamic_features, edge_index, edge_attr, batch=batch
        )

        out = out.reshape(-1, self.num_output_features)
        return out


if __name__ == "__main__":
    """
    Run doctests for this module.

    From the command line, run:
    python  models/models_new.py
    """
    import doctest

    doctest.testmod(verbose=True)
    print("Doctests complete.")
