# mswegnn/models/adforce_processors.py
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
from torch.linalg import vector_norm
from torch_geometric.data.batch import Batch
from torch_geometric.utils import scatter
import numpy as np

# Import the helper, not the old MLP class
from .adforce_helpers import make_mlp, activation_functions


class GNN_Layer(nn.Module):
    """
    A simple wrapper for standard PyG GNN layers (GCN, SAGE, GIN, GAT).
    (Original from adforce_gnn.py)
    """

    def __init__(self, in_features, out_features, gnn_activation, type_gnn="GCN"):
        super().__init__()
        if type_gnn == "GCN":
            self.conv = GCNConv(in_features, out_features)
        elif type_gnn == "SAGE":
            self.conv = SAGEConv(in_features, out_features)
        elif type_gnn == "GIN":
            self.conv = GINConv(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                    nn.Linear(out_features, out_features),
                )
            )
        elif type_gnn == "GAT":
            self.conv = GATConv(in_features, out_features)
        else:
            raise ValueError("GNN type not implemented")

        if gnn_activation == "relu":
            self.activation = nn.ReLU()
        elif gnn_activation == "prelu":
            self.activation = nn.PReLU()
        elif gnn_activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("Activation type not implemented")

    def forward(self, x, edge_index, edge_attr=None):
        # GCNConv, SAGEConv, and GINConv do not accept edge_attr by default.
        # We pass only the x and edge_index.
        x = self.conv(x, edge_index)
        x = self.activation(x)
        return x


class GNN_Adforce(nn.Module):
    """
    Refactored GNN processor class.
    (Original from adforce_gnn.py)

    --- REFACTORED ---
    Now uses `make_mlp` from .helpers instead of the local MLP class.
    """

    def __init__(
        self,
        in_features,
        hid_features,
        num_output_features,
        mlp_layers,
        gnn_activation="tanh",
        mlp_activation="prelu",
        type_gnn="GCN",
        with_filter_matrix=False,
        K=3,
        **kwargs,  # Added **kwargs to catch unused args
    ):
        super().__init__()

        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = num_output_features
        self.bias = kwargs.get("bias", False)

        self.gnn = GNN_Layer(
            in_features, hid_features, gnn_activation, type_gnn=type_gnn
        )

        # --- REFACTORED ---
        # Was: MLP(hid_features, self.out_features, hid_features, mlp_layers, mlp_activation)
        self.decoder = make_mlp(
            input_size=hid_features,
            output_size=self.out_features,
            hidden_size=hid_features,
            n_layers=mlp_layers,
            activation=mlp_activation,
            bias=self.bias,
        )
        # --- END REFACTOR ---

    def forward(
        self, static_features, dynamic_features, edge_index, edge_attr, **kwargs
    ):
        x = torch.cat([static_features, dynamic_features], -1)

        x = self.gnn(x, edge_index, edge_attr)
        x = self.decoder(x)

        return x


class SWEGNN(nn.Module):
    r"""
    Shallow Water Equations (SWE) inspired Graph Neural Network layer.
    (Original from adforce_gnn.py)

    This is the core physics-inspired layer. It correctly uses `make_mlp`
    from its inception.
    """

    def __init__(
        self,
        static_node_features: int,
        dynamic_node_features: int,
        edge_features: int,
        K: int = 2,
        normalize: bool = True,
        with_filter_matrix: bool = True,
        with_gradient: bool = True,
        upwind_mode: bool = False,
        # device: str = "cpu",
        **mlp_kwargs,
    ):
        super().__init__()
        self.edge_features = edge_features
        self.edge_input_size = (
            edge_features + static_node_features * 2 + dynamic_node_features * 2
        )
        self.edge_output_size = dynamic_node_features
        hidden_size = self.edge_output_size * 2

        self.normalize = normalize
        self.K = K
        self.with_filter_matrix = with_filter_matrix
        # self.device = device
        self.with_gradient = with_gradient
        self.upwind_mode = upwind_mode

        # Rename 'mlp_layers' to 'n_layers' for make_mlp
        n_layers = mlp_kwargs.pop("mlp_layers", 2)
        mlp_kwargs["n_layers"] = n_layers
        mlp_kwargs.pop("edge_mlp", None)
        self.bias = mlp_kwargs.get("bias", False)
        mlp_kwargs['bias'] = self.bias

        self.edge_mlp = make_mlp(
            self.edge_input_size,
            self.edge_output_size,
            hidden_size=hidden_size,
            # device=device,
            # bias=self.bias,
            **mlp_kwargs,
        )

        if with_filter_matrix:
            self.filter_matrix = torch.nn.ModuleList(
                [
                    nn.Linear(
                        dynamic_node_features,
                        dynamic_node_features,
                        bias=self.bias,
                        # device=device,
                    )
                    for _ in range(K + 1)
                ]
            )

    def forward(
        self,
        static_nodes: Tensor,
        dynamic_nodes: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
    ) -> Tensor:
        row, col = edge_index[0], edge_index[1]
        num_nodes = dynamic_nodes.size(0)

        if self.with_filter_matrix:
            h = self.filter_matrix[0](dynamic_nodes.clone())
        else:
            h = dynamic_nodes.clone()

        for k in range(self.K):
            mask_active_nodes = h.sum(dim=1) != 0
            mask_row = mask_active_nodes[row]
            mask_col = mask_active_nodes[col]
            edge_mask = mask_row | mask_col
            row_masked, col_masked = row[edge_mask], col[edge_mask]

            edge_input_concat = torch.cat(
                [
                    static_nodes[row_masked],
                    static_nodes[col_masked],
                    dynamic_nodes[row_masked],
                    dynamic_nodes[col_masked],
                ],
                dim=1,
            )

            if self.edge_features > 0 and edge_features is not None:
                edge_input_concat = torch.cat(
                    [edge_input_concat, edge_features[edge_mask]], dim=1
                )

            edge_message = self.edge_mlp(edge_input_concat)

            if self.normalize:
                norm = vector_norm(edge_message, dim=1, keepdim=True)
                edge_message = edge_message / norm
                edge_message.masked_fill_(torch.isnan(edge_message), 0)

            if self.with_gradient:
                hydraulic_gradient = h[col_masked] - h[row_masked]
                if self.upwind_mode:
                    hydraulic_gradient.clamp_(min=0)
                flux = hydraulic_gradient * edge_message
            else:
                flux = edge_message * h[row_masked]

            aggregated_flux = scatter(
                src=flux,
                index=col_masked,
                reduce="sum",
                dim=0,
                dim_size=num_nodes,
            )

            if self.with_filter_matrix:
                aggregated_flux = self.filter_matrix[k + 1](aggregated_flux)

            h = h + aggregated_flux

        return h


class SWEGNN_Adforce(nn.Module):
    """
    Wrapper for the SWEGNN layer to match the Adforce pipeline.
    (Original from adforce_models.py)

    This class replicates the encoder-processor-decoder structure
    from the original gnn.py.

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
        type_gnn: str = "SWEGNN",  # Catches 'type_gnn' from config
        **gnn_kwargs,
    ):
        super().__init__()

        self.hid_features = hid_features
        self.bias = gnn_kwargs.get("bias", False)

        # 1. Encoders
        # --- REFACTORED ---
        self.static_node_encoder = make_mlp(
            input_size=in_features_static,
            output_size=hid_features,
            hidden_size=hid_features,
            n_layers=mlp_layers,
            activation=mlp_activation,
            bias=self.bias,
        )
        self.dynamic_node_encoder = make_mlp(
            input_size=in_features_dynamic,
            output_size=hid_features,
            hidden_size=hid_features,
            n_layers=mlp_layers,
            activation=mlp_activation,
            bias=self.bias,
        )
        # --- END REFACTOR ---

        # 2. Optional Edge Encoder
        self.edge_mlp_flag = gnn_kwargs.get("edge_mlp", True)
        self.num_edge_features_for_gnn = in_features_edge

        if self.edge_mlp_flag:
            self.num_edge_features_for_gnn = hid_features
            # --- REFACTORED ---
            self.edge_encoder = make_mlp(
                input_size=in_features_edge,
                output_size=hid_features,
                hidden_size=hid_features,
                n_layers=mlp_layers,
                activation=mlp_activation,
                bias=self.bias,
            )
            # --- END REFACTOR ---

        gnn_kwargs.pop("model_type", None)
        gnn_kwargs.pop("type_gnn", None)

        # 3. GNN Processor (The SWEGNN layer itself)
        self.gnn = SWEGNN(
            static_node_features=hid_features,
            dynamic_node_features=hid_features,
            edge_features=self.num_edge_features_for_gnn,
            mlp_layers=mlp_layers,
            activation=mlp_activation,
            # bias=self.bias,
            **gnn_kwargs,
        )

        # 4. GNN Activation
        self.gnn_activation = activation_functions(gnn_activation)
        if self.gnn_activation is None:
            self.gnn_activation = nn.Identity()

        # 5. Decoder
        # --- REFACTORED ---
        self.decoder = make_mlp(
            input_size=hid_features,
            output_size=num_output_features,
            hidden_size=hid_features,
            n_layers=mlp_layers,
            bias=self.bias,
            activation=mlp_activation,
        )
        # --- END REFACTOR ---

    def forward(
        self,
        static_features: Tensor,
        dynamic_features: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        **kwargs,  # Catches the 'batch' argument
    ) -> Tensor:

        # 1. Encode Nodes
        x_s = self.static_node_encoder(static_features)
        x_d = self.dynamic_node_encoder(dynamic_features)

        # 2. Encode Edges
        e_attr_for_gnn = edge_attr
        if self.edge_mlp_flag and edge_attr is not None:
            e_attr_for_gnn = self.edge_encoder(edge_attr)

        # 3. Process
        x = self.gnn(x_s, x_d, edge_index, edge_features=e_attr_for_gnn)

        # 4. GNN Activation
        x = self.gnn_activation(x)

        # 5. Decode
        x = self.decoder(x)
        return x
