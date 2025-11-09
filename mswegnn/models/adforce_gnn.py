# models/gnn_new.py
# (This is a new file, replacing models/gnn.py)
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
from mswegnn.models.adforce_helpers import make_mlp

# --- Helper classes (MLP, GNN_Layer) are unchanged ---
# ... (Copy the full MLP, GNN_Layer, and MSGNN_Layer classes from models/gnn.py here) ...
# ... (No changes are needed in those helper classes) ...


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hid_features,
        mlp_layers,
        activation="relu",
        edge_mlp=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hid_features = hid_features

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("Activation type not implemented")

        self.mlp_layers = mlp_layers
        self.edge_mlp = edge_mlp

        assert mlp_layers >= 1, "Number of MLP layers must be at least 1"

        if mlp_layers == 1:
            self.layers = self._get_layer(in_features, out_features)
        else:
            self.layers = nn.ModuleList()
            self.layers.append(self._get_layer(in_features, hid_features))
            for _ in range(mlp_layers - 2):
                self.layers.append(self._get_layer(hid_features, hid_features))
            self.layers.append(self._get_layer(hid_features, out_features))

    def _get_layer(self, in_features, out_features):
        layer = nn.Sequential(nn.Linear(in_features, out_features), self.activation)
        return layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GNN_Layer(nn.Module):
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


class MSGNN_Layer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        gnn_activation,
        type_gnn="GCN",
        with_filter_matrix=False,
        K=3,
    ):
        super().__init__()
        self.conv = GNN_Layer(
            in_features, out_features, gnn_activation, type_gnn=type_gnn
        )
        self.with_filter_matrix = with_filter_matrix
        self.K = K

        if self.with_filter_matrix:
            self.filter_matrix = nn.Parameter(torch.ones(self.K))

    def forward(self, x, edge_index, edge_attr=None, intra_mesh_edge_index=None):
        if self.with_filter_matrix:
            x_conv = []
            x_conv.append(self.conv(x, edge_index, edge_attr=edge_attr))
            for k in range(1, self.K):
                x_conv.append(self.conv(x_conv[-1], edge_index, edge_attr=edge_attr))

            x_conv = torch.stack(x_conv, dim=-1)
            x_conv = torch.matmul(x_conv, self.filter_matrix)

        else:
            x_conv = self.conv(x, edge_index, edge_attr=edge_attr)

        return x_conv


# --- Main GNN Classes (Refactored) ---


class GNN_Adforce(nn.Module):
    """
    Refactored GNN class.

    The `out_features` argument is no longer hardcoded to 2.
    It must be passed in as `num_output_features`.
    """

    def __init__(
        self,
        in_features,
        hid_features,
        num_output_features,  # <-- CHANGED: Was 'out_features=2'
        mlp_layers,
        gnn_activation="tanh",
        mlp_activation="prelu",
        type_gnn="GCN",
        with_filter_matrix=False,
        K=3,
        **kwargs,
    ):  # Added **kwargs to catch unused args
        super().__init__()

        # --- CHANGED: Parameterized output features ---
        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = num_output_features  # <-- RENAMED/PARAMETERIZED
        # --- END CHANGE ---

        self.gnn = GNN_Layer(
            in_features, hid_features, gnn_activation, type_gnn=type_gnn
        )

        self.decoder = MLP(
            hid_features, self.out_features, hid_features, mlp_layers, mlp_activation
        )

    def forward(
        self, static_features, dynamic_features, edge_index, edge_attr, **kwargs
    ):
        x = torch.cat([static_features, dynamic_features], -1)

        x = self.gnn(x, edge_index, edge_attr)
        x = self.decoder(x)

        return x


class MSGNN_Adforce(nn.Module):
    """
    Refactored MSGNN class.

    The `out_features` argument is no longer hardcoded to 2.
    It must be passed in as `num_output_features`.
    """

    def __init__(
        self,
        in_features,
        hid_features,
        num_output_features,  # <-- CHANGED: Was 'out_features=2'
        mlp_layers,
        num_scales,
        gnn_activation="tanh",
        mlp_activation="prelu",
        type_gnn="GCN",
        with_filter_matrix=False,
        K=3,
        learned_pooling=False,
        skip_connections=True,
        **kwargs,
    ):  # Added **kwargs to catch unused args
        super().__init__()

        # --- CHANGED: Parameterized output features ---
        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = num_output_features  # <-- RENAMED/PARAMETERIZED
        # --- END CHANGE ---

        self.num_scales = num_scales
        self.learned_pooling = learned_pooling
        self.skip_connections = skip_connections

        self.encoder = MLP(
            in_features, hid_features, hid_features, mlp_layers, mlp_activation
        )

        self.gnn_layers = nn.ModuleList(
            [
                MSGNN_Layer(
                    hid_features,
                    hid_features,
                    gnn_activation,
                    type_gnn=type_gnn,
                    with_filter_matrix=with_filter_matrix,
                    K=K,
                )
                for _ in range(num_scales)
            ]
        )

        self.decoder = MLP(
            hid_features, self.out_features, hid_features, mlp_layers, mlp_activation
        )

        if self.learned_pooling:
            self.pooling_layers = nn.ModuleList(
                [nn.Linear(hid_features, hid_features) for _ in range(num_scales - 1)]
            )

    def forward(
        self, static_features, dynamic_features, edge_index, edge_attr, batch, **kwargs
    ):
        x = torch.cat([static_features, dynamic_features], -1)
        x = self.encoder(x)

        x_scales = self._create_scale_features(x, batch)

        x_scales_out = []
        for i in range(self.num_scales):
            x_conv = self.gnn_layers[i](
                x_scales[i],
                edge_index[i],
                edge_attr=edge_attr[i],
                intra_mesh_edge_index=None,
            )
            x_scales_out.append(x_conv)

        x_out = self._pool(x_scales_out, batch)

        x_out = self.decoder(x_out)

        return x_out

    def _pool(self, x_scales, batch):
        # ... (Copy the _pool and _create_scale_features methods from models/gnn.py here) ...
        # ... (No changes are needed in those helper methods) ...
        finest_scale = x_scales[0]
        if self.skip_connections:
            for i in range(self.num_scales - 1):
                if self.learned_pooling:
                    x_pool = self.pooling_layers[i](x_scales[i + 1])
                else:
                    x_pool = x_scales[i + 1]

                finest_scale = finest_scale + scatter(
                    x_pool,
                    batch.node_ptr[
                        batch.intra_edge_ptr[i] : batch.intra_edge_ptr[i + 1], 1
                    ],
                    dim=0,
                    reduce="mean",
                )
        return finest_scale

    def _create_scale_features(self, x, batch):
        if isinstance(batch, Batch):
            x_scales = [
                x[batch.node_ptr[i, 0] : batch.node_ptr[i, -1]]
                for i in range(batch.num_graphs)
            ]
            x_scales = [
                torch.cat(
                    [
                        x_graph[batch.node_ptr[i, j] : batch.node_ptr[i, j + 1]]
                        for i in range(batch.num_graphs)
                    ]
                )
                for j in range(self.num_scales)
            ]
        else:
            x_scales = [
                x[batch.node_ptr[i] : batch.node_ptr[i + 1]]
                for i in range(self.num_scales)
            ]
        return x_scales


class SWEGNN(nn.Module):
    r"""
    Shallow Water Equations (SWE) inspired Graph Neural Network layer.

    This layer performs ``K`` rounds of message passing, where the update
    rule is physically-informed by a numerical discretization of the
    Shallow Water Equations.

    The core idea is to learn a message (``edge_message``) for each edge,
    which acts like a 'conductivity' or 'flux coefficient'. The node
    update is then driven by a 'flux' calculated using this message.

    Attributes:
        with_gradient (bool): If ``True``, uses a physics-inspired update
            based on the *gradient* between node states ($H_j - H_i$). This
            models pressure gradient-driven flow.
        upwind_mode (bool): If ``True`` (and ``with_gradient=True``),
            applies an upwinding scheme (flux = max(0, $H_j - H_i$) * msg),
            a common technique in fluid dynamics for numerical stability,
            ensuring flow only moves down-gradient.
        with_filter_matrix (bool): If ``True``, applies a learnable
            linear transformation (filter) to the node states at the
            beginning ($W_0$) and after each aggregation step ($W_k$).
        K (int): The number of message-passing iterations (time steps).
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
        device: str = "cpu",
        **mlp_kwargs,
    ):
        """
        Args:
            static_node_features (int): Dimension of static node features.
            dynamic_node_features (int): Dimension of dynamic node features.
                                         This is also the output dimension.
            edge_features (int): Dimension of edge features.
            K (int, optional): Number of message passing iterations.
                               Defaults to 2.
            normalize (bool, optional): If ``True``, L2-normalize the
                                      computed edge message. Defaults to True.
            with_filter_matrix (bool, optional): If ``True``, add learnable
                                               filter matrices W_k.
                                               Defaults to True.
            with_gradient (bool, optional): If ``True``, use the
                                          physics-inspired gradient update.
                                          Defaults to True.
            upwind_mode (bool, optional): If ``True``, use upwinding.
                                        Only active if `with_gradient=True`.
                                        Defaults to False.
            device (str, optional): Device to place tensors on.
                                    Defaults to "cpu".
            **mlp_kwargs: Additional keyword arguments passed to the
                          `make_mlp` factory function (e.g., `mlp_layers`, `activation`).
        """
        super().__init__()
        self.edge_features = edge_features

        # The edge MLP input is the concatenation of:
        # [src_static, dst_static, src_dynamic, dst_dynamic, edge_features]
        self.edge_input_size = (
            edge_features + static_node_features * 2 + dynamic_node_features * 2
        )
        # The edge MLP output has the same dim as the dynamic features
        self.edge_output_size = dynamic_node_features
        hidden_size = self.edge_output_size * 2

        self.normalize = normalize
        self.K = K
        self.with_filter_matrix = with_filter_matrix
        self.device = device
        self.with_gradient = with_gradient
        self.upwind_mode = upwind_mode

        # --- FIX 1: Rename 'mlp_layers' to 'n_layers' for make_mlp ---
        n_layers = mlp_kwargs.pop("mlp_layers", 2)
        mlp_kwargs["n_layers"] = n_layers

        # --- FIX 2: Remove 'edge_mlp' from kwargs ---
        # This argument is used by the *wrapper* (SWEGNN_Adforce),
        # not by the internal `make_mlp` function.
        mlp_kwargs.pop("edge_mlp", None)
        # --- END FIXES ---

        # This MLP learns the edge-wise message, m_ij
        self.edge_mlp = make_mlp(
            self.edge_input_size,
            self.edge_output_size,
            hidden_size=hidden_size,
            # device=device,
            **mlp_kwargs,  # Now contains 'n_layers' and NOT 'edge_mlp'
        )

        if with_filter_matrix:
            # Create K+1 filter matrices:
            # W_0 (for initial state) and W_1...W_K (for aggregated flux)
            self.filter_matrix = torch.nn.ModuleList(
                [
                    nn.Linear(
                        dynamic_node_features,
                        dynamic_node_features,
                        bias=False,
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
        """
        Performs the forward pass of the SWEGNN layer.

        Args:
            static_nodes (Tensor): Static node features (e.g., elevation,
                                   node type) of shape [num_nodes,
                                   static_node_features].
            dynamic_nodes (Tensor): Dynamic node features (e.g., water
                                    height, velocity) of shape [num_nodes,
                                    dynamic_node_features]. This is the
                                    state that will be updated.
            edge_index (Tensor): Graph connectivity in COO format
                                 of shape [2, num_edges].
            edge_features (Optional[Tensor], optional): Edge features
                                                       (e.g., length,
                                                       slope) of shape
                                                       [num_edges,
                                                       edge_features].
                                                       Defaults to None.

        Returns:
            Tensor: The updated dynamic node features (state) after K
                    iterations, of shape [num_nodes,
                    dynamic_node_features].
        """
        row, col = edge_index[0], edge_index[1]
        num_nodes = dynamic_nodes.size(0)

        # --- 1. Initialize Evolving Node State (H) ---
        # 'h' represents the evolving state (e.g., water height, momentum)
        if self.with_filter_matrix:
            # h_0 = W_0 * H_initial
            h = self.filter_matrix[0](dynamic_nodes.clone())
        else:
            h = dynamic_nodes.clone()

        # --- 2. Perform K Iterations (Simulation Steps) ---
        for k in range(self.K):

            # --- 3. Edge Masking ---
            # Filter out 'inactive' nodes (where all dynamic features are 0)
            # This might represent 'dry' nodes in a simulation.
            # An edge (i, j) is kept if *either* i or j is active.
            mask_active_nodes = h.sum(dim=1) != 0
            mask_row = mask_active_nodes[row]
            mask_col = mask_active_nodes[col]
            edge_mask = mask_row | mask_col  # logical OR

            # Apply mask to edge indices
            row_masked, col_masked = row[edge_mask], col[edge_mask]

            # --- 4. Edge Message Calculation (Learnable Conductivity) ---

            # Concatenate features for the MLP:
            # [s_i, s_j, d_i, d_j] for all active edges (i,j)
            # Note: The *original* dynamic features (dynamic_nodes) are used
            # as input to the MLP, not the evolving state 'h'.
            edge_input_concat = torch.cat(
                [
                    static_nodes[row_masked],
                    static_nodes[col_masked],
                    dynamic_nodes[row_masked],
                    dynamic_nodes[col_masked],
                ],
                dim=1,
            )

            # Add edge features if they exist
            if self.edge_features > 0 and edge_features is not None:
                edge_input_concat = torch.cat(
                    [edge_input_concat, edge_features[edge_mask]], dim=1
                )

            # m_ij = MLP(features)
            # This message acts as a learnable conductivity coefficient
            edge_message = self.edge_mlp(edge_input_concat)

            if self.normalize:
                # m_ij = m_ij / ||m_ij||
                norm = vector_norm(edge_message, dim=1, keepdim=True)
                edge_message = edge_message / norm
                edge_message.masked_fill_(torch.isnan(edge_message), 0)

            # --- 5. Flux Calculation (The SWE-inspired step) ---

            # This is where the physics logic is applied, using the
            # *evolving* state 'h'.

            if self.with_gradient:
                # Physics-Inspired: Flux is proportional to the gradient
                # F_ij = m_ij * (H_j - H_i)

                # Gradient: (H_j - H_i)
                hydraulic_gradient = h[col_masked] - h[row_masked]

                if self.upwind_mode:
                    # Upwinding: F_ij = m_ij * max(0, H_j - H_i)
                    # Ensures flow only moves from high (j) to low (i)
                    hydraulic_gradient.clamp_(min=0)

                flux = hydraulic_gradient * edge_message

            else:
                # Standard GNN: Message is scaled by the source node state
                # F_ij = m_ij * H_i
                flux = edge_message * h[row_masked]

            # --- 6. Aggregation (Summing Fluxes) ---
            # Each node 'j' (represented by 'col_masked') sums all
            # incoming fluxes. This scatter sum acts as a discrete
            # divergence, calculating the net flow into each node.
            aggregated_flux = scatter(
                src=flux,
                index=col_masked,
                reduce="sum",
                dim=0,
                dim_size=num_nodes,
            )

            # --- 7. Node State Update (Euler Time Step) ---
            if self.with_filter_matrix:
                # Apply filter W_{k+1} to the aggregated flux
                aggregated_flux = self.filter_matrix[k + 1](aggregated_flux)

            # H^{k+1} = H^k + Aggregated_Flux
            # This is a discrete forward Euler integration step:
            # H_new = H_old + dt * F_net
            # (where dt is implicitly 1 or learned in the MLP/filters)
            h = h + aggregated_flux

        # --- 8. Return Final State ---
        return h
