# models/gnn_new.py
# (This is a new file, replacing models/gnn.py)

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
from torch_geometric.data.batch import Batch
from torch_geometric.utils import scatter
import numpy as np

# --- Helper classes (MLP, GNN_Layer) are unchanged ---
# ... (Copy the full MLP, GNN_Layer, and MSGNN_Layer classes from models/gnn.py here) ...
# ... (No changes are needed in those helper classes) ...

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hid_features, mlp_layers, activation='relu', edge_mlp=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hid_features = hid_features

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'tanh':
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
        layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            self.activation
        )
        return layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GNN_Layer(nn.Module):
    def __init__(self, in_features, out_features, gnn_activation, type_gnn='GCN'):
        super().__init__()
        if type_gnn == 'GCN':
            self.conv = GCNConv(in_features, out_features)
        elif type_gnn == 'SAGE':
            self.conv = SAGEConv(in_features, out_features)
        elif type_gnn == 'GIN':
            self.conv = GINConv(nn.Sequential(nn.Linear(in_features, out_features),
                                               nn.ReLU(), nn.Linear(out_features, out_features)))
        elif type_gnn == 'GAT':
            self.conv = GATConv(in_features, out_features)
        else:
            raise ValueError('GNN type not implemented')

        if gnn_activation == 'relu':
            self.activation = nn.ReLU()
        elif gnn_activation == 'prelu':
            self.activation = nn.PReLU()
        elif gnn_activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Activation type not implemented")

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.activation(x)
        return x

class MSGNN_Layer(nn.Module):
    def __init__(self, in_features, out_features, gnn_activation, type_gnn='GCN',
                 with_filter_matrix=False, K=3):
        super().__init__()
        self.conv = GNN_Layer(in_features, out_features, gnn_activation, type_gnn=type_gnn)
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

class GNN_new(nn.Module):
    """
    Refactored GNN class.

    The `out_features` argument is no longer hardcoded to 2.
    It must be passed in as `num_output_features`.
    """
    def __init__(self, in_features, hid_features,
                 num_output_features, # <-- CHANGED: Was 'out_features=2'
                 mlp_layers,
                 gnn_activation='tanh',
                 mlp_activation='prelu',
                 type_gnn='GCN',
                 with_filter_matrix=False, K=3,
                 **kwargs): # Added **kwargs to catch unused args
        super().__init__()

        # --- CHANGED: Parameterized output features ---
        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = num_output_features # <-- RENAMED/PARAMETERIZED
        # --- END CHANGE ---

        self.gnn = GNN_Layer(in_features, hid_features, gnn_activation, type_gnn=type_gnn)

        self.decoder = MLP(hid_features, self.out_features, hid_features,
                           mlp_layers, mlp_activation)

    def forward(self, static_features, dynamic_features, edge_index, edge_attr, **kwargs):
        x = torch.cat([static_features, dynamic_features], -1)

        x = self.gnn(x, edge_index, edge_attr)
        x = self.decoder(x)

        return x


class MSGNN_new(nn.Module):
    """
    Refactored MSGNN class.

    The `out_features` argument is no longer hardcoded to 2.
    It must be passed in as `num_output_features`.
    """
    def __init__(self, in_features, hid_features,
                 num_output_features, # <-- CHANGED: Was 'out_features=2'
                 mlp_layers,
                 num_scales,
                 gnn_activation='tanh',
                 mlp_activation='prelu',
                 type_gnn='GCN',
                 with_filter_matrix=False, K=3,
                 learned_pooling=False,
                 skip_connections=True,
                 **kwargs): # Added **kwargs to catch unused args
        super().__init__()

        # --- CHANGED: Parameterized output features ---
        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = num_output_features # <-- RENAMED/PARAMETERIZED
        # --- END CHANGE ---

        self.num_scales = num_scales
        self.learned_pooling = learned_pooling
        self.skip_connections = skip_connections

        self.encoder = MLP(in_features, hid_features, hid_features,
                           mlp_layers, mlp_activation)

        self.gnn_layers = nn.ModuleList([MSGNN_Layer(hid_features, hid_features, gnn_activation, type_gnn=type_gnn,
                                                with_filter_matrix=with_filter_matrix, K=K) for _ in range(num_scales)])

        self.decoder = MLP(hid_features, self.out_features, hid_features,
                           mlp_layers, mlp_activation)

        if self.learned_pooling:
            self.pooling_layers = nn.ModuleList([nn.Linear(hid_features, hid_features) for _ in range(num_scales-1)])

    def forward(self, static_features, dynamic_features, edge_index, edge_attr, batch, **kwargs):
        x = torch.cat([static_features, dynamic_features], -1)
        x = self.encoder(x)

        x_scales = self._create_scale_features(x, batch)

        x_scales_out = []
        for i in range(self.num_scales):
            x_conv = self.gnn_layers[i](x_scales[i],
                                     edge_index[i],
                                     edge_attr=edge_attr[i],
                                     intra_mesh_edge_index=None)
            x_scales_out.append(x_conv)

        x_out = self._pool(x_scales_out, batch)

        x_out = self.decoder(x_out)

        return x_out

    def _pool(self, x_scales, batch):
        # ... (Copy the _pool and _create_scale_features methods from models/gnn.py here) ...
        # ... (No changes are needed in those helper methods) ...
        finest_scale = x_scales[0]
        if self.skip_connections:
            for i in range(self.num_scales-1):
                if self.learned_pooling:
                    x_pool = self.pooling_layers[i](x_scales[i+1])
                else:
                    x_pool = x_scales[i+1]

                finest_scale = finest_scale + scatter(x_pool, batch.node_ptr[batch.intra_edge_ptr[i]:batch.intra_edge_ptr[i+1],1],
                                                      dim=0, reduce='mean')
        return finest_scale

    def _create_scale_features(self, x, batch):
        if isinstance(batch, Batch):
            x_scales = [x[batch.node_ptr[i,0]:batch.node_ptr[i,-1]] for i in range(batch.num_graphs)]
            x_scales = [torch.cat([x_graph[batch.node_ptr[i,j]:batch.node_ptr[i,j+1]] for i in range(batch.num_graphs)])
                        for j in range(self.num_scales)]
        else:
            x_scales = [x[batch.node_ptr[i]:batch.node_ptr[i+1]] for i in range(self.num_scales)]
        return x_scales
