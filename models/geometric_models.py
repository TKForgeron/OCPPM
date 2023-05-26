import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

torch.manual_seed(42)

import torch
from torch.nn import Sequential, MSELoss
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCN(torch.nn.Module):
    def __init__(self, num_node_features: int, hyperparams: dict):
        super().__init__()

        self.gconv1 = GCNConv(num_node_features, hyperparams["num_hidden_features"])
        self.gconv2 = GCNConv(
            hyperparams["num_hidden_features"], hyperparams["num_hidden_features"]
        )
        self.out = Linear(hyperparams["num_hidden_features"], 1)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index

        # First Message Passing Layer (Transformation)
        x = self.gconv1(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.gconv2(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)

        # Output layer
        # x = F.mse_loss()
        # x = tf.reshape(h, (int(h.shape[0] / 4), (4 * 24))) MAKE THIS IN PYTORCH?
        out = self.out(x)
        return out

    def get_class_name(self) -> str:
        return str(self).split("(")[0]


class GAT(torch.nn.Module):
    def __init__(self, num_node_features: int, hyperparams: dict):
        super().__init__()

        self.gconv1 = GATConv(num_node_features, hyperparams["num_hidden_features"])
        self.gconv2 = GATConv(
            hyperparams["num_hidden_features"], hyperparams["num_hidden_features"]
        )
        self.out = Linear(hyperparams["num_hidden_features"], 1)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index

        # First Message Passing Layer (Transformation)
        x = self.gconv1(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.gconv2(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)

        # Output layer
        # x = F.mse_loss()
        # x = tf.reshape(h, (int(h.shape[0] / 4), (4 * 24))) MAKE THIS IN PYTORCH?
        out = self.out(x)
        return out

    def get_class_name(self) -> str:
        return str(self).split("(")[0]

