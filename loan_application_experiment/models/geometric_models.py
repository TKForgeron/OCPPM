import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ModuleList, MSELoss, Sequential
from torch_geometric.nn import GATConv, GCNConv, TopKPooling, TransformerConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


class GraphModel(torch.nn.Module):
    def get_class_name(self) -> str:
        """Returns the class name"""
        return str(self).split("(")[0]


class AdamsGCN(GraphModel):
    """Implementation of a Graph Convolutional Network as in Adams et al. (2022)"""

    def __init__(self, num_node_features: int, hyperparams: dict):
        super().__init__()
        self.hyperparams = hyperparams
        self.gconv1 = GCNConv(num_node_features, hyperparams["num_hidden_features"])
        self.gconv2 = GCNConv(
            hyperparams["num_hidden_features"], hyperparams["num_hidden_features"]
        )
        self.out = Linear(
            hyperparams["num_hidden_features"] * hyperparams["size_subgraph_samples"], 1
        )

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index

        # First Message Passing Layer (Transformation)
        x = self.gconv1(x, edge_index)
        x = x.relu()

        # Second Message Passing Layer
        x = self.gconv2(x, edge_index)
        x = x.relu()

        # Reshape layer, to account for graph-level predictions,
        # since we're given concatenated subgraph samples each mini batch
        x = torch.reshape(
            x,
            (
                int(x.shape[0] / self.hyperparams["size_subgraph_samples"]),
                int(
                    x.shape[0]
                    * x.shape[1]
                    / (x.shape[0] / self.hyperparams["size_subgraph_samples"])
                ),
            ),
        )

        # Output layer
        out = self.out(x)
        return out


class CGNN(torch.nn.Module):
    """Implementation of a Graph Convolutional Network"""

    def __init__(self, num_node_features: int, hyperparams: dict):
        super().__init__()
        self.hyperparams = hyperparams
        self.gconv1 = GCNConv(num_node_features, hyperparams["num_hidden_features"])
        self.gconv2 = GCNConv(
            hyperparams["num_hidden_features"], hyperparams["num_hidden_features"]
        )
        # self.dropout = torch.nn.Dropout(p=0.2)
        self.h3 = Linear(
            hyperparams["num_hidden_features"] * hyperparams["size_subgraph_samples"],
            hyperparams["num_hidden_features"] * hyperparams["size_subgraph_samples"],
        )
        self.out = Linear(
            hyperparams["num_hidden_features"] * hyperparams["size_subgraph_samples"], 1
        )

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index

        # First Message Passing Layer (Transformation)
        x = self.gconv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.25, training=self.training)

        # Second Message Passing Layer
        x = self.gconv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.25, training=self.training)

        # Reshape layer, to account for graph-level predictions,
        # since we're given concatenated subgraph samples each mini batch
        x = torch.reshape(
            x,
            (
                int(x.shape[0] / self.hyperparams["size_subgraph_samples"]),
                int(
                    x.shape[0]
                    * x.shape[1]
                    / (x.shape[0] / self.hyperparams["size_subgraph_samples"])
                ),
            ),
        )
        # Third hidden: Linear
        x = self.h3(x)
        x = x.relu()
        x = F.dropout(x, p=0.25, training=self.training)

        # Output layer
        out = self.out(x)
        return out

    def get_class_name(self) -> str:
        """Returns the class name"""
        return str(self).split("(")[0]


class AGNN(torch.nn.Module):
    """Implementation of a Graph Attention Network"""

    def __init__(self, num_node_features: int, hyperparams: dict):
        super().__init__()
        self.hyperparams = hyperparams
        self.gconv1 = GATConv(num_node_features, hyperparams["num_hidden_features"])
        self.gconv2 = GATConv(
            hyperparams["num_hidden_features"], hyperparams["num_hidden_features"]
        )
        self.dropout = torch.nn.Dropout(0.2)
        self.out = Linear(
            hyperparams["num_hidden_features"] * hyperparams["size_subgraph_samples"], 1
        )

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index

        # First Message Passing Layer (Transformation)
        x = self.gconv1(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.gconv2(x, edge_index)
        x = x.relu()
        x = self.dropout(x)

        # Output layer
        # x = F.mse_loss()
        # Reshape layer, to account for graph-level predictions,
        # since we're given concatenated subgraph samples each mini batch
        a = int(x.shape[0] / self.hyperparams["size_subgraph_samples"])
        b = int(x.shape[1] * self.hyperparams["size_subgraph_samples"])
        x = torch.reshape(x, (a, b))
        out = self.out(x)
        return out

    def get_class_name(self) -> str:
        return str(self).split("(")[0]
