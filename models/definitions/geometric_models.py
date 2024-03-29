import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch.nn import BatchNorm1d, ModuleList, MSELoss, Sequential
from torch_geometric.nn import (
    BatchNorm,
    GATConv,
    GATv2Conv,
    GCNConv,
    GeneralConv,
    GINConv,
    Linear,
    SAGEConv,
    TopKPooling,
    TransformerConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    to_hetero,
)


class GraphModel(torch.nn.Module):
    def get_class_name(self) -> str:
        """Returns the class name"""
        return str(self).split("(")[0]


class AdamsGCN(GraphModel):
    """Implementation of a Graph Convolutional Network as in Adams et al. (2022)"""

    def __init__(
        self,
        hidden_channels: int = 24,
        out_channels: int = 1,
        regression_target: bool = True,
        graph_level_prediction: bool = True,
        squeeze: bool = False,
    ):
        super().__init__()
        self.squeeze = squeeze
        self.conv1 = GCNConv(-1, hidden_channels)
        self.conv2 = GCNConv(-1, hidden_channels)
        self.pool1 = global_mean_pool
        self.lin_out = Linear(-1, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.pool1(x, batch)
        x = self.lin_out(x)
        x = torch.squeeze(x) if self.squeeze else x
        return x


class SimpleGNN_EFG(GraphModel):
    # SimpleGNN_EFG(64, 1): 0.4382 MAE (test), 6k params
    def __init__(
        self,
        hidden_channels: int = 64,
        out_channels: int = 1,
        regression_target: bool = True,
        graph_level_prediction: bool = True,
    ):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_channels)
        self.conv2 = GCNConv(-1, hidden_channels)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.pool1 = lambda x, batch: x
        if graph_level_prediction:
            self.pool1 = pygnn.global_add_pool
        self.lin_out = Linear(-1, out_channels)
        self.probs_out = lambda x: x
        if not regression_target:
            self.probs_out = nn.Softmax(dim=1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.conv2(x, edge_index)
        x = self.act2(x)
        x = self.pool1(x, batch)
        x = self.lin_out(x)
        return self.probs_out(x)


# Model Configuration
class AGNN_EFG(GraphModel):
    """Implementation of a Attentional Graph Neural Network for EFG"""

    # AGNN_EFG(256, 1): 0.54 MAE (test)

    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_channels, add_self_loops=True)
        self.bn1 = BatchNorm(hidden_channels)
        self.act1 = nn.LeakyReLU()
        self.lin1 = Linear(-1, hidden_channels)
        self.pool1 = TopKPooling(hidden_channels)
        self.conv2 = GATv2Conv(-1, hidden_channels, add_self_loops=True)
        self.act2 = nn.LeakyReLU()
        self.pool2 = TopKPooling(hidden_channels)
        self.lin2 = Linear(-1, hidden_channels)
        self.pool3 = global_add_pool
        self.lin3 = Linear(-1, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.act1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = self.conv2(x, edge_index)
        x = self.act2(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin2(x)
        x = self.pool3(x, batch)
        x = self.lin3(x)
        return x


class SimpleHigherOrderGNN_EFG(GraphModel):
    def __init__(
        self,
        hidden_channels: int = 64,
        out_channels: int = 1,
        regression_target: bool = True,
        graph_level_prediction: bool = True,
    ):
        super().__init__()
        self.conv1 = pygnn.GraphConv(-1, hidden_channels)
        self.act1 = nn.PReLU()
        self.pool1 = lambda x, batch: x
        if graph_level_prediction:
            self.pool1 = pygnn.global_mean_pool
        self.lin_out = pygnn.Linear(-1, out_channels)
        self.probs_out = lambda x: x
        if not regression_target:
            self.probs_out = nn.Softmax(dim=1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.pool1(x, batch)
        x = self.lin_out(x)
        return self.probs_out(x)


class HigherOrderGNN_EFG_SG(GraphModel):
    # HigherOrderGNN_EFG(48, 1): 0.4040 MAE (test)

    def __init__(
        self,
        hidden_channels: int = 48,
        out_channels: int = 1,
        regression_target: bool = True,
        graph_level_prediction: bool = True,
    ):
        super().__init__()
        self.conv1 = pygnn.GraphConv(-1, hidden_channels)
        self.conv2 = pygnn.GraphConv(-1, hidden_channels)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.pool1 = lambda x, batch: x
        if graph_level_prediction:
            self.pool1 = pygnn.global_add_pool
        self.lin_out = pygnn.Linear(-1, out_channels)
        self.probs_out = lambda x: x
        if not regression_target:
            self.probs_out = nn.Softmax(dim=1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.conv2(x, edge_index)
        x = self.act2(x)
        x = self.pool1(x, batch)
        x = self.lin_out(x)
        return self.probs_out(x)


class HigherOrderGNN(GraphModel):
    """

    Implementation of a Higher Order Graph Neural Network for node-level
    prediction, presented by Morris et al. (2019). This type of GNN
    operator is especially suitable for graph-level prediction tasks.

    The authors show that GNNs have the same expressiveness as the
    1-WL in terms of distinguishing non-isomorphic (sub-)graphs,
    and propose a generalization of GNNs, called k-dimensional GNNs
    (k-GNNs), which can take higher-order graph structures at multiple
    scales into account.

    The "k" in k-dimensional refers to the number of scales or levels of
    higher-order structures that the network can capture.
        For example, a 2-dimensional GNN could capture both pairwise
        connections between nodes (1st order) and communities of nodes
        (2nd order), while a 3-dimensional GNN could capture even more
        complex patterns that involve groups of communities (3rd order).

    In this context, higher-order graph structures refer to patterns
    or relationships that exist between groups of nodes in a graph,
    beyond just the pairwise connections between individual nodes.
        For example, in a social network, higher-order structures
        could include communities of users who interact with each
        other more frequently than with users outside of their community.
        In a molecule graph, higher-order structures could include
        functional groups that are composed of multiple atoms and have
        specific chemical properties.
    By taking these higher-order structures into account, k-dimensional
    GNNs can capture more complex relationships between nodes in a graph
    and potentially improve the accuracy of graph classification and
    regression tasks.

    """

    def __init__(
        self,
        hidden_channels: int = 32,
        out_channels: int = 1,
        no_preprocessing_layers: int = 0,
        no_messagepassing_layers: int = 2,
        no_postprocessing_layers: int = 1,
        squeeze: bool = False,
        graph_level_prediction: bool = False,
    ):
        super().__init__()
        self.squeeze = squeeze
        self.pre_layers = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.post_layers = nn.ModuleList()
        for _ in range(no_preprocessing_layers):
            self.pre_layers.append(pygnn.Linear(-1, hidden_channels))
        for _ in range(no_messagepassing_layers):
            self.convs.append(pygnn.GraphConv(-1, hidden_channels))
            self.acts.append(nn.PReLU())
        self.gpool = lambda x, batch: x
        if graph_level_prediction:
            self.gpool = pygnn.global_add_pool
        for i in range(no_postprocessing_layers):
            if i != (no_postprocessing_layers - 1):
                self.post_layers.append(pygnn.Linear(-1, hidden_channels))
            else:
                self.post_layers.append(pygnn.Linear(-1, out_channels))

    def forward(self, x, edge_index, batch=None):
        for pre_processing in self.pre_layers:
            x = pre_processing(x)
        for conv, act in zip(self.convs, self.acts):
            x = conv(x, edge_index)
            x = act(x)
        x = self.gpool(x, batch)
        for post_processing in self.post_layers:
            x = post_processing(x)
        x = torch.squeeze(x) if self.squeeze else x
        return x


"""
What has been tried:
Convolutional layers:
    - SimpleConv
    - GCNConv
    - ChebConv
    - SAGEConv
    - GraphConv
    - GATConv
    - GATv2Conv
    - TransformerConv
    - GeneralConv
Global pooling layers:
    - global_add_pool
    - global_mean_pool
    - global_max_pool
    - ASAPooling
Local pooling layers:
    - TopKPooling
    - SAGPooling
    - ASAPooling
Hidden layer size:
    - 32
    - 48
    - 64
    - 128
    - 256
    - 512
Subgraph size:
    - 4
Batch size:
    - 32
Epochs:
    - 30 (early stopping: 5)
    - 75 (early stopping: 10)


"""
