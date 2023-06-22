import logging

import torch.nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ModuleList, MSELoss, Sequential
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.nn import GATConv, GCNConv, HGTConv, TopKPooling, TransformerConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


class HCGNN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        meta_data = (
            ["event", "application", "offer"],
            [
                ("event", "follows", "event"),
                ("event", "interacts", "application"),
                ("event", "interacts", "offer"),
                ("application", "interacts", "application"),
                ("application", "rev_interacts", "event"),
                ("offer", "rev_interacts", "event"),
            ],
        )

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in [meta_data[0]]:
            logging.debug(node_type)
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                hidden_channels,
                hidden_channels,
                meta_data,
                num_heads,
                group="sum",
            )
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict: dict, edge_index_dict: dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict["event"])
