import pickle

import torch
from torch_geometric.nn import GATConv, Linear, to_hetero

from ....models.geometric_models import AGNN, CGNN

with open("../data/BPI17/feature_encodings/BPI17_OFG.pkl", "rb") as het_data_pkl:
    het_data = pickle.load(het_data_pkl)


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


model = GAT(hidden_channels=64, out_channels=1)
model = to_hetero(model, het_data.metadata(), aggr="sum")
