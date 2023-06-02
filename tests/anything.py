# %%
hetero_data.to_homogeneous()

# %%
hetero_data["application", "interacts with", "offer"]

# %%
G = nx.Graph()
G.add_edges_from(graph)
some_graph = {x for x in graph if "Application_2114646933" in x}
sorted_cc = sorted(list(nx.connected_components(G)), key=len, reverse=True)
sg = G.subgraph(sorted_cc[4])
# sg = nx.Graph()
# sg.add_edges_from(some_graph_adjacency)
nx.draw(sg, with_labels=True)
plt.savefig("../../tests/objects_interaction_graph.png")

# %%
from torch_geometric.datasets import OGB_MAG

dataset = OGB_MAG(root="./data", preprocess="metapath2vec")
citation_network = dataset[0]
citation_network

# %%
example_edge = citation_network.edge_items()[0]
edge_tensor = example_edge[1].edge_index

# %%
edge_tensor.shape

# %%
# testing to check whether we can indicate correct nodes from each node type in the edge_index
orders = {"price": [1250, 678], "discount": [33, 0], "black_friday_sale": [1, 0]}
items = {"weight": [3.5, 3.0, 26.0], "size": [2, 2, 3]}
order_x_tensor = torch.tensor(list(orders.values()))[1:].T
order_y_tensor = torch.tensor(list(orders.values()))[
    0
].T  # order_price will be the target
item_x_tensor = torch.tensor(list(items.values())).T
# o_o_interaction = torch.tensor([[],[]])
o_i_interaction = torch.tensor(
    [[0, 0, 1], [0, 1, 2]]
)  # directional (only from order to item)
i_i_interaction = torch.tensor([[2, 1], [1, 2]])
test_het = HeteroData(
    {"order": {"x": order_x_tensor, "y": order_y_tensor}, "item": {"x": item_x_tensor}},
    #   order__interacts_with__order={'edge_index': o_o_interaction},
    order__interacts_with__item={"edge_index": o_i_interaction},
    item__interacts_with__item={"edge_index": i_i_interaction},
)

# %%
test_het
g = utils.to_networkx(test_het.to_homogeneous(), to_undirected=False)
nx.draw(g, with_labels=True)

# %%
fake_het = FakeHeteroDataset(
    num_node_types=2,
    num_edge_types=3,
    avg_num_nodes=2.5,
    num_classes=2,
    avg_degree=1,
    avg_num_channels=3,
)
fake_het = fake_het[0]
# fake_het.generate_ids()
fake_het

# %%
fake_het["v0"]

# %%
fake_het
fake_het.to_dict()

# %%
fake_het["(v1, e0, v1)"]

# %%


# %%


# %%


# %%
# define a format for the adjacency matrix (use real oid or custom object_index?)

# %%
# Define HeteroData in PyG, using offer_features, application_features, and the adjacency matrix

# %%
# build OFG class that can hold features
# enable this class to be ported to PyG


# %%
# COULD MOVE THIS TO A UNIT TEST FILE
# example_edges = [('o1','i1'),('o1','i2'),('i2','i3'),('i3','o2')]
# test_split = split_on_edge_types(edge_list=to_undirected(example_edges), edge_types=[('o','i'), ('i','i')])
# test_split
