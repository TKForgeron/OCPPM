# %%
# import logging
import pickle
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import pm4py
import pm4py.ocel
import pm4py.read
import torch
from pm4py.algo.transformation.ocel.features.objects import (
    algorithm as object_feature_factory,
)
from torch_geometric.data import HeteroData

ocel_file = "data/BPI17/BPI2017-Final.jsonocel"


# %%
def get_index_map(df: pd.DataFrame, index_col_a: str, index_col_b: str) -> dict:
    return pd.Series(df[index_col_b].values, index=df[index_col_a]).to_dict()


def add_object_type_index(df: pd.DataFrame, object_type: str) -> pd.DataFrame:
    return (
        df.reset_index()
        .drop("index", axis=1)
        .reset_index()
        .rename(columns={"index": f"{object_type}_index"})
    )


def object_map_to_node_map(object_map: dict, node_map: dict, object_name: str) -> dict:
    return {
        k: node_map[v]
        for (k, v) in object_map.items()
        if object_name.lower() in k.lower()
    }


def split_on_edge_types(
    edge_list: list[tuple[str, str]], edge_types: list[tuple[str, str]]
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """
    Function that splits edges based on a given list of edge types.
    It returns a dict, with a key for each (found) edge type and a list for the corresponding edges.

    Currently, the function assumes directed edges. So do specify both ways, if undirected edges are wanted.

    Procedure:
    Per edge type (in edge_types), append all edges (from edge_list)
    if both the source and destination type of the edge corresponds to the edge type's source and destination
    """

    edges_split_on_type = dict()
    for edge_type in edge_types:
        for edge in edge_list:
            if edge_type[0] in edge[0].lower() and edge_type[1] in edge[1].lower():
                if edge_type in edges_split_on_type:
                    # if the key/edge_type already exists, append
                    edges_split_on_type[edge_type].append(edge)
                else:
                    # if this edge is the first of this edge_type
                    edges_split_on_type[edge_type] = [
                        edge
                    ]  # define a new key and value (list with the edge inside)
    return edges_split_on_type


def to_undirected(edge_list: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    return list(set(edge_list + [tuple(reversed(edge)) for edge in edge_list]))


def rename_edges(
    edges: list[tuple[Any, Any]],
    renaming_map: dict,
) -> list[tuple[Any, Any]]:
    renamed_edges = []
    for edge in edges:
        if edge[0] in renaming_map and edge[1] in renaming_map:
            renamed_edges.append((renaming_map[edge[0]], renaming_map[edge[1]]))
        elif edge[0] in renaming_map:
            renamed_edges.append((renaming_map[edge[0]], edge[1]))
        elif edge[1] in renaming_map:
            renamed_edges.append((edge[0], renaming_map[edge[1]]))
        else:
            renamed_edges.append(edge)
    return renamed_edges


def rename_edges_in_split_dict(
    split_edges: dict[tuple[str, str], list[tuple[Any, Any]]], renaming_map: dict
) -> dict[tuple[str, str], list[tuple[Any, Any]]]:
    return {
        edge_type: rename_edges(edges, renaming_map)
        for (edge_type, edges) in split_edges.items()
    }


def to_torch_coo_format(edge_list: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(edge_list, dtype=torch.int64).T


# %%
ocel = pm4py.read.read_ocel(ocel_file)
ocel.objects.dtypes

# %%
ocel.objects["event_Accepted"] = ocel.objects["event_Accepted"].replace(
    {True: 1, False: 0}
)
ocel.objects["event_Selected"] = ocel.objects["event_Selected"].replace(
    {True: 1, False: 0}
)
ocel.objects = ocel.objects.reset_index().rename(columns={"index": "object_index"})

# %%
application_attributes = {
    "str": [
        "event_LoanGoal",
        "event_ApplicationType",
    ],
    "num": [
        "event_RequestedAmount",
    ],
}
offer_attributes = {
    "str": [],
    "num": [
        "event_NumberOfTerms",
        "event_Accepted",
        "event_Selected",
        "event_OfferedAmount",
        "event_CreditScore",
        "event_FirstWithdrawalAmount",
        "event_MonthlyCost",
    ],
}

# %%
# create object-level feature matrix
data, feature_names = object_feature_factory.apply(
    ocel,
    parameters={
        "str_obj_attr": ["ocel:type"]
        + application_attributes["str"]
        + offer_attributes["str"],
        "num_obj_attr": ["object_index"]  # include object_index for reference
        + application_attributes["num"]
        + offer_attributes["num"],
    },
)

# %%
# make pd.DataFrame from feature matrix
object_features = pd.DataFrame(data, columns=feature_names)
# retrieve the mapper from ocel object id to the object index in the pd.DataFrame (e.g. 'Offer_148581083':1)
oid_object_index_map = get_index_map(ocel.objects, "ocel:oid", "object_index")

# reset column name from object_index that was passed to the object-level feature matrix factory
object_features = object_features.rename(
    columns={"@@event_num_object_index": "object_index"}
)
object_features["object_index"] = object_features["object_index"].astype(int)

# %%


# Split object feature matrix into one feature matrix per object type
offer_features = object_features[
    object_features["@@object_attr_value_ocel:type_offer"] == 1
]
application_features = object_features[
    object_features["@@object_attr_value_ocel:type_application"] == 1
]

# Subset features to only include object attribute features, excluding object interaction features
flatten = lambda l: [item for sublist in l for item in sublist]
application_attribute_feature_idxs = flatten(
    [
        np.where(application_features.columns.str.contains(attr_name))[0]
        for attr_name in application_attributes["str"]
        + application_attributes["num"]
        + ["object_index"]
    ]
)
offer_attribute_feature_idxs = flatten(
    [
        np.where(offer_features.columns.str.contains(attr_name))[0]
        for attr_name in offer_attributes["str"]
        + offer_attributes["num"]
        + ["object_index"]
    ]
)
# subset application features, with correct columns
application_features = application_features.iloc[:, application_attribute_feature_idxs]
# create object_index to application_index mapper
application_features = add_object_type_index(application_features, "application")
object_index_application_index_map = get_index_map(
    application_features, "object_index", "application_index"
)

# subset offer features, with correct columns
offer_features = offer_features.iloc[:, offer_attribute_feature_idxs]
# create object_index to offer_index mapper
offer_features = add_object_type_index(offer_features, "offer")
object_index_offer_index_map = get_index_map(
    offer_features, "object_index", "offer_index"
)

# %%
# calculate object graph (we select object_interaction here, but other graphs are possible)
graph = pm4py.ocel.discover_objects_graph(ocel, graph_type="object_interaction")


# %%
# define object relation types (edge types)
bpi17_edge_types = [
    ("offer", "offer"),
    ("application", "offer"),
    ("application", "application"),
]
# assign edge tuples to correct edge types
bpi17_edges_per_edge_type = split_on_edge_types(list(graph), bpi17_edge_types)

# %%
# create ocel object index to application node index (for HeteroData) mapper
application_to_node_map = object_map_to_node_map(
    oid_object_index_map, object_index_application_index_map, "application"
)
# create ocel object index to offer node index (for HeteroData) mapper
offer_to_node_map = object_map_to_node_map(
    oid_object_index_map, object_index_offer_index_map, "offer"
)

# %%
# rename edges to have correct edge_index for HeteroData
bpi17_edges_per_edge_type = rename_edges_in_split_dict(
    bpi17_edges_per_edge_type, application_to_node_map
)
bpi17_edges_per_edge_type = rename_edges_in_split_dict(
    bpi17_edges_per_edge_type, offer_to_node_map
)

# %%
# define heteregeneous graph
hetero_data = HeteroData()
# attach node feature vectors for both "application" type and "offer" type
hetero_data["application"].x = torch.tensor(
    application_features.drop(["application_index", "object_index"], axis=1).values
)
hetero_data["offer"].x = torch.tensor(
    offer_features.drop(["offer_index", "object_index"], axis=1).values
)

# with edge types: application->offer, offer<->offer
hetero_data["application", "interacts", "application"].edge_index = torch.tensor(
    [[], []], dtype=torch.int64
)
hetero_data["application", "interacts", "offer"].edge_index = to_torch_coo_format(
    bpi17_edges_per_edge_type[("application", "offer")]
)
hetero_data["offer", "interacts", "offer"].edge_index = to_torch_coo_format(
    to_undirected(bpi17_edges_per_edge_type[("offer", "offer")])
)

# %%
with open("data/BPI17/feature_encodings/BPI17_OFG.pkl", "wb") as binary_file:
    pickle.dump(hetero_data, binary_file)
