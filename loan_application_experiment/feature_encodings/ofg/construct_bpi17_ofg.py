# %%
# import logging
import pickle
from typing import Any

import hetero_graph_encoding_utils as utils
import numpy as np
import pandas as pd
import pm4py
import pm4py.ocel
import pm4py.read
import torch
from pm4py.algo.transformation.ocel.features.objects import (
    algorithm as object_feature_factory,
)
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData

ocel_file = "data/BPI17/source/BPI2017-Final.jsonocel"


# %%
# load OCEL
ocel = pm4py.read.read_ocel(ocel_file)

# %%
# encode boolean variables
ocel.objects["event_Accepted"] = ocel.objects["event_Accepted"].replace(
    {True: 1, False: 0}
)
ocel.objects["event_Selected"] = ocel.objects["event_Selected"].replace(
    {True: 1, False: 0}
)
ocel.objects = ocel.objects.reset_index().rename(columns={"index": "object_index"})

# %%
# define object attributes per object type
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
# NORMALIZE "@@object_lifecycle_duration" (JUST FOR TESTING)
object_features.iloc[:, 1:2] = StandardScaler().fit_transform(
    object_features.iloc[:, 1:2]
)
# retrieve the mapper from ocel object id to the object index in the pd.DataFrame (e.g. 'Offer_148581083':1)
oid_object_index_map = utils.get_index_map(ocel.objects, "ocel:oid", "object_index")

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
        + ["object_index", "object_lifecycle_duration"]
    ]
)
offer_attribute_feature_idxs = flatten(
    [
        np.where(offer_features.columns.str.contains(attr_name))[0]
        for attr_name in offer_attributes["str"]
        + offer_attributes["num"]
        + ["object_index", "object_lifecycle_duration"]
    ]
)
# subset application features, with correct columns
application_features = application_features.iloc[:, application_attribute_feature_idxs]
# create object_index to application_index mapper
application_features = utils.add_object_type_index(application_features, "application")
object_index_application_index_map = utils.get_index_map(
    application_features, "object_index", "application_index"
)

# subset offer features, with correct columns
offer_features = offer_features.iloc[:, offer_attribute_feature_idxs]
# create object_index to offer_index mapper
offer_features = utils.add_object_type_index(offer_features, "offer")
object_index_offer_index_map = utils.get_index_map(
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
bpi17_edges_per_edge_type = utils.split_on_edge_types(list(graph), bpi17_edge_types)

# %%
# create ocel object index to application node index (for HeteroData) mapper
application_to_node_map = utils.object_map_to_node_map(
    oid_object_index_map, object_index_application_index_map, "application"
)
# create ocel object index to offer node index (for HeteroData) mapper
offer_to_node_map = utils.object_map_to_node_map(
    oid_object_index_map, object_index_offer_index_map, "offer"
)


# %%
# rename edges to have correct edge_index for HeteroData
bpi17_edges_per_edge_type = utils.rename_edges_in_split_dict(
    bpi17_edges_per_edge_type, application_to_node_map
)
bpi17_edges_per_edge_type = utils.rename_edges_in_split_dict(
    bpi17_edges_per_edge_type, offer_to_node_map
)

# %%
# define heterogeneous graph
hetero_data = HeteroData()
# define target variable for both "application" type and "offer" type
hetero_data["application"].y = torch.tensor(
    application_features["@@object_lifecycle_duration"].values
)
hetero_data["offer"].y = torch.tensor(
    offer_features["@@object_lifecycle_duration"].values
)
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
hetero_data["application", "interacts", "offer"].edge_index = utils.to_torch_coo_format(
    bpi17_edges_per_edge_type[("application", "offer")]
)
hetero_data["offer", "interacts", "offer"].edge_index = utils.to_torch_coo_format(
    bpi17_edges_per_edge_type[("offer", "offer")]
    # utils.to_undirected(bpi17_edges_per_edge_type[("offer", "offer")])
)


# %%
# save HeteroData object
with open(
    "data/BPI17/feature_encodings/OFG/ofg/raw/BPI17_OFG.pkl", "wb"
) as binary_file:
    pickle.dump(hetero_data, binary_file)

# save object interaction graph information (for HOEG encoding)
with open(
    "data/BPI17/feature_encodings/HOEG/hoeg/raw/bpi17_ofg+oi_graph+app_node_map+off_node_map.pkl",
    "wb",
) as binary_file:
    objects_data = {
        "ofg": hetero_data,
        "objects_interaction_graph": graph,
        "object_feature_vector_map": {
            "application": application_to_node_map,
            "offer": offer_to_node_map,
        },
        "object_feature_matrices": {
            "application": application_features,
            "offer": offer_features,
        },
    }
    pickle.dump(objects_data, binary_file)
