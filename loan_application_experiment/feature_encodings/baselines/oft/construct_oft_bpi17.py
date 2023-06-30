import re
from typing import Any

import numpy as np
import pandas as pd
import pm4py
import pm4py.ocel
import pm4py.read
from pm4py.algo.transformation.ocel.features.objects import (
    algorithm as object_feature_factory,
)
from sklearn.preprocessing import StandardScaler

ocel_file = "../../../data/BPI17/source/BPI2017-Final.jsonocel"
oft_out_file = "../../../data/BPI17/feature_encodings/OFT/application_features.csv"

# load OCEL
ocel = pm4py.read.read_ocel(ocel_file)

# encode boolean variables
ocel.objects["event_Accepted"] = ocel.objects["event_Accepted"].replace(
    {True: 1, False: 0}
)
ocel.objects["event_Selected"] = ocel.objects["event_Selected"].replace(
    {True: 1, False: 0}
)
ocel.objects = ocel.objects.reset_index().rename(columns={"index": "object_index"})

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

# make pd.DataFrame from feature matrix
object_features = pd.DataFrame(data, columns=feature_names)

# NORMALIZE "@@object_lifecycle_duration" (JUST FOR TESTING)
object_features.iloc[:, 1:2] = StandardScaler().fit_transform(
    object_features.iloc[:, 1:2]
)

# Split object feature matrix into one feature matrix per object type
offer_features = object_features[
    object_features["@@object_attr_value_ocel:type_offer"] == 1
]
application_features = object_features[
    object_features["@@object_attr_value_ocel:type_application"] == 1
]

# clean application features
flatten = lambda l: [item for sublist in l for item in sublist]

# select used columns/features
application_attribute_feature_idxs = flatten(
    [
        np.where(application_features.columns.str.contains(attr_name))[0]
        for attr_name in application_attributes["str"]
        + application_attributes["num"]
        + ["object_lifecycle_duration"]
    ]
)
application_features = application_features.iloc[:, application_attribute_feature_idxs]

# strip JSON special characters from feature names, as they are now supported in LightGBM
application_features = application_features.rename(
    columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x)
)

# export the dataframe to csv
application_features.to_csv(oft_out_file, index=False)
