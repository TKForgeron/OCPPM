# Python natives
import pickle
import re

# General data handling
import pandas as pd

# Object centric data handling
from ocpa.algo.predictive_monitoring import tabular

feature_storage_in_file = "data/BPI17/feature_encodings/EFG/efg/raw/BPI2017-feature_storage-split-[C1-3,C5,P1-6,O2,O3,O5].fs"

event_feature_table_out_file = (
    "data/BPI17/feature_encodings/EFT/event_based_features.csv"
)

with open(
    feature_storage_in_file,
    "rb",
) as file:
    feature_storage = pickle.load(file)

eft = tabular.construct_table(feature_storage)
# rename columns that contain JSON special characters (as they are not supported by LightGBM)
eft = eft.rename(columns=lambda col_name: re.sub("[^A-Za-z0-9_]+", "", str(col_name)))


eft.to_csv(event_feature_table_out_file, index=False)
