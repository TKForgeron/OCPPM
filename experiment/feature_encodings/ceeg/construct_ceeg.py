# SELF REMINDER. Copy the 'ocpa' directory to the forked one from github, so that I can push updates to github.
# Timing
import timeit

start = timeit.default_timer()

# Python natives
import pickle

import ocpa.algo.predictive_monitoring.factory as feature_factory

# Object centric process mining
from ocpa.objects.log.importer.ocel import factory as ocel_import_factory

# Simple machine learning models, procedure tools, and evaluation metrics
from sklearn.preprocessing import PowerTransformer, StandardScaler


def print_time_taken(start_time: float, prefix: str = "") -> float:
    elapsed_time = timeit.default_timer() - start_time
    minutes, seconds = int(elapsed_time / 60), int(elapsed_time % 60)
    print(f"{prefix} time taken: {minutes}m{seconds}s")


BASE_DIR = "data/BPI17"
oeeg_out_file = f"{BASE_DIR}/feature_encodings/BPI17_OEEG.pkl"
TARGET_LABEL = feature_factory.EVENT_REMAINING_TIME

ocel_file = f"{BASE_DIR}/BPI2017-Final.jsonocel"

print("Constructing OCEL object")
ocel = ocel_import_factory.apply(ocel_file)

activities = ocel.log.log["event_activity"].unique().tolist()

feature_set = {
    "event-level": [
        (feature_factory.EVENT_PRECEDING_ACTIVITIES, (act,)) for act in activities
    ]
    + [  # C2
        (
            feature_factory.EVENT_AGG_PREVIOUS_CHAR_VALUES,
            ("event_RequestedAmount", max),
        ),  # D1
        (feature_factory.EVENT_ELAPSED_TIME, ()),  # P2
        TARGET_LABEL,  # P3
        (feature_factory.EVENT_PREVIOUS_TYPE_COUNT, ("offer",)),  # O3
    ],
    "execution-level": [feature_factory.EXECUTION_THROUGHPUT],  # P1
}
print("Constructing FeatureStorage object")
feature_storage = feature_factory.apply(
    ocel,
    event_based_features=feature_set["event-level"],
    execution_based_features=feature_set["execution-level"],
)

print("Pickling FeatureStorage object")
with open(
    f"{BASE_DIR}/intermediates/ceeg/BPI17-feature_storage-[C2,D1,P2,P3,O3,P1].fs",
    "wb",
) as file:
    pickle.dump(feature_storage, file)

print_time_taken(start, "Total")
