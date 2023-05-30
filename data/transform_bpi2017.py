# %%
""" This script transforms BPI2017 to a full OCEL, with separated events and objects and their attributes. """

# Python natives
import json
import pickle
import random
from copy import copy
from typing import Any

import ocpa.objects.log.importer.csv.factory as csv_import_factory
# Object centric process mining
import ocpa.objects.log.importer.ocel.factory as ocel_import_factory
from ocpa.objects.log.exporter.ocel import factory as ocel_export_factory
from ocpa.objects.log.ocel import OCEL


def load_ocel(
    ocel_filename: str,
    parameters: dict[str, Any] = None,
) -> OCEL:
    extension = ocel_filename.split(".")[-1]
    if extension == "csv":
        return csv_import_factory.apply(
            ocel_filename, csv_import_factory.TO_OCEL, parameters
        )
    elif extension == "jsonocel":
        return ocel_import_factory.apply(ocel_filename)
    elif extension == "xmlocel":
        raise Exception("XMLOCEL is not supported yet")
    else:
        raise Exception(f"{extension} is not supported yet")

ocel_in_file = "BPI17/BPI2017-Final.csv"
ocel_intermediary_file = "BPI17/intermediates/BPI2017.jsonocel"
ocel_out_file =  "BPI17/BPI2017-Final.jsonocel"

# %%
# Define event attributes
event_attributes = [
    "event_Action",
    "event_EventOrigin",
    "event_OrgResource",
]

# Define object attributes
application_attributes = {
    "event_RequestedAmount",
    "event_ApplicationType",
    "event_LoanGoal",
}
offer_attributes = {
    "event_OfferedAmount",
    "event_CreditScore",
    "event_Selected",
    "event_MonthlyCost",
    "event_NumberOfTerms",
    "event_Accepted",
    "event_FirstWithdrawalAmount",
}

object_attributes=offer_attributes.union(application_attributes)
# Load full BPI17 CSV as OCEL
ocel = load_ocel(
    ocel_filename=ocel_in_file,
    parameters={
        "obj_names": ["application", "offer"],
        "val_names": event_attributes + list(object_attributes),
        "act_name": "event_activity",
        "time_name": "event_timestamp",
        "sep": ",",
    },
)

# Export BPI17 OCEL as JSON
ocel_export_factory.apply(ocel, ocel_intermediary_file)
print("Intermediary jsonocel export success!")


# %%
# Load JSONOCEL
with open(ocel_intermediary_file) as json_data:
    data = json.load(json_data)

# %%
def sample_dict(n: int, dy: dict, seed:int=42) -> dict:
    random.seed(seed)
    return {k:dy[k] for k in random.sample(dy.keys(), n)}

events = data['ocel:events']
objects = data['ocel:objects']

# %%
def get_related_event(
    object_id: str, events: dict[str, Any], attr_names: set[str]
) -> str:
    # there may be multiple events related to an object, but we just take one of them
    event_keys = {ekey for ekey in events if object_id in events[ekey]["ocel:omap"]}
    if event_keys:
        return next(iter(event_keys))
    elif len(event_keys)>1:
        return next(next(iter(event_keys)))


def object_attributes_from_event_attributes(
    attr_names: set[str],
    object_type: str,
    events: dict[str, Any],
    objects: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # for given object type get related events per object
    oid_events_map = {
        oid: get_related_event(oid, events,attr_names)
        for oid in (oid for oid, v in objects.items() if v["ocel:type"] == object_type)
    }
    oid_events_map = {
        k: v for k, v in oid_events_map.items() if v
    }  # filter out objects that have no related events

    # lookup event attributes and append to each object
    for oid, eid in oid_events_map.items():
        # gather object attributes and remove from events dictionary
        object_attributes = {
            attr_name: events[eid]["ocel:vmap"].pop(attr_name, None)
            for attr_name in attr_names
        }

        # append object attributes to objects dictionary
        objects[oid]["ocel:ovmap"] = object_attributes

    return events, objects


def remove_event_attributes(events:dict[str,dict[str,Any]], attributes:set[str]) -> dict[str,set[str]]:
    result = {'success': {},'fail':copy(attributes)}
    for event in events:
        for attribute_to_remove in attributes:
            try:
                del events[event]['ocel:vmap'][attribute_to_remove]
                result["success"] = attribute_to_remove
                result["fail"].discard(attribute_to_remove)
            except:
                pass
    return result


# %%
# Make application object attributes
events, objects = object_attributes_from_event_attributes(
    application_attributes, "application", events, objects
)


# # %%
# with open("BPI17/intermediates/BPI2017_events(application).pkl", 'wb') as events_obj:
#     pickle.dump(events, events_obj)

# with open("BPI17/intermediates/BPI2017_objects(application).pkl", 'wb') as objects_obj:
#     pickle.dump(objects, objects_obj)

# %%
# Make offer object attributes
events, objects = object_attributes_from_event_attributes(
    offer_attributes, "offer", events, objects
)

# Remove event attributes that have been transformed to object attributes
res = remove_event_attributes(data['ocel:events'], object_attributes.union({'event_index','event_id'})))

# # %%
# with open("BPI17/intermediates/BPI2017_events.pkl", 'wb') as events_obj:
#     pickle.dump(events, events_obj)

# with open("BPI17/intermediates/BPI2017_objects.pkl", 'wb') as objects_obj:
#     pickle.dump(objects, objects_obj)

# %%
data['ocel:events'] = events
data['ocel:objects'] = objects

# %%
with open(ocel_out_file, "w") as json_data:
    json.dump(data, json_data)
