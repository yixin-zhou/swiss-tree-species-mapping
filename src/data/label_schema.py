import json


LABEL_SCHEMA_VERSION = "nfi_basal_area_11class_v1"
STAND_TYPE_TO_ID = {"mixed": 0, "pure": 1}
STAND_NUM_CLASSES = 2
LEAF_TYPE_ORDER = ("Broadleaf", "Coniferous")
LEAF_NUM_CLASSES = len(LEAF_TYPE_ORDER)
SPECIES_CLASS_ORDER = (
    "Spruce",
    "Fir",
    "Pine",
    "Larch",
    "Arolla pine",
    "Beech",
    "Maple",
    "Ash",
    "Oak",
    "Chestnut",
    "Other",
)
SPECIES_NUM_CLASSES = len(SPECIES_CLASS_ORDER)
OTHER_SPECIES_INDEX = SPECIES_CLASS_ORDER.index("Other")
BROADLEAF_SPECIES_INDICES = tuple(
    SPECIES_CLASS_ORDER.index(name)
    for name in ("Beech", "Maple", "Ash", "Oak", "Chestnut")
)
CONIFEROUS_SPECIES_INDICES = tuple(
    SPECIES_CLASS_ORDER.index(name)
    for name in ("Spruce", "Fir", "Pine", "Larch", "Arolla pine")
)


def decode_json_attr(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return tuple(json.loads(str(value)))
