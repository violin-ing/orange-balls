"""
Column schema for the GDELT 2.0 Events table.

The export files are tab-delimited and ship without a header row, so these
61 names must be supplied at read time. 

The order is fixed by the GDELT 2.0 Event Codebook and must not be rearranged.

Source: 
http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf
"""

from __future__ import annotations

EVENT_COLUMNS: tuple[str, ...] = (
    # --- Event identification ---
    "GlobalEventID",
    "Day",
    "MonthYear",
    "Year",
    "FractionDate",
    # --- Actor 1 ---
    "Actor1Code",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor1KnownGroupCode",
    "Actor1EthnicCode",
    "Actor1Religion1Code",
    "Actor1Religion2Code",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",
    # --- Actor 2 ---
    "Actor2Code",
    "Actor2Name",
    "Actor2CountryCode",
    "Actor2KnownGroupCode",
    "Actor2EthnicCode",
    "Actor2Religion1Code",
    "Actor2Religion2Code",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",
    # --- Event action ---
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    # --- Actor 1 geography ---
    "Actor1Geo_Type",
    "Actor1Geo_FullName",
    "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code",
    "Actor1Geo_ADM2Code",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    # --- Actor 2 geography ---
    "Actor2Geo_Type",
    "Actor2Geo_FullName",
    "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code",
    "Actor2Geo_ADM2Code",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    # --- Action geography (where the event happened) ---
    "ActionGeo_Type",
    "ActionGeo_FullName",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_ADM2Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "ActionGeo_FeatureID",
    # --- Data management ---
    "DATEADDED",
    "SOURCEURL",
)

# Subset of columns that the country-day aggregation actually needs
USED_COLUMNS: tuple[str, ...] = (
    "GlobalEventID",
    "Day",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumArticles",
    "AvgTone",
    "ActionGeo_CountryCode",
    "SOURCEURL",
)

# CAMEO event root codes
# Root code is the first 2 digits of EventCode (gives the coarse category) 
CAMEO_ROOT_LABELS: dict[str, str] = {
    "01": "Make public statement",
    "02": "Appeal",
    "03": "Express intent to cooperate",
    "04": "Consult",
    "05": "Engage in diplomatic cooperation",
    "06": "Engage in material cooperation",
    "07": "Provide aid",
    "08": "Yield",
    "09": "Investigate",
    "10": "Demand",
    "11": "Disapprove",
    "12": "Reject",
    "13": "Threaten",
    "14": "Protest",
    "15": "Exhibit force posture",
    "16": "Reduce relations",
    "17": "Coerce",
    "18": "Assault",
    "19": "Fight",
    "20": "Use unconventional mass violence",
}

# QuadClass groups the root codes into 4 bands
# Useful as a cooperation/conflict signal
QUAD_CLASS_LABELS: dict[int, str] = {
    1: "Verbal cooperation",
    2: "Material cooperation",
    3: "Verbal conflict",
    4: "Material conflict",
}


def assert_schema_length() -> None:
    """
    Guard against accidental edits to EVENT_COLUMNS.
    """
    if len(EVENT_COLUMNS) != 61:
        raise AssertionError(
            f"GDELT 2.0 Events has 61 columns, schema has {len(EVENT_COLUMNS)}"
        )
