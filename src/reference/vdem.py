"""
Loads the V-Dem country-year data used by this project.

The original V-Dem file contains thousands of columns. This loader reads
only the country code, country name, year, and the four data fields needed
for each V-Dem variable listed in src/reference/variables.py:
    - value: the estimated measurement
    - sd: the standard deviation
    - low: the lower uncertainty bound
    - high: the upper uncertainty bound

Important details:
    - V-Dem uses its own country codes. Codes that do not match this
      project's ISO3 codes must be listed in DROPPED or the load stops.
    - Rows before min_year are excluded. This also removes historical
      states that do not have an ISO3 code.
    - The uncertainty interval for the low/high bounds is not recorded.

The loader converts the selected data into the standard long reference
table, removes rows without a value, checks for duplicate entries, and
returns the result sorted by variable, year and country.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reference.common import (
    REFERENCE_COLUMNS,
    check_codes,
    check_unique,
)
from src.reference.variables import variables_from

CODE = "country_text_id"
NAME = "country_name"
YEAR = "year"
SUFFIXES = {"value": "", "sd": "_sd", "low": "_codelow", "high": "_codehigh"}

# V-Dem codes with no ISO3 equivalent in this project
# This list will be filled in as we progress and perform more runs
DROPPED: dict[str, str] = {
    # GDELT and WGI treat these as part of Somalia (SOM) and Tanzania (TZA)
    "SML": "Somaliland",
    "ZZB": "Zanzibar",

    # GDELT (GZ and WE) and WGI combine West Bank & Gaza as PSE
    "PSG": "Palestine/Gaza",
}


def load_vdem(path: Path | str, min_year: int = 2000) -> pd.DataFrame:
    """
    Read the V-Dem variables this project uses into the long reference
    table (see src/reference/common.py). Years before min_year are left
    out.
    """
    variables = variables_from("vdem")
    value_columns = [
        f"{v.source_name}{suffix}"
        for v in variables
        for suffix in SUFFIXES.values()
    ]

    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in [CODE, NAME, YEAR, *value_columns]
               if c not in header]
    if missing:
        raise ValueError(
            f"V-Dem file is missing columns {missing}. Check the variable "
            "names against this version's codebook."
        )

    frame = pd.read_csv(
        path,
        usecols=[CODE, NAME, YEAR, *value_columns],
        dtype={CODE: "string", NAME: "string"},
        keep_default_na=False,
        na_values={c: ["", "NA"] for c in value_columns},
        low_memory=False,
    )

    frame = frame[frame[YEAR] >= min_year]
    frame[CODE] = frame[CODE].str.strip()
    keep = check_codes(frame[CODE], frame[NAME], DROPPED, "V-Dem")
    frame = frame[keep]

    frames: list[pd.DataFrame] = []
    for variable in variables:
        columns = {
            field: pd.to_numeric(
                frame[f"{variable.source_name}{suffix}"], errors="coerce"
            )
            for field, suffix in SUFFIXES.items()
        }

        frames.append(pd.DataFrame({
            "iso3": frame[CODE].astype(str),
            "year": frame[YEAR].astype("int64"),
            "variable": variable.name,
            "source": "vdem",
            **columns,
            "interval": float("nan"),
        }))

    table = pd.concat(frames, ignore_index=True)
    table = table.dropna(subset=["value"])
    
    check_unique(table, "V-Dem")

    return table[REFERENCE_COLUMNS].sort_values(
        ["variable", "year", "iso3"], ignore_index=True
    )
