"""
Loads the V-Dem Country-Year (Full+Others) CSV file.

The file has over 4,000 columns. Only country_text_id, year and, for each
variable in src/reference/variables.py, four columns are read:

    <var>            the measurement model's point estimate
    <var>_sd         its standard deviation
    <var>_codelow    lower uncertainty bound
    <var>_codehigh   upper uncertainty bound

Things to note:
    - country_text_id is V-Dem's own three-letter code. It is mostly the
      same as ISO3, but V-Dem also covers places that are not separate
      countries in GDELT. Unknown codes stop the load (see DROPPED).
    - Years before min_year are left out. This also skips historical
      states (e.g. 19th-century German states) that have no ISO3 code.
    - The width of the _codelow/_codehigh interval is not recorded here
      (interval is left missing). Check the codebook's "Variable Versions
      and Suffixes" section before using those bounds; use sd meanwhile.
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

# V-Dem codes with no ISO3 equivalent in this project, dropped on purpose.
# Fill in from the first real run: the loader lists any unknown codes.
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
        # Only empty cells and "NA" count as missing, and only in the
        # number columns, so a country code is never read as missing
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
