"""
Loads the Worldwide Governance Indicators (WGI) data from an Excel file.

Each governance dimension has its own sheet. The loader reads only the
dimensions listed in src/reference/variables.py and the columns needed
for each one:
    - country code and name
    - year
    - governance score (0 - 100)
    - standard error of the governance score

Things to note:
    - WGI marks missing values with "..".
    - The standard error is used to calculate the lower and upper
      uncertainty bounds. These represent a 90% interval, using
      (score ± 1.645 * standard error).
    - WGI does not publish data for every year. Available years start
      with 1996, 1998 and 2000, followed by annual data from 2002.

Rows before min_year are excluded. Country codes are checked, rows
without a score are removed, and duplicate country-year-variable
records are rejected.

The result is returned in the standard long reference table format
(see src/reference/common.py).
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

CODE = "Economy (code)"
NAME = "Economy (name)"
YEAR = "Year"
SCORE = "Governance score (0-100)"
SCORE_ERROR = "Standard error (gov. score)"
INTERVAL = 0.90
Z_90 = 1.6448536269514722 # Z-value for 90% CI

DROPPED: dict[str, str] = {
    "ANT": "Netherlands Antilles (dissolved in 2010)",
}


def load_wgi(path: Path | str, min_year: int = 2000) -> pd.DataFrame:
    """
    Read the WGI dimensions this project uses into the long reference table
    (see src/reference/common.py). Years before min_year are excluded.
    """
    frames: list[pd.DataFrame] = []
    for variable in variables_from("wgi"):
        sheet = pd.read_excel(
            path,
            sheet_name=variable.source_name,
            keep_default_na=False, # stops "NA" from being read as missing
            na_values=[".."], # WGI's missing marker is ".."
        )
        needed = [CODE, NAME, YEAR, SCORE, SCORE_ERROR]
        missing = [c for c in needed if c not in sheet.columns]
        if missing:
            raise ValueError(
                f"WGI sheet {variable.source_name!r} is missing columns "
                f"{missing}. The file layout may have changed."
            )

        sheet = sheet[sheet[YEAR] >= min_year]
        sheet[CODE] = sheet[CODE].astype(str).str.strip()
        keep = check_codes(sheet[CODE], sheet[NAME], DROPPED, "WGI")
        sheet = sheet[keep]

        score = pd.to_numeric(sheet[SCORE], errors="coerce")
        error = pd.to_numeric(sheet[SCORE_ERROR], errors="coerce")
        frames.append(pd.DataFrame({
            "iso3": sheet[CODE],
            "year": sheet[YEAR].astype("int64"),
            "variable": variable.name,
            "source": "wgi",
            "value": score,
            "sd": error,
            "low": score - Z_90 * error,
            "high": score + Z_90 * error,
            "interval": INTERVAL,
        }))

    table = pd.concat(frames, ignore_index=True)
    table = table.dropna(subset=["value"])

    check_unique(table, "WGI")

    return table[REFERENCE_COLUMNS].sort_values(
        ["variable", "year", "iso3"], ignore_index=True
    )
