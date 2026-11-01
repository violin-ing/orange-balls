"""
Loads the Worldwide Governance Indicators (WGI) Excel file.

The file has one sheet per governance dimension (va, pv, ge, rq, rl, cc),
each with one row per economy and year. Only the dimensions listed in
src/reference/variables.py are read.

Things to note:
    - Missing values are written as "..".
    - Uncertainty: "Standard error (estimate)" and a 90% interval
      ("Lower/Upper threshold (90% conf. int. estimate)").
    - Years are not continuous: 1996, 1998, 2000, then every year from
      2002.
    - The "with source data" edition also lists which outside sources
      feed each dimension (columns ending " mean"). They are not loaded
      here, but they answer which sources sit inside WGI.
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
ESTIMATE = "Governance estimate (approx. -2.5 to +2.5)"
STD_ERROR = "Standard error (estimate)"
LOWER = "Lower threshold (90% conf. int. estimate)"
UPPER = "Upper threshold (90% conf. int. estimate)"
INTERVAL = 0.90

# Codes in the file that are not countries this project scores
DROPPED: dict[str, str] = {
    "ANT": "Netherlands Antilles, dissolved in 2010",
}


def load_wgi(path: Path | str, min_year: int = 2000) -> pd.DataFrame:
    """
    Read the WGI dimensions this project uses into the long reference
    table (see src/reference/common.py). Years before min_year are left
    out.
    """
    frames: list[pd.DataFrame] = []
    for variable in variables_from("wgi"):
        sheet = pd.read_excel(
            path,
            sheet_name=variable.source_name,
            # keep_default_na=False stops codes such as "NA" being read as
            # missing; WGI's own missing marker is ".."
            keep_default_na=False,
            na_values=[".."],
        )
        needed = [CODE, NAME, YEAR, ESTIMATE, STD_ERROR, LOWER, UPPER]
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

        frames.append(pd.DataFrame({
            "iso3": sheet[CODE],
            "year": sheet[YEAR].astype("int64"),
            "variable": variable.name,
            "source": "wgi",
            "value": pd.to_numeric(sheet[ESTIMATE], errors="coerce"),
            "sd": pd.to_numeric(sheet[STD_ERROR], errors="coerce"),
            "low": pd.to_numeric(sheet[LOWER], errors="coerce"),
            "high": pd.to_numeric(sheet[UPPER], errors="coerce"),
            "interval": INTERVAL,
        }))

    table = pd.concat(frames, ignore_index=True)
    table = table.dropna(subset=["value"])
    check_unique(table, "WGI")
    return table[REFERENCE_COLUMNS].sort_values(
        ["variable", "year", "iso3"], ignore_index=True
    )
