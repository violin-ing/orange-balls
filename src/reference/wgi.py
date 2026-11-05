"""
Loads the Worldwide Governance Indicators (WGI) Excel file.

The file has one sheet per governance dimension (va, pv, ge, rq, rl, cc),
each with one row per economy and year. Only the dimensions listed in
src/reference/variables.py are read.

Things to note:
    - Missing values are written as "..".
    - Two scales are published. The estimate (about -2.5 to +2.5) is
      centred on each year's global average, so it is relative. The
      governance score (0-100) is the same estimate anchored to two fixed
      benchmark countries, so it is comparable across years. The score is
      used (2025 methodology revision, section 4.1).
    - Uncertainty: "Standard error (gov. score)". The file gives a 90%
      interval only for the estimate; the score is a straight-line
      rescaling of it, so its 90% interval is score -+ 1.645 x standard
      error (checked: the estimate's published interval is estimate -+
      1.645 x its standard error).
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
SCORE = "Governance score (0-100)"
SCORE_ERROR = "Standard error (gov. score)"
INTERVAL = 0.90
Z_90 = 1.6448536269514722     # normal quantile for a 90% interval

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
