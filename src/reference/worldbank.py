"""
Loads GDP per capita data from a World Bank CSV download.

GDP per capita is used as an independent check on the country
scores. It is not included in the scoring calculations.

The World Bank download contains three files. Extract all three
into data/reference/worldbank/:

    - The main CSV file containing country data.
    - Country metadata identifying countries and regions.
    - Indicator metadata (not used by this script).

The script:

    1. Reads the country data and identifies the indicators used
       by this project.
    2. Removes regional and income-group totals, keeping individual
       countries only.
    3. Checks country codes and handles known exceptions.
    4. Converts the data from one column per year into one row
       per country, indicator and year.
    5. Keeps data from min_year onwards (2000 by default).
    6. Returns the data in the standard reference table format.

The World Bank does not provide uncertainty estimates for these
values, so standard errors and confidence intervals are left empty.

Main function:

    load_worldbank()
        Reads the downloaded files, cleans the data and returns
        the country-level reference data.
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

CODE = "Country Code"
NAME = "Country Name"
INDICATOR = "Indicator Code"
REGION = "Region"
HEADER_LINES = 4

# Drop economies with no ISO3 equivalent
DROPPED: dict[str, str] = {
    "CHI": "Channel Islands",
    "CUW": "Curacao",
    "SXM": "Sint Maarten (Dutch part)",
    "INX": "Not classified",
}


def metadata_path(values_path: Path) -> Path:
    """
    The country metadata file that came in the same zip.
    """
    pattern = f"Metadata_Country_{values_path.name}"
    found = sorted(values_path.parent.glob(pattern))

    if len(found) != 1:
        raise FileNotFoundError(
            f"expected Metadata_Country_{values_path.name} next to the "
            "values file: unzip all three files from the download"
        )
    
    return found[0]


def load_worldbank(path: Path | str, min_year: int = 2000) -> pd.DataFrame:
    """
    Read the World Bank variables this project uses into the long
    reference table (see src/reference/common.py).
    """
    path = Path(path)
    values = pd.read_csv(
        path, skiprows=HEADER_LINES, keep_default_na=False, na_values=[""],
        dtype={CODE: "string", NAME: "string", INDICATOR: "string"},
    )

    values = values.loc[:, ~values.columns.str.startswith("Unnamed")]
    countries = pd.read_csv(
        metadata_path(path), keep_default_na=False, na_values=[""],
        dtype={CODE: "string"},
    )

    totals = set(countries.loc[countries[REGION].isna(), CODE])

    frames: list[pd.DataFrame] = []

    for variable in variables_from("wb"):
        rows = values[values[INDICATOR] == variable.source_name]

        if rows.empty:
            raise ValueError(
                f"{path.name} holds no {variable.source_name} rows: is it "
                "the right download?"
            )
        
        rows = rows[~rows[CODE].isin(totals)]
        keep = check_codes(rows[CODE], rows[NAME], DROPPED, "World Bank")
        rows = rows[keep]

        years = [c for c in rows.columns if c.isdigit()]
        long = rows.melt(id_vars=[CODE], value_vars=years,
                         var_name="year", value_name="value")
        
        long["year"] = long["year"].astype("int64")
        long = long[long["year"] >= min_year]

        frames.append(pd.DataFrame({
            "iso3": long[CODE].astype(str),
            "year": long["year"],
            "variable": variable.name,
            "source": "wb",
            "value": pd.to_numeric(long["value"], errors="coerce"),
            "sd": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "interval": float("nan"),
        }))

    table = pd.concat(frames, ignore_index=True)
    table = table.dropna(subset=["value"])
    check_unique(table, "World Bank")
    
    return table[REFERENCE_COLUMNS].sort_values(
        ["variable", "year", "iso3"], ignore_index=True
    )
