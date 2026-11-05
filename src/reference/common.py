"""
Shared functions used by all reference-data loaders.

Different data sources provide country information in different
formats. This module makes sure they all use the same table
structure so their data can be stored and used together.

Each row represents one country, one year and one variable.

The standard table contains:

    iso3       Country code
    year       Year of the observation
    variable   Name of the indicator
    source     Where the data came from
    value      The measured value
    sd         Standard deviation or standard error
    low/high   Lower and upper uncertainty limits
    interval   Confidence level of those limits, where available

Using this format means new indicators can be added as new rows
without changing the database structure.

The module also checks country codes and prevents duplicate data:

    known_iso3()
        Returns the country codes recognised by the project.

    check_codes()
        Checks that every country code is recognised.
        Known non-country entries can be deliberately excluded.
        Unexpected codes stop the import rather than silently
        dropping countries.

    check_unique()
        Checks that each country, year and indicator combination
        appears only once.

These checks help ensure that reference data from different
sources can be safely combined.
"""

from __future__ import annotations

import pandas as pd

from src.countries.codes import load_crosswalk

REFERENCE_COLUMNS: list[str] = [
    "iso3", "year", "variable", "source",
    "value", "sd", "low", "high", "interval",
]


class UnknownReferenceCode(ValueError):
    """
    A dataset holds a country code that is neither a known ISO3 code nor
    on that loader's list of codes to drop.
    """


def known_iso3() -> set[str]:
    """
    Every ISO3 code this project uses (the GDELT crosswalk's codes).
    """
    return set(load_crosswalk().values())


def check_codes(
        codes: pd.Series,
        names: pd.Series,
        dropped: dict[str, str],
        source: str,
    ) -> pd.Series:
    """
    Return True for rows whose code is a known ISO3 code, False for rows
    whose code is on the dropped list, and raise for anything else.
    """
    recognised = known_iso3() | set(dropped)
    unknown = codes[~codes.isin(recognised)]
    if not unknown.empty:
        examples = (
            pd.DataFrame({"code": codes, "name": names})
            .loc[unknown.index]
            .drop_duplicates("code")
            .sort_values("code")
        )

        listing = ", ".join(
            f"{r.code} ({r.name})" for r in examples.itertuples()
        )

        raise UnknownReferenceCode(
            f"{source}: country codes that match no ISO3 code: {listing}. "
            "Add each to fips_iso3.csv if it is a country GDELT also "
            f"uses, or to the {source} loader's DROPPED list if not."
        )
    
    return codes.isin(known_iso3())


def check_unique(table: pd.DataFrame, source: str) -> None:
    """
    Each (iso3, year, variable) must appear once.
    """
    repeats = table.duplicated(["iso3", "year", "variable"], keep=False)

    if repeats.any():
        sample = table.loc[repeats, ["iso3", "year", "variable"]].head(6)
        
        raise ValueError(
            f"{source}: repeated country-year-variable rows:\n"
            f"{sample.to_string(index=False)}"
        )
