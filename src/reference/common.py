"""
Shared pieces for the reference-data loaders.

Every loader returns the same long table, one row per country, year and
variable:

    iso3, year, variable, source, value, sd, low, high, interval

    value     the point estimate
    sd        its standard deviation (WGI: standard error) -- comparable
              across sources, so this is what uncertainty bands should use
    low/high  the source's own uncertainty bounds
    interval  how wide those bounds are, e.g. 0.90 for a 90% interval;
              missing if the source does not say

A long table means a new variable is a new set of rows, not a new column,
so the database never needs rebuilding when the criteria change.
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
    """Every ISO3 code this project uses (the GDELT crosswalk's codes)."""
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

    Like the GDELT crosswalk, this stops on purpose: a code that matches
    nothing would otherwise make a country silently disappear from the
    join.
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
    """Each (iso3, year, variable) must appear once."""
    repeats = table.duplicated(["iso3", "year", "variable"], keep=False)
    if repeats.any():
        sample = table.loc[repeats, ["iso3", "year", "variable"]].head(6)
        raise ValueError(
            f"{source}: repeated country-year-variable rows:\n"
            f"{sample.to_string(index=False)}"
        )
