"""
Picks reference values for one year and joins them to GDELT country-days.

The year is always passed in explicitly. Nothing here looks up "the latest
available year", so a replay of 2020 uses 2020 values (or whatever year the
caller chooses) and gives the same answer whenever it is run.

Which reference year belongs to an assessment date is a method choice, not
a coding one: V-Dem and WGI publish a year's values in the following
spring, so a real-time assessment in mid-2020 could only have seen 2019
values. The caller decides; see the scoring settings.
"""

from __future__ import annotations

import pandas as pd

from src.reference.variables import VARIABLES


def for_year(reference: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    One row per country for the given year, with two columns per
    variable: <variable> (the value) and <variable>_sd.

    Every variable in VARIABLES gets its columns, even if the year has no
    data for it, so the output shape never depends on the data.
    """
    rows = reference[reference["year"] == year]
    wide = rows.pivot(index="iso3", columns="variable", values="value")
    sds = rows.pivot(index="iso3", columns="variable", values="sd")
    sds.columns = [f"{c}_sd" for c in sds.columns]
    wide = wide.join(sds)

    ordered: list[str] = []
    for name in VARIABLES:
        ordered += [name, f"{name}_sd"]
    wide = wide.reindex(columns=ordered)
    wide.columns.name = None
    wide = wide.reset_index().sort_values("iso3", ignore_index=True)
    wide.attrs["year"] = year
    return wide


def join_reference(
        country_day: pd.DataFrame,
        reference_year: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Add the reference columns to a country-day table, matching on iso3.

    Every country-day row is kept. A country with no reference data gets
    missing values, and the number of such countries is recorded in
    table.attrs["reference_coverage"].
    """
    joined = country_day.merge(reference_year, on="iso3", how="left")

    names = list(VARIABLES)
    has_all = joined[names].notna().all(axis=1)
    countries = joined["iso3"].nunique()
    without_any = joined.loc[joined[names].isna().all(axis=1), "iso3"]
    joined.attrs["reference_coverage"] = {
        "reference_year": reference_year.attrs.get("year"),
        "countries": countries,
        "with_all_variables": joined.loc[has_all, "iso3"].nunique(),
        "with_no_variables": without_any.nunique(),
        "events_without_reference": int(
            joined.loc[~has_all, "n_events"].sum()
        ),
        "events_total": int(joined["n_events"].sum()),
    }
    return joined
