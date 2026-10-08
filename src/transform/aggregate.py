"""
Turns GDELT event rows into one row per country per day.

Input:  the table from read_events() or read_day(), one row per event.
Output: one row per (iso3, date) with plain measures only.

This module stores counts and averages. It does not produce scores or
levels. Those belong to the scoring layer, so the scoring design can change
without touching this code.

Things to note:
    - The date is the day GDELT recorded the event (DATEADDED, in UTC),
      not the Day column. Day can be a week, a month or a year earlier
      when an article talks about a past event.
    - Rows with no country are dropped, and the number dropped is recorded.
    - SOURCEURL is not carried into the output.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.countries.codes import to_iso3
from src.gdelt.schema import CAMEO_ROOT_LABELS, QUAD_CLASS_LABELS

logger = logging.getLogger(__name__)

# Columns this module needs from the events table
REQUIRED_COLUMNS: tuple[str, ...] = (
    "GlobalEventID",
    "DATEADDED",
    "ActionGeo_CountryCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumArticles",
    "AvgTone",
)

ROOT_CODES: list[str] = sorted(CAMEO_ROOT_LABELS)     # "01" to "20"
QUAD_CLASSES: list[int] = sorted(QUAD_CLASS_LABELS)   # 1 to 4


def to_country_day(events: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate event rows into a country-day table.

    Output columns:
        iso3, date          the country and the day (UTC)
        n_events            number of events
        n_mentions          total mentions of those events
        n_articles          total articles about those events
        tone_mean           plain average of AvgTone
        tone_mean_weighted  average of AvgTone, weighted by NumArticles
        goldstein_mean      plain average of GoldsteinScale
        quad_1 .. quad_4    number of events in each QuadClass
        root_01 .. root_20  number of events with each CAMEO root code
        root_other          events whose root code is not one of the 20

    The returned table also carries a small record of what was dropped,
    in table.attrs["dropped"].
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events table is missing columns: {missing}")

    rows_in = len(events)

    # Work on a copy so the caller's table is not changed.
    work = events[list(REQUIRED_COLUMNS)].copy()

    # --- 1. Remove repeated events ---
    # Each GlobalEventID should appear once. 
    # A repeat means the same file was read twice, and counting it twice would 
    # inflate that day.
    work = work.drop_duplicates(subset="GlobalEventID")
    rows_duplicate = rows_in - len(work)

    # --- 2. Country: FIPS -> ISO3 ---
    work["iso3"] = to_iso3(work["ActionGeo_CountryCode"])

    # --- 3. Date: first 8 characters of DATEADDED (YYYYMMDDHHMMSS) ---
    # errors="coerce" turns anything that is not a real date into a missing 
    # value, so one bad row cannot stop the run.
    work["date"] = pd.to_datetime(
        work["DATEADDED"].astype("string").str.slice(0, 8),
        format="%Y%m%d",
        errors="coerce",
    )

    # --- 4. Drop rows that cannot be placed in a country-day ---
    no_country = work["iso3"].isna()
    bad_date = work["date"].isna() & ~no_country
    work = work[~no_country & ~bad_date].copy()

    dropped = {
        "rows_in": rows_in,
        "duplicate_event": int(rows_duplicate),
        "no_country": int(no_country.sum()),
        "bad_date": int(bad_date.sum()),
        "rows_used": len(work),
    }
    logger.info("country-day aggregation: %s", dropped)

    keys = ["iso3", "date"]

    # --- 5. Totals and plain averages ---
    table = work.groupby(keys).agg(
        n_events=("GlobalEventID", "size"),
        n_mentions=("NumMentions", "sum"),
        n_articles=("NumArticles", "sum"),
        tone_mean=("AvgTone", "mean"),
        goldstein_mean=("GoldsteinScale", "mean"),
    )

    # --- 6. Tone weighted by number of articles ---
    # An event with no tone value gets weight 0 so it does not count.
    # weighted mean = sum(tone * articles) / sum(articles)
    weight = work["NumArticles"].where(work["AvgTone"].notna(), 0)
    work["_tone_x_weight"] = work["AvgTone"] * weight
    work["_weight"] = weight
    sums = work.groupby(keys)[["_tone_x_weight", "_weight"]].sum()
    table["tone_mean_weighted"] = sums["_tone_x_weight"] / sums["_weight"]

    # --- 7. Event counts per QuadClass ---
    # `crosstab` counts how many rows fall in each (country-day, class) cell.
    # `reindex` makes sure all four columns exist, even if a class is absent.
    quad = pd.crosstab([work["iso3"], work["date"]], work["QuadClass"])
    quad = quad.reindex(columns=QUAD_CLASSES, fill_value=0)
    quad.columns = [f"quad_{q}" for q in QUAD_CLASSES]

    # --- 8. Event counts per CAMEO root code ---
    root = pd.crosstab([work["iso3"], work["date"]], work["EventRootCode"])
    root = root.reindex(columns=ROOT_CODES, fill_value=0)
    root.columns = [f"root_{code}" for code in ROOT_CODES]

    # A country-day whose events all lack a class or root code would be
    # missing from the crosstab, so line both up with the main table.
    quad = quad.reindex(table.index, fill_value=0)
    root = root.reindex(table.index, fill_value=0)

    table = table.join(quad).join(root)

    # Events with a missing or unusual root code, so that nothing is lost:
    # n_events == root_01 + ... + root_20 + root_other
    table["root_other"] = table["n_events"] - root.sum(axis=1)

    # --- 9. Tidy up ---
    column_order = (
        ["n_events", "n_mentions", "n_articles",
         "tone_mean", "tone_mean_weighted", "goldstein_mean"]
        + list(quad.columns)
        + list(root.columns)
        + ["root_other"]
    )
    table = table[column_order].reset_index()
    table = table.sort_values(["date", "iso3"], ignore_index=True)

    table.attrs["dropped"] = dropped
    return table
