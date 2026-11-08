"""
Turns raw GDELT event records into daily summary tables for each country.

The input is an event table from read_events() or read_day(), with one
row per event.

The module produces two output tables:
    - to_country_day(): one row per country per day, containing event
      totals and the measurements needed by the scoring criteria.
    - to_event_code_day(): one row per country, day, event code,
      relationship, media-target status and lead-paragraph status,
      preserving the full CAMEO event code detail.

This module only counts events and adds up measurements. It does not
calculate percentages, weights, scores or overall levels. Those
calculations belong in the scoring module.

Countries are counted in two different ways:
    - By location: events that happened inside the country. Used for
      general totals, tone, domestic unrest and pressure on the media.
    - By actor: events in which the country took part, even if they
      happened somewhere else. Used for international conflict and
      cooperation.

GoldsteinScale is not a measure of how severe an individual event was.
It is a value assigned to the event type by GDELT. The module stores
Goldstein totals alongside event counts, and keeps the full event-code
breakdown so the scoring rules can distinguish specific types of events.

Important details:
    - The event date is taken from DATEADDED, when GDELT recorded the
      event, rather than the Day field, which may refer to an earlier
      event mentioned in an article.
    - Each criterion stores event counts, article counts and Goldstein
      totals. The scoring module decides which measure to use.
    - SOURCEURL is not included in the output.
    - IsRootEvent (1 = the event was taken from an article's lead
      paragraph) is kept as the is_root column of the event-code table,
      so scoring can use lead-paragraph events only (a standard GDELT
      noise filter). The country-day table counts all events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.countries.actor_codes import (
    actor_to_iso3,
    clean_actor_codes,
    unmatched_actor_codes,
)
from src.countries.codes import to_iso3
from src.gdelt.schema import CAMEO_ROOT_LABELS, QUAD_CLASS_LABELS

logger = logging.getLogger(__name__)

# Columns needed from the events table
REQUIRED_COLUMNS: tuple[str, ...] = (
    "GlobalEventID",
    "DATEADDED",
    "ActionGeo_CountryCode",
    "Actor1CountryCode",
    "Actor2CountryCode",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",
    "EventCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumArticles",
    "AvgTone",
    "IsRootEvent",
)

ROOT_CODES: list[str] = sorted(CAMEO_ROOT_LABELS) # "01" to "20"
QUAD_CLASSES: list[int] = sorted(QUAD_CLASS_LABELS) # 1 to 4


# === EVENT SELECTION RULES ===

# Domestic unrest codes...
# Strict definition: protest, assault, fighting and mass violence
UNREST_ROOT_CODES: frozenset[str] = frozenset({"14", "18", "19", "20"})

# Broader definition: also includes state coercion (code 17)
UNREST_LOOSE_ROOT_CODES: frozenset[str] = UNREST_ROOT_CODES | {"17"}


# Pressure on media codes...
# Includes disapproval, rejection, threats, coercion and assault
MEDIA_PRESSURE_ROOT_CODES: frozenset[str] = frozenset(
    {"11", "12", "13", "17", "18"}
)
MEDIA_TYPE_CODE = "MED"

# Intergovernmental organisations (IGOs), e.g. UN, NATO, EU
IGO_TYPE_CODE = "IGO"

# QuadClasses:
# > 3 and 4 = conflict
# > 1 and 2 = cooperation
CONFLICT_QUAD_CLASSES: frozenset[int] = frozenset({3, 4})
COOPERATION_QUAD_CLASSES: frozenset[int] = frozenset({1, 2})

# How event actors relate to the country where the event happened:
# 1. domestic: actor 1 is local; actor 2 is local or absent
# 2. no_country: no foreign actor, but the event is not domestic
# 3. foreign: at least one actor is foreign
RELATION_DOMESTIC = "domestic"
RELATION_NO_COUNTRY = "no_country"
RELATION_FOREIGN = "foreign"

# Measures calculated for events by location and by actor
RELATIONS: list[str] = [
    RELATION_DOMESTIC, RELATION_NO_COUNTRY, RELATION_FOREIGN,
]

# Each criterion has three measures:
# 1. events = number of events
# 2. articles = number of articles
# 3. goldstein = sum of GoldsteinScale values
LOCATION_COUNTS: list[str] = ["unrest", "unrest_loose", "media_pressure"]
ACTOR_COUNTS: list[str] = ["actor", "intl_conflict", "intl_coop"]
COUNT_KINDS: tuple[str, ...] = ("events", "articles", "goldstein")

# Columns in the detailed event-code output table.
EVENT_CODE_COLUMNS: list[str] = [
    "iso3", "date", "event_code", "relation", "media_target", "is_root",
    "n_events", "n_articles", "goldstein_sum",
]


def criterion_columns() -> list[str]:
    """
    Every criterion column name, in output order.
    """
    return [
        f"{name}_{kind}"
        for name in LOCATION_COUNTS + ACTOR_COUNTS
        for kind in COUNT_KINDS
    ]


def _has_type(work: pd.DataFrame, actor: str, type_code: str) -> pd.Series:
    """
    True where the actor ("1" or "2") has type_code in any type slot.
    """
    has = pd.Series(False, index=work.index)
    for slot in ("Type1Code", "Type2Code", "Type3Code"):
        has |= (work[f"Actor{actor}{slot}"] == type_code).fillna(False)
    return has.astype(bool)


@dataclass
class _Prepared:
    """
    The cleaned events shared by both public functions.
    """
    work: pd.DataFrame # one row per event with a valid date
    dropped: dict[str, int] # what was left out, and why
    unmatched: dict[str, int] # actor codes that are not a country


def _prepare(events: pd.DataFrame) -> _Prepared:
    """
    Check the input, remove repeats, convert countries and dates, and work
    out how each event's actors relate to its location.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events table is missing columns: {missing}")

    rows_in = len(events)

    # Work on a copy
    work = events[list(REQUIRED_COLUMNS)].copy()

    # --- 1. Remove repeated events ---

    # Each GlobalEventID should appear once
    work = work.drop_duplicates(subset="GlobalEventID")
    rows_duplicate = rows_in - len(work)


    # --- 2. Countries ---
    
    # Location: FIPS -> ISO3; Actors: CAMEO -> ISO3
    work["iso3"] = to_iso3(work["ActionGeo_CountryCode"])
    work["actor1"] = actor_to_iso3(work["Actor1CountryCode"])
    work["actor2"] = actor_to_iso3(work["Actor2CountryCode"])

    unmatched: dict[str, int] = {}
    for column in ("Actor1CountryCode", "Actor2CountryCode"):
        for code, count in unmatched_actor_codes(work[column]).items():
            unmatched[code] = unmatched.get(code, 0) + count
    unmatched = dict(sorted(unmatched.items(), key=lambda kv: -kv[1]))


    # --- 3. Date: first 8 characters of DATEADDED (YYYYMMDDHHMMSS) ---

    # errors="coerce" converts malformed dates into missing values
    work["date"] = pd.to_datetime(
        work["DATEADDED"].astype("string").str.slice(0, 8),
        format="%Y%m%d",
        errors="coerce",
    )
    bad_date = work["date"].isna()
    work = work[~bad_date].copy()


    # --- 4. How the actors relate to the location ---

    work["actor1_igo"] = _has_type(work, "1", IGO_TYPE_CODE)
    work["actor2_igo"] = _has_type(work, "2", IGO_TYPE_CODE)

    # An actor is "foreign" when it has a country code that is not the
    # location, or is an IGO (a blank code is not foreign)
    actor1_blank = clean_actor_codes(work["Actor1CountryCode"]) == ""
    actor2_blank = clean_actor_codes(work["Actor2CountryCode"]) == ""
    actor1_local = (work["actor1"] == work["iso3"]).fillna(False)
    actor2_local = (work["actor2"] == work["iso3"]).fillna(False)
    actor1_foreign = (~actor1_blank & ~actor1_local) | work["actor1_igo"]
    actor2_foreign = (~actor2_blank & ~actor2_local) | work["actor2_igo"]

    no_foreign = ~actor1_foreign & ~actor2_foreign
    domestic = actor1_local & ~actor2_foreign
    work["relation"] = RELATION_FOREIGN
    work.loc[no_foreign, "relation"] = RELATION_NO_COUNTRY
    work.loc[domestic, "relation"] = RELATION_DOMESTIC

    # The target (actor 2) is coded as media in any of its type slots
    work["media_target"] = _has_type(work, "2", MEDIA_TYPE_CODE)

    # Taken from an article's lead paragraph (IsRootEvent = 1)
    # Anything else (e.g. missing value) counts as not
    is_root = pd.to_numeric(work["IsRootEvent"], errors="coerce") == 1
    work["is_root"] = is_root.fillna(False).astype(bool)

    # Missing Goldstein values add nothing to a sum
    work["_goldstein"] = work["GoldsteinScale"].fillna(0.0)

    dropped = {
        "rows_in": rows_in,
        "duplicate_event": int(rows_duplicate),
        "no_country": int(work["iso3"].isna().sum()),
        "bad_date": int(bad_date.sum()),
        "rows_used": int(work["iso3"].notna().sum()),
    }

    return _Prepared(work=work, dropped=dropped, unmatched=unmatched)


def _criterion_sums(
        frame: pd.DataFrame,
        keys: list[str],
        flags: dict[str, pd.Series],
    ) -> pd.DataFrame:
    """
    For each flag, count the flagged events, sum their articles and sum
    their Goldstein values, grouped by the key columns.

    A flag is a True/False column saying whether each event counts.
    Gives three columns per flag: 
        <name>_events 
        <name>_articles
        <name>_goldstein
    """
    work = frame[keys].copy()
    for name, flag in flags.items():
        flag = flag.fillna(False).astype(bool)
        work[f"{name}_events"] = flag.astype("int64")

        # Articles and Goldstein count only where the flag is True
        work[f"{name}_articles"] = frame["NumArticles"].where(flag, 0)
        work[f"{name}_goldstein"] = frame["_goldstein"].where(flag, 0.0)

    return work.groupby(keys).sum()


def to_country_day(events: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise GDELT events for each country and day.

    Includes:
        - Total events, mentions and articles.
        - Average event tone and Goldstein scores.
        - Event counts by QuadClass and CAMEO root code.
        - Domestic unrest and media pressure measures.
        - International conflict and cooperation measures.

    Each criterion has three measures: event count, article count
    and Goldstein total.

    Countries involved in events are included even if no events
    were located inside their borders.

    Information about excluded rows and unmatched actor codes is
    stored in table.attrs.
    """
    prepared = _prepare(events)
    work = prepared.work
    logger.info("country-day aggregation: %s", prepared.dropped)

    root = work["EventRootCode"]
    target_is_media = work["media_target"]


    # --- Events that happened inside each country ---

    located = work[work["iso3"].notna()].copy()
    keys = ["iso3", "date"]

    # Calculate daily event totals and averages
    table = located.groupby(keys).agg(
        n_events=("GlobalEventID", "size"),
        n_mentions=("NumMentions", "sum"),
        n_articles=("NumArticles", "sum"),
        tone_mean=("AvgTone", "mean"),
        goldstein_mean=("GoldsteinScale", "mean"),
    )

    # Calculate average tone weighted by article counts (events without a tone
    # value receive no weight)
    weight = located["NumArticles"].where(located["AvgTone"].notna(), 0)
    located["_tone_x_weight"] = located["AvgTone"] * weight
    located["_weight"] = weight
    sums = located.groupby(keys)[["_tone_x_weight", "_weight"]].sum()
    table["tone_mean_weighted"] = sums["_tone_x_weight"] / sums["_weight"]

    # Count events by QuadClass and CAMEO root code
    index = [located["iso3"], located["date"]]

    quad_counts = pd.crosstab(index, located["QuadClass"])
    quad_counts = quad_counts.reindex(columns=QUAD_CLASSES, fill_value=0)
    quad_counts.columns = [f"quad_{q}" for q in QUAD_CLASSES]

    root_counts = pd.crosstab(index, located["EventRootCode"])
    root_counts = root_counts.reindex(columns=ROOT_CODES, fill_value=0)
    root_counts.columns = [f"root_{code}" for code in ROOT_CODES]

    # Keep all country-days, including those with missing event codes
    quad_counts = quad_counts.reindex(table.index, fill_value=0)
    root_counts = root_counts.reindex(table.index, fill_value=0)
    table = table.join(quad_counts).join(root_counts)

    # Count events with missing or unrecognised root codes
    table["root_other"] = table["n_events"] - root_counts.sum(axis=1)

    # Exclude media-targeted events from unrest to avoid double counting
    domestic_not_media = (
        (work["relation"] == RELATION_DOMESTIC) & ~target_is_media
    )

    # Calculate unrest and media-pressure measures
    location_sums = _criterion_sums(
        located,
        keys,
        {
            "unrest": root.isin(UNREST_ROOT_CODES) & domestic_not_media,
            "unrest_loose": (
                root.isin(UNREST_LOOSE_ROOT_CODES) & domestic_not_media
            ),
            "media_pressure": (
                root.isin(MEDIA_PRESSURE_ROOT_CODES) & target_is_media
            ),
        },
    )
    table = table.join(location_sums)


    # --- Events involving each country ---

    # Create one record for each country involved in an event (only count once
    # if both actors are the same country)
    different = (work["actor1"] != work["actor2"]).fillna(False)
    both_countries = work["actor1"].notna() & work["actor2"].notna()

    actor_columns = [
        "GlobalEventID", "date", "NumArticles", "QuadClass", "_goldstein",
    ]
    first = work.loc[work["actor1"].notna(), actor_columns].assign(
        iso3=work["actor1"],
        international=(both_countries & different) | work["actor2_igo"],
    )
    second = work.loc[work["actor2"].notna(), actor_columns].assign(
        iso3=work["actor2"],
        international=(both_countries & different) | work["actor1_igo"],
    )

    taking_part = pd.concat([first, second], ignore_index=True)
    taking_part = taking_part.drop_duplicates(
        subset=["GlobalEventID", "iso3"], ignore_index=True
    )

    # Count international conflict and cooperation events
    is_cross_border = taking_part["international"].astype(bool)
    part_quad = taking_part["QuadClass"]
    actor_sums = _criterion_sums(
        taking_part,
        keys,
        {
            "actor": pd.Series(True, index=taking_part.index),
            "intl_conflict": (
                is_cross_border & part_quad.isin(CONFLICT_QUAD_CLASSES)
            ),
            "intl_coop": (
                is_cross_border & part_quad.isin(COOPERATION_QUAD_CLASSES)
            ),
        },
    )


    # --- Combine location and actor measures ---

    # Keep countries that appear in either set of measures
    table = table.join(actor_sums, how="outer")

    # Replace missing counts and sums with zero
    int_columns = (
        ["n_events", "n_mentions", "n_articles"]
        + list(quad_counts.columns)
        + list(root_counts.columns)
        + ["root_other"]
        + [c for c in criterion_columns() if not c.endswith("_goldstein")]
    )

    goldstein_columns = [
        c for c in criterion_columns() if c.endswith("_goldstein")
    ]

    table[int_columns] = table[int_columns].fillna(0).astype("int64")
    table[goldstein_columns] = table[goldstein_columns].fillna(0.0)

    # Arrange columns and sort by date and country
    column_order = (
        ["n_events", "n_mentions", "n_articles",
         "tone_mean", "tone_mean_weighted", "goldstein_mean"]
        + list(quad_counts.columns)
        + list(root_counts.columns)
        + ["root_other"]
        + criterion_columns()
    )

    table = table[column_order].reset_index()
    table = table.sort_values(["date", "iso3"], ignore_index=True)

    # Attach info about excluded rows and unmatched actor codes
    table.attrs["dropped"] = prepared.dropped
    table.attrs["actor_codes_unmatched"] = prepared.unmatched

    return table


def to_event_code_day(events: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise located events by full CAMEO event code.

    One row per country, day, event code, actor relationship,
    media-target status and lead-paragraph status (is_root: 1 when
    GDELT took the event from an article's lead paragraph).

    Columns:
        n_events       number of events
        n_articles     total number of articles
        goldstein_sum  sum of GoldsteinScale values

    relation describes how the event's actors relate to the country:
        domestic    actor 1 is local; actor 2 is local or absent
        no_country  no foreign actor, but not domestic
        foreign     at least one actor is foreign

    Only events with a country location are included. Missing event
    codes are stored as an empty string.

    The table preserves the detail needed to check the event counts
    used in to_country_day().
    """

    # Prepare events and keep only those with a country location
    work = _prepare(events).work
    located = work[work["iso3"].notna()].copy()

    # Standardise event codes and media-target flags
    located["event_code"] = located["EventCode"].fillna("").astype(str)
    located["media_target"] = located["media_target"].astype("int64")
    located["is_root"] = located["is_root"].astype("int64")

    # Group by country, day, event code, actor relationship and flags
    keys = [
        "iso3", "date", "event_code", "relation", "media_target",
        "is_root",
    ]
    table = located.groupby(keys).agg(
        n_events=("GlobalEventID", "size"),
        n_articles=("NumArticles", "sum"),
        goldstein_sum=("_goldstein", "sum"),
    ).reset_index()

    # Store counts as integers and keep the defined column order
    table["n_events"] = table["n_events"].astype("int64")
    table["n_articles"] = table["n_articles"].astype("int64")
    table = table[EVENT_CODE_COLUMNS]

    return table.sort_values(keys, ignore_index=True)
