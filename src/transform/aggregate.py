"""
Turns GDELT event rows into country-day tables.

Example GDELT event row (in CSV, but written in JSON here for clarity):
{
  "GlobalEventID": 987654321,
  "SQLDATE": 20261014,
  "Actor1Name": "UNITED STATES",
  "Actor1CountryCode": "USA",
  "Actor2Name": "CHINA",
  "Actor2CountryCode": "CHN",
  "EventCode": "040",
  "EventBaseCode": "040",
  "EventRootCode": "04",
  "QuadClass": 1,
  "GoldsteinScale": 2.8,
  "NumMentions": 15,
  "NumSources": 5,
  "NumArticles": 12,
  "AvgTone": 1.5,
  "ActionGeo_CountryCode": "CH",
  "ActionGeo_Lat": 39.9042,
  "ActionGeo_Long": 116.4074,
  "SOURCEURL": "https://poo.com/news"
}

Input:  the table from read_events() or read_day(), one row per event.
Output: two tables, both holding plain measures only.
    to_country_day()      one row per (iso3, date), the criteria inputs
    to_event_code_day()   one row per (iso3, date, event_code, relation),
                          the same events at full CAMEO code detail

This module stores counts and sums. It does not produce shares, weights,
scores or levels. Those belong to the scoring layer, so the scoring design
can change without touching this code.

A country gets measures in two different ways:
    - by LOCATION: events that happened in the country
      (ActionGeo_CountryCode). Used for totals, tone, unrest and pressure
      on media.
    - by ACTOR: events the country took part in, wherever they happened
      (Actor1CountryCode, Actor2CountryCode). Used for international
      conflict and cooperation.

Severity:
    GDELT gives no severity for a single event. GoldsteinScale is fixed by
    the CAMEO event code (a riot is always the same value, however many
    were hurt), so it measures how severe that TYPE of event usually is.
    Two things are stored so that severity can be applied in scoring:
    - a Goldstein sum next to every criterion count (<name>_goldstein)
    - the full event code table, so criteria can be defined at the level
      of e.g. "violent protest" (145) rather than "protest" (14)

Things to note:
    - The date is the day GDELT recorded the event (DATEADDED, in UTC),
      not the Day column. Day can be a week, a month or a year earlier
      when an article talks about a past event.
    - Every criterion is stored as a number of events, a number of articles
      and a Goldstein sum. The scoring layer picks which to use.
    - SOURCEURL is not carried into the output.
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

# Columns this module needs from the events table
REQUIRED_COLUMNS: tuple[str, ...] = (
    "GlobalEventID",
    "DATEADDED",
    "ActionGeo_CountryCode",
    "Actor1CountryCode",
    "Actor2CountryCode",
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
)

ROOT_CODES: list[str] = sorted(CAMEO_ROOT_LABELS) # "01" to "20"
QUAD_CLASSES: list[int] = sorted(QUAD_CLASS_LABELS) # 1 to 4

# === WHICH EVENTS COUNT TOWARDS WHAT CRITERION ===

# Domestic unrest: protest, coerce, assault, fight, mass violence
UNREST_ROOT_CODES: frozenset[str] = frozenset({"14", "17", "18", "19", "20"})

# Pressure on media: disapprove, reject, threaten, coerce, assault (the target 
# (actor 2) is coded as media)
MEDIA_PRESSURE_ROOT_CODES: frozenset[str] = frozenset(
    {"11", "12", "13", "17", "18"}
)
MEDIA_TYPE_CODE = "MED"

CONFLICT_QUAD_CLASSES: frozenset[int] = frozenset({3, 4})
COOPERATION_QUAD_CLASSES: frozenset[int] = frozenset({1, 2})

# How the actors of a located event relate to the place it happened
# Every located event gets exactly one of these
RELATION_DOMESTIC = "domestic" # actor 1 local; actor 2 local or absent
RELATION_NO_COUNTRY = "no_country" # no foreign actor, but not domestic
RELATION_FOREIGN = "foreign" # at least one actor is foreign
RELATIONS: list[str] = [
    RELATION_DOMESTIC, RELATION_NO_COUNTRY, RELATION_FOREIGN,
]

# Criterion columns, in output order
# Each name is stored three times:
# - <name>_events: number of events
# - <name>_articles: number of articles about those events
# - <name>_goldstein: sum of GoldsteinScale over those events
LOCATION_COUNTS: list[str] = ["unrest", "unrest_loose", "media_pressure"]
ACTOR_COUNTS: list[str] = ["actor", "intl_conflict", "intl_coop"]
COUNT_KINDS: tuple[str, ...] = ("events", "articles", "goldstein")

EVENT_CODE_COLUMNS: list[str] = [
    "iso3", "date", "event_code", "relation",
    "n_events", "n_articles", "goldstein_sum",
]


def criterion_columns() -> list[str]:
    """Every criterion column name, in output order."""
    return [
        f"{name}_{kind}"
        for name in LOCATION_COUNTS + ACTOR_COUNTS
        for kind in COUNT_KINDS
    ]


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

    # Work on a copy so the caller's table is not changed.
    work = events[list(REQUIRED_COLUMNS)].copy()

    # 1. Remove repeated events
    # Each GlobalEventID should appear once
    # A repeat means the same file was read twice, and counting it twice 
    # would inflate that day
    work = work.drop_duplicates(subset="GlobalEventID")
    rows_duplicate = rows_in - len(work)

    # 2. Countries
    # Location: FIPS -> ISO3
    # Actors: CAMEO -> ISO3
    work["iso3"] = to_iso3(work["ActionGeo_CountryCode"])
    work["actor1"] = actor_to_iso3(work["Actor1CountryCode"])
    work["actor2"] = actor_to_iso3(work["Actor2CountryCode"])

    unmatched: dict[str, int] = {}
    for column in ("Actor1CountryCode", "Actor2CountryCode"):
        for code, count in unmatched_actor_codes(work[column]).items():
            unmatched[code] = unmatched.get(code, 0) + count
    unmatched = dict(sorted(unmatched.items(), key=lambda kv: -kv[1]))

    # 3. Date: first 8 characters of DATEADDED (YYYYMMDDHHMMSS)
    # errors="coerce" turns anything that is not a real date into a missing 
    # value, so one bad row cannot stop the run
    work["date"] = pd.to_datetime(
        work["DATEADDED"].astype("string").str.slice(0, 8),
        format="%Y%m%d",
        errors="coerce",
    )
    bad_date = work["date"].isna()
    work = work[~bad_date].copy()

    # 4. How the actors relate to the location
    # An actor is "foreign" when it has a country code that is not the location
    # > A blank code is not foreign
    # > A code that is not a country (e.g. EUR for Europe is foreign)
    actor1_blank = clean_actor_codes(work["Actor1CountryCode"]) == ""
    actor2_blank = clean_actor_codes(work["Actor2CountryCode"]) == ""
    actor1_local = (work["actor1"] == work["iso3"]).fillna(False)
    actor2_local = (work["actor2"] == work["iso3"]).fillna(False)
    actor1_foreign = ~actor1_blank & ~actor1_local
    actor2_foreign = ~actor2_blank & ~actor2_local

    no_foreign = ~actor1_foreign & ~actor2_foreign
    domestic = actor1_local & ~actor2_foreign
    work["relation"] = RELATION_FOREIGN
    work.loc[no_foreign, "relation"] = RELATION_NO_COUNTRY
    work.loc[domestic, "relation"] = RELATION_DOMESTIC

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
    Gives three columns per flag: <name>_events, <name>_articles and
    <name>_goldstein.
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
    Aggregate event rows into a country-day table.

    Output columns:
        iso3, date          the country and the day (UTC)

      By location (events that happened in the country):
        n_events            number of events
        n_mentions          total mentions of those events
        n_articles          total articles about those events
        tone_mean           plain average of AvgTone
        tone_mean_weighted  average of AvgTone, weighted by NumArticles
        goldstein_mean      plain average of GoldsteinScale
        quad_1 .. quad_4    number of events in each QuadClass
        root_01 .. root_20  number of events with each CAMEO root code
        root_other          events whose root code is not one of the 20
        unrest_*            unrest events whose relation is "domestic"
        unrest_loose_*      unrest events with no foreign actor
                            ("domestic" or "no_country")
        media_pressure_*    pressure events whose target is media

      By actor (events the country took part in):
        actor_*             all events with this country as an actor
        intl_conflict_*     conflict events between two different countries
        intl_coop_*         cooperation events between two different
                            countries

    Each * is stored as _events, _articles and _goldstein.

    A country with actor events but no located events still gets a row,
    with n_events = 0 and missing averages.

    The returned table carries two records in table.attrs:
        "dropped"                what was left out of the location measures
        "actor_codes_unmatched"  actor codes that are not a known country
    """
    prepared = _prepare(events)
    work = prepared.work
    logger.info("country-day aggregation: %s", prepared.dropped)

    root = work["EventRootCode"]
    is_unrest_code = root.isin(UNREST_ROOT_CODES)
    target_is_media = (
        (work["Actor2Type1Code"] == MEDIA_TYPE_CODE).fillna(False)
        | (work["Actor2Type2Code"] == MEDIA_TYPE_CODE).fillna(False)
        | (work["Actor2Type3Code"] == MEDIA_TYPE_CODE).fillna(False)
    )


    # === LOCATION MEASURES ===
    
    located = work[work["iso3"].notna()].copy()
    keys = ["iso3", "date"]

    table = located.groupby(keys).agg(
        n_events=("GlobalEventID", "size"),
        n_mentions=("NumMentions", "sum"),
        n_articles=("NumArticles", "sum"),
        tone_mean=("AvgTone", "mean"),
        goldstein_mean=("GoldsteinScale", "mean"),
    )

    # Tone weighted by number of articles:
    # => weighted mean = sum(tone * articles) / sum(articles)
    # An event with no tone value gets weight 0 so it does not count
    weight = located["NumArticles"].where(located["AvgTone"].notna(), 0)
    located["_tone_x_weight"] = located["AvgTone"] * weight
    located["_weight"] = weight
    sums = located.groupby(keys)[["_tone_x_weight", "_weight"]].sum()
    table["tone_mean_weighted"] = sums["_tone_x_weight"] / sums["_weight"]

    # Event counts per QuadClass and per CAMEO root code
    # - crosstab counts the rows in each (country-day, code) cell
    # - reindex makes sure every column exists, even if a code did not occur
    index = [located["iso3"], located["date"]]
    quad_counts = pd.crosstab(index, located["QuadClass"])
    quad_counts = quad_counts.reindex(columns=QUAD_CLASSES, fill_value=0)
    quad_counts.columns = [f"quad_{q}" for q in QUAD_CLASSES]

    root_counts = pd.crosstab(index, located["EventRootCode"])
    root_counts = root_counts.reindex(columns=ROOT_CODES, fill_value=0)
    root_counts.columns = [f"root_{code}" for code in ROOT_CODES]

    # A country-day whose events all lack a class or root code would be
    # missing from the crosstab, so line both up with the main table
    quad_counts = quad_counts.reindex(table.index, fill_value=0)
    root_counts = root_counts.reindex(table.index, fill_value=0)
    table = table.join(quad_counts).join(root_counts)

    # Events with a missing or unusual root code, so that nothing is lost:
    # => n_events == root_01 + ... + root_20 + root_other
    table["root_other"] = table["n_events"] - root_counts.sum(axis=1)

    relation = work["relation"]
    location_sums = _criterion_sums(
        located,
        keys,
        {
            "unrest": is_unrest_code & (relation == RELATION_DOMESTIC),
            "unrest_loose": (
                is_unrest_code & (relation != RELATION_FOREIGN)
            ),
            "media_pressure": (
                root.isin(MEDIA_PRESSURE_ROOT_CODES) & target_is_media
            ),
        },
    )
    table = table.join(location_sums)


    # === ACTOR MEASURES ===

    # One row per (event, country taking part)
    # An event between two different countries gives two rows, one for each
    # An event where both actors are the same country gives one row
    work["cross_border"] = (
        work["actor1"].notna()
        & work["actor2"].notna()
        & (work["actor1"] != work["actor2"]).fillna(False)
    )
    actor_columns = [
        "GlobalEventID", "date", "NumArticles", "QuadClass", "_goldstein",
        "cross_border",
    ]
    first = work.loc[work["actor1"].notna(), actor_columns].assign(
        iso3=work["actor1"]
    )
    second = work.loc[work["actor2"].notna(), actor_columns].assign(
        iso3=work["actor2"]
    )
    taking_part = pd.concat([first, second], ignore_index=True)
    taking_part = taking_part.drop_duplicates(
        subset=["GlobalEventID", "iso3"], ignore_index=True
    )

    is_cross_border = taking_part["cross_border"].astype(bool)
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


    # === COMBINE THE TWO TOGETHER ===

    # how="outer" keeps a country that appears on only one side
    table = table.join(actor_sums, how="outer")

    # A country on only one side has no value for the other side's counts
    # and sums (for a count or a sum, "no value" = zero)
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

    table.attrs["dropped"] = prepared.dropped
    table.attrs["actor_codes_unmatched"] = prepared.unmatched
    return table


def to_event_code_day(events: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate located events by full CAMEO event code.

    One row per (iso3, date, event_code, relation), for every combination
    that occurred. Columns: n_events, n_articles, goldstein_sum.

    relation says how the actors relate to the location:
        "domestic"    actor 1 is local; actor 2 is local or absent
        "no_country"  no foreign actor, but not "domestic" (e.g. both blank)
        "foreign"     at least one actor is foreign

    So, for example, the strict unrest count in to_country_day() equals the
    sum of n_events over rows whose relation is "domestic" and whose code
    starts with an unrest root code. The tests check this.

    Only located events are included, the same ones as n_events in
    to_country_day(). A missing event code is stored as "".
    """
    work = _prepare(events).work
    located = work[work["iso3"].notna()].copy()
    located["event_code"] = located["EventCode"].fillna("").astype(str)

    keys = ["iso3", "date", "event_code", "relation"]
    table = located.groupby(keys).agg(
        n_events=("GlobalEventID", "size"),
        n_articles=("NumArticles", "sum"),
        goldstein_sum=("_goldstein", "sum"),
    ).reset_index()

    table["n_events"] = table["n_events"].astype("int64")
    table["n_articles"] = table["n_articles"].astype("int64")
    table = table[EVENT_CODE_COLUMNS]
    return table.sort_values(keys, ignore_index=True)
