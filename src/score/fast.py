"""
Calculates four indicators of a country's political and social conditions
using GDELT event data, for a single assessment date.

The aim is to identify whether conditions in a country are unusual compared 
with what is normal for that country.

The four indicators are:
    1. unrest
        Measures domestic unrest events, excluding events aimed at the media. 
        Events can be weighted by severity, so more serious events contribute 
        more to the score.

    2. media_pressure
        Measures events involving pressure or threats directed at the media, 
        as a share of all located events.

    3. intl_conflict
        Measures conflict involving another country or an international 
        governmental organisation (IGO).

    4. intl_coop
        Measures cooperation involving another country or an IGO.

For each indicator, we count the relevant events (k) and divide by the 
appropriate total number of events (n). This gives us the observed share.

We then compare this with the country's historical pattern:
    - The baseline is calculated from the country's own history.
    - A small contribution from the overall share across all countries helps 
      when a country has limited historical data.
    - The baseline is used to estimate how many events we would normally expect
      on the assessment date.
    - The observed count is compared with that expectation.

The resulting score is interpreted as:
    1.0 = broadly in line with the country's normal pattern
    > 1.0 = more events than normally expected
    < 1.0 = fewer events than normally expected

Media pressure is treated differently by default because these events are 
relatively rare and often absent on individual days. Instead, its smoothed 
share over a longer period is used, allowing countries to be compared later.

Unrest counts only events from an article's lead paragraph (GDELT's 
IsRootEvent), in both the numerator and the denominator, so the filter does not
change the share merely by how much of a country's coverage is lead-paragraph
material. Setting `Settings.unrest_root_only` = `False` uses all events.

Unrest can use severity weights between 0 and 1. The weighted event count is 
used for the score, while the sum of squared weights is used when estimating 
uncertainty. With no weighting, this is equivalent to ordinary event counting.

The output includes the score, its standard error, the event counts used in the
calculation, the calculation method and a low-data flag.

Performance and separation of responsibilities:

    load_fast_inputs()  reads the required database data once and prepares the
                        event counts and history.

    fast_criteria()     calculates the four indicators using those prepared 
                        inputs (it performs no database reads or other 
                        input/output operations).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.score.settings import Settings
from src.store.database import (
    load_country_day,
    load_event_codes,
    load_pull_log,
)
from src.transform.aggregate import (
    RELATION_DOMESTIC,
    UNREST_LOOSE_ROOT_CODES,
    UNREST_ROOT_CODES,
)

FAST_CRITERIA: tuple[str, ...] = (
    "unrest", "media_pressure", "intl_conflict", "intl_coop",
)

# True when a high value = better conditions for an announcement
FAST_HIGHER_IS_BETTER: dict[str, bool] = {
    "unrest": False,
    "media_pressure": False,
    "intl_conflict": False,
    "intl_coop": True,
}

TIERS_PATH = Path(__file__).with_name("unrest_tiers.csv")

OUTPUT_COLUMNS: list[str] = [
    "iso3", "criterion", "value", "se", "method",
    "k_today", "n_today", "k_window", "n_window", "low_data",
]


class InsufficientHistory(ValueError):
    """
    Not enough completely pulled days to compute a criterion.
    """


def load_tiers(path: Path = TIERS_PATH) -> pd.DataFrame:
    """
    The unrest severity tiers: code_prefix, tier, label, reason.

    The values in unrest_tiers.csv:
        Tier 3: assault (18) and fight (19)
        Tier 2: riot (145) with coerce (17)
    """
    tiers = pd.read_csv(path, dtype={"code_prefix": "string"})
    if tiers["code_prefix"].duplicated().any():
        raise ValueError(f"{path}: a code prefix is listed twice")
    if not (tiers["tier"] >= 1).all():
        raise ValueError(f"{path}: tiers must be 1 or more")
    return tiers


def tier_weights(codes: pd.Series, tiers: pd.DataFrame) -> pd.Series:
    """
    Weight for each event code: tier / top tier, using the longest matching
    code prefix (so 145 beats 14). A code with no tier stops the run, so a code
    added to the unrest set cannot silently weigh 0.
    """
    top = tiers["tier"].max()
    weights = pd.Series(np.nan, index=codes.index)
    best = pd.Series(0, index=codes.index)

    for row in tiers.itertuples():
        prefix = str(row.code_prefix)
        match = codes.str.startswith(prefix) & (len(prefix) > best)
        weights[match] = row.tier / top
        best[match] = len(prefix)

    if weights.isna().any():
        missing = sorted(codes[weights.isna()].unique())
        raise ValueError(
            f"unrest codes with no severity tier: {missing}. Add them to "
            f"{TIERS_PATH.name}."
        )
    
    return weights


@dataclass
class FastInputs:
    """
    Everything fast_criteria() needs, loaded once.
    """
    counts: pd.DataFrame # iso3, date, criterion, k, k2, n
    complete_days: list[date] # days pulled with all 96 files


def build_counts(
        country_day: pd.DataFrame,
        event_codes: pd.DataFrame,
        settings: Settings,
        tiers: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
    """
    Numerator k, its sum of squared weights k2, and denominator n per country,
    day and criterion, for the variant chosen in settings (unrest code set,
    weighting and denominator).
    """
    keys = ["iso3", "date"]
    pieces: list[pd.DataFrame] = []


    # --- unrest (from the event code table) ---

    # Lead-paragraph events only (both sides of the share), by default
    if settings.unrest_root_only:
        if "is_root" not in event_codes.columns:
            raise ValueError(
                "the event code table has no is_root column: it was built "
                "by older code. Delete the database file and run "
                "src.pull_day again to rebuild it from the cache."
            )
        event_codes = event_codes[event_codes["is_root"] == 1]

    # Obtain event codes
    codes = (UNREST_ROOT_CODES if settings.unrest_codes == "strict"
             else UNREST_LOOSE_ROOT_CODES)
    root = event_codes["event_code"].str[:2]
    domestic = event_codes["relation"] == RELATION_DOMESTIC
    unrest_rows = event_codes[
        root.isin(codes) & domestic & (event_codes["media_target"] == 0)
    ].copy()

    if settings.unrest_weighting == "tiers":
        weight = tier_weights(unrest_rows["event_code"],
                              load_tiers() if tiers is None else tiers)
        
    elif settings.unrest_weighting == "goldstein":
        # Goldstein is fixed per code => sum / events is the code's value
        weight = (unrest_rows["goldstein_sum"].abs()
                  / unrest_rows["n_events"] / 10).clip(upper=1.0)
        
    else:
        weight = pd.Series(1.0, index=unrest_rows.index)

    unrest_rows["k"] = weight * unrest_rows["n_events"]
    unrest_rows["k2"] = weight ** 2 * unrest_rows["n_events"]
    k = unrest_rows.groupby(keys)[["k", "k2"]].sum()

    if settings.unrest_denominator == "domestic":
        # Same actor-coding filter on both sides
        n = event_codes[domestic].groupby(keys)["n_events"].sum()
    else:
        # All located events
        n = event_codes.groupby(keys)["n_events"].sum()

    unrest = k.join(n.rename("n"), how="outer").fillna(0)
    pieces.append(unrest.reset_index().assign(criterion="unrest"))


    # --- the other three (from the country-day table) ---

    for criterion, k_col, n_col in (
        ("media_pressure", "media_pressure_events", "n_events"),
        ("intl_conflict", "intl_conflict_events", "actor_events"),
        ("intl_coop", "intl_coop_events", "actor_events"),
    ):
        piece = country_day[keys + [k_col, n_col]].rename(
            columns={k_col: "k", n_col: "n"}
        )
        pieces.append(piece.assign(criterion=criterion, k2=piece["k"]))

    counts = pd.concat(pieces, ignore_index=True)
    counts["k"] = counts["k"].astype("float64")
    counts["k2"] = counts["k2"].astype("float64")
    counts["n"] = counts["n"].astype("int64")
    counts["date"] = pd.to_datetime(counts["date"]).dt.date
    counts = counts[counts["n"] > 0]

    bad = counts[counts["k"] > counts["n"] + 1e-9]
    if not bad.empty:
        raise ValueError(
            "numerator larger than denominator:\n"
            f"{bad.head().to_string(index=False)}"
        )
    
    return counts[["iso3", "date", "criterion", "k", "k2", "n"]]


def load_fast_inputs(
        connection: sqlite3.Connection,
        assessment_date: date,
        settings: Settings,
        first_date: date | None = None,
    ) -> FastInputs:
    """
    Read the days needed for one assessment date, once. With first_date,
    read enough for every assessment date from first_date to assessment_date
    (used to compute bounds over a reference period).
    """
    window = max(settings.baseline_days, settings.media_window_days)
    start = (first_date or assessment_date) - timedelta(days=window)
    country_day = load_country_day(connection, start, assessment_date)
    event_codes = load_event_codes(connection, start, assessment_date)

    log = load_pull_log(connection)
    complete = sorted(
        pd.to_datetime(log.loc[log["complete"] == 1, "date"]).dt.date
    )

    return FastInputs(
        counts=build_counts(country_day, event_codes, settings),
        complete_days=complete,
    )


def _smoothed(k, n, prior, strength):
    """
    Smoothed share and its binomial standard error.
    """
    share = (k + strength * prior) / (n + strength)
    se = np.sqrt(share * (1 - share) / (n + strength))
    return share, se


def _window_days(
        complete: list[date],
        first: date,
        last: date,
        needed: int,
        settings: Settings,
        what: str,
    ) -> list[date]:
    days = [d for d in complete if first <= d <= last]

    if len(days) < settings.min_baseline_coverage * needed:
        raise InsufficientHistory(
            f"{what}: {len(days)} complete days between {first} and "
            f"{last}, need at least "
            f"{int(np.ceil(settings.min_baseline_coverage * needed))} "
            f"of {needed}. Pull the missing days with src.pull_day."
        )
    
    return days


def selected_fast_criteria(settings: Settings) -> list[str]:
    """
    The GDELT criteria this run uses, in a fixed order.
    """
    if not settings.criteria:
        return list(FAST_CRITERIA)
    return [c for c in FAST_CRITERIA if c in settings.criteria]


def fast_criteria(
        inputs: FastInputs,
        assessment_date: date,
        settings: Settings,
    ) -> pd.DataFrame:
    """
    One row per country and criterion for the assessment date, with columns
    value, se (standard error), method ("deviation" or "share"), the counts
    behind it, and a low_data flag.

    Only the criteria chosen in settings.criteria are computed (all four by 
    default). Never returns a missing or infinite value: it raises instead, so
    nothing undefined reaches normalisation.
    """
    settings.validate()
    chosen = selected_fast_criteria(settings)
    if not chosen:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    
    if assessment_date not in inputs.complete_days:
        raise InsufficientHistory(
            f"{assessment_date} has not been pulled completely"
        )

    baseline = _window_days(
        inputs.complete_days,
        assessment_date - timedelta(days=settings.baseline_days),
        assessment_date - timedelta(days=1),
        settings.baseline_days, settings, "baseline",
    )
    media_window = _window_days(
        inputs.complete_days,
        assessment_date - timedelta(days=settings.media_window_days - 1),
        assessment_date,
        settings.media_window_days, settings, "media window",
    )

    counts = inputs.counts
    if "k2" not in counts.columns: # unweighted counts
        counts = counts.assign(k2=counts["k"])

    a = settings.prior_strength
    rows: list[pd.DataFrame] = []
    sums = ["k", "k2", "n"]

    for criterion in chosen:
        c = counts[counts["criterion"] == criterion]
        today = c[c["date"] == assessment_date].groupby("iso3")[sums]
        today = today.sum()

        use_share = (criterion == "media_pressure"
                     and settings.media_method == "cross_country")
        window_days = media_window if use_share else baseline
        window = c[c["date"].isin(window_days)].groupby("iso3")[sums]
        window = window.sum()

        # The all-country share over the window (for smoothing)
        total_n = window["n"].sum()
        if total_n == 0 or window["k"].sum() == 0:
            raise InsufficientHistory(
                f"{criterion}: no qualifying events in any country over "
                "the window, so it cannot be computed"
            )
        
        prior = window["k"].sum() / total_n

        frame = window.rename(columns={
            "k": "k_window", "k2": "k2_window", "n": "n_window",
        })

        frame = frame.join(
            today.rename(columns={
                "k": "k_today", "k2": "k2_today", "n": "n_today",
            }),
            how="outer",
        ).fillna(0)

        if use_share:
            value, se = _smoothed(
                frame["k_window"], frame["n_window"], prior, a
            )
            method = "share"

        else:
            c = settings.pseudo_events
            p_b, se_b = _smoothed(frame["k_window"], frame["n_window"],
                                  prior, a)
            expected = frame["n_today"] * p_b
            observed = frame["k_today"]
            if c == 0 and (expected == 0).any():
                raise InsufficientHistory(
                    f"{criterion}: nothing expected today for some "
                    "countries, so with pseudo_events 0 the ratio is "
                    "undefined; use pseudo_events > 0"
                )
            value = (observed + c) / (expected + c)

            # k2 = sum(w^2 * events): the plain count when unweighted
            var_observed = (frame["k2_today"] + c) * (1 - p_b)
            var_expected = (frame["n_today"] * se_b) ** 2

            # delta method: relative errors add in quadrature
            se = value * np.sqrt(
                var_observed / (observed + c) ** 2
                + var_expected / (expected + c) ** 2
            )

            method = "deviation"

        out = frame.assign(
            criterion=criterion, value=value, se=se, method=method,
            low_data=frame["n_today"] < settings.low_data_events,
        ).reset_index()
        rows.append(out)

    result = pd.concat(rows, ignore_index=True)
    for column in ("n_today", "n_window"):
        result[column] = result[column].astype("int64")
    result = result[OUTPUT_COLUMNS]

    valid_values = np.isfinite(result[["value", "se"]]).all(axis=1)
    undefined = ~valid_values

    if undefined.any():
        raise ValueError(
            "Missing or invalid values found before normalisation:\n"
            f"{result[undefined].head().to_string(index=False)}"
        )
    
    return result.sort_values(["criterion", "iso3"], ignore_index=True)
