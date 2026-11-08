"""
Calculates and stores the reference limits used to put country indicators onto
a common scale.

AIM: make scores comparable across different countries and assessment dates.

Without fixed limits, a country's score would depend on which other countries
happened to be included in that particular run. This would make it difficult to
compare results over time.

Instead, we calculate the limits once using a chosen reference period and store
them in the database. The same limits are then reused for future assessments.

For each indicator, we store:

    - lower / upper   The limits used for min-max normalisation.
    - mean / sd       The average and spread used for z-scores.
    - method          How the limits were calculated.
    - reference_start / reference_end
                      The period used to calculate the limits.
    - variant         Identifies the GDELT settings used.
    - computed_at     When the limits were calculated.

Different indicators use different methods to set their limits:

    theoretical
        Uses the natural limits of a measurement scale.

    reference_range
        Uses the lowest and highest values found in the reference period.

    percentile_cap
        Uses zero as the lower limit and a chosen high percentile as the upper
        limit. This prevents unusually high GDELT values from stretching the 
        scale too far.

Values outside the stored limits are capped when normalised.

GDELT settings are recorded in a variant code. If settings that affect the
scores change, a different set of normalisation limits is required.

Main functions:

    slow_bounds()
        Calculates reference limits for structural indicators.

    fast_reference_values()
        Calculates GDELT indicator values across the reference period.

    fast_bounds()
        Calculates reference limits for the GDELT indicators.

    fast_variant()
        Identifies the GDELT settings used, so the correct normalisation limits
        are applied.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from src.reference.variables import VARIABLES
from src.score.fast import (
    TIERS_PATH,
    FastInputs,
    InsufficientHistory,
    fast_criteria,
)
from src.score.settings import Settings
from src.store.database import BOUNDS_COLUMNS

SLOW_VARIANT = "-"

# Settings that change the value of a GDELT criterion
FAST_SETTINGS: tuple[str, ...] = (
    "baseline_days", "prior_strength", "pseudo_events", "unrest_codes",
    "unrest_weighting", "unrest_denominator", "unrest_root_only",
    "media_method",
    "media_window_days",
)


def fast_variant(settings: Settings) -> str:
    """
    A short code naming the GDELT settings in use (e.g. "a3f09c21d4e5").
    Two runs with the same code compute GDELT values the same way, so they can
    share bounds.
    """
    described = {name: getattr(settings, name) for name in FAST_SETTINGS}

    if settings.unrest_weighting == "tiers":
        described["tiers"] = TIERS_PATH.read_text(encoding="utf-8")

    text = json.dumps(described, sort_keys=True)

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def slow_bounds(
        reference: pd.DataFrame,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
    """
    Bounds for every structural variable that has rows in the reference table
    over [start_year, end_year]. 
    Variables used only for checks (i.e. role "validation") are skipped.
    """
    window = reference[reference["year"].between(start_year, end_year)]
    rows = []

    for name, variable in VARIABLES.items():

        if variable.role == "validation":
            continue

        values = window.loc[window["variable"] == name, "value"]

        if values.empty:
            continue

        if variable.bounds is not None:
            lower, upper = variable.bounds
            method = "theoretical"

        else:
            lower, upper = float(values.min()), float(values.max())
            method = "reference_range"

        rows.append({
            "criterion": name, "variant": SLOW_VARIANT,
            "lower": lower, "upper": upper,
            "mean": float(values.mean()), "sd": float(values.std()),
            "method": method,
            "reference_start": str(start_year),
            "reference_end": str(end_year),
            "computed_at": _now(),
        })

    return pd.DataFrame(rows, columns=BOUNDS_COLUMNS)


def fast_reference_values(
        inputs: FastInputs,
        days: Iterable[date],
        settings: Settings,
    ) -> tuple[pd.DataFrame, list[date]]:
    """
    Every country-day value of the GDELT criteria over the given days.
    Days without enough history are skipped and returned separately.
    """
    frames: list[pd.DataFrame] = []
    skipped: list[date] = []
    last_reason = ""

    for day in days:
        try:
            values = fast_criteria(inputs, day, settings)
        except InsufficientHistory as error:
            skipped.append(day)
            last_reason = str(error)
            continue
        frames.append(values.assign(date=day))

    if not frames:
        raise InsufficientHistory(
            "No day in the reference period could be used to compute the "
            f"GDELT criteria. Reason given for the last day: {last_reason}"
        )
    
    return pd.concat(frames, ignore_index=True), skipped


def fast_bounds(
        values: pd.DataFrame,
        settings: Settings,
    ) -> pd.DataFrame:
    """
    Bounds for the GDELT criteria from their country-day values over the
    reference period: Starts from 0 up to the cap_percentile-th percentile.
    """
    variant = fast_variant(settings)
    start, end = str(values["date"].min()), str(values["date"].max())
    rows = []

    for criterion, group in values.groupby("criterion"):
        x = group["value"].to_numpy(dtype=float)
        rows.append({
            "criterion": criterion, "variant": variant,
            "lower": 0.0,
            "upper": float(np.percentile(x, settings.cap_percentile)),
            "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "method": "percentile_cap",
            "reference_start": start, "reference_end": end,
            "computed_at": _now(),
        })
        
    return pd.DataFrame(rows, columns=BOUNDS_COLUMNS)
