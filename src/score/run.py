"""
Calculates an overall score for each country by combining GDELT indicators with
longer-term structural information.

The process has two steps:

    inputs = load_score_inputs(connection, assessment_date, settings)
    result = score_countries(inputs, assessment_date, settings)

First, load_score_inputs() collects everything needed from the database:

    - GDELT indicators for the assessment date.
    - Structural information for the chosen reference year.
    - The stored limits used to make indicators comparable.

Then score_countries() uses those inputs to:

    1. Select the indicators included in the assessment.
    2. Bring the GDELT and structural data together.
    3. Put all indicators onto a common scale, with higher values
       always meaning better conditions.
    4. Combine them into one overall score per country.
    5. Carry the standard errors through to show uncertainty.

The indicators included in a run are controlled by Settings.criteria.

    Empty (default)
        Uses all four GDELT indicators and all structural variables marked as
        criteria.

    A list of names
        Uses only the selected indicators. This allows different combinations
        to be tested, such as structural indicators alone, to see whether
        adding GDELT improves the assessment.

Candidate variables can also be included when explicitly selected.
Validation variables cannot be used as scoring criteria.

The structural reference year is the assessment year minus reference_lag_years.
GDELT indicators use the assessment date itself.

The database is read only by load_score_inputs(). Once the inputs are loaded,
score_countries() performs the calculations without further database access.
This allows the same data to be rescored with different settings during
sensitivity testing.

The result contains the original indicator values, their standard errors,
normalised values, the final country scores and supporting information about
the calculation.

Main functions:

    chosen_criteria()
        Determines which indicators are included.

    directions()
        Defines whether higher values mean better conditions.

    load_score_inputs()
        Reads and prepares all required data from the database.

    score_countries()
        Combines the indicators and calculates the final scores.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
import pandas as pd

from src.reference.variables import VARIABLES, with_role
from src.score.bands import assign_bands, cut_points
from src.score.bounds import SLOW_VARIANT, fast_variant, score_variant
from src.score.combine import aggregate, normalise
from src.score.fast import (
    FAST_CRITERIA,
    FAST_HIGHER_IS_BETTER,
    FastInputs,
    InsufficientHistory,
    fast_criteria,
    load_fast_inputs,
)
from src.score.settings import Settings
from src.store.database import (
    BAND_CUT_COLUMNS,
    load_band_cuts,
    load_bounds,
    load_reference,
)


@dataclass
class ScoreInputs:
    """
    Everything score_countries() needs, loaded once.
    """
    fast: FastInputs
    reference: pd.DataFrame # long reference table
    bounds: pd.DataFrame # stored normalisation bounds
    band_cuts: pd.DataFrame = field(  # stored band cut points
        default_factory=lambda: pd.DataFrame(columns=BAND_CUT_COLUMNS))


@dataclass
class ScoreResult:
    values: pd.DataFrame # country x criterion, raw values
    ses: pd.DataFrame # their standard errors
    norm: pd.DataFrame # normalised, higher = better
    norm_se: pd.DataFrame
    scores: pd.DataFrame # score, score_se, n_criteria, band, band_low,
                         # band_high, band_uncertain
    fast: pd.DataFrame # fast_criteria() output, with counts
    reference_year: int
    fast_variant: str
    clipped: dict[str, int] = field(default_factory=dict)
    cuts: np.ndarray | None = None # the band cut points used
    score_variant: str = ""


def chosen_criteria(settings: Settings) -> list[str]:
    """
    The criteria a run scores, in a fixed order.
    """
    if not settings.criteria:
        return list(FAST_CRITERIA) + with_role("criterion")
    
    allowed = list(FAST_CRITERIA) + with_role("criterion") \
        + with_role("candidate")
    unknown = [c for c in settings.criteria if c not in allowed]

    if unknown:
        raise ValueError(
            f"criteria that cannot be scored: {unknown}. Allowed: "
            f"{allowed}"
        )
    
    return list(settings.criteria)


def directions() -> dict[str, bool]:
    """
    Return whether higher values are better for each criterion.
    """
    result = FAST_HIGHER_IS_BETTER.copy()

    for name, variable in VARIABLES.items():
        result[name] = variable.higher_is_better

    return result


def load_score_inputs(
        connection: sqlite3.Connection,
        assessment_date: date,
        settings: Settings,
    ) -> ScoreInputs:
    settings.validate()
    chosen = chosen_criteria(settings)
    year = assessment_date.year - settings.reference_lag_years
    slow = [c for c in chosen if c in VARIABLES]
    
    if any(c in FAST_CRITERIA for c in chosen):
        fast = load_fast_inputs(connection, assessment_date, settings)
    else:
        fast = FastInputs(counts=pd.DataFrame(), complete_days=[])

    reference = load_reference(connection, variables=slow,
                               start_year=year, end_year=year)
    bounds = load_bounds(connection,
                         [SLOW_VARIANT, fast_variant(settings)])
    band_cuts = load_band_cuts(connection, [score_variant(settings)])

    return ScoreInputs(fast=fast, reference=reference, bounds=bounds,
                       band_cuts=band_cuts)


def _wide(long: pd.DataFrame, column: str, names: list[str]) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame(columns=names)
    
    return long.pivot(index="iso3", columns="criterion", values=column)


def score_countries(
        inputs: ScoreInputs,
        assessment_date: date,
        settings: Settings,
    ) -> ScoreResult:
    settings.validate()
    chosen = chosen_criteria(settings)
    year = assessment_date.year - settings.reference_lag_years
    variant = fast_variant(settings)

    # Structural criteria for the reference year
    slow_names = [c for c in chosen if c in VARIABLES]
    slow = inputs.reference[
        (inputs.reference["year"] == year)
        & inputs.reference["variable"].isin(slow_names)
    ].rename(columns={"variable": "criterion", "sd": "se"})

    # GDELT criteria for the assessment date
    fast_names = [c for c in chosen if c in FAST_CRITERIA]

    if fast_names:
        fast = fast_criteria(
            inputs.fast, assessment_date,
            replace(settings, criteria=tuple(fast_names)),
        )

    else:
        fast = pd.DataFrame(columns=["iso3", "criterion", "value", "se"])

    both = pd.concat(
        [slow[["iso3", "criterion", "value", "se"]],
         fast[["iso3", "criterion", "value", "se"]]],
         ignore_index=True
    )
    
    values = _wide(both, "value", chosen).reindex(columns=chosen)
    
    ses = _wide(both, "se", chosen).reindex(columns=chosen)

    bounds = inputs.bounds

    slow_rows = (
        (bounds["variant"] == SLOW_VARIANT)
        & bounds["criterion"].isin(slow_names)
    )

    fast_rows = (
        (bounds["variant"] == variant)
        & bounds["criterion"].isin(fast_names)
    )

    wanted = slow_rows | fast_rows
    bounds = bounds[wanted].set_index("criterion")

    norm, norm_se = normalise(values, ses, directions(), settings, bounds)
    cuts = cut_points(settings, inputs.band_cuts)
    scores = assign_bands(aggregate(norm, norm_se, settings), cuts)

    return ScoreResult(
        values=values, ses=ses, norm=norm, norm_se=norm_se,
        scores=scores, fast=fast, reference_year=year,
        fast_variant=variant, clipped=norm.attrs.get("clipped", {}),
        cuts=cuts, score_variant=score_variant(settings),
    )


def reference_scores(
        inputs: ScoreInputs,
        days: Iterable[date],
        settings: Settings,
    ) -> tuple[pd.DataFrame, list[date]]:
    """
    Every country's composite score on every given day, for working out the
    band cut points (decision log 14.1). One row per country and day:
    date, iso3, score. Days without enough GDELT history are skipped and
    returned separately.

    inputs must cover the whole period: GDELT counts from before the first
    day, and structural values for every reference year needed.
    """
    # Bands are not needed here (they are what is being worked out), so
    # score with equal-width cuts, which need nothing stored
    plain = replace(settings, band_method="equal_width", band_cuts=())
    frames: list[pd.DataFrame] = []
    skipped: list[date] = []

    for day in days:
        try:
            result = score_countries(inputs, day, plain)
        except InsufficientHistory:
            skipped.append(day)
            continue

        frames.append(pd.DataFrame({
            "date": day,
            "iso3": result.scores.index,
            "score": result.scores["score"].to_numpy(),
        }))

    if not frames:
        raise InsufficientHistory(
            "no reference day could be scored, so band cut points cannot "
            "be worked out"
        )

    return pd.concat(frames, ignore_index=True), skipped
