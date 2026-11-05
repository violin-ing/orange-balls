"""
Makes the four country indicators comparable and combines them into
one overall score for each country.

The four indicators come from the earlier GDELT scoring step:
    - unrest
    - media pressure
    - international conflict
    - international cooperation

First, each indicator is converted onto a common scale so that
they can be combined fairly.

Normalisation can use either:

    minmax
        Rescales values using fixed lower and upper limits.
        Values outside those limits are capped.

    zscore
        Shows how far a country's value is above or below the
        average, measured in standard deviations.

The direction of each indicator is also adjusted so that higher
always means better conditions. For example, lower unrest and
lower conflict are treated as better.

The limits and averages can come from a fixed reference period
or from the countries included in the current run. Fixed limits
should be used when comparing results across different dates.

Next, the normalised indicators are combined into one country
score using their assigned weights.

Two methods are available:

    arithmetic
        Calculates a weighted average of the indicators.

    geometric
        Combines the indicators using a weighted geometric mean.
        All values must be above zero.

If a country is missing an indicator, it can either be excluded
or scored using the indicators that are available, depending
on the settings.

Standard errors are carried through both steps to show the
uncertainty in the final score. These calculations assume that
the errors in different indicators are independent.

The output contains one row per country, with:

    score       Overall country score
    score_se    Estimated standard error
    n_criteria  Number of indicators used

Main functions:

    normalise()  Makes the indicators comparable and adjusts
                 their direction so higher always means better.

    aggregate()  Combines the indicators into one overall score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.score.settings import Settings


class MissingBounds(ValueError):
    """
    Fixed bounds were asked for but none are stored for a criterion.
    """


def _per_run_bounds(x: pd.Series, criterion: str) -> dict[str, float]:
    low, high = float(x.min()), float(x.max())

    if not np.isfinite(high - low) or high == low:
        raise ValueError(
            f"{criterion}: every country has the same value, so it "
            "cannot be min-max normalised"
        )
    
    spread = float(x.std())

    return {"lower": low, "upper": high, "mean": float(x.mean()),
            "sd": spread}


def normalise(
        values: pd.DataFrame,
        ses: pd.DataFrame,
        higher_is_better: dict[str, bool],
        settings: Settings,
        bounds: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return normalised values and their standard errors, same shape as the
    input, with higher always meaning better.

    bounds: indexed by criterion, with columns lower, upper, mean and sd.
    Required when settings.bounds is "fixed".
    """
    settings.validate()
    missing = [c for c in values.columns if c not in higher_is_better]

    if missing:
        raise ValueError(f"no direction recorded for criteria {missing}")
    
    if settings.bounds == "fixed":
        absent = [c for c in values.columns
                  if bounds is None or c not in bounds.index]
        if absent:
            raise MissingBounds(
                f"no stored bounds for {absent}. Run src.compute_bounds "
                "for these settings or bounds='per_run' for a sensitivity "
                "check."
            )

    norm = pd.DataFrame(index=values.index)
    norm_se = pd.DataFrame(index=values.index)

    clipped: dict[str, int] = {}
    for criterion in values.columns:
        x = values[criterion].astype(float)
        se = ses[criterion].astype(float)

        if settings.bounds == "fixed":
            b = bounds.loc[criterion]
        else:
            b = _per_run_bounds(x.dropna(), criterion)

        good = higher_is_better[criterion]

        if settings.normalisation == "minmax":
            lower, upper = float(b["lower"]), float(b["upper"])
            outside = (x < lower) | (x > upper)
            clipped[criterion] = int(outside.sum())

            t = (x.clip(lower, upper) - lower) / (upper - lower)

            if not good:
                t = 1 - t

            span = (1 - settings.minmax_floor) / (upper - lower)

            norm[criterion] = settings.minmax_floor + (
                1 - settings.minmax_floor) * t
            norm_se[criterion] = se * span

        else:
            spread = float(b["sd"])

            if not spread > 0:
                raise ValueError(
                    f"{criterion}: no spread, so it cannot be turned "
                    "into z-scores"
                )
            
            z = (x - float(b["mean"])) / spread
            norm[criterion] = z if good else -z
            norm_se[criterion] = se / spread
            clipped[criterion] = 0

    norm.attrs["clipped"] = clipped

    return norm, norm_se


def aggregate(
        norm: pd.DataFrame,
        norm_se: pd.DataFrame,
        settings: Settings,
    ) -> pd.DataFrame:
    """
    One row per country: score, score_se and n_criteria (how many criteria
    the score is built from).

    If settings.criteria is set, only those criteria are combined; a
    named criterion missing from the input stops the run.
    With require_all_criteria, countries missing any criterion are left
    out. Otherwise the weights of the criteria a country has are used.
    """
    settings.validate()

    if settings.criteria:
        absent = [c for c in settings.criteria if c not in norm.columns]

        if absent:
            raise ValueError(f"criteria not in the input: {absent}")
        
        norm = norm[list(settings.criteria)]
        norm_se = norm_se[list(settings.criteria)]

    weights = pd.Series(
        {c: settings.weights.get(c, 1.0) for c in norm.columns}
    )

    unknown = sorted(set(settings.weights) - set(norm.columns))

    if unknown:
        raise ValueError(f"weights given for unknown criteria {unknown}")

    present = norm.notna() & norm_se.notna()

    if settings.require_all_criteria:
        keep = present.all(axis=1)
        norm, norm_se, present = norm[keep], norm_se[keep], present[keep]

    w = present.mul(weights, axis=1) # 0 where a criterion is missing
    w_sum = w.sum(axis=1)
    share = w.div(w_sum, axis=0) # each row's weights sum to 1
    x = norm.where(present, 0.0)
    s = norm_se.where(present, 0.0)

    if settings.aggregation == "arithmetic":
        score = (share * x).sum(axis=1)
        score_se = np.sqrt((share ** 2 * s ** 2).sum(axis=1))
    else:
        if (x[present] <= 0).any().any():
            raise ValueError(
                "geometric aggregation met a value of 0 or below; check "
                "minmax_floor"
            )
        
        safe = x.where(present, 1.0)
        score = np.exp((share * np.log(safe)).sum(axis=1))
        relative = (share ** 2 * (s / safe) ** 2).sum(axis=1)
        score_se = score * np.sqrt(relative)

    result = pd.DataFrame({
        "score": score,
        "score_se": score_se,
        "n_criteria": present.sum(axis=1).astype("int64"),
    })
    
    return result[w_sum > 0]
