"""
Sensitivity analysis to test whether country rankings or stability categories
depend too heavily on the choice of weights.

The method is the weighted arithmetic mean with equal weights (will justify 
choice in report). The weights are drawn at random many times, every country is
re-scored with each draw, and the result is how often each country's band
differs from its band under the base weights.

How the weights are drawn...
    From a Dirichlet distribution (i.e. random weights that are all positive
    and add up to 1. Its parameters are:

         concentration x (no. of criteria) x (base share of each criterion)
    
    As a result, the draws equal the base weights on average.

        concentration = 1   with equal base weights, every possible set of
                            weights is equally likely (the widest reasonable
                            test: "we know nothing about the weights")
        concentration > 1   draws stay closer to the base weights (e.g. 10 or
                            50: "the weights are roughly right")

    Note that the seed is fixed, so a run can be repeated exactly.

Why it is cheap...
    The weights do not affect normalisation, so the normalised table from
    one scoring run (ScoreResult.norm) is reused: each draw is one matrix
    product. 1,000 draws for 180 countries take well under a second.

What it reports (weights_monte_carlo)...
    per country    base band; changed_share, the share of draws in which the
                   band differs from the base band; lowest, highest and most
                   common band over the draws; and the share of draws in each
                   band (band_1 ... band_N)
    summary        number of countries and draws, the mean changed share, and
                   how many countries change band in at least 10%, 25% and 50%
                   of draws

Only countries with every criterion are used (the default, 
require_all_criteria). Arithmetic aggregation is the method; with
geometric_variant() settings the draws use the geometric mean, so the
geometric variant can be reported the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.score.bands import band_of, cut_points
from src.score.run import ScoreResult
from src.score.settings import Settings

CHANGE_THRESHOLDS: tuple[float, ...] = (0.10, 0.25, 0.50)


@dataclass
class MonteCarloResult:
    countries: pd.DataFrame # one row per country, see module notes
    summary: dict[str, float]
    weights: pd.DataFrame # one row per draw, one column per criterion


def base_weights(criteria: list[str], settings: Settings) -> np.ndarray:
    """
    The base weights as shares adding up to 1 (equal unless set).
    """
    raw = np.array([settings.weights.get(c, 1.0) for c in criteria],
                   dtype=float)
    
    return raw / raw.sum()


def draw_weights(
        base: np.ndarray,
        draws: int,
        concentration: float,
        seed: int,
    ) -> np.ndarray:
    """
    Generate random sets of weights that add up to 1.
    Concentration controls how much weights vary from the original.
    """
    if draws < 1:
        raise ValueError("draws must be at least 1")
    
    if not concentration > 0:
        raise ValueError("concentration must be above 0")
    
    rng = np.random.default_rng(seed)

    return rng.dirichlet(concentration * len(base) * base, size=draws)


def scores_for_weights(
        norm: pd.DataFrame,
        weights: np.ndarray,
        settings: Settings,
    ) -> np.ndarray:
    """
    Calculate each country's score using every set of weights.
    Supports arithmetic and geometric means.
    """
    normalised_scores = norm.to_numpy(dtype=float)

    if settings.aggregation == "geometric":
        if (normalised_scores <= 0).any():
            raise ValueError("geometric aggregation needs values above 0")
        
        return np.exp(np.log(normalised_scores) @ weights.T)
    
    return normalised_scores @ weights.T


def weights_monte_carlo(
        result: ScoreResult,
        settings: Settings,
        draws: int = 1000,
        concentration: float = 1.0,
        seed: int = 0,
    ) -> MonteCarloResult:
    """
    Re-score the countries of one run under many random weights and report how
    often each changes band. settings must be the Settings the run was scored
    with.
    """
    settings.validate()
    cuts = cut_points(settings)
    if cuts is None:
        raise ValueError(
            "these settings give no bands (z-scores without band_cuts), "
            "so band changes cannot be counted"
        )

    norm = result.norm.dropna()
    if norm.empty:
        raise ValueError("no country has every criterion")
    
    criteria = list(norm.columns)
    base = base_weights(criteria, settings)

    base_band = band_of(scores_for_weights(norm, base[None, :],
                                           settings)[:, 0], cuts)
    
    weights = draw_weights(base, draws, concentration, seed)
    bands = band_of(scores_for_weights(norm, weights, settings), cuts)

    countries = pd.DataFrame({
        "base_band": base_band,
        "changed_share": (bands != base_band[:, None]).mean(axis=1),
        "lowest_band": bands.min(axis=1),
        "highest_band": bands.max(axis=1),
        "most_common_band": [np.bincount(row).argmax() for row in bands],
    }, index=norm.index)

    for band in range(1, len(cuts) + 2):
        countries[f"band_{band}"] = (bands == band).mean(axis=1)

    countries = countries.sort_values("changed_share", ascending=False)

    changed = countries["changed_share"]
    summary: dict[str, float] = {
        "countries": len(countries),
        "draws": draws,
        "concentration": concentration,
        "seed": seed,
        "mean_changed_share": float(changed.mean()),
    }

    for threshold in CHANGE_THRESHOLDS:
        summary[f"countries_changing_{int(threshold * 100)}pc"] = int(
            (changed >= threshold).sum()
        )

    return MonteCarloResult(
        countries=countries,
        summary=summary,
        weights=pd.DataFrame(weights, columns=criteria),
    )
