"""
Tests the scoring system to make sure its calculations work correctly.

Run from the repository root:

    python -m tests.score_test

The tests use small, made-up datasets rather than the real database.
This makes them quick to run and ensures the results are predictable.

The tests cover:

    1. Settings
       Checks that valid settings are accepted and invalid
       combinations are rejected.

    2. GDELT event counts
       Checks how events are counted, including unrest severity
       weights, media targeting and denominator choices.

    3. GDELT indicator calculations
       Checks historical comparisons, smoothing, media pressure,
       uncertainty and handling of countries with few or no events.

    4. Normalisation
       Checks that indicators are put onto a common scale,
       correctly adjusted so higher always means better, and
       handled correctly when values fall outside the reference limits.

    5. Combining indicators
       Checks that weights, arithmetic and geometric averages,
       missing data and standard errors are handled correctly.

    6. Reference limits
       Checks how fixed limits are calculated, stored and linked
       to the correct GDELT settings.

    7. Complete country assessments
       Checks that the full process works, from individual
       indicators through to the final country score.

The tests also check that missing history, invalid settings and
undefined results are caught rather than allowed to produce
unreliable country scores.

Each test reports its result. The script finishes by confirming
how many tests passed.
"""

from __future__ import annotations

import argparse
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.reference.variables import with_role
from src.score.bounds import (
    SLOW_VARIANT,
    fast_bounds,
    fast_reference_values,
    fast_variant,
    slow_bounds,
)
from src.score.combine import MissingBounds, aggregate, normalise
from src.score.fast import (
    FAST_CRITERIA,
    FastInputs,
    InsufficientHistory,
    build_counts,
    fast_criteria,
    load_tiers,
    tier_weights,
)
from src.score.run import ScoreInputs, score_countries
from src.score.settings import InvalidSettings, Settings
from src.store.database import BOUNDS_COLUMNS

DAY = date(2026, 10, 31)


def expect_error(error_type, function, *args, contains: str = "") -> None:
    try:
        function(*args)
    except error_type as error:
        assert contains in str(error), str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}")


# ------------------------------------------------------------ settings

def test_default_settings_are_valid() -> None:
    Settings().validate()


def test_geometric_with_zero_floor_is_refused() -> None:
    settings = Settings(aggregation="geometric", minmax_floor=0.0)
    expect_error(InvalidSettings, settings.validate, contains="floor")


def test_geometric_with_zscores_is_refused() -> None:
    settings = Settings(aggregation="geometric", normalisation="zscore",
                        minmax_floor=0.1)
    expect_error(InvalidSettings, settings.validate, contains="z-scores")


def test_geometric_with_a_floor_is_allowed() -> None:
    Settings(aggregation="geometric", minmax_floor=0.1).validate()


def test_bad_values_are_refused() -> None:
    for bad in (Settings(unrest_codes="medium"), Settings(bands=1),
                Settings(prior_strength=-1), Settings(weights={"x": 0}),
                Settings(unrest_weighting="severity"),
                Settings(bounds="moving"), Settings(cap_percentile=20),
                Settings(criteria=("unrest", "unrest"))):
        expect_error(InvalidSettings, bad.validate)


# ------------------------------------------------- building the counts

def test_unrest_denominator_uses_the_same_filter() -> None:
    """Domestic-coded events on both sides; 'all' uses every event."""
    d = pd.Timestamp(DAY)
    country_day = pd.DataFrame({
        "iso3": ["USA"], "date": [d], "n_events": [100],
        "media_pressure_events": [1], "actor_events": [80],
        "intl_conflict_events": [5], "intl_coop_events": [7],
    })
    codes = pd.DataFrame({
        "iso3": "USA", "date": d,
        "event_code": ["141", "190", "173", "010", "010"],
        "relation": ["domestic", "domestic", "domestic", "domestic",
                     "foreign"],
        "media_target": [0, 0, 0, 0, 0],
        "n_events": [3, 2, 4, 31, 60],
        # Goldstein is fixed per code: 141 -6.5, 190 -10, 173 -5, 010 0
        "goldstein_sum": [-19.5, -20.0, -20.0, 0.0, 0.0],
    })
    def unrest(**changes):
        settings = Settings(unrest_weighting="count", **changes)
        counts = build_counts(country_day, codes, settings)
        row = counts[counts["criterion"] == "unrest"].iloc[0]
        return int(row["k"]), int(row["n"])

    # strict: 141 and 190; domestic denominator: 3 + 2 + 4 + 31 = 40
    assert unrest() == (5, 40)
    # loose adds the arrest (173)
    assert unrest(unrest_codes="loose") == (9, 40)
    # all located events as denominator
    assert unrest(unrest_denominator="all") == (5, 100)

    counts = build_counts(country_day, codes, Settings())
    conflict = counts[counts["criterion"] == "intl_conflict"].iloc[0]
    assert (conflict["k"], conflict["n"]) == (5, 80)


UNREST_DAY = pd.DataFrame({
    "iso3": ["USA"], "date": [pd.Timestamp(DAY)], "n_events": [100],
    "media_pressure_events": [1], "actor_events": [80],
    "intl_conflict_events": [5], "intl_coop_events": [7],
})
UNREST_CODES = pd.DataFrame({
    "iso3": "USA", "date": pd.Timestamp(DAY),
    "event_code": ["141", "145", "190", "173", "010"],
    "relation": "domestic", "media_target": 0,
    "n_events": [3, 1, 2, 4, 30],
    "goldstein_sum": [-19.5, -7.5, -20.0, -20.0, 0.0],
})


def unrest_row(settings: Settings) -> pd.Series:
    counts = build_counts(UNREST_DAY, UNREST_CODES, settings)
    return counts[counts["criterion"] == "unrest"].iloc[0]


def test_unrest_is_weighted_by_severity_tier() -> None:
    """Default tiers: protest 1/4, riot 2/4, fight 3/4 (of the top tier)."""
    row = unrest_row(Settings())
    assert abs(row["k"] - (3 * 0.25 + 1 * 0.5 + 2 * 0.75)) < 1e-12
    # k2 = sum of squared weights, used for the count's variance
    assert abs(row["k2"] - (3 * 0.0625 + 1 * 0.25 + 2 * 0.5625)) < 1e-12
    assert row["n"] == 40
    # loose adds the arrest (173, coerce, tier 2 of 4)
    loose = unrest_row(Settings(unrest_codes="loose"))
    assert abs(loose["k"] - (row["k"] + 4 * 0.5)) < 1e-12


def test_goldstein_weighting_is_a_sensitivity_variant() -> None:
    row = unrest_row(Settings(unrest_weighting="goldstein"))
    assert abs(row["k"] - (3 * 0.65 + 1 * 0.75 + 2 * 1.0)) < 1e-12


def test_tiers_use_the_longest_matching_prefix() -> None:
    tiers = load_tiers()
    weights = tier_weights(pd.Series(["141", "145", "1451", "200"]), tiers)
    assert weights.tolist() == [0.25, 0.5, 0.5, 1.0]
    without_20 = tiers[tiers["code_prefix"] != "20"]
    expect_error(ValueError, tier_weights, pd.Series(["200"]), without_20,
                 contains="no severity tier")


# --------------------------------------------- computing the criteria

def make_inputs(
        days: int = 90,
        small_k_today: int = 0,
        media_all_zero_for_small: bool = True,
    ) -> FastInputs:
    """
    BIG: 3,000 events a day, 3% unrest every day (so its ratio is 1).
    SMALL: 20 events a day, never any unrest or media events before today.
    MID: 500 events a day, 3% unrest normally, 6% today.
    """
    rows = []
    dates = [DAY - timedelta(days=i) for i in range(days + 1)]
    for d in dates:
        today = d == DAY
        for iso3, n, k in (
            ("BIG", 3000, 90),
            ("SMALL", 20, small_k_today if today else 0),
            ("MID", 500, 30 if today else 15),
        ):
            for criterion in ("unrest", "intl_conflict", "intl_coop"):
                rows.append((iso3, d, criterion, k, n))
            media_k = 0 if (iso3 == "SMALL" and media_all_zero_for_small) \
                else max(k // 10, 1)
            rows.append((iso3, d, "media_pressure", media_k, n))
    counts = pd.DataFrame(rows, columns=["iso3", "date", "criterion", "k",
                                         "n"])
    return FastInputs(counts=counts, complete_days=sorted(dates))


def get(result: pd.DataFrame, iso3: str, criterion: str) -> pd.Series:
    match = result[(result["iso3"] == iso3)
                   & (result["criterion"] == criterion)]
    assert len(match) == 1
    return match.iloc[0]


def test_zero_baseline_still_gives_a_defined_value() -> None:
    result = fast_criteria(make_inputs(), DAY, Settings())
    small = get(result, "SMALL", "unrest")
    assert math.isfinite(small["value"]) and math.isfinite(small["se"])
    # No events today and none before: about "normal", with a wide band
    assert abs(small["value"] - 1.0) < 0.01
    assert 0.9 < small["se"] < 1.1


def test_zero_expected_without_smoothing_is_refused() -> None:
    expect_error(InsufficientHistory, fast_criteria, make_inputs(), DAY,
                 Settings(prior_strength=0, pseudo_events=0),
                 contains="pseudo_events")


def test_few_events_mean_a_wider_band() -> None:
    """Same share, very different amounts of data."""
    result = fast_criteria(make_inputs(), DAY, Settings())
    big = get(result, "BIG", "unrest")
    small = get(result, "SMALL", "unrest")
    assert abs(big["value"] - 1.0) < 0.01
    assert small["se"] > 5 * big["se"]
    assert bool(small["low_data"]) and not bool(big["low_data"])


def test_one_unusual_event_in_a_small_country_is_damped() -> None:
    """One event where none ever happened: unsmoothed, the ratio would be
    infinite. Smoothed, it is about 2, with a wide band."""
    result = fast_criteria(make_inputs(small_k_today=1), DAY, Settings())
    small = get(result, "SMALL", "unrest")
    assert 1.5 < small["value"] < 2.5
    assert small["se"] > 1.0


def test_deviation_matches_a_hand_calculation() -> None:
    settings = Settings()
    result = fast_criteria(make_inputs(), DAY, settings)
    mid = get(result, "MID", "unrest")

    # Prior: all countries over the 90 baseline days
    k_all = 90 * (90 + 0 + 15)
    n_all = 90 * (3000 + 20 + 500)
    p0 = k_all / n_all
    a, c = settings.prior_strength, settings.pseudo_events
    p_b = (90 * 15 + a * p0) / (90 * 500 + a)
    se_b = math.sqrt(p_b * (1 - p_b) / (90 * 500 + a))
    expected = 500 * p_b                     # about 15
    ratio = (30 + c) / (expected + c)        # about 1.94
    se = ratio * math.sqrt(
        (30 + c) * (1 - p_b) / (30 + c) ** 2
        + (500 * se_b) ** 2 / (expected + c) ** 2
    )

    assert abs(mid["value"] - ratio) < 1e-12
    assert abs(mid["se"] - se) < 1e-12
    assert mid["method"] == "deviation"


def test_media_is_compared_across_countries_by_default() -> None:
    result = fast_criteria(make_inputs(), DAY, Settings())
    media = get(result, "BIG", "media_pressure")
    assert media["method"] == "share"
    assert media["n_window"] == 30 * 3000          # the 30-day window
    assert abs(media["value"] - 9 / 3000) < 1e-3   # a share, not a ratio

    deviation = fast_criteria(make_inputs(), DAY,
                              Settings(media_method="deviation"))
    assert get(deviation, "BIG", "media_pressure")["method"] == "deviation"


def test_too_little_history_is_refused() -> None:
    expect_error(InsufficientHistory, fast_criteria, make_inputs(days=30),
                 DAY, Settings(), contains="baseline")


def test_an_unpulled_assessment_day_is_refused() -> None:
    expect_error(InsufficientHistory, fast_criteria, make_inputs(),
                 DAY + timedelta(days=1), Settings(),
                 contains="not been pulled")


def test_no_undefined_value_reaches_scoring() -> None:
    """Research chat item 1: whatever the settings, no NaN or infinity
    leaves fast_criteria, even for a country with no events at all."""
    variants = [
        Settings(),
        Settings(media_method="deviation"),
        Settings(unrest_denominator="all"),
        Settings(prior_strength=0),
        Settings(prior_strength=0, media_method="deviation"),
        Settings(pseudo_events=0.5, baseline_days=60),
    ]
    for settings in variants:
        result = fast_criteria(make_inputs(), DAY, settings)
        assert np.isfinite(result[["value", "se"]]).all().all(), settings
        assert len(result) == 3 * len(FAST_CRITERIA)


def test_only_the_chosen_criteria_are_computed() -> None:
    settings = Settings(criteria=("unrest", "wgi_ge"))
    result = fast_criteria(make_inputs(), DAY, settings)
    assert set(result["criterion"]) == {"unrest"}
    none = fast_criteria(make_inputs(), DAY,
                         Settings(criteria=("wgi_ge",)))
    assert none.empty


# === NORMALISE AND COMBINE ===

def table(rows: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=["A", "B", "C"])


DIRECTIONS = {"good": True, "bad": False}


def test_minmax_flips_and_floors() -> None:
    values = table({"good": [0.0, 5.0, 10.0], "bad": [0.0, 5.0, 10.0]})
    ses = table({"good": [1.0, 1.0, 1.0], "bad": [1.0, 1.0, 1.0]})
    norm, norm_se = normalise(values, ses, DIRECTIONS,
                              Settings(minmax_floor=0.1, bounds="per_run"))
    
    assert norm["good"].round(6).tolist() == [0.1, 0.55, 1.0]
    assert norm["bad"].round(6).tolist() == [1.0, 0.55, 0.1]

    # se scaled by (1 - floor) / (max - min) = 0.9 / 10
    assert abs(norm_se.loc["A", "good"] - 0.09) < 1e-12


def test_constant_criterion_is_refused() -> None:
    values = table({"good": [2.0, 2.0, 2.0]})
    expect_error(ValueError, normalise, values, values, DIRECTIONS,
                 Settings(bounds="per_run"), contains="same value")


FIXED = pd.DataFrame(
    {
        "lower": [0.0, 0.0], 
        "upper": [10.0, 10.0], 
        "mean": [5.0, 5.0],
        "sd": [2.0, 2.0]
    },
    index=pd.Index(["good", "bad"], name="criterion"),
)


def test_fixed_bounds_do_not_depend_on_the_other_countries() -> None:
    ses = table({"good": [0.5] * 3, "bad": [0.5] * 3})
    wide = table({"good": [5.0, 0.0, 10.0], "bad": [5.0, 0.0, 10.0]})
    narrow = table({"good": [5.0, 4.0, 6.0], "bad": [5.0, 4.0, 6.0]})

    settings = Settings() # bounds="fixed"

    a_wide, _ = normalise(wide, ses, DIRECTIONS, settings, FIXED)
    a_narrow, _ = normalise(narrow, ses, DIRECTIONS, settings, FIXED)

    assert a_wide.loc["A"].tolist() == a_narrow.loc["A"].tolist() \
        == [0.5, 0.5]
    
    # per-run bounds, by contrast, move A
    per_run, _ = normalise(
        narrow, ses, DIRECTIONS,
        Settings(bounds="per_run")
    )
    assert per_run.loc["A", "good"] == 0.5 and per_run.loc["B", "good"] == 0


def test_values_outside_fixed_bounds_are_clipped_and_counted() -> None:
    values = table({"good": [12.0, 5.0, -1.0], "bad": [12.0, 5.0, 5.0]})
    ses = table({"good": [1.0] * 3, "bad": [1.0] * 3})
    norm, norm_se = normalise(values, ses, DIRECTIONS, Settings(), FIXED)

    assert norm["good"].tolist() == [1.0, 0.5, 0.0]
    assert norm.loc["A", "bad"] == 0.0
    assert norm.attrs["clipped"] == {"good": 2, "bad": 1}
    assert abs(norm_se.loc["A", "good"] - 0.1) < 1e-12


def test_missing_fixed_bounds_are_refused() -> None:
    values = table({"good": [1.0, 2.0, 3.0]})
    expect_error(MissingBounds, normalise, values, values, DIRECTIONS,
                 Settings(), FIXED.drop(index="good"),
                 contains="compute_bounds")


def test_zscores_use_the_reference_mean_and_sd() -> None:
    values = table({"good": [5.0, 9.0, 1.0], "bad": [5.0, 9.0, 1.0]})
    ses = table({"good": [1.0] * 3, "bad": [1.0] * 3})
    norm, norm_se = normalise(values, ses, DIRECTIONS,
                              Settings(normalisation="zscore"), FIXED)
    
    assert norm["good"].tolist() == [0.0, 2.0, -2.0]
    assert norm["bad"].tolist() == [0.0, -2.0, 2.0]
    assert norm_se.loc["A", "good"] == 0.5


def test_arithmetic_aggregation_and_its_error() -> None:
    norm = table({"good": [0.2, 0.6, 1.0], "bad": [1.0, 0.6, 0.2]})
    se = table({"good": [0.1, 0.1, 0.1], "bad": [0.2, 0.2, 0.2]})
    result = aggregate(norm, se, Settings(weights={"good": 3, "bad": 1}))

    # (3 * 0.2 + 1 * 1.0) / 4 = 0.4
    assert abs(result.loc["A", "score"] - 0.4) < 1e-12

    # sqrt(0.75^2 * 0.1^2 + 0.25^2 * 0.2^2)
    expected = math.sqrt(0.75 ** 2 * 0.01 + 0.25 ** 2 * 0.04)
    assert abs(result.loc["A", "score_se"] - expected) < 1e-12


def test_geometric_bottom_country_is_not_zeroed() -> None:
    values = table({"good": [0.0, 5.0, 10.0], "bad": [9.0, 0.0, 10.0]})
    ses = table({"good": [0.5, 0.5, 0.5], "bad": [0.5, 0.5, 0.5]})
    settings = Settings(aggregation="geometric", minmax_floor=0.1,
                        bounds="per_run")
    norm, norm_se = normalise(values, ses, DIRECTIONS, settings)

    result = aggregate(norm, norm_se, settings)
    assert (result["score"] > 0).all()

    # A is worst on "good" but strong on "bad" -> it must beat C, which is
    # worst on "bad" (or at least not be forced to 0)
    a_score = result.loc["A", "score"]
    assert abs(a_score - math.sqrt(0.1 * norm.loc["A", "bad"])) < 1e-12
    assert np.isfinite(result["score_se"]).all()


def test_countries_missing_a_criterion_are_left_out() -> None:
    norm = table({"good": [0.2, np.nan, 1.0], "bad": [1.0, 0.6, 0.2]})
    se = table({"good": [0.1, np.nan, 0.1], "bad": [0.1, 0.1, 0.1]})
    assert aggregate(norm, se, Settings()).index.tolist() == ["A", "C"]

    partial = aggregate(norm, se, Settings(require_all_criteria=False))
    assert partial.loc["B", "n_criteria"] == 1
    assert abs(partial.loc["B", "score"] - 0.6) < 1e-12


def test_aggregate_uses_only_the_chosen_criteria() -> None:
    norm = table({"good": [0.2, 0.6, 1.0], "bad": [1.0, 0.6, 0.2]})
    se = table({"good": [0.1] * 3, "bad": [0.1] * 3})
    result = aggregate(norm, se, Settings(criteria=("good",)))

    assert result["score"].tolist() == [0.2, 0.6, 1.0]
    assert (result["n_criteria"] == 1).all()

    expect_error(ValueError, aggregate, norm, se,
                 Settings(criteria=("good", "tone")), contains="tone")


# === BOUNDS ===

def test_fast_variant_follows_the_settings_that_change_values() -> None:
    base = fast_variant(Settings())

    assert fast_variant(Settings(prior_strength=10)) != base
    assert fast_variant(Settings(unrest_weighting="count")) != base

    # weights, aggregation and the criteria list do not change the values
    assert fast_variant(Settings(weights={"unrest": 2},
                                 aggregation="geometric",
                                 minmax_floor=0.1,
                                 criteria=("unrest",))) == base


def reference_rows(rows: list[tuple]) -> pd.DataFrame:
    """
    rows: (iso3, year, variable, value, sd)
    """
    return pd.DataFrame([
        {"iso3": i, "year": y, "variable": v, "source": "x", "value": x,
         "sd": s, "low": np.nan, "high": np.nan, "interval": np.nan}
        for i, y, v, x, s in rows
    ])


def test_slow_bounds_use_the_scale_or_the_reference_range() -> None:
    reference = reference_rows([
        ("A", 2010, "wgi_ge", 30.0, 3.0), ("B", 2010, "wgi_ge", 70.0, 3.0),
        ("A", 2010, "v2mecenefm", -2.0, 0.2),
        ("B", 2020, "v2mecenefm", 3.0, 0.2),
        ("C", 1990, "v2mecenefm", -4.0, 0.2), # before the period
        ("A", 2010, "gdp_pc", 1000.0, np.nan),
    ])

    bounds = slow_bounds(reference, 2000, 2024).set_index("criterion")

    assert list(bounds.columns) == BOUNDS_COLUMNS[1:]
    assert (bounds.loc["wgi_ge", "lower"], bounds.loc["wgi_ge", "upper"],
            bounds.loc["wgi_ge", "method"]) == (0.0, 100.0, "theoretical")
    assert (bounds.loc["v2mecenefm", "lower"],
            bounds.loc["v2mecenefm", "upper"]) == (-2.0, 3.0)
    assert bounds.loc["v2mecenefm", "method"] == "reference_range"
    assert "gdp_pc" not in bounds.index
    assert (bounds["variant"] == SLOW_VARIANT).all()


def test_fast_bounds_cap_at_a_percentile() -> None:
    values = pd.DataFrame({
        "criterion": "unrest", "iso3": "A",
        "date": [DAY - timedelta(days=i) for i in range(101)],
        "value": [float(i) for i in range(101)],
    })

    bounds = fast_bounds(values, Settings(cap_percentile=90))
    row = bounds.iloc[0]

    assert (row["lower"], row["upper"]) == (0.0, 90.0)
    assert row["variant"] == fast_variant(Settings())


def test_reference_days_without_history_are_skipped() -> None:
    inputs = make_inputs(days=100)
    days = sorted(inputs.complete_days)
    values, skipped = fast_reference_values(inputs, days, Settings())

    # A day needs 72 of its 90 baseline days (80%) pulled
    assert len(skipped) == 72 and len(days) - len(skipped) == 29
    assert values["date"].nunique() == 29


# === FULL RUN ===

SLOW = with_role("criterion")


def make_score_inputs(settings: Settings) -> ScoreInputs:
    year = DAY.year - settings.reference_lag_years
    rows = []

    for iso3, level in (("BIG", 0.8), ("SMALL", 0.5), ("MID", 0.2)):
        for name in SLOW:
            value = 100 * level if name.startswith("wgi") else 4 * level
            rows.append((iso3, year, name, value, 0.1))
            rows.append((iso3, year + 1, name, 0.0, 0.1))

    reference = reference_rows(rows)
    slow = slow_bounds(reference, year, year)

    # Make the V-Dem range a little wider than the three countries
    slow.loc[slow["method"] == "reference_range", "lower"] = 0.0
    slow.loc[slow["method"] == "reference_range", "upper"] = 4.0
    fast_values = fast_criteria(make_inputs(), DAY, settings).assign(
        date=DAY)
    fast = fast_bounds(fast_values, settings)
    fast["upper"] = fast["upper"] * 2

    return ScoreInputs(
        fast=make_inputs(), 
        reference=reference,
        bounds=pd.concat([slow, fast], ignore_index=True)
    )


def test_a_whole_run_scores_every_country() -> None:
    settings = Settings()
    result = score_countries(make_score_inputs(settings), DAY, settings)

    assert result.reference_year == DAY.year - 1
    assert list(result.values.columns) == list(FAST_CRITERIA) + SLOW
    assert sorted(result.scores.index) == ["BIG", "MID", "SMALL"]
    assert (result.scores["n_criteria"] == 9).all()
    assert np.isfinite(result.scores[["score", "score_se"]]).all().all()

    # BIG is best on every structural criterion except shutdown capacity
    # (where high = bad) so it should beat MID
    assert result.scores.loc["BIG", "score"] \
        > result.scores.loc["MID", "score"]


def test_a_run_can_score_the_structural_criteria_only() -> None:
    """
    Nine criteria versus the slow five.
    """
    settings = Settings(criteria=tuple(SLOW))
    result = score_countries(make_score_inputs(Settings()), DAY, settings)

    assert list(result.values.columns) == SLOW
    assert result.fast.empty
    assert (result.scores["n_criteria"] == 5).all()


def test_unknown_or_validation_criteria_are_refused() -> None:
    inputs = make_score_inputs(Settings())

    for bad in (("unrest", "tone"), ("gdp_pc",)):
        expect_error(ValueError, score_countries, inputs, DAY,
                     Settings(criteria=bad), contains="cannot be scored")


def main() -> None:
    argparse.ArgumentParser().parse_args()
    tests = [
        test_default_settings_are_valid,
        test_geometric_with_zero_floor_is_refused,
        test_geometric_with_zscores_is_refused,
        test_geometric_with_a_floor_is_allowed,
        test_bad_values_are_refused,
        test_unrest_denominator_uses_the_same_filter,
        test_unrest_is_weighted_by_severity_tier,
        test_goldstein_weighting_is_a_sensitivity_variant,
        test_tiers_use_the_longest_matching_prefix,
        test_zero_baseline_still_gives_a_defined_value,
        test_zero_expected_without_smoothing_is_refused,
        test_few_events_mean_a_wider_band,
        test_one_unusual_event_in_a_small_country_is_damped,
        test_deviation_matches_a_hand_calculation,
        test_media_is_compared_across_countries_by_default,
        test_too_little_history_is_refused,
        test_an_unpulled_assessment_day_is_refused,
        test_no_undefined_value_reaches_scoring,
        test_only_the_chosen_criteria_are_computed,
        test_minmax_flips_and_floors,
        test_constant_criterion_is_refused,
        test_fixed_bounds_do_not_depend_on_the_other_countries,
        test_values_outside_fixed_bounds_are_clipped_and_counted,
        test_missing_fixed_bounds_are_refused,
        test_zscores_use_the_reference_mean_and_sd,
        test_arithmetic_aggregation_and_its_error,
        test_geometric_bottom_country_is_not_zeroed,
        test_countries_missing_a_criterion_are_left_out,
        test_aggregate_uses_only_the_chosen_criteria,
        test_fast_variant_follows_the_settings_that_change_values,
        test_slow_bounds_use_the_scale_or_the_reference_range,
        test_fast_bounds_cap_at_a_percentile,
        test_reference_days_without_history_are_skipped,
        test_a_whole_run_scores_every_country,
        test_a_run_can_score_the_structural_criteria_only,
        test_unknown_or_validation_criteria_are_refused,
    ]

    for test in tests:
        test()
        print(f"ok  {test.__name__}")
        
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
