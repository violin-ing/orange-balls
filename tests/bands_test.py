"""
Tests for src/score/bands.py and src/score/sensitivity.py. Run from the
repository root:

    python -m tests.bands_test

The tests use small, made-up scores, never the real database.

They cover:
    1. Bands: cut points (frozen reference-period percentiles, equal widths,
        explicit), which band a score falls in, the band range from the
        score's uncertainty and the band_uncertain flag.
    2. The band description table: the shipped file is valid, a public scale
        with fewer bands than the internal one is allowed, and broken tables
        are refused.
    3. The weights sensitivity analysis: countries whose criteria all agree
        never change band, a country near a cut point does, results repeat with
        the same seed, and the base band matches the scoring run.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.score.bands import (
    BadDescriptions,
    MissingBandCuts,
    assign_bands,
    band_of,
    check_descriptions,
    cut_points,
    describe,
    load_descriptions,
    public_band,
    reference_cut_rows,
)
from src.score.combine import aggregate
from src.score.run import ScoreResult
from src.score.sensitivity import (
    base_weights,
    draw_weights,
    weights_monte_carlo,
)
from src.score.settings import (
    EQUAL_FIFTHS,
    InvalidSettings,
    Settings,
    geometric_variant,
)

EQUAL = Settings(band_method="equal_width")
EQUAL_CUTS = np.array([0.2, 0.4, 0.6, 0.8])


def expect_error(error_type, function, *args, contains: str = "") -> None:
    try:
        function(*args)
    except error_type as error:
        assert contains in str(error), str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}")


# === BANDS ===

def reference(n: int = 101) -> pd.DataFrame:
    """
    Made-up reference-period scores: 0.00, 0.01, ... 1.00 spread over a
    few days, so each percentile is easy to work out by hand.
    """
    start = date(2026, 8, 1)
    return pd.DataFrame({
        "date": [start + timedelta(days=i % 5) for i in range(n)],
        "iso3": [f"C{i}" for i in range(n)],
        "score": np.linspace(0, 1, n),
    })


def test_reference_cut_points_are_the_stored_percentiles() -> None:
    rows = reference_cut_rows(reference(), Settings())

    # 5th, 20th, 50th and 80th percentiles of 0.00 ... 1.00
    assert np.allclose(rows["value"], [0.05, 0.2, 0.5, 0.8])
    assert np.allclose(cut_points(Settings(), rows), [0.05, 0.2, 0.5, 0.8])
    assert rows["reference_start"].iloc[0] == "2026-08-01"
    assert rows["reference_end"].iloc[0] == "2026-08-05"

    # they put 5/15/30/30/20 per cent of the reference scores in bands 1-5
    bands = band_of(reference(1001)["score"], cut_points(Settings(), rows))
    shares = np.bincount(bands, minlength=6)[1:] / 1001
    assert np.allclose(shares, [0.05, 0.15, 0.30, 0.30, 0.20], atol=0.002)


def test_equal_fifths_are_stored_alongside() -> None:
    fifths = replace(Settings(), band_percentiles=EQUAL_FIFTHS)
    rows = pd.concat([reference_cut_rows(reference(), Settings()),
                      reference_cut_rows(reference(), fifths)])

    assert np.allclose(cut_points(fifths, rows), [0.2, 0.4, 0.6, 0.8])
    assert np.allclose(cut_points(Settings(), rows),
                       [0.05, 0.2, 0.5, 0.8])


def test_missing_reference_cut_points_are_refused() -> None:
    expect_error(MissingBandCuts, cut_points, Settings(), None,
                 contains="compute_bounds")

    # cut points belong to the settings that made the scores
    rows = reference_cut_rows(reference(), Settings())
    expect_error(MissingBandCuts, cut_points, geometric_variant(Settings()),
                 rows, contains="compute_bounds")


def test_equal_width_and_explicit_cut_points() -> None:
    assert np.allclose(cut_points(EQUAL), EQUAL_CUTS)

    # geometric variant: equal widths over [0.1, 1]
    assert np.allclose(cut_points(geometric_variant(EQUAL)),
                       [0.28, 0.46, 0.64, 0.82])
    assert np.allclose(cut_points(replace(EQUAL, bands=3)), [1 / 3, 2 / 3])

    # explicit cut points override either method
    explicit = Settings(band_cuts=(0.3, 0.5, 0.7, 0.9))
    assert np.allclose(cut_points(explicit), [0.3, 0.5, 0.7, 0.9])

    # z-scores have no fixed range
    assert cut_points(replace(EQUAL, normalisation="zscore")) is None


def test_bad_cut_points_are_refused() -> None:
    for bad in (Settings(band_cuts=(0.5,)),
                Settings(band_cuts=(0.2, 0.6, 0.4, 0.8)),
                Settings(band_percentiles=(5.0, 20.0, 50.0)),
                Settings(band_percentiles=(20.0, 5.0, 50.0, 80.0)),
                Settings(band_percentiles=(0.0, 20.0, 50.0, 80.0)),
                Settings(band_method="quantile")):
        expect_error(InvalidSettings, bad.validate)


def test_band_1_holds_the_lowest_scores() -> None:
    cuts = EQUAL_CUTS
    found = band_of([0.0, 0.19, 0.2, 0.55, 0.99, 1.0], cuts).tolist()

    # A score exactly on a cut point goes to the higher band
    assert found == [1, 1, 2, 3, 5, 5]


def test_band_range_comes_from_the_standard_error() -> None:
    scores = pd.DataFrame({"score": [0.5, 0.5, 0.05],
                           "score_se": [0.0, 0.1, 0.1]},
                          index=["A", "B", "C"])
    
    banded = assign_bands(scores, EQUAL_CUTS)
    assert banded["band"].tolist() == [3, 3, 1]

    # 0.5 -+ 1.96 x 0.1 = 0.304 to 0.696
    assert banded["band_low"].tolist() == [3, 2, 1]
    assert banded["band_high"].tolist() == [3, 4, 2]

    # flagged when the data does not pin the country to one band (14.5)
    assert banded["band_uncertain"].tolist() == [False, True, True]


def test_zscores_get_no_band_without_cut_points() -> None:
    scores = pd.DataFrame({"score": [0.3], "score_se": [0.1]})

    banded = assign_bands(scores, None)
    assert banded["band"].isna().all()
    assert banded["band_uncertain"].isna().all()


# === DESCRIPTIONS ===

def table(public: list[tuple[int, str]], bands: int = 5) -> pd.DataFrame:
    """
    Internal rows 1..bands, and public rows (band, internal_bands).
    """
    rows = [(b, "internal", str(b), f"in {b}", f"do {b}", "draft")
            for b in range(1, bands + 1)]
    rows += [(b, "public", covers, f"out {b}", f"see {covers}", "draft")
             for b, covers in public]
    
    return pd.DataFrame(rows, columns=[
        "band", "audience", "internal_bands", "label", "action", "status",
    ])


def test_the_shipped_description_table_is_valid() -> None:
    shipped = check_descriptions(load_descriptions(), Settings().bands)
    assert set(shipped["audience"]) == {"internal", "public"}
    assert (shipped["status"] == "draft").all()


def test_the_shipped_public_scale_has_three_bands_grouped_1_2_2() -> None:
    # Decision log 21.1: Severe = 1, Elevated = 2-3, Routine = 4-5
    shipped = check_descriptions(load_descriptions(), Settings().bands)
    found = public_band(pd.Series([1, 2, 3, 4, 5]), shipped).tolist()
    assert found == [1, 2, 2, 3, 3]


def test_fewer_public_bands_than_internal_ones_are_allowed() -> None:
    three = check_descriptions(table([(1, "1;2"), (2, "3"), (3, "4;5")]),
                               5)
    found = public_band(pd.Series([1, 2, 3, 4, 5]), three).tolist()
    assert found == [1, 1, 2, 3, 3]


def test_broken_description_tables_are_refused() -> None:
    broken = {
        "misses internal band 5": table([(1, "1;2"), (2, "3;4")]),
        "not consecutive": table([(1, "1;3"), (2, "2"), (3, "4;5")]),
        "covers band 3 twice": table([(1, "1;2;3"), (2, "3;4;5")]),
        "public bands with a gap": table([(1, "1;2"), (3, "3;4;5")]),
        "internal band missing": table([(1, "1;2;3;4")], bands=4),
    }

    for name, bad in broken.items():
        try:
            check_descriptions(bad, 5)
        except BadDescriptions:
            continue
        raise AssertionError(f"Should be refused: {name}")


def test_describe_gives_each_audience_its_own_wording() -> None:
    t = table([(1, "1;2"), (2, "3"), (3, "4;5")])
    banded = pd.DataFrame({"band": [1, 4]}, index=["A", "B"])

    internal = describe(banded, t, "internal")
    public = describe(banded, t, "public")

    assert internal["label"].tolist() == ["in 1", "in 4"]
    assert public["band"].tolist() == [1, 3]
    assert public["action"].tolist() == ["see 1;2", "see 4;5"]


# === WEIGHTS SENSITIVITY ===

def result_for(
        norm: pd.DataFrame,
        settings: Settings,
        cuts: np.ndarray | None = EQUAL_CUTS,
    ) -> ScoreResult:
    se = norm * 0 + 0.01
    scores = assign_bands(aggregate(norm, se, settings), cuts)
    empty = pd.DataFrame()

    return ScoreResult(
        values=norm, ses=se, norm=norm, norm_se=se, scores=scores,
        fast=empty, reference_year=2025, fast_variant="x", cuts=cuts,
    )


NORM = pd.DataFrame({
    # STEADY: every criterion 0.9, so any weights give 0.9 (band 5)
    # EDGE: averages 0.41 (band 3, just above the 0.4 cut), but the
    #       criteria disagree, so other weights move it up or down
    "a": [0.9, 0.9, 0.05],
    "b": [0.9, 0.1, 0.06],
    "c": [0.9, 0.23, 0.07],
}, index=["STEADY", "EDGE", "LOW"])


def test_weight_draws_add_up_to_one_and_centre_on_the_base() -> None:
    base = base_weights(["a", "b", "c"], Settings(weights={"a": 2.0}))
    assert np.allclose(base, [0.5, 0.25, 0.25])

    draws = draw_weights(base, 5000, concentration=50, seed=1)
    assert np.allclose(draws.sum(axis=1), 1.0)
    assert np.allclose(draws.mean(axis=0), base, atol=0.01)


def test_countries_whose_criteria_agree_never_change_band() -> None:
    settings = Settings()

    mc = weights_monte_carlo(result_for(NORM, settings), settings,
                             draws=500, seed=0)
    c = mc.countries
    assert c.loc["STEADY", "changed_share"] == 0.0
    assert c.loc["STEADY", "base_band"] == 5
    assert c.loc["LOW", "changed_share"] == 0.0
    assert 0.1 < c.loc["EDGE", "changed_share"] < 0.9
    assert c.loc["EDGE", "base_band"] == 3
    assert c.loc["EDGE", "lowest_band"] <= 2
    assert c.loc["EDGE", "highest_band"] >= 4

    shares = c[[f"band_{b}" for b in range(1, 6)]].sum(axis=1)
    assert np.allclose(shares, 1.0)
    assert mc.summary["countries"] == 3 and mc.summary["draws"] == 500


def test_the_base_band_matches_the_scoring_run() -> None:
    settings = Settings(weights={"a": 3.0})
    result = result_for(NORM, settings)

    mc = weights_monte_carlo(result, settings, draws=10)
    assert (mc.countries["base_band"]
            == result.scores.loc[mc.countries.index, "band"]).all()


def test_the_same_seed_gives_the_same_answer() -> None:
    settings = Settings()
    result = result_for(NORM, settings)
    
    first = weights_monte_carlo(result, settings, draws=200, seed=7)
    again = weights_monte_carlo(result, settings, draws=200, seed=7)
    other = weights_monte_carlo(result, settings, draws=200, seed=8)
    pd.testing.assert_frame_equal(first.countries, again.countries)

    assert not first.weights.equals(other.weights)


def test_the_geometric_variant_can_be_tested_too() -> None:
    settings = geometric_variant(Settings())
    norm = NORM.clip(lower=0.1)

    cuts = cut_points(geometric_variant(EQUAL))
    mc = weights_monte_carlo(result_for(norm, settings, cuts), settings,
                             draws=200)
    assert mc.countries.loc["STEADY", "changed_share"] == 0.0


def test_a_run_without_bands_is_refused() -> None:
    result = result_for(NORM, Settings(), cuts=None)

    expect_error(ValueError, weights_monte_carlo, result, Settings(),
                 contains="no bands")


def main() -> None:
    argparse.ArgumentParser().parse_args()
    tests = [
        test_reference_cut_points_are_the_stored_percentiles,
        test_equal_fifths_are_stored_alongside,
        test_missing_reference_cut_points_are_refused,
        test_equal_width_and_explicit_cut_points,
        test_bad_cut_points_are_refused,
        test_band_1_holds_the_lowest_scores,
        test_band_range_comes_from_the_standard_error,
        test_zscores_get_no_band_without_cut_points,
        test_the_shipped_description_table_is_valid,
        test_the_shipped_public_scale_has_three_bands_grouped_1_2_2,
        test_fewer_public_bands_than_internal_ones_are_allowed,
        test_broken_description_tables_are_refused,
        test_describe_gives_each_audience_its_own_wording,
        test_weight_draws_add_up_to_one_and_centre_on_the_base,
        test_countries_whose_criteria_agree_never_change_band,
        test_the_base_band_matches_the_scoring_run,
        test_the_same_seed_gives_the_same_answer,
        test_the_geometric_variant_can_be_tested_too,
        test_a_run_without_bands_is_refused,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
