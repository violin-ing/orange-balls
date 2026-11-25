"""
Tests for src/score/pairs.py (Bilateral View scores) and the JSON written by
src/export_web.py. Run from the repository root:

    python -m tests.pairs_test

The tests use small, made-up pair tables, never the real database.

They cover:
    1. Pair scores: both directions add up, the smoothing, the "too little
       contact" rule, and bands from cut points.
    2. Cut points: the fast reference calculation agrees with scoring each
       day on its own, and days without history are skipped.
    3. The export: the bilateral JSON lists each pair from both sides, the
       international JSON holds every scored country, and the relation band
       labels file is valid.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.export_web import bilateral_json, international_json
from src.score.fast import InsufficientHistory
from src.score.pairs import (
    load_relation_bands,
    pair_cut_rows,
    pair_reference_values,
    pair_scores,
    pair_variant,
    stored_pair_cuts,
)
from src.score.run import score_countries
from src.score.settings import Settings
from tests.score_test import DAY as SCORE_DAY
from tests.score_test import make_score_inputs, with_cuts

DAY = date(2026, 10, 31)


def expect_error(error_type, function, *args) -> None:
    try:
        function(*args)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def make_pair_day(days: int = 90) -> tuple[pd.DataFrame, list[date]]:
    """
    Every day for `days` days up to DAY:
        RUS -> UKR  4 conflict, UKR -> RUS 2 conflict, 1 cooperative
        USA -> GBR  3 cooperative, GBR -> USA 1 cooperative
        FRA -> DEU  1 cooperative on the last day only (too little contact)
    """
    dates = [DAY - timedelta(days=i) for i in range(days)]
    rows = []
    for d in dates:
        ts = pd.Timestamp(d)
        rows += [
            (ts, "RUS", "UKR", 4, 0, 4, 3),
            (ts, "UKR", "RUS", 3, 1, 2, 2),
            (ts, "USA", "GBR", 3, 3, 0, 0),
            (ts, "GBR", "USA", 1, 1, 0, 0),
        ]
    rows.append((pd.Timestamp(DAY), "FRA", "DEU", 1, 1, 0, 0))
    table = pd.DataFrame(rows, columns=[
        "date", "actor1", "actor2", "n_events", "n_coop", "n_conflict",
        "n_material_conflict",
    ])
    return table, sorted(dates)


def row(scores: pd.DataFrame, a: str, b: str) -> pd.Series:
    match = scores[(scores["a"] == a) & (scores["b"] == b)]
    assert len(match) == 1
    return match.iloc[0]


# === PAIR SCORES ===

def test_both_directions_are_added_together() -> None:
    table, days = make_pair_day()
    scores = pair_scores(table, DAY, Settings(), days, None)

    # written one way round, alphabetically
    rus = row(scores, "RUS", "UKR")
    assert rus["n"] == 90 * 7  # 6 conflict + 1 cooperative a day
    assert rus["conflict"] == 90 * 6
    assert (rus["a_to_b"], rus["b_to_a"]) == (90 * 4, 90 * 3)

    usa = row(scores, "GBR", "USA")
    assert (usa["a_to_b"], usa["b_to_a"]) == (90 * 1, 90 * 3)


def test_the_score_is_a_smoothed_cooperative_share() -> None:
    table, days = make_pair_day()
    scores = pair_scores(table, DAY, Settings(), days, None)

    coop = 90 * 1 + 90 * 4 + 1
    total = 90 * 7 + 90 * 4 + 1
    prior = coop / total
    a = Settings().prior_strength

    rus = row(scores, "RUS", "UKR")
    expected = (90 * 1 + a * prior) / (90 * 7 + a)
    assert abs(rus["value"] - expected) < 1e-12
    assert row(scores, "GBR", "USA")["value"] > rus["value"]


def test_too_little_contact_gets_no_band() -> None:
    table, days = make_pair_day()
    cuts = np.array([0.2, 0.4, 0.6, 0.8])
    scores = pair_scores(table, DAY, Settings(), days, cuts)

    fra = row(scores, "DEU", "FRA")
    assert not fra["enough"] and fra["band"] == 0

    # enough contact: the conflictual pair in band 1, the friendly one in 5
    assert row(scores, "RUS", "UKR")["band"] == 1
    assert row(scores, "GBR", "USA")["band"] == 5

    # without cut points nobody gets a band
    plain = pair_scores(table, DAY, Settings(), days, None)
    assert (plain["band"] == 0).all()


def test_a_day_without_history_is_refused() -> None:
    table, days = make_pair_day(days=30)
    expect_error(InsufficientHistory, pair_scores, table, DAY, Settings(),
                 days, None)


# === CUT POINTS ===

def test_reference_values_match_scoring_each_day() -> None:
    table, days = make_pair_day(days=120)
    settings = Settings()
    reference = [DAY - timedelta(days=i) for i in range(3)]

    values, skipped = pair_reference_values(table, reference, settings, days)
    assert skipped == []

    for day in reference:
        direct = pair_scores(table, day, settings, days, None)
        direct = direct[direct["enough"]].set_index(["a", "b"])["value"]
        fast = values[values["date"] == day].set_index(["a", "b"])["value"]
        assert set(direct.index) == set(fast.index)
        assert np.allclose(direct.sort_index(), fast.sort_index())


def test_days_without_history_are_skipped() -> None:
    table, days = make_pair_day(days=100)
    early = [days[0] + timedelta(days=5), DAY]

    values, skipped = pair_reference_values(table, early, Settings(), days)
    assert skipped == [early[0]]
    assert set(values["date"]) == {DAY}


def test_cut_rows_are_stored_under_their_own_variant() -> None:
    table, days = make_pair_day(days=120)
    settings = Settings()
    values, _ = pair_reference_values(table, [DAY], settings, days)

    rows = pair_cut_rows(values, settings)
    assert len(rows) == 4
    assert (rows["variant"] == pair_variant(settings)).all()
    assert pair_variant(settings).startswith("pairs-")
    assert np.allclose(stored_pair_cuts(rows, settings),
                       np.percentile(values["value"], [5, 20, 50, 80]))

    # different settings, different variant: not found
    assert stored_pair_cuts(rows, Settings(baseline_days=60)) is None


# === EXPORT ===

def test_the_relation_band_labels_are_valid() -> None:
    labels = load_relation_bands()
    assert labels["band"].tolist() == [0, 1, 2, 3, 4, 5]
    assert (labels["label"].str.strip() != "").all()


def test_bilateral_json_lists_each_pair_from_both_sides() -> None:
    table, days = make_pair_day()
    cuts = np.array([0.2, 0.4, 0.6, 0.8])
    scores = pair_scores(table, DAY, Settings(), days, cuts)
    data = bilateral_json(DAY, scores, cuts, Settings())

    rus_ukr = data["partners"]["RUS"]["UKR"]
    ukr_rus = data["partners"]["UKR"]["RUS"]

    # band and score agree; "out" and "in" swap sides
    assert rus_ukr[:4] == ukr_rus[:4]
    assert (rus_ukr[4], rus_ukr[5]) == (ukr_rus[5], ukr_rus[4])
    assert data["fields"] == ["band", "score", "events", "conflict", "out",
                              "in"]
    assert data["partners"]["FRA"]["DEU"][0] == 0


def test_international_json_holds_every_scored_country() -> None:
    settings = Settings()
    inputs = with_cuts(make_score_inputs(settings), settings)
    result = score_countries(inputs, SCORE_DAY, settings)
    data = international_json(SCORE_DAY, result, settings)

    assert set(data["countries"]) == {"BIG", "MID", "SMALL"}
    big = data["countries"]["BIG"]
    assert set(big["values"]) == {c["key"] for c in data["criteria"]}
    assert 1 <= big["band"] <= 5
    assert big["band_low"] <= big["band"] <= big["band_high"]

    # every criterion has a group, a name and a text
    groups = {g["key"] for g in data["groups"]}
    assert all(c["group"] in groups and c["text"] for c in data["criteria"])
    # band actions are split into sentences with a capital letter
    assert all(a[0].isupper() for b in data["bands"] for a in b["actions"])


def main() -> None:
    argparse.ArgumentParser().parse_args()
    tests = [
        test_both_directions_are_added_together,
        test_the_score_is_a_smoothed_cooperative_share,
        test_too_little_contact_gets_no_band,
        test_a_day_without_history_is_refused,
        test_reference_values_match_scoring_each_day,
        test_days_without_history_are_skipped,
        test_cut_rows_are_stored_under_their_own_variant,
        test_the_relation_band_labels_are_valid,
        test_bilateral_json_lists_each_pair_from_both_sides,
        test_international_json_holds_every_scored_country,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
