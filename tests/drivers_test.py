"""
Tests for src/score/drivers.py. Run from the repository root:

    python -m tests.drivers_test

The tests use small, made-up countries, never the real database.

They cover:
    1. The criterion text table: the shipped file has one text for each
       criterion, and broken tables are refused.
    2. Driver selection: the weakest criteria are named weakest first, at
       most driver_count of them, only when notably below average, and never
       a criterion on which every country is the same.
    3. Composition: band description first, then one line per driver.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.score.drivers import (
    BadCriterionTexts,
    check_criterion_texts,
    compose,
    load_criterion_texts,
    select_drivers,
)
from src.score.run import chosen_criteria
from src.score.settings import InvalidSettings, Settings


def expect_error(error_type, function, *args) -> None:
    try:
        function(*args)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


# === CRITERION TEXTS ===

def test_the_shipped_texts_cover_every_criterion() -> None:
    criteria = chosen_criteria(Settings())
    assert len(criteria) == 11

    shipped = check_criterion_texts(load_criterion_texts(), criteria)
    assert set(shipped["criterion"]) == set(criteria)
    assert (shipped["status"] == "draft").all()


def texts(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """
    A text table from (criterion, label, text, status) rows.
    """
    return pd.DataFrame(
        [(c, "internal", label, text, status)
         for c, label, text, status in rows],
        columns=["criterion", "audience", "label", "text", "status"],
    )


def test_broken_text_tables_are_refused() -> None:
    good = [("a", "A", "about a", "draft"), ("b", "B", "about b", "draft")]

    broken = {
        "missing b": texts(good[:1]),
        "a twice": texts(good + [("a", "A2", "again", "draft")]),
        "empty text": texts([good[0], ("b", "B", " ", "draft")]),
        "unknown status": texts([good[0], ("b", "B", "about b", "agreed")]),
    }

    for name, bad in broken.items():
        try:
            check_criterion_texts(bad, ["a", "b"])
        except BadCriterionTexts:
            continue
        raise AssertionError(f"should be refused: {name}")


# === DRIVERS ===

# Six ordinary countries, plus WEAK (low on c and a, a little low on b) and
# FLAT (average everywhere). Criterion "same" is equal for every country.
NORM = pd.DataFrame({
    "a": [0.6, 0.5, 0.7, 0.6, 0.5, 0.7, 0.30, 0.6],
    "b": [0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.50, 0.55],
    "c": [0.7, 0.8, 0.7, 0.8, 0.7, 0.8, 0.10, 0.65],
    "same": [0.5] * 8,
}, index=["P", "Q", "R", "S", "T", "U", "WEAK", "FLAT"])


def test_the_weakest_criteria_are_named_weakest_first() -> None:
    found = select_drivers(NORM, Settings())
    weak = found[found["iso3"] == "WEAK"]

    # c is further below its average, in standard deviations, than a
    assert weak["criterion"].tolist() == ["c", "a"]
    assert weak["rank"].tolist() == [1, 2]
    assert (weak["gap_sd"] <= -0.5).all()


def test_an_average_country_gets_no_driver() -> None:
    found = select_drivers(NORM, Settings())
    assert "FLAT" not in set(found["iso3"])


def test_driver_count_and_floor_are_respected() -> None:
    one = select_drivers(NORM, Settings(driver_count=1))
    assert one[one["iso3"] == "WEAK"]["criterion"].tolist() == ["c"]

    none = select_drivers(NORM, Settings(driver_count=0))
    assert none.empty

    # With a very high floor nothing is notably weak
    strict = select_drivers(NORM, Settings(driver_floor_sd=5.0))
    assert strict.empty


def test_a_criterion_with_no_spread_is_never_named() -> None:
    found = select_drivers(NORM, Settings(driver_floor_sd=0.0))
    assert "same" not in set(found["criterion"])


def test_bad_driver_settings_are_refused() -> None:
    for bad in (Settings(driver_count=-1), Settings(driver_floor_sd=-0.1)):
        expect_error(InvalidSettings, bad.validate)


# === COMPOSITION ===

def test_compose_gives_the_band_then_one_line_per_driver() -> None:
    table = texts([
        ("a", "Weak a", "a matters", "draft"),
        ("b", "Weak b", "b matters", "draft"),
        ("c", "Weak c", "c matters", "draft"),
        ("same", "Same", "same", "draft"),
    ])
    band_row = pd.Series({"label": "Band two", "action": "do two"})
    drivers = select_drivers(NORM, Settings())

    assert compose("WEAK", band_row, drivers, table) == [
        "Band two: do two", "Weak c: c matters", "Weak a: a matters",
    ]
    assert compose("FLAT", band_row, drivers, table) == ["Band two: do two"]


def main() -> None:
    argparse.ArgumentParser().parse_args()
    tests = [
        test_the_shipped_texts_cover_every_criterion,
        test_broken_text_tables_are_refused,
        test_the_weakest_criteria_are_named_weakest_first,
        test_an_average_country_gets_no_driver,
        test_driver_count_and_floor_are_respected,
        test_a_criterion_with_no_spread_is_never_named,
        test_bad_driver_settings_are_refused,
        test_compose_gives_the_band_then_one_line_per_driver,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
