"""
Tests for src/countries/codes.py -- run from the repository root.

Unit tests (no data files needed):

    python3 -m tests.countries_test

Check against the real GDELT files already downloaded into the cache:

    python3 -m tests.countries_test --real
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.countries.codes import (
    NON_COUNTRY_CODES,
    CrosswalkError,
    UnknownCountryCode,
    load_crosswalk,
    to_iso3,
)
from src.gdelt.download import read_events

CACHE_DIR = Path("data/cache/gdelt")


def test_crosswalk_loads() -> None:
    """The real file passes its own checks and has a sensible size."""
    crosswalk = load_crosswalk()
    assert len(crosswalk) > 240, len(crosswalk)


def test_codes_that_differ_between_fips_and_iso() -> None:
    """
    The cases where skipping the conversion would give the wrong country.
    """
    crosswalk = load_crosswalk()
    expected = {
        "NI": "NGA",  # Nigeria (ISO 'NI' is Nicaragua)
        "NU": "NIC",  # Nicaragua
        "IS": "ISR",  # Israel (ISO 'IS' is Iceland)
        "IC": "ISL",  # Iceland
        "RS": "RUS",  # Russia (ISO 'RS' is Serbia)
        "RI": "SRB",  # Serbia
        "CH": "CHN",  # China (ISO 'CH' is Switzerland)
        "SZ": "CHE",  # Switzerland
        "UK": "GBR",  # United Kingdom
        "GB": "GAB",  # Gabon (ISO 'GB' is the United Kingdom)
        "GM": "DEU",  # Germany (ISO 'GM' is Gambia)
        "SN": "SGP",  # Singapore (ISO 'SN' is Senegal)
        "WA": "NAM",  # Namibia
        "US": "USA",
    }
    for fips, iso3 in expected.items():
        assert crosswalk[fips] == iso3, (fips, crosswalk[fips], iso3)


def test_to_iso3_basic() -> None:
    """
    Countries convert; empty cells and non-countries become missing.
    """
    codes = pd.Series(["US", "NI", None, "", "OS", " uk "], dtype="string")
    result = to_iso3(codes)

    assert result.tolist()[:2] == ["USA", "NGA"]
    assert result.isna().tolist() == [False, False, True, True, True, False]
    assert result.iloc[5] == "GBR"  # spaces and lower case are tidied
    assert len(result) == len(codes)


def test_to_iso3_keeps_the_index() -> None:
    """
    Needed so the result lines up when added back as a new column.
    """
    codes = pd.Series(["FR", "JA"], index=[10, 20], dtype="string")
    result = to_iso3(codes)
    assert result.index.tolist() == [10, 20]
    assert result.tolist() == ["FRA", "JPN"]


def test_unknown_code_stops_the_run() -> None:
    codes = pd.Series(["US", "ZZ", "ZZ"], dtype="string")
    try:
        to_iso3(codes)
    except UnknownCountryCode as error:
        assert "'ZZ': 2" in str(error), str(error)
    else:
        raise AssertionError("an unknown code should raise")


def test_bad_crosswalk_file_is_rejected(tmp_path: Path | None = None) -> None:
    """
    A row with the ISO3 code and the name swapped must not load.
    """
    import tempfile

    folder = tmp_path or Path(tempfile.mkdtemp())
    bad_file = folder / "bad.csv"
    bad_file.write_text("fips,iso3,name\nRP,Philippines,PHL\n", encoding="utf-8")
    try:
        load_crosswalk(bad_file)
    except CrosswalkError:
        pass
    else:
        raise AssertionError("a malformed crosswalk should raise")


def check_real_files() -> None:
    """
    Convert every country code in the cached GDELT files.
    """
    paths = sorted(CACHE_DIR.glob("*.export.CSV.zip"))
    if not paths:
        raise SystemExit(f"no cached files in {CACHE_DIR}, run the fetch test first")

    events = pd.concat([read_events(p) for p in paths], ignore_index=True)
    fips = events["ActionGeo_CountryCode"]
    events["iso3"] = to_iso3(fips)  # raises if any code is unrecognised

    print(f"{len(paths)} files, {len(events):,} rows")
    print(f"rows with a country:    {events['iso3'].notna().mean():.1%}")
    print(f"rows with no location:  {fips.isna().mean():.1%}")
    print(f"rows with a non-country code: "
          f"{fips.isin(NON_COUNTRY_CODES).mean():.1%}")

    print("\n--- ten most common countries (FIPS -> ISO3) ---")
    pairs = events.dropna(subset=["iso3"])
    print(pairs.groupby(["ActionGeo_CountryCode", "iso3"]).size()
          .sort_values(ascending=False).head(10))
    print("\nREAL DATA CHECK OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="check the cached GDELT files instead")
    args = parser.parse_args()

    if args.real:
        check_real_files()
        return

    tests = [
        test_crosswalk_loads,
        test_codes_that_differ_between_fips_and_iso,
        test_to_iso3_basic,
        test_to_iso3_keeps_the_index,
        test_unknown_code_stops_the_run,
        test_bad_crosswalk_file_is_rejected,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
