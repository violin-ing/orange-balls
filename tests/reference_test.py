"""
Tests for src/reference/ and the reference tables -- run from the repo root:

    python3 -m tests.reference_test

Uses small hand-made files in a temporary folder, never the real ones.
To load the real files, run python3 -m src.load_reference.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from src.reference.common import REFERENCE_COLUMNS, UnknownReferenceCode
from src.reference.select import for_year, join_reference
from src.reference.variables import VARIABLES
from src.reference.vdem import load_vdem
from src.reference.wgi import (
    CODE,
    ESTIMATE,
    LOWER,
    NAME,
    STD_ERROR,
    UPPER,
    YEAR,
    load_wgi,
)
from src.store.database import (
    connect,
    load_reference,
    load_reference_files,
    save_reference,
)


def temp_dir() -> Path:
    return Path(tempfile.mkdtemp())


# === WGI ===

def wgi_sheet(rows: list[tuple]) -> pd.DataFrame:
    """
    rows: (code, name, year, estimate, se). 
    Bounds are estimate -+ 1.
    """
    return pd.DataFrame({
        "ID variable (economy code/ gov. dimension/ year)": "x",
        NAME: [r[1] for r in rows],
        CODE: [r[0] for r in rows],
        YEAR: [r[2] for r in rows],
        ESTIMATE: [r[3] for r in rows],
        STD_ERROR: [r[4] for r in rows],
        LOWER: [r[3] if r[3] == ".." else r[3] - 1 for r in rows],
        UPPER: [r[3] if r[3] == ".." else r[3] + 1 for r in rows],
        "VDM mean": "..",
    })


def make_wgi(path: Path, extra_row: tuple | None = None) -> Path:
    ge = [
        ("GBR", "United Kingdom", 2020, 1.4, 0.2),
        ("NAM", "Namibia", 2020, 0.1, 0.3), # "NA"-like code
        ("ANT", "Netherlands Antilles (former)", 2004, 0.5, 0.4), # dropped
        ("GBR", "United Kingdom", 1998, 1.2, 0.2), # before min_year
        ("NGA", "Nigeria", 2020, "..", ".."), # missing value
    ]
    if extra_row:
        ge.append(extra_row)
    pv = [
        ("GBR", "United Kingdom", 2020, 0.5, 0.25),
        ("NGA", "Nigeria", 2020, -1.8, 0.3),
    ]
    with pd.ExcelWriter(path) as writer:
        wgi_sheet(ge).to_excel(writer, sheet_name="ge", index=False)
        wgi_sheet(pv).to_excel(writer, sheet_name="pv", index=False)
        wgi_sheet(pv).to_excel(writer, sheet_name="va", index=False)
    return path


def test_wgi_loads_the_right_rows() -> None:
    table = load_wgi(make_wgi(temp_dir() / "wgi.xlsx"))
    assert list(table.columns) == REFERENCE_COLUMNS

    found = [(r.variable, r.iso3, r.year, r.value) for r in table.itertuples()]
    assert found == [
        ("wgi_ge", "GBR", 2020, 1.4),
        ("wgi_ge", "NAM", 2020, 0.1),
        ("wgi_pv", "GBR", 2020, 0.5),
        ("wgi_pv", "NGA", 2020, -1.8),
    ], found

    gbr = table[(table["iso3"] == "GBR") & (table["variable"] == "wgi_ge")]
    assert gbr["sd"].item() == 0.2 and gbr["interval"].item() == 0.90

    assert abs(gbr["low"].item() - 0.4) < 1e-9
    assert abs(gbr["high"].item() - 2.4) < 1e-9


def test_wgi_unknown_code_stops_the_load() -> None:
    path = make_wgi(temp_dir() / "wgi.xlsx",
                    extra_row=("ZZZ", "Nowhere", 2020, 0.0, 0.1))
    try:
        load_wgi(path)
    except UnknownReferenceCode as error:
        assert "ZZZ (Nowhere)" in str(error), str(error)
    else:
        raise AssertionError("an unknown code should raise")


# === V-Dem ===

VDEM_VARS = ["v2mecenefm", "v2meharjrn", "v2smgovshutcap"]


def make_vdem(path: Path, extra_code: str | None = None) -> Path:
    rows = [
        # code, name, year, value for each variable
        ("GBR", "United Kingdom", 2020, 2.5),
        ("NAM", "Namibia", 2020, 1.0),
        ("BAD", "Baden", 1850, 0.0), # historical, before min_year
        ("NGA", "Nigeria", 2020, "NA"), # missing value
    ]
    if extra_code:
        rows.append((extra_code, "Somewhere", 2020, 0.3))
    frame = pd.DataFrame({
        "country_name": [r[1] for r in rows],
        "country_text_id": [r[0] for r in rows],
        "year": [r[2] for r in rows],
        "v2x_polyarchy": 0.5, # a column not read
    })
    for var in VDEM_VARS:
        frame[var] = [r[3] for r in rows]
        frame[f"{var}_sd"] = 0.1
        frame[f"{var}_codelow"] = [
            r[3] if r[3] == "NA" else r[3] - 0.2 for r in rows
        ]
        frame[f"{var}_codehigh"] = [
            r[3] if r[3] == "NA" else r[3] + 0.2 for r in rows
        ]
        frame[f"{var}_osp"] = 9.9 # a version not read
    frame.to_csv(path, index=False)
    return path


def test_vdem_loads_the_right_rows() -> None:
    table = load_vdem(make_vdem(temp_dir() / "vdem.csv"))
    assert list(table.columns) == REFERENCE_COLUMNS
    assert sorted(set(table["iso3"])) == ["GBR", "NAM"]
    assert sorted(set(table["variable"])) == VDEM_VARS
    gbr = table[(table["iso3"] == "GBR") & (table["variable"] == "v2meharjrn")]
    assert gbr["value"].item() == 2.5 and gbr["sd"].item() == 0.1
    assert abs(gbr["low"].item() - 2.3) < 1e-9
    assert gbr["interval"].isna().all() # not recorded for V-Dem


def test_vdem_unknown_code_stops_the_load() -> None:
    path = make_vdem(temp_dir() / "vdem.csv", extra_code="XYZ")
    try:
        load_vdem(path)
    except UnknownReferenceCode as error:
        assert "XYZ (Somewhere)" in str(error), str(error)
    else:
        raise AssertionError("an unknown code should raise")


def test_vdem_missing_column_is_reported() -> None:
    path = make_vdem(temp_dir() / "vdem.csv")
    frame = pd.read_csv(path).drop(columns="v2meharjrn_sd")
    frame.to_csv(path, index=False)
    try:
        load_vdem(path)
    except ValueError as error:
        assert "v2meharjrn_sd" in str(error)
    else:
        raise AssertionError("a missing column should raise")


# === Variables ===

def test_directions_match_the_codebooks() -> None:
    """
    From the V-Dem v16 codebook answer scales (see variables.py).
    """
    better_when_high = {n: v.higher_is_better for n, v in VARIABLES.items()}
    assert better_when_high == {
        "wgi_ge": True,
        "wgi_pv": True,
        "v2mecenefm": True,      # 4 = government rarely censors
        "v2meharjrn": True,      # 4 = journalists never harassed
        "v2smgovshutcap": False, # 4 = can shut down all Internet access
    }


# === Selecting & Joining ===

def make_reference() -> pd.DataFrame:
    folder = temp_dir()
    return pd.concat([
        load_wgi(make_wgi(folder / "wgi.xlsx")),
        load_vdem(make_vdem(folder / "vdem.csv")),
    ], ignore_index=True)


def test_for_year_is_exact_and_has_a_fixed_shape() -> None:
    wide = for_year(make_reference(), 2020)
    expected = ["iso3"]
    for name in VARIABLES:
        expected += [name, f"{name}_sd"]
    assert list(wide.columns) == expected
    assert wide["iso3"].tolist() == ["GBR", "NAM", "NGA"]

    # A year with no data gives no rows, never another year's values
    assert len(for_year(make_reference(), 2019)) == 0


def test_join_keeps_every_country_day() -> None:
    days = pd.DataFrame({
        "iso3": ["GBR", "NGA", "GUM"],
        "date": pd.Timestamp("2020-06-01"),
        "n_events": [10, 5, 1],
    })
    joined = join_reference(days, for_year(make_reference(), 2020))
    assert len(joined) == 3
    assert joined.loc[joined["iso3"] == "GUM", "wgi_ge"].isna().all()
    coverage = joined.attrs["reference_coverage"]
    assert coverage["reference_year"] == 2020
    assert coverage["with_all_variables"] == 1 # GBR only
    assert coverage["with_no_variables"] == 1 # GUM
    assert coverage["events_without_reference"] == 6


# === Database ===

def test_reference_is_saved_and_replaced_by_source() -> None:
    reference = make_reference()
    wgi = reference[reference["source"] == "wgi"].reset_index(drop=True)
    vdem = reference[reference["source"] == "vdem"].reset_index(drop=True)

    connection = connect(":memory:")
    save_reference(connection, wgi, "wgi", "wgi.xlsx", "abc")
    save_reference(connection, vdem, "vdem", "vdem.csv", "def")
    save_reference(connection, wgi, "wgi", "wgi.xlsx", "abc") # again

    loaded = load_reference(connection)
    assert len(loaded) == len(wgi) + len(vdem) # replaced, not added
    files = load_reference_files(connection)
    assert files["source"].tolist() == ["vdem", "wgi"]
    assert files["sha256"].tolist() == ["def", "abc"]

    only = load_reference(connection, variables=["wgi_pv"], start_year=2020)
    assert set(only["variable"]) == {"wgi_pv"} and len(only) == 2
    connection.close()


def test_rows_from_another_source_are_refused() -> None:
    reference = make_reference()
    connection = connect(":memory:")
    try:
        save_reference(connection, reference, "wgi", "x", "y")
    except ValueError as error:
        assert "vdem" in str(error)
    else:
        raise AssertionError("mixed sources should raise")
    connection.close()


def main() -> None:
    argparse.ArgumentParser().parse_args()
    tests = [
        test_wgi_loads_the_right_rows,
        test_wgi_unknown_code_stops_the_load,
        test_vdem_loads_the_right_rows,
        test_vdem_unknown_code_stops_the_load,
        test_vdem_missing_column_is_reported,
        test_directions_match_the_codebooks,
        test_for_year_is_exact_and_has_a_fixed_shape,
        test_join_keeps_every_country_day,
        test_reference_is_saved_and_replaced_by_source,
        test_rows_from_another_source_are_refused,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
