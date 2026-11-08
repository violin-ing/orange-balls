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
    NAME,
    SCORE,
    SCORE_ERROR,
    YEAR,
    Z_90,
    load_wgi,
)
from src.reference.worldbank import load_worldbank
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
    rows: (code, name, year, score 0-100, its standard error). The
    -2.5 to +2.5 estimate columns are present but must not be read.
    """
    return pd.DataFrame({
        "ID variable (economy code/ gov. dimension/ year)": "x",
        NAME: [r[1] for r in rows],
        CODE: [r[0] for r in rows],
        YEAR: [r[2] for r in rows],
        "Governance estimate (approx. -2.5 to +2.5)": 9.9,
        "Standard error (estimate)": 9.9,
        SCORE: [r[3] for r in rows],
        SCORE_ERROR: [r[4] for r in rows],
        "VDM mean": "..",
    })


def make_wgi(path: Path, extra_row: tuple | None = None) -> Path:
    ge = [
        ("GBR", "United Kingdom", 2020, 84.0, 4.0),
        ("NAM", "Namibia", 2020, 51.0, 6.0),       # "NA"-like code
        ("ANT", "Netherlands Antilles (former)", 2004, 60.0, 8.0),  # dropped
        ("GBR", "United Kingdom", 1998, 80.0, 4.0),  # before min_year
        ("NGA", "Nigeria", 2020, "..", ".."),       # missing value
    ]

    if extra_row:
        ge.append(extra_row)

    pv = [
        ("GBR", "United Kingdom", 2020, 70.0, 5.0),
        ("NGA", "Nigeria", 2020, 20.0, 6.0),
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
        ("wgi_ge", "GBR", 2020, 84.0),
        ("wgi_ge", "NAM", 2020, 51.0),
        ("wgi_pv", "GBR", 2020, 70.0),
        ("wgi_pv", "NGA", 2020, 20.0),
    ], found

    gbr = table[(table["iso3"] == "GBR") & (table["variable"] == "wgi_ge")]
    assert gbr["sd"].item() == 4.0 and gbr["interval"].item() == 0.90
    assert abs(gbr["low"].item() - (84.0 - Z_90 * 4.0)) < 1e-9
    assert abs(gbr["high"].item() - (84.0 + Z_90 * 4.0)) < 1e-9


def test_wgi_unknown_code_stops_the_load() -> None:
    path = make_wgi(
        temp_dir() / "wgi.xlsx",
        extra_row=("ZZZ", "Nowhere", 2020, 50.0, 5.0)
    )
    
    try:
        load_wgi(path)
    except UnknownReferenceCode as error:
        assert "ZZZ (Nowhere)" in str(error), str(error)
    else:
        raise AssertionError("an unknown code should raise")


# === V-Dem ===

VDEM_VARS = ["v2mecenefm", "v2meharjrn", "v2smgovshutcap",
             "v2smgovdom", "v2smpardom"]


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
    assert sorted(set(table["variable"])) == sorted(VDEM_VARS)
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
        raise AssertionError("An unknown code should raise")


def test_vdem_missing_column_is_reported() -> None:
    path = make_vdem(temp_dir() / "vdem.csv")

    frame = pd.read_csv(path).drop(columns="v2meharjrn_sd")
    frame.to_csv(path, index=False)

    try:
        load_vdem(path)
    except ValueError as error:
        assert "v2meharjrn_sd" in str(error)
    else:
        raise AssertionError("A missing column should raise")


# === World Bank ===

def make_worldbank(folder: Path) -> Path:
    name = "API_NY.GDP.PCAP.PP.KD_DS2_en_csv_v2_1.csv"

    header = (
        '"Data Source","World Development Indicators",\n\n'
        '"Last Updated Date","2026-10-01",\n\n'
    )

    values = pd.DataFrame({
        "Country Name": ["United Kingdom", "World", "Namibia",
                         "Channel Islands"],
        "Country Code": ["GBR", "WLD", "NAM", "CHI"],
        "Indicator Name": "GDP per capita, PPP",
        "Indicator Code": "NY.GDP.PCAP.PP.KD",
        "1999": [40000.0, 15000.0, 7000.0, 50000.0],
        "2020": [45000.0, 17000.0, None, 55000.0],
        "2021": [46000.0, 18000.0, 9000.0, 56000.0],
    })
    values["Unnamed"] = "" # download ends each line with a comma

    path = folder / name
    path.write_text(header + values.to_csv(index=False), encoding="utf-8")
    pd.DataFrame({
        "Country Code": ["GBR", "WLD", "NAM", "CHI"],
        "Region": ["Europe & Central Asia", "", "Sub-Saharan Africa",
                   "Europe & Central Asia"],
    }).to_csv(folder / f"Metadata_Country_{name}", index=False)

    return path


def test_worldbank_drops_totals_and_old_years() -> None:
    table = load_worldbank(make_worldbank(temp_dir()))
    assert list(table.columns) == REFERENCE_COLUMNS

    found = [(r.iso3, r.year, r.value) for r in table.itertuples()]
    
    # WLD (a total) and CHI (dropped) are gone
    # 1999 is before min_year
    # Namibia's missing 2020 value is left out
    assert found == [
        ("GBR", 2020, 45000.0),
        ("GBR", 2021, 46000.0),
        ("NAM", 2021, 9000.0),
    ], found
    assert set(table["variable"]) == {"gdp_pc"}
    assert table["sd"].isna().all()


# === Variables ===

def test_directions_match_the_codebooks() -> None:
    """From the V-Dem v16 codebook answer scales; see variables.py."""
    better_when_high = {n: v.higher_is_better for n, v in VARIABLES.items()}
    assert better_when_high == {
        "wgi_ge": True,
        "wgi_pv": True,
        "v2mecenefm": True, # 4 = government rarely censors
        "v2meharjrn": True, # 4 = journalists never harassed
        "v2smgovshutcap": False, # 4 = can shut down all Internet access
        "v2smgovdom": True, # 4 = government never spreads false info
        "v2smpardom": True, # 4 = parties never spread false info
        "gdp_pc": True, # not scored; validation only
    }
    roles = {n: v.role for n, v in VARIABLES.items()}
    assert [n for n, r in roles.items() if r == "criterion"] == [
        "wgi_ge", "wgi_pv", "v2mecenefm", "v2meharjrn", "v2smgovshutcap",
        "v2smgovdom",
    ]
    assert roles["v2smpardom"] == "candidate"
    assert roles["gdp_pc"] == "validation"


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

    # A year with no data gives no rows
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
    save_reference(connection, wgi, "wgi", "wgi.xlsx", "abc")  # again

    loaded = load_reference(connection)
    assert len(loaded) == len(wgi) + len(vdem)      # replaced, not added
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
        test_worldbank_drops_totals_and_old_years,
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
