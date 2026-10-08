"""
Converts GDELT country codes (FIPS 10-4) into ISO 3166-1 alpha-3 codes (ISO3).

Why this is needed:
    - GDELT's ActionGeo_CountryCode column uses FIPS 10-4, an old two-letter
      United States government standard.
    - V-Dem and WGI (the datasets this will be joined with) use ISO3.
    - Many two-letter codes exist in both systems but mean different
      countries. FIPS 'NI' is Nigeria, ISO 'NI' is Nicaragua. A join without
      a conversion would not crash, but would put events in the wrong country.

The conversion table lives in fips_iso3.csv, next to this file. It covers
every code in GDELT's own country list:
http://data.gdeltproject.org/api/v2/guides/LOOKUP-COUNTRIES.TXT

Things to note:
    - Several FIPS codes can share one ISO3 code. Gaza Strip (GZ) and
      West Bank (WE) both become PSE.
    - Kosovo (KV) becomes XKX. This is not an official ISO code, but it is
      the code WGI and V-Dem use.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CROSSWALK_PATH = Path(__file__).parent / "fips_iso3.csv"

# Codes in GDELT's country list that are not countries, or that have no ISO3
# code. Rows with these codes get a missing value instead of a country.
NON_COUNTRY_CODES: frozenset[str] = frozenset(
    {
        "AT",  # Ashmore and Cartier Islands
        "AX",  # Akrotiri Sovereign Base Area
        "CR",  # Coral Sea Islands
        "DX",  # Dhekelia Sovereign Base Area
        "IP",  # Clipperton Island
        "NM",  # No Man's Land
        "NT",  # Netherlands Antilles (dissolved in 2010)
        "OS",  # Oceans
        "PF",  # Paracel Islands
        "PG",  # Spratly Islands
        "PJ",  # Etorofu, Habomai, Kunashiri and Shikotan Islands
        "UF",  # Undersea Features
        "UU",  # Undesignated Sovereignty
    }
)


class CrosswalkError(ValueError):
    """
    The crosswalk file itself is wrong (bad code, duplicate, wrong columns).
    """


class UnknownCountryCode(ValueError):
    """
    The data contains a code that is in neither the crosswalk nor
    NON_COUNTRY_CODES.
    """


def load_crosswalk(path: Path = CROSSWALK_PATH) -> dict[str, str]:
    """
    Load fips_iso3.csv and return it as a {FIPS: ISO3} dictionary.

    The file is checked every time it is loaded. It is edited by hand, and a
    typo here would put a country's events in the wrong place for every
    result that follows.
    """
    # keep_default_na=False stops pandas turning text such as "NA" into a
    # missing value. Country codes are exactly the kind of text it mistakes.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)

    expected = ["fips", "iso3", "name"]
    if list(frame.columns) != expected:
        raise CrosswalkError(
            f"expected columns {expected}, got {list(frame.columns)}"
        )

    bad_fips = frame.loc[~frame["fips"].str.fullmatch(r"[A-Z]{2}"), "fips"]
    if not bad_fips.empty:
        raise CrosswalkError(
            f"FIPS codes must be two capital letters: {bad_fips.tolist()}"
        )

    bad_iso3 = frame.loc[~frame["iso3"].str.fullmatch(r"[A-Z]{3}"), "iso3"]
    if not bad_iso3.empty:
        raise CrosswalkError(
            f"ISO3 codes must be three capital letters: {bad_iso3.tolist()}"
        )

    # A FIPS code listed twice would make the conversion ambiguous.
    # (An ISO3 code listed twice is allowed, see the module docstring.)
    duplicates = frame.loc[frame["fips"].duplicated(), "fips"]
    if not duplicates.empty:
        raise CrosswalkError(f"duplicate FIPS codes: {duplicates.tolist()}")

    # A code cannot be both a country and a non-country.
    overlap = sorted(set(frame["fips"]) & NON_COUNTRY_CODES)
    if overlap:
        raise CrosswalkError(
            f"codes in both the crosswalk and NON_COUNTRY_CODES: {overlap}"
        )

    return dict(zip(frame["fips"], frame["iso3"]))


def to_iso3(
        codes: pd.Series,
        crosswalk: dict[str, str] | None = None,
    ) -> pd.Series:
    """
    Convert a column of FIPS codes into a column of ISO3 codes.

    The result has the same length and index as the input, so it can be
    assigned straight back as a new column.

    Three possible outcomes for each row:
        - known country      -> its ISO3 code
        - empty, or a code
          in NON_COUNTRY_CODES -> missing value
        - anything else      -> UnknownCountryCode is raised

    The third case stops the run on purpose. Skipping unknown codes quietly
    would leave countries missing from the results with nothing to show why.
    """
    if crosswalk is None:
        crosswalk = load_crosswalk()

    # Tidy the input: empty cells become "", stray spaces and lower case go.
    cleaned = codes.astype("string").fillna("").str.strip().str.upper()

    recognised = set(crosswalk) | NON_COUNTRY_CODES | {""}
    unknown = cleaned[~cleaned.isin(recognised)]
    if not unknown.empty:
        counts = unknown.value_counts().to_dict()
        raise UnknownCountryCode(
            f"unrecognised country codes (code: number of rows): {counts}. "
            "Add each one to fips_iso3.csv if it is a country, "
            "or to NON_COUNTRY_CODES if it is not."
        )

    # .map() looks each code up in the dictionary. Codes that are not keys
    # (i.e. "" and the non-country codes) come back as missing values.
    return cleaned.map(crosswalk).astype("string")
