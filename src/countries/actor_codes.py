"""
Converts GDELT actor country codes into ISO3 codes.

GDELT uses three different country code systems:
    - ActionGeo_CountryCode   FIPS, two letters      -> see codes.py
    - Actor1CountryCode and
      Actor2CountryCode       CAMEO, three letters   -> this file
    - V-Dem and WGI           ISO3, three letters

Most CAMEO country codes are the same as ISO3, so the rule here is:
    1. If the code is in cameo_iso3.csv, use the ISO3 code given there.
       That file only lists the codes that differ.
    2. Otherwise, if the code is an ISO3 code this project knows
       (any ISO3 code in fips_iso3.csv), keep it as it is.
    3. Otherwise it is not a country: the result is a missing value.

Step 3 does not stop the run, unlike codes.py. The actor field also holds
codes that are not countries at all, such as regions (EUR for Europe), so
an unmatched code is normal here. Unmatched codes are counted and reported
instead, so that a real country code that needs adding can be spotted.

Note: cameo_iso3.csv is not a full list of differences. Will add to it as we
keep going.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.countries.codes import load_crosswalk

OVERRIDES_PATH = Path(__file__).parent / "cameo_iso3.csv"


class ActorCodeError(ValueError):
    """
    The override file itself is wrong (bad code, duplicate, wrong columns).
    """


def load_actor_overrides(path: Path = OVERRIDES_PATH) -> dict[str, str]:
    """
    Load cameo_iso3.csv and return it as a {CAMEO: ISO3} dictionary.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)

    expected = ["cameo", "iso3", "name"]
    if list(frame.columns) != expected:
        raise ActorCodeError(
            f"expected columns {expected}, got {list(frame.columns)}"
        )

    for column in ("cameo", "iso3"):
        bad = frame.loc[~frame[column].str.fullmatch(r"[A-Z]{3}"), column]
        if not bad.empty:
            raise ActorCodeError(
                f"{column} codes must be three capital letters: "
                f"{bad.tolist()}"
            )

    duplicates = frame.loc[frame["cameo"].duplicated(), "cameo"]
    if not duplicates.empty:
        raise ActorCodeError(f"duplicate CAMEO codes: {duplicates.tolist()}")

    return dict(zip(frame["cameo"], frame["iso3"], strict=True))


def clean_actor_codes(codes: pd.Series) -> pd.Series:
    """
    Tidy a column of actor codes: empty cells become "", and stray spaces
    and lower case are removed.
    """
    return codes.astype("string").fillna("").str.strip().str.upper()


def actor_to_iso3(codes: pd.Series) -> pd.Series:
    """
    Convert a column of CAMEO actor country codes into ISO3 codes.

    The result has the same length and index as the input. A blank code,
    or a code that is not a country, gives a missing value.
    """
    overrides = load_actor_overrides()
    known_iso3 = set(load_crosswalk().values())

    cleaned = clean_actor_codes(codes)

    # Step 1: codes that differ from ISO3 are replaced
    converted = cleaned.replace(overrides)

    # Steps 2 and 3: keep known ISO3 codes, blank out everything else
    return converted.where(converted.isin(known_iso3)).astype("string")


def unmatched_actor_codes(codes: pd.Series) -> dict[str, int]:
    """
    Count the non-blank actor codes that did not convert to a country.

    Returns {code: number of rows}, most common first.
    """
    cleaned = clean_actor_codes(codes)
    unmatched = cleaned[(cleaned != "") & actor_to_iso3(codes).isna()]
    return {str(k): int(v) for k, v in unmatched.value_counts().items()}