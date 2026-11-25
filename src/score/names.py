"""
Readable names for the criteria, and the groups they are shown in. Used by
the printed outputs and the web export, so both use the same words.
"""

from __future__ import annotations

CRITERION_NAMES: dict[str, str] = {
    "wgi_ge": "Government effectiveness",
    "wgi_pv": "Political stability",
    "v2mecenefm": "Media censorship",
    "v2meharjrn": "Harassment of journalists",
    "v2smgovshutcap": "Internet shutdown capacity",
    "v2smgovdom": "Government disinformation",
    "unrest": "Domestic unrest",
    "media_pressure": "Pressure on media",
    "intl_conflict": "International conflict",
    "intl_coop": "International cooperation",
    "relations": "International relations",
}

# The three groups the International View shows the criteria in
GROUPS: dict[str, str] = {
    "structural": "Structural conditions (yearly)",
    "current": "Current conditions (news events)",
    "international": "International relations (news events)",
}

CRITERION_GROUPS: dict[str, str] = {
    "wgi_ge": "structural",
    "wgi_pv": "structural",
    "v2mecenefm": "structural",
    "v2meharjrn": "structural",
    "v2smgovshutcap": "structural",
    "v2smgovdom": "structural",
    "unrest": "current",
    "media_pressure": "current",
    "intl_conflict": "international",
    "intl_coop": "international",
    "relations": "international",
}


def name(criterion: str) -> str:
    """
    The readable name of a criterion, or its code if it has none.
    """
    return CRITERION_NAMES.get(criterion, criterion)
