"""
Defines the structural indicators used in the country assessment.

All structural indicators are listed here in one place, along
with their data source, description and scoring direction.

For each indicator, we record:

    name
        The name used throughout the project.

    source
        Where the data comes from (V-Dem, WGI or World Bank).

    source_name
        The original indicator name in the source data.

    label
        A description of what the indicator measures.

    higher_is_better
        Whether a higher value represents better or worse
        conditions for receiving an announcement.

    role
        Whether the indicator is used for scoring:

            criterion
                Included in the default country assessment.

            candidate
                Available for testing, but only included when
                explicitly selected in Settings.criteria.

            validation
                Not included in scoring. Used for independent
                checks, such as comparing scores with GDP per capita.

    bounds
        The natural minimum and maximum values, where the
        measurement scale has fixed limits.

Scoring direction is based on the original source documentation.
It is important to get this right because the scoring system
automatically adjusts each indicator so that higher always means
better conditions.

Data sources:

    V-Dem
        Measures media freedom, journalist safety, Internet
        shutdown capacity and other political conditions.

    Digital Society Project
        Provides indicators of disinformation: government
        dissemination of false information is a criterion (it bears
        on the Declaration's rumour-management provision); party
        dissemination is a candidate kept for sensitivity analysis.
        Polarisation of society (v2smpolsoc) was considered and
        excluded: it measures general polarisation, not the
        information environment (decision log 11.7).

    WGI (Worldwide Governance Indicators)
        Provides measures of government effectiveness and
        political stability, on a 0-100 scale.

    World Bank
        Provides GDP per capita for independent validation only.

V-Dem estimates do not have fixed minimum and maximum values.
Their normalisation limits are therefore calculated from the
chosen reference period.

Main functions:

    variables_from()
        Returns the indicators provided by a particular source.

    with_role()
        Returns the indicators assigned to a particular role.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variable:
    name: str # name used in this project's tables
    source: str # "vdem" or "wgi"
    source_name: str # column (V-Dem) or sheet (WGI) in the file
    label: str
    higher_is_better: bool
    role: str = "criterion"
    bounds: tuple[float, float] | None = None


VARIABLES: dict[str, Variable] = {
    v.name: v
    for v in (

        Variable(
            "wgi_ge", "wgi", "ge",
            "Government effectiveness (score 0-100)",
            higher_is_better=True, bounds=(0.0, 100.0),
        ),

        Variable(
            "wgi_pv", "wgi", "pv",
            "Political stability and absence of violence/terrorism "
            "(score 0-100)",
            higher_is_better=True, bounds=(0.0, 100.0),
        ),

        Variable(
            "v2mecenefm", "vdem", "v2mecenefm",
            "Government censorship effort, media (high = less censorship)",
            higher_is_better=True,
        ),

        Variable(
            "v2meharjrn", "vdem", "v2meharjrn",
            "Harassment of journalists (high = less harassment)",
            higher_is_better=True,
        ),

        Variable(
            "v2smgovshutcap", "vdem", "v2smgovshutcap",
            "Government Internet shut down capacity (high = more capacity)",
            higher_is_better=False,
        ),

        Variable(
            "v2smgovdom", "vdem", "v2smgovdom",
            "Government dissemination of false information, domestic "
            "(high = less)",
            higher_is_better=True,
        ),

        Variable(
            "v2smpardom", "vdem", "v2smpardom",
            "Party dissemination of false information, domestic "
            "(high = less)",
            higher_is_better=True, role="candidate",
        ),

        Variable(
            "gdp_pc", "wb", "NY.GDP.PCAP.PP.KD",
            "GDP per capita, PPP (constant 2021 international $)",
            higher_is_better=True, role="validation",
        ),
    )
}

ROLES = ("criterion", "candidate", "validation")


def variables_from(source: str) -> list[Variable]:
    """
    The variables that come from one source, in a fixed order.
    """
    return [v for v in VARIABLES.values() if v.source == source]


def with_role(role: str) -> list[str]:
    """
    Names of the variables with one role, in a fixed order.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    
    return [n for n, v in VARIABLES.items() if v.role == role]
