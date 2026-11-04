"""
Every method choice in scoring, in one object.

Each function in src/score/ takes a Settings.

Sensitivity analysis means building many Settings and scoring with each,
so anything that is not here cannot be varied.

Settings.validate() refuses combinations that cannot work. Therefore, a bad
combination will fail at the start of a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NORMALISATIONS = ("minmax", "zscore")
AGGREGATIONS = ("arithmetic", "geometric")
UNREST_CODE_SETS = ("strict", "loose")
DENOMINATORS = ("domestic", "all")
MEDIA_METHODS = ("cross_country", "deviation")
UNREST_WEIGHTINGS = ("tiers", "count", "goldstein")
BOUNDS = ("fixed", "per_run")


class InvalidSettings(ValueError):
    """
    A setting (or a combination of settings) that cannot be used.
    """


@dataclass(frozen=True)
class Settings:
    # --- GDELT criteria (src/score/fast.py) ---

    # Days before the assessment date that define a country's "normal"
    baseline_days: int = 90

    # Share of those days that must have been pulled completely
    min_baseline_coverage: float = 0.8

    # Smoothing of shares over a window (baseline + media's 30-day share)
    prior_strength: float = 20.0 # 0 = no impact

    # Smoothing of the deviation ratio (observed / expected events today)
    pseudo_events: float = 1.0

    # Unrest:
    # > "strict" = root codes 14, 18, 19, 20
    # > "loose" = root code 17
    unrest_codes: str = "strict"

    # How unrest events are weighted:
    # > "tiers": by severity tier from the CAMEO hierarchy
    # > "count": every event counts 1
    # > "goldstein": |Goldstein| / 10
    unrest_weighting: str = "tiers"

    # Unrest share denominator: 
    # > "domestic" = only domestic-coded events
    # > "all" = all located events (sensitivity variant)
    unrest_denominator: str = "domestic"

    # Note: Pressure on media is mostly zeros -> by default it is NOT compared
    # with the country's own baseline (its share over a longer window is
    # compared across countries instead)
    media_method: str = "cross_country"
    media_window_days: int = 30

    # Below this many events in a day = low data
    low_data_events: int = 50


    # --- Combining criteria (src/score/combine.py) ---

    # Which criteria are scored:
    # - Empty = the default set (the four GDELT criteria and the structural 
    #           variables marked "criterion" in src/reference/variables.py)
    criteria: tuple[str, ...] = ()

    # "fixed": normalisation bounds come from the norm_bounds table
    # "per_run": from whichever countries are in this run
    bounds: str = "fixed"

    # GDELT criteria have no natural top -> upper bound is this percentile of 
    # all country-day values over the reference period
    cap_percentile: float = 99.0

    normalisation: str = "minmax"
    minmax_floor: float = 0.0
    aggregation: str = "arithmetic"

    # Criterion name -> weight (empty means equal weights)
    weights: dict[str, float] = field(default_factory=dict)

    # Score only countries that have every criterion
    require_all_criteria: bool = True
    bands: int = 5


    # --- Reference data ---

    reference_lag_years: int = 1

    def validate(self) -> Settings:
        """
        Raise InvalidSettings for anything unusable; return self.
        """
        problems: list[str] = []

        def check(ok: bool, message: str) -> None:
            if not ok:
                problems.append(message)

        check(self.baseline_days >= 7, "baseline_days must be at least 7")

        check(0 < self.min_baseline_coverage <= 1,
              "min_baseline_coverage must be in (0, 1]")
        
        check(self.prior_strength >= 0, "prior_strength cannot be negative")

        check(self.pseudo_events >= 0, "pseudo_events cannot be negative")

        check(self.unrest_codes in UNREST_CODE_SETS,
              f"unrest_codes must be one of {UNREST_CODE_SETS}")
        
        check(self.unrest_denominator in DENOMINATORS,
              f"unrest_denominator must be one of {DENOMINATORS}")
        
        check(self.media_method in MEDIA_METHODS,
              f"media_method must be one of {MEDIA_METHODS}")
        
        check(self.unrest_weighting in UNREST_WEIGHTINGS,
              f"unrest_weighting must be one of {UNREST_WEIGHTINGS}")
        
        check(self.bounds in BOUNDS, f"bounds must be one of {BOUNDS}")

        check(50 <= self.cap_percentile <= 100,
              "cap_percentile must be between 50 and 100")
        
        check(len(set(self.criteria)) == len(self.criteria),
              "criteria are listed more than once")
        
        check(self.media_window_days >= 1,
              "media_window_days must be at least 1")
        
        check(self.normalisation in NORMALISATIONS,
              f"normalisation must be one of {NORMALISATIONS}")
        
        check(0 <= self.minmax_floor < 1, "minmax_floor must be in [0, 1)")

        check(self.aggregation in AGGREGATIONS,
              f"aggregation must be one of {AGGREGATIONS}")
        
        check(all(w > 0 for w in self.weights.values()),
              "weights must all be positive")
        
        check(self.bands >= 2, "bands must be at least 2")

        check(self.reference_lag_years >= 0,
              "reference_lag_years cannot be negative")

        # Every normalised value must be strictly positive
        if self.aggregation == "geometric":
            check(self.normalisation == "minmax",
                  "geometric aggregation needs min-max normalisation "
                  "(z-scores can be zero or negative)")
            check(self.minmax_floor > 0,
                  "geometric aggregation needs minmax_floor above 0, e.g. "
                  "0.1: with a floor of 0 the worst country on any "
                  "criterion scores 0 overall")

        if problems:
            raise InvalidSettings("; ".join(problems))
        return self
