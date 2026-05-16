"""Public data types for wcsim. PRD §8-aligned. FIFA fields on Team are
optional to support Elo-only inputs (Spike 1 finding)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Team:
    name: str
    iso3: str
    confederation: str
    elo: float
    elo_updated: date | None = None
    fifa_points: float | None = None
    fifa_rank: int | None = None
    fifa_updated: date | None = None


@dataclass(frozen=True)
class MatchResult:
    home: str
    away: str
    home_goals: int
    away_goals: int
    stage: str
    neutral: bool
    extra_time: bool
    went_to_pens: bool
    pen_winner: str | None
    home_rating_before: float
    away_rating_before: float


@dataclass(frozen=True)
class TournamentResult:
    seed: int
    rating_mode: str
    matches: list[MatchResult]
    placements: dict[str, str]
    final_ratings: dict[str, float]


@dataclass(frozen=True)
class Params:
    """PRD v1.7 §5.5 model parameters. All defaults match the PRD."""
    c_elo: float = 300.0
    c_fifa: float = 450.0
    mu: float = 1.35
    lambda_min: float = 0.05
    blend_w: float = 0.7
    e0: float = 1500.0
    f0: float = 1300.0
    home_bonus_elo: float = 100.0
    home_bonus_fifa: float = 150.0
    rho: float = 0.0
    k_elo: float = 60.0
    k_fifa: float = 60.0
    shrinkage: float = 1.0  # 1.0 = use raw ratings; <1 compresses toward field mean


@dataclass(frozen=True)
class SimulationResult:
    """Aggregated output of N tournament simulations."""
    n: int
    seed: int
    probabilities: dict[str, dict[str, float]]
    ci_lo: dict[str, dict[str, float]]
    ci_hi: dict[str, dict[str, float]]
    mean_goals_for: dict[str, float]
    mean_goals_against: dict[str, float]
