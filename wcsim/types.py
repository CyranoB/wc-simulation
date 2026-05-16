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
