"""Pluggable rating-system interface. Concrete classes (EloRating,
FifaRating, BlendRating) implement this Protocol structurally — no
inheritance required."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Params, Team


@runtime_checkable
class RatingSystem(Protocol):
    name: str
    scale: float
    c: float
    home_bonus: float

    def rating_of(self, team: Team) -> float: ...

    def rating_diff(
        self, a: Team, b: Team, a_is_host: bool, b_is_host: bool,
    ) -> float: ...

    def win_expectation(self, diff: float) -> float: ...

    def lambdas(
        self, diff: float, mu: float, lambda_min: float,
    ) -> tuple[float, float]: ...

    def update(
        self, before: float, expected: float,
        score_home: int, score_away: int,
    ) -> float: ...
