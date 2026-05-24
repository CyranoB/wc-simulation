"""Blended Elo + FIFA rating per PRD §5.5. R_blend = w*R_elo + (1-w)*R_fifa*E0/F0.
Predictions use Elo-space params (S=400, c=300, H=100). Update applies
Elo-style update to the blended value directly (simplified; the PRD says
both components update independently, but the single-value Protocol interface
requires a single before/after value)."""
from __future__ import annotations

from ..types import Params, Team
from .elo import EloRating


class BlendRating:
    name = "blend"
    scale = 400.0

    def __init__(self, params: Params):
        self._params = params
        self.c = params.c_elo
        self.home_bonus = params.home_bonus_elo
        self.k = params.k_elo
        self._w = params.blend_w
        self._e0 = params.e0
        self._f0 = params.f0
        self._elo = EloRating(params)

    def rating_of(self, team: Team) -> float:
        if team.fifa_points is None:
            raise ValueError(f"Team {team.iso3} has no fifa_points; BlendRating requires it.")
        fifa_in_elo_space = team.fifa_points * self._e0 / self._f0
        return self._w * team.elo + (1.0 - self._w) * fifa_in_elo_space

    def rating_diff(self, a: Team, b: Team, a_is_host: bool, b_is_host: bool) -> float:
        host_diff = float(a_is_host) - float(b_is_host)
        return (self.rating_of(a) - self.rating_of(b)) + self.home_bonus * host_diff

    def win_expectation(self, diff: float) -> float:
        return 1.0 / (1.0 + 10.0 ** (-diff / self.scale))

    def lambdas(self, diff: float, mu: float, lambda_min: float) -> tuple[float, float]:
        lam_a = max(lambda_min, mu + diff / (2.0 * self.c))
        lam_b = max(lambda_min, mu - diff / (2.0 * self.c))
        return lam_a, lam_b

    def update(self, before: float, expected: float, score_home: int, score_away: int) -> float:
        return self._elo.update(before, expected, score_home, score_away)
