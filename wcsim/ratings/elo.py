"""Elo rating system per PRD §5.5. Scale S=400, c=300, K=60.
Goal-margin multiplier G_m: 1 (draw or 1-goal), 1.5 (2-goal),
(11 + |Δ|) / 8 (margins of 3+)."""
from __future__ import annotations

from ..types import Params, Team


class EloRating:
    name = "elo"
    scale = 400.0

    def __init__(self, params: Params):
        self._params = params
        self.c = params.c_elo
        self.home_bonus = params.home_bonus_elo
        self.k = params.k_elo

    def rating_of(self, team: Team) -> float:
        return team.elo

    def rating_diff(self, a: Team, b: Team, a_is_host: bool, b_is_host: bool) -> float:
        host_diff = float(a_is_host) - float(b_is_host)
        return (a.elo - b.elo) + self.home_bonus * host_diff

    def win_expectation(self, diff: float) -> float:
        return 1.0 / (1.0 + 10.0 ** (-diff / self.scale))

    def lambdas(self, diff: float, mu: float, lambda_min: float) -> tuple[float, float]:
        lam_a = max(lambda_min, mu + diff / (2.0 * self.c))
        lam_b = max(lambda_min, mu - diff / (2.0 * self.c))
        return lam_a, lam_b

    def update(self, before: float, expected: float, score_home: int, score_away: int) -> float:
        delta = score_home - score_away
        if score_home == score_away:
            w_actual = 0.5
        elif delta > 0:
            w_actual = 1.0
        else:
            w_actual = 0.0
        abs_delta = abs(delta)
        if abs_delta <= 1:
            gm = 1.0
        elif abs_delta == 2:
            gm = 1.5
        else:
            gm = (11.0 + abs_delta) / 8.0
        return before + self.k * gm * (w_actual - expected)
