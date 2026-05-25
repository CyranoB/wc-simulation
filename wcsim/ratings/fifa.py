"""FIFA Men's World Ranking system per PRD §5.5. Scale S=600, c=450,
home bonus 150. Update: R' = R + I * (W - W_e) with I = k_fifa (60)."""
from __future__ import annotations

from ..types import Params, Team


class FifaRating:
    name = "fifa"
    scale = 600.0

    def __init__(self, params: Params):
        self._params = params
        self.c = params.c_fifa
        self.home_bonus = params.home_bonus_fifa
        self.k = params.k_fifa

    def rating_of(self, team: Team) -> float:
        if team.fifa_points is None:
            raise ValueError(
                f"Team {team.iso3} has no fifa_points; FifaRating requires it."
            )
        return team.fifa_points

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
        delta = score_home - score_away
        if score_home == score_away:
            w_actual = 0.5
        elif delta > 0:
            w_actual = 1.0
        else:
            w_actual = 0.0
        return before + self.k * (w_actual - expected)
