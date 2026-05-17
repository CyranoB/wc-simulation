"""Player-value rating: team strength from squad market values (Transfermarkt).
Log-normalized to Elo-equivalent scale so existing model parameters work."""
from __future__ import annotations

import math
import statistics

from ..types import Params, Team


class PlayerRating:
    name = "player"
    scale = 400.0

    def __init__(self, params: Params, squad_data: dict[str, dict]):
        self._params = params
        self.c = params.c_elo
        self.home_bonus = params.home_bonus_elo
        self.k = params.k_elo

        squad_values = {
            iso3: data["total_value_eur"]
            for iso3, data in squad_data.items()
        }
        log_vals = [math.log10(max(v, 1)) for v in squad_values.values()]
        log_mean = statistics.mean(log_vals)
        log_std = statistics.stdev(log_vals) if len(log_vals) > 1 else 1.0
        target_std = 190.0
        alpha = target_std / log_std if log_std > 0 else 1.0

        self._ratings: dict[str, float] = {
            iso3: 1500.0 + alpha * (math.log10(max(val, 1)) - log_mean)
            for iso3, val in squad_values.items()
        }

    def rating_of(self, team: Team) -> float:
        if team.iso3 not in self._ratings:
            raise ValueError(f"Team {team.iso3} has no squad data")
        return self._ratings[team.iso3]

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
        return before
