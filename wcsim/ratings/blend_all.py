"""Three-way Elo + FIFA + Player blend rating.

R = w_player * R_player + (1 - w_player) * R_elo_fifa_blend

where R_elo_fifa_blend is the existing two-way blend (w*R_elo + (1-w)*R_fifa*E0/F0).
When blend_player=0, this is identical to BlendRating."""
from __future__ import annotations

from ..types import Params, Team
from .blend import BlendRating
from .elo import EloRating
from .player import PlayerRating


class BlendAllRating:
    name = "blend_all"
    scale = 400.0

    def __init__(self, params: Params, squad_data: dict[str, dict]):
        self._params = params
        self.c = params.c_elo
        self.home_bonus = params.home_bonus_elo
        self.k = params.k_elo
        self._w_player = params.blend_player
        self._blend = BlendRating(params)
        self._player = PlayerRating(params, squad_data)
        self._elo = EloRating(params)

    def rating_of(self, team: Team) -> float:
        wp = self._w_player
        r_player = self._player.rating_of(team)
        r_base = self._blend.rating_of(team)
        return wp * r_player + (1.0 - wp) * r_base

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
