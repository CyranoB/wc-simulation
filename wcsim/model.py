"""Match model: Poisson + Dixon-Coles τ. Pure deterministic predict_match."""
from __future__ import annotations
import numpy as np
from .ratings.base import RatingSystem
from .types import Params, Team

SCORE_GRID_MAX = 8


def _poisson_pmf(lmbda: float, max_goals: int) -> np.ndarray:
    if lmbda == 0.0:
        pmf = np.zeros(max_goals + 1)
        pmf[0] = 1.0
        return pmf
    k = np.arange(max_goals + 1)
    log_fact = np.cumsum(np.log(np.maximum(k, 1)))
    log_fact[0] = 0.0
    log_pmf = k * np.log(lmbda) - lmbda - log_fact
    return np.exp(log_pmf)


def _apply_tau(grid: np.ndarray, lam_a: float, lam_b: float, rho: float) -> np.ndarray:
    if rho == 0.0:
        return grid
    grid[0, 0] *= max(0.0, 1.0 - lam_a * lam_b * rho)
    grid[0, 1] *= max(0.0, 1.0 + lam_a * rho)
    grid[1, 0] *= max(0.0, 1.0 + lam_b * rho)
    grid[1, 1] *= max(0.0, 1.0 - rho)
    return grid


def _outcome_probs(lam_a: float, lam_b: float, rho: float) -> tuple[float, float, float]:
    pa = _poisson_pmf(lam_a, SCORE_GRID_MAX)
    pb = _poisson_pmf(lam_b, SCORE_GRID_MAX)
    grid = _apply_tau(np.outer(pa, pb), lam_a, lam_b, rho)
    p_home = float(np.tril(grid, k=-1).sum())
    p_draw = float(np.trace(grid))
    p_away = float(np.triu(grid, k=1).sum())
    s = p_home + p_draw + p_away
    return p_home / s, p_draw / s, p_away / s


def predict_match(
    team_a: Team, team_b: Team, *,
    rating: RatingSystem, params: Params | None = None,
    a_is_host: bool = False, b_is_host: bool = False,
) -> tuple[float, float, float]:
    """Return (P(team_a wins), P(draw), P(team_b wins)). Pure function."""
    p = params if params is not None else Params()
    diff = rating.rating_diff(team_a, team_b, a_is_host, b_is_host)
    lam_a, lam_b = rating.lambdas(diff, p.mu, p.lambda_min)
    return _outcome_probs(lam_a, lam_b, p.rho)
