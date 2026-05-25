"""Match model: Poisson + Dixon-Coles τ. predict_match + sample_match."""
from __future__ import annotations

import numpy as np

from .ratings.base import RatingSystem
from .types import MatchResult, Params, Team

SCORE_GRID_MAX = 8


def _poisson_pmf(lmbda: float, max_goals: int) -> np.ndarray:
    if abs(lmbda) < 1e-15:
        pmf = np.zeros(max_goals + 1)
        pmf[0] = 1.0
        return pmf
    k = np.arange(max_goals + 1)
    log_fact = np.cumsum(np.log(np.maximum(k, 1)))
    log_fact[0] = 0.0
    log_pmf = k * np.log(lmbda) - lmbda - log_fact
    return np.exp(log_pmf)


def _apply_tau(grid: np.ndarray, lam_a: float, lam_b: float, rho: float) -> np.ndarray:
    if abs(rho) < 1e-15:
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


ET_LAMBDA_SCALE = 30.0 / 90.0


def _sample_score(lam_a: float, lam_b: float, rho: float, rng: np.random.Generator) -> tuple[int, int]:
    """Sample (home_goals, away_goals) from the τ-corrected joint Poisson grid.
    When rho is ~0, uses fast independent Poisson sampling."""
    if abs(rho) < 1e-15:
        ha = int(min(rng.poisson(lam_a), SCORE_GRID_MAX))
        hb = int(min(rng.poisson(lam_b), SCORE_GRID_MAX))
        return ha, hb
    pa = _poisson_pmf(lam_a, SCORE_GRID_MAX)
    pb = _poisson_pmf(lam_b, SCORE_GRID_MAX)
    grid = _apply_tau(np.outer(pa, pb), lam_a, lam_b, rho)
    grid = grid / grid.sum()
    flat = grid.flatten()
    idx = int(rng.choice(len(flat), p=flat))
    n = SCORE_GRID_MAX + 1
    return idx // n, idx % n


def sample_match(
    team_a: Team, team_b: Team, *,
    rating: RatingSystem, params: Params | None = None,
    a_is_host: bool = False, b_is_host: bool = False,
    rng: np.random.Generator,
    stage: str = "group",
    live_ratings: dict[str, float] | None = None,
) -> MatchResult:
    """Sample a single match. For non-group stages, plays ET then pens if tied.
    If live_ratings is provided, uses those values instead of rating.rating_of()
    for computing the rating diff (supports shrinkage + in-tournament updates)."""
    p = params if params is not None else Params()
    if live_ratings:
        host_diff = float(a_is_host) - float(b_is_host)
        diff = (live_ratings[team_a.iso3] - live_ratings[team_b.iso3]) + rating.home_bonus * host_diff
        r_a = live_ratings[team_a.iso3]
        r_b = live_ratings[team_b.iso3]
    else:
        diff = rating.rating_diff(team_a, team_b, a_is_host, b_is_host)
        r_a = rating.rating_of(team_a)
        r_b = rating.rating_of(team_b)
    lam_a, lam_b = rating.lambdas(diff, p.mu, p.lambda_min)

    home_goals, away_goals = _sample_score(lam_a, lam_b, p.rho, rng)
    extra_time = False
    went_to_pens = False
    pen_winner: str | None = None

    if stage != "group" and home_goals == away_goals:
        extra_time = True
        et_h, et_a = _sample_score(lam_a * ET_LAMBDA_SCALE, lam_b * ET_LAMBDA_SCALE, p.rho, rng)
        home_goals += et_h
        away_goals += et_a
        if home_goals == away_goals:
            went_to_pens = True
            w_e = rating.win_expectation(diff)
            pen_winner = team_a.iso3 if rng.random() < w_e else team_b.iso3

    return MatchResult(
        home=team_a.iso3, away=team_b.iso3,
        home_goals=home_goals, away_goals=away_goals,
        stage=stage,
        neutral=not (a_is_host or b_is_host),
        extra_time=extra_time, went_to_pens=went_to_pens, pen_winner=pen_winner,
        home_rating_before=r_a,
        away_rating_before=r_b,
    )
