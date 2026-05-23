"""Spike-only Dixon-Coles attack/defence model.

This module intentionally lives outside the public ``wcsim`` package. It fits
team attack and defence parameters from match scores, with rating-derived
priors used only as regularization anchors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

SCORE_GRID_MAX = 8
EPS = 1e-12


@dataclass(frozen=True)
class FitConfig:
    """Configuration for the validation-spike fit."""

    half_life_days: float = 730.0
    prior_sd: float = 0.35
    unmapped_prior_sd: float = 0.20
    rho_bounds: tuple[float, float] = (-0.2, 0.2)
    maxiter: int = 500


@dataclass(frozen=True)
class MatchRecord:
    date: date
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    tournament: str
    neutral: bool


@dataclass(frozen=True)
class PriorMaps:
    strength: dict[str, float]
    attack: dict[str, float]
    defense: dict[str, float]
    sd: dict[str, float]
    mapped_teams: set[str]
    unmapped_teams: set[str]


def _poisson_pmf(lmbda: float, max_goals: int = SCORE_GRID_MAX) -> np.ndarray:
    k = np.arange(max_goals + 1)
    log_fact = np.cumsum(np.log(np.maximum(k, 1)))
    log_fact[0] = 0.0
    return np.exp(k * math.log(max(lmbda, EPS)) - lmbda - log_fact)


def _tau(lam_h: float, lam_a: float, home_goals: int, away_goals: int, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lam_h * lam_a * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lam_h * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lam_a * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def score_probability_grid(lam_home: float, lam_away: float, rho: float) -> np.ndarray:
    """Return normalized 9x9 score grid under the project DC convention."""
    home = _poisson_pmf(lam_home)
    away = _poisson_pmf(lam_away)
    grid = np.outer(home, away)
    grid[0, 0] *= max(0.0, _tau(lam_home, lam_away, 0, 0, rho))
    grid[0, 1] *= max(0.0, _tau(lam_home, lam_away, 0, 1, rho))
    grid[1, 0] *= max(0.0, _tau(lam_home, lam_away, 1, 0, rho))
    grid[1, 1] *= max(0.0, _tau(lam_home, lam_away, 1, 1, rho))
    return grid / grid.sum()


def outcome_probs_from_lambdas(lam_home: float, lam_away: float, rho: float) -> tuple[float, float, float]:
    grid = score_probability_grid(lam_home, lam_away, rho)
    p_home = float(np.tril(grid, k=-1).sum())
    p_draw = float(np.trace(grid))
    p_away = float(np.triu(grid, k=1).sum())
    return p_home, p_draw, p_away


def competition_weight(tournament: str) -> float:
    """Coarse competition weights for broad international-results training."""
    t = tournament.lower()
    if t == "fifa world cup" or (
        any(x in t for x in ("uefa euro", "copa am", "african cup", "asian cup", "gold cup"))
        and "qualification" not in t
        and "qualifying" not in t
    ):
        return 4.0
    if "qualification" in t or "qualifying" in t or "nations league" in t:
        return 2.0
    if t == "friendly":
        return 0.5
    return 0.25


def match_weight(record: MatchRecord, cutoff_date: date, config: FitConfig) -> float:
    days_old = max((cutoff_date - record.date).days, 0)
    recency = 0.5 ** (days_old / config.half_life_days)
    return competition_weight(record.tournament) * recency


def _player_rating_by_iso(squad_data: dict[str, dict]) -> dict[str, float]:
    if not squad_data:
        return {}
    log_values = {
        iso3: math.log10(max(float(data["total_value_eur"]), 1.0))
        for iso3, data in squad_data.items()
    }
    vals = list(log_values.values())
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
    alpha = 190.0 / std if std > 0 else 1.0
    return {iso3: 1500.0 + alpha * (value - mean) for iso3, value in log_values.items()}


def _rating_by_iso(
    iso3: str,
    *,
    elo_by_iso: dict[str, float],
    fifa_by_iso: dict[str, float],
    f0: float,
    player_by_iso: dict[str, float],
    player_weight: float,
) -> float | None:
    parts: list[tuple[float, float]] = []
    if iso3 in elo_by_iso:
        parts.append((0.7 * (1.0 - player_weight), float(elo_by_iso[iso3])))
    if iso3 in fifa_by_iso:
        parts.append((0.3 * (1.0 - player_weight), float(fifa_by_iso[iso3]) * 1500.0 / f0))
    if player_weight > 0 and iso3 in player_by_iso:
        parts.append((player_weight, player_by_iso[iso3]))
    total_weight = sum(w for w, _ in parts)
    if total_weight <= 0:
        return None
    return sum(w * value for w, value in parts) / total_weight


def build_prior_maps(
    team_names: Iterable[str],
    *,
    name_to_iso3: dict[str, str],
    elo_by_iso: dict[str, float],
    fifa_by_iso: dict[str, float],
    f0: float,
    config: FitConfig,
    squad_data: dict[str, dict] | None = None,
    player_weight: float = 0.0,
) -> PriorMaps:
    """Create attack/defence priors keyed by raw team name."""
    teams = sorted(set(team_names))
    player_by_iso = _player_rating_by_iso(squad_data or {})
    raw_ratings: dict[str, float] = {}
    mapped: set[str] = set()
    unmapped: set[str] = set()

    for team in teams:
        iso3 = name_to_iso3.get(team)
        rating = None
        if iso3 is not None:
            rating = _rating_by_iso(
                iso3,
                elo_by_iso=elo_by_iso,
                fifa_by_iso=fifa_by_iso,
                f0=f0,
                player_by_iso=player_by_iso,
                player_weight=player_weight,
            )
        if rating is None:
            unmapped.add(team)
        else:
            mapped.add(team)
            raw_ratings[team] = rating

    center = float(np.mean(list(raw_ratings.values()))) if raw_ratings else 1500.0
    strength: dict[str, float] = {}
    attack: dict[str, float] = {}
    defense: dict[str, float] = {}
    sd: dict[str, float] = {}
    for team in teams:
        if team in raw_ratings:
            s = (raw_ratings[team] - center) / 400.0
            strength[team] = s
            attack[team] = 0.5 * s
            defense[team] = 0.5 * s
            sd[team] = config.prior_sd
        else:
            strength[team] = 0.0
            attack[team] = 0.0
            defense[team] = 0.0
            sd[team] = config.unmapped_prior_sd
    return PriorMaps(strength, attack, defense, sd, mapped, unmapped)


@dataclass(frozen=True)
class DixonColesAttackDefenseModel:
    teams: list[str]
    attacks: dict[str, float]
    defenses: dict[str, float]
    log_mu: float
    home_adv: float
    rho: float
    prior_attacks: dict[str, float] | None = None
    prior_defenses: dict[str, float] | None = None

    def _attack(self, team: str) -> float:
        if team in self.attacks:
            return self.attacks[team]
        return (self.prior_attacks or {}).get(team, 0.0)

    def _defense(self, team: str) -> float:
        if team in self.defenses:
            return self.defenses[team]
        return (self.prior_defenses or {}).get(team, 0.0)

    def expected_goals(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral: bool,
        a_is_host: bool = False,
        b_is_host: bool = False,
    ) -> tuple[float, float]:
        home_bonus = 0.0 if neutral else self.home_adv
        if a_is_host:
            home_bonus += self.home_adv
        away_bonus = self.home_adv if b_is_host else 0.0
        lam_home = math.exp(self.log_mu + self._attack(home_team) - self._defense(away_team) + home_bonus)
        lam_away = math.exp(self.log_mu + self._attack(away_team) - self._defense(home_team) + away_bonus)
        return lam_home, lam_away

    def predict(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral: bool,
        a_is_host: bool = False,
        b_is_host: bool = False,
    ) -> tuple[float, float, float]:
        lam_home, lam_away = self.expected_goals(
            home_team, away_team, neutral=neutral, a_is_host=a_is_host, b_is_host=b_is_host,
        )
        return outcome_probs_from_lambdas(lam_home, lam_away, self.rho)


def _log_poisson(goals: int, lmbda: float) -> float:
    return goals * math.log(max(lmbda, EPS)) - lmbda - math.lgamma(goals + 1)


def _initial_log_mu(records: list[MatchRecord]) -> float:
    if not records:
        return math.log(1.35)
    total_goals = sum(r.home_goals + r.away_goals for r in records)
    return math.log(max(total_goals / (2.0 * len(records)), 0.2))


def fit_model(
    records: list[MatchRecord],
    priors: PriorMaps,
    *,
    config: FitConfig,
    cutoff_date: date,
) -> DixonColesAttackDefenseModel:
    """Fit a penalized Dixon-Coles attack/defence model."""
    if not records:
        teams = sorted(priors.attack)
        return DixonColesAttackDefenseModel(
            teams=teams,
            attacks=dict(priors.attack),
            defenses=dict(priors.defense),
            log_mu=math.log(1.35),
            home_adv=0.0,
            rho=0.0,
            prior_attacks=dict(priors.attack),
            prior_defenses=dict(priors.defense),
        )

    teams = sorted({r.home_team for r in records} | {r.away_team for r in records} | set(priors.attack))
    idx = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    weights = np.array([match_weight(r, cutoff_date, config) for r in records], dtype=float)

    x0 = np.zeros(2 * n + 3)
    for team, i in idx.items():
        x0[i] = priors.attack.get(team, 0.0)
        x0[n + i] = priors.defense.get(team, 0.0)
    x0[2 * n] = _initial_log_mu(records)
    x0[2 * n + 1] = 0.0
    x0[2 * n + 2] = 0.0

    bounds = [(-3.0, 3.0)] * (2 * n)
    bounds.extend([(math.log(0.2), math.log(4.0)), (-0.7, 0.7), config.rho_bounds])

    def objective_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        attacks = x[:n]
        defenses = x[n:2 * n]
        log_mu = float(x[2 * n])
        home_adv = float(x[2 * n + 1])
        rho = float(x[2 * n + 2])

        nll = 0.0
        grad = np.zeros_like(x)
        for rec, w in zip(records, weights):
            h = idx[rec.home_team]
            a = idx[rec.away_team]
            adv = 0.0 if rec.neutral else home_adv
            lam_h = math.exp(log_mu + attacks[h] - defenses[a] + adv)
            lam_a = math.exp(log_mu + attacks[a] - defenses[h])
            tau = _tau(lam_h, lam_a, rec.home_goals, rec.away_goals, rho)
            if tau <= 0.0:
                return 1e12, grad
            tau_eta_h = 0.0
            tau_eta_a = 0.0
            tau_rho = 0.0
            if rec.home_goals == 0 and rec.away_goals == 0:
                tau_eta_h = -lam_h * lam_a * rho / tau
                tau_eta_a = -lam_h * lam_a * rho / tau
                tau_rho = -lam_h * lam_a / tau
            elif rec.home_goals == 0 and rec.away_goals == 1:
                tau_eta_h = lam_h * rho / tau
                tau_rho = lam_h / tau
            elif rec.home_goals == 1 and rec.away_goals == 0:
                tau_eta_a = lam_a * rho / tau
                tau_rho = lam_a / tau
            elif rec.home_goals == 1 and rec.away_goals == 1:
                tau_rho = -1.0 / tau

            d_eta_h = rec.home_goals - lam_h + tau_eta_h
            d_eta_a = rec.away_goals - lam_a + tau_eta_a
            ll = (
                _log_poisson(rec.home_goals, lam_h)
                + _log_poisson(rec.away_goals, lam_a)
                + math.log(tau)
            )
            nll -= w * ll
            grad[h] -= w * d_eta_h
            grad[n + a] += w * d_eta_h
            grad[a] -= w * d_eta_a
            grad[n + h] += w * d_eta_a
            grad[2 * n] -= w * (d_eta_h + d_eta_a)
            if not rec.neutral:
                grad[2 * n + 1] -= w * d_eta_h
            grad[2 * n + 2] -= w * tau_rho

        penalty = 100.0 * float(np.mean(attacks) ** 2 + np.mean(defenses) ** 2)
        grad[:n] += 200.0 * float(np.mean(attacks)) / n
        grad[n:2 * n] += 200.0 * float(np.mean(defenses)) / n
        for team, i in idx.items():
            sd = priors.sd.get(team, config.unmapped_prior_sd)
            penalty += 0.5 * ((attacks[i] - priors.attack.get(team, 0.0)) / sd) ** 2
            penalty += 0.5 * ((defenses[i] - priors.defense.get(team, 0.0)) / sd) ** 2
            grad[i] += (attacks[i] - priors.attack.get(team, 0.0)) / (sd ** 2)
            grad[n + i] += (defenses[i] - priors.defense.get(team, 0.0)) / (sd ** 2)
        penalty += 0.5 * (home_adv / 0.30) ** 2
        penalty += 0.5 * (rho / 0.10) ** 2
        grad[2 * n + 1] += home_adv / (0.30 ** 2)
        grad[2 * n + 2] += rho / (0.10 ** 2)
        return nll + penalty, grad

    result = minimize(
        objective_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": config.maxiter},
    )
    x = result.x
    attacks = {team: float(x[idx[team]]) for team in teams}
    defenses = {team: float(x[n + idx[team]]) for team in teams}
    return DixonColesAttackDefenseModel(
        teams=teams,
        attacks=attacks,
        defenses=defenses,
        log_mu=float(x[2 * n]),
        home_adv=float(x[2 * n + 1]),
        rho=float(x[2 * n + 2]),
        prior_attacks=dict(priors.attack),
        prior_defenses=dict(priors.defense),
    )
