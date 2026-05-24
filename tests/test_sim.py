"""Tests for Monte Carlo simulation engine."""


def test_run_simulations_returns_simulation_result(default_params):
    from wcsim.ratings.elo import EloRating
    from wcsim.sim import run_simulations
    from wcsim.types import Team
    teams = {f"T{i:02d}": Team(name=f"T{i:02d}", iso3=f"T{i:02d}",
                               confederation="UNK", elo=1500.0 + i * 10)
             for i in range(32)}
    draw = {chr(ord("A") + g): [f"T{g*4+j:02d}" for j in range(4)] for g in range(8)}
    result = run_simulations(
        teams=teams, draw=draw, hosts=set(),
        rating=EloRating(default_params), params=default_params,
        n=10, seed=42, workers=1,
    )
    assert result.n == 10
    assert result.seed == 42
    assert "T00" in result.probabilities
    assert any(v.get("Win", 0) > 0 for v in result.probabilities.values())


def test_run_simulations_deterministic_across_workers(default_params):
    from wcsim.ratings.elo import EloRating
    from wcsim.sim import run_simulations
    from wcsim.types import Team
    teams = {f"T{i:02d}": Team(name=f"T{i:02d}", iso3=f"T{i:02d}",
                               confederation="UNK", elo=1500.0 + i * 10)
             for i in range(32)}
    draw = {chr(ord("A") + g): [f"T{g*4+j:02d}" for j in range(4)] for g in range(8)}
    r1 = run_simulations(teams=teams, draw=draw, hosts=set(),
                         rating=EloRating(default_params), params=default_params,
                         n=20, seed=42, workers=1)
    r2 = run_simulations(teams=teams, draw=draw, hosts=set(),
                         rating=EloRating(default_params), params=default_params,
                         n=20, seed=42, workers=2)
    assert r1.probabilities == r2.probabilities


def test_run_simulations_probabilities_sum_to_one(default_params):
    from wcsim.ratings.elo import EloRating
    from wcsim.sim import run_simulations
    from wcsim.types import Team
    teams = {f"T{i:02d}": Team(name=f"T{i:02d}", iso3=f"T{i:02d}",
                               confederation="UNK", elo=1500.0 + i * 10)
             for i in range(32)}
    draw = {chr(ord("A") + g): [f"T{g*4+j:02d}" for j in range(4)] for g in range(8)}
    result = run_simulations(teams=teams, draw=draw, hosts=set(),
                             rating=EloRating(default_params), params=default_params,
                             n=50, seed=1, workers=1)
    win_sum = sum(v.get("Win", 0) for v in result.probabilities.values())
    assert abs(win_sum - 1.0) < 0.01
