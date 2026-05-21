"""Monte Carlo simulation engine. Runs N tournaments in parallel using
ProcessPoolExecutor with counter-seeded RNG for deterministic output."""
from __future__ import annotations
import os
import random as _random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from .ratings.base import RatingSystem
from .report import wilson_ci
from .tournament import simulate_tournament
from .types import Params, SimulationResult, Team


def _simulate_one(args: tuple) -> tuple[dict[str, str], dict[str, int], dict[str, int]]:
    """Worker function (top-level for pickling). Returns (placements,
    goals_for per team, goals_against per team)."""
    teams, draw, hosts, rating, params, seed_i, group_venues, knockout_host = args
    result = simulate_tournament(
        teams=teams, draw=draw, hosts=hosts,
        rating=rating, params=params, seed=seed_i,
        group_venues=group_venues, knockout_host=knockout_host,
    )
    gf: dict[str, int] = dict.fromkeys(teams, 0)
    ga: dict[str, int] = dict.fromkeys(teams, 0)
    for m in result.matches:
        gf[m.home] += m.home_goals
        ga[m.home] += m.away_goals
        gf[m.away] += m.away_goals
        ga[m.away] += m.home_goals
    return result.placements, gf, ga


def _run_parallel(args_list: list[tuple], workers: int) -> list:
    """Execute simulations either serially or in parallel."""
    if workers == 1:
        return [_simulate_one(a) for a in args_list]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_simulate_one, args_list))


def _aggregate_results(
    all_results: list,
) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, int]]:
    """Aggregate placements and goal totals across all simulations."""
    stage_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_gf: dict[str, int] = defaultdict(int)
    total_ga: dict[str, int] = defaultdict(int)
    for placements, gf, ga in all_results:
        for iso3, stage in placements.items():
            stage_counts[iso3][stage] += 1
        for iso3, goals in gf.items():
            total_gf[iso3] += goals
        for iso3, goals in ga.items():
            total_ga[iso3] += goals
    return stage_counts, total_gf, total_ga


def _compute_probabilities(
    stage_counts: dict[str, dict[str, int]], n: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Compute cumulative probabilities + Wilson CIs from stage counts."""
    all_stages_seen: set[str] = set()
    for counts in stage_counts.values():
        all_stages_seen.update(counts.keys())
    stage_order = ["Champion", "Final", "SF", "QF", "R16", "R32", "GroupOut"]
    stages = [s for s in stage_order if s in all_stages_seen]
    output_stages = ["Win" if s == "Champion" else s for s in stages]

    probabilities: dict[str, dict[str, float]] = {}
    ci_lo: dict[str, dict[str, float]] = {}
    ci_hi: dict[str, dict[str, float]] = {}

    for iso3, counts in stage_counts.items():
        exclusive = {stage: counts.get(stage, 0) / n for stage in stages}
        probs: dict[str, float] = {}
        lo: dict[str, float] = {}
        hi: dict[str, float] = {}
        cumulative = 0.0
        for stage, out_name in zip(stages, output_stages):
            if stage == "GroupOut":
                p = exclusive[stage]
            else:
                cumulative += exclusive[stage]
                p = cumulative
            probs[out_name] = p
            ci = wilson_ci(p, n)
            lo[out_name] = ci[0]
            hi[out_name] = ci[1]
        probabilities[iso3] = probs
        ci_lo[iso3] = lo
        ci_hi[iso3] = hi
    return probabilities, ci_lo, ci_hi


def run_simulations(
    teams: dict[str, Team], draw: dict[str, list[str]], hosts: set[str],
    *, rating: RatingSystem, params: Params = Params(),
    n: int = 100_000, seed: int | None = None, workers: int | None = None,
    group_venues: dict[str, str] | None = None,
    knockout_host: str | None = None,
) -> SimulationResult:
    """Run N tournaments deterministically. seed_i = base_seed + i per sim."""
    if seed is None:
        seed = _random.randint(0, 2**31)
    if workers is None:
        workers = os.cpu_count() or 1

    args_list = [
        (teams, draw, hosts, rating, params, seed + i, group_venues, knockout_host)
        for i in range(n)
    ]
    all_results = _run_parallel(args_list, workers)
    stage_counts, total_gf, total_ga = _aggregate_results(all_results)
    probabilities, ci_lo, ci_hi = _compute_probabilities(stage_counts, n)

    return SimulationResult(
        n=n, seed=seed,
        probabilities=probabilities,
        ci_lo=ci_lo, ci_hi=ci_hi,
        mean_goals_for={iso3: total_gf[iso3] / n for iso3 in teams},
        mean_goals_against={iso3: total_ga[iso3] / n for iso3 in teams},
    )
