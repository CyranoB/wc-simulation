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


def _simulate_one(args: tuple) -> dict[str, str]:
    """Worker function (top-level for pickling). Returns placements dict."""
    teams, draw, hosts, rating, params, seed_i = args
    result = simulate_tournament(
        teams=teams, draw=draw, hosts=hosts,
        rating=rating, params=params, seed=seed_i,
    )
    return result.placements


def run_simulations(
    teams: dict[str, Team], draw: dict[str, list[str]], hosts: set[str],
    *, rating: RatingSystem, params: Params = Params(),
    n: int = 100_000, seed: int | None = None, workers: int | None = None,
) -> SimulationResult:
    """Run N tournaments deterministically. seed_i = base_seed + i per sim."""
    if seed is None:
        seed = _random.randint(0, 2**31)
    if workers is None:
        workers = os.cpu_count() or 1

    args_list = [(teams, draw, hosts, rating, params, seed + i) for i in range(n)]

    if workers == 1:
        all_placements = [_simulate_one(a) for a in args_list]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            all_placements = list(executor.map(_simulate_one, args_list))

    # Aggregate.
    stage_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for placements in all_placements:
        for iso3, stage in placements.items():
            stage_counts[iso3][stage] += 1

    # Stage order from deepest to earliest exit. Output is CUMULATIVE
    # ("reached at least this stage") per PRD §5.4, not exclusive.
    all_stages_seen: set[str] = set()
    for counts in stage_counts.values():
        all_stages_seen.update(counts.keys())
    stage_order = ["Champion", "Final", "SF", "QF", "R16", "R32", "GroupOut"]
    stages = [s for s in stage_order if s in all_stages_seen]
    output_stages = ["Win" if s == "Champion" else s for s in stages]

    # Compute CUMULATIVE probabilities: P(reached at least stage X).
    # A team that won also reached Final, SF, QF, etc.
    # GroupOut stays exclusive (fraction that didn't advance past groups).
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

    return SimulationResult(
        n=n, seed=seed,
        probabilities=probabilities,
        ci_lo=ci_lo, ci_hi=ci_hi,
        mean_goals_for={iso3: 0.0 for iso3 in teams},
        mean_goals_against={iso3: 0.0 for iso3 in teams},
    )
