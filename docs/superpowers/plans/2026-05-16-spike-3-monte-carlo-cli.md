# Spike 3 — Monte Carlo Runner + CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a working `wcsim run` CLI command that simulates a tournament N times in parallel and outputs per-team probabilities with Wilson CIs. Plus `match`, `teams`, `version` commands.

**Architecture:** `ProcessPoolExecutor.map` runs N simulations (each seeded as `base_seed + i`) preserving index order for deterministic aggregation. Results collected into a `SimulationResult` dataclass, formatted by `report.py`, cached by `cache.py`, exposed via Typer CLI. Data loaded from bundled spike CSVs by `data.py`.

**Tech Stack:** Python ≥ 3.11, NumPy, pandas, typer ≥ 0.9, pyarrow ≥ 14, pytest. Run via `python -m wcsim.cli`.

---

## Task 1: Install new deps + add `__main__.py`

**Files:**
- Modify: `spikes/01-validation/requirements.txt`
- Create: `wcsim/__main__.py`

- [ ] **Step 1:** Append to `spikes/01-validation/requirements.txt`:
```
typer>=0.9
pyarrow>=14
```

- [ ] **Step 2:** Install:
```bash
spikes/01-validation/.venv/bin/pip install --quiet typer pyarrow
```

- [ ] **Step 3:** Create `wcsim/__main__.py`:
```python
"""Entry point for `python -m wcsim.cli`."""
from wcsim.cli import app

if __name__ == "__main__":
    app()
```

- [ ] **Step 4:** Verify typer installed:
```bash
spikes/01-validation/.venv/bin/python -c "import typer; print(typer.__version__)"
```

- [ ] **Step 5:** Commit:
```bash
git add spikes/01-validation/requirements.txt wcsim/__main__.py
git commit -m "Spike 3 Task 1: add typer + pyarrow deps, __main__.py entry"
```

---

## Task 2: `wcsim/data.py` — teams + draw loaders

**Files:**
- Create: `wcsim/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1:** Write `tests/test_data.py`:
```python
"""Tests for data loaders."""
from pathlib import Path
import pytest

SPIKE_DATA = Path(__file__).parent.parent / "spikes" / "01-validation" / "data" / "raw"


def test_load_teams_returns_dict_of_teams():
    from wcsim.data import load_teams
    teams = load_teams(SPIKE_DATA / "elo_history.csv", snapshot_date="2026-06-10")
    assert len(teams) == 48
    assert "ARG" in teams
    assert teams["ARG"].elo > 2000


def test_load_draw_returns_12_groups():
    from wcsim.data import load_draw
    draw = load_draw(SPIKE_DATA / "wc2026_draw.json")
    assert len(draw) == 12
    assert all(len(v) == 4 for v in draw.values())
    assert "ARG" in draw["J"]


def test_load_teams_raises_on_missing_file():
    from wcsim.data import load_teams
    with pytest.raises(FileNotFoundError):
        load_teams(Path("/nonexistent.csv"), snapshot_date="2026-06-10")
```

- [ ] **Step 2:** Run to fail.

- [ ] **Step 3:** Implement `wcsim/data.py`:
```python
"""Data loaders for the CLI. Reads bundled CSVs/JSON into wcsim types."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .types import Team

SPIKE_DATA = Path(__file__).parent.parent / "spikes" / "01-validation" / "data" / "raw"
DEFAULT_TEAMS_PATH = SPIKE_DATA / "elo_history.csv"
DEFAULT_DRAW_PATH = SPIKE_DATA / "wc2026_draw.json"

# Import the name mapping from the spike for ISO3 resolution.
import sys
_spike_dir = str(Path(__file__).parent.parent / "spikes" / "01-validation")
if _spike_dir not in sys.path:
    sys.path.insert(0, _spike_dir)
from name_to_iso3 import to_iso3


def load_teams(csv_path: Path, snapshot_date: str = "2026-06-10") -> dict[str, Team]:
    """Load teams from an elo_history.csv, filtering to a specific snapshot date."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Teams file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    target = pd.to_datetime(snapshot_date)
    df = df[df["date"] <= target].sort_values("date")
    latest = df.groupby("team").tail(1)
    teams: dict[str, Team] = {}
    for _, row in latest.iterrows():
        try:
            iso3 = to_iso3(row["team"])
        except KeyError:
            continue
        teams[iso3] = Team(
            name=row["team"], iso3=iso3, confederation="UNK",
            elo=float(row["rating"]),
        )
    return teams


def load_draw(json_path: Path) -> dict[str, list[str]]:
    """Load a draw JSON (group letter -> list of ISO3 codes)."""
    if not json_path.exists():
        raise FileNotFoundError(f"Draw file not found: {json_path}")
    with json_path.open() as f:
        return json.load(f)
```

- [ ] **Step 4:** Run to pass: `spikes/01-validation/.venv/bin/python -m pytest tests/test_data.py -v`

- [ ] **Step 5:** Commit:
```bash
git add wcsim/data.py tests/test_data.py
git commit -m "Spike 3 Task 2: data.py loaders (teams CSV + draw JSON)"
```

---

## Task 3: `wcsim/report.py` — Wilson CI + formatters

**Files:**
- Create: `wcsim/report.py`
- Create: `tests/test_report.py`

- [ ] **Step 1:** Write `tests/test_report.py`:
```python
"""Tests for report formatters and Wilson CI."""
import math


def test_wilson_ci_at_p_half_n_100():
    from wcsim.report import wilson_ci
    lo, hi = wilson_ci(0.5, 100)
    assert 0.40 < lo < 0.42
    assert 0.58 < hi < 0.60


def test_wilson_ci_at_p_zero():
    from wcsim.report import wilson_ci
    lo, hi = wilson_ci(0.0, 1000)
    assert lo == 0.0
    assert hi > 0.0


def test_wilson_ci_at_p_one():
    from wcsim.report import wilson_ci
    lo, hi = wilson_ci(1.0, 1000)
    assert lo < 1.0
    assert hi == 1.0


def test_format_csv_produces_header_and_rows():
    from wcsim.report import format_csv
    from wcsim.types import SimulationResult
    result = SimulationResult(
        n=100, seed=42,
        probabilities={"ARG": {"Win": 0.28, "GroupOut": 0.05}},
        ci_lo={"ARG": {"Win": 0.20, "GroupOut": 0.02}},
        ci_hi={"ARG": {"Win": 0.36, "GroupOut": 0.10}},
        mean_goals_for={"ARG": 2.1},
        mean_goals_against={"ARG": 0.8},
    )
    csv_str = format_csv(result)
    assert "team" in csv_str.split("\n")[0]
    assert "ARG" in csv_str


def test_format_table_returns_string():
    from wcsim.report import format_table
    from wcsim.types import SimulationResult
    result = SimulationResult(
        n=100, seed=42,
        probabilities={"ARG": {"Win": 0.28, "GroupOut": 0.05}},
        ci_lo={"ARG": {"Win": 0.20, "GroupOut": 0.02}},
        ci_hi={"ARG": {"Win": 0.36, "GroupOut": 0.10}},
        mean_goals_for={"ARG": 2.1},
        mean_goals_against={"ARG": 0.8},
    )
    table = format_table(result)
    assert "ARG" in table
    assert "28" in table  # 0.28 displayed as percentage
```

- [ ] **Step 2:** Run to fail.

- [ ] **Step 3:** Add `SimulationResult` to `wcsim/types.py`:
```python
@dataclass(frozen=True)
class SimulationResult:
    n: int
    seed: int
    probabilities: dict[str, dict[str, float]]
    ci_lo: dict[str, dict[str, float]]
    ci_hi: dict[str, dict[str, float]]
    mean_goals_for: dict[str, float]
    mean_goals_against: dict[str, float]
```

- [ ] **Step 4:** Implement `wcsim/report.py`:
```python
"""Report formatters: table, CSV, JSON. Plus Wilson CI computation."""
from __future__ import annotations
import csv
import io
import json
import math
from .types import SimulationResult


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for proportion p at sample size n."""
    if n == 0:
        return (0.0, 1.0)
    if p == 0.0:
        return (0.0, 1.0 - (1.0 - 0.95) ** (1.0 / n))
    if p == 1.0:
        return ((1.0 - 0.95) ** (1.0 / n), 1.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def format_table(result: SimulationResult, verbose: bool = False) -> str:
    """Plain-text table for stdout."""
    if not result.probabilities:
        return "(no results)"
    stages = list(next(iter(result.probabilities.values())).keys())
    header = f"{'Team':<6}" + "".join(f"{s:>10}" for s in stages)
    lines = [header, "-" * len(header)]
    sorted_teams = sorted(
        result.probabilities.items(),
        key=lambda kv: -kv[1].get("Win", 0),
    )
    for iso3, probs in sorted_teams:
        row = f"{iso3:<6}"
        for stage in stages:
            pct = probs.get(stage, 0) * 100
            row += f"{pct:>9.1f}%"
        lines.append(row)
    return "\n".join(lines)


def format_csv(result: SimulationResult, include_ci: bool = True) -> str:
    """CSV with one row per team."""
    if not result.probabilities:
        return ""
    stages = list(next(iter(result.probabilities.values())).keys())
    buf = io.StringIO()
    fieldnames = ["team"]
    for s in stages:
        fieldnames.append(s)
        if include_ci:
            fieldnames.extend([f"{s}_ci_lo", f"{s}_ci_hi"])
    fieldnames.extend(["mean_goals_for", "mean_goals_against", "simulations", "seed"])
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for iso3 in sorted(result.probabilities.keys()):
        row: dict = {"team": iso3}
        for s in stages:
            row[s] = f"{result.probabilities[iso3].get(s, 0):.6f}"
            if include_ci:
                row[f"{s}_ci_lo"] = f"{result.ci_lo[iso3].get(s, 0):.6f}"
                row[f"{s}_ci_hi"] = f"{result.ci_hi[iso3].get(s, 0):.6f}"
        row["mean_goals_for"] = f"{result.mean_goals_for.get(iso3, 0):.4f}"
        row["mean_goals_against"] = f"{result.mean_goals_against.get(iso3, 0):.4f}"
        row["simulations"] = result.n
        row["seed"] = result.seed
        w.writerow(row)
    return buf.getvalue()


def format_json(result: SimulationResult, meta_det: dict, meta_env: dict) -> str:
    """JSON with meta blocks + rows."""
    rows = []
    stages = list(next(iter(result.probabilities.values())).keys()) if result.probabilities else []
    for iso3 in sorted(result.probabilities.keys()):
        row = {"team": iso3}
        for s in stages:
            row[s] = result.probabilities[iso3].get(s, 0)
            row[f"{s}_ci_lo"] = result.ci_lo[iso3].get(s, 0)
            row[f"{s}_ci_hi"] = result.ci_hi[iso3].get(s, 0)
        row["mean_goals_for"] = result.mean_goals_for.get(iso3, 0)
        row["mean_goals_against"] = result.mean_goals_against.get(iso3, 0)
        rows.append(row)
    return json.dumps({
        "meta": {"deterministic": meta_det, "environment": meta_env},
        "rows": rows,
    }, indent=2)
```

- [ ] **Step 5:** Run to pass.

- [ ] **Step 6:** Commit:
```bash
git add wcsim/types.py wcsim/report.py tests/test_report.py
git commit -m "Spike 3 Task 3: report.py (Wilson CI + table/CSV/JSON formatters)"
```

---

## Task 4: `wcsim/sim.py` — Monte Carlo engine

**Files:**
- Create: `wcsim/sim.py`
- Create: `tests/test_sim.py`

- [ ] **Step 1:** Write `tests/test_sim.py`:
```python
"""Tests for Monte Carlo simulation engine."""
import numpy as np
import pytest


def test_run_simulations_returns_simulation_result(default_params):
    from wcsim.sim import run_simulations
    from wcsim.ratings.elo import EloRating
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
    # Win probability for at least one team should be > 0
    assert any(v.get("Win", 0) > 0 for v in result.probabilities.values())


def test_run_simulations_deterministic_across_workers(default_params):
    from wcsim.sim import run_simulations
    from wcsim.ratings.elo import EloRating
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
    from wcsim.sim import run_simulations
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Team
    teams = {f"T{i:02d}": Team(name=f"T{i:02d}", iso3=f"T{i:02d}",
                               confederation="UNK", elo=1500.0 + i * 10)
             for i in range(32)}
    draw = {chr(ord("A") + g): [f"T{g*4+j:02d}" for j in range(4)] for g in range(8)}
    result = run_simulations(teams=teams, draw=draw, hosts=set(),
                             rating=EloRating(default_params), params=default_params,
                             n=50, seed=1, workers=1)
    # Win% across all teams should sum to ~1.0
    win_sum = sum(v.get("Win", 0) for v in result.probabilities.values())
    assert abs(win_sum - 1.0) < 0.01  # within 1% (small n, so some noise)
```

- [ ] **Step 2:** Run to fail.

- [ ] **Step 3:** Implement `wcsim/sim.py`:
```python
"""Monte Carlo simulation engine. Runs N tournaments in parallel using
ProcessPoolExecutor with counter-seeded RNG for deterministic output."""
from __future__ import annotations
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

from .ratings.base import RatingSystem
from .report import wilson_ci
from .tournament import simulate_tournament
from .types import Params, SimulationResult, Team


def _simulate_one(args: tuple) -> dict[str, str]:
    """Worker function. Must be top-level for pickling."""
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
    """Run N tournaments. Each sim i uses seed = base_seed + i.
    Deterministic: same (teams, draw, hosts, rating, params, seed) ->
    byte-identical SimulationResult regardless of workers."""
    import random as _random
    if seed is None:
        seed = _random.randint(0, 2**31)
    if workers is None:
        workers = os.cpu_count() or 1

    # Build args for each simulation.
    args_list = [
        (teams, draw, hosts, rating, params, seed + i)
        for i in range(n)
    ]

    # Run in parallel (or serial if workers=1).
    if workers == 1:
        all_placements = [_simulate_one(a) for a in args_list]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            all_placements = list(executor.map(_simulate_one, args_list))

    # Aggregate placements into probabilities.
    stage_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    goals_for: dict[str, float] = defaultdict(float)
    goals_against: dict[str, float] = defaultdict(float)

    for placements in all_placements:
        for iso3, stage in placements.items():
            stage_counts[iso3][stage] += 1

    # Compute probabilities and Wilson CIs.
    probabilities: dict[str, dict[str, float]] = {}
    ci_lo: dict[str, dict[str, float]] = {}
    ci_hi: dict[str, dict[str, float]] = {}

    # Determine the stage ordering from the first result.
    all_stages_seen: set[str] = set()
    for counts in stage_counts.values():
        all_stages_seen.update(counts.keys())
    # Order: Champion -> Final -> SF -> ... -> GroupOut
    stage_order = ["Champion", "Final", "SF", "QF", "R16", "R32", "GroupOut"]
    stages = [s for s in stage_order if s in all_stages_seen]
    # Rename "Champion" to "Win" for the output.
    output_stages = ["Win" if s == "Champion" else s for s in stages]

    for iso3, counts in stage_counts.items():
        probs: dict[str, float] = {}
        lo: dict[str, float] = {}
        hi: dict[str, float] = {}
        for stage, out_name in zip(stages, output_stages):
            p = counts.get(stage, 0) / n
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
        mean_goals_for={iso3: 0.0 for iso3 in teams},  # TODO: track in future
        mean_goals_against={iso3: 0.0 for iso3 in teams},
    )
```

- [ ] **Step 4:** Run to pass: `spikes/01-validation/.venv/bin/python -m pytest tests/test_sim.py -v`

- [ ] **Step 5:** Commit:
```bash
git add wcsim/sim.py tests/test_sim.py
git commit -m "Spike 3 Task 4: sim.py Monte Carlo engine (ProcessPoolExecutor)"
```

---

## Task 5: `wcsim/cache.py` — last-run persistence

**Files:**
- Create: `wcsim/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1:** Write `tests/test_cache.py`:
```python
"""Tests for cache.py."""
from pathlib import Path
import tempfile


def test_write_and_read_cache():
    from wcsim.cache import write_cache, read_cache_meta
    from wcsim.types import SimulationResult
    result = SimulationResult(
        n=10, seed=42,
        probabilities={"ARG": {"Win": 0.3}},
        ci_lo={"ARG": {"Win": 0.2}},
        ci_hi={"ARG": {"Win": 0.4}},
        mean_goals_for={"ARG": 2.0},
        mean_goals_against={"ARG": 0.8},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        write_cache(result, {"seed": 42}, {"runtime": 1.0}, cache_dir=cache_dir)
        meta = read_cache_meta(cache_dir=cache_dir)
        assert meta is not None
        assert meta["deterministic"]["seed"] == 42
```

- [ ] **Step 2:** Run to fail.

- [ ] **Step 3:** Implement `wcsim/cache.py`:
```python
"""Cache: persists last-run results for wcsim bracket (future) and re-use."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .types import SimulationResult

DEFAULT_CACHE_DIR = Path.home() / ".wcsim"


def write_cache(
    result: SimulationResult,
    meta_det: dict, meta_env: dict,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> None:
    """Write last_run.parquet + last_run.meta.json."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Parquet: probabilities as a DataFrame.
    rows = []
    for iso3, probs in result.probabilities.items():
        row = {"team": iso3, **probs}
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(cache_dir / "last_run.parquet", index=False)
    # Meta JSON.
    meta = {"deterministic": meta_det, "environment": meta_env}
    with (cache_dir / "last_run.meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


def read_cache_meta(cache_dir: Path = DEFAULT_CACHE_DIR) -> dict | None:
    """Read last_run.meta.json; returns None if not found."""
    meta_path = cache_dir / "last_run.meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open() as f:
        return json.load(f)
```

- [ ] **Step 4:** Run to pass.

- [ ] **Step 5:** Commit:
```bash
git add wcsim/cache.py tests/test_cache.py
git commit -m "Spike 3 Task 5: cache.py (parquet + meta.json)"
```

---

## Task 6: `wcsim/cli.py` — run command

**Files:**
- Create: `wcsim/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1:** Write `tests/test_cli.py`:
```python
"""Tests for the CLI (subprocess-level integration)."""
import subprocess
import sys
from pathlib import Path

PYTHON = str(Path(sys.executable))  # Use the venv python


def test_cli_run_produces_output():
    result = subprocess.run(
        [PYTHON, "-m", "wcsim.cli", "run", "-n", "10", "--seed", "42", "--workers", "1"],
        capture_output=True, text=True, timeout=30,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Win" in result.stdout or "%" in result.stdout


def test_cli_match_prints_probabilities():
    result = subprocess.run(
        [PYTHON, "-m", "wcsim.cli", "match", "Brazil", "France", "--neutral"],
        capture_output=True, text=True, timeout=10,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "draw" in result.stdout.lower() or "Draw" in result.stdout


def test_cli_version():
    result = subprocess.run(
        [PYTHON, "-m", "wcsim.cli", "version"],
        capture_output=True, text=True, timeout=5,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0
    assert "wcsim" in result.stdout.lower() or "0." in result.stdout
```

- [ ] **Step 2:** Run to fail.

- [ ] **Step 3:** Implement `wcsim/cli.py`:
```python
"""CLI entry point using Typer. Run via `python -m wcsim.cli`."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional
import typer

app = typer.Typer(name="wcsim", help="Football Tournament Monte Carlo Simulator")


@app.command()
def run(
    n: int = typer.Option(100_000, "-n", "--simulations", help="Number of simulations"),
    seed: Optional[int] = typer.Option(None, "--seed", help="RNG seed"),
    teams_path: Optional[Path] = typer.Option(None, "--teams", help="Teams CSV path"),
    draw_path: Optional[Path] = typer.Option(None, "--draw", help="Draw JSON path"),
    rating_mode: str = typer.Option("elo", "--rating", help="Rating: elo, fifa, blend"),
    out: Optional[Path] = typer.Option(None, "--out", help="Output file (CSV/JSON)"),
    format: str = typer.Option("table", "--format", help="Output format: table, csv, json"),
    workers: Optional[int] = typer.Option(None, "--workers", help="Parallel workers"),
    ci: bool = typer.Option(True, "--ci/--no-ci", help="Include confidence intervals"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
):
    """Run Monte Carlo tournament simulation."""
    from .data import load_teams, load_draw, DEFAULT_TEAMS_PATH, DEFAULT_DRAW_PATH
    from .ratings.elo import EloRating
    from .ratings.fifa import FifaRating
    from .ratings.blend import BlendRating
    from .sim import run_simulations
    from .report import format_table, format_csv, format_json
    from .cache import write_cache
    from .types import Params
    import random

    tp = teams_path or DEFAULT_TEAMS_PATH
    dp = draw_path or DEFAULT_DRAW_PATH
    teams = load_teams(tp)
    draw = load_draw(dp)
    hosts = {"USA", "MEX", "CAN"}  # hardcoded for WC 2026 (--hosts deferred)

    params = Params()
    rating_cls = {"elo": EloRating, "fifa": FifaRating, "blend": BlendRating}[rating_mode]
    rating = rating_cls(params)

    actual_seed = seed if seed is not None else random.randint(0, 2**31)
    if not quiet:
        typer.echo(f"Running {n} simulations (seed={actual_seed}, workers={workers or 'auto'})...")

    start = time.time()
    result = run_simulations(
        teams=teams, draw=draw, hosts=hosts,
        rating=rating, params=params,
        n=n, seed=actual_seed, workers=workers,
    )
    elapsed = time.time() - start
    if not quiet:
        typer.echo(f"Done in {elapsed:.1f}s.")

    # Output.
    if format == "csv" or (out and str(out).endswith(".csv")):
        output = format_csv(result, include_ci=ci)
    elif format == "json" or (out and str(out).endswith(".json")):
        meta_det = {"seed": actual_seed, "simulations": n, "rating_mode": rating_mode}
        meta_env = {"runtime_seconds": elapsed}
        output = format_json(result, meta_det, meta_env)
    else:
        output = format_table(result, verbose=verbose)

    if out:
        out.write_text(output)
        if not quiet:
            typer.echo(f"Written to {out}")
    else:
        typer.echo(output)

    # Cache.
    meta_det = {"seed": actual_seed, "simulations": n, "rating_mode": rating_mode}
    meta_env = {"runtime_seconds": elapsed}
    write_cache(result, meta_det, meta_env)


@app.command()
def match(
    team_a: str = typer.Argument(..., help="Team A name"),
    team_b: str = typer.Argument(..., help="Team B name"),
    neutral: bool = typer.Option(False, "--neutral"),
    home: Optional[str] = typer.Option(None, "--home", help="Which team is home: A, B, or none"),
    rating_mode: str = typer.Option("elo", "--rating"),
):
    """Print win/draw/loss probabilities for a single match."""
    from .data import load_teams, DEFAULT_TEAMS_PATH
    from .model import predict_match
    from .ratings.elo import EloRating
    from .ratings.fifa import FifaRating
    from .ratings.blend import BlendRating
    from .types import Params
    from name_to_iso3 import to_iso3

    teams = load_teams(DEFAULT_TEAMS_PATH)
    params = Params()
    rating_cls = {"elo": EloRating, "fifa": FifaRating, "blend": BlendRating}[rating_mode]
    rating = rating_cls(params)

    iso_a = to_iso3(team_a)
    iso_b = to_iso3(team_b)
    t_a, t_b = teams[iso_a], teams[iso_b]

    a_is_host = (home == "A") if home else False
    b_is_host = (home == "B") if home else False
    if neutral:
        a_is_host = b_is_host = False

    p = predict_match(t_a, t_b, rating=rating, params=params,
                      a_is_host=a_is_host, b_is_host=b_is_host)

    typer.echo(f"{team_a} vs {team_b} ({rating_mode} mode, "
               f"{'neutral' if neutral else 'home=' + (home or 'A')}):")
    typer.echo(f"  {team_a} wins: {p[0]*100:.1f}%")
    typer.echo(f"  Draw:         {p[1]*100:.1f}%")
    typer.echo(f"  {team_b} wins: {p[2]*100:.1f}%")


@app.command()
def teams(
    teams_path: Optional[Path] = typer.Option(None, "--teams"),
):
    """List loaded teams with their Elo rating."""
    from .data import load_teams, DEFAULT_TEAMS_PATH
    all_teams = load_teams(teams_path or DEFAULT_TEAMS_PATH)
    typer.echo(f"{'ISO3':<6}{'Name':<25}{'Elo':>8}")
    typer.echo("-" * 39)
    for iso3, t in sorted(all_teams.items(), key=lambda kv: -kv[1].elo):
        typer.echo(f"{iso3:<6}{t.name:<25}{t.elo:>8.1f}")


@app.command()
def version():
    """Print version."""
    typer.echo("wcsim 0.3.0-dev (Spike 3)")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4:** Run to pass: `spikes/01-validation/.venv/bin/python -m pytest tests/test_cli.py -v`

- [ ] **Step 5:** Update `wcsim/__init__.py` to also export `SimulationResult` and `run_simulations`:
```python
from .types import SimulationResult
from .sim import run_simulations
```

- [ ] **Step 6:** Commit:
```bash
git add wcsim/cli.py wcsim/__main__.py wcsim/__init__.py tests/test_cli.py
git commit -m "Spike 3 Task 6: cli.py (run + match + teams + version)"
```

---

## Task 7: Determinism integration test

**Files:**
- Modify: `tests/test_sim.py`

- [ ] **Step 1:** Append a full-stack determinism test:
```python
def test_full_determinism_csv_output(default_params):
    """Same inputs → byte-identical CSV regardless of workers."""
    from wcsim.sim import run_simulations
    from wcsim.report import format_csv
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Team
    teams = {f"T{i:02d}": Team(name=f"T{i:02d}", iso3=f"T{i:02d}",
                               confederation="UNK", elo=1500.0 + i * 10)
             for i in range(32)}
    draw = {chr(ord("A") + g): [f"T{g*4+j:02d}" for j in range(4)] for g in range(8)}

    def csv_for(w):
        r = run_simulations(teams=teams, draw=draw, hosts=set(),
                            rating=EloRating(default_params), params=default_params,
                            n=20, seed=99, workers=w)
        return format_csv(r)

    assert csv_for(1) == csv_for(2) == csv_for(4)
```

- [ ] **Step 2:** Run: `spikes/01-validation/.venv/bin/python -m pytest tests/test_sim.py -v`

- [ ] **Step 3:** Commit:
```bash
git add tests/test_sim.py
git commit -m "Spike 3 Task 7: determinism integration test (workers 1/2/4)"
```

---

## Task 8: Performance smoke test

**Files:**
- Modify: `tests/test_sim.py`

- [ ] **Step 1:** Append:
```python
import time

def test_performance_1000_sims_under_10s(default_params):
    """1000 sims single-core should complete in <10s (extrapolates to ~100s
    for 100k; with 8 cores -> ~12s, well under 60s target)."""
    from wcsim.sim import run_simulations
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Team
    teams = {f"T{i:02d}": Team(name=f"T{i:02d}", iso3=f"T{i:02d}",
                               confederation="UNK", elo=1500.0 + i * 10)
             for i in range(32)}
    draw = {chr(ord("A") + g): [f"T{g*4+j:02d}" for j in range(4)] for g in range(8)}
    start = time.time()
    run_simulations(teams=teams, draw=draw, hosts=set(),
                    rating=EloRating(default_params), params=default_params,
                    n=1000, seed=1, workers=1)
    elapsed = time.time() - start
    assert elapsed < 10.0, f"1000 sims took {elapsed:.1f}s (limit 10s)"
```

- [ ] **Step 2:** Run: `spikes/01-validation/.venv/bin/python -m pytest tests/test_sim.py::test_performance_1000_sims_under_10s -v`

- [ ] **Step 3:** Commit:
```bash
git add tests/test_sim.py
git commit -m "Spike 3 Task 8: performance smoke (1000 sims < 10s single-core)"
```

---

## Task 9: Coverage + final cleanup

**Files:**
- Modify: `wcsim/__init__.py` (ensure all re-exports)

- [ ] **Step 1:** Run full suite with coverage:
```bash
spikes/01-validation/.venv/bin/python -m pytest tests/ --cov=wcsim --cov-report=term-missing -q
```

- [ ] **Step 2:** Fix any uncovered lines with targeted tests if needed.

- [ ] **Step 3:** Final commit:
```bash
git add -A && git commit -m "Spike 3 Task 9: coverage + final cleanup"
```

---

## Self-Review

- **Spec §4 SimulationResult** → Task 3 (added to types.py)
- **Spec §5 sim.py** → Task 4
- **Spec §6 data.py** → Task 2
- **Spec §7 report.py** → Task 3
- **Spec §8 cache.py** → Task 5
- **Spec §9 cli.py** → Task 6
- **Spec §10 determinism** → Tasks 4 + 7
- **Spec §10 performance** → Task 8
- **Spec §11 ACs 1-8** → covered across Tasks 2-8
- **Type consistency:** `SimulationResult` used in sim.py, report.py, cache.py, cli.py — same shape throughout. `run_simulations` signature consistent between test and implementation.
