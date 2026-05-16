# Spike 3: Monte Carlo Runner + Report + CLI

**Status:** Approved by brainstorm 2026-05-16
**Parent:** PRD `wcsim` v1.7 §5.1–§5.4, §6, §7
**Prereqs:** Spike 2 (library with types, ratings, model, tournament)

## 1. Goal

Ship a usable `wcsim run` command that simulates a tournament 100,000 times in parallel and produces a probability table showing each team's chances of reaching each round. Plus `wcsim match`, `wcsim teams`, and `wcsim version` commands. Deterministic output regardless of worker count.

**Binary success:**

1. `wcsim run -n 1000 --seed 42 --out results.csv` finishes in <5s and produces a CSV with 48 rows × probability columns + Wilson CIs.
2. `wcsim run --seed 1 --out r.csv` produces byte-identical CSV across `--workers 1` and `--workers 4`.
3. `wcsim run -n 100000 --seed 1` finishes in ≤60s on an 8-core machine (or ≤5 minutes single-core).
4. `wcsim match Brazil France --neutral --rating elo` prints win/draw/loss probabilities matching `wcsim.predict_match` output.
5. `wcsim teams` lists all bundled teams with Elo + group assignment.
6. `wcsim version` prints the version string.

## 2. Non-Goals

- `wcsim bracket` (needs cache read logic + greedy walk renderer — deferred)
- `wcsim update-ratings` (scraper migration — deferred)
- `wcsim backtest` (Spike 1's validator, not production CLI yet — deferred)
- `pyproject.toml` / pip-install — still flat layout; user runs via `python -m wcsim.cli`
- Rich table rendering (defer `rich` dep; use plain formatted stdout for now)

## 3. New modules

```
wcsim/
├── data.py         # load_teams(csv_path) -> dict[str, Team]
│                   # load_draw(json_path) -> dict[str, list[str]]
├── sim.py          # run_simulations(...) -> SimulationResult
├── report.py       # format_table(), format_csv(), format_json(), wilson_ci()
├── cache.py        # write_cache(result, meta_det, meta_env)
└── cli.py          # Typer app: run, match, teams, version
```

Existing modules unchanged: `types.py`, `ratings/`, `model.py`, `tournament.py`, `__init__.py` (adds re-exports for new public symbols).

## 4. New type: `SimulationResult`

```python
@dataclass(frozen=True)
class SimulationResult:
    """Aggregated output of N tournament simulations."""
    n: int
    seed: int
    probabilities: dict[str, dict[str, float]]  # {iso3: {"Win": 0.28, "Final": 0.42, ...}}
    ci_lo: dict[str, dict[str, float]]          # Wilson 95% lower bound per cell
    ci_hi: dict[str, dict[str, float]]          # Wilson 95% upper bound per cell
    mean_goals_for: dict[str, float]
    mean_goals_against: dict[str, float]
```

Stages in the probability dict depend on the tournament format:
- WC 2018/2022 (32 teams): `["Win", "Final", "SF", "QF", "R16", "GroupOut"]`
- WC 2026 (48 teams): `["Win", "Final", "SF", "QF", "R16", "R32", "GroupOut"]`

## 5. `wcsim/sim.py` — Monte Carlo engine

```python
from concurrent.futures import ProcessPoolExecutor

def _simulate_one(args: tuple) -> dict[str, str]:
    """Worker function. Receives (teams, draw, hosts, rating_cls, params, seed_i).
    Returns placements dict {iso3: exit_stage}."""

def run_simulations(
    teams: dict[str, Team], draw: dict[str, list[str]], hosts: set[str],
    *, rating: RatingSystem, params: Params = Params(),
    n: int = 100_000, seed: int | None = None, workers: int | None = None,
) -> SimulationResult:
    """Run N tournaments. Each sim i uses seed = base_seed + i.
    Workers partition by index; results aggregated in index order.
    Returns SimulationResult with probabilities and Wilson CIs."""
```

**Determinism contract:**
- `base_seed` = user-provided seed, or a random seed that gets printed/recorded.
- Simulation $i$ gets `seed_i = base_seed + i`.
- Workers process sims in any order via `ProcessPoolExecutor.map(...)`.
- Aggregation is over the index-ordered results (since each sim produces a placement dict keyed by iso3, and we just count occurrences).
- Because `ProcessPoolExecutor.map` preserves input order, we don't even need to sort — the results come back in index order already.

**Performance:**
- Worker function must be picklable → top-level function (not a lambda/closure).
- Teams, draw, hosts, params are serialized once per worker (via initializer or repeated arg). For 100k sims with 48 teams, the bottleneck is the group-stage simulation (72 `sample_match` calls per sim × 100k = 7.2M calls). Each `sample_match` is ~50μs of NumPy → single-core estimate ~6 minutes. With 8 cores, ~45s. Should meet the ≤60s target.

## 6. `wcsim/data.py` — data loaders for CLI

```python
def load_teams(csv_path: Path) -> dict[str, Team]:
    """Parse the bundled elo_history.csv + name_to_iso3 into Team objects."""

def load_draw(json_path: Path) -> dict[str, list[str]]:
    """Parse the draw JSON into {group_letter: [iso3, iso3, iso3, iso3]}."""

DEFAULT_TEAMS_PATH = Path(__file__).parent.parent / "spikes" / "01-validation" / "data" / "raw" / "elo_history.csv"
DEFAULT_DRAW_PATH = Path(__file__).parent.parent / "spikes" / "01-validation" / "data" / "raw" / "wc2026_draw.json"
```

For Spike 3, the "bundled" data is the spike's bundled CSVs. Proper `wcsim/data/` bundled snapshots are a packaging concern (Spike 6).

## 7. `wcsim/report.py` — formatters

```python
def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for proportion p at sample size n."""

def format_table(result: SimulationResult, verbose: bool = False) -> str:
    """Plain-text table for stdout. Includes CIs if verbose."""

def format_csv(result: SimulationResult, include_ci: bool = True) -> str:
    """CSV with one row per team, columns for each stage probability + CIs."""

def format_json(result: SimulationResult, meta_det: dict, meta_env: dict) -> str:
    """JSON with meta.deterministic + meta.environment + rows array."""
```

## 8. `wcsim/cache.py` — last-run persistence

```python
CACHE_DIR = Path.home() / ".wcsim"

def write_cache(result: SimulationResult, meta_det: dict, meta_env: dict) -> None:
    """Write last_run.parquet + last_run.meta.json to ~/.wcsim/."""

def read_cache_meta() -> dict | None:
    """Read last_run.meta.json; returns None if cache doesn't exist."""
```

Uses pandas DataFrame → parquet for the probability table. The meta JSON is the same `{deterministic: {...}, environment: {...}}` structure from PRD §5.4.

## 9. `wcsim/cli.py` — Typer commands

```python
import typer
app = typer.Typer()

@app.command()
def run(
    n: int = 100_000,
    seed: int | None = None,
    teams: Path | None = None,
    draw: Path | None = None,
    rating: str = "elo",
    out: Path | None = None,
    format: str = "table",
    workers: int | None = None,
    ci: bool = True,
    verbose: bool = False,
    quiet: bool = False,
): ...

@app.command()
def match(team_a: str, team_b: str, neutral: bool = False, home: str | None = None, rating: str = "elo"): ...

@app.command()
def teams(teams_path: Path | None = None): ...

@app.command()
def version(): ...
```

Entry point: `python -m wcsim.cli` (no pip-install needed; flat layout means `python -m wcsim.cli run ...` from repo root).

## 10. Test strategy

- TDD per module (same pattern as Spike 2).
- **Determinism test**: `run_simulations(..., seed=42, workers=1)` == `run_simulations(..., seed=42, workers=4)` — byte-identical `SimulationResult`.
- **Performance smoke**: `run_simulations(..., n=1000, workers=1)` completes in <5s (extrapolates to ~8min for 100k single-core; multi-core brings it under 60s).
- **Wilson CI test**: for known (p, n), `wilson_ci(p, n)` matches `statsmodels.proportion_confint` to 1e-6.
- **CLI integration**: `subprocess.run(["python", "-m", "wcsim.cli", "run", "-n", "100", "--seed", "42"])` exits 0 and stdout contains probability numbers.

## 11. Acceptance Criteria

1. `python -m wcsim.cli run -n 1000 --seed 42 --out results.csv` produces a valid CSV with Wilson CIs.
2. Same command with `--workers 1` and `--workers 4` produces byte-identical CSV.
3. `python -m wcsim.cli run -n 100000 --seed 1` finishes ≤60s on 8-core (benchmarked on the dev machine and documented).
4. `python -m wcsim.cli match Brazil France --neutral` prints 3 probabilities summing to 1.
5. `python -m wcsim.cli teams` lists 48 teams with Elo ratings.
6. `python -m wcsim.cli version` prints a version string.
7. Cache files written to `~/.wcsim/last_run.parquet` + `last_run.meta.json` after every `run`.
8. ≥95% line coverage on the new modules; 100% on `wilson_ci`, `_simulate_one`.

## 12. Dependencies

Add to requirements: `typer>=0.9`, `pyarrow>=14` (for parquet). No `rich` yet (plain text table for now).

## 13. Out of scope

- `wcsim bracket` (render from cache — deferred)
- `wcsim update-ratings` / scrapers in CLI
- `wcsim backtest`
- Proper bundled data under `wcsim/data/` (still reads from spike's data/raw/)
- `pyproject.toml` / entry-point scripts
- `--dump-sims` NDJSON log
- `--hosts` flag (hardcoded to WC 2026 hosts for now)
