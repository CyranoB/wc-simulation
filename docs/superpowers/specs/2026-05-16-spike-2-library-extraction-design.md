# Spike 2: Library Extraction

**Status:** Approved by brainstorm 2026-05-16
**Parent:** PRD `wcsim` v1.7 §7 *Architecture*; Spike 1 (PR #1) provides the validated math.
**Work location:** repo root, new `wcsim/` and `tests/` directories on a feature branch.

## 1. Goal

Extract the validated match model from `spikes/01-validation/validate.py` into a clean Python library `wcsim/` per PRD §7's architecture, extend it with tournament logic (group stage + knockout bracket) supporting **both** WC 2018/2022 format (32 teams, 8 groups × 4) and WC 2026 format (48 teams, 12 groups × 4), and bundle WC 2026 data so the library can produce a forward-looking 2026 simulation. Build it test-first with pytest. Spike 2 unblocks the Monte Carlo runner (Spike 3+) and the eventual hosted-service API.

**Binary success:**

1. `tests/test_regression.py` passes — for every WC 2018+2022 match in the bundled snapshot, `wcsim.predict_match(...)` reproduces `validate.predict(...)` to within `1e-9` for Elo, FIFA, and Blend modes.
2. `wcsim.simulate_tournament(teams=..., draw=WC2022_draw, hosts={"QAT"}, rating=EloRating(), seed=42)` runs end-to-end and returns a `TournamentResult` with 64 matches, a single champion, and complete `placements` (all 32 participants get an exit-stage label).
3. `wcsim.simulate_tournament(teams=..., draw=WC2026_draw, hosts={"USA","MEX","CAN"}, rating=EloRating(), seed=42)` runs end-to-end and returns a `TournamentResult` with 103 matches (12 groups × 6 group matches + 16 R32 + 8 R16 + 4 QF + 2 SF + 1 Final = 103; no 3rd-place playoff in WC 2026), a single champion, and complete `placements` for all 48 participants.
4. `python -m pytest tests/ -v` reports ≥ 95% line coverage of `wcsim/`; 100% on the math hot path (`predict_match`, `_apply_tau`, `RatingSystem.lambdas`, `RatingSystem.update`).

## 2. Non-Goals

- No CLI, no Typer (Spike 5).
- No Monte Carlo runner / multi-process (Spike 3 or 4).
- No scrapers in the library — they stay under `spikes/01-validation/scrapers/`. Spike 2 extends them to fetch WC 2026 data (draw + current Elo/FIFA snapshots) and bundles the results under `spikes/01-validation/data/raw/`. Migration into `wcsim/scrapers/` is a later spike.
- No tournament-structure JSON schema — both supported formats (WC 2018/2022 and WC 2026) are hardcoded constants in `tournament.py`. Selection is by `len(teams)` dispatch (32 vs 48). Schema implementation is a later spike.
- No `pyproject.toml` / pip-installable packaging. Flat directory layout; `python -m pytest` works via `conftest.py` adding repo root to `sys.path`.
- No structural model changes beyond what's already in PRD v1.7 (Dixon-Coles is included).

## 3. Module layout

```
wcsim/                          # library, flat sibling of spikes/
├── __init__.py                 # re-exports public API
├── types.py                    # Team, MatchResult, TournamentResult, Params
├── ratings/
│   ├── __init__.py
│   ├── base.py                 # RatingSystem Protocol
│   ├── elo.py                  # EloRating
│   ├── fifa.py                 # FifaRating
│   └── blend.py                # BlendRating
├── model.py                    # predict_match, sample_match, Dixon-Coles
└── tournament.py               # simulate_group_stage, simulate_knockout, simulate_tournament

tests/
├── conftest.py                 # fixtures: sample Teams, default Params, bundled snapshot
├── test_types.py
├── test_ratings/
│   ├── test_base.py            # Protocol structural-typing check
│   ├── test_elo.py
│   ├── test_fifa.py
│   └── test_blend.py
├── test_model.py
├── test_tournament.py
└── test_regression.py          # cross-checks wcsim against validate.py to 1e-9
```

## 4. Public API

`wcsim/__init__.py` re-exports the public surface:

```python
from wcsim.types import Team, MatchResult, TournamentResult, Params
from wcsim.ratings.elo import EloRating
from wcsim.ratings.fifa import FifaRating
from wcsim.ratings.blend import BlendRating
from wcsim.model import predict_match, sample_match
from wcsim.tournament import simulate_tournament
```

Usage:

```python
import wcsim

elo = wcsim.EloRating(wcsim.Params())
# Single match (pure function, returns 3 probabilities summing to 1)
p_home, p_draw, p_away = wcsim.predict_match(team_a, team_b, rating=elo)

# Single tournament (deterministic given seed)
result = wcsim.simulate_tournament(
    teams=teams_by_iso3, draw=draw, hosts={"QAT"},
    rating=elo, params=wcsim.Params(), seed=42,
)
```

Internal modules (`wcsim.model._poisson_pmf`, etc.) remain importable but are not part of the public API.

## 5. Types

PRD §8-aligned. FIFA fields on `Team` are optional (Spike 1 finding: Elo-only inputs must be valid at the data-model level).

```python
from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True)
class Team:
    name: str            # display name
    iso3: str            # canonical 3-letter key
    confederation: str
    elo: float
    elo_updated: date | None = None
    fifa_points: float | None = None
    fifa_rank: int | None = None
    fifa_updated: date | None = None

@dataclass(frozen=True)
class Params:
    c_elo: float = 300.0
    c_fifa: float = 450.0
    mu: float = 1.35
    lambda_min: float = 0.05
    blend_w: float = 0.7
    e0: float = 1500.0
    f0: float = 1300.0
    home_bonus_elo: float = 100.0
    home_bonus_fifa: float = 150.0
    rho: float = 0.0     # Dixon-Coles correlation; 0 = independent Poisson

@dataclass(frozen=True)
class MatchResult:
    home: str            # iso3
    away: str            # iso3
    home_goals: int      # post-ET for knockouts; regulation otherwise
    away_goals: int
    stage: str           # "group" | "R32" | "R16" | "QF" | "SF" | "Final" | "3rd"
    neutral: bool
    extra_time: bool
    went_to_pens: bool
    pen_winner: str | None  # iso3 or None
    home_rating_before: float
    away_rating_before: float

@dataclass(frozen=True)
class TournamentResult:
    seed: int
    rating_mode: str        # "elo" | "fifa" | "blend"
    matches: list[MatchResult]   # chronological order
    placements: dict[str, str]   # iso3 -> "GroupOut" | "R32" | "R16" | "QF" | "SF" | "Final" | "Champion"
    final_ratings: dict[str, float]  # iso3 -> post-tournament rating in the rating mode's space
```

## 6. `RatingSystem` Protocol

```python
# wcsim/ratings/base.py
from typing import Protocol
from ..types import Team, Params

class RatingSystem(Protocol):
    """Pluggable rating-system interface. Each implementation knows its scale
    S, its rating-to-goal constant c, and how to compute rating diffs,
    win-expectations, λ-rates, and post-match rating updates."""

    name: str          # "elo" | "fifa" | "blend"
    scale: float       # S: 400 for Elo/Blend, 600 for FIFA
    c: float           # rating-to-goal scaling constant
    home_bonus: float  # H, in this system's native units (Elo bonus is rescaled in FIFA)

    def rating_of(self, team: Team) -> float: ...
    def rating_diff(self, a: Team, b: Team,
                    a_is_host: bool, b_is_host: bool) -> float: ...
    def win_expectation(self, diff: float) -> float: ...
    def lambdas(self, diff: float, mu: float, lambda_min: float
                ) -> tuple[float, float]: ...
    def update(self, before: float, expected: float,
               score_home: int, score_away: int) -> float: ...
```

Concrete classes (`EloRating`, `FifaRating`, `BlendRating`) take a `Params` in `__init__` and implement these methods per PRD §5.5. `BlendRating` composes an `EloRating` and `FifaRating` internally and produces a blended rating in Elo space per PRD §5.5's normalisation formula.

## 7. Match model

Pure functions in `wcsim/model.py`:

```python
SCORE_GRID_MAX = 8   # inclusive => 9x9 grid; > 99.99% mass

def _poisson_pmf(lmbda: float, max_goals: int) -> np.ndarray: ...
def _apply_tau(grid: np.ndarray, lam_a: float, lam_b: float, rho: float) -> np.ndarray:
    """Dixon-Coles τ adjustment on the four lowest-scoring joint outcomes."""

def predict_match(team_a, team_b, *, rating, params=Params(),
                  a_is_host=False, b_is_host=False) -> tuple[float, float, float]:
    """(P(home_win), P(draw), P(away_win)). Pure; no RNG."""

def sample_match(team_a, team_b, *, rating, params=Params(),
                 a_is_host=False, b_is_host=False, rng: np.random.Generator,
                 stage: str = "group") -> MatchResult:
    """Sample (home_goals, away_goals) from the joint Poisson+τ grid using rng.
    For non-'group' stages, also handles extra time (λ scaled by 30/90) and
    penalty shootouts (rating-weighted Bernoulli)."""
```

`predict_match` is the canonical interface for probability outputs; `sample_match` is for tournament simulation. Both go through the same `_apply_tau`-corrected score grid.

## 8. Tournament module

Two hardcoded formats — selected by team count. Schema-driven structures are a later spike.

```python
# wcsim/tournament.py
from dataclasses import dataclass

@dataclass(frozen=True)
class TournamentStructure:
    name: str                    # "WC2018-2022" | "WC2026"
    groups_count: int            # 8 or 12
    group_size: int              # 4
    top_per_group: int           # 2 (both formats)
    best_thirds: int             # 0 (2018/2022) or 8 (2026)
    knockout_stages: list[str]   # ["R16","QF","SF","Final"] or ["R32","R16","QF","SF","Final"]
    third_place_playoff: bool    # True for 2018/2022, False for 2026

STRUCTURE_2018_2022 = TournamentStructure(
    name="WC2018-2022", groups_count=8, group_size=4,
    top_per_group=2, best_thirds=0,
    knockout_stages=["R16", "QF", "SF", "Final"],
    third_place_playoff=True,    # both 2018 and 2022 had 3rd-place playoffs
)

STRUCTURE_2026 = TournamentStructure(
    name="WC2026", groups_count=12, group_size=4,
    top_per_group=2, best_thirds=8,
    knockout_stages=["R32", "R16", "QF", "SF", "Final"],
    third_place_playoff=False,    # 2026 WC does not have a 3rd-place playoff
)


def _structure_for(team_count: int) -> TournamentStructure:
    """Dispatch: 32 teams → WC2018-2022; 48 → WC2026. Other counts raise ValueError."""
    if team_count == 32:
        return STRUCTURE_2018_2022
    if team_count == 48:
        return STRUCTURE_2026
    raise ValueError(f"Unsupported tournament size: {team_count} teams "
                     "(supported: 32 for WC 2018/2022, 48 for WC 2026)")


def simulate_group_stage(
    teams: dict[str, Team], draw: dict[str, list[str]],
    *, rating: RatingSystem, params: Params, rng: np.random.Generator,
    hosts: set[str],
) -> tuple[list[MatchResult], dict[str, int]]:
    """6 matches per group × groups_count. Tiebreakers (per PRD §5.5):
    points → GD → GF → head-to-head → random. Returns (matches,
    finishing_position) where position ∈ {1..group_size}."""

def best_third_place_teams(group_results, n: int) -> list[str]:
    """Rank the third-place teams by points → GD → GF, return top n iso3s.
    For 2018/2022 (n=0), returns []. For 2026 (n=8), returns 8 iso3s."""

def seed_knockout(structure, group_winners, group_runners_up, best_thirds) -> list[str]:
    """Apply the format's seeding rule. Returns iso3 codes in bracket order
    (`structure.groups_count * structure.top_per_group + structure.best_thirds`
    in total — 16 for WC 2018/2022, 32 for WC 2026)."""

def simulate_knockout(
    seeded: list[str], teams: dict[str, Team], structure,
    *, rating: RatingSystem, params: Params, rng: np.random.Generator,
) -> tuple[list[MatchResult], dict[str, str]]:
    """Runs `structure.knockout_stages` rounds. Each match sampled with extra
    time + penalties handling. If `structure.third_place_playoff`, plays an
    additional 3rd-place match between the two semifinal losers. Returns
    (matches, {iso3 -> exit_stage})."""

def simulate_tournament(
    teams: dict[str, Team], draw: dict[str, list[str]], hosts: set[str],
    *, rating: RatingSystem, params: Params = Params(), seed: int,
) -> TournamentResult:
    """Top-level deterministic entry point. Selects format via `_structure_for(len(teams))`.
    Combines group stage + knockout (+ 3rd-place playoff if format requires)."""
```

The `rng` is created from `seed` at the top of `simulate_tournament` using `numpy.random.default_rng(seed)` and threaded through every sampling call so the simulation is fully deterministic.

In-tournament rating updates apply after each match via `rating.update(...)`, and the updated ratings are used for subsequent predictions (per PRD §5.5).

**Bundled 2026 data**: Spike 2 also produces three new data files under `spikes/01-validation/data/raw/`:
- `wc2026_draw.json` — the 12 groups × 4 teams as drawn on 2025-12-05 (public; sourced from FIFA's official site or Wikipedia, hand-curated if needed).
- `elo_history.csv` — augmented with current (May 2026) Elo for all 48 WC 2026 participants; scraper extended to handle a third snapshot date.
- `fifa_ranking.csv` — augmented with the latest pre-tournament FIFA ranking (March or April 2026); already covered by the existing scraper at a new dateId.

## 9. Test strategy

- **One test file per source module**, mirroring `wcsim/` layout under `tests/`.
- **TDD cycle per public function**: write the failing test first, run to confirm fail, write minimum implementation, run to confirm pass, commit. The implementation plan will spell this out per task.
- **conftest.py** provides:
  - `sample_team_brazil`, `sample_team_france` fixtures with known Elo/FIFA values
  - `default_params` fixture: `Params()` with PRD v1.7 defaults
  - `bundled_elo_history` / `bundled_matches` fixtures: pandas DataFrames loaded from `spikes/01-validation/data/raw/`
  - `paths` autouse fixture: prepends `<repo_root>` to `sys.path` (so `import wcsim` works) and `<repo_root>/spikes/01-validation/` to `sys.path` (so `import validate` works — the spike directory's name `01-validation` starts with a digit and can't be a normal Python package, so adding it to `sys.path` and importing `validate` as a top-level module is the cleanest path)
- **Regression guard** (`tests/test_regression.py`): the highest-value test in the suite. For each rating mode in `(elo, fifa, blend)`, iterates every WC 2018+2022 match, calls `wcsim.predict_match(...)` AND the corresponding call into `spikes.01-validation.validate`, asserts the (p_home, p_draw, p_away) tuples match to within `1e-9`. Failing this test means the extraction silently changed behavior — Spike 2 is broken.
- **Tournament tests** (`tests/test_tournament.py`):
  - **WC 2022 pin**: with `seed=42` and the bundled WC 2022 draw + 2022-11-19 ratings, pin `(champion_iso3, set_of_R16_advancers)` to a reference computed once on the first green run.
  - **WC 2026 pin**: with `seed=42` and the bundled WC 2026 draw + May 2026 ratings, pin `(champion_iso3, set_of_R32_advancers)` similarly.
  - **Format dispatch test**: pass 32 teams → assert `_structure_for(32) == STRUCTURE_2018_2022`; 48 → `STRUCTURE_2026`; other counts raise `ValueError`.
  - **Third-place playoff**: assert WC 2022 result includes a 3rd-place match; WC 2026 result does not.
  - Both pins lock the implementation against silent changes to RNG seeding or tournament logic.
- **Coverage target**: ≥ 95% line on `wcsim/`; 100% on `predict_match`, `_apply_tau`, `RatingSystem.lambdas`, `RatingSystem.update`.

## 10. Deterministic-RNG contract

`simulate_tournament(..., seed=N)` must produce byte-identical `TournamentResult` for the same inputs across:

- Multiple invocations in the same process
- Cold restart of the Python interpreter
- Different NumPy versions (within minor version bumps; `default_rng(seed)` is stable per NumPy's compat policy)

The RNG is created once at the top of `simulate_tournament` and threaded through every `sample_match` call. No module-level RNG state.

Spike 3's Monte Carlo runner will rely on this contract (counter-based seeding per simulation index) — Spike 2 must not break it.

## 11. Risks & Open Questions

- **In-tournament rating updates change predictions across matches.** validate.py freezes ratings at the snapshot date (it does only single-match predictions, never a sequence). The library must implement updates per PRD §5.5 for `simulate_tournament` to be correct. The regression test in §9 is single-match only (matches what validate.py does); the tournament pins in §9 are the only checks that updates are wired correctly. **Risk: subtle bug in `rating.update()` could go undetected.** Mitigation: unit-test `update()` against PRD §5.5's formulas for each rating system independently.
- **2026 WC seeding rule needs a fixed 15-row decision table.** The first-knockout-round (R32) bracket pairing for 12 groups + 8 best thirds is determined by which groups produced the qualifying thirds. The constants live in `tournament.py:_BEST_THIRDS_BRACKET_LOOKUP_2026`. Sourcing: FIFA's published 2026 WC bracket diagram (the same table CONMEBOL and Euro 2024 derivatives used). To be confirmed during implementation by cross-referencing FIFA's officially-published 2026 bracket pairings.
- **WC 2026 data completeness.** The 2026 draw was held on 2025-12-05; if any qualifying playoffs were unresolved at draw time, those slots are placeholders (e.g., "AFC playoff winner") rather than real teams. As of 2026-05-16, qualification is effectively complete — the spec assumes 48 real teams are known. If 1-2 slots remain placeholders at implementation time, the bundled draw substitutes a reasonable proxy (e.g., the higher-ranked team in the playoff that's still pending) and `wc2026_draw.json` documents the substitution. Tournament test would then pin against the substituted draw.
- **WC 2026 Elo / FIFA snapshot reproducibility.** Unlike 2018 and 2022 (historical, frozen), the May 2026 snapshot is "current at scrape time." Re-running the scrapers later returns updated values. We capture a fixed snapshot at Spike 2 time and commit the resulting CSV so future runs are reproducible. Scraper-output drift is documented but accepted.
- **`f0` default** in `Params` is `1300.0` as a stand-in. The real value should come from a snapshot metadata file (PRD §5.5). Spike 2 keeps the stand-in; a `Params.from_snapshot(snapshot_meta)` constructor is a later-spike concern. Note: Spike 1 already showed `f0_2018=877.7` vs `f0_2022=1579.1` — Spike 2's bundled snapshot uses per-snapshot values inside the tests (loaded from data), not the `Params` default.
- **No tournament-structure schema yet.** Hardcoded 2018/2022 and 2026 formats means Euro 2024 (24 teams) / Africa Cup / Copa América etc. can't be simulated by Spike 2. Acceptable for the spike's purpose (extract the math + cover the two formats we have data for); the schema lands in a later spike.

## 12. Acceptance Criteria

1. `wcsim/` exists with the 9 source files in §3.
2. `tests/` exists with the 9 test files in §3.
3. `python -m pytest tests/ -v` passes 100% of tests.
4. `tests/test_regression.py` includes assertions for all three rating modes × all 128 WC 2018+2022 matches; all pass within `1e-9`.
5. `tests/test_tournament.py` runs `simulate_tournament` at `seed=42` on **both** the WC 2022 draw (32 teams, 64 matches, 3rd-place playoff included) and the WC 2026 draw (48 teams, 104 matches, no 3rd-place playoff); pins champion + first-knockout-round advancers for each; passes.
6. `_structure_for(team_count)` correctly dispatches: 32 → `STRUCTURE_2018_2022`, 48 → `STRUCTURE_2026`, other → `ValueError`.
7. `python -m pytest tests/ --cov=wcsim --cov-report=term-missing` reports ≥ 95% line coverage on `wcsim/`.
8. `wcsim.__init__` re-exports exactly the public API in §4; no extra symbols leak.
9. `Params(rho=0.0)` reproduces independent-Poisson behavior (regression test confirms by matching validate.py at the v1.6 baseline).
10. `spikes/01-validation/data/raw/wc2026_draw.json` exists and contains 12 groups × 4 ISO3 codes; the scraper enhancements that produced the May-2026 Elo + FIFA snapshots are committed; running the scrapers from scratch reproduces the bundled CSVs (modulo intentional drift documented in §11).

## 13. Out of scope

- Monte Carlo runner (`wcsim.sim`) — Spike 3.
- CLI / Typer — Spike 5.
- Migration of scrapers from `spikes/01-validation/scrapers/` into `wcsim/scrapers/` — later spike. (Spike 2 *extends* the spike-resident scrapers to fetch 2026 data; it does not move them.)
- Tournament-structure JSON schema (arbitrary group counts, non-standard advancement rules) — later spike. Spike 2 hardcodes 8-group and 12-group formats only.
- Other tournament formats (Euro 24, Copa América 16, AFCON 24, etc.) — later spike.
- `pyproject.toml` / pip-installable packaging — Spike 6.
- Performance optimization (≤ 60s for 100k sims) — Spike 3+ concern.
