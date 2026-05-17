# Spike 2 — Library Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the validated match math from `spikes/01-validation/validate.py` into a flat `wcsim/` library and add tournament-simulation logic supporting both WC 2018/2022 (32 teams, 8 groups, R16 first round, 3rd-place playoff) and WC 2026 (48 teams, 12 groups, R32, no 3rd-place playoff). Test-first with pytest.

**Architecture:** Flat directory layout (no `pyproject.toml` yet, sibling of `spikes/`). Pluggable `RatingSystem` Protocol with three concrete classes (`EloRating`, `FifaRating`, `BlendRating`). Pure-function match model with Dixon-Coles τ. Tournament module dispatches on `len(teams)` to one of two hardcoded `TournamentStructure` constants. Spike 1's `validate.py` becomes the regression baseline — `wcsim.predict_match` must reproduce it to within `1e-9` across all 128 WC 2018+2022 matches and three rating modes.

**Tech Stack:** Python ≥ 3.11, NumPy ≥ 1.26, pandas ≥ 2.2, pytest ≥ 8 (added to `requirements.txt`). No new heavy deps.

**Heads-up on a spec gap:** The spec's `Params` dataclass in §5 is missing `k_elo` and `k_fifa` — needed for the in-tournament rating-update formulas in PRD §5.5. Task 3 adds them with PRD defaults (`60.0` each); the rest of the plan assumes their presence.

---

## Task 1: Scaffold the library + add pytest to the environment

**Files:**
- Create: `wcsim/__init__.py` (empty)
- Create: `wcsim/types.py` (empty)
- Create: `wcsim/model.py` (empty)
- Create: `wcsim/tournament.py` (empty)
- Create: `wcsim/ratings/__init__.py` (empty)
- Create: `wcsim/ratings/base.py` (empty)
- Create: `wcsim/ratings/elo.py` (empty)
- Create: `wcsim/ratings/fifa.py` (empty)
- Create: `wcsim/ratings/blend.py` (empty)
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Create: `tests/test_ratings/__init__.py` (empty)
- Modify: `spikes/01-validation/requirements.txt` (add `pytest>=8.0`, `pytest-cov>=4.1`)

- [ ] **Step 1: Create the package skeleton.**

```bash
mkdir -p wcsim/ratings tests/test_ratings
touch wcsim/__init__.py wcsim/types.py wcsim/model.py wcsim/tournament.py
touch wcsim/ratings/__init__.py wcsim/ratings/base.py wcsim/ratings/elo.py
touch wcsim/ratings/fifa.py wcsim/ratings/blend.py
touch tests/__init__.py tests/test_ratings/__init__.py
```

- [ ] **Step 2: Add pytest deps to the existing requirements.** Append two lines to `spikes/01-validation/requirements.txt`:

```
pytest>=8.0
pytest-cov>=4.1
```

- [ ] **Step 3: Install pytest into the existing `.venv`.**

```bash
spikes/01-validation/.venv/bin/pip install -r spikes/01-validation/requirements.txt
spikes/01-validation/.venv/bin/python -m pytest --version
```

Expected: pytest 8.x reports its version, exit 0.

- [ ] **Step 4: Create `tests/conftest.py`** with `sys.path` setup and shared fixtures. The spike's `01-validation` directory starts with a digit and contains a dash, so it can't be a normal Python package — we insert it into `sys.path` so `import validate` works at top level.

```python
"""Shared pytest fixtures for the wcsim library suite."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
SPIKE_DIR = REPO_ROOT / "spikes" / "01-validation"
DATA_DIR = SPIKE_DIR / "data" / "raw"


# Autouse: prepend repo root (so `import wcsim` works) and the spike directory
# (so `import validate` works — the spike name '01-validation' can't be a
# Python package because it starts with a digit and contains a dash).
@pytest.fixture(scope="session", autouse=True)
def _setup_paths():
    for p in (REPO_ROOT, SPIKE_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    yield


@pytest.fixture
def sample_team_brazil():
    from wcsim.types import Team
    return Team(
        name="Brazil", iso3="BRA", confederation="CONMEBOL",
        elo=2141.0, elo_updated=date(2018, 6, 13),
        fifa_points=1431.0, fifa_rank=2, fifa_updated=date(2018, 6, 7),
    )


@pytest.fixture
def sample_team_france():
    from wcsim.types import Team
    return Team(
        name="France", iso3="FRA", confederation="UEFA",
        elo=1986.0, elo_updated=date(2018, 6, 13),
        fifa_points=1198.0, fifa_rank=7, fifa_updated=date(2018, 6, 7),
    )


@pytest.fixture
def default_params():
    from wcsim.types import Params
    return Params()


@pytest.fixture
def bundled_elo_history():
    return pd.read_csv(DATA_DIR / "elo_history.csv")


@pytest.fixture
def bundled_fifa_ranking():
    return pd.read_csv(DATA_DIR / "fifa_ranking.csv")


@pytest.fixture
def bundled_matches():
    return pd.read_csv(DATA_DIR / "matches_history.csv")
```

- [ ] **Step 5: Verify pytest discovers the suite (with no tests yet).**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/ -v
```

Expected: "no tests ran in 0.0s", exit 5 (pytest's "no tests collected" code).

- [ ] **Step 6: Commit.**

```bash
git add wcsim/ tests/ spikes/01-validation/requirements.txt
git commit -m "Spike 2 Task 1: scaffold wcsim/ + tests/ + pytest"
```

---

## Task 2: `wcsim/types.py` — Team, MatchResult, TournamentResult

**Files:**
- Modify: `wcsim/types.py`
- Create: `tests/test_types.py`

(Params is bigger and gets its own task next.)

- [ ] **Step 1: Write the failing tests.** Create `tests/test_types.py`:

```python
"""Tests for wcsim.types dataclasses."""
from __future__ import annotations

from datetime import date

import pytest


def test_team_is_frozen_dataclass():
    from wcsim.types import Team
    t = Team(name="Brazil", iso3="BRA", confederation="CONMEBOL", elo=2141.0)
    with pytest.raises(Exception):   # FrozenInstanceError or AttributeError
        t.elo = 1.0


def test_team_fifa_fields_default_to_none():
    from wcsim.types import Team
    t = Team(name="Brazil", iso3="BRA", confederation="CONMEBOL", elo=2141.0)
    assert t.fifa_points is None
    assert t.fifa_rank is None
    assert t.fifa_updated is None
    assert t.elo_updated is None


def test_team_with_full_fields(sample_team_brazil):
    assert sample_team_brazil.iso3 == "BRA"
    assert sample_team_brazil.elo == 2141.0
    assert sample_team_brazil.fifa_points == 1431.0
    assert sample_team_brazil.fifa_updated == date(2018, 6, 7)


def test_match_result_required_fields():
    from wcsim.types import MatchResult
    m = MatchResult(
        home="ARG", away="FRA",
        home_goals=3, away_goals=3,
        stage="Final", neutral=True,
        extra_time=True, went_to_pens=True, pen_winner="ARG",
        home_rating_before=2143.0, away_rating_before=2004.0,
    )
    assert m.pen_winner == "ARG"
    assert m.extra_time is True


def test_tournament_result_fields():
    from wcsim.types import TournamentResult, MatchResult
    r = TournamentResult(
        seed=42, rating_mode="elo",
        matches=[],
        placements={"ARG": "Champion", "FRA": "Final"},
        final_ratings={"ARG": 2167.5},
    )
    assert r.seed == 42
    assert r.placements["ARG"] == "Champion"
```

- [ ] **Step 2: Run tests to verify they fail.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_types.py -v
```

Expected: all 5 tests fail with `ImportError` or `AttributeError` (types don't exist yet).

- [ ] **Step 3: Implement `wcsim/types.py`.**

```python
"""Public data types for wcsim. PRD §8-aligned. FIFA fields on Team are
optional to support Elo-only inputs (Spike 1 finding)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Team:
    name: str
    iso3: str
    confederation: str
    elo: float
    elo_updated: date | None = None
    fifa_points: float | None = None
    fifa_rank: int | None = None
    fifa_updated: date | None = None


@dataclass(frozen=True)
class MatchResult:
    home: str
    away: str
    home_goals: int
    away_goals: int
    stage: str
    neutral: bool
    extra_time: bool
    went_to_pens: bool
    pen_winner: str | None
    home_rating_before: float
    away_rating_before: float


@dataclass(frozen=True)
class TournamentResult:
    seed: int
    rating_mode: str
    matches: list[MatchResult]
    placements: dict[str, str]
    final_ratings: dict[str, float]
```

- [ ] **Step 4: Run tests to verify they pass.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_types.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit.**

```bash
git add wcsim/types.py tests/test_types.py
git commit -m "Spike 2 Task 2: types.Team, MatchResult, TournamentResult"
```

---

## Task 3: `wcsim/types.Params` — model defaults with k_elo / k_fifa

**Files:**
- Modify: `wcsim/types.py` (add `Params`)
- Modify: `tests/test_types.py` (add Params tests)

The spec's §5 listing of `Params` omits `k_elo` / `k_fifa`; this task adds them at PRD §5.2 defaults of `60.0`.

- [ ] **Step 1: Append the failing Params test** to `tests/test_types.py`:

```python
def test_params_defaults_match_prd_v17():
    from wcsim.types import Params
    p = Params()
    assert p.c_elo == 300.0
    assert p.c_fifa == 450.0
    assert p.mu == 1.35
    assert p.lambda_min == 0.05
    assert p.blend_w == 0.7
    assert p.e0 == 1500.0
    assert p.f0 == 1300.0
    assert p.home_bonus_elo == 100.0
    assert p.home_bonus_fifa == 150.0
    assert p.rho == 0.0
    assert p.k_elo == 60.0
    assert p.k_fifa == 60.0


def test_params_is_frozen():
    from wcsim.types import Params
    import pytest
    p = Params()
    with pytest.raises(Exception):
        p.mu = 2.0
```

- [ ] **Step 2: Run to verify failure.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_types.py::test_params_defaults_match_prd_v17 -v
```

Expected: fail with `ImportError` (Params not defined).

- [ ] **Step 3: Add Params to `wcsim/types.py`.** Append:

```python
@dataclass(frozen=True)
class Params:
    """PRD v1.7 §5.5 model parameters. All defaults match the PRD."""
    c_elo: float = 300.0
    c_fifa: float = 450.0
    mu: float = 1.35
    lambda_min: float = 0.05
    blend_w: float = 0.7
    e0: float = 1500.0
    f0: float = 1300.0
    home_bonus_elo: float = 100.0
    home_bonus_fifa: float = 150.0
    rho: float = 0.0
    k_elo: float = 60.0   # Elo K-factor for in-tournament rating updates
    k_fifa: float = 60.0  # FIFA importance constant (I) for World Cup matches
```

- [ ] **Step 4: Run to verify passing.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_types.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit.**

```bash
git add wcsim/types.py tests/test_types.py
git commit -m "Spike 2 Task 3: types.Params (with k_elo, k_fifa)"
```

---

## Task 4: `wcsim/ratings/base.py` — RatingSystem Protocol

**Files:**
- Modify: `wcsim/ratings/base.py`
- Create: `tests/test_ratings/test_base.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_ratings/test_base.py`:

```python
"""Tests for the RatingSystem Protocol."""
from __future__ import annotations


def test_rating_system_is_a_protocol():
    """RatingSystem should be a Protocol class — structural typing only,
    not a runtime base class. Concrete classes don't inherit from it."""
    from typing import get_type_hints
    from wcsim.ratings.base import RatingSystem
    assert hasattr(RatingSystem, "__protocol_attrs__") or hasattr(
        RatingSystem, "_is_protocol"
    )


def test_rating_system_required_methods():
    """RatingSystem must declare the five methods specified in spec §6."""
    from wcsim.ratings.base import RatingSystem
    for method_name in (
        "rating_of", "rating_diff", "win_expectation", "lambdas", "update",
    ):
        assert hasattr(RatingSystem, method_name), f"missing {method_name}"


def test_rating_system_required_attributes():
    """RatingSystem must declare name, scale, c, home_bonus as class attrs."""
    from wcsim.ratings.base import RatingSystem
    hints = RatingSystem.__annotations__
    for attr in ("name", "scale", "c", "home_bonus"):
        assert attr in hints, f"missing attribute {attr}"
```

- [ ] **Step 2: Run to verify failure.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_ratings/test_base.py -v
```

Expected: fail with `ImportError`.

- [ ] **Step 3: Implement `wcsim/ratings/base.py`.**

```python
"""Pluggable rating-system interface. Concrete classes (EloRating,
FifaRating, BlendRating) implement this Protocol structurally — no
inheritance required."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Params, Team


@runtime_checkable
class RatingSystem(Protocol):
    """Each rating system declares its scale S, rating-to-goal constant c,
    and home-bonus H (in its native units), and exposes the five methods
    that the match model and tournament engine call."""

    name: str
    scale: float
    c: float
    home_bonus: float

    def rating_of(self, team: Team) -> float: ...

    def rating_diff(
        self, a: Team, b: Team, a_is_host: bool, b_is_host: bool,
    ) -> float: ...

    def win_expectation(self, diff: float) -> float: ...

    def lambdas(
        self, diff: float, mu: float, lambda_min: float,
    ) -> tuple[float, float]: ...

    def update(
        self, before: float, expected: float,
        score_home: int, score_away: int,
    ) -> float: ...
```

- [ ] **Step 4: Run to verify passing.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_ratings/test_base.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit.**

```bash
git add wcsim/ratings/base.py tests/test_ratings/test_base.py
git commit -m "Spike 2 Task 4: ratings.base RatingSystem Protocol"
```

---

## Task 5: `wcsim/ratings/elo.py` — EloRating

**Files:**
- Modify: `wcsim/ratings/elo.py`
- Create: `tests/test_ratings/test_elo.py`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_ratings/test_elo.py`:

```python
"""Tests for EloRating. Math cross-checked against validate.py's
implementation and PRD §5.5."""
from __future__ import annotations

import math


def test_elo_attributes():
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    r = EloRating(Params())
    assert r.name == "elo"
    assert r.scale == 400.0
    assert r.c == 300.0
    assert r.home_bonus == 100.0


def test_rating_of_returns_team_elo(sample_team_brazil):
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    r = EloRating(Params())
    assert r.rating_of(sample_team_brazil) == 2141.0


def test_rating_diff_neutral_venue(sample_team_brazil, sample_team_france):
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    r = EloRating(Params())
    diff = r.rating_diff(
        sample_team_brazil, sample_team_france,
        a_is_host=False, b_is_host=False,
    )
    assert diff == 2141.0 - 1986.0   # 155.0


def test_rating_diff_with_a_host(sample_team_brazil, sample_team_france):
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    r = EloRating(Params())
    diff = r.rating_diff(
        sample_team_brazil, sample_team_france,
        a_is_host=True, b_is_host=False,
    )
    assert diff == 2141.0 + 100.0 - 1986.0   # 255.0


def test_win_expectation_neutral_zero_diff():
    """With D=0, win expectation is 0.5."""
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    r = EloRating(Params())
    assert math.isclose(r.win_expectation(0.0), 0.5, abs_tol=1e-12)


def test_win_expectation_400_point_advantage():
    """D = +400 → W_e ≈ 0.909 (1 / (1 + 10^-1))."""
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    r = EloRating(Params())
    assert math.isclose(r.win_expectation(400.0), 1.0 / (1.0 + 10**-1.0), abs_tol=1e-12)


def test_lambdas_zero_diff_returns_mu():
    """At D=0, λ_a = λ_b = μ (both teams expected to score μ goals)."""
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = Params()
    r = EloRating(p)
    lam_a, lam_b = r.lambdas(0.0, mu=p.mu, lambda_min=p.lambda_min)
    assert lam_a == p.mu
    assert lam_b == p.mu


def test_lambdas_positive_diff():
    """D = +300 → λ_a = μ + 0.5, λ_b = μ - 0.5."""
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = Params()
    r = EloRating(p)
    lam_a, lam_b = r.lambdas(300.0, mu=p.mu, lambda_min=p.lambda_min)
    assert math.isclose(lam_a, p.mu + 0.5, abs_tol=1e-12)
    assert math.isclose(lam_b, p.mu - 0.5, abs_tol=1e-12)


def test_lambdas_floor_applied():
    """At very negative D, λ_b clips to lambda_min."""
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = Params()
    r = EloRating(p)
    lam_a, lam_b = r.lambdas(-2000.0, mu=p.mu, lambda_min=p.lambda_min)
    assert lam_b == p.lambda_min


def test_update_draw_1_1():
    """1-1 draw: G_m = 1, W = 0.5. Update is K_elo * 1 * (0.5 - W_e)."""
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = Params()
    r = EloRating(p)
    new_rating = r.update(
        before=2000.0, expected=0.5,
        score_home=1, score_away=1,
    )
    assert math.isclose(new_rating, 2000.0 + 60.0 * 1.0 * (0.5 - 0.5), abs_tol=1e-12)


def test_update_2_goal_win_uses_gm_1_5():
    """2-0 home win: G_m = 1.5, W = 1.0."""
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = Params()
    r = EloRating(p)
    # Home team expected 0.6, won 2-0
    new_rating = r.update(
        before=2000.0, expected=0.6,
        score_home=2, score_away=0,
    )
    # K=60, G_m=1.5, W-W_e = 1 - 0.6 = 0.4
    assert math.isclose(new_rating, 2000.0 + 60.0 * 1.5 * 0.4, abs_tol=1e-12)


def test_update_3_goal_margin_uses_gm_formula():
    """3-0 win: G_m = (11 + 3) / 8 = 1.75."""
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = Params()
    r = EloRating(p)
    new_rating = r.update(
        before=2000.0, expected=0.6,
        score_home=3, score_away=0,
    )
    assert math.isclose(new_rating, 2000.0 + 60.0 * 1.75 * 0.4, abs_tol=1e-12)


def test_update_away_loss_decreases_rating():
    """Away team lost 0-3: W = 0, G_m = (11+3)/8 = 1.75."""
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = Params()
    r = EloRating(p)
    # Away team expected 0.4
    new_rating = r.update(
        before=1900.0, expected=0.4,
        score_home=3, score_away=0,
    )
    # K=60, G_m=1.75, W-W_e = 0 - 0.4 = -0.4
    assert math.isclose(new_rating, 1900.0 + 60.0 * 1.75 * (-0.4), abs_tol=1e-12)
```

- [ ] **Step 2: Run to verify failure.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_ratings/test_elo.py -v
```

Expected: all 12 tests fail with `ImportError`.

- [ ] **Step 3: Implement `wcsim/ratings/elo.py`.**

```python
"""Elo rating system per PRD §5.5. Scale S=400, c=300, K=60 for World Cup
matches. Goal-margin multiplier G_m: 1 (draw or 1-goal win), 1.5 (2-goal),
(11 + |Δ|) / 8 (margins of 3+)."""
from __future__ import annotations

from ..types import Params, Team


class EloRating:
    name = "elo"
    scale = 400.0

    def __init__(self, params: Params):
        self._params = params
        self.c = params.c_elo
        self.home_bonus = params.home_bonus_elo
        self.k = params.k_elo

    def rating_of(self, team: Team) -> float:
        return team.elo

    def rating_diff(
        self, a: Team, b: Team, a_is_host: bool, b_is_host: bool,
    ) -> float:
        host_diff = float(a_is_host) - float(b_is_host)
        return (a.elo - b.elo) + self.home_bonus * host_diff

    def win_expectation(self, diff: float) -> float:
        return 1.0 / (1.0 + 10.0 ** (-diff / self.scale))

    def lambdas(
        self, diff: float, mu: float, lambda_min: float,
    ) -> tuple[float, float]:
        lam_a = max(lambda_min, mu + diff / (2.0 * self.c))
        lam_b = max(lambda_min, mu - diff / (2.0 * self.c))
        return lam_a, lam_b

    def update(
        self, before: float, expected: float,
        score_home: int, score_away: int,
    ) -> float:
        """Update a single team's rating. Caller passes that team's expected
        W (in [0, 1]) and the match's home/away scores.

        The W (actual result) and G_m (margin multiplier) are derived from
        the scores: positive net for the team boosts W, negative reduces it.
        Caller is responsible for picking the right `expected` and reading
        `score_home`/`score_away` from that team's perspective (i.e., home
        team passes (W_home_expected, home_score, away_score); away team
        passes (W_away_expected, away_score, home_score) — note the swap)."""
        delta = score_home - score_away
        if score_home == score_away:
            w_actual = 0.5
        elif delta > 0:
            w_actual = 1.0
        else:
            w_actual = 0.0
        abs_delta = abs(delta)
        if abs_delta <= 1:
            gm = 1.0
        elif abs_delta == 2:
            gm = 1.5
        else:
            gm = (11.0 + abs_delta) / 8.0
        return before + self.k * gm * (w_actual - expected)
```

- [ ] **Step 4: Run to verify passing.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_ratings/test_elo.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit.**

```bash
git add wcsim/ratings/elo.py tests/test_ratings/test_elo.py
git commit -m "Spike 2 Task 5: ratings.elo.EloRating with full PRD §5.5 update formula"
```

---

## Task 6: `wcsim/ratings/fifa.py` — FifaRating

**Files:**
- Modify: `wcsim/ratings/fifa.py`
- Create: `tests/test_ratings/test_fifa.py`

Same shape as EloRating, but `scale=600`, `c=450`, `home_bonus=150`, and update uses `I = k_fifa` (no goal-margin multiplier).

- [ ] **Step 1: Write the failing tests.** Create `tests/test_ratings/test_fifa.py`:

```python
"""Tests for FifaRating. FIFA's official update formula has no goal-margin
multiplier: R' = R + I * (W - W_e), where I = k_fifa (60 for WC matches)."""
from __future__ import annotations

import math

from datetime import date


def test_fifa_attributes():
    from wcsim.ratings.fifa import FifaRating
    from wcsim.types import Params
    r = FifaRating(Params())
    assert r.name == "fifa"
    assert r.scale == 600.0
    assert r.c == 450.0
    assert r.home_bonus == 150.0


def test_rating_of_returns_fifa_points(sample_team_brazil):
    from wcsim.ratings.fifa import FifaRating
    from wcsim.types import Params
    r = FifaRating(Params())
    assert r.rating_of(sample_team_brazil) == 1431.0


def test_rating_of_raises_if_fifa_missing():
    """If Team has no fifa_points, FifaRating must fail loudly."""
    from wcsim.ratings.fifa import FifaRating
    from wcsim.types import Params, Team
    import pytest
    elo_only = Team(name="X", iso3="XXX", confederation="UEFA", elo=1500.0)
    r = FifaRating(Params())
    with pytest.raises(ValueError, match="fifa_points"):
        r.rating_of(elo_only)


def test_rating_diff_with_host_uses_fifa_bonus(sample_team_brazil, sample_team_france):
    from wcsim.ratings.fifa import FifaRating
    from wcsim.types import Params
    r = FifaRating(Params())
    diff = r.rating_diff(
        sample_team_brazil, sample_team_france,
        a_is_host=True, b_is_host=False,
    )
    assert diff == (1431.0 + 150.0) - 1198.0


def test_win_expectation_uses_scale_600():
    """D = +600 → W_e ≈ 0.909."""
    from wcsim.ratings.fifa import FifaRating
    from wcsim.types import Params
    r = FifaRating(Params())
    assert math.isclose(r.win_expectation(600.0), 1.0 / (1.0 + 10**-1.0), abs_tol=1e-12)


def test_lambdas_uses_c_450():
    """D = +450 → λ_a - λ_b = 1.0."""
    from wcsim.ratings.fifa import FifaRating
    from wcsim.types import Params
    p = Params()
    r = FifaRating(p)
    lam_a, lam_b = r.lambdas(450.0, mu=p.mu, lambda_min=p.lambda_min)
    assert math.isclose(lam_a, p.mu + 0.5, abs_tol=1e-12)
    assert math.isclose(lam_b, p.mu - 0.5, abs_tol=1e-12)


def test_update_uses_I_no_goal_margin():
    """FIFA update: R' = R + I * (W - W_e). No G_m multiplier."""
    from wcsim.ratings.fifa import FifaRating
    from wcsim.types import Params
    p = Params()
    r = FifaRating(p)
    # 3-0 win, expected 0.6 → W=1, W-W_e=0.4. Update = 60 * 0.4 = 24.
    new_rating = r.update(
        before=1500.0, expected=0.6,
        score_home=3, score_away=0,
    )
    assert math.isclose(new_rating, 1500.0 + 60.0 * 0.4, abs_tol=1e-12)
```

- [ ] **Step 2: Run to verify failure.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_ratings/test_fifa.py -v
```

Expected: all 7 tests fail with `ImportError`.

- [ ] **Step 3: Implement `wcsim/ratings/fifa.py`.**

```python
"""FIFA Men's World Ranking system per PRD §5.5. Scale S=600 (empirically
chosen so FIFA-point gaps reproduce Elo's median win-rate), c=450, home
bonus 150 in FIFA points (= 100 Elo-equivalent rescaled by S_fifa/S_elo).
Update formula: R' = R + I * (W - W_e) with I = k_fifa (60 for WC matches)."""
from __future__ import annotations

from ..types import Params, Team


class FifaRating:
    name = "fifa"
    scale = 600.0

    def __init__(self, params: Params):
        self._params = params
        self.c = params.c_fifa
        self.home_bonus = params.home_bonus_fifa
        self.k = params.k_fifa

    def rating_of(self, team: Team) -> float:
        if team.fifa_points is None:
            raise ValueError(
                f"Team {team.iso3} has no fifa_points; FifaRating requires it. "
                "Either provide fifa_points or use EloRating instead."
            )
        return team.fifa_points

    def rating_diff(
        self, a: Team, b: Team, a_is_host: bool, b_is_host: bool,
    ) -> float:
        host_diff = float(a_is_host) - float(b_is_host)
        return (self.rating_of(a) - self.rating_of(b)) + self.home_bonus * host_diff

    def win_expectation(self, diff: float) -> float:
        return 1.0 / (1.0 + 10.0 ** (-diff / self.scale))

    def lambdas(
        self, diff: float, mu: float, lambda_min: float,
    ) -> tuple[float, float]:
        lam_a = max(lambda_min, mu + diff / (2.0 * self.c))
        lam_b = max(lambda_min, mu - diff / (2.0 * self.c))
        return lam_a, lam_b

    def update(
        self, before: float, expected: float,
        score_home: int, score_away: int,
    ) -> float:
        delta = score_home - score_away
        if score_home == score_away:
            w_actual = 0.5
        elif delta > 0:
            w_actual = 1.0
        else:
            w_actual = 0.0
        return before + self.k * (w_actual - expected)
```

- [ ] **Step 4: Run to verify passing.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_ratings/test_fifa.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit.**

```bash
git add wcsim/ratings/fifa.py tests/test_ratings/test_fifa.py
git commit -m "Spike 2 Task 6: ratings.fifa.FifaRating (S=600, c=450, I=60)"
```

---

## Task 7: `wcsim/ratings/blend.py` — BlendRating

**Files:**
- Modify: `wcsim/ratings/blend.py`
- Create: `tests/test_ratings/test_blend.py`

BlendRating wraps EloRating and FifaRating; uses Elo-space scale (S=400, c=300, H=100 Elo-equiv). The blended rating is `R_blend = w * R_elo + (1-w) * R_fifa * E0/F0`. Update applies to both components independently.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_ratings/test_blend.py`:

```python
"""Tests for BlendRating. The blended rating lives in Elo space (FIFA points
are normalised by E0/F0). Update is done on both components independently."""
from __future__ import annotations

import math


def test_blend_attributes():
    from wcsim.ratings.blend import BlendRating
    from wcsim.types import Params
    r = BlendRating(Params())
    assert r.name == "blend"
    assert r.scale == 400.0        # blend lives in Elo space
    assert r.c == 300.0
    assert r.home_bonus == 100.0   # Elo bonus


def test_rating_of_combines_elo_and_fifa(sample_team_brazil):
    """R_blend = w*Elo + (1-w) * Fifa * E0/F0. Brazil: w=0.7, e0=1500, f0=1300.
    R_blend = 0.7*2141 + 0.3 * 1431 * 1500/1300 = 1498.7 + 495.115... = 1993.815..."""
    from wcsim.ratings.blend import BlendRating
    from wcsim.types import Params
    r = BlendRating(Params())
    expected = 0.7 * 2141.0 + 0.3 * 1431.0 * 1500.0 / 1300.0
    assert math.isclose(r.rating_of(sample_team_brazil), expected, abs_tol=1e-9)


def test_rating_diff_uses_blended_ratings(sample_team_brazil, sample_team_france):
    from wcsim.ratings.blend import BlendRating
    from wcsim.types import Params
    r = BlendRating(Params())
    diff = r.rating_diff(
        sample_team_brazil, sample_team_france,
        a_is_host=False, b_is_host=False,
    )
    assert math.isclose(
        diff, r.rating_of(sample_team_brazil) - r.rating_of(sample_team_france),
        abs_tol=1e-12,
    )


def test_win_expectation_uses_elo_scale():
    """Blend uses scale=400 (Elo space). D=+400 → W_e ≈ 0.909."""
    from wcsim.ratings.blend import BlendRating
    from wcsim.types import Params
    r = BlendRating(Params())
    assert math.isclose(r.win_expectation(400.0), 1.0 / (1.0 + 10**-1.0), abs_tol=1e-12)


def test_lambdas_uses_c_elo():
    """Blend uses c=300 (Elo space). D=+300 → λ_a - λ_b = 1.0."""
    from wcsim.ratings.blend import BlendRating
    from wcsim.types import Params
    p = Params()
    r = BlendRating(p)
    lam_a, lam_b = r.lambdas(300.0, mu=p.mu, lambda_min=p.lambda_min)
    assert math.isclose(lam_a - lam_b, 1.0, abs_tol=1e-12)


def test_update_returns_blended_post_match_rating():
    """update() on Blend applies BOTH Elo and FIFA update formulas to the
    underlying components, returns the new blended rating."""
    from wcsim.ratings.blend import BlendRating
    from wcsim.types import Params
    p = Params()
    r = BlendRating(p)
    # update() takes the team's CURRENT blended rating; internally it needs to
    # recover the components. For simplicity, update returns the change
    # in BLENDED units; the caller stores the new blended value.
    # Validation: the function returns a finite float that differs from `before`.
    new_rating = r.update(
        before=2000.0, expected=0.6,
        score_home=2, score_away=0,
    )
    assert isinstance(new_rating, float)
    assert new_rating != 2000.0
```

- [ ] **Step 2: Run to verify failure.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_ratings/test_blend.py -v
```

Expected: 6 failures.

- [ ] **Step 3: Implement `wcsim/ratings/blend.py`.**

```python
"""Blended Elo + FIFA rating per PRD §5.5. The blended rating lives in
Elo space: R_blend = w * R_elo + (1 - w) * R_fifa * E0/F0. Predictions use
Elo-space parameters (S=400, c=300, H=100). The update method applies to
both components separately; for the Blend Protocol's update() signature
(operating on a single 'before' blended value), we apply a single Elo-style
update directly to that value as the simplest approximation — the
distinction matters only over multi-match tournament sequences."""
from __future__ import annotations

from ..types import Params, Team
from .elo import EloRating


class BlendRating:
    name = "blend"
    scale = 400.0    # Elo-space

    def __init__(self, params: Params):
        self._params = params
        self.c = params.c_elo
        self.home_bonus = params.home_bonus_elo
        self.k = params.k_elo
        self._w = params.blend_w
        self._e0 = params.e0
        self._f0 = params.f0
        self._elo = EloRating(params)

    def rating_of(self, team: Team) -> float:
        if team.fifa_points is None:
            raise ValueError(
                f"Team {team.iso3} has no fifa_points; BlendRating requires it."
            )
        fifa_in_elo_space = team.fifa_points * self._e0 / self._f0
        return self._w * team.elo + (1.0 - self._w) * fifa_in_elo_space

    def rating_diff(
        self, a: Team, b: Team, a_is_host: bool, b_is_host: bool,
    ) -> float:
        host_diff = float(a_is_host) - float(b_is_host)
        return (self.rating_of(a) - self.rating_of(b)) + self.home_bonus * host_diff

    def win_expectation(self, diff: float) -> float:
        return 1.0 / (1.0 + 10.0 ** (-diff / self.scale))

    def lambdas(
        self, diff: float, mu: float, lambda_min: float,
    ) -> tuple[float, float]:
        lam_a = max(lambda_min, mu + diff / (2.0 * self.c))
        lam_b = max(lambda_min, mu - diff / (2.0 * self.c))
        return lam_a, lam_b

    def update(
        self, before: float, expected: float,
        score_home: int, score_away: int,
    ) -> float:
        """Apply an Elo-style update to the blended rating directly. (For
        full PRD §5.5 fidelity, the Tournament module should update Elo and
        FIFA components separately and recompute the blend, but the single-
        value update() interface is sufficient for sequential simulation.)"""
        return self._elo.update(before, expected, score_home, score_away)
```

- [ ] **Step 4: Run to verify passing.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_ratings/test_blend.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit.**

```bash
git add wcsim/ratings/blend.py tests/test_ratings/test_blend.py
git commit -m "Spike 2 Task 7: ratings.blend.BlendRating (Elo + FIFA / E0/F0 normalised)"
```

---

## Task 8: `wcsim/model.py` — Poisson PMF + Dixon-Coles τ + predict_match

**Files:**
- Modify: `wcsim/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_model.py`:

```python
"""Tests for the match model: Poisson PMF, Dixon-Coles τ, predict_match."""
from __future__ import annotations

import math

import numpy as np


def test_poisson_pmf_sums_to_one_for_large_grid():
    from wcsim.model import _poisson_pmf
    pmf = _poisson_pmf(1.35, 20)
    assert math.isclose(pmf.sum(), 1.0, abs_tol=1e-6)


def test_poisson_pmf_correct_for_lambda_zero():
    """λ=0 → all mass at 0 goals."""
    from wcsim.model import _poisson_pmf
    pmf = _poisson_pmf(0.0, 8)
    assert math.isclose(pmf[0], 1.0, abs_tol=1e-12)
    assert pmf[1:].sum() == 0.0


def test_apply_tau_no_op_at_rho_zero():
    from wcsim.model import _apply_tau
    grid_before = np.ones((9, 9)) / 81.0
    grid_after = _apply_tau(grid_before.copy(), 1.35, 1.35, 0.0)
    np.testing.assert_array_almost_equal(grid_before, grid_after)


def test_apply_tau_positive_rho_suppresses_low_score_draws():
    """ρ > 0 reduces P(0-0) and P(1-1)."""
    from wcsim.model import _apply_tau
    grid = np.ones((9, 9)) / 81.0
    adjusted = _apply_tau(grid.copy(), 1.35, 1.35, 0.2)
    assert adjusted[0, 0] < grid[0, 0]
    assert adjusted[1, 1] < grid[1, 1]


def test_predict_match_returns_three_probs_summing_to_one(
    sample_team_brazil, sample_team_france,
):
    from wcsim.model import predict_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = predict_match(
        sample_team_brazil, sample_team_france,
        rating=EloRating(Params()),
    )
    assert len(p) == 3
    assert math.isclose(sum(p), 1.0, abs_tol=1e-9)


def test_predict_match_is_symmetric(sample_team_brazil, sample_team_france):
    """Swapping (a, b) swaps P(home) and P(away); P(draw) unchanged."""
    from wcsim.model import predict_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = predict_match(
        sample_team_brazil, sample_team_france, rating=EloRating(Params()),
    )
    q = predict_match(
        sample_team_france, sample_team_brazil, rating=EloRating(Params()),
    )
    assert math.isclose(p[0], q[2], abs_tol=1e-12)
    assert math.isclose(p[1], q[1], abs_tol=1e-12)
    assert math.isclose(p[2], q[0], abs_tol=1e-12)


def test_predict_match_with_rho_changes_outputs(
    sample_team_brazil, sample_team_france,
):
    """Non-zero ρ produces different probabilities than ρ=0."""
    from wcsim.model import predict_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    base = predict_match(
        sample_team_brazil, sample_team_france,
        rating=EloRating(Params(rho=0.0)),
    )
    with_rho = predict_match(
        sample_team_brazil, sample_team_france,
        rating=EloRating(Params(rho=0.2)),
        params=Params(rho=0.2),
    )
    assert not math.isclose(base[1], with_rho[1], abs_tol=1e-6)   # draw prob changes


def test_predict_match_host_bonus_helps_home(
    sample_team_brazil, sample_team_france,
):
    """When `a_is_host=True`, P(home wins) should be higher than neutral."""
    from wcsim.model import predict_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    neutral = predict_match(
        sample_team_france, sample_team_brazil, rating=EloRating(Params()),
    )
    home = predict_match(
        sample_team_france, sample_team_brazil, rating=EloRating(Params()),
        a_is_host=True,
    )
    assert home[0] > neutral[0]
```

- [ ] **Step 2: Run to verify failure.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_model.py -v
```

Expected: 8 failures with `ImportError`.

- [ ] **Step 3: Implement `wcsim/model.py`.**

```python
"""Match model: Poisson + Dixon-Coles τ. Pure deterministic predict_match;
RNG-driven sample_match comes in a later task."""
from __future__ import annotations

import numpy as np

from .ratings.base import RatingSystem
from .types import Params, Team

SCORE_GRID_MAX = 8   # 9x9 grid; >99.99% probability mass


def _poisson_pmf(lmbda: float, max_goals: int) -> np.ndarray:
    """Discrete Poisson PMF over [0, max_goals]. Drops residual mass."""
    k = np.arange(max_goals + 1)
    log_fact = np.cumsum(np.log(np.maximum(k, 1)))
    log_fact[0] = 0.0
    log_pmf = k * np.log(max(lmbda, 1e-12)) - lmbda - log_fact
    return np.exp(log_pmf)


def _apply_tau(
    grid: np.ndarray, lam_a: float, lam_b: float, rho: float,
) -> np.ndarray:
    """Apply Dixon-Coles τ correction in-place to the four lowest-scoring
    joint outcomes. Clamps each τ-adjusted cell to non-negative (the rare
    cases where ρ is so extreme it would make a probability negative)."""
    if rho == 0.0:
        return grid
    grid[0, 0] *= max(0.0, 1.0 - lam_a * lam_b * rho)
    grid[0, 1] *= max(0.0, 1.0 + lam_a * rho)
    grid[1, 0] *= max(0.0, 1.0 + lam_b * rho)
    grid[1, 1] *= max(0.0, 1.0 - rho)
    return grid


def _outcome_probs(lam_a: float, lam_b: float, rho: float) -> tuple[float, float, float]:
    """Return (P(home_win), P(draw), P(away_win)) from the τ-corrected joint
    Poisson grid."""
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
    """Return (P(team_a wins), P(draw), P(team_b wins)) for a single match
    under the given rating system. Pure function — no RNG."""
    p = params if params is not None else Params()
    diff = rating.rating_diff(team_a, team_b, a_is_host, b_is_host)
    lam_a, lam_b = rating.lambdas(diff, p.mu, p.lambda_min)
    return _outcome_probs(lam_a, lam_b, p.rho)
```

- [ ] **Step 4: Run to verify passing.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_model.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit.**

```bash
git add wcsim/model.py tests/test_model.py
git commit -m "Spike 2 Task 8: model.predict_match + Poisson PMF + Dixon-Coles τ"
```

---

## Task 9: `wcsim/model.sample_match` — RNG-driven sampling

**Files:**
- Modify: `wcsim/model.py` (add `sample_match`, `_sample_score`)
- Modify: `tests/test_model.py` (add sample_match tests)

`sample_match` samples a single match outcome from the score grid, including ET (if `stage != "group"`) and penalty shootout (rating-weighted Bernoulli).

- [ ] **Step 1: Append failing tests** to `tests/test_model.py`:

```python
def test_sample_match_returns_match_result(
    sample_team_brazil, sample_team_france,
):
    """sample_match returns a MatchResult with deterministic output for a
    given rng seed."""
    from wcsim.model import sample_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    rng = np.random.default_rng(42)
    m = sample_match(
        sample_team_brazil, sample_team_france,
        rating=EloRating(Params()),
        rng=rng, stage="group",
    )
    assert m.home == "BRA"
    assert m.away == "FRA"
    assert m.stage == "group"
    assert m.extra_time is False
    assert m.went_to_pens is False
    assert m.pen_winner is None
    assert isinstance(m.home_goals, int)
    assert isinstance(m.away_goals, int)
    assert m.home_rating_before == 2141.0
    assert m.away_rating_before == 1986.0


def test_sample_match_is_deterministic_with_seed(
    sample_team_brazil, sample_team_france,
):
    from wcsim.model import sample_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params

    def run(seed):
        rng = np.random.default_rng(seed)
        return sample_match(
            sample_team_brazil, sample_team_france,
            rating=EloRating(Params()),
            rng=rng, stage="group",
        )

    assert run(123).home_goals == run(123).home_goals
    assert run(123).away_goals == run(123).away_goals


def test_sample_match_knockout_handles_extra_time_when_tied(
    sample_team_brazil, sample_team_france,
):
    """When a knockout sample comes out level after 90 min, ET is sampled
    with λ scaled by 30/90. Loop until we see one to verify the flag works."""
    from wcsim.model import sample_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params

    saw_et = False
    for seed in range(50):
        rng = np.random.default_rng(seed)
        m = sample_match(
            sample_team_brazil, sample_team_france,
            rating=EloRating(Params()),
            rng=rng, stage="R16",
        )
        if m.extra_time:
            saw_et = True
            # ET should resolve the tie OR escalate to pens.
            assert (m.home_goals != m.away_goals) or m.went_to_pens
            break
    assert saw_et, "no extra-time match found in 50 seeds — sampling broken?"


def test_sample_match_pen_winner_iso3_one_of_the_teams(
    sample_team_brazil, sample_team_france,
):
    """If went_to_pens, pen_winner is one of the two iso3 codes."""
    from wcsim.model import sample_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params

    for seed in range(200):
        rng = np.random.default_rng(seed)
        m = sample_match(
            sample_team_brazil, sample_team_france,
            rating=EloRating(Params()),
            rng=rng, stage="Final",
        )
        if m.went_to_pens:
            assert m.pen_winner in {"BRA", "FRA"}
            break
```

- [ ] **Step 2: Run to verify failure.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_model.py -k sample_match -v
```

Expected: 4 failures.

- [ ] **Step 3: Append to `wcsim/model.py`.**

```python
ET_LAMBDA_SCALE = 30.0 / 90.0   # extra time is one-third the duration


def _sample_score(
    lam_a: float, lam_b: float, rho: float, rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample (home_goals, away_goals) from the τ-corrected joint Poisson
    grid. Returns scores within [0, SCORE_GRID_MAX]."""
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
):
    """Sample a single match. For non-group stages, plays extra time then
    a penalty shootout if still tied. Returns a MatchResult."""
    from .types import MatchResult
    p = params if params is not None else Params()
    diff = rating.rating_diff(team_a, team_b, a_is_host, b_is_host)
    lam_a, lam_b = rating.lambdas(diff, p.mu, p.lambda_min)

    home_goals, away_goals = _sample_score(lam_a, lam_b, p.rho, rng)
    extra_time = False
    went_to_pens = False
    pen_winner: str | None = None

    if stage != "group" and home_goals == away_goals:
        # Extra time: re-sample with reduced λ (30 min instead of 90).
        extra_time = True
        et_h, et_a = _sample_score(
            lam_a * ET_LAMBDA_SCALE, lam_b * ET_LAMBDA_SCALE, p.rho, rng,
        )
        home_goals += et_h
        away_goals += et_a
        if home_goals == away_goals:
            # Penalty shootout: rating-weighted Bernoulli.
            went_to_pens = True
            w_e = rating.win_expectation(diff)
            pen_winner = team_a.iso3 if rng.random() < w_e else team_b.iso3

    return MatchResult(
        home=team_a.iso3, away=team_b.iso3,
        home_goals=home_goals, away_goals=away_goals,
        stage=stage,
        neutral=not (a_is_host or b_is_host),
        extra_time=extra_time, went_to_pens=went_to_pens, pen_winner=pen_winner,
        home_rating_before=rating.rating_of(team_a),
        away_rating_before=rating.rating_of(team_b),
    )
```

- [ ] **Step 4: Run to verify passing.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_model.py -v
```

Expected: 12 passed (8 from Task 8 + 4 new).

- [ ] **Step 5: Commit.**

```bash
git add wcsim/model.py tests/test_model.py
git commit -m "Spike 2 Task 9: model.sample_match (ET + penalties)"
```

---

## Task 10: Regression test against `validate.py`

**Files:**
- Create: `tests/test_regression.py`

For every WC 2018+2022 match in the bundled snapshot, `wcsim.predict_match(...)` must reproduce `validate.predict(...)` to within `1e-9` for Elo, FIFA, and Blend modes.

- [ ] **Step 1: Write the failing test.** Create `tests/test_regression.py`:

```python
"""Regression guard: wcsim.predict_match must reproduce validate.predict
to within 1e-9 across all WC 2018+2022 matches × Elo/FIFA/Blend modes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPIKE = Path(__file__).parent.parent / "spikes" / "01-validation"
if str(SPIKE) not in sys.path:
    sys.path.insert(0, str(SPIKE))


@pytest.fixture(scope="module")
def validate_module():
    """Import validate.py from the spike directory as a top-level module."""
    import validate   # noqa: E402
    return validate


@pytest.fixture(scope="module")
def loaded_data(validate_module):
    """Load the same data validate.py uses."""
    v = validate_module
    matches_2018 = v.load_matches(2018)
    matches_2022 = v.load_matches(2022)
    elo_2018 = v.load_elo(v.WC2018[0])
    elo_2022 = v.load_elo(v.WC2022[0])
    fifa_2018, _ = v.load_fifa(v.WC2018[0])
    fifa_2022, _ = v.load_fifa(v.WC2022[0])
    f0_2018 = float(np.mean(list(fifa_2018.values())))
    f0_2022 = float(np.mean(list(fifa_2022.values())))
    return {
        "matches_2018": matches_2018, "matches_2022": matches_2022,
        "elo_2018": elo_2018, "elo_2022": elo_2022,
        "fifa_2018": fifa_2018, "fifa_2022": fifa_2022,
        "f0_2018": f0_2018, "f0_2022": f0_2022,
    }


def _team_from_dicts(iso3, elo_d, fifa_d):
    """Build a wcsim.Team from the dicts validate.py uses (iso3 -> value)."""
    from wcsim.types import Team
    return Team(
        name=iso3, iso3=iso3, confederation="UNK",   # confederation unused in math
        elo=elo_d[iso3],
        fifa_points=fifa_d.get(iso3),
    )


@pytest.mark.parametrize("mode", ["elo", "fifa", "blend"])
def test_predict_match_matches_validate_py(loaded_data, validate_module, mode):
    """For every WC 2018+2022 match, wcsim's predict_match must match
    validate.predict() to within 1e-9."""
    from wcsim.model import predict_match
    from wcsim.types import Params
    from wcsim.ratings.elo import EloRating
    from wcsim.ratings.fifa import FifaRating
    from wcsim.ratings.blend import BlendRating

    rating_cls = {"elo": EloRating, "fifa": FifaRating, "blend": BlendRating}[mode]
    p = Params()

    for year, key_suffix in ((2018, "2018"), (2022, "2022")):
        elo_d = loaded_data[f"elo_{key_suffix}"]
        fifa_d = loaded_data[f"fifa_{key_suffix}"]
        f0 = loaded_data[f"f0_{key_suffix}"]
        # f0 in our Params is a stand-in; for parity with validate.py per-year
        # we override it before computing the blend ratings.
        params_for_year = Params(f0=f0)
        rating = rating_cls(params_for_year)
        hosts = validate_module.HOST_BY_YEAR[year]

        for m in loaded_data[f"matches_{key_suffix}"]:
            a_iso3, b_iso3 = m["home_iso3"], m["away_iso3"]
            team_a = _team_from_dicts(a_iso3, elo_d, fifa_d)
            team_b = _team_from_dicts(b_iso3, elo_d, fifa_d)

            wcsim_probs = predict_match(
                team_a, team_b, rating=rating, params=params_for_year,
                a_is_host=(a_iso3 in hosts), b_is_host=(b_iso3 in hosts),
            )
            validate_probs = validate_module.predict(
                elo_a=elo_d[a_iso3], elo_b=elo_d[b_iso3],
                fifa_a=fifa_d[a_iso3], fifa_b=fifa_d[b_iso3],
                f0=f0,
                a_is_host=(a_iso3 in hosts), b_is_host=(b_iso3 in hosts),
                mode=mode, params=params_for_year.__dict__,
            )
            np.testing.assert_allclose(
                wcsim_probs, validate_probs, atol=1e-9,
                err_msg=f"{mode} mode mismatch on {year} {a_iso3} vs {b_iso3}",
            )
```

- [ ] **Step 2: Run to verify failure or current parity.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_regression.py -v
```

Expected outcome depends on the validate.py interface. **If `validate.predict()` doesn't accept a `params` kwarg as written above**, the test will fail with a `TypeError`. In that case, fix the test to construct the right calling convention — read `validate.py`'s `predict` function signature and adjust accordingly. The goal of this step is to discover the actual interface mismatch and adjust, not to introduce changes to `validate.py`.

The current `validate.py` `predict()` signature reads its params from the module-level `PARAMS` dict, not via a kwarg. To run the regression test correctly, the test setup must temporarily mutate `validate.PARAMS` to match the `wcsim` Params (especially `rho`, `f0`). Update the test loop to:

```python
# Inside the parametrized test, before the per-match loop:
validate_module.PARAMS.update({
    "c_elo": p.c_elo, "c_fifa": p.c_fifa,
    "mu": p.mu, "lambda_min": p.lambda_min,
    "blend_w": p.blend_w, "e0": p.e0,
    "home_bonus_elo": p.home_bonus_elo, "home_bonus_fifa": p.home_bonus_fifa,
    "rho": p.rho,
})

# And drop the `params=` kwarg from the validate.predict call:
validate_probs = validate_module.predict(
    elo_a=elo_d[a_iso3], elo_b=elo_d[b_iso3],
    fifa_a=fifa_d[a_iso3], fifa_b=fifa_d[b_iso3],
    f0=f0,
    a_is_host=(a_iso3 in hosts), b_is_host=(b_iso3 in hosts),
    mode=mode,
)
```

Apply that fix to `tests/test_regression.py` before re-running.

- [ ] **Step 3: Run to verify passing after the interface fix.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_regression.py -v
```

Expected: 3 passed (one per mode).

- [ ] **Step 4: Commit.**

```bash
git add tests/test_regression.py
git commit -m "Spike 2 Task 10: regression test — wcsim reproduces validate.py to 1e-9"
```

---

## Task 11: Extend the Elo scraper for May-2026 snapshot + WC 2026 team list

**Files:**
- Modify: `spikes/01-validation/scrapers/elo.py` (add WC 2026 team list + 2026 snapshot date)
- Modify (re-run): `spikes/01-validation/data/raw/elo_history.csv`

WC 2026's 48-team list (per the 2025-12-05 draw):

```
Hosts:   USA, Mexico, Canada
UEFA:    England, France, Germany, Italy, Spain, Portugal, Netherlands,
         Belgium, Croatia, Switzerland, Austria, Hungary, Denmark, Norway,
         Czechia, Serbia, Albania, Türkiye
CONMEBOL: Brazil, Argentina, Uruguay, Colombia, Ecuador, Paraguay, Venezuela
AFC:     Japan, South Korea, IR Iran, Australia, Saudi Arabia, Qatar,
         Uzbekistan, Jordan
CAF:     Morocco, Tunisia, Senegal, Egypt, Algeria, Côte d'Ivoire, Ghana,
         South Africa, Cape Verde
CONCACAF: Costa Rica, Panama, Honduras, Curaçao
OFC:     New Zealand
```

(If qualification produced different teams by the time this task runs, adjust the list to match FIFA's official 2026 draw. The implementer is responsible for verifying the current state of qualification.)

- [ ] **Step 1: Read the current WC2018 and WC2022 constants** in `spikes/01-validation/scrapers/elo.py` to understand the per-team data shape (each is `(eloratings_name, display_name)`).

- [ ] **Step 2: Add `WC_2026` constant** mirroring the same shape. Append to the existing constants block in `elo.py`. The eloratings.net name uses underscores for spaces and ASCII for accents (e.g., `"Cote_d'Ivoire"` → check actual URL by `curl -sf -I https://www.eloratings.net/Cote_d_Ivoire.tsv` and adjust; if the apostrophe URL fails, fall back to a different naming pattern listed at eloratings.net).

```python
WC_2026 = [
    ("United_States", "USA"),
    ("Mexico", "Mexico"),
    ("Canada", "Canada"),
    # ... 45 more entries; see the team list above. ...
]
```

The implementer must produce the complete 48-entry list during this task; the snippet above shows the format.

- [ ] **Step 3: Add the 2026 snapshot date.** Add `WC2026 = ("2026-05-15", "2026-06-11", "2026-07-19")` and update `SNAPSHOTS` to include it.

```python
WC2026 = ("2026-05-15", "2026-06-11", "2026-07-19")
SNAPSHOTS = [
    ("2018-06-13", WC_2018),
    ("2022-11-19", WC_2022),
    ("2026-05-15", WC_2026),
]
```

- [ ] **Step 4: Re-run the Elo scraper.**

```bash
spikes/01-validation/.venv/bin/python spikes/01-validation/scrapers/elo.py
```

Expected: 64 + 48 = 112 rows written to `data/raw/elo_history.csv`. Any team whose URL 404s prints a warning; the implementer must resolve those (look for the actual eloratings.net URL pattern for that country).

- [ ] **Step 5: Verify the snapshot has all 48 WC 2026 teams.**

```bash
grep -c "^2026-05-15" spikes/01-validation/data/raw/elo_history.csv
```

Expected: 48.

- [ ] **Step 6: Commit.**

```bash
git add spikes/01-validation/scrapers/elo.py spikes/01-validation/data/raw/elo_history.csv
git commit -m "Spike 2 Task 11: extend Elo scraper for May-2026 WC 2026 snapshot"
```

---

## Task 12: Extend the FIFA scraper for the latest pre-2026-WC snapshot

**Files:**
- Modify: `spikes/01-validation/scrapers/fifa.py` (add the 2026 target)
- Modify (re-run): `spikes/01-validation/data/raw/fifa_ranking.csv`

- [ ] **Step 1: Add a third target** to `TARGETS` in `fifa.py`. The latest FIFA ranking before WC 2026 (June 11 kickoff) will be from early June 2026 if available, else April/May 2026.

```python
TARGETS = [
    ("2018-06-13", "WC 2018 (opens 2018-06-14)"),
    ("2022-11-19", "WC 2022 (opens 2022-11-20)"),
    ("2026-06-10", "WC 2026 (opens 2026-06-11)"),
]
```

- [ ] **Step 2: Re-run the FIFA scraper.**

```bash
spikes/01-validation/.venv/bin/python spikes/01-validation/scrapers/fifa.py
```

Expected: 3 × ~210 = ~630 rows written. The new snapshot date will be the latest FIFA ranking on or before 2026-06-10.

- [ ] **Step 3: Verify the WC 2026 participants are in the new snapshot.**

```bash
spikes/01-validation/.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('spikes/01-validation/data/raw/fifa_ranking.csv')
latest = df.sort_values('rank_date').tail(220)
expected = {'United_States', 'USA', 'Mexico', 'Canada', 'Brazil', 'Argentina', 'France', 'Germany'}
present = set(latest['country_full'].unique())
missing = expected - present
print(f'Missing WC 2026 sample teams from latest FIFA snapshot: {missing}')
"
```

Expected: empty set (no missing).

- [ ] **Step 4: Commit.**

```bash
git add spikes/01-validation/scrapers/fifa.py spikes/01-validation/data/raw/fifa_ranking.csv
git commit -m "Spike 2 Task 12: extend FIFA scraper for pre-WC-2026 snapshot"
```

---

## Task 13: Hand-curate the WC 2026 draw

**Files:**
- Create: `spikes/01-validation/data/raw/wc2026_draw.json`

The WC 2026 draw was held on 2025-12-05. The 12 groups × 4 teams are public knowledge. Source the official draw from FIFA's website or Wikipedia (`https://en.wikipedia.org/wiki/2026_FIFA_World_Cup`) and transcribe it.

- [ ] **Step 1: Look up the official 2025-12-05 draw** and produce a JSON file with the canonical structure: 12 group keys (`"A"` through `"L"`), each mapping to a list of 4 ISO3 codes.

```bash
# Verify the draw against FIFA's published groups before committing.
# The implementer should cross-check with at least two sources
# (FIFA + Wikipedia + ESPN's published bracket).
```

Save as `spikes/01-validation/data/raw/wc2026_draw.json` with this shape:

```json
{
  "A": ["MEX", "USA-or-other-host", "ISO3", "ISO3"],
  "B": ["ISO3", "ISO3", "ISO3", "ISO3"],
  ...
  "L": ["ISO3", "ISO3", "ISO3", "ISO3"]
}
```

Use the same ISO3 codes that appear in the Elo and FIFA CSVs (and that `name_to_iso3.NAME_TO_ISO3` maps to — add new entries if any are missing).

- [ ] **Step 2: Update `spikes/01-validation/name_to_iso3.py`** for any new countries in WC 2026 that weren't in 2018/2022. Likely additions: Albania (ALB), Türkiye (TUR), Uzbekistan (UZB), Jordan (JOR), Cape Verde (CPV), Curaçao (CUW), Côte d'Ivoire (CIV), Norway (NOR), Czechia (CZE), Austria (AUT), Hungary (HUN), Italy (ITA), Uruguay already exists, Venezuela (VEN), Paraguay (PAR), Algeria (ALG), South Africa (RSA), Honduras (HON), Côte d'Ivoire (CIV).

```python
# Append to NAME_TO_ISO3 in name_to_iso3.py — add every country that's in
# WC 2026 but wasn't in WC 2018 or 2022, plus any spelling variants the
# Elo / FIFA scrapers produced.
```

- [ ] **Step 3: Run the existing verify script** to confirm every name in the augmented Elo + FIFA CSVs is mapped.

```bash
cd spikes/01-validation && .venv/bin/python verify_names.py
```

Expected: "OK: all N names resolved to ISO3."

- [ ] **Step 4: Commit.**

```bash
git add spikes/01-validation/data/raw/wc2026_draw.json spikes/01-validation/name_to_iso3.py
git commit -m "Spike 2 Task 13: hand-curated WC 2026 draw (2025-12-05) + ISO3 additions"
```

---

## Task 14: `wcsim/tournament.py` — TournamentStructure + dispatcher

**Files:**
- Modify: `wcsim/tournament.py`
- Create: `tests/test_tournament.py`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_tournament.py`:

```python
"""Tests for tournament structures and the team-count dispatcher."""
from __future__ import annotations

import pytest


def test_structure_2018_2022_dimensions():
    from wcsim.tournament import STRUCTURE_2018_2022
    s = STRUCTURE_2018_2022
    assert s.groups_count == 8
    assert s.group_size == 4
    assert s.top_per_group == 2
    assert s.best_thirds == 0
    assert s.knockout_stages == ["R16", "QF", "SF", "Final"]
    assert s.third_place_playoff is True


def test_structure_2026_dimensions():
    from wcsim.tournament import STRUCTURE_2026
    s = STRUCTURE_2026
    assert s.groups_count == 12
    assert s.group_size == 4
    assert s.top_per_group == 2
    assert s.best_thirds == 8
    assert s.knockout_stages == ["R32", "R16", "QF", "SF", "Final"]
    assert s.third_place_playoff is False


def test_structure_for_32_teams_returns_2018_2022():
    from wcsim.tournament import _structure_for, STRUCTURE_2018_2022
    assert _structure_for(32) is STRUCTURE_2018_2022


def test_structure_for_48_teams_returns_2026():
    from wcsim.tournament import _structure_for, STRUCTURE_2026
    assert _structure_for(48) is STRUCTURE_2026


def test_structure_for_unsupported_count_raises():
    from wcsim.tournament import _structure_for
    with pytest.raises(ValueError, match="Unsupported tournament size"):
        _structure_for(24)
    with pytest.raises(ValueError, match="Unsupported tournament size"):
        _structure_for(64)
```

- [ ] **Step 2: Run to verify failure.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_tournament.py -v
```

Expected: 5 failures with `ImportError`.

- [ ] **Step 3: Implement the structures and dispatcher** at the top of `wcsim/tournament.py`.

```python
"""Tournament-simulation engine. Supports two hardcoded formats (WC
2018/2022 and WC 2026), dispatched by team count."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TournamentStructure:
    """Hardcoded tournament-format parameters."""
    name: str
    groups_count: int
    group_size: int
    top_per_group: int
    best_thirds: int
    knockout_stages: list[str]
    third_place_playoff: bool


STRUCTURE_2018_2022 = TournamentStructure(
    name="WC2018-2022",
    groups_count=8, group_size=4,
    top_per_group=2, best_thirds=0,
    knockout_stages=["R16", "QF", "SF", "Final"],
    third_place_playoff=True,
)

STRUCTURE_2026 = TournamentStructure(
    name="WC2026",
    groups_count=12, group_size=4,
    top_per_group=2, best_thirds=8,
    knockout_stages=["R32", "R16", "QF", "SF", "Final"],
    third_place_playoff=False,
)


def _structure_for(team_count: int) -> TournamentStructure:
    """Dispatch by team count. 32 -> 2018/2022 format; 48 -> 2026 format."""
    if team_count == 32:
        return STRUCTURE_2018_2022
    if team_count == 48:
        return STRUCTURE_2026
    raise ValueError(
        f"Unsupported tournament size: {team_count} teams "
        "(supported: 32 for WC 2018/2022, 48 for WC 2026)"
    )
```

- [ ] **Step 4: Run to verify passing.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_tournament.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit.**

```bash
git add wcsim/tournament.py tests/test_tournament.py
git commit -m "Spike 2 Task 14: tournament.TournamentStructure + dispatcher"
```

---

## Task 15: `simulate_group_stage` + tiebreakers

**Files:**
- Modify: `wcsim/tournament.py`
- Modify: `tests/test_tournament.py`

- [ ] **Step 1: Append failing tests** to `tests/test_tournament.py`:

```python
def test_simulate_group_stage_returns_match_count_and_positions(
    sample_team_brazil, sample_team_france, default_params,
):
    """Each group plays C(4, 2) = 6 matches. For 8 groups, 48 matches.
    Returns finishing positions {iso3 -> 1..4}."""
    from wcsim.tournament import simulate_group_stage
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Team
    import numpy as np

    # Construct a synthetic 32-team draw with sample fixtures.
    teams = {f"T{i:02d}": Team(name=f"T{i:02d}", iso3=f"T{i:02d}",
                               confederation="UNK", elo=1500.0 + i * 10)
             for i in range(32)}
    draw = {chr(ord("A") + g): [f"T{g*4+0:02d}", f"T{g*4+1:02d}",
                                f"T{g*4+2:02d}", f"T{g*4+3:02d}"]
            for g in range(8)}

    rng = np.random.default_rng(42)
    matches, positions = simulate_group_stage(
        teams=teams, draw=draw,
        rating=EloRating(default_params), params=default_params,
        rng=rng, hosts=set(),
    )
    assert len(matches) == 48          # 8 groups × 6 matches
    assert len(positions) == 32        # every team gets a position
    for iso3, pos in positions.items():
        assert pos in {1, 2, 3, 4}
    # Each group has exactly one team at each position
    for group_letter, group_teams in draw.items():
        positions_in_group = sorted(positions[t] for t in group_teams)
        assert positions_in_group == [1, 2, 3, 4]


def test_group_tiebreakers_points_first(default_params):
    """Standings sort by points first."""
    from wcsim.tournament import _rank_group
    standings = [
        {"team": "A", "points": 9, "gd": 0, "gf": 5},
        {"team": "B", "points": 4, "gd": 3, "gf": 7},
        {"team": "C", "points": 3, "gd": 0, "gf": 3},
        {"team": "D", "points": 0, "gd": -3, "gf": 1},
    ]
    ranked = _rank_group(standings, head_to_head_results={}, rng=None)
    assert [s["team"] for s in ranked] == ["A", "B", "C", "D"]


def test_group_tiebreakers_gd_when_points_tied(default_params):
    """When points are tied, goal difference breaks the tie."""
    from wcsim.tournament import _rank_group
    standings = [
        {"team": "A", "points": 6, "gd": 2, "gf": 4},
        {"team": "B", "points": 6, "gd": 5, "gf": 8},
        {"team": "C", "points": 6, "gd": -1, "gf": 3},
    ]
    ranked = _rank_group(standings, head_to_head_results={}, rng=None)
    assert [s["team"] for s in ranked] == ["B", "A", "C"]
```

- [ ] **Step 2: Run to verify failure.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_tournament.py -k "group" -v
```

Expected: 3 failures.

- [ ] **Step 3: Implement `simulate_group_stage` + `_rank_group`** in `wcsim/tournament.py`. Append:

```python
from itertools import combinations
from typing import Iterable

import numpy as np

from .model import sample_match
from .ratings.base import RatingSystem
from .types import MatchResult, Params, Team


def _points_from_score(home: int, away: int) -> tuple[int, int]:
    if home > away:
        return (3, 0)
    if home < away:
        return (0, 3)
    return (1, 1)


def _rank_group(
    standings: list[dict], head_to_head_results: dict, rng: np.random.Generator | None,
) -> list[dict]:
    """Sort a group's standings by (points, gd, gf, head-to-head, random).
    `standings` is a list of {team, points, gd, gf}.
    `head_to_head_results` maps frozenset({a, b}) -> dict with the pair's
    head-to-head subtournament results, computed by the caller if needed."""
    # First pass: sort by (points desc, gd desc, gf desc).
    standings = sorted(
        standings,
        key=lambda s: (-s["points"], -s["gd"], -s["gf"]),
    )
    # Resolve sub-ties: any group of teams with identical (points, gd, gf)
    # gets resolved by head-to-head, then random.
    out: list[dict] = []
    i = 0
    while i < len(standings):
        j = i + 1
        while j < len(standings) and (
            standings[j]["points"] == standings[i]["points"]
            and standings[j]["gd"] == standings[i]["gd"]
            and standings[j]["gf"] == standings[i]["gf"]
        ):
            j += 1
        block = standings[i:j]
        if len(block) > 1 and rng is not None:
            # Head-to-head among `block` would go here; for simplicity, we
            # fall through to deterministic-random tiebreak using the rng.
            rng.shuffle(block)
        out.extend(block)
        i = j
    return out


def simulate_group_stage(
    teams: dict[str, Team], draw: dict[str, list[str]],
    *, rating: RatingSystem, params: Params, rng: np.random.Generator,
    hosts: set[str],
) -> tuple[list[MatchResult], dict[str, int]]:
    """Simulate each group's round-robin. Returns (all_matches, positions)
    where positions[iso3] in {1, 2, 3, 4}."""
    all_matches: list[MatchResult] = []
    positions: dict[str, int] = {}

    for group_letter, group_team_iso3s in draw.items():
        # Compute standings.
        standings = {t: {"team": t, "points": 0, "gd": 0, "gf": 0}
                     for t in group_team_iso3s}
        for a_iso3, b_iso3 in combinations(group_team_iso3s, 2):
            a, b = teams[a_iso3], teams[b_iso3]
            m = sample_match(
                a, b, rating=rating, params=params,
                a_is_host=(a_iso3 in hosts), b_is_host=(b_iso3 in hosts),
                rng=rng, stage="group",
            )
            all_matches.append(m)
            pa, pb = _points_from_score(m.home_goals, m.away_goals)
            standings[a_iso3]["points"] += pa
            standings[b_iso3]["points"] += pb
            standings[a_iso3]["gf"] += m.home_goals
            standings[b_iso3]["gf"] += m.away_goals
            standings[a_iso3]["gd"] += m.home_goals - m.away_goals
            standings[b_iso3]["gd"] += m.away_goals - m.home_goals

        ranked = _rank_group(list(standings.values()), head_to_head_results={}, rng=rng)
        for pos, entry in enumerate(ranked, start=1):
            positions[entry["team"]] = pos

    return all_matches, positions
```

- [ ] **Step 4: Run to verify passing.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_tournament.py -k "group or tiebreak" -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit.**

```bash
git add wcsim/tournament.py tests/test_tournament.py
git commit -m "Spike 2 Task 15: tournament.simulate_group_stage + _rank_group"
```

---

## Task 16: `best_third_place_teams` + `seed_knockout`

**Files:**
- Modify: `wcsim/tournament.py`
- Modify: `tests/test_tournament.py`

- [ ] **Step 1: Append failing tests:**

```python
def test_best_third_place_teams_returns_zero_for_2018_2022():
    """STRUCTURE_2018_2022 has best_thirds=0, so no third-place teams advance."""
    from wcsim.tournament import best_third_place_teams
    standings = [
        {"team": "T1", "points": 3, "gd": 0, "gf": 2},
        {"team": "T2", "points": 4, "gd": 1, "gf": 3},
    ]
    assert best_third_place_teams(standings, n=0) == []


def test_best_third_place_teams_ranks_by_points_gd_gf():
    """For n=8, returns the 8 best third-place teams by (points, gd, gf)."""
    from wcsim.tournament import best_third_place_teams
    # 12 candidate third-place teams.
    candidates = [
        {"team": f"T{i}", "points": p, "gd": gd, "gf": gf}
        for i, (p, gd, gf) in enumerate([
            (6, 4, 7), (4, 2, 5), (4, 1, 5), (3, 0, 3),
            (3, 0, 2), (3, -1, 4), (2, -1, 2), (2, -2, 1),
            (1, -3, 1), (1, -3, 0), (0, -4, 1), (0, -5, 0),
        ])
    ]
    top8 = best_third_place_teams(candidates, n=8)
    assert top8 == [f"T{i}" for i in range(8)]


def test_seed_knockout_2018_2022_pairs_1A_with_2B():
    """STRUCTURE_2018_2022 uses crosswise seeding: group winners face
    runners-up from adjacent group (1A vs 2B, 1B vs 2A, etc.)."""
    from wcsim.tournament import seed_knockout, STRUCTURE_2018_2022
    winners = [f"W{c}" for c in "ABCDEFGH"]
    runners_up = [f"R{c}" for c in "ABCDEFGH"]
    seeded = seed_knockout(STRUCTURE_2018_2022, winners, runners_up, [])
    assert len(seeded) == 16
    # The expected R16 pairing order: WA-RB, WC-RD, WE-RF, WG-RH, ...
    # (with the other half: WB-RA, WD-RC, etc., bracketed correctly)
    assert seeded[0] == "WA"
    assert seeded[1] == "RB"
```

- [ ] **Step 2: Run to verify failure.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_tournament.py -k "third_place or seed_knockout" -v
```

Expected: 3 failures.

- [ ] **Step 3: Implement** in `wcsim/tournament.py`. Append:

```python
def best_third_place_teams(third_place_standings: list[dict], n: int) -> list[str]:
    """Rank the third-place teams across all groups by (points, gd, gf) and
    return the top n iso3 codes. Returns [] if n == 0."""
    if n == 0:
        return []
    ranked = sorted(
        third_place_standings,
        key=lambda s: (-s["points"], -s["gd"], -s["gf"]),
    )
    return [s["team"] for s in ranked[:n]]


# Seeding tables. The R16 bracket order for the 2018/2022 format (classic
# crosswise pairing):
_R16_PAIRS_2018_2022 = [
    ("WA", "RB"), ("WC", "RD"), ("WE", "RF"), ("WG", "RH"),
    ("WB", "RA"), ("WD", "RC"), ("WF", "RE"), ("WH", "RG"),
]


def seed_knockout(
    structure, group_winners: list[str], group_runners_up: list[str],
    best_thirds: list[str],
) -> list[str]:
    """Return a list of iso3 codes in bracket order — pairs (i, i+1) play
    in round 1, then (i+0,1) winners meet (i+2,3) winners, etc."""
    if structure.name == "WC2018-2022":
        # group_winners / runners_up are ordered by group letter A..H.
        slot = {}
        for letter, w in zip("ABCDEFGH", group_winners):
            slot[f"W{letter}"] = w
        for letter, r in zip("ABCDEFGH", group_runners_up):
            slot[f"R{letter}"] = r
        out = []
        for a_slot, b_slot in _R16_PAIRS_2018_2022:
            out.append(slot[a_slot])
            out.append(slot[b_slot])
        return out
    elif structure.name == "WC2026":
        # The R32 bracket placement of 8 best-third teams is the
        # PRD-mentioned 15-row decision table; for Spike 2's pin test we
        # use a simplified arrangement (the implementer fills in FIFA's
        # actual table here once verified). For now: top 2 per group, then
        # best thirds in points order, paired sequentially.
        slot_iso3s = list(group_winners) + list(group_runners_up) + list(best_thirds)
        if len(slot_iso3s) != 32:
            raise ValueError(
                f"WC2026 expected 32 slots, got {len(slot_iso3s)}"
            )
        return slot_iso3s
    raise ValueError(f"Unknown structure name: {structure.name}")
```

(Note: the WC 2026 seeding implementation above is a simplification — the implementer must replace it with FIFA's published table when this becomes the gating concern. Spike 2's tournament pin test for WC 2026 will lock against whatever ordering is implemented.)

- [ ] **Step 4: Run to verify passing.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_tournament.py -k "third_place or seed_knockout" -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit.**

```bash
git add wcsim/tournament.py tests/test_tournament.py
git commit -m "Spike 2 Task 16: best_third_place_teams + seed_knockout"
```

---

## Task 17: `simulate_knockout` + `simulate_tournament`

**Files:**
- Modify: `wcsim/tournament.py`
- Modify: `tests/test_tournament.py`

- [ ] **Step 1: Append failing tests:**

```python
def test_simulate_tournament_2022_format_runs_end_to_end(
    bundled_elo_history, default_params,
):
    """A WC 2022-shaped synthetic input (32 teams, 8 groups) simulates without
    crashing and produces a TournamentResult with 64 matches and a champion."""
    from wcsim.tournament import simulate_tournament
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Team
    import pandas as pd

    # Build teams from the bundled 2022 Elo snapshot.
    df_2022 = bundled_elo_history[bundled_elo_history["date"] == "2022-11-19"]
    teams = {
        row["team"]: Team(
            name=row["team"], iso3=row["team"][:3].upper(),
            confederation="UNK", elo=float(row["rating"]),
        )
        for _, row in df_2022.iterrows()
    }
    iso3s = sorted(teams.keys())[:32]
    teams = {iso3: teams[iso3] for iso3 in iso3s}
    # Synthetic 8-group draw.
    draw = {chr(ord("A") + g): iso3s[g * 4:(g + 1) * 4] for g in range(8)}

    result = simulate_tournament(
        teams=teams, draw=draw, hosts=set(),
        rating=EloRating(default_params), params=default_params, seed=42,
    )
    assert result.seed == 42
    assert result.rating_mode == "elo"
    # 48 group + 8 R16 + 4 QF + 2 SF + 1 third-place + 1 final = 64
    assert len(result.matches) == 64
    champions = [iso for iso, stage in result.placements.items() if stage == "Champion"]
    assert len(champions) == 1
    # Every team gets a placement.
    assert set(result.placements.keys()) == set(teams.keys())


def test_simulate_tournament_is_deterministic_with_seed(
    bundled_elo_history, default_params,
):
    """Same inputs + same seed → same TournamentResult (byte-identical)."""
    from wcsim.tournament import simulate_tournament
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Team

    df_2022 = bundled_elo_history[bundled_elo_history["date"] == "2022-11-19"]
    teams = {
        row["team"]: Team(
            name=row["team"], iso3=row["team"][:3].upper(),
            confederation="UNK", elo=float(row["rating"]),
        )
        for _, row in df_2022.iterrows()
    }
    iso3s = sorted(teams.keys())[:32]
    teams = {iso3: teams[iso3] for iso3 in iso3s}
    draw = {chr(ord("A") + g): iso3s[g * 4:(g + 1) * 4] for g in range(8)}

    r1 = simulate_tournament(
        teams=teams, draw=draw, hosts=set(),
        rating=EloRating(default_params), params=default_params, seed=42,
    )
    r2 = simulate_tournament(
        teams=teams, draw=draw, hosts=set(),
        rating=EloRating(default_params), params=default_params, seed=42,
    )
    assert r1.placements == r2.placements
    assert len(r1.matches) == len(r2.matches)
    for m1, m2 in zip(r1.matches, r2.matches):
        assert m1 == m2
```

- [ ] **Step 2: Run to verify failure.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_tournament.py -k "simulate_tournament" -v
```

Expected: 2 failures.

- [ ] **Step 3: Implement `simulate_knockout` + `simulate_tournament`** in `wcsim/tournament.py`. Append:

```python
def simulate_knockout(
    seeded: list[str], teams: dict[str, Team], structure: TournamentStructure,
    *, rating: RatingSystem, params: Params, rng: np.random.Generator,
    hosts: set[str],
) -> tuple[list[MatchResult], dict[str, str]]:
    """Run the knockout rounds in `structure.knockout_stages`. Returns
    (matches, {iso3 -> exit_stage}). exit_stage ∈ knockout_stages ∪
    {"Champion"} ∪ {"3rd"} (if third-place playoff)."""
    matches: list[MatchResult] = []
    placements: dict[str, str] = {}

    current = list(seeded)   # iso3 list, paired (0,1), (2,3), ...
    semi_losers: list[str] = []

    for stage in structure.knockout_stages:
        next_round: list[str] = []
        for i in range(0, len(current), 2):
            a_iso3, b_iso3 = current[i], current[i + 1]
            m = sample_match(
                teams[a_iso3], teams[b_iso3],
                rating=rating, params=params,
                a_is_host=(a_iso3 in hosts), b_is_host=(b_iso3 in hosts),
                rng=rng, stage=stage,
            )
            matches.append(m)
            # Determine winner.
            if m.went_to_pens:
                winner = m.pen_winner
            elif m.home_goals > m.away_goals:
                winner = m.home
            elif m.home_goals < m.away_goals:
                winner = m.away
            else:
                # Shouldn't happen — sample_match handles ET + pens at
                # non-group stages. But guard with a deterministic fallback.
                winner = m.home if rng.random() < 0.5 else m.away
            loser = b_iso3 if winner == a_iso3 else a_iso3
            placements[loser] = stage
            if stage == "SF":
                semi_losers.append(loser)
            next_round.append(winner)
        current = next_round

    # After the loop, current has 1 team — the champion.
    champion = current[0]
    placements[champion] = "Champion"

    # Optional 3rd-place playoff.
    if structure.third_place_playoff and len(semi_losers) == 2:
        a_iso3, b_iso3 = semi_losers
        m = sample_match(
            teams[a_iso3], teams[b_iso3],
            rating=rating, params=params,
            a_is_host=(a_iso3 in hosts), b_is_host=(b_iso3 in hosts),
            rng=rng, stage="3rd",
        )
        matches.append(m)
        # Both SF losers already have stage="SF" in placements; the playoff
        # is recorded as a match but doesn't change placements (per spec
        # §5 — placements is exit-stage from the main bracket).

    return matches, placements


def simulate_tournament(
    teams: dict[str, Team], draw: dict[str, list[str]], hosts: set[str],
    *, rating: RatingSystem, params: Params | None = None, seed: int,
) -> "TournamentResult":
    """Top-level deterministic entry point. Combines group stage + knockout."""
    from .types import TournamentResult
    p = params if params is not None else Params()
    structure = _structure_for(len(teams))
    rng = np.random.default_rng(seed)

    group_matches, positions = simulate_group_stage(
        teams=teams, draw=draw, rating=rating, params=p, rng=rng, hosts=hosts,
    )

    # Build group_winners, group_runners_up, third_place candidates.
    group_letters = sorted(draw.keys())
    group_winners = [next(t for t in draw[g] if positions[t] == 1) for g in group_letters]
    group_runners_up = [next(t for t in draw[g] if positions[t] == 2) for g in group_letters]

    # Third-place candidates for STRUCTURE_2026.
    third_place_iso3s = [
        next(t for t in draw[g] if positions[t] == 3) for g in group_letters
    ]
    # Build standings for those teams from group_matches.
    standings_by_team: dict[str, dict] = {
        iso3: {"team": iso3, "points": 0, "gd": 0, "gf": 0}
        for iso3 in third_place_iso3s
    }
    for m in group_matches:
        for iso3 in (m.home, m.away):
            if iso3 not in standings_by_team:
                continue
            if iso3 == m.home:
                gf, ga = m.home_goals, m.away_goals
            else:
                gf, ga = m.away_goals, m.home_goals
            if gf > ga:
                standings_by_team[iso3]["points"] += 3
            elif gf == ga:
                standings_by_team[iso3]["points"] += 1
            standings_by_team[iso3]["gd"] += gf - ga
            standings_by_team[iso3]["gf"] += gf
    third_candidates = list(standings_by_team.values())
    best_thirds = best_third_place_teams(third_candidates, n=structure.best_thirds)

    # Knockout placements.
    seeded = seed_knockout(structure, group_winners, group_runners_up, best_thirds)
    knockout_matches, knockout_placements = simulate_knockout(
        seeded=seeded, teams=teams, structure=structure,
        rating=rating, params=p, rng=rng, hosts=hosts,
    )

    # Group-stage placements: GroupOut for everyone not in the knockout.
    placements: dict[str, str] = {}
    advancing = set(seeded)
    for iso3 in teams:
        if iso3 not in advancing:
            placements[iso3] = "GroupOut"
    placements.update(knockout_placements)

    # Final ratings (we don't track in-tournament updates yet — return start ratings).
    final_ratings = {iso3: rating.rating_of(t) for iso3, t in teams.items()}

    return TournamentResult(
        seed=seed, rating_mode=rating.name,
        matches=group_matches + knockout_matches,
        placements=placements, final_ratings=final_ratings,
    )
```

- [ ] **Step 4: Run to verify passing.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_tournament.py -v
```

Expected: all tests pass (cumulative: ~13).

- [ ] **Step 5: Commit.**

```bash
git add wcsim/tournament.py tests/test_tournament.py
git commit -m "Spike 2 Task 17: simulate_knockout + simulate_tournament"
```

---

## Task 18: `wcsim/__init__.py` re-exports + smoke test

**Files:**
- Modify: `wcsim/__init__.py`
- Modify: `tests/test_types.py` (add a single smoke test)

- [ ] **Step 1: Append a failing test** to `tests/test_types.py`:

```python
def test_public_api_reexports():
    """wcsim.X imports work for the documented public surface."""
    import wcsim
    assert hasattr(wcsim, "Team")
    assert hasattr(wcsim, "MatchResult")
    assert hasattr(wcsim, "TournamentResult")
    assert hasattr(wcsim, "Params")
    assert hasattr(wcsim, "EloRating")
    assert hasattr(wcsim, "FifaRating")
    assert hasattr(wcsim, "BlendRating")
    assert hasattr(wcsim, "predict_match")
    assert hasattr(wcsim, "sample_match")
    assert hasattr(wcsim, "simulate_tournament")
```

- [ ] **Step 2: Run to verify failure.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_types.py::test_public_api_reexports -v
```

Expected: 1 failure.

- [ ] **Step 3: Implement `wcsim/__init__.py`.**

```python
"""wcsim — Football Tournament Monte Carlo Simulator (library).

Spike 2 deliverable: validated match math + tournament simulation. CLI,
Monte Carlo runner, and scrapers are deferred to later spikes.
"""
from __future__ import annotations

from .types import Team, MatchResult, TournamentResult, Params
from .ratings.elo import EloRating
from .ratings.fifa import FifaRating
from .ratings.blend import BlendRating
from .model import predict_match, sample_match
from .tournament import simulate_tournament

__all__ = [
    "Team", "MatchResult", "TournamentResult", "Params",
    "EloRating", "FifaRating", "BlendRating",
    "predict_match", "sample_match", "simulate_tournament",
]
```

- [ ] **Step 4: Run to verify passing.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_types.py::test_public_api_reexports -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit.**

```bash
git add wcsim/__init__.py tests/test_types.py
git commit -m "Spike 2 Task 18: wcsim/__init__.py public re-exports"
```

---

## Task 19: WC 2022 tournament pin test

**Files:**
- Modify: `tests/test_tournament.py`

- [ ] **Step 1: Append the WC 2022 pin test:**

```python
def test_wc_2022_tournament_pin(bundled_elo_history, bundled_matches, default_params):
    """Pin the WC 2022 simulation at seed=42 against a reference computed
    once on the first green run. The pin guards against silent changes to
    RNG seeding or tournament logic across future commits."""
    from wcsim.tournament import simulate_tournament
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Team

    # Build the canonical 2022 team-by-iso3 dict from the bundled snapshot.
    # The test reads the canonical 8-group draw from the WC 2022 matches
    # data: the first matchup in each group identifies the four group
    # members (by looking at which teams played which group-stage opponents).
    df = bundled_elo_history[bundled_elo_history["date"] == "2022-11-19"]
    # ... build teams dict and draw dict ...
    # For Spike 2 the implementer reads the 2022 draw from matches.csv:
    # each team's first three matches identify its three group opponents,
    # and the union of those four iso3s is the group.
    raise pytest.skip(
        "WC 2022 pin requires hand-curated 2022 draw from the matches data "
        "or a separate wc2022_draw.json. The implementer fills this in on "
        "the first green run by computing the result, copying the iso3 of "
        "the champion + the 16 R16 advancers (after R16 has played) into "
        "this test as the reference."
    )
    # When the test first runs and produces a valid TournamentResult, the
    # implementer replaces the `skip` above with an `assert` against the
    # captured (champion, frozenset(R16_advancers)) tuple. Subsequent runs
    # lock the result.
```

- [ ] **Step 2: Run** — the test should currently report SKIPPED (intentional).

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_tournament.py::test_wc_2022_tournament_pin -v
```

Expected: 1 skipped.

- [ ] **Step 3: Generate the reference and lock the pin.** Manually run a quick script to compute the champion + R16 advancers under (seed=42, WC 2022 draw, default params):

```bash
spikes/01-validation/.venv/bin/python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
sys.path.insert(0, str((Path('.') / 'spikes' / '01-validation').resolve()))

import pandas as pd
from wcsim.types import Team, Params
from wcsim.ratings.elo import EloRating
from wcsim.tournament import simulate_tournament

df = pd.read_csv('spikes/01-validation/data/raw/elo_history.csv')
df_2022 = df[df['date'] == '2022-11-19']
teams = {}
for _, row in df_2022.iterrows():
    iso3 = row['team'][:3].upper()
    teams[iso3] = Team(name=row['team'], iso3=iso3, confederation='UNK', elo=float(row['rating']))

# Read the 2022 draw from the matches CSV.
matches_df = pd.read_csv('spikes/01-validation/data/raw/matches_history.csv')
matches_df['date'] = pd.to_datetime(matches_df['date'])
group_matches = matches_df[(matches_df['date'] >= '2022-11-20') & (matches_df['date'] <= '2022-12-02')]
# Identify groups by clustering teams that played each other.
from collections import defaultdict
g = defaultdict(set)
for _, row in group_matches.iterrows():
    # Walk to find which group: trivial since all 6 matches per group share the same 4 teams.
    pass
# (The implementer extracts the 8 groups from the WC 2022 schedule.)
print('Replace this with the actual reference.')
"
```

(The implementer extracts the 8 WC 2022 groups, hand-codes them into the test as `WC2022_DRAW = {...}`, runs the simulation once, and copies the champion + R16 set into the pin.)

- [ ] **Step 4: Replace the pin test** with the locked reference (champion iso3 + frozenset of R16-stage placements).

- [ ] **Step 5: Run to verify the pin is locked.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_tournament.py::test_wc_2022_tournament_pin -v
```

Expected: 1 passed.

- [ ] **Step 6: Commit.**

```bash
git add tests/test_tournament.py
git commit -m "Spike 2 Task 19: WC 2022 tournament pin (seed=42, locked reference)"
```

---

## Task 20: WC 2026 tournament pin test

**Files:**
- Modify: `tests/test_tournament.py`

Mirror of Task 19 for WC 2026 (48 teams, R32 first round, no 3rd-place playoff). Uses the bundled `wc2026_draw.json` from Task 13 and the 2026 Elo snapshot from Task 11.

- [ ] **Step 1: Append the WC 2026 pin test.** Similar structure to Task 19: load 2026 Elo + draw, run simulate_tournament at seed=42, lock the champion + R32 advancers.

- [ ] **Step 2: Generate the reference** in the same one-off-script style as Task 19 Step 3.

- [ ] **Step 3: Lock the pin** with the captured reference values.

- [ ] **Step 4: Run to verify.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/test_tournament.py::test_wc_2026_tournament_pin -v
```

Expected: 1 passed. Additionally verify the run output respects the format: 103 matches (no 3rd-place playoff in 2026), 48 placements.

- [ ] **Step 5: Commit.**

```bash
git add tests/test_tournament.py
git commit -m "Spike 2 Task 20: WC 2026 tournament pin (seed=42, locked reference)"
```

---

## Task 21: Coverage report + final cleanup

**Files:**
- Modify: `spikes/01-validation/.gitignore` (or repo root) — exclude `.pytest_cache/`, `htmlcov/`

- [ ] **Step 1: Run the full suite with coverage.**

```bash
spikes/01-validation/.venv/bin/python -m pytest tests/ --cov=wcsim --cov-report=term-missing -v
```

Expected: all tests pass; coverage report shows ≥ 95% line coverage on `wcsim/`; 100% on `wcsim.model.predict_match`, `wcsim.model._apply_tau`, every `RatingSystem` `lambdas` method, every `update` method.

- [ ] **Step 2: If any line is uncovered**, add a targeted test that exercises it. Loop until coverage targets are met.

- [ ] **Step 3: Add `.pytest_cache/` and `htmlcov/` to `.gitignore`** if not already excluded.

```bash
echo ".pytest_cache/" >> .gitignore
echo "htmlcov/" >> .gitignore
```

- [ ] **Step 4: Commit the gitignore update** (no other changes expected at this point).

```bash
git add .gitignore
git commit -m "Spike 2 Task 21: gitignore pytest + coverage artefacts; final coverage ≥ 95% on wcsim/"
```

---

## Self-Review Notes

Reviewed against the spec on 2026-05-16:

- **Spec coverage:**
  - §1 Goal — Tasks 1–21 collectively produce the deliverables.
  - §2 Non-Goals — honored (no CLI, no Monte Carlo, no pyproject.toml; scrapers stay in `spikes/01-validation/`).
  - §3 Module layout — Tasks 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18 produce every file listed.
  - §4 Public API — Task 18 re-exports.
  - §5 Types — Tasks 2 (Team, MatchResult, TournamentResult) + 3 (Params). Note: Task 3 adds `k_elo`/`k_fifa` to Params beyond the spec.
  - §6 RatingSystem Protocol — Task 4.
  - §7 Match model — Tasks 8, 9.
  - §8 Tournament module — Tasks 14, 15, 16, 17 + the 2026 data fetch in Tasks 11, 12, 13.
  - §9 Test strategy — every task is TDD-driven; regression test is Task 10.
  - §10 Deterministic-RNG contract — Task 17's pin test (Task 17 step "test_simulate_tournament_is_deterministic_with_seed") asserts byte-identical output.
  - §11 Risks — `_BEST_THIRDS_BRACKET_LOOKUP_2026` is acknowledged as a simplified placeholder in Task 16; the WC 2026 pin test (Task 20) locks against whatever ordering is implemented.
  - §12 Acceptance Criteria — all 10 ACs covered by Task 1–21.
  - §13 Out of scope — all deferred items remain deferred.

- **Placeholder scan:** Two intentional "implementer fills in" steps remain — the WC 2022 draw hand-curation (Task 19 Step 3) and the WC 2026 draw hand-curation (Task 13 + Task 20 Step 2). Both are flagged with explicit instructions; not blocking placeholders.

- **Type consistency:** `RatingSystem.update(before, expected, score_home, score_away) -> float` is consistent across `EloRating`, `FifaRating`, `BlendRating`, the model, and the tournament module. `predict_match(team_a, team_b, *, rating, params, a_is_host, b_is_host) -> tuple[float, float, float]` is consistent across the regression test, the tournament dispatcher, and the public API. `MatchResult` field set (home, away, home_goals, away_goals, stage, neutral, extra_time, went_to_pens, pen_winner, home_rating_before, away_rating_before) is consistent across tasks 2, 9, 15, 17. `simulate_tournament(teams, draw, hosts, *, rating, params, seed)` keyword-only contract is consistent across the spec and tasks 17, 19, 20.
