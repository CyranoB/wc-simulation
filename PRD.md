# PRD: `wcsim` — Football Tournament Monte Carlo Simulator CLI

**Author:** Research Agent
**Status:** Draft v1.5
**Last updated:** 2026-05-15

**Changelog v1.5 (second-round peer-review responses):**
- **Host bonus is now rating-mode-aware.** §5.5 specifies the rescaling: $H_{\text{elo}} = H_{\text{blend}} =$ `--home-bonus` (default 100); $H_{\text{fifa}} =$ `--home-bonus` $\cdot\ S_{\text{fifa}}/S_{\text{elo}}$ (default 150). The rescaling preserves the win-probability bump across modes.
- **Blend's $S$ stated explicitly.** §5.5 now says $S = 400$ for blend (because blend ratings are normalised to Elo space). Blend's $c$ was already specified as `--c-elo`.
- **`wcsim_version` moved from `meta.deterministic` to `meta.environment`.** The cache-invalidation logic in §5.4 still keys on it, but output reproducibility no longer depends on the build string — so an upgrade no longer breaks byte-identical reproducibility by definition.
- **Elo source clarified: `eloratings.net`** is the correct upstream for international-team Elo (an earlier draft incorrectly named `clubelo.com`, which covers clubs only). §7 and §11 updated. Spike 1 uses the same source via a bundled snapshot, so cross-source re-validation risk remains eliminated.

**Changelog v1.4 (peer-review responses):**
- Redefined `wcsim bracket` as the **per-slot most-likely matchup** walked top-down from the modal champion, *not* "the modal full bracket" (which is essentially never repeated across 100k sims). The cache now stores a per-slot matchup frequency table alongside the per-team aggregates.
- Added §5.2.1 with `wcsim match`-specific flags: `--home {A,B,none}` (default `A`) and its alias `--neutral`, resolving the user-story-vs-AC1 flag mismatch.
- Extended `--pens` enum with `blend`; default still tracks `--rating`, so `--rating blend` no longer falls off the enum.
- Narrowed v1 *Tournament structure* scope to a concrete JSON schema: group count/size + simple top-N + best-K-third-place advancement, with `"seeding": "1A_vs_2B"` or `"wc2026"`. Arbitrary custom third-place permutations are deferred to v2.
- `Team` dataclass marks FIFA fields *optional* so the data model agrees with §5.3 ("missing FIFA columns warn but don't block `--rating elo`"). Loaders raise a data error only when `--rating` needs FIFA.
- Split the JSON `meta` block into `meta.deterministic` and `meta.environment`; reproducibility (§6, AC3) explicitly applies only to data rows + `meta.deterministic`, not to `runtime_seconds`.
- AC10 replaced with a concrete check: half-widths match the Wilson 95% closed form to within $10^{-6}$.

**Changelog v1.3:**
- Reconciled architecture (§7), data model (§8), milestones (§9), and acceptance criteria (§10) with v1.2's first-class FIFA support — there is now a `ratings/` package, a FIFA scraper, FIFA/blend milestones, and FIFA/blend ACs.
- Restored markdown formatting in §7–11.
- Fixed the blended-rating formula so the mixing weight is **roster-independent**: FIFA points are normalised against a snapshot-frozen constant $F_0$, not the field-mean $\bar{R}_{\text{fifa}}$.
- Split `--k` into `--k-elo` / `--k-fifa` and `--c` into `--c-elo` / `--c-fifa`; aligned FIFA K with the official $I = 60$ for World-Cup tournament matches.
- Specified reproducibility under `--workers` via counter-seeded RNG; added a corresponding AC.
- Confidence intervals (95% Wilson) are emitted in CSV/JSON, and shown in the stdout table when `--verbose`.
- Added §12 *Validation & Calibration* with back-test methodology and Brier-score targets.
- Documented the $\lambda$-floor as an approximation; exposed `--lambda-min`.
- Added missing risks: FIFA scraper fragility, blended-rating snapshot-pinning, worker-determinism, perf regression from FIFA tracking, $\lambda$-floor discontinuity.
- `MatchResult` now records `stage`, `neutral`, `extra_time`, and pre-match ratings, so analysis and the bracket renderer don't need to reverse-engineer match context.

---

## 1. Overview

`wcsim` is a command-line tool that simulates international football tournaments using a Monte Carlo approach. It takes a roster of teams with **Elo ratings and FIFA rankings**, plus a tournament structure (groups + knockout bracket), then plays the entire tournament thousands of times using a rating-derived Poisson goal model. It outputs per-team probabilities of advancing through each round, winning the trophy, and expected goal statistics — with 95% confidence intervals.

The default configuration ships with the 48-team, 12-group format used by the 2026 FIFA World Cup, but the engine is structure-agnostic and accepts any group-stage + knockout layout. Users can choose which rating system drives match predictions — Elo (default), FIFA, or a configurable blend of both.

## 2. Goals & Non-Goals

### Goals

- Provide statistically defensible tournament forecasts grounded in the well-established Elo + Poisson goal model.
- Support **multiple rating sources** (Elo, FIFA, blended) so users can compare forecasts across systems.
- Run 100,000 tournament simulations in under 60 seconds on a modern multi-core laptop.
- Produce both human-readable terminal summaries and machine-readable CSV/JSON output for downstream charting, including confidence intervals.
- Be **fully deterministic** given a seed, **across any worker count**, so results are reproducible on any machine.
- Allow users to plug in custom rating CSVs, custom group draws, and custom tournament structures (within the v1 schema in §5.3) without touching code.
- Ship a documented back-test against recent tournaments so users can judge model calibration.

### Non-Goals

- A web UI, REST API, or live-data daemon.
- Player-level modelling (injuries, lineups, xG from event data).
- Live in-tournament re-forecasting after real matches play out (planned for v2).
- Betting odds integration or financial features.
- Venue-aware home advantage (the host bonus is team-attached, not match-attached — see §5.5).
- Arbitrary custom third-place-team advancement mappings. The 2026 mapping is bundled; v1's tournament-structure schema covers group count/size and simple "top-N + best-K-third-place" advancement only (see §5.3).

## 3. Target Users

- Data-curious football fans who want to run "what if" tournament scenarios.
- Sports analytics hobbyists comparing model variants and rating systems.
- Educators teaching Monte Carlo methods, Elo ratings, or Poisson regression with a concrete, engaging example.
- Journalists and bloggers producing pre-tournament probability charts.

## 4. User Stories

- *As a fan*, I run `wcsim run` and see a ranked list of trophy probabilities.
- *As an analyst*, I run `wcsim run -n 1000000 --seed 42 --out results.csv` and get a reproducible CSV with confidence intervals.
- *As a comparator*, I run `wcsim run --rating fifa` and `wcsim run --rating elo` to see how the two systems disagree.
- *As a tinkerer*, I edit `teams.csv` to bump my country's Elo or FIFA points and rerun to see the impact.
- *As a viewer*, I run `wcsim bracket` and get an ASCII bracket of the most likely outcome.
- *As a skeptic*, I run `wcsim match Spain Brazil --neutral` to inspect a single head-to-head probability.
- *As a teacher*, I run `wcsim run -n 1000 --dump-sims sims.ndjson` to show students individual simulated tournaments and chaotic outliers.
- *As a sceptic-2*, I run `wcsim backtest --year 2022` to see how the model would have done at the last World Cup.

## 5. Functional Requirements

### 5.1 Commands

| Command | Purpose |
|---|---|
| `wcsim run` | Run the full Monte Carlo and output aggregate probabilities with confidence intervals. |
| `wcsim match <A> <B>` | Print win/draw/loss and expected goals for a single match. |
| `wcsim bracket` | Print the **per-slot most-likely** knockout bracket, walked top-down from the modal champion using slot-matchup frequencies from the latest `run` cache. |
| `wcsim teams` | List loaded teams with their Elo, FIFA points, FIFA rank, and group. |
| `wcsim update-ratings [--source elo\|fifa\|all]` | Re-fetch ratings from the configured upstream sources. |
| `wcsim backtest --year YYYY [--tournament wc\|euro]` | Re-simulate a finished tournament using pre-tournament ratings; report Brier score and calibration plot data. |
| `wcsim version` | Print version and per-source data snapshot dates. |

### 5.2 Global flags

- `-n, --simulations INT` — number of tournament simulations (default `100000`).
- `--seed INT` — RNG seed for reproducibility (default: random; printed on exit).
- `--teams PATH` — override teams CSV (default `~/.wcsim/teams.csv`). Replaces the previous `--elo` flag; legacy `--elo` accepted as alias.
- `--draw PATH` — override the group-draw JSON file.
- `--hosts STR[,STR...]` — host nations that receive the home-field bonus.
- `--home-bonus INT` — home-field rating bonus in **Elo-equivalent units** (default `100`). Rescaled per rating mode in §5.5 — FIFA mode uses $H \cdot S_{\text{fifa}}/S_{\text{elo}}$ (150 by default); Elo and blend use the value as-is.
- `--rating {elo,fifa,blend}` — rating system used for predictions (default `elo`).
- `--blend FLOAT` — blend weight when `--rating blend`; weight on Elo, FIFA gets `1 - weight` (default `0.7`).
- `--k-elo FLOAT` — Elo K-factor for in-tournament updates (default `60`).
- `--k-fifa FLOAT` — FIFA importance constant $I$ for in-tournament updates (default `60`, matching FIFA's official value for World-Cup tournament matches).
- `--c-elo FLOAT` — rating-to-goal scaling constant for Elo and Blend (default `300`).
- `--c-fifa FLOAT` — rating-to-goal scaling constant for pure-FIFA mode (default `450`).
- `--mu FLOAT` — Poisson baseline goals per team per game (default `1.35`).
- `--lambda-min FLOAT` — floor applied to Poisson rates (default `0.05`); see §5.5 for the rare-discontinuity caveat.
- `--no-update-ratings` — keep ratings frozen across the tournament (faster, less realistic).
- `--pens {coinflip,elo,fifa,blend}` — penalty shootout model (default: match `--rating`; `coinflip` always available). With `--pens blend`, the shootout uses the blended rating with the same `--blend` weight as match prediction.
- `--out PATH` — write results to CSV or JSON (extension-detected).
- `--format {table,json,csv}` — control stdout format (default `table`).
- `--workers INT` — parallel processes for simulation (default: CPU count). Output is byte-identical regardless of worker count for the same seed (see §6).
- `--dump-sims PATH` — write per-simulation NDJSON log of every match outcome. ⚠️ Can produce multi-GB files at `-n 100000`; use `--dump-sample N` to keep only `N` random sims.
- `--ci / --no-ci` — toggle 95% Wilson confidence intervals in output (default on for CSV/JSON, off for stdout table).
- `-q, --quiet` / `-v, --verbose` — log level control; `-v` enables CIs in the stdout table.

### 5.2.1 `wcsim match`-specific flags

`wcsim match` accepts the global rating / scale flags (`--rating`, `--blend`, `--c-*`, `--k-*`, `--mu`, `--lambda-min`, `--seed`) plus:

- `--home {A,B,none}` — which positional argument receives the host bonus; `none` is a neutral venue. Default `A`.
- `--neutral` — alias for `--home none`. If both `--home` and `--neutral` are passed, `--neutral` wins so it can be used as an explicit override.

### 5.3 Inputs

- **Teams data** — CSV with columns:
  - `team` — display name
  - `iso3` — three-letter ISO code
  - `confederation` — `UEFA`, `CONMEBOL`, `CONCACAF`, `AFC`, `CAF`, `OFC`
  - `elo` — current Elo rating (float)
  - `fifa_points` — current FIFA Men's World Ranking points (float)
  - `fifa_rank` — integer position in the latest FIFA ranking
  - `fifa_updated` — ISO date of the FIFA snapshot
  - `elo_updated` — ISO date of the Elo snapshot

  A bundled snapshot ships with the tool. A sidecar `teams.meta.json` carries snapshot-level normalisation constants ($E_0$, $F_0$ — see §5.5) and per-source last-update dates. Missing FIFA columns trigger a warning but do not block `--rating elo` runs; loaders raise a data error (exit code `2`) only when `--rating fifa` or `--rating blend` is requested without FIFA data.

- **Group draw** — JSON: `{"A": ["Team1","Team2","Team3","Team4"], "B": [...], ...}`.
- **Host config** — JSON: `{"hosts": ["TeamX","TeamY"]}` for home-bonus assignment.
- **Tournament structure** (optional) — JSON conforming to the v1 schema below. Defaults to the bundled 2026 World Cup structure.

  ```json
  {
    "groups":      {"count": 8, "size": 4},
    "advancement": {"top_per_group": 2, "best_thirds": 0},
    "knockout":    {"first_round": "R16", "seeding": "1A_vs_2B"}
  }
  ```

  - `groups.count` × `groups.size` must equal the number of teams in the draw.
  - `advancement.top_per_group` × `groups.count` + `advancement.best_thirds` must be a power of two (the size of `knockout.first_round`).
  - `knockout.seeding` accepts `"1A_vs_2B"` (classic crosswise pairing) or `"wc2026"` (the bundled 12-group third-place permutation). Arbitrary custom permutations are out of scope for v1; see §2 *Non-Goals* and §11.

### 5.4 Outputs

- **Stdout table** for `run` (FIFA rank shown for context; CI half-widths shown with `-v`):

  ```
  Team        FIFA  Elo    Win%   Final%  SF%   QF%   R16%   R32%   GroupOut%
  Spain       1     2161   28.3   42.1    58.0  72.3  86.5   99.2   0.8
  Argentina   2     2143   12.7   25.4    ...
  ```

- **CSV/JSON** with one row per team and columns for every round probability, plus its `_ci_lo` / `_ci_hi` (95% Wilson) when `--ci` is on, plus `mean_goals_for`, `mean_goals_against`, `simulations`, `elo_start`, `fifa_points_start`, `fifa_rank_start`. JSON additionally carries a top-level `meta` block split into two sub-blocks:
  - `meta.deterministic` — `{seed, simulations, rating_mode, blend_weight, k_elo, k_fifa, c_elo, c_fifa, mu, lambda_min, e0, f0, ci_method: "wilson_95", snapshot_dates: {...}}`. Covered by the byte-identical reproducibility guarantee (§6, AC3).
  - `meta.environment` — `{wcsim_version, runtime_seconds, host_os, host_cores}`. **Not** part of the reproducibility guarantee. Cache invalidation (see *Cache* below) keys on `wcsim_version` from this block; an upgrade therefore invalidates the cache without breaking the in-scope reproducibility claim.
- **Bracket file** (`--out bracket.txt`) — per-slot most-likely knockout bracket. Construction is a top-down greedy walk: pick the modal champion, then for each preceding slot pick the most frequent matchup involving the already-fixed teams, then recurse. This is **not** the modal *full* bracket (which is essentially unique across 100k sims) — it is a slot-coherent rendering of the cache, and the file header states this explicitly.
- **Per-simulation log** (`--dump-sims PATH`) NDJSON of every simulation's full results — useful for surfacing outlier or chaotic scenarios. Use `--dump-sample N` to cap size.
- **Cache** — every `run` writes three files under `~/.wcsim/`:
  - `last_run.parquet` — per-team aggregate probabilities.
  - `last_run_slots.parquet` — per-slot matchup frequency table keyed by `(stage, slot_id, home_team, away_team)`; `wcsim bracket` reads this.
  - `last_run.meta.json` — same `meta.deterministic` / `meta.environment` blocks as the JSON output.

  A `wcsim_version` mismatch invalidates the cache with a clear error rather than silently rendering stale data.

### 5.5 Core algorithm

For two teams A and B with chosen rating $$R$$ (Elo, FIFA points, or blended), and home advantage $$H$$ applied to the home side:

- **Effective rating difference:** $$D = (R_A + H) - R_B$$.
- **Win expectation:** $$W_e = \frac{1}{1 + 10^{-D/S}}$$, where the scale $$S$$ depends on rating system: $$S = 400$$ for Elo, $$S = 600$$ for FIFA points, and $$S = 400$$ for blend (since the blended rating is normalised to Elo space — see *Blended rating* below). The FIFA scale is empirically chosen so that FIFA-point gaps in the bundled snapshot reproduce the same median win-rate as Elo on the same matchups; the calibration script is shipped in `tools/calibrate_fifa_scale.py` and is rerun whenever the snapshot is refreshed.
- **Goal expectations:** $$\lambda_A = \max(\lambda_{\min},\ \mu + D/(2c)),\quad \lambda_B = \max(\lambda_{\min},\ \mu - D/(2c))$$. The constant $$c$$ is `--c-elo` for Elo/Blend (default `300`) and `--c-fifa` for pure FIFA (default `450`). The floor $$\lambda_{\min}$$ (default `0.05`) introduces a small discontinuity for $$|D| \gtrsim 2c(\mu - \lambda_{\min})$$, which for default values is $$\approx 780$$ Elo points — present in $$< 1\%$$ of international fixtures. See §11 for an open question on switching to a multiplicative form.
- **Goals sampled independently:** $$G_A \sim \text{Poisson}(\lambda_A),\quad G_B \sim \text{Poisson}(\lambda_B)$$.
- **Post-match rating update** follows the chosen system's official formula:
  - **Elo:** $$R' = R + K_{\text{elo}} \cdot G_m \cdot (W - W_e)$$, with $$G_m = 1$$ for draws or 1-goal wins, $$1.5$$ for 2-goal wins, and $$(11 + |\Delta|)/8$$ for margins of 3+.
  - **FIFA:** $$R' = R + I \cdot (W - W_e)$$, with $$I = K_{\text{fifa}}$$ (default `60`, the official FIFA value for World-Cup tournament matches).
  - **Blend:** Elo and FIFA points are each updated independently using their own formulas; the blended rating is recomputed from the two new values for the next match's prediction.
- **Blended rating** (roster-independent normalisation):
  $$R_{\text{blend}} = w \cdot R_{\text{elo}} + (1 - w) \cdot \frac{R_{\text{fifa}} \cdot E_0}{F_0}$$
  where $$E_0 = 1500$$ (canonical Elo reference mean) and $$F_0$$ is the **global** FIFA-points mean across the full international pool at snapshot time, frozen in `teams.meta.json`. Because $$E_0$$ and $$F_0$$ do not depend on which teams are in the tournament, the blend weight $$w$$ has a stable interpretation across runs and rosters.
- **Group-stage tiebreakers:** points → goal difference → goals for → head-to-head → fair-play (skipped; documented as a deliberate simplification) → random draw.
- **Host bonus is team-attached and mode-rescaled:** the configured host nations receive $$+H$$ in every match they play, regardless of venue. The user-set `--home-bonus` is expressed in Elo-equivalent units; effective $$H$$ per mode is $$H_{\text{elo}} = H_{\text{blend}} =$$ `--home-bonus` and $$H_{\text{fifa}} =$$ `--home-bonus` $$\cdot\ S_{\text{fifa}}/S_{\text{elo}} = 1.5 \times$$ `--home-bonus` (150 by default). The rescaling preserves the win-probability bump across rating modes. Venue-aware bonuses are non-goal (§2) and tracked in §11.
- **Knockout draws:** extra time is sampled as a second match with $$\lambda$$ scaled by $$30/90$$; if still tied, a penalty shootout is decided by `--pens` (rating-weighted Bernoulli for `elo`/`fifa`, fair 50/50 for `coinflip`). `--pens` defaults to whatever `--rating` is.

## 6. Non-Functional Requirements

- **Performance:** 100k sims in ≤ 60s on an 8-core machine for `--rating elo` and `--rating fifa`; ≤ 66s for `--rating blend` (10% allowance per §11). Single-core completes within 5 minutes.
- **Reproducibility:**
  - **Scope:** the byte-identical guarantee covers the data rows of CSV / JSON / Parquet outputs and the `meta.deterministic` JSON sub-block. The `meta.environment` sub-block (runtime, host info) is explicitly excluded.
  - Identical in-scope output for identical `(seed, teams, draw, params)`.
  - **Worker-count invariant:** in-scope output is byte-identical for any value of `--workers`. The Monte Carlo runner uses a counter-based RNG (numpy `PCG64DXSM`); simulation $i$ derives its state from `(seed, i)`, so workers can claim sims in any order and merging aggregates is associative.
- **Portability:** pure Python ≥ 3.11, runs on macOS, Linux, Windows.
- **Footprint:** install size ≤ 25 MB including bundled rating snapshots.
- **Offline-first:** fully functional without internet after install; only `update-ratings` and `backtest` (when downloading historical ratings) need network.
- **Exit codes:** `0` success, `1` user error, `2` data error, `3` network error.

## 7. Architecture

```
wcsim/
├── cli.py                # Typer command tree.
├── data.py               # Loaders: teams CSV (+ teams.meta.json), group-draw JSON,
│                         # host config, tournament-structure JSON.
├── scrapers/             # One module per upstream rating source.
│   ├── elo.py            #   eloratings.net scraper (matches the Spike 1 validation source).
│   │                     #   Bundled historical snapshots in wcsim/data/snapshots/
│   │                     #   are derived from the same source.
│   └── fifa.py           #   fifa.com Men's World Ranking (JSON endpoint where available,
│                         #   HTML fallback).
├── ratings/              # Rating-system math (pluggable).
│   ├── base.py           #   Common interface: win_expectation(), update(), to_elo_space().
│   ├── elo.py            #   Elo formulas (W_e, G_m, K_elo).
│   ├── fifa.py           #   FIFA formulas (W_e, I).
│   └── blend.py          #   Blended rating using fixed E0/F0 from snapshot metadata.
├── model.py              # Match simulator: Poisson sampling, extra time, penalties.
├── tournament.py         # Group stage, third-place ranking, knockout bracket.
├── sim.py                # Monte Carlo runner with counter-seeded multiprocessing.Pool.
├── report.py             # Table / CSV / JSON formatters, Wilson CIs, ASCII bracket renderer.
├── cache.py              # Persists aggregates to ~/.wcsim/last_run.parquet plus meta sidecar.
└── validate.py           # backtest command: re-runs historical tournaments, computes Brier
                          # score and calibration buckets.
```

Dependencies: `numpy`, `pandas`, `typer`, `rich` (pretty tables), `httpx` (rating refresh), `pyarrow` (cache), `lxml` (HTML parsing for scrapers).

## 8. Data Model

```python
from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True)
class Team:
    name: str
    iso3: str
    confederation: str
    elo: float
    elo_updated: date
    group: str
    # FIFA fields are optional at the data-model level so Elo-only inputs are valid
    # (see §5.3). data.py raises an error if --rating fifa or --rating blend is
    # requested against a Team missing these fields.
    fifa_points: float | None = None
    fifa_rank: int | None = None
    fifa_updated: date | None = None

@dataclass(frozen=True)
class MatchResult:
    home: str
    away: str
    home_goals: int
    away_goals: int
    stage: str             # "group", "R32", "R16", "QF", "SF", "F", "3rd"
    neutral: bool
    extra_time: bool
    went_to_pens: bool
    winner: str | None     # None only for group-stage draws
    home_rating_before: float
    away_rating_before: float

@dataclass(frozen=True)
class TournamentResult:
    seed: int
    rating_mode: str       # "elo" | "fifa" | "blend"
    matches: list[MatchResult]
    placements: dict[str, str]   # team -> "GroupOut" | "R32" | "R16" | "QF" | "SF" | "Final" | "Champion"
    final_ratings: dict[str, float]  # team -> post-tournament rating in the chosen space
```

The `Team` dataclass mirrors the CSV schema in §5.3 exactly, including the `*_updated` dates. `MatchResult` records enough context (`stage`, `neutral`, `extra_time`, pre-match ratings) that downstream analysis and the bracket renderer don't need to reverse-engineer it from sequence position.

## 9. Milestones

- **M1 — Core math (week 1):** `wcsim match` returns single-match probabilities for Elo, FIFA, and Blend, each verified against a closed-form reference within 0.5 pp.
- **M2 — Tournament engine (week 2):** Group stage with full tiebreakers and a knockout bracket simulated once end-to-end for all three rating modes.
- **M3 — Monte Carlo (week 3):** `wcsim run` with counter-seeded multiprocessing, deterministic across worker counts, CSV/JSON output with Wilson CIs.
- **M4 — Data & UX (week 4):** Elo + FIFA scrapers, `update-ratings`, ASCII bracket, rich tables, docs, CI on macOS/Linux/Windows.
- **M5 — Validation (week 5):** `wcsim backtest` against the last two completed World Cups and the most recent Euros; calibration of `c` against the ~25% historical group-stage draw rate; published Brier scores and calibration plots in `docs/validation.md`.

## 10. Acceptance Criteria

1. `wcsim match A B --home A --rating elo` returns a win probability matching the closed-form Elo $W_e$ within 0.5 pp; the same holds for `--rating fifa` against the FIFA formula and for `--rating blend --blend 0.7` against the analytic blended $W_e$.
2. `wcsim run -n 100000 --seed 1 --rating elo` finishes within 60 s on an 8-core laptop; the same with `--rating fifa`; with `--rating blend`, within 66 s (≤10% regression budget).
3. `wcsim run --seed 1 --out r.csv` produces **byte-identical** CSV across two invocations *and* across `--workers 1`, `--workers 4`, and `--workers 8`.
4. Every team in the input teams CSV appears in the output with all round-probability columns populated (no NaNs, no missing rows).
5. The sum of `win_pct` across all teams equals `1.0 ± 0.001`.
6. For two teams with identical FIFA rank but different Elo, group-survival% under `--rating elo` is monotonic in Elo; the symmetric statement holds for `--rating fifa` (different FIFA, identical Elo).
7. **Calibration:** with default parameters, the simulated group-stage draw rate is within 2 pp of the historical international rate (~25%).
8. **Back-test:** on the 2018 and 2022 World Cups, the per-match Brier score across the simulated probabilities is below 0.21 (better than a 0.222 uniform-probability baseline).
9. `wcsim update-ratings --source fifa` produces a `teams.csv` in which every team has populated `fifa_points` and `fifa_rank`; equivalent for `--source elo` and `--source all`.
10. CSV / JSON output includes `_ci_lo` / `_ci_hi` columns for every probability when `--ci` is on. For every reported probability $\hat{p}$ at sample size $n$, the bounds match the Wilson 95% closed form to within $10^{-6}$ — verified by a unit test against `statsmodels.stats.proportion.proportion_confint(..., method="wilson")`.

## 11. Risks & Open Questions

- **Elo source stability** — `eloratings.net` publishes international-team Elo via HTML; fragility is mitigated with a versioned scraper, bundled snapshot fallback, and a CI job that fetches and parses weekly. The Spike 1 validation uses the same source (via a bundled snapshot of the same data), so a passing spike means production calibration is not at risk from a source switch. `clubelo.com` is **not** an option — it covers European club football only, not international teams.
- **FIFA source stability** — `fifa.com` JSON endpoints have changed twice in the last two years; same mitigation as Elo, plus an HTML fallback parser.
- **Group-draw updates** — if the configured tournament reseeds, ship updated `draw.json` as a patch release.
- **Calibration of $c$** — needs tuning so the simulated draw rate matches historical reality (~25% in international football). Exposed as `--c-elo` / `--c-fifa` and documented in `docs/validation.md`.
- **Penalty shootout model** — pure coin-flip vs rating-weighted is a methodological choice; default tracks `--rating`, all three are exposed.
- **Blended-rating snapshot pinning** — $F_0$ is frozen at snapshot build time. If the global FIFA mean drifts substantially (e.g. after a rating-system overhaul like the 2018 FIFA reform), `--rating blend` results across two snapshots are not strictly comparable. Document the snapshot date prominently; consider exposing `--f0 FLOAT` for advanced users.
- **$\lambda$-floor discontinuity** — at extreme rating gaps the additive Poisson form requires a floor and introduces a small kink. Open question: switch to a multiplicative form $\lambda_{A/B} = \mu \cdot 10^{\pm D / (2c \cdot S)}$ in v1.4 after back-test data is in.
- **Worker-count reproducibility** — relies on counter-seeded RNG (PCG64DXSM); regression here is caught by AC3 in CI.
- **Performance regression from FIFA tracking** — measuring two ratings per team during in-tournament updates costs ~5–10%; CI benchmarks both pure-Elo and pure-FIFA to enforce the 10% budget.
- **Venue-aware home bonus** — currently team-attached. WC 2026 has three hosts spread across a continent, and applying a single bonus to all of them in every match is a known simplification. Tracked for v2 alongside live re-forecasting.
- **Legal / ToS** — scrapers respect `robots.txt` and rate-limit to ≤ 1 request / 2 s; bundled snapshots are redistributed as factual data. Users running `update-ratings` are themselves responsible for compliance.
- **Bracket greedy-walk coherence** — the top-down walk used by `wcsim bracket` (§5.4) is slot-coherent by construction, but it is not the modal *full* bracket and can differ noticeably from the per-slot independent argmax when two finalists are nearly equiprobable. Documented in the bracket file header; an alternative "argmax-per-slot, then reconcile" renderer is an open candidate for v1.5 if users find the greedy walk confusing.
- **Custom tournament structure scope** — v1's schema (§5.3) covers group count/size, top-N + best-K-third-place advancement, and two named seeding modes (`1A_vs_2B`, `wc2026`). Formats with byes, unequal group sizes, or arbitrary third-place permutations (AFCON, COSAFA Cup, some Olympic formats) are out of scope until v2.

## 12. Validation & Calibration

Two artifacts produced at M5 and rerun on every release:

### 12.1 Calibration of `c`

`tools/calibrate_c.py` sweeps `--c-elo` over `[200, 400]` (and `--c-fifa` over `[350, 600]`) and selects the value that minimises $|p_{\text{draw,sim}} - 0.25|$ on the bundled snapshot. The chosen defaults (`300` / `450`) are the result of this sweep on the 2026-Q2 snapshot; the script reruns on every snapshot refresh and the defaults are updated if the new minimum moves by more than 5%.

### 12.2 Back-test

`wcsim backtest --year 2022` (and `--year 2018`) replays a completed World Cup using the ratings as they stood the week before kickoff (snapshots are bundled in `wcsim/data/backtest/`). For each completed match $m$ we record the model's predicted probability $\hat{p}_m$ over `{home win, draw, away win}` and the realised outcome $y_m$, then compute:

- **Per-match Brier score:** $B = \frac{1}{N} \sum_m \sum_k (\hat{p}_{m,k} - y_{m,k})^2$. Target: $B < 0.21$ (a uniform-probability baseline scores $\approx 0.222$).
- **Calibration plot:** bucket predictions into deciles of $\hat{p}$ and plot mean predicted vs. observed frequency; the diagonal is perfect calibration.
- **Trophy-probability comparison:** the simulated trophy probabilities are compared to FiveThirtyEight's published pre-tournament forecasts (where available) as an external sanity check.

These outputs are committed to `docs/validation.md` so users can judge calibration without rerunning.
