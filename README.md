# wcsim — Football Tournament Monte Carlo Simulator

Simulate the FIFA World Cup thousands of times using Elo ratings, FIFA rankings, and squad market values. Get per-team probabilities for winning the trophy, reaching the final, and advancing through each round.

## Quick start

```bash
# Set up the environment
python3 -m venv .venv
.venv/bin/pip install -r spikes/01-validation/requirements.txt

# Simulate the 2026 World Cup (100k sims, three-way blend, deterministic)
.venv/bin/python -m wcsim.cli run -n 100000 --seed 1 --rating blend --blend-player 0.5 --shrinkage 0.85

# Quick 1000-sim run with pure Elo
.venv/bin/python -m wcsim.cli run -n 1000 --seed 42

# Predict a single match
.venv/bin/python -m wcsim.cli match France England --rating blend --blend-player 0.5 --neutral

# List all teams with ratings
.venv/bin/python -m wcsim.cli teams
```

## Recommended configuration

The three-way blend with shrinkage produces predictions closest to professional forecasts (Opta Supercomputer):

```bash
.venv/bin/python -m wcsim.cli run \
  -n 100000 \
  --seed 1 \
  --rating blend \
  --blend-player 0.5 \
  --shrinkage 0.85
```

This blends 50% squad market values + 35% Elo + 15% FIFA rankings, with regression-to-mean shrinkage (0.85) to prevent top teams from being overrated by volatile recent results.

## Sample output

```
$ wcsim run -n 100000 --seed 1 --rating blend --blend-player 0.5
Team         Win     Final        SF        QF       R16       R32  GroupOut
----------------------------------------------------------------------------
ESP        10.3%     17.8%     29.0%     43.7%     64.5%     97.0%      3.0%
FRA        10.0%     17.1%     28.3%     45.5%     68.8%     94.1%      5.9%
ARG         8.1%     14.5%     24.6%     38.9%     58.3%     94.0%      6.0%
ENG         7.9%     14.0%     23.9%     41.8%     65.9%     92.8%      7.2%
BRA         5.3%     10.0%     18.9%     31.3%     53.1%     90.1%      9.9%
POR         5.5%     10.5%     19.6%     35.8%     58.9%     90.8%      9.2%
...
```

## Commands

| Command | Description |
|---|---|
| `wcsim run` | Run N tournament simulations and output probability table |
| `wcsim match A B` | Single-match win/draw/loss probabilities |
| `wcsim teams` | List loaded teams with Elo ratings |
| `wcsim version` | Print version |

### `wcsim run` flags

| Flag | Default | Description |
|---|---|---|
| `-n, --simulations` | 100000 | Number of tournament simulations |
| `--seed` | random | RNG seed for reproducibility |
| `--rating` | elo | Rating mode: `elo`, `fifa`, `blend`, `player` |
| `--blend-player` | 0.0 | Player-value weight in three-way blend (0=disabled, 0.5=recommended) |
| `--shrinkage` | 1.0 | Rating regression to mean (1.0=none, 0.85=moderate) |
| `--workers` | CPU count | Parallel workers |
| `--out PATH` | stdout | Write to CSV or JSON file |
| `--format` | table | Output format: `table`, `csv`, `json` |
| `--ci / --no-ci` | on | Include 95% Wilson confidence intervals |
| `--teams PATH` | bundled | Override teams CSV |
| `--draw PATH` | bundled | Override draw JSON |
| `-v, --verbose` | off | Show CIs in table output |
| `-q, --quiet` | off | Suppress progress messages |

## Model

The match model follows PRD §5.5:

1. **Rating difference** D between teams A and B (with optional home bonus)
2. **Poisson goal rates** λ_A = μ + D/(2c), λ_B = μ - D/(2c)
3. **Dixon-Coles correlation** τ adjusts the four lowest-scoring joint outcomes (0-0, 0-1, 1-0, 1-1) via parameter ρ
4. **Goals sampled** from the joint distribution on a 9×9 score grid
5. **Knockout matches** add extra time (λ scaled by 30/90) then penalty shootout (rating-weighted Bernoulli)

Four pluggable rating systems:
- **Elo** (S=400, c=300) — default, well-calibrated against historical WC results
- **FIFA** (S=600, c=450) — official FIFA Men's World Ranking points
- **Player** — squad market values (Transfermarkt), log-normalized to Elo scale
- **Blend** (three-way Elo + FIFA + Player) — `--blend-player 0.5` gives 50% Player + 35% Elo + 15% FIFA

## Tournament formats

| Format | Teams | Groups | Knockout | 3rd place |
|---|---|---|---|---|
| WC 2018/2022 | 32 | 8 × 4 | R16 → QF → SF → Final | Yes |
| WC 2026 | 48 | 12 × 4 | R32 → R16 → QF → SF → Final | No |

Format auto-detected from team count in the draw. The WC 2026 bracket uses the official FIFA R32 structure (Dec 2025 draw): crosswise winner-vs-runner-up pairings, runner-up-vs-runner-up matches, and constraint-based third-place slot assignment matching FIFA's permutation table.

## Reproducibility

- Same `--seed` produces byte-identical output regardless of `--workers` count
- Counter-based RNG: simulation i uses seed = base_seed + i
- `ProcessPoolExecutor.map` preserves index order for deterministic aggregation

## Data sources

Bundled snapshots scraped from canonical sources:
- **Elo ratings**: [eloratings.net](https://eloratings.net) per-team TSV files
- **FIFA rankings**: [inside.fifa.com](https://inside.fifa.com/fifa-world-ranking/men) hidden JSON API
- **Squad market values**: [Kaggle davidcariboo/player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) (Transfermarkt mirror)
- **Match results**: [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json) (WC 2018 + 2022)
- **WC 2026 draw**: official FIFA draw (2025-12-05)

Re-run the scrapers to refresh:
```bash
.venv/bin/python spikes/01-validation/scrapers/elo.py
.venv/bin/python spikes/01-validation/scrapers/fifa.py
.venv/bin/python spikes/01-validation/scrapers/matches.py
.venv/bin/python spikes/01-validation/scrapers/squads.py  # requires players.csv from Kaggle
```

## Project structure

```
wcsim/                    # Library
├── types.py              # Team, Params, MatchResult, TournamentResult, SimulationResult
├── ratings/              # Pluggable rating systems
│   ├── base.py           # RatingSystem Protocol
│   ├── elo.py            # EloRating
│   ├── fifa.py           # FifaRating
│   ├── blend.py          # BlendRating (two-way Elo+FIFA)
│   ├── player.py         # PlayerRating (squad market values)
│   └── blend_all.py      # BlendAllRating (three-way Elo+FIFA+Player)
├── squad_data.py         # Squad value loader (reads squads_2026.json)
├── model.py              # predict_match, sample_match (Poisson + Dixon-Coles)
├── tournament.py         # Group stage + knockout (WC 2018/2022 + WC 2026 FIFA bracket)
├── sim.py                # Monte Carlo runner (ProcessPoolExecutor)
├── report.py             # Table / CSV / JSON formatters + Wilson CIs
├── cache.py              # Last-run persistence (~/.wcsim/)
├── data.py               # Teams CSV + draw JSON loaders
└── cli.py                # Typer CLI entry point

tests/                    # pytest suite (99 tests, 98%+ coverage)
spikes/01-validation/     # Spike 1: model validation back-test
├── validate.py           # RPS scoring against WC 2018+2022
├── scrapers/             # Canonical-source data fetchers
└── data/raw/             # Bundled snapshots (Elo, FIFA, matches, draw, squads)
```

## Development

```bash
# Run the test suite
.venv/bin/python -m pytest tests/ -v

# Run with coverage
.venv/bin/python -m pytest tests/ --cov=wcsim --cov-report=term-missing

# Run the Spike 1 validation (model calibration check)
cd spikes/01-validation && ../.venv/bin/python validate.py
```

## Validation

Spike 1 validated the model against WC 2018 + 2022 (128 historical matches):
- **RPS (Ranked Probability Score)**: 0.2088 for Elo mode (gate: < 0.215, baseline: 0.222)
- **Dixon-Coles structural fix** reduced worst calibration error from 0.21 to 0.058 (73% improvement, within sampling noise at 128 matches)
- Full sweep results in `spikes/01-validation/results/sweep.json`

## Status

- [x] **Spike 1**: Model validation back-test (PRD v1.7, Dixon-Coles)
- [x] **Spike 2**: Library extraction (types + ratings + model + tournament)
- [x] **Spike 3**: Monte Carlo runner + CLI (`wcsim run`)
- [x] **Spike 4**: Player-value rating + three-way blend + official FIFA 2026 bracket
- [ ] **Spike 5**: Performance optimization + `pyproject.toml` packaging
- [ ] **Spike 6**: `wcsim bracket` + `wcsim update-ratings`
- [ ] **Spike 7**: Documentation + PyPI release

## License

TBD
