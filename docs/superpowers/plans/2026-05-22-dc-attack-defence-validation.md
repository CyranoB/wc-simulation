# Dixon-Coles Attack/Defence Validation Spike

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a spike-only Dixon-Coles attack/defence backend trained on broader international results and compare it against the existing blend baseline.

**Architecture:** Keep all fitting, validation, and data refresh code under `spikes/01-validation` so normal `wcsim` CLI behavior is unchanged. Use raw team names for the broad training dataset, attach rating-derived priors only when a raw name maps to ISO3, and evaluate on the existing held-out 2018/2022 World Cup validation set.

**Tech Stack:** Python, NumPy, pandas, SciPy, pytest, existing `wcsim` validation data.

---

### Task 1: Add Failing Tests

**Files:**
- Create: `tests/test_dc_attack_defence.py`
- Create: `tests/test_validation_dc.py`

- [ ] Write tests for probability normalization, symmetry, rho behavior, prior construction, synthetic fitting, fixture parsing, validation JSON output, and existing regression safety.
- [ ] Run the new tests and confirm they fail because the spike modules do not exist yet.

### Task 2: Add Spike Model

**Files:**
- Create: `spikes/01-validation/dc_attack_defence.py`

- [ ] Implement `FitConfig`, `MatchRecord`, `PriorMaps`, `DixonColesAttackDefenseModel`, `build_prior_maps`, `score_probability_grid`, and `fit_model`.
- [ ] Fit `attack`, `defense`, `log_mu`, `home_adv`, and bounded `rho` with SciPy L-BFGS-B using weighted penalized likelihood.
- [ ] Use positive `rho` to suppress low-score draws, matching the existing project convention.

### Task 3: Add Data Refresh Scraper

**Files:**
- Create: `spikes/01-validation/scrapers/international_results.py`

- [ ] Download `martj42/international_results` `results.csv`.
- [ ] Validate the expected martj42 column schema before writing `spikes/01-validation/data/raw/international_results.csv`.

### Task 4: Add Validation Script

**Files:**
- Create: `spikes/01-validation/validate_dc.py`

- [ ] Load broad training data from `international_results.csv`.
- [ ] Train rolling four-year 2018/2022 models with Elo/FIFA priors only.
- [ ] Evaluate against the existing held-out World Cup matches and write `results/dc_attack_defence.json`.
- [ ] Add `--forecast-2026` to train a current model with `50% player + 35% Elo + 15% FIFA` priors and write `results/dc_2026_preview.json`.

### Task 5: Dependencies And Verification

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `spikes/01-validation/requirements.txt`

- [ ] Add `scipy>=1.12` to the `data` optional dependency.
- [ ] Add `scipy>=1.12` to the spike requirements file.
- [ ] Run `uv lock`.
- [ ] Run:

```bash
uv run --extra data --extra dev pytest tests/test_dc_attack_defence.py tests/test_validation_dc.py tests/test_regression.py -q
uv run --extra data python spikes/01-validation/validate_dc.py --years 2018,2022
uv run --extra data python spikes/01-validation/validate_dc.py --forecast-2026
```
