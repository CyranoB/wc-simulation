# Spike 1: Model Validation Back-Test

**Decision: `soft_fail` — and the prescribed parameter sweep does not rescue it.** The headline RPS metric **passes** (Elo 90-min RPS = **0.2088** vs the < 0.215 gate), but the Elo-mode calibration curve fails the spec gate in two deciles. The follow-up `c_elo / μ` sweep in `tools/sweep.py` evaluated **231 (c_elo, μ) combinations** across the full prescribed range and found **0 fully-passing settings**: every cell passes RPS, but no cell fits calibration within ±0.05. The persistent failure is **decile 6** (predicted 0.55-0.65 range), worst-failing in 65% of swept cells. **Spike 2 (library extraction) should not start until PRD §5.5 gets a structural model change** — parameter tuning alone is insufficient. FIFA-mode RPS = 0.2140, Blend-mode RPS = 0.2082. ET divergence (post-ET − 90-min) is right at the 0.01 flag threshold for Elo (0.011). Simulated group-stage draw rate is 23.8% (Elo) vs 19.8% observed — also above the ±2pp flag threshold.

## Summary table

| Mode  | RPS 90-min | RPS post-ET | ET divergence | Sim draw rate | Δ vs observed |
|-------|-----------:|------------:|--------------:|--------------:|--------------:|
| Elo   |     0.2088 |      0.2198 |        0.0110 |         23.8% |        +4.0pp |
| FIFA  |     0.2140 |      0.2228 |        0.0089 |         24.1% |        +4.3pp |
| Blend |     0.2082 |      0.2180 |        0.0098 |         22.7% |        +2.9pp |

Uniform-predictor baseline: 2/9 ≈ 0.2222 (verified by a startup assertion in `validate.py`). Observed group-stage draw rate: 19.8%.

Calibration plot: `results/calibration.png`. Decision record: `results/brier.json`.

## Calibration miss detail

Two Elo-mode 90-min deciles exceed the ±0.05 tolerance against the diagonal:

| Decile | mean predicted | mean observed | n   | abs err | Verdict |
|--------|---------------:|--------------:|----:|--------:|---------|
| 1      |          0.165 |         0.098 |  41 |   0.068 | FAIL (just 0.018 over threshold; within sampling noise) |
| 6      |          0.634 |         0.842 |  19 |   0.208 | FAIL (~2σ miss; model is too cautious in this range) |

Decile 6 is the meaningful issue — when the model gives a team a ~63% chance, that team actually wins ~84% of the time. The model is under-confident at moderately-favoured. This is the kind of thing the `c_elo` sweep should be able to fix without structural changes.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python validate.py
```

All inputs are bundled under `data/raw/` from three canonical-source scrapers (`scrapers/elo.py`, `scrapers/fifa.py`, `scrapers/matches.py`). If you want to re-run the scrape (it's offline-bundled by default), use the scrapers in that folder. The script is single-threaded NumPy; runtime is under a minute.

## Data sources

- **Elo**: scraped from `eloratings.net` via per-team TSVs (`<TeamName>.tsv` — spaces → underscores). For each WC participant, we fetch their TSV and extract the post-match Elo from their last match before the tournament's opening day.
- **FIFA**: scraped from `inside.fifa.com` via their hidden API. Picks `2018-06-07` for WC 2018 and `2022-10-06` for WC 2022 (latest ranking ≤ pre-tournament).
- **Matches**: pulled from `openfootball/worldcup.json` (public-domain JSON dumps including FT, ET, and penalty scores).
- **Knockout ET/penalty supplement**: hand-curated `data/raw/knockout_supplement.csv` (10 rows). The plan's `REQUIRED_ET_PENS_MATCHES` set asserts coverage at load time.

## Caveats

- **FIFA reform** — the FIFA ranking algorithm changed in mid-2018, so the 2018 and 2022 snapshots have different point distributions. `F0` is computed per-tournament here (`f0_by_year` in `brier.json`: 877.7 for 2018, 1579.1 for 2022). Production will use a single bundled snapshot.
- **Knockout convention** — `rps_90min` is the gating metric (the match model predicts 90 minutes); `rps_post_et` is informational. The observed ET divergence (0.011 for Elo) is right at the 0.01 flag threshold — slightly above means the model handles ET-bound matches a bit differently from its standalone 90-min predictions. Spike 2 should investigate before wiring up the tournament engine.
- **Sample size noise** — 128 matches → RPS has ~±0.015 noise. The 0.215 gate has ~0.6σ headroom over the 0.222 uniform baseline. If the calibration sweep doesn't bring the curve in line, extend the dataset to Euro 2020 (51 matches) before declaring a hard fail.

## Sweep finding (soft_fail follow-up)

`tools/sweep.py` runs the spec-prescribed `c_elo ∈ [200, 400]` / `μ ∈ [1.1, 1.6]` sweep at step sizes 10 and 0.05 (231 cells total). Full grid is committed to `results/sweep.json`.

| Outcome | Count |
|---|---:|
| Cells passing the RPS gate (rps_90min < 0.215) | **231 / 231** |
| Cells passing the calibration gate (every decile with n ≥ 8 within ±0.05 of diagonal) | **0 / 231** |
| Cells fully passing both | **0 / 231** |

The best-calibrated cell in the grid is `(c_elo=380, μ=1.55)` with `worst_err = 0.0731` in decile 5 — still 46% over the 0.05 tolerance. Decile 6 (model predicted 0.55-0.65 range) is the persistent culprit, worst-failing in 149 of 231 cells (65%).

**Conclusion:** the spec's soft_fail fallback action applies — *"revisit model definition."* The score-grid Poisson model under-predicts moderate-favourite outcomes in a way that parameter tuning cannot fix.

## Next step

Update PRD §5.5 with a structural model change before Spike 2 starts. Candidates listed in the PRD's existing risks section (§11), ordered by relevance to the observed miss:

1. **Dixon-Coles correlation** — adjusts the low-score outcomes (0-0, 1-0, 0-1, 1-1) to better fit observed soccer score distributions. Directly addresses the under-prediction of moderate-favourite wins AND the over-prediction of draws (the two flags we triggered) in a single change. **Most likely fix.**
2. **Multiplicative Poisson form** — `λ_{A/B} = μ · 10^(±D/(2cS))` instead of additive. Removes the `λ_min` floor and changes the shape at moderate rating gaps; might or might not solve the calibration miss on its own.
3. **Separate home/away μ** — usually 1.45 home / 1.25 away in club football. Less applicable to WC (mostly neutral venues); probably orthogonal to the observed miss.

Implementation path: amend PRD §5.5 with the chosen structural change, re-run `validate.py` and `tools/sweep.py`. If the updated model achieves `worst_err ≤ 0.05` at a defensible `(c_elo, μ)`, update the PRD defaults and proceed to Spike 2. Otherwise, escalate further (consider asking whether 128 matches is too small a calibration sample given the model's structure).
