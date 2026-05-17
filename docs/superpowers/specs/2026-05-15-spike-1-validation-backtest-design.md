# Spike 1: Model Validation Back-Test

**Status:** Approved by brainstorm 2026-05-15
**Parent:** PRD `wcsim` v1.6 §12.2 *Back-test*
**Work location:** `spikes/01-validation/` on branch `CyranoB/review-prd` (in-place; the Conductor workspace already provides the working branch)

## 1. Goal

Retire model risk **before** any production `wcsim` code is written. Demonstrate that the match model in PRD §5.5 (rating + Poisson with $\lambda$-floor) is calibrated on historical World Cup matches across all three rating modes (Elo, FIFA, blend).

**Binary success:** RPS (Ranked Probability Score) on the combined 2018 + 2022 WC match set < **0.215** for **Elo mode** (the production default), using PRD-default parameters. The threshold sits 0.005 above the PRD's AC8 target of < 0.21 as a tolerance band for the ~±0.015 sampling noise inherent in a 128-match dataset. The uniform-predictor baseline on balanced 3-outcome data is exactly $2/9 \approx 0.222$, asserted at startup in `validate.py`. FIFA and blend modes are measured alongside; their RPS values are reported but their pass/fail is informational, not gating.

## 2. Non-Goals

- No CLI, no Typer, no `wcsim/` package — throwaway script.
- No tournament simulation, no Monte Carlo loop, no multiprocessing.
- No scrapers — inputs are bundled and committed.
- No unit tests beyond the validation outputs themselves.
- No closed-form sanity tests for the math (those are PRD AC1, deferred to Spike 2).

## 3. Inputs

All datasets are fetched once, normalised, and committed under `spikes/01-validation/data/`. The spike is offline-reproducible afterwards.

- **Pre-tournament Elo** — eloratings.net snapshot for the day before each tournament's opening match (2018-06-13, 2022-11-19). Sourced via a public Kaggle mirror that scrapes eloratings.net daily (see Task 2 in the plan for the specific dataset), so we get a stable historical snapshot without scraping HTML ourselves.
- **Pre-tournament FIFA points** — Kaggle "FIFA Men's World Ranking" dataset, monthly snapshot immediately before each tournament (2018-06-07 and 2022-10-06 rankings).
- **WC match results** — Kaggle "FIFA World Cup" dataset (fallback: football-data.co.uk). 64 matches per tournament; both 90-minute and post-extra-time scores; penalty winner where applicable.

Team-name normalisation across the three sources is via ISO3 codes; this is the single largest non-modelling task in the spike.

## 4. Method

For every historical match $m$ between teams A and B (A = "home" side per dataset, even at neutral venues):

1. Look up team A and B's Elo and FIFA points as of the day before kickoff.
2. Compute three predicted distributions over `{home win, draw, away win}`, one per rating mode, using PRD §5.5 with default parameters:

   | Mode | $R$ source | $S$ | $c$ | $\mu$ | $\lambda_{\min}$ | $H$ (host only) |
   |---|---|---|---|---|---|---|
   | Elo | clubelo Elo | 400 | 300 | 1.35 | 0.05 | 100 |
   | FIFA | FIFA points | 600 | 450 | 1.35 | 0.05 | 150 |
   | Blend | $0.7 R_{\text{elo}} + 0.3 \cdot R_{\text{fifa}} \cdot E_0/F_0$ | 400 | 300 | 1.35 | 0.05 | 100 |

   The FIFA host bonus of 150 is the Elo bonus rescaled to preserve the same win-probability bump: $H_{\text{fifa}} = H_{\text{elo}} \cdot S_{\text{fifa}}/S_{\text{elo}}$. Blend constants: $E_0 = 1500$ (canonical Elo reference mean); $F_0$ = mean FIFA points across the relevant snapshot, computed per-tournament for the spike (production will use a single bundled snapshot, per PRD §5.5).

3. Convert each $(\lambda_A, \lambda_B)$ pair to a 3-outcome distribution by summing the joint Poisson PMF on a $9 \times 9$ score grid (truncation at 8 goals/side covers > 99.99% of mass).
4. Record $(\hat{p}_{m,\text{mode}}, y_m)$ where $y_m$ is the one-hot realised outcome.
5. **Knockout matches — dual-convention scoring.** The match model in PRD §5.5 predicts **90-minute outcomes**; ET and penalties are separate sampling mechanisms in the production tournament engine. Evaluating a 90-min predictor against a post-ET outcome conflates the two and slightly flatters the model on draws that broke in extra time. The spike therefore computes RPS under **both** conventions and reports both:
   - **`rps_90min`:** $y_m$ uses the 90-minute regulation score. Group-stage matches: identical to step 4. Knockout matches that went to ET: counted as 90-minute draws.
   - **`rps_post_et`:** $y_m$ uses the post-ET score; penalty winners are attributed to the post-ET winner; the "draw" outcome is impossible for knockouts.

   `rps_90min` is the **gating metric** (it is what the model actually predicts). `rps_post_et` is informational. A divergence `rps_post_et - rps_90min > 0.01` is reported as a flag for Spike 2 to investigate ET handling before the production tournament engine is built.
6. **Informational: simulated draw rate.** Sum the diagonal mass of each match's $9 \times 9$ score grid to get the predicted draw probability; average across the 96 group-stage matches (48 per tournament × 2). Compare to the observed group-stage draw frequency (~25% historically). PRD §12.1 targets ≤ 2 pp deviation; a wide miss (e.g., 35% or 15%) is an early signal that `c` needs the §12.1 sweep **before** Spike 2 — much cheaper than discovering it inside the tournament engine.

After all matches, compute:

- **RPS per mode and convention:** $$\text{RPS}_{\text{mode},\text{conv}} = \frac{1}{N(K-1)} \sum_m \sum_{k=1}^{K-1} \left( \sum_{j \le k} \hat{p}_{m,j,\text{mode}} - \sum_{j \le k} y_{m,j,\text{conv}} \right)^2$$ with $K = 3$ and outcome ordering `[home_win, draw, away_win]`. Six values total (3 modes × 2 conventions). A uniform predictor on balanced one-hot outcomes scores exactly $2/9 \approx 0.222$ — `validate.py` asserts this at startup as a guard rail.
- **Simulated draw rate per mode** vs observed group-stage draw rate.
- **Calibration plot (Elo mode, 90-min convention):** bucket predictions into deciles of $\hat{p}$, plot mean predicted vs observed frequency. The diagonal is perfect calibration.
- **Breakdowns:** RPS broken out by tournament (2018, 2022, combined), by mode, by convention, and by outcome bucket (`favourite_wins`, `underdog_wins`, `draw`) for sanity. *Favourite* is defined by higher Elo at kickoff, independent of which mode is being scored — so the buckets stay comparable across modes.

## 5. Output / Artifacts

```
spikes/01-validation/
├── validate.py                # ~100–200 lines of NumPy
├── data/
│   ├── elo_2018.csv
│   ├── elo_2022.csv
│   ├── fifa_2018.csv
│   ├── fifa_2022.csv
│   ├── matches_2018.csv
│   └── matches_2022.csv
├── results/
│   ├── brier.json             # see schema below
│   └── calibration.png        # Elo-mode decile calibration
└── README.md                  # how to reproduce, with conclusion paragraph at top
```

`brier.json` schema (filename kept as `brier.json` for stability across the iteration history — the score inside is RPS):

```json
{
  "decision": "pass" | "pass_with_caveat" | "soft_fail" | "hard_fail",
  "params": {
    "c_elo": 300, "c_fifa": 450, "mu": 1.35, "lambda_min": 0.05,
    "blend_w": 0.7, "e0": 1500,
    "f0_by_year": {"2018": 1280.4, "2022": 1342.7},
    "home_bonus_elo": 100, "home_bonus_fifa": 150
  },
  "n_matches": {"total": 128, "group_stage": 96, "knockout": 32},
  "rps_90min": {
    "elo":   {"total": 0.207, "2018": 0.209, "2022": 0.205},
    "fifa":  {"total": 0.221, "2018": 0.224, "2022": 0.218},
    "blend": {"total": 0.205, "2018": 0.207, "2022": 0.203}
  },
  "rps_post_et": {
    "elo":   {"total": 0.211, "2018": 0.213, "2022": 0.209},
    "fifa":  {"total": 0.224, "2018": 0.227, "2022": 0.221},
    "blend": {"total": 0.209, "2018": 0.211, "2022": 0.207}
  },
  "et_divergence": {"elo": 0.004, "fifa": 0.003, "blend": 0.004},
  "draw_rate": {
    "observed_group_stage": 0.260,
    "simulated_group_stage": {"elo": 0.249, "fifa": 0.231, "blend": 0.244},
    "delta_pp": {"elo": -1.1, "fifa": -2.9, "blend": -1.6}
  },
  "by_bucket": {"favourite_wins": {...}, "underdog_wins": {...}, "draw": {...}}
}
```

All committed to git on the current working branch (`CyranoB/review-prd` in the Conductor workspace; the plan does not create a separate spike branch).

## 6. Pass / Fail

The gating metric is **`rps_90min[elo].total`** (the 90-min RPS on Elo mode — what the production match model actually predicts).

| Outcome | Criterion | Action |
|---|---|---|
| **Pass** | `rps_90min[elo].total` < 0.215 **AND** Elo-mode calibration within ±0.05 of diagonal in every decile with ≥ 8 matches (sparser deciles reported but not gating) | Proceed to Spike 2. Report all 6 RPS values + draw-rate comparison alongside. |
| **Pass w/ caveat** | Elo passes, but `rps_90min[fifa].total` or `rps_90min[blend].total` > 0.225 | Proceed to Spike 2. Document the underperforming mode in PRD §11 as "exposed for comparison, not recommended as default." |
| **Soft fail** | `rps_90min[elo].total` in [0.215, 0.222] | Sweep `c-elo` ∈ [200, 400] and `μ` ∈ [1.1, 1.6]; if a setting passes, update PRD defaults and proceed. Otherwise revisit model definition. |
| **Hard fail** | `rps_90min[elo].total` ≥ 0.222 | **Stop.** Rework PRD §5.5 (candidates: multiplicative Poisson form, separate home/away $\mu$, Dixon-Coles correlation) before any further engineering. |

**Informational flags** (do not gate, but are documented in `README.md`):

- `rps_post_et[elo] - rps_90min[elo] > 0.01` → ET handling is a Spike 2 investigation item before the tournament engine is wired up.
- `|draw_rate.delta_pp[elo]| > 2` → `c-elo` likely needs the PRD §12.1 sweep; surface this for Spike 2 to address before Monte Carlo work.

Calibration-curve failure with a passing RPS counts as soft fail.

## 7. Risks & Contingencies

- **Elo source aligned with production** — eloratings.net is the source for both this spike (via a bundled Kaggle mirror snapshot) and the production scraper (PRD v1.5 §7). This eliminates cross-source re-validation risk. Documented in `README.md`.
- **Kaggle Elo mirror staleness** — the public mirror dataset may lag the live eloratings.net by a few days. For 2018 and 2022 historical snapshots this is irrelevant; for the production scraper, it is mitigated by hitting eloratings.net directly.
- **FIFA reform discontinuity** — FIFA reformed its ranking algorithm in mid-2018. The 2018 and 2022 snapshots have different point distributions, so $F_0$ is computed per-tournament here. Flagged in `README.md`; production uses a single snapshot.
- **Team-name normalisation** — clubelo, Kaggle FIFA, and match-results datasets disagree on spellings ("South Korea" / "Korea Republic" / "KOR"). Normalise to ISO3; expect 30–60 minutes of data cleanup.
- **Sample size noise** — 128 matches → RPS has ~±0.015 noise. If the Elo total lands in [0.215, 0.218], extend the dataset by adding Euro 2020 (51 matches) before declaring soft fail.
- **Source availability** — snapshot once and commit; the spike never re-downloads after initial setup.
- **ET / penalties convention** — the convention in §4 is one defensible choice. If calibration looks systematically off on knockout matches specifically, fall back to "90-minute score only" and document the limitation.

## 8. Estimated Effort

**10–14 hours** of focused work for solo + LLM-assisted. Breakdown:

- ~4 hours: fetch + normalise the three data sources (most of the effort is team-name reconciliation).
- ~3 hours: implement the three rating-mode predictors and the Poisson score-grid → 3-outcome conversion.
- ~2 hours: RPS (both conventions) + draw-rate + calibration plot + decision logic.
- ~2 hours: write `README.md`, commit, document conclusion.
- ~3 hours buffer for unexpected data issues or a soft-fail sweep.

**Workflow tip — front-load data prep.** Download all three CSVs and spot-check them (`head -20`, verify column names, look up a known match like Argentina-France 2022) **before** writing any model code. Team-name normalisation is the most common surprise — catching it during data prep is cheap; debugging it inside the model pipeline is not.

## 9. Decision Record

The outcome — measured RPS values, params used, and the four-way decision label — is the input to Spike 2's go/no-go gate. It lives at `spikes/01-validation/results/brier.json` and is summarised at the top of `spikes/01-validation/README.md`. Spike 2 cannot start until this artifact exists with `decision` ∈ {`pass`, `pass_with_caveat`}.
