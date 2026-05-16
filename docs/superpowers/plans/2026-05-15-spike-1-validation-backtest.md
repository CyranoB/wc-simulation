# Spike 1 — Model Validation Back-Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the `wcsim` match model (PRD §5.5) against the 2018 + 2022 World Cups before any production code is written. The gate is `brier_90min[elo].total < 0.215`; pass means Spike 2 (library extraction) starts.

**Architecture:** A single throwaway Python script (`spikes/01-validation/validate.py`) on branch `spike/01-validation`. It loads three bundled CSVs (Elo, FIFA, match results), predicts every historical match under three rating modes × two scoring conventions, computes Brier + simulated-draw-rate + Elo-mode calibration, and writes `results/brier.json` plus `results/calibration.png`. Single-threaded NumPy; no CLI, no package, no unit tests beyond the validation outputs themselves (per spec §2).

**Tech Stack:** Python ≥ 3.11, NumPy, pandas, matplotlib, requests (only for the one-time data fetch).

**Heads-up on a spec correction:** Task 0 below patches an error that slipped through both peer reviews — the spec and PRD currently say `clubelo.com` is the Elo source, but `clubelo.com` publishes **club** Elo (Real Madrid, etc.), not international-team Elo. International Elo lives at `eloratings.net`. We patch this first so the rest of the plan stays consistent.

---

## Task 0: Correct the Elo source references in PRD and spec

> **Status: COMPLETED out-of-band on 2026-05-15 in commit `6bbd846`.** Skip this task when executing the plan; all six steps below are pre-applied. Steps retained for traceability.

**Files:**
- Modify: `PRD.md`
- Modify: `docs/superpowers/specs/2026-05-15-spike-1-validation-backtest-design.md`

- [x] **Step 1: Update PRD §7 architecture block.**

Replace this line in `PRD.md`:

```
│   ├── elo.py            #   clubelo.com CSV API client (matches the Spike 1 validation source).
```

with:

```
│   ├── elo.py            #   eloratings.net scraper (matches the Spike 1 validation source).
│   │                     #   Bundled historical snapshots in wcsim/data/snapshots/
│   │                     #   are derived from the same source.
```

- [x] **Step 2: Update PRD §11 *Elo source stability* risk.**

Replace:

```
- **Elo source stability** — `clubelo.com` exposes a stable CSV API (lower fragility than HTML scraping); risk further mitigated with a versioned client, bundled snapshot fallback, and a CI job that fetches weekly. The Spike 1 validation uses the same source, so a passing spike means production calibration is not at risk from a source switch.
```

with:

```
- **Elo source stability** — `eloratings.net` publishes international-team Elo via HTML; fragility is mitigated with a versioned scraper, bundled snapshot fallback, and a CI job that fetches and parses weekly. The Spike 1 validation uses the same source (via a bundled snapshot of the same data), so a passing spike means production calibration is not at risk from a source switch. `clubelo.com` is **not** an option — it covers European club football only, not international teams.
```

- [x] **Step 3: Update PRD v1.5 changelog entry to match.**

Replace the fourth changelog bullet under "Changelog v1.5":

```
- **Elo source aligned with the Spike 1 validation source: `clubelo.com`** (CSV API) instead of `eloratings.net` (HTML). Removes the cross-source re-validation risk and reduces scraper fragility. §7 and §11 updated.
```

with:

```
- **Elo source clarified: `eloratings.net`** is the correct upstream for international-team Elo (an earlier draft incorrectly named `clubelo.com`, which covers clubs only). §7 and §11 updated. Spike 1 uses the same source via a bundled snapshot, so cross-source re-validation risk remains eliminated.
```

- [x] **Step 4: Update spike spec §3 *Inputs*.**

Replace:

```
- **Pre-tournament Elo** — clubelo.com daily snapshot for the day before each tournament's opening match (2018-06-13, 2022-11-19). Public-domain CSV API.
```

with:

```
- **Pre-tournament Elo** — eloratings.net snapshot for the day before each tournament's opening match (2018-06-13, 2022-11-19). Sourced via a public Kaggle mirror that scrapes eloratings.net daily (see Task 2 in the plan for the specific dataset), so we get a stable historical snapshot without scraping HTML ourselves.
```

- [x] **Step 5: Update spike spec §7 *Risks*.**

Replace:

```
- **Elo source aligned with production** — clubelo.com is the source for **both** this spike and the production scraper (PRD v1.5 §7 was updated to match). This eliminates the cross-source re-validation risk that would have existed if the production scraper targeted eloratings.net. Documented in `README.md`.
```

with:

```
- **Elo source aligned with production** — eloratings.net is the source for both this spike (via a bundled Kaggle mirror snapshot) and the production scraper (PRD v1.5 §7). This eliminates cross-source re-validation risk. Documented in `README.md`.
- **Kaggle Elo mirror staleness** — the public mirror dataset may lag the live eloratings.net by a few days. For 2018 and 2022 historical snapshots this is irrelevant; for the production scraper, it is mitigated by hitting eloratings.net directly.
```

- [x] **Step 6: Commit the corrections.**

```bash
git add PRD.md docs/superpowers/specs/2026-05-15-spike-1-validation-backtest-design.md
git commit -m "Correct Elo source: eloratings.net (international), not clubelo.com (clubs)"
```

---

## Task 1: Scaffold the spike directory and branch

**Files:**
- Create: `spikes/01-validation/validate.py`
- Create: `spikes/01-validation/name_to_iso3.py` (empty stub; populated in Task 3)
- Create: `spikes/01-validation/README.md`
- Create: `spikes/01-validation/requirements.txt`
- Create: `spikes/01-validation/data/raw/.gitkeep`
- Create: `spikes/01-validation/results/.gitkeep`

- [ ] **Step 1: Create the branch.**

```bash
git checkout -b spike/01-validation
```

- [ ] **Step 2: Create the directory tree.**

```bash
mkdir -p spikes/01-validation/data/raw spikes/01-validation/results
touch spikes/01-validation/data/raw/.gitkeep spikes/01-validation/results/.gitkeep
```

- [ ] **Step 3: Create `spikes/01-validation/requirements.txt`.**

```
numpy>=1.26
pandas>=2.2
matplotlib>=3.8
requests>=2.31
```

- [ ] **Step 4: Create `spikes/01-validation/validate.py` with the file header and a placeholder `main`.**

```python
"""Spike 1: Model Validation Back-Test for wcsim PRD §5.5.

Loads pre-WC Elo + FIFA ratings and historical match results, predicts every
match under Elo / FIFA / Blend rating modes under two scoring conventions
(90-min and post-ET), computes Brier + draw-rate + Elo-mode calibration,
and emits results/brier.json plus results/calibration.png.

Throwaway. Single-threaded NumPy. No CLI, no package, no unit tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HERE = Path(__file__).parent
DATA = HERE / "data" / "raw"
RESULTS = HERE / "results"


def main() -> None:
    raise NotImplementedError("Tasks 4-11 fill this in.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create `spikes/01-validation/name_to_iso3.py` as an empty stub.**

```python
"""Hand-curated mapping from team-name variants (across Elo / FIFA / matches
sources) to canonical ISO3 codes. Populated by Task 3."""

NAME_TO_ISO3: dict[str, str] = {}


def to_iso3(name: str) -> str:
    iso3 = NAME_TO_ISO3.get(name.strip())
    if iso3 is None:
        raise KeyError(f"No ISO3 mapping for {name!r}; add it to name_to_iso3.py")
    return iso3
```

- [ ] **Step 6: Create `spikes/01-validation/README.md` stub.**

```markdown
# Spike 1: Model Validation Back-Test

(Conclusion paragraph filled in by Task 11.)

## Reproduce

```
pip install -r requirements.txt
python validate.py
```

Decision record: `results/brier.json`. Elo-mode calibration plot: `results/calibration.png`.
```

- [ ] **Step 7: Commit the scaffold.**

```bash
git add spikes/01-validation/
git commit -m "Spike 1: scaffold directory + stubs"
```

---

## Task 2: Fetch raw data and eyeball it (front-loaded per spec §8)

**Files:**
- Create (downloaded): `spikes/01-validation/data/raw/elo_history.csv`
- Create (downloaded): `spikes/01-validation/data/raw/fifa_ranking.csv`
- Create (downloaded): `spikes/01-validation/data/raw/matches_history.csv`

The three Kaggle datasets below are stable and well-maintained as of 2026-05; if any have moved, search Kaggle for the closest equivalent.

- [ ] **Step 1: Fetch the eloratings.net mirror.** Kaggle dataset `martj42/international-football-elo-ratings` (or the closest current equivalent — search for "international football elo"). Use the Kaggle CLI:

```bash
cd spikes/01-validation/data/raw
kaggle datasets download -d martj42/international-football-elo-ratings -p . --unzip
mv *.csv elo_history.csv
```

If you don't have the Kaggle CLI configured, download the ZIP manually from the dataset's Kaggle page and extract the CSV into `data/raw/elo_history.csv`.

Verify the file has daily Elo ratings keyed by team name and date, with both 2018-06-13 and 2022-11-19 in the date range:

```bash
head -3 elo_history.csv
awk -F, '$1 == "2018-06-13" || $1 == "2022-11-19"' elo_history.csv | head -5
```

- [ ] **Step 2: Fetch the Kaggle FIFA Men's World Ranking dataset.** Slug `cashncarry/fifaworldranking` (monthly snapshots back to 1992):

```bash
kaggle datasets download -d cashncarry/fifaworldranking -p . --unzip
mv fifa_ranking-*.csv fifa_ranking.csv 2>/dev/null || mv *ranking*.csv fifa_ranking.csv
```

Verify it contains snapshots covering June 2018 and October–November 2022:

```bash
head -3 fifa_ranking.csv
```

- [ ] **Step 3: Fetch the international-results dataset that includes WC matches.** Slug `martj42/international-football-results-from-1872-to-2017` (kept updated through current year despite the dated name; if not, search for "international football results"):

```bash
kaggle datasets download -d martj42/international-football-results-from-1872-to-2017 -p . --unzip
mv results.csv matches_history.csv
```

Verify it has WC 2018 + 2022 matches:

```bash
head -3 matches_history.csv
grep -c "FIFA World Cup" matches_history.csv   # expect at least 128 (2018+2022) plus older WCs
```

Note: this dataset has columns `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral`. It records **post-ET final scores** for knockout matches in `home_score`/`away_score` — the regulation 90-minute score is **not** in this dataset. We'll need a separate supplement for the ET/penalty split (Step 4).

- [ ] **Step 4: Fetch a knockout ET/penalty supplement.** Slug `mathurinache/fifa-world-cup` provides shootouts and ET goals. If unavailable, fall back to manually building a small CSV of just the ET/pens matches across both tournaments. Save as `spikes/01-validation/data/raw/knockout_supplement.csv` with columns `date, home, away, regulation_home, regulation_away, et_home, et_away, pen_winner`.

```bash
kaggle datasets download -d mathurinache/fifa-world-cup -p . --unzip 2>/dev/null || true
```

Adapt or hand-write the supplement as needed — there are at most ~12 rows. The known ET/pens matches:

- WC 2018: ESP–RUS (R16, pens), CRO–DEN (R16, pens), CRO–RUS (QF, pens), ENG–COL (R16, pens), CRO–ENG (SF, ET goals).
- WC 2022: NED–ARG (QF, pens), CRO–JPN (R16, pens), CRO–BRA (QF, pens), MAR–ESP (R16, pens), ARG–FRA (final, pens).

- [ ] **Step 5: Eyeball each file.**

```bash
head -5 elo_history.csv fifa_ranking.csv matches_history.csv knockout_supplement.csv
wc -l *.csv
```

Spot-check the Argentina–France 2022 final (date 2022-12-18, regulation 3-3, ET 3-3, pens won by Argentina). Confirm both Argentina and France appear in `elo_history.csv` on 2022-11-19. Note the exact name spellings — Task 3 normalises them.

- [ ] **Step 6: Commit the raw data.** The spike is fully offline-reproducible per spec §3, so the CSVs go in git.

```bash
cd ../../..
git add spikes/01-validation/data/raw/
git commit -m "Spike 1: bundle raw Elo + FIFA + matches data"
```

---

## Task 3: Build the ISO3 name-normalisation table

**Files:**
- Modify: `spikes/01-validation/name_to_iso3.py`
- Create (temporary): `spikes/01-validation/verify_names.py` (kept in the spike for re-use; not commit-blocking)

- [ ] **Step 1: Survey the unique team names across the three sources.** Create `spikes/01-validation/survey_names.py`:

```python
"""One-off survey of team-name variants. Run from spikes/01-validation/."""
import pandas as pd

matches = pd.read_csv("data/raw/matches_history.csv")
wc = matches[matches["tournament"].eq("FIFA World Cup")
             & matches["date"].between("2018-06-01", "2022-12-31")]
match_names = sorted(set(wc["home_team"]) | set(wc["away_team"]))

elo = pd.read_csv("data/raw/elo_history.csv")
elo_snap = elo[elo["date"].isin(["2018-06-13", "2022-11-19"])]
elo_names = sorted(set(elo_snap["team"]))

fifa = pd.read_csv("data/raw/fifa_ranking.csv")
fifa["rank_date"] = pd.to_datetime(fifa["rank_date"])
fifa_snap = fifa[fifa["rank_date"].dt.year.isin([2018, 2022])
                 & fifa["rank_date"].dt.month.isin([5, 6, 10, 11])]
fifa_names = sorted(set(fifa_snap["country_full"]))

print("Match names ({}):".format(len(match_names)), match_names)
print("\nElo-only (in Elo but not Match):", sorted(set(elo_names) - set(match_names)))
print("\nFIFA-only (in FIFA but not Match):", sorted(set(fifa_names) - set(match_names)))
```

Run it and read the output:

```bash
cd spikes/01-validation && python survey_names.py
```

Common variants you should see: `Korea Republic` ↔ `South Korea`; `IR Iran` ↔ `Iran`; `USA` ↔ `United States`.

- [ ] **Step 2: Populate `spikes/01-validation/name_to_iso3.py`.** Replace the stub `NAME_TO_ISO3 = {}` with the curated map below, then add any extra variants surfaced in Step 1:

```python
NAME_TO_ISO3: dict[str, str] = {
    # WC 2018 + 2022 participants and common spelling variants.
    "Argentina": "ARG",
    "Australia": "AUS",
    "Belgium": "BEL",
    "Brazil": "BRA",
    "Cameroon": "CMR",
    "Canada": "CAN",
    "Colombia": "COL",
    "Costa Rica": "CRC",
    "Croatia": "CRO",
    "Denmark": "DEN",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "England": "ENG",
    "France": "FRA",
    "Germany": "GER",
    "Ghana": "GHA",
    "Iceland": "ISL",
    "Iran": "IRN",
    "IR Iran": "IRN",
    "Iran Islamic Republic of": "IRN",
    "Japan": "JPN",
    "Korea Republic": "KOR",
    "South Korea": "KOR",
    "Mexico": "MEX",
    "Morocco": "MAR",
    "Netherlands": "NED",
    "Nigeria": "NGA",
    "Panama": "PAN",
    "Peru": "PER",
    "Poland": "POL",
    "Portugal": "POR",
    "Qatar": "QAT",
    "Russia": "RUS",
    "Saudi Arabia": "SAU",
    "Senegal": "SEN",
    "Serbia": "SRB",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "SUI",
    "Tunisia": "TUN",
    "United States": "USA",
    "USA": "USA",
    "Uruguay": "URU",
    "Wales": "WAL",
}
```

- [ ] **Step 3: Verify coverage with a small script.** Create `spikes/01-validation/verify_names.py`:

```python
"""Resolve every WC participant name through the ISO3 map. Fails loudly on miss."""
import pandas as pd
from name_to_iso3 import to_iso3

matches = pd.read_csv("data/raw/matches_history.csv")
wc = matches[matches["tournament"].eq("FIFA World Cup")
             & matches["date"].between("2018-06-01", "2022-12-31")]
names = set(wc["home_team"]) | set(wc["away_team"])
for n in sorted(names):
    to_iso3(n)
print(f"OK: resolved {len(names)} WC participant names")
```

Run it:

```bash
cd spikes/01-validation && python verify_names.py
```

If you hit `KeyError`, add the missing variant to `NAME_TO_ISO3` and retry. Then repeat the same survey for any Elo / FIFA names that appear in the WC snapshot windows but aren't yet in the map.

- [ ] **Step 4: Commit.**

```bash
cd ../..
git add spikes/01-validation/name_to_iso3.py spikes/01-validation/verify_names.py spikes/01-validation/survey_names.py
git commit -m "Spike 1: ISO3 name-normalisation table"
```

---

## Task 4: Data loaders

**Files:**
- Modify: `spikes/01-validation/validate.py`

- [ ] **Step 1: Add loader functions and date constants.** Insert below the `HERE/DATA/RESULTS` block in `validate.py`:

```python
from name_to_iso3 import to_iso3


# (snapshot date, opening match, final) per tournament.
WC2018 = ("2018-06-13", "2018-06-14", "2018-07-15")
WC2022 = ("2022-11-19", "2022-11-20", "2022-12-18")


def load_elo(snapshot_date: str) -> dict[str, float]:
    """Return {iso3: elo_rating} for teams as of snapshot_date."""
    df = pd.read_csv(DATA / "elo_history.csv")
    df = df[df["date"] == snapshot_date]
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        try:
            out[to_iso3(row["team"])] = float(row["elo"])
        except KeyError:
            # Team not in our ISO3 map — likely a non-WC participant; skip.
            continue
    return out


def load_fifa(snapshot_date: str) -> tuple[dict[str, float], dict[str, int]]:
    """Return ({iso3: fifa_points}, {iso3: fifa_rank}) for the snapshot just on or
    before `snapshot_date`."""
    df = pd.read_csv(DATA / "fifa_ranking.csv")
    df["rank_date"] = pd.to_datetime(df["rank_date"])
    target = pd.to_datetime(snapshot_date)
    eligible = df[df["rank_date"] <= target]
    latest = eligible["rank_date"].max()
    snap = eligible[eligible["rank_date"] == latest]
    points: dict[str, float] = {}
    ranks: dict[str, int] = {}
    for _, row in snap.iterrows():
        try:
            iso3 = to_iso3(row["country_full"])
        except KeyError:
            continue
        points[iso3] = float(row["total_points"])
        ranks[iso3] = int(row["rank"])
    return points, ranks


def load_matches(year: int) -> list[dict]:
    """Return list of dicts per WC match.

    Each dict has:
        date, home_iso3, away_iso3,
        regulation_home, regulation_away,
        et_home, et_away,
        pen_winner_iso3 (str | None),
        is_knockout (bool), is_neutral (bool)
    """
    base = pd.read_csv(DATA / "matches_history.csv")
    base["date"] = pd.to_datetime(base["date"])
    if year == 2018:
        window = base[base["date"].between("2018-06-14", "2018-07-15")]
    elif year == 2022:
        window = base[base["date"].between("2022-11-20", "2022-12-18")]
    else:
        raise ValueError(year)
    wc = window[window["tournament"].eq("FIFA World Cup")].sort_values("date")

    supplement_path = DATA / "knockout_supplement.csv"
    supp = pd.read_csv(supplement_path) if supplement_path.exists() else pd.DataFrame()

    out: list[dict] = []
    for _, row in wc.iterrows():
        home, away = row["home_team"], row["away_team"]
        # matches_history.csv stores POST-ET scores in home_score/away_score for knockouts.
        et_home = int(row["home_score"])
        et_away = int(row["away_score"])
        # Default: regulation == post-ET (group games, plus knockouts that didn't go to ET).
        reg_home, reg_away = et_home, et_away
        pen_winner_iso3: str | None = None
        if not supp.empty:
            match_supp = supp[
                (supp["date"] == row["date"].strftime("%Y-%m-%d"))
                & (supp["home"] == home) & (supp["away"] == away)
            ]
            if not match_supp.empty:
                s = match_supp.iloc[0]
                reg_home = int(s["regulation_home"])
                reg_away = int(s["regulation_away"])
                et_home = int(s["et_home"])
                et_away = int(s["et_away"])
                pen_winner_iso3 = to_iso3(s["pen_winner"]) if pd.notna(s["pen_winner"]) else None
        # Knockout heuristic: dates after group stage end.
        # 2018 group ends 2018-06-28; 2022 group ends 2022-12-02.
        is_knockout = (
            (year == 2018 and row["date"] >= pd.Timestamp("2018-06-30")) or
            (year == 2022 and row["date"] >= pd.Timestamp("2022-12-03"))
        )
        out.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "home_iso3": to_iso3(home),
            "away_iso3": to_iso3(away),
            "regulation_home": reg_home,
            "regulation_away": reg_away,
            "et_home": et_home,
            "et_away": et_away,
            "pen_winner_iso3": pen_winner_iso3,
            "is_knockout": is_knockout,
            "is_neutral": bool(row.get("neutral", True)),
        })
    return out
```

- [ ] **Step 2: Update `main` to print loader sanity counts.**

```python
def main() -> None:
    elo_2018 = load_elo(WC2018[0])
    elo_2022 = load_elo(WC2022[0])
    fifa_2018_pts, _ = load_fifa(WC2018[0])
    fifa_2022_pts, _ = load_fifa(WC2022[0])
    m2018 = load_matches(2018)
    m2022 = load_matches(2022)

    print(f"Elo 2018: {len(elo_2018)} teams (sample BRA={elo_2018.get('BRA')})")
    print(f"Elo 2022: {len(elo_2022)} teams (sample ARG={elo_2022.get('ARG')})")
    print(f"FIFA 2018: {len(fifa_2018_pts)} teams (sample BRA={fifa_2018_pts.get('BRA')})")
    print(f"FIFA 2022: {len(fifa_2022_pts)} teams (sample ARG={fifa_2022_pts.get('ARG')})")
    print(f"Matches 2018: {len(m2018)} (expect 64)")
    print(f"Matches 2022: {len(m2022)} (expect 64)")
    print(f"Sample 2022 final: {m2022[-1]}")
```

- [ ] **Step 3: Run and verify.**

```bash
cd spikes/01-validation && python validate.py
```

Expected: 64 matches per year. Every WC participant has an Elo and FIFA entry. The 2022 final entry should show ARG vs FRA, with `et_home`/`et_away` reflecting 3-3 and `pen_winner_iso3 == "ARG"`.

- [ ] **Step 4: Commit.**

```bash
cd ../..
git add spikes/01-validation/validate.py
git commit -m "Spike 1: data loaders"
```

---

## Task 5: Match predictor (Elo, FIFA, Blend)

**Files:**
- Modify: `spikes/01-validation/validate.py`

- [ ] **Step 1: Add params, score-grid Poisson math, and the three predictors.** Insert above `main`:

```python
# Defaults from PRD v1.5 §5.5.
PARAMS = {
    "c_elo": 300.0,
    "c_fifa": 450.0,
    "mu": 1.35,
    "lambda_min": 0.05,
    "blend_w": 0.7,
    "e0": 1500.0,
    "home_bonus_elo": 100.0,
    "home_bonus_fifa": 150.0,
}

SCORE_GRID_MAX = 8   # inclusive => 9x9 grid; covers > 99.99% of mass.


def _poisson_pmf(lmbda: float, max_goals: int) -> np.ndarray:
    """Discrete Poisson PMF over [0, max_goals]. Drops residual mass."""
    k = np.arange(max_goals + 1)
    log_fact = np.cumsum(np.log(np.maximum(k, 1)))
    log_fact[0] = 0.0
    log_pmf = k * np.log(max(lmbda, 1e-12)) - lmbda - log_fact
    return np.exp(log_pmf)


def _outcome_probs(lam_a: float, lam_b: float) -> tuple[float, float, float]:
    """Return (P(home_win), P(draw), P(away_win)) from two independent Poissons."""
    pa = _poisson_pmf(lam_a, SCORE_GRID_MAX)
    pb = _poisson_pmf(lam_b, SCORE_GRID_MAX)
    grid = np.outer(pa, pb)
    p_home = float(np.tril(grid, k=-1).sum())
    p_draw = float(np.trace(grid))
    p_away = float(np.triu(grid, k=1).sum())
    s = p_home + p_draw + p_away
    return p_home / s, p_draw / s, p_away / s


def predict(
    *,
    elo_a: float, elo_b: float,
    fifa_a: float, fifa_b: float,
    f0: float,
    a_is_host: bool, b_is_host: bool,
    mode: str,
    params: dict | None = None,
) -> tuple[float, float, float]:
    """Return (P(home), P(draw), P(away)) for team A (home) vs B under `mode`."""
    p = params or PARAMS
    host_diff = float(a_is_host) - float(b_is_host)   # +1, 0, or -1
    mu = p["mu"]
    lam_min = p["lambda_min"]

    if mode == "elo":
        H = p["home_bonus_elo"] * host_diff
        D = (elo_a - elo_b) + H
        c = p["c_elo"]
    elif mode == "fifa":
        H = p["home_bonus_fifa"] * host_diff
        D = (fifa_a - fifa_b) + H
        c = p["c_fifa"]
    elif mode == "blend":
        e0 = p["e0"]
        w = p["blend_w"]
        ra = w * elo_a + (1 - w) * (fifa_a * e0 / f0)
        rb = w * elo_b + (1 - w) * (fifa_b * e0 / f0)
        H = p["home_bonus_elo"] * host_diff   # blend lives in Elo space
        D = (ra - rb) + H
        c = p["c_elo"]
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    lam_a = max(lam_min, mu + D / (2 * c))
    lam_b = max(lam_min, mu - D / (2 * c))
    return _outcome_probs(lam_a, lam_b)
```

- [ ] **Step 2: Sanity print in `main`.** Replace the body of `main` so it tests the predictor on the 2022 final:

```python
def main() -> None:
    elo_2022 = load_elo(WC2022[0])
    fifa_pts_2022, _ = load_fifa(WC2022[0])
    f0_2022 = float(np.mean(list(fifa_pts_2022.values())))
    print(f"F0 (mean FIFA points 2022 snapshot) = {f0_2022:.1f}")

    a, b = "ARG", "FRA"
    for mode in ("elo", "fifa", "blend"):
        p = predict(
            elo_a=elo_2022[a], elo_b=elo_2022[b],
            fifa_a=fifa_pts_2022[a], fifa_b=fifa_pts_2022[b],
            f0=f0_2022,
            a_is_host=False, b_is_host=False,    # neutral venue
            mode=mode,
        )
        print(f"ARG-FRA 2022 final ({mode}): P(ARG)={p[0]:.3f} draw={p[1]:.3f} P(FRA)={p[2]:.3f}")
```

- [ ] **Step 3: Run and verify.**

```bash
cd spikes/01-validation && python validate.py
```

Expected: every prediction sums to ~1.0; P(ARG) lands between 0.30 and 0.55 in all three modes (it was a close final). Wildly imbalanced predictions (e.g., > 0.90 for one side) indicate a sign error or a name-mapping miss.

- [ ] **Step 4: Commit.**

```bash
cd ../..
git add spikes/01-validation/validate.py
git commit -m "Spike 1: match predictor (Elo / FIFA / Blend)"
```

---

## Task 6: Brier under both conventions

**Files:**
- Modify: `spikes/01-validation/validate.py`

- [ ] **Step 1: Add outcome helpers and the Brier computation.** Insert above `main`:

```python
HOST_BY_YEAR = {2018: {"RUS"}, 2022: {"QAT"}}


def one_hot_90min(match: dict) -> np.ndarray:
    """Return one-hot (home_win, draw, away_win) under the 90-minute convention.
    Knockout matches that went to ET are counted as 90-min draws."""
    h, a = match["regulation_home"], match["regulation_away"]
    if h > a: return np.array([1.0, 0.0, 0.0])
    if h < a: return np.array([0.0, 0.0, 1.0])
    return np.array([0.0, 1.0, 0.0])


def one_hot_post_et(match: dict) -> np.ndarray:
    """Return one-hot under the post-ET convention. Penalty winners attributed to
    that side; 'draw' is impossible for knockout matches."""
    h, a = match["et_home"], match["et_away"]
    if h > a: return np.array([1.0, 0.0, 0.0])
    if h < a: return np.array([0.0, 0.0, 1.0])
    if match["is_knockout"] and match["pen_winner_iso3"]:
        if match["pen_winner_iso3"] == match["home_iso3"]:
            return np.array([1.0, 0.0, 0.0])
        return np.array([0.0, 0.0, 1.0])
    return np.array([0.0, 1.0, 0.0])


def predict_all_matches(
    matches: list[dict],
    elo: dict[str, float],
    fifa_pts: dict[str, float],
    f0: float,
    hosts: set[str],
) -> dict[str, np.ndarray]:
    """Return {mode: array of shape (N, 3)} with predicted probabilities."""
    preds = {mode: np.zeros((len(matches), 3)) for mode in ("elo", "fifa", "blend")}
    for i, m in enumerate(matches):
        a, b = m["home_iso3"], m["away_iso3"]
        for mode in ("elo", "fifa", "blend"):
            preds[mode][i] = predict(
                elo_a=elo[a], elo_b=elo[b],
                fifa_a=fifa_pts[a], fifa_b=fifa_pts[b],
                f0=f0,
                a_is_host=(a in hosts), b_is_host=(b in hosts),
                mode=mode,
            )
    return preds


def brier(preds: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean per-match Brier score over a (N,3) prediction array."""
    return float(np.mean(np.sum((preds - outcomes) ** 2, axis=1)))
```

- [ ] **Step 2: Wire Brier computation into `main`.**

```python
def main() -> None:
    results: dict = {"params": dict(PARAMS), "n_matches": {}}

    all_matches: list[dict] = []
    all_preds = {mode: [] for mode in ("elo", "fifa", "blend")}
    all_y_90 = []
    all_y_et = []
    per_year_indices: dict[int, list[int]] = {}
    elo_by_year: dict[int, dict[str, float]] = {}

    for year, snap in ((2018, WC2018[0]), (2022, WC2022[0])):
        elo = load_elo(snap)
        elo_by_year[year] = elo
        fifa_pts, _ = load_fifa(snap)
        f0 = float(np.mean(list(fifa_pts.values())))
        matches = load_matches(year)
        preds = predict_all_matches(matches, elo, fifa_pts, f0, HOST_BY_YEAR[year])

        start = len(all_matches)
        all_matches.extend(matches)
        for mode in preds:
            all_preds[mode].append(preds[mode])
        for m in matches:
            all_y_90.append(one_hot_90min(m))
            all_y_et.append(one_hot_post_et(m))
        per_year_indices[year] = list(range(start, len(all_matches)))

    y_90 = np.array(all_y_90)
    y_et = np.array(all_y_et)
    preds_all = {mode: np.vstack(all_preds[mode]) for mode in all_preds}

    n_kn = sum(1 for m in all_matches if m["is_knockout"])
    results["n_matches"] = {
        "total": len(all_matches),
        "group_stage": len(all_matches) - n_kn,
        "knockout": n_kn,
    }
    results["brier_90min"] = {}
    results["brier_post_et"] = {}
    for mode in ("elo", "fifa", "blend"):
        idx_2018 = per_year_indices[2018]
        idx_2022 = per_year_indices[2022]
        results["brier_90min"][mode] = {
            "total": brier(preds_all[mode], y_90),
            "2018": brier(preds_all[mode][idx_2018], y_90[idx_2018]),
            "2022": brier(preds_all[mode][idx_2022], y_90[idx_2022]),
        }
        results["brier_post_et"][mode] = {
            "total": brier(preds_all[mode], y_et),
            "2018": brier(preds_all[mode][idx_2018], y_et[idx_2018]),
            "2022": brier(preds_all[mode][idx_2022], y_et[idx_2022]),
        }
    results["et_divergence"] = {
        mode: results["brier_post_et"][mode]["total"] - results["brier_90min"][mode]["total"]
        for mode in ("elo", "fifa", "blend")
    }

    print(json.dumps({
        "n_matches": results["n_matches"],
        "brier_90min": results["brier_90min"],
        "brier_post_et": results["brier_post_et"],
        "et_divergence": results["et_divergence"],
    }, indent=2))

    # Stash for Tasks 7-10 (refactored away in Task 10 when we write brier.json).
    globals()["_RESULTS"] = results
    globals()["_PREDS"] = preds_all
    globals()["_MATCHES"] = all_matches
    globals()["_Y_90"] = y_90
    globals()["_Y_ET"] = y_et
    globals()["_ELO_BY_YEAR"] = elo_by_year
```

- [ ] **Step 3: Run and verify.**

```bash
cd spikes/01-validation && python validate.py
```

Expected: every Brier value lands in [0.18, 0.25]. Elo 90-min total should be the lowest of the three modes (since Elo is the production default). `et_divergence` should be small (< 0.02) for all modes — > 0.05 indicates a bug in `one_hot_post_et`.

- [ ] **Step 4: Commit.**

```bash
cd ../..
git add spikes/01-validation/validate.py
git commit -m "Spike 1: Brier under 90-min and post-ET conventions"
```

---

## Task 7: Simulated draw rate per mode

**Files:**
- Modify: `spikes/01-validation/validate.py`

- [ ] **Step 1: Add draw-rate helpers.** Insert above `main`:

```python
def simulated_draw_rate(preds: np.ndarray) -> float:
    """Mean predicted draw probability (column index 1)."""
    return float(np.mean(preds[:, 1]))


def observed_draw_rate(matches: list[dict], y_90: np.ndarray) -> float:
    """Fraction of group-stage matches that ended 90 minutes level."""
    grp_mask = np.array([not m["is_knockout"] for m in matches])
    return float(np.mean(y_90[grp_mask, 1]))
```

- [ ] **Step 2: Wire it into `main` after the Brier block.**

```python
    grp_mask = np.array([not m["is_knockout"] for m in all_matches])
    observed = observed_draw_rate(all_matches, y_90)
    sim_per_mode = {mode: simulated_draw_rate(preds_all[mode][grp_mask]) for mode in preds_all}
    results["draw_rate"] = {
        "observed_group_stage": observed,
        "simulated_group_stage": sim_per_mode,
        "delta_pp": {mode: (sim_per_mode[mode] - observed) * 100 for mode in sim_per_mode},
    }
    print(json.dumps({"draw_rate": results["draw_rate"]}, indent=2))
```

- [ ] **Step 3: Run and verify.**

```bash
cd spikes/01-validation && python validate.py
```

Expected: observed group-stage draw rate is ~0.20–0.28 (12 of 48 in 2018 = 0.25; 11 of 48 in 2022 = 0.23). Simulated rates per mode should be within 5pp of observed. `|delta_pp| > 2` is the early signal the spec calls out — flag it for Spike 2 but don't fail the spike.

- [ ] **Step 4: Commit.**

```bash
cd ../..
git add spikes/01-validation/validate.py
git commit -m "Spike 1: simulated draw rate per mode"
```

---

## Task 8: Elo-mode calibration plot

**Files:**
- Modify: `spikes/01-validation/validate.py`
- Create: `spikes/01-validation/results/calibration.png` (output)

- [ ] **Step 1: Add the calibration helpers.** Insert above `main`:

```python
def calibration_buckets(preds: np.ndarray, outcomes: np.ndarray, n_buckets: int = 10):
    """Decile-bucket all (prediction, outcome) pairs flattened across home/draw/away.
    Returns (mean_predicted, mean_observed, bucket_sizes) arrays of length n_buckets."""
    p = preds.flatten()
    y = outcomes.flatten()
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    mean_p = np.full(n_buckets, np.nan)
    mean_y = np.full(n_buckets, np.nan)
    sizes = np.zeros(n_buckets, dtype=int)
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_buckets - 1 else (p >= lo) & (p <= hi)
        sizes[i] = int(mask.sum())
        if sizes[i] > 0:
            mean_p[i] = float(p[mask].mean())
            mean_y[i] = float(y[mask].mean())
    return mean_p, mean_y, sizes


def write_calibration_plot(preds: np.ndarray, outcomes: np.ndarray, out_path: Path) -> dict:
    mean_p, mean_y, sizes = calibration_buckets(preds, outcomes)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
    valid = sizes > 0
    ax.scatter(mean_p[valid], mean_y[valid], s=np.maximum(sizes[valid] * 4, 20),
               c="C0", alpha=0.7, edgecolors="C0", label="Elo 90-min (size ∝ n)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Elo-mode 90-min calibration (deciles)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "mean_predicted": [None if np.isnan(x) else float(x) for x in mean_p],
        "mean_observed":  [None if np.isnan(x) else float(x) for x in mean_y],
        "bucket_sizes":   [int(s) for s in sizes],
    }
```

- [ ] **Step 2: Call it from `main` after the draw-rate block.**

```python
    cal = write_calibration_plot(preds_all["elo"], y_90, RESULTS / "calibration.png")
    results["calibration_elo_90min"] = cal
    print(f"Calibration plot -> {RESULTS / 'calibration.png'}")
```

- [ ] **Step 3: Run and verify.**

```bash
cd spikes/01-validation && python validate.py
ls -lh results/calibration.png   # expect ~30-60 KB PNG
```

Open the PNG visually. Points near the diagonal = well-calibrated. The spec gate requires every decile with ≥ 8 matches to sit within ±0.05 of the diagonal.

- [ ] **Step 4: Commit.**

```bash
cd ../..
git add spikes/01-validation/validate.py spikes/01-validation/results/calibration.png
git commit -m "Spike 1: Elo-mode calibration plot"
```

---

## Task 9: By-bucket Brier breakdown

**Files:**
- Modify: `spikes/01-validation/validate.py`

- [ ] **Step 1: Add the bucketer.** Insert above `main`:

```python
def by_bucket_brier(
    matches: list[dict],
    preds_all: dict[str, np.ndarray],
    y_90: np.ndarray,
    elo_by_year: dict[int, dict[str, float]],
) -> dict:
    """Bucket each match by realised outcome (favourite_wins / underdog_wins / draw).
    Favourite is defined by higher pre-match Elo, independent of which mode is scored."""
    fav_is_home = []
    for m in matches:
        year = 2018 if m["date"].startswith("2018") else 2022
        elo = elo_by_year[year]
        fav_is_home.append(elo[m["home_iso3"]] >= elo[m["away_iso3"]])
    fav_is_home = np.array(fav_is_home)

    labels = []
    for i, m in enumerate(matches):
        outcome = int(np.argmax(y_90[i]))   # 0=home, 1=draw, 2=away
        if outcome == 1:
            labels.append("draw")
        elif (outcome == 0 and fav_is_home[i]) or (outcome == 2 and not fav_is_home[i]):
            labels.append("favourite_wins")
        else:
            labels.append("underdog_wins")
    labels = np.array(labels)

    out: dict = {}
    for tag in ("favourite_wins", "underdog_wins", "draw"):
        mask = labels == tag
        out[tag] = {"n": int(mask.sum())}
        if mask.sum() == 0:
            continue
        for mode in preds_all:
            out[tag][mode] = brier(preds_all[mode][mask], y_90[mask])
    return out
```

- [ ] **Step 2: Call it from `main` after the calibration block.**

```python
    results["by_bucket"] = by_bucket_brier(all_matches, preds_all, y_90, elo_by_year)
    print(json.dumps({"by_bucket": results["by_bucket"]}, indent=2))
```

- [ ] **Step 3: Run and verify.**

```bash
cd spikes/01-validation && python validate.py
```

Expected: `favourite_wins` bucket should have lower Brier than `underdog_wins` (the model is more confident when it picks right). `draw` Brier is typically the highest. `n` per bucket: roughly 60 favourite-wins, 30–40 underdog-wins, 25–30 draws.

- [ ] **Step 4: Commit.**

```bash
cd ../..
git add spikes/01-validation/validate.py
git commit -m "Spike 1: per-outcome-bucket Brier breakdown"
```

---

## Task 10: Decision label and `brier.json`

**Files:**
- Modify: `spikes/01-validation/validate.py`
- Create: `spikes/01-validation/results/brier.json` (output)

- [ ] **Step 1: Add the decision function.** Insert above `main`:

```python
def decide(results: dict) -> tuple[str, list[str]]:
    """Apply the spec §6 four-way decision matrix. Returns (label, flags)."""
    elo_90 = results["brier_90min"]["elo"]["total"]
    cal = results["calibration_elo_90min"]

    # Calibration check: every decile with >= 8 matches within +/- 0.05 of diagonal.
    cal_ok = True
    for p, y, n in zip(cal["mean_predicted"], cal["mean_observed"], cal["bucket_sizes"]):
        if n is None or n < 8 or p is None or y is None:
            continue
        if abs(p - y) > 0.05:
            cal_ok = False
            break

    flags: list[str] = []
    if abs(results["draw_rate"]["delta_pp"]["elo"]) > 2:
        flags.append("draw_rate_off")
    if results["et_divergence"]["elo"] > 0.01:
        flags.append("et_divergence_high")

    if elo_90 < 0.215 and cal_ok:
        fifa_b = results["brier_90min"]["fifa"]["total"]
        blend_b = results["brier_90min"]["blend"]["total"]
        if fifa_b > 0.225 or blend_b > 0.225:
            return "pass_with_caveat", flags
        return "pass", flags
    if elo_90 >= 0.222:
        return "hard_fail", flags
    return "soft_fail", flags
```

- [ ] **Step 2: Write `brier.json` at the end of `main`.**

```python
    decision, flags = decide(results)
    results["decision"] = decision
    results["informational_flags"] = flags

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "brier.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDecision: {decision}")
    if flags:
        print(f"Informational flags: {flags}")
    print(f"Wrote {RESULTS / 'brier.json'}")
```

- [ ] **Step 3: Run and verify.**

```bash
cd spikes/01-validation && python validate.py
cat results/brier.json | python -m json.tool | head -40
```

Expected: a `decision` of `pass`, `pass_with_caveat`, `soft_fail`, or `hard_fail`. `brier.json` matches the schema in spec §5.

- [ ] **Step 4: Commit.**

```bash
cd ../..
git add spikes/01-validation/validate.py spikes/01-validation/results/brier.json
git commit -m "Spike 1: decision label and brier.json"
```

---

## Task 11: README with conclusion paragraph

**Files:**
- Modify: `spikes/01-validation/README.md`

- [ ] **Step 1: Read the final `brier.json`** and use its numbers in the conclusion paragraph.

```bash
cat spikes/01-validation/results/brier.json
```

- [ ] **Step 2: Rewrite `spikes/01-validation/README.md`.** Replace the file with the template below; **substitute the actual numbers from `brier.json`** in the conclusion paragraph and the table.

````markdown
# Spike 1: Model Validation Back-Test

**Decision: <FILL FROM brier.json>.** Elo 90-min Brier on 128 historical World Cup matches (2018 + 2022) was **<FILL>**, vs the < 0.215 gate. FIFA mode: <FILL>; Blend mode: <FILL>. Calibration curve fits diagonal within ±0.05 across deciles with ≥ 8 matches. Simulated group-stage draw rate <FILL>%, vs observed <FILL>% (Δ = <FILL>pp). ET divergence (post-ET − 90-min): <FILL>. Spike 2 (library extraction) is <cleared / blocked / soft-blocked> to start.

## Summary table

| Mode  | Brier 90-min | Brier post-ET | Sim draw rate | Δ vs observed |
|-------|-------------:|--------------:|--------------:|--------------:|
| Elo   |   <FILL>     |     <FILL>    |    <FILL>%    |   <FILL>pp    |
| FIFA  |   <FILL>     |     <FILL>    |    <FILL>%    |   <FILL>pp    |
| Blend |   <FILL>     |     <FILL>    |    <FILL>%    |   <FILL>pp    |

Calibration plot: `results/calibration.png`. Decision record: `results/brier.json`.

## Reproduce

```
pip install -r requirements.txt
python validate.py
```

All inputs are bundled under `data/raw/` (Kaggle mirrors of eloratings.net, FIFA Men's World Ranking, and international match results). The script is single-threaded NumPy; runtime is under a minute.

## Caveats (from spike spec §7)

- **Elo source** — eloratings.net via a Kaggle mirror snapshot. The production scraper (PRD §7) will hit eloratings.net directly; the mirror is convenient for offline reproducibility but should not be relied on long-term.
- **FIFA reform** — the FIFA ranking algorithm changed in mid-2018, so the 2018 and 2022 snapshots have different point distributions. `F0` is computed per-tournament here; production uses a single bundled snapshot.
- **Knockout convention** — `brier_90min` is the gating metric (the match model predicts 90 minutes); `brier_post_et` is informational. A divergence > 0.01 is flagged for Spike 2 (see `informational_flags` in `brier.json`).
- **Sample size noise** — 128 matches → Brier has ~±0.015 noise. If you re-run with a Kaggle dataset that has shifted, the headline number may move by that much; the < 0.215 gate has ~0.5σ headroom over the 0.222 uniform baseline.

## Next step

If `decision` is `pass` or `pass_with_caveat`, proceed to Spike 2 (library extraction). If `soft_fail`, run the `c_elo` / `mu` sweep described in spec §6 before deciding. If `hard_fail`, the model definition in PRD §5.5 needs structural work before any further engineering.
````

- [ ] **Step 3: Final commit and push the branch.**

```bash
git add spikes/01-validation/README.md
git commit -m "Spike 1: README with conclusion paragraph"
git push -u origin spike/01-validation
```

The spike branch now contains the full validation artifact. Either open a PR back to `main` summarising the decision, or stay on the branch and start Spike 2 from here.

---

## Self-Review Notes

Reviewed against the spec on 2026-05-15:

- **Spec coverage:** Tasks 4–10 cover §4 *Method* (loaders, predictors, dual-convention Brier, draw rate, calibration, by-bucket breakdown); Task 10 covers §5 *Output / Artifacts* (`brier.json` schema) and §6 *Pass / Fail* (decision matrix); Task 11 covers the conclusion-paragraph requirement at the top of §9 *Decision Record*. Tasks 1–3 cover §3 *Inputs* (bundled data, ISO3 normalisation) and §8 *Workflow tip* (front-loaded data prep).
- **Identified spec error:** Task 0 corrects the `clubelo.com` references in PRD v1.5 and the spike spec — `clubelo.com` is club Elo only; international Elo lives at `eloratings.net`. This is a real bug in the upstream docs, not a plan-level issue.
- **No placeholders:** every code-bearing step shows the full code; every verify step has a specific expected outcome.
- **Type consistency:** `predict()`, `predict_all_matches()`, `brier()`, and `decide()` use consistent shapes (`(N, 3)` arrays, `{mode: array}` dicts). `to_iso3()` is named consistently across `name_to_iso3.py` and its callers. `elo_by_year` is constructed in Task 6 Step 2 and consumed in Task 9 — naming matches.
