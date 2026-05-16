"""Calibration sweep for the Spike 1 soft_fail follow-up.

Per spec §6, soft_fail prescribes: 'Sweep c_elo ∈ [200, 400] and μ ∈ [1.1, 1.6];
if a setting passes, update PRD defaults and proceed.'

This script:
1. Loads the bundled Elo + match data via validate.py helpers.
2. For each (c_elo, μ) on a 21×11 grid (c step 10, μ step 0.05), recomputes
   Elo-mode 90-min predictions and:
     - rps_90min[elo] (must stay below 0.215, the headline gate)
     - calibration buckets (every decile with n ≥ 8 must be within ±0.05
       of the diagonal — this is what failed in the baseline run)
3. Reports the best fully-passing setting (if any) and the worst-failing
   decile + its error magnitude for the runner-up.

Outputs:
- results/sweep.json: full grid with rps_90min and calibration metrics per cell
- stdout: top candidates and recommendation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SPIKE_ROOT = HERE.parent
sys.path.insert(0, str(SPIKE_ROOT))

# Import the data-loading + math helpers from validate.py. validate.py only
# runs main() when invoked as __main__, so importing is side-effect-free.
from validate import (  # noqa: E402
    WC2018, WC2022, HOST_BY_YEAR,
    load_elo, load_matches,
    one_hot_90min, rps, calibration_buckets, _outcome_probs,
)

RESULTS = SPIKE_ROOT / "results"

# Fixed parameters (kept at PRD defaults during the sweep).
LAM_MIN = 0.05
HOME_BONUS_ELO = 100.0
GATE_RPS = 0.215
GATE_CAL_TOLERANCE = 0.05
GATE_CAL_MIN_N = 8


def elo_predictions(
    matches: list[dict],
    elo: dict[str, float],
    hosts: set[str],
    c_elo: float,
    mu: float,
) -> np.ndarray:
    """Compute Elo-mode 3-outcome predictions for all matches with given (c, μ).
    Other params (lam_min, home_bonus) are held at PRD defaults."""
    out = np.zeros((len(matches), 3))
    for i, m in enumerate(matches):
        a, b = m["home_iso3"], m["away_iso3"]
        host_diff = float(a in hosts) - float(b in hosts)
        D = (elo[a] - elo[b]) + HOME_BONUS_ELO * host_diff
        lam_a = max(LAM_MIN, mu + D / (2 * c_elo))
        lam_b = max(LAM_MIN, mu - D / (2 * c_elo))
        out[i] = _outcome_probs(lam_a, lam_b)
    return out


def evaluate(preds: np.ndarray, y_90: np.ndarray) -> dict:
    """Return rps and per-decile calibration verdict for one (c, μ) cell."""
    rps_val = rps(preds, y_90)
    mean_p, mean_y, sizes = calibration_buckets(preds, y_90)
    worst_err = 0.0
    worst_dec = -1
    for i, (p, y, n) in enumerate(zip(mean_p, mean_y, sizes)):
        if n < GATE_CAL_MIN_N or np.isnan(p) or np.isnan(y):
            continue
        err = abs(p - y)
        if err > worst_err:
            worst_err = err
            worst_dec = i
    cal_ok = worst_err <= GATE_CAL_TOLERANCE
    return {
        "rps_90min": float(rps_val),
        "rps_passes": rps_val < GATE_RPS,
        "calibration_ok": bool(cal_ok),
        "worst_decile_err": float(worst_err),
        "worst_decile_idx": int(worst_dec),
        "fully_passes": bool(cal_ok and rps_val < GATE_RPS),
    }


def main() -> int:
    # Load data once (much cheaper than per-cell).
    matches_2018 = load_matches(2018)
    matches_2022 = load_matches(2022)
    elo_2018 = load_elo(WC2018[0])
    elo_2022 = load_elo(WC2022[0])

    all_matches = matches_2018 + matches_2022
    y_90 = np.array([one_hot_90min(m) for m in all_matches])

    # Sweep grid: c_elo in [200, 400] step 10, mu in [1.10, 1.60] step 0.05.
    c_grid = list(range(200, 401, 10))   # 21 values
    mu_grid = [round(1.10 + 0.05 * i, 3) for i in range(11)]  # 1.10, 1.15, ..., 1.60

    print(f"Sweeping {len(c_grid)} × {len(mu_grid)} = {len(c_grid) * len(mu_grid)} cells "
          f"(c_elo step 10, μ step 0.05)...")

    grid: list[dict] = []
    for c in c_grid:
        for mu in mu_grid:
            preds_2018 = elo_predictions(matches_2018, elo_2018, HOST_BY_YEAR[2018], c, mu)
            preds_2022 = elo_predictions(matches_2022, elo_2022, HOST_BY_YEAR[2022], c, mu)
            preds = np.vstack([preds_2018, preds_2022])
            cell = evaluate(preds, y_90)
            cell["c_elo"] = c
            cell["mu"] = mu
            grid.append(cell)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "sweep.json").open("w") as f:
        json.dump({"grid": grid, "gate": {"rps": GATE_RPS, "cal_tolerance": GATE_CAL_TOLERANCE}}, f, indent=2)

    fully = [c for c in grid if c["fully_passes"]]
    print(f"\n{len(fully)} of {len(grid)} cells fully pass "
          f"(rps_90min < {GATE_RPS} AND every decile with n≥{GATE_CAL_MIN_N} within ±{GATE_CAL_TOLERANCE} of diagonal).")

    # Current baseline.
    baseline = next(c for c in grid if c["c_elo"] == 300 and abs(c["mu"] - 1.35) < 1e-6)
    print(f"\nBaseline (c_elo=300, μ=1.35): rps={baseline['rps_90min']:.4f}  "
          f"worst_err={baseline['worst_decile_err']:.4f} (decile {baseline['worst_decile_idx']})  "
          f"fully_passes={baseline['fully_passes']}")

    if fully:
        # Rank fully-passing by tightest calibration margin, then by lowest RPS.
        ranked = sorted(fully, key=lambda c: (c["worst_decile_err"], c["rps_90min"]))
        print(f"\nTop 5 fully-passing settings:")
        print(f"  {'c_elo':>5}  {'mu':>5}  {'rps':>7}  {'worst_err':>9}")
        for c in ranked[:5]:
            print(f"  {c['c_elo']:>5}  {c['mu']:>5.2f}  {c['rps_90min']:>7.4f}  {c['worst_decile_err']:>9.4f}")
        best = ranked[0]
        print(f"\nRECOMMENDED DEFAULTS:  c_elo = {best['c_elo']}, μ = {best['mu']:.2f}")
        print(f"  rps_90min = {best['rps_90min']:.4f}  worst calibration error = {best['worst_decile_err']:.4f}")
        return 0
    else:
        # No fully-passing cell. Find closest-to-passing.
        # First filter to cells that pass the RPS gate (a soft_fail with passing RPS
        # is still preferred over a hard_fail with both failing).
        rps_passing = [c for c in grid if c["rps_passes"]]
        print(f"\nNo (c, μ) setting fully passes. {len(rps_passing)} of {len(grid)} pass the RPS gate.")
        ranked = sorted(rps_passing, key=lambda c: c["worst_decile_err"])
        print(f"\nTop 5 closest-to-passing (RPS passes; calibration sorted by worst-decile error):")
        print(f"  {'c_elo':>5}  {'mu':>5}  {'rps':>7}  {'worst_err':>9}  {'decile':>6}")
        for c in ranked[:5]:
            print(f"  {c['c_elo']:>5}  {c['mu']:>5.2f}  {c['rps_90min']:>7.4f}  "
                  f"{c['worst_decile_err']:>9.4f}  {c['worst_decile_idx']:>6}")
        print(f"\nCONCLUSION: parameter sweep alone cannot fix calibration; "
              f"the model likely needs a structural change per spec §6 hard_fail action "
              f"(multiplicative Poisson, separate home/away μ, or Dixon-Coles).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
