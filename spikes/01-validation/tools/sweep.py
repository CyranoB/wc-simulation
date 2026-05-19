"""Calibration sweep for Spike 1 — now 3D (c_elo, μ, ρ) after PRD v1.7's
Dixon-Coles addition.

Per spec §6, soft_fail prescribes a sweep over `c_elo ∈ [200, 400]` and
`μ ∈ [1.1, 1.6]`. The original 2D sweep (v1.6) confirmed no setting in that
grid passes the calibration gate. PRD v1.7 added `ρ` (Dixon-Coles
correlation) to the model; this sweep extends to that third dimension to
test whether the structural change rescues calibration.

For each (c_elo, μ, ρ) cell:
- rps_90min[elo] must stay below 0.215 (headline gate)
- every calibration decile with n ≥ 8 must be within ±0.05 of diagonal

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
    rho: float = 0.0,
) -> np.ndarray:
    """Compute Elo-mode 3-outcome predictions for all matches with given
    (c_elo, μ, ρ). Other params (lam_min, home_bonus) are held at PRD defaults."""
    out = np.zeros((len(matches), 3))
    for i, m in enumerate(matches):
        a, b = m["home_iso3"], m["away_iso3"]
        host_diff = float(a in hosts) - float(b in hosts)
        D = (elo[a] - elo[b]) + HOME_BONUS_ELO * host_diff
        lam_a = max(LAM_MIN, mu + D / (2 * c_elo))
        lam_b = max(LAM_MIN, mu - D / (2 * c_elo))
        out[i] = _outcome_probs(lam_a, lam_b, rho)
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

    # 3D sweep grid:
    #   c_elo ∈ [200, 400] step 10  (21 values)
    #   μ     ∈ [1.10, 1.60] step 0.05  (11 values)
    #   ρ     ∈ [−0.40, 0.40] step 0.025 (33 values; widened past PRD's safe
    #         band of |ρ|≤0.2 to probe whether larger ρ fully passes — τ clamps
    #         to 0 at the boundary in _outcome_probs so it stays well-defined).
    c_grid = list(range(200, 401, 10))
    mu_grid = [round(1.10 + 0.05 * i, 3) for i in range(11)]
    rho_grid = [round(-0.40 + 0.025 * i, 4) for i in range(33)]

    total = len(c_grid) * len(mu_grid) * len(rho_grid)
    print(f"Sweeping {len(c_grid)} × {len(mu_grid)} × {len(rho_grid)} = {total} cells "
          f"(c_elo step 10, μ step 0.05, ρ step 0.025)...")

    grid: list[dict] = []
    for c in c_grid:
        for mu in mu_grid:
            for rho in rho_grid:
                preds_2018 = elo_predictions(matches_2018, elo_2018, HOST_BY_YEAR[2018], c, mu, rho)
                preds_2022 = elo_predictions(matches_2022, elo_2022, HOST_BY_YEAR[2022], c, mu, rho)
                preds = np.vstack([preds_2018, preds_2022])
                cell = evaluate(preds, y_90)
                cell["c_elo"] = c
                cell["mu"] = mu
                cell["rho"] = rho
                grid.append(cell)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "sweep.json").open("w") as f:
        json.dump({"grid": grid, "gate": {"rps": GATE_RPS, "cal_tolerance": GATE_CAL_TOLERANCE}}, f, indent=2)

    fully = [c for c in grid if c["fully_passes"]]
    print(f"\n{len(fully)} of {len(grid)} cells fully pass "
          f"(rps_90min < {GATE_RPS} AND every decile with n≥{GATE_CAL_MIN_N} within ±{GATE_CAL_TOLERANCE} of diagonal).")

    # Current baseline (rho=0 reproduces the v1.6 independent-Poisson result).
    baseline = next(
        c for c in grid
        if c["c_elo"] == 300 and abs(c["mu"] - 1.35) < 1e-6 and abs(c["rho"]) < 1e-6
    )
    print(f"\nBaseline (c_elo=300, μ=1.35, ρ=0): rps={baseline['rps_90min']:.4f}  "
          f"worst_err={baseline['worst_decile_err']:.4f} (decile {baseline['worst_decile_idx']})  "
          f"fully_passes={baseline['fully_passes']}")

    if fully:
        # Rank fully-passing by tightest calibration margin, then by lowest RPS,
        # then prefer settings closer to the PRD defaults (smaller change).
        def closeness(c):
            return (
                c["worst_decile_err"],
                c["rps_90min"],
                abs(c["c_elo"] - 300) / 100.0 + abs(c["mu"] - 1.35) + abs(c["rho"]) * 5,
            )
        ranked = sorted(fully, key=closeness)
        print("\nTop 10 fully-passing settings:")
        print(f"  {'c_elo':>5}  {'mu':>5}  {'rho':>6}  {'rps':>7}  {'worst_err':>9}")
        for c in ranked[:10]:
            print(f"  {c['c_elo']:>5}  {c['mu']:>5.2f}  {c['rho']:>+6.3f}  "
                  f"{c['rps_90min']:>7.4f}  {c['worst_decile_err']:>9.4f}")
        best = ranked[0]
        print(f"\nRECOMMENDED DEFAULTS:  c_elo = {best['c_elo']}  μ = {best['mu']:.2f}  ρ = {best['rho']:+.3f}")
        print(f"  rps_90min = {best['rps_90min']:.4f}  worst calibration error = {best['worst_decile_err']:.4f}")
        return 0
    else:
        # No fully-passing cell. Find closest-to-passing.
        rps_passing = [c for c in grid if c["rps_passes"]]
        print(f"\nNo (c, μ, ρ) setting fully passes. {len(rps_passing)} of {len(grid)} pass the RPS gate.")
        ranked = sorted(rps_passing, key=lambda c: c["worst_decile_err"])
        print("\nTop 5 closest-to-passing (RPS passes; calibration sorted by worst-decile error):")
        print(f"  {'c_elo':>5}  {'mu':>5}  {'rho':>6}  {'rps':>7}  {'worst_err':>9}  {'decile':>6}")
        for c in ranked[:5]:
            print(f"  {c['c_elo']:>5}  {c['mu']:>5.2f}  {c['rho']:>+6.3f}  "
                  f"{c['rps_90min']:>7.4f}  {c['worst_decile_err']:>9.4f}  {c['worst_decile_idx']:>6}")
        print("\nCONCLUSION: Dixon-Coles alone is not sufficient. Further structural "
              "change required (multiplicative Poisson or larger calibration sample).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
