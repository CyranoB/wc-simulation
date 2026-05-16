"""Report formatters: table, CSV, JSON. Plus Wilson CI computation."""
from __future__ import annotations
import csv
import io
import json
import math
from .types import SimulationResult


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for proportion p at sample size n."""
    if n == 0:
        return (0.0, 1.0)
    if p == 0.0:
        return (0.0, 1.0 - (1.0 - 0.95) ** (1.0 / n))
    if p == 1.0:
        return ((1.0 - 0.95) ** (1.0 / n), 1.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def format_table(result: SimulationResult, verbose: bool = False) -> str:
    """Plain-text table for stdout."""
    if not result.probabilities:
        return "(no results)"
    stages = list(next(iter(result.probabilities.values())).keys())
    header = f"{'Team':<6}" + "".join(f"{s:>10}" for s in stages)
    lines = [header, "-" * len(header)]
    sorted_teams = sorted(result.probabilities.items(), key=lambda kv: -kv[1].get("Win", 0))
    for iso3, probs in sorted_teams:
        row = f"{iso3:<6}"
        for stage in stages:
            pct = probs.get(stage, 0) * 100
            row += f"{pct:>9.1f}%"
        lines.append(row)
    return "\n".join(lines)


def format_csv(result: SimulationResult, include_ci: bool = True) -> str:
    """CSV with one row per team."""
    if not result.probabilities:
        return ""
    stages = list(next(iter(result.probabilities.values())).keys())
    buf = io.StringIO()
    fieldnames = ["team"]
    for s in stages:
        fieldnames.append(s)
        if include_ci:
            fieldnames.extend([f"{s}_ci_lo", f"{s}_ci_hi"])
    fieldnames.extend(["mean_goals_for", "mean_goals_against", "simulations", "seed"])
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for iso3 in sorted(result.probabilities.keys()):
        row_dict: dict = {"team": iso3}
        for s in stages:
            row_dict[s] = f"{result.probabilities[iso3].get(s, 0):.6f}"
            if include_ci:
                row_dict[f"{s}_ci_lo"] = f"{result.ci_lo[iso3].get(s, 0):.6f}"
                row_dict[f"{s}_ci_hi"] = f"{result.ci_hi[iso3].get(s, 0):.6f}"
        row_dict["mean_goals_for"] = f"{result.mean_goals_for.get(iso3, 0):.4f}"
        row_dict["mean_goals_against"] = f"{result.mean_goals_against.get(iso3, 0):.4f}"
        row_dict["simulations"] = result.n
        row_dict["seed"] = result.seed
        w.writerow(row_dict)
    return buf.getvalue()


def format_json(result: SimulationResult, meta_det: dict, meta_env: dict) -> str:
    """JSON with meta blocks + rows."""
    rows = []
    stages = list(next(iter(result.probabilities.values())).keys()) if result.probabilities else []
    for iso3 in sorted(result.probabilities.keys()):
        row = {"team": iso3}
        for s in stages:
            row[s] = result.probabilities[iso3].get(s, 0)
            row[f"{s}_ci_lo"] = result.ci_lo[iso3].get(s, 0)
            row[f"{s}_ci_hi"] = result.ci_hi[iso3].get(s, 0)
        row["mean_goals_for"] = result.mean_goals_for.get(iso3, 0)
        row["mean_goals_against"] = result.mean_goals_against.get(iso3, 0)
        rows.append(row)
    return json.dumps({"meta": {"deterministic": meta_det, "environment": meta_env}, "rows": rows}, indent=2)
