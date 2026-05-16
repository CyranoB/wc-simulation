"""CLI entry point using Typer. Run via `python -m wcsim.cli`."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional
import typer

app = typer.Typer(name="wcsim", help="Football Tournament Monte Carlo Simulator")


def _make_rating(rating_mode: str, params):
    from .ratings.elo import EloRating
    from .ratings.fifa import FifaRating
    from .ratings.blend import BlendRating
    return {"elo": EloRating, "fifa": FifaRating, "blend": BlendRating}[rating_mode](params)


def _format_output(result, fmt, out, ci, verbose, actual_seed, n, rating_mode, elapsed):
    from .report import format_table, format_csv, format_json
    if fmt == "csv" or (out and str(out).endswith(".csv")):
        return format_csv(result, include_ci=ci)
    if fmt == "json" or (out and str(out).endswith(".json")):
        meta_det = {"seed": actual_seed, "simulations": n, "rating_mode": rating_mode}
        meta_env = {"runtime_seconds": elapsed}
        return format_json(result, meta_det, meta_env)
    return format_table(result, verbose=verbose)


@app.command()
def run(
    n: int = typer.Option(100_000, "-n", "--simulations"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    teams_path: Optional[Path] = typer.Option(None, "--teams"),
    draw_path: Optional[Path] = typer.Option(None, "--draw"),
    rating_mode: str = typer.Option("elo", "--rating"),
    out: Optional[Path] = typer.Option(None, "--out"),
    fmt: str = typer.Option("table", "--format"),
    workers: Optional[int] = typer.Option(None, "--workers"),
    ci: bool = typer.Option(True, "--ci/--no-ci"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
):
    """Run Monte Carlo tournament simulation."""
    from .data import load_teams, load_draw, load_venues, DEFAULT_TEAMS_PATH, DEFAULT_DRAW_PATH
    from .sim import run_simulations
    from .cache import write_cache
    from .types import Params
    import random as _random

    teams = load_teams(teams_path or DEFAULT_TEAMS_PATH)
    draw = load_draw(draw_path or DEFAULT_DRAW_PATH)
    venues = load_venues()
    hosts = {"USA", "MEX", "CAN"}
    group_venues = venues.group_venues if venues else None
    knockout_host_iso3 = venues.knockout_venue if venues else None
    params = Params()
    rating = _make_rating(rating_mode, params)

    actual_seed = seed if seed is not None else _random.randint(0, 2**31)
    if not quiet:
        typer.echo(f"Running {n} simulations (seed={actual_seed}, workers={workers or 'auto'})...")

    start = time.time()
    result = run_simulations(
        teams=teams, draw=draw, hosts=hosts,
        rating=rating, params=params, n=n, seed=actual_seed, workers=workers,
        group_venues=group_venues, knockout_host=knockout_host_iso3,
    )
    elapsed = time.time() - start
    if not quiet:
        typer.echo(f"Done in {elapsed:.1f}s.")

    output = _format_output(result, fmt, out, ci, verbose, actual_seed, n, rating_mode, elapsed)
    if out:
        out.write_text(output)
        if not quiet:
            typer.echo(f"Written to {out}")
    else:
        typer.echo(output)

    meta_det = {"seed": actual_seed, "simulations": n, "rating_mode": rating_mode}
    meta_env = {"runtime_seconds": elapsed}
    write_cache(result, meta_det, meta_env)


@app.command()
def match(
    team_a: str = typer.Argument(...),
    team_b: str = typer.Argument(...),
    neutral: bool = typer.Option(False, "--neutral"),
    home: Optional[str] = typer.Option(None, "--home"),
    rating_mode: str = typer.Option("elo", "--rating"),
):
    """Print win/draw/loss probabilities for a single match."""
    from .data import load_teams, DEFAULT_TEAMS_PATH
    from .model import predict_match
    from .ratings.elo import EloRating
    from .ratings.fifa import FifaRating
    from .ratings.blend import BlendRating
    from .types import Params
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "spikes" / "01-validation"))
    from name_to_iso3 import to_iso3

    teams = load_teams(DEFAULT_TEAMS_PATH)
    params = Params()
    rating_cls = {"elo": EloRating, "fifa": FifaRating, "blend": BlendRating}[rating_mode]
    rating = rating_cls(params)

    iso_a = to_iso3(team_a)
    iso_b = to_iso3(team_b)
    t_a, t_b = teams[iso_a], teams[iso_b]

    effective_home = "none" if neutral else (home or "A")
    a_is_host = effective_home == "A"
    b_is_host = effective_home == "B"

    p = predict_match(t_a, t_b, rating=rating, params=params,
                      a_is_host=a_is_host, b_is_host=b_is_host)
    venue = "neutral" if effective_home == "none" else f"home={effective_home}"
    typer.echo(f"{team_a} vs {team_b} ({rating_mode}, {venue}):")
    typer.echo(f"  {team_a} wins: {p[0]*100:.1f}%")
    typer.echo(f"  Draw:         {p[1]*100:.1f}%")
    typer.echo(f"  {team_b} wins: {p[2]*100:.1f}%")


@app.command()
def teams(teams_path: Optional[Path] = typer.Option(None, "--teams")):
    """List loaded teams with Elo rating."""
    from .data import load_teams, DEFAULT_TEAMS_PATH
    all_teams = load_teams(teams_path or DEFAULT_TEAMS_PATH)
    typer.echo(f"{'ISO3':<6}{'Name':<25}{'Elo':>8}")
    typer.echo("-" * 39)
    for iso3, t in sorted(all_teams.items(), key=lambda kv: -kv[1].elo):
        typer.echo(f"{iso3:<6}{t.name:<25}{t.elo:>8.1f}")


@app.command()
def version():
    """Print version."""
    typer.echo("wcsim 0.3.0-dev (Spike 3)")


if __name__ == "__main__":
    app()
