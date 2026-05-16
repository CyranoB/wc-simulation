"""CLI entry point using Typer. Run via `python -m wcsim.cli`."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional
import typer

app = typer.Typer(name="wcsim", help="Football Tournament Monte Carlo Simulator")


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
    from .data import load_teams, load_draw, DEFAULT_TEAMS_PATH, DEFAULT_DRAW_PATH
    from .ratings.elo import EloRating
    from .ratings.fifa import FifaRating
    from .ratings.blend import BlendRating
    from .sim import run_simulations
    from .report import format_table, format_csv, format_json
    from .cache import write_cache
    from .types import Params
    import random as _random

    tp = teams_path or DEFAULT_TEAMS_PATH
    dp = draw_path or DEFAULT_DRAW_PATH
    teams = load_teams(tp)
    draw = load_draw(dp)
    hosts = {"USA", "MEX", "CAN"}

    params = Params()
    rating_cls = {"elo": EloRating, "fifa": FifaRating, "blend": BlendRating}[rating_mode]
    rating = rating_cls(params)

    actual_seed = seed if seed is not None else _random.randint(0, 2**31)
    if not quiet:
        typer.echo(f"Running {n} simulations (seed={actual_seed}, workers={workers or 'auto'})...")

    start = time.time()
    result = run_simulations(
        teams=teams, draw=draw, hosts=hosts,
        rating=rating, params=params, n=n, seed=actual_seed, workers=workers,
    )
    elapsed = time.time() - start
    if not quiet:
        typer.echo(f"Done in {elapsed:.1f}s.")

    if fmt == "csv" or (out and str(out).endswith(".csv")):
        output = format_csv(result, include_ci=ci)
    elif fmt == "json" or (out and str(out).endswith(".json")):
        meta_det = {"seed": actual_seed, "simulations": n, "rating_mode": rating_mode}
        meta_env = {"runtime_seconds": elapsed}
        output = format_json(result, meta_det, meta_env)
    else:
        output = format_table(result, verbose=verbose)

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

    a_is_host = (home == "A") if home else False
    b_is_host = (home == "B") if home else False
    if neutral:
        a_is_host = b_is_host = False

    p = predict_match(t_a, t_b, rating=rating, params=params,
                      a_is_host=a_is_host, b_is_host=b_is_host)
    typer.echo(f"{team_a} vs {team_b} ({rating_mode}, {'neutral' if neutral else 'home=' + (home or 'A')}):")
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
