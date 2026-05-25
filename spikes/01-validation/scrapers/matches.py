"""WC match-results downloader for openfootball/worldcup.json.

This isn't really a scraper — openfootball publishes clean JSON dumps over
HTTPS, so we just download and reshape into the CSV format that load_matches
in validate.py expects.

openfootball format per match:
    date, time, round, team1, team2,
    score: {ft: [h, a], ht: [h, a], et: [h, a] (optional), p: [h, a] (optional)},
    goals1, goals2, group, ground

martj42 / load_matches format (target):
    date, home_team, away_team, home_score, away_score,
    tournament, city, country, neutral

Notes on the score columns:
- `home_score` / `away_score` carry the POST-ET total for knockout matches
  (matches martj42 convention; the regulation/ET split lives in
  knockout_supplement.csv).
- All WC matches except those involving the host nation are at neutral
  venues; we set neutral=True everywhere and rely on the host-bonus logic in
  predict() to handle host games via HOST_BY_YEAR.
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

SOURCES = {
    2018: "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2018/worldcup.json",
    2022: "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2022/worldcup.json",
}
USER_AGENT = "Mozilla/5.0 (wcsim Spike 1 validation back-test; +eddie@pick.fr)"
SOURCE_HOST = "raw.githubusercontent.com"
HERE = Path(__file__).parent
OUTPUT = HERE.parent / "data" / "raw" / "matches_history.csv"


def fetch_json(url: str) -> dict:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != SOURCE_HOST:
        raise ValueError(f"Unexpected match data URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def team_name(t) -> str:
    """openfootball stores team as either a plain string or {name, code}."""
    if isinstance(t, dict):
        return t.get("name") or t.get("code") or str(t)
    return str(t)


def scrape() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with OUTPUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "home_team", "away_team", "home_score", "away_score",
                    "tournament", "city", "country", "neutral"])
        for year, url in SOURCES.items():
            print(f"Fetching WC {year} matches from openfootball...")
            data = fetch_json(url)
            matches = data.get("matches", [])
            print(f"  {len(matches)} matches in {data.get('name', year)}")
            for m in matches:
                score = m.get("score") or {}
                ft = score.get("ft") or [None, None]
                et = score.get("et")  # may be None for non-ET matches
                final = et if et else ft  # post-ET if it went to ET, else FT
                home_score, away_score = final
                if home_score is None or away_score is None:
                    # Future / cancelled match — shouldn't happen for finished WCs.
                    continue
                w.writerow([
                    m["date"],
                    team_name(m.get("team1")),
                    team_name(m.get("team2")),
                    int(home_score),
                    int(away_score),
                    "FIFA World Cup",
                    (m.get("ground") or {}).get("city", "") if isinstance(m.get("ground"), dict) else "",
                    (m.get("ground") or {}).get("country", "") if isinstance(m.get("ground"), dict) else "",
                    "True",  # WC matches are at neutral venues except for hosts; host-bonus applies via HOST_BY_YEAR
                ])
                rows_written += 1

    print(f"\nWrote {rows_written} rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(scrape())
