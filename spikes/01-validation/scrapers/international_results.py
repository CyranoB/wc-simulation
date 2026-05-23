"""Download broad international match results from martj42/international_results."""
from __future__ import annotations

import csv
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
USER_AGENT = "Mozilla/5.0 (wcsim Dixon-Coles validation spike)"
HERE = Path(__file__).parent
OUTPUT = HERE.parent / "data" / "raw" / "international_results.csv"
EXPECTED_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]


def fetch_csv(url: str = URL) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def validate_schema(csv_text: str) -> None:
    header = next(csv.reader(csv_text.splitlines()))
    if header != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected martj42 schema: {header!r}")


def scrape() -> int:
    text = fetch_csv()
    validate_schema(text)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text)
    rows = max(len(text.splitlines()) - 1, 0)
    print(f"Wrote {rows} rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(scrape())
