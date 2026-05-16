"""FIFA Men's World Ranking scraper for inside.fifa.com.

Hits the hidden API documented in the cnc8 + hericlibong scrapers:

1. GET https://inside.fifa.com/fifa-world-ranking/men  (HTML)
   Extract the embedded `<script id="__NEXT_DATA__">` JSON. The list of
   historical rankings lives at:
       props.pageProps.pageData.ranking.dates
   Each entry is a year object with a `dates` array of {id, iso}.

2. For each target tournament, find the latest ranking date strictly before
   the tournament's opening match.

3. GET https://inside.fifa.com/api/ranking-overview?locale=en&dateId=<id>
   The response is JSON with a `rankings` array where each entry has:
       rankingItem.name           team name
       rankingItem.rank           current rank
       rankingItem.previousRank   prior rank
       rankingItem.totalPoints    FIFA points
       previousPoints             prior points
       tag.text                   confederation

Output: `data/raw/fifa_ranking.csv` with columns matching what `load_fifa()`
expects (rank_date, country_full, total_points, rank).
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

FIFA_BASE = "https://inside.fifa.com"
FIFA_LANDING = f"{FIFA_BASE}/fifa-world-ranking/men"
USER_AGENT = "Mozilla/5.0 (wcsim Spike 1 validation back-test; +eddie@pick.fr)"
HERE = Path(__file__).parent
OUTPUT = HERE.parent / "data" / "raw" / "fifa_ranking.csv"
REQUEST_DELAY_S = 1.0  # politeness for FIFA's API

# Target snapshots: latest ranking strictly before tournament opening day.
TARGETS = [
    ("2018-06-13", "WC 2018 (opens 2018-06-14)"),
    ("2022-11-19", "WC 2022 (opens 2022-11-20)"),
]


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_json(url: str) -> dict:
    return json.loads(fetch_text(url))


def extract_next_data(html: str) -> dict:
    """Parse the embedded `<script id="__NEXT_DATA__" type="application/json">`
    block from the FIFA landing page."""
    m = re.search(
        r'<script[^>]+id="__NEXT_DATA__"[^>]+>(.*?)</script>',
        html, flags=re.DOTALL,
    )
    if not m:
        raise RuntimeError("__NEXT_DATA__ script tag not found in FIFA landing HTML")
    return json.loads(m.group(1))


def all_ranking_dates(next_data: dict) -> list[dict]:
    """Flatten the year-grouped dates array into a single list of {id, iso}."""
    ranking = next_data["props"]["pageProps"]["pageData"]["ranking"]
    flat = []
    for year_entry in ranking["dates"]:
        for d in year_entry["dates"]:
            flat.append({"id": d["id"], "iso": d["iso"]})  # iso looks like "2018-06-07T00:00:00.000Z"
    return flat


def iso_to_date(iso: str) -> str:
    """'2018-06-07T00:00:00.000Z' -> '2018-06-07'."""
    return iso.split("T", 1)[0]


def pick_latest_before(dates: list[dict], target: str) -> dict:
    """Among the (id, iso) dates, pick the one whose date-only iso is the
    latest <= target. Raises if none qualify."""
    eligible = [d for d in dates if iso_to_date(d["iso"]) <= target]
    if not eligible:
        raise RuntimeError(f"No FIFA ranking date <= {target}")
    return max(eligible, key=lambda d: iso_to_date(d["iso"]))


def fetch_ranking(date_id: str) -> dict:
    url = f"{FIFA_BASE}/api/ranking-overview?locale=en&dateId={date_id}"
    return fetch_json(url)


def scrape() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching FIFA landing page to discover historical date IDs...")
    landing_html = fetch_text(FIFA_LANDING)
    next_data = extract_next_data(landing_html)
    dates = all_ranking_dates(next_data)
    print(f"  Found {len(dates)} historical ranking snapshots (oldest: "
          f"{iso_to_date(min(dates, key=lambda d: d['iso'])['iso'])}, newest: "
          f"{iso_to_date(max(dates, key=lambda d: d['iso'])['iso'])})")
    time.sleep(REQUEST_DELAY_S)

    fetches: list[tuple[str, dict]] = []  # (rank_date_iso, payload)
    for target_date, label in TARGETS:
        chosen = pick_latest_before(dates, target_date)
        chosen_date = iso_to_date(chosen["iso"])
        print(f"  Target {target_date} [{label}] -> using ranking from {chosen_date} (id={chosen['id']})")
        payload = fetch_ranking(chosen["id"])
        fetches.append((chosen_date, payload))
        time.sleep(REQUEST_DELAY_S)

    rows_written = 0
    with OUTPUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank_date", "country_full", "total_points", "rank",
                    "previous_points", "previous_rank", "confederation"])
        for rank_date, payload in fetches:
            rankings = payload.get("rankings", [])
            if not rankings:
                print(f"WARNING: empty rankings array for {rank_date}", file=sys.stderr)
                continue
            for item in rankings:
                ri = item.get("rankingItem", {})
                w.writerow([
                    rank_date,
                    ri.get("name", ""),
                    ri.get("totalPoints", ""),
                    ri.get("rank", ""),
                    item.get("previousPoints", ""),
                    ri.get("previousRank", ""),
                    (item.get("tag") or {}).get("text", ""),
                ])
                rows_written += 1

    print(f"\nWrote {rows_written} rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(scrape())
