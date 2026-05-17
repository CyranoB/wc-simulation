"""Elo scraper for eloratings.net.

For each World Cup participant, fetches the per-team TSV at
`https://www.eloratings.net/<TeamName>.tsv` (spaces → underscores), finds the
team's last match strictly before the target snapshot date, and extracts the
team's post-match Elo rating. The TSV row format is:

    year  month  day  home_code  away_code  home_goals  away_goals  type  venue
    points_exchanged  home_post_elo  away_post_elo  home_rank_chg  away_rank_chg
    home_post_rank  away_post_rank

Confirmed empirically by cross-referencing consecutive matches for the same
team (e.g., Brazil 2026-03-26 post-Elo = 1970 = Brazil 2026-03-31 pre-Elo).

Output: `data/raw/elo_history.csv` with columns (date, team, rating), one row
per (team, snapshot_date) pair. Teams that played both tournaments appear once
per tournament with their respective pre-tournament Elo.
"""
from __future__ import annotations

import csv
import sys
import time
import urllib.request
from pathlib import Path

# WC participants. Each entry is (eloratings_name, display_name).
# eloratings_name is the URL slug (underscores for spaces, ASCII only — per
# the convention in ratings.js line 856). display_name is what we write to
# the CSV so name_to_iso3.py can map it.
WC_2018 = [
    ("Argentina", "Argentina"), ("Australia", "Australia"), ("Belgium", "Belgium"),
    ("Brazil", "Brazil"), ("Colombia", "Colombia"), ("Costa_Rica", "Costa Rica"),
    ("Croatia", "Croatia"), ("Denmark", "Denmark"), ("Egypt", "Egypt"),
    ("England", "England"), ("France", "France"), ("Germany", "Germany"),
    ("Iceland", "Iceland"), ("Iran", "Iran"), ("Japan", "Japan"),
    ("South_Korea", "South Korea"), ("Mexico", "Mexico"), ("Morocco", "Morocco"),
    ("Nigeria", "Nigeria"), ("Panama", "Panama"), ("Peru", "Peru"),
    ("Poland", "Poland"), ("Portugal", "Portugal"), ("Russia", "Russia"),
    ("Saudi_Arabia", "Saudi Arabia"), ("Senegal", "Senegal"), ("Serbia", "Serbia"),
    ("Spain", "Spain"), ("Sweden", "Sweden"), ("Switzerland", "Switzerland"),
    ("Tunisia", "Tunisia"), ("Uruguay", "Uruguay"),
]

WC_2022 = [
    ("Argentina", "Argentina"), ("Australia", "Australia"), ("Belgium", "Belgium"),
    ("Brazil", "Brazil"), ("Cameroon", "Cameroon"), ("Canada", "Canada"),
    ("Costa_Rica", "Costa Rica"), ("Croatia", "Croatia"), ("Denmark", "Denmark"),
    ("Ecuador", "Ecuador"), ("England", "England"), ("France", "France"),
    ("Germany", "Germany"), ("Ghana", "Ghana"), ("Iran", "Iran"),
    ("Japan", "Japan"), ("South_Korea", "South Korea"), ("Mexico", "Mexico"),
    ("Morocco", "Morocco"), ("Netherlands", "Netherlands"), ("Poland", "Poland"),
    ("Portugal", "Portugal"), ("Qatar", "Qatar"), ("Saudi_Arabia", "Saudi Arabia"),
    ("Senegal", "Senegal"), ("Serbia", "Serbia"), ("Spain", "Spain"),
    ("Switzerland", "Switzerland"), ("Tunisia", "Tunisia"),
    ("United_States", "United States"), ("Uruguay", "Uruguay"), ("Wales", "Wales"),
]

WC_2026 = [
    # Group A
    ("Mexico", "Mexico"), ("South_Africa", "South Africa"),
    ("South_Korea", "South Korea"), ("Denmark", "Denmark"),
    # Group B
    ("Canada", "Canada"), ("Italy", "Italy"),
    ("Qatar", "Qatar"), ("Switzerland", "Switzerland"),
    # Group C
    ("Brazil", "Brazil"), ("Morocco", "Morocco"),
    ("Haiti", "Haiti"), ("Scotland", "Scotland"),
    # Group D
    ("United_States", "United States"), ("Paraguay", "Paraguay"),
    ("Australia", "Australia"), ("Turkey", "Turkey"),
    # Group E
    ("Germany", "Germany"), ("Curacao", "Curacao"),
    ("Ivory_Coast", "Ivory Coast"), ("Ecuador", "Ecuador"),
    # Group F
    ("Netherlands", "Netherlands"), ("Japan", "Japan"),
    ("Ukraine", "Ukraine"), ("Tunisia", "Tunisia"),
    # Group G
    ("Belgium", "Belgium"), ("Egypt", "Egypt"),
    ("Iran", "Iran"), ("New_Zealand", "New Zealand"),
    # Group H
    ("Spain", "Spain"), ("Cape_Verde", "Cape Verde"),
    ("Saudi_Arabia", "Saudi Arabia"), ("Uruguay", "Uruguay"),
    # Group I
    ("France", "France"), ("Senegal", "Senegal"),
    ("Bolivia", "Bolivia"), ("Norway", "Norway"),
    # Group J
    ("Argentina", "Argentina"), ("Algeria", "Algeria"),
    ("Austria", "Austria"), ("Jordan", "Jordan"),
    # Group K
    ("Portugal", "Portugal"), ("Jamaica", "Jamaica"),
    ("Uzbekistan", "Uzbekistan"), ("Colombia", "Colombia"),
    # Group L
    ("England", "England"), ("Croatia", "Croatia"),
    ("Ghana", "Ghana"), ("Panama", "Panama"),
]

SNAPSHOTS = [
    # (snapshot_date, [(eloratings_name, display_name), ...])
    # Day before tournament opening match. Captures the team's Elo after their
    # last pre-tournament fixture (typically a warm-up friendly).
    ("2018-06-13", WC_2018),  # WC 2018 opened 2018-06-14
    ("2022-11-19", WC_2022),  # WC 2022 opened 2022-11-20
    ("2026-06-10", WC_2026),  # WC 2026 opens 2026-06-11
]

ELORATINGS_BASE = "https://www.eloratings.net"
USER_AGENT = "Mozilla/5.0 (wcsim Spike 1 validation back-test; +eddie@pick.fr)"
REQUEST_DELAY_S = 0.5  # politeness: one fetch every 0.5s
HERE = Path(__file__).parent
OUTPUT = HERE.parent / "data" / "raw" / "elo_history.csv"

# Two-letter ISO codes for the host countries (used to skip rows from neutral
# tournaments that aren't relevant — actually unused right now; we just want
# the team's most recent post-match Elo regardless of opponent).


def fetch_team_tsv(eloratings_name: str) -> list[list[str]]:
    """Fetch one team's full match history from eloratings.net.

    Returns a list of TSV rows (each row is a list of string fields). Raises
    urllib.error.HTTPError on 404 / 5xx so the caller knows the team name
    didn't resolve.
    """
    url = f"{ELORATINGS_BASE}/{eloratings_name}.tsv"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return [line.split("\t") for line in body.splitlines() if line]


def extract_elo_before(rows: list[list[str]], snapshot_date: str, team_code: str) -> float | None:
    """Return the team's post-match Elo on the latest match strictly before
    snapshot_date. team_code is the 2-letter eloratings.net code (so we know
    whether to read column 11 [home post-Elo] or column 12 [away post-Elo]).
    """
    target_y, target_m, target_d = (int(x) for x in snapshot_date.split("-"))
    target_tuple = (target_y, target_m, target_d)
    latest_elo: float | None = None
    for row in rows:
        if len(row) < 12:
            continue
        try:
            y, m, d = int(row[0]), int(row[1]), int(row[2])
        except ValueError:
            continue
        if (y, m, d) >= target_tuple:
            break  # rows are chronological
        home_code, away_code = row[3], row[4]
        if home_code == team_code:
            latest_elo = float(row[10])  # col 11 (0-indexed 10): home post-Elo
        elif away_code == team_code:
            latest_elo = float(row[11])  # col 12 (0-indexed 11): away post-Elo
        # else: this team_code didn't play, but the row shouldn't be here at
        # all if we fetched the right team's TSV. Defensive skip.
    return latest_elo


def infer_team_code(rows: list[list[str]]) -> str:
    """Find the 2-letter code that appears most often in the home/away columns
    of this team's TSV (it's the team's own code, since the file is per-team)."""
    from collections import Counter
    codes = Counter()
    for row in rows:
        if len(row) < 5:
            continue
        codes[row[3]] += 1
        codes[row[4]] += 1
    # The team's own code appears in every row, opponents appear at most once
    # per match each. So the most common code is the team's code.
    return codes.most_common(1)[0][0]


def scrape() -> int:
    """Fetch all required team TSVs, extract pre-WC Elo, write CSV.

    Returns the number of (team, snapshot) rows written. Exit code 0 on
    success, 1 if any team failed to resolve.
    """
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Dedupe team list — many teams played both WCs; fetch each TSV once.
    unique_teams: dict[str, str] = {}  # eloratings_name -> display_name
    for _, team_list in SNAPSHOTS:
        for elo_name, display in team_list:
            unique_teams[elo_name] = display
    print(f"Fetching {len(unique_teams)} unique team TSVs from eloratings.net "
          f"(estimated {len(unique_teams) * REQUEST_DELAY_S:.0f}s with rate-limit)...")

    team_data: dict[str, tuple[list[list[str]], str]] = {}  # elo_name -> (rows, code)
    errors: list[str] = []
    for i, (elo_name, _display) in enumerate(unique_teams.items(), 1):
        try:
            rows = fetch_team_tsv(elo_name)
            code = infer_team_code(rows)
            team_data[elo_name] = (rows, code)
            print(f"  [{i}/{len(unique_teams)}] {elo_name} -> {code} ({len(rows)} matches)")
        except Exception as e:
            errors.append(f"{elo_name}: {e}")
            print(f"  [{i}/{len(unique_teams)}] {elo_name}: FAILED ({e})", file=sys.stderr)
        time.sleep(REQUEST_DELAY_S)

    if errors:
        print(f"\n{len(errors)} team(s) failed to fetch:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    rows_written = 0
    with OUTPUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "team", "rating"])
        for snapshot_date, team_list in SNAPSHOTS:
            missing: list[str] = []
            for elo_name, display in team_list:
                rows, code = team_data[elo_name]
                elo = extract_elo_before(rows, snapshot_date, code)
                if elo is None:
                    missing.append(f"{display} ({elo_name}/{code})")
                    continue
                w.writerow([snapshot_date, display, f"{elo:.1f}"])
                rows_written += 1
            if missing:
                print(f"WARNING: no pre-{snapshot_date} match found for: {missing}",
                      file=sys.stderr)

    print(f"\nWrote {rows_written} rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(scrape())
