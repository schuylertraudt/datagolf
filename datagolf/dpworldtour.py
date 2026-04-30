"""
DP World Tour stats fetcher.

Pulls stat leaderboards from europeantour.com stats pages via the
embedded __NEXT_DATA__ JSON (no API key required).

Each stat is identified by a URL slug from the europeantour.com stats pages:
  https://www.europeantour.com/dpworld-tour/stats/{year}/{slug}/
  e.g., slug = "strokes-gained-total" for SG: Total

Usage:
    from datagolf.dpworldtour import DPWorldTourStats
    client = DPWorldTourStats()
    df = client.get_combined_stat("strokes-gained-total", label="SG: Total")
"""

import json
import re
import requests
import pandas as pd
import numpy as np
from typing import Optional
from datetime import date, datetime


def auto_season_weight(today: date = None) -> tuple[float, str]:
    """
    Calculate an appropriate current-season blend weight based on where
    we are in the DP World Tour season.

    The DP World Tour season runs approximately Nov 1 → Sep 30 each year.
    Weight ramps linearly from 0.10 (season start) to 0.90 (season end).

    Returns (weight, description_string).
    """
    if today is None:
        today = date.today()

    month = today.month
    year = today.year

    # Between seasons in October
    if month == 10:
        return 0.05, "off-season / between seasons (Oct)"

    if month >= 11:
        season_start = date(year,     11,  1)
        season_end   = date(year + 1,  9, 30)
    else:
        season_start = date(year - 1, 11,  1)
        season_end   = date(year,      9, 30)

    total_days   = (season_end   - season_start).days
    elapsed_days = (today        - season_start).days
    elapsed_days = max(0, min(elapsed_days, total_days))

    progress = elapsed_days / total_days
    weight   = round(0.10 + progress * 0.80, 2)

    if progress < 0.20:
        stage = "early season — leaning on previous season"
    elif progress < 0.55:
        stage = "mid season — balanced blend"
    else:
        stage = "late season — current season dominant"

    pct_through = int(progress * 100)
    desc = f"{weight:.0%} current / {1-weight:.0%} previous  ({pct_through}% through season — {stage})"
    return weight, desc


_BASE_URL   = "https://www.europeantour.com"
_TOUR_SLUG  = "dpworld-tour"

# Browser-like headers to avoid 403 on direct page fetches
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.europeantour.com/dpworld-tour/stats/",
}


# Curated stat catalog — slugs from europeantour.com/dpworld-tour/stats/{year}/{slug}/
# Verify slugs at: https://www.europeantour.com/dpworld-tour/stats/
DP_CURATED_STATS = [
    {"category": "Strokes Gained", "stats": [
        {"id": "strokes-gained-total",               "title": "SG: Total"},
        {"id": "strokes-gained-putting",             "title": "SG: Putting"},
        {"id": "strokes-gained-approach-to-the-green", "title": "SG: Approach"},
        {"id": "strokes-gained-off-the-tee",         "title": "SG: Off the Tee"},
        {"id": "strokes-gained-around-the-green",    "title": "SG: Around the Green"},
        {"id": "strokes-gained-tee-to-green",        "title": "SG: Tee-to-Green"},
    ]},
    {"category": "Driving", "stats": [
        {"id": "driving-distance", "title": "Driving Distance"},
        {"id": "driving-accuracy", "title": "Driving Accuracy %"},
    ]},
    {"category": "Approach the Green", "stats": [
        {"id": "greens-in-regulation", "title": "Greens in Regulation %"},
        {"id": "proximity-to-hole",    "title": "Proximity to Hole"},
    ]},
    {"category": "Around the Green", "stats": [
        {"id": "scrambling",  "title": "Scrambling"},
        {"id": "sand-saves",  "title": "Sand Save %"},
    ]},
    {"category": "Putting", "stats": [
        {"id": "putts-per-round",        "title": "Putts per Round"},
        {"id": "one-putts",              "title": "1-Putt %"},
        {"id": "three-putts-avoided",    "title": "3-Putt Avoidance"},
    ]},
    {"category": "Scoring", "stats": [
        {"id": "scoring-average",  "title": "Scoring Average"},
        {"id": "birdie-average",   "title": "Birdie Average"},
        {"id": "eagle-average",    "title": "Eagle Average"},
        {"id": "bogey-average",    "title": "Bogey Average"},
    ]},
]


class DPWorldTourStats:
    def __init__(self, timeout: int = 20):
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self.timeout = timeout
        self._current_year = datetime.utcnow().year

    def get_stat_categories(self) -> tuple[list[dict], str]:
        """Return the curated DP World Tour stat catalog."""
        return DP_CURATED_STATS, ""

    def _fetch_stat(self, stat_slug: str, year: int = None) -> tuple[list, str]:
        """
        Fetch player rankings for a stat by URL slug and year.
        Parses the __NEXT_DATA__ JSON embedded in the stats page HTML.
        Returns (rows, title).
        """
        year = year or self._current_year
        url = f"{_BASE_URL}/{_TOUR_SLUG}/stats/{year}/{stat_slug}/"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return _parse_next_data(resp.text, stat_slug)

    def get_combined_stat(
        self,
        stat_id: str,
        label: Optional[str] = None,
        current_weight: float = 0.6,
    ) -> pd.DataFrame:
        """
        Fetch a stat for the current season and previous season, blend by
        percentile rank, and return a single combined_rank per player.

        Same interface as PGATourStats.get_combined_stat().

        Returns DataFrame with columns:
            player_name, stat_id, label, combined_rank, cur_rank, cur_value, prev_rank, prev_value
        """
        cur_entries,  title = self._fetch_stat(stat_id)
        prev_entries, _     = self._fetch_stat(stat_id, year=self._current_year - 1)

        cur_df  = _entries_to_df(cur_entries,  suffix="cur")
        prev_df = _entries_to_df(prev_entries, suffix="prev")

        df = cur_df.merge(prev_df, on="player_name", how="outer")

        n_cur  = df["cur_rank"].notna().sum()
        n_prev = df["prev_rank"].notna().sum()

        if n_cur > 1:
            df["cur_pct"]  = 1 - (df["cur_rank"]  - 1) / (n_cur  - 1)
        else:
            df["cur_pct"]  = np.nan

        if n_prev > 1:
            df["prev_pct"] = 1 - (df["prev_rank"] - 1) / (n_prev - 1)
        else:
            df["prev_pct"] = np.nan

        prev_weight = 1.0 - current_weight
        has_both = df["cur_pct"].notna() & df["prev_pct"].notna()
        has_cur  = df["cur_pct"].notna() & df["prev_pct"].isna()
        has_prev = df["cur_pct"].isna()  & df["prev_pct"].notna()

        df["blended_pct"] = np.nan
        df.loc[has_both, "blended_pct"] = (
            current_weight * df.loc[has_both, "cur_pct"] +
            prev_weight    * df.loc[has_both, "prev_pct"]
        )
        df.loc[has_cur,  "blended_pct"] = df.loc[has_cur,  "cur_pct"]
        df.loc[has_prev, "blended_pct"] = df.loc[has_prev, "prev_pct"]

        df = df.sort_values("blended_pct", ascending=False).reset_index(drop=True)
        df["combined_rank"] = df.index + 1
        df["stat_id"] = stat_id
        df["label"]   = label or title or stat_id

        return df[["player_name", "stat_id", "label", "combined_rank",
                   "cur_rank", "cur_value", "prev_rank", "prev_value"]]


# ---------------------------------------------------------------------------
# __NEXT_DATA__ parser
# ---------------------------------------------------------------------------

def _parse_next_data(html: str, stat_slug: str) -> tuple[list, str]:
    """
    Extract a stat leaderboard from __NEXT_DATA__ JSON embedded in a
    europeantour.com stats page. Returns (rows, title).
    rows = [{"playerName": ..., "rank": ..., "value": ...}]
    """
    m = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        html, re.DOTALL
    )
    if not m:
        return [], ""

    try:
        data = json.loads(m.group(1))
    except Exception:
        return [], ""

    title = _find_title(data) or stat_slug.replace("-", " ").title()
    rows  = _find_leaderboard(data)
    return rows, title


def _find_title(obj, depth: int = 0) -> str:
    """Recursively search for a stat title string in the JSON."""
    if depth > 8:
        return ""
    if isinstance(obj, dict):
        for key in ("statTitle", "statName", "title", "name", "categoryName"):
            v = obj.get(key)
            if isinstance(v, str) and 3 < len(v) < 80:
                return v
        for v in obj.values():
            result = _find_title(v, depth + 1)
            if result:
                return result
    elif isinstance(obj, list):
        for item in obj[:3]:
            result = _find_title(item, depth + 1)
            if result:
                return result
    return ""


def _find_leaderboard(obj, depth: int = 0) -> list:
    """
    Recursively search a JSON structure for a player stat leaderboard.
    Looks for an array of objects with player name + rank fields.
    """
    if depth > 12:
        return []

    if isinstance(obj, list) and len(obj) > 3:
        if obj and isinstance(obj[0], dict):
            keys = set(obj[0].keys())
            has_name = bool(keys & {"playerName", "name", "fullName", "playerfullname",
                                    "playerFullName", "player_name"})
            has_rank = bool(keys & {"rank", "position", "pos", "rankPos"})
            if has_name and has_rank:
                rows = _extract_rows(obj)
                if rows:
                    return rows
        for item in obj:
            result = _find_leaderboard(item, depth + 1)
            if result:
                return result

    elif isinstance(obj, dict):
        # Prioritise keys that sound like a leaderboard container
        priority = ["leaderboard", "stats", "entries", "players", "rows",
                    "statEntries", "data", "results", "leaders"]
        for key in priority:
            if key in obj:
                result = _find_leaderboard(obj[key], depth + 1)
                if result:
                    return result
        for value in obj.values():
            result = _find_leaderboard(value, depth + 1)
            if result:
                return result

    return []


def _extract_rows(entries: list) -> list:
    """Normalise raw JSON player entries to {"playerName", "rank", "value"}."""
    rows = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        name = (
            e.get("playerName") or e.get("playerFullName") or
            e.get("fullName")   or e.get("name") or
            e.get("playerfullname") or e.get("player_name") or ""
        )
        rank_raw = (
            e.get("rank") or e.get("position") or
            e.get("pos")  or e.get("rankPos") or None
        )
        value_raw = (
            e.get("value")     or e.get("statValue") or
            e.get("yards")     or e.get("percentage") or
            e.get("average")   or e.get("metres") or
            e.get("total")     or None
        )
        if name and rank_raw is not None:
            rows.append({"playerName": name, "rank": rank_raw, "value": value_raw})
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entries_to_df(entries: list, suffix: str) -> pd.DataFrame:
    """Convert raw stat rows to a normalised DataFrame."""
    rows = []
    for e in entries:
        name  = _normalize_name(e.get("playerName") or "")
        if not name:
            continue
        rank  = _to_int(e.get("rank"))
        value = _to_float(e.get("value"))
        rows.append({"player_name": name, f"{suffix}_rank": rank, f"{suffix}_value": value})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["player_name", f"{suffix}_rank", f"{suffix}_value"]
    )


def _normalize_name(name: str) -> str:
    """Lowercase, handle 'Last, First' → 'first last'."""
    name = name.strip()
    if "," in name:
        parts = name.split(",", 1)
        name  = f"{parts[1].strip()} {parts[0].strip()}"
    return name.lower()


def _to_int(val) -> Optional[int]:
    try:
        return int(str(val).replace("T", "").replace("-", "").strip())
    except Exception:
        return None


def _to_float(val) -> Optional[float]:
    try:
        return float(str(val).replace(",", "").replace("%", "").strip())
    except Exception:
        return None
