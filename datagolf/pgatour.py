"""
PGA Tour stats fetcher.

Pulls stat leaderboards from the PGA Tour's orchestrator GraphQL API
(the same API the pgatour.com website calls internally).

Each stat is identified by a numeric ID found in the pgatour.com URL:
  https://www.pgatour.com/stats/detail/02564  →  stat_id = "02564"

Usage:
    from datagolf.pgatour import PGATourStats
    client = PGATourStats()
    df = client.get_combined_stat("02564", label="SG: T2G")
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional
from datetime import date, datetime


def auto_season_weight(today: date = None) -> tuple[float, str]:
    """
    Calculate an appropriate current-season blend weight based on where
    we are in the PGA Tour season.

    The PGA Tour season runs approximately Oct 15 → Aug 31 each year.
    Weight ramps linearly from 0.10 (season start) to 0.90 (season end).

      Early season (Oct–Nov): ~0.10–0.20  — little current data, lean on prev season
      Mid season   (Jan–Apr): ~0.30–0.60  — balanced blend
      Late season  (May–Aug): ~0.65–0.90  — current season dominates

    Returns (weight, description_string).
    """
    if today is None:
        today = date.today()

    month = today.month
    year  = today.year

    # PGA Tour season: starts ~Oct 15, ends ~Aug 31
    # If we're in Sep, we're between seasons — clamp to just-started
    if month == 9:
        return 0.05, "off-season / between seasons (Sep)"

    # Season start: Oct 15 of current year (if Oct–Dec) or previous year (if Jan–Aug)
    if month >= 10:
        season_start = date(year,     10, 15)
        season_end   = date(year + 1,  8, 31)
    else:
        season_start = date(year - 1, 10, 15)
        season_end   = date(year,      8, 31)

    total_days   = (season_end   - season_start).days
    elapsed_days = (today        - season_start).days
    elapsed_days = max(0, min(elapsed_days, total_days))

    progress = elapsed_days / total_days          # 0.0 → 1.0
    weight   = round(0.10 + progress * 0.80, 2)  # 0.10 → 0.90

    # Human-readable stage
    if progress < 0.20:
        stage = "early season — leaning on previous season"
    elif progress < 0.55:
        stage = "mid season — balanced blend"
    else:
        stage = "late season — current season dominant"

    pct_through = int(progress * 100)
    desc = f"{weight:.0%} current / {1-weight:.0%} previous  ({pct_through}% through season — {stage})"
    return weight, desc

# Public API key embedded in pgatour.com's JavaScript bundle.
# If requests start failing with 401, inspect network traffic on pgatour.com/stats
# to find the updated key.
_API_URL = "https://orchestrator.pgatour.com/graphql"
_API_KEY  = "da2-gsrx5bibzbb4njvhl7t37wqyl4"

_STAT_QUERY = """
query StatDetails($tourCode: TourCode!, $statId: String!, $year: Int) {
  statDetails(tourCode: $tourCode, statId: $statId, year: $year) {
    statId
    statTitle
    rows {
      ... on StatDetailsPlayer {
        playerName
        rank
        stats {
          statValue
          statId
        }
      }
    }
  }
}
"""


_CATEGORIES_QUERY = """
query StatCategories($tourCode: TourCode!) {
  statCategories(tourCode: $tourCode) {
    displayName
    stats {
      statId
      statTitle
    }
  }
}
"""


# Curated fallback stat catalog — used when the API doesn't return categories.
# IDs sourced from pgatour.com/stats/detail/<ID> URLs.
CURATED_STATS = [
    {"category": "Strokes Gained", "stats": [
        {"id": "02675", "title": "SG: Total"},
        {"id": "02674", "title": "SG: Putting"},
        {"id": "02568", "title": "SG: Approach the Green"},
        {"id": "02567", "title": "SG: Off the Tee"},
        {"id": "02569", "title": "SG: Around the Green"},
        {"id": "02564", "title": "SG: Tee-to-Green"},
    ]},
    {"category": "Driving", "stats": [
        {"id": "02330", "title": "Driving Distance"},
        {"id": "02401", "title": "Driving Accuracy %"},
        {"id": "02534", "title": "Total Driving"},
    ]},
    {"category": "Approach the Green", "stats": [
        {"id": "02329", "title": "Greens in Regulation %"},
        {"id": "02388", "title": "Proximity to Hole"},
        {"id": "02393", "title": "Proximity 100-125 yards"},
        {"id": "02394", "title": "Proximity 125-150 yards"},
        {"id": "02395", "title": "Proximity 150-175 yards"},
        {"id": "02396", "title": "Proximity 175-200 yards"},
        {"id": "02397", "title": "Proximity 200+ yards"},
        {"id": "02463", "title": "Proximity 50-125 yards"},
    ]},
    {"category": "Around the Green", "stats": [
        {"id": "02429", "title": "Scrambling"},
        {"id": "02430", "title": "Sand Save %"},
        {"id": "130",   "title": "Scrambling from Sand"},
        {"id": "02431", "title": "Scrambling from Rough"},
    ]},
    {"category": "Putting", "stats": [
        {"id": "02428", "title": "Putts per Round"},
        {"id": "02415", "title": "1-Putt %"},
        {"id": "02416", "title": "3-Putt Avoidance"},
        {"id": "02383", "title": "Putting from 5 feet"},
        {"id": "02384", "title": "Putting from 10 feet"},
        {"id": "02385", "title": "Putting from 15 feet"},
        {"id": "02386", "title": "Putting from 20 feet"},
        {"id": "101",   "title": "One-Putt %"},
    ]},
    {"category": "Scoring", "stats": [
        {"id": "120",   "title": "Scoring Average"},
        {"id": "02511", "title": "Birdie Average"},
        {"id": "02512", "title": "Eagle Average"},
        {"id": "02513", "title": "Bogey Average"},
        {"id": "02333", "title": "Par 3 Scoring Average"},
        {"id": "02334", "title": "Par 4 Scoring Average"},
        {"id": "02335", "title": "Par 5 Scoring Average"},
    ]},
]


class PGATourStats:
    def __init__(self, timeout: int = 20):
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": _API_KEY,
            "x-amz-user-agent": "aws-amplify/3.8.21",
            "Content-Type": "application/json",
        })
        self.timeout = timeout
        self._current_year = datetime.utcnow().year

    def get_stat_categories(self) -> tuple[list[dict], str]:
        """
        Fetch the full stat catalog grouped by category.
        Returns (categories, error_message). categories is empty on failure.
        """
        payload = {
            "query": _CATEGORIES_QUERY,
            "variables": {"tourCode": "R"},
        }
        try:
            resp = self.session.post(_API_URL, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            errors = data.get("errors")
            if errors:
                return [], f"GraphQL error: {errors[0].get('message', errors)}"
            raw = data.get("data", {}).get("statCategories") or []
            if not raw:
                return [], "statCategories returned empty — query may not be supported"
            categories = [
                {
                    "category": cat.get("displayName", "Other"),
                    "stats": [
                        {"id": s["statId"], "title": s["statTitle"]}
                        for s in (cat.get("stats") or [])
                        if s.get("statId") and s.get("statTitle")
                    ],
                }
                for cat in raw
                if cat.get("stats")
            ]
            return categories, ""
        except Exception as exc:
            return [], str(exc)

    def _fetch_stat(self, stat_id: str, year: int = None) -> tuple:
        """Fetch raw stat entries for a stat + optional year. Returns (rows, title)."""
        variables = {"tourCode": "R", "statId": stat_id}
        if year is not None:
            variables["year"] = year
        payload = {"query": _STAT_QUERY, "variables": variables}
        resp = self.session.post(_API_URL, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        details = (data.get("data") or {}).get("statDetails") or {}
        rows = [r for r in (details.get("rows") or []) if r.get("playerName")]
        return rows, details.get("statTitle", "")

    def get_combined_stat(
        self,
        stat_id: str,
        label: Optional[str] = None,
        current_weight: float = 0.6,
    ) -> pd.DataFrame:
        """
        Fetch a stat for the current (2026) and previous (2025) season, blend
        by percentile rank, and return a single combined_rank per player.

        current_weight: 0–1. Weight given to current season (rest goes to previous).

        Returns DataFrame with columns:
            player_name, stat_id, label, combined_rank, cur_rank, cur_value, prev_rank, prev_value
        """
        cur_entries,  title = self._fetch_stat(stat_id, year=2026)
        prev_entries, _     = self._fetch_stat(stat_id, year=2025)

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
# Helpers
# ---------------------------------------------------------------------------

def _entries_to_df(entries: list, suffix: str) -> pd.DataFrame:
    """Convert raw GraphQL stat rows (StatDetailsPlayer) to a normalized DataFrame."""
    rows = []
    for e in entries:
        name = _normalize_name(e.get("playerName") or "")
        if not name:
            continue
        rank = _to_int(e.get("rank"))
        # Primary value: first entry in stats list
        stat_vals = e.get("stats") or []
        value = _to_float(stat_vals[0].get("statValue")) if stat_vals else None
        rows.append({"player_name": name, f"{suffix}_rank": rank, f"{suffix}_value": value})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["player_name", f"{suffix}_rank", f"{suffix}_value"]
    )


def _normalize_name(name: str) -> str:
    """Lowercase, handle 'Last, First' → 'first last'."""
    name = name.strip()
    if "," in name:
        parts = name.split(",", 1)
        name = f"{parts[1].strip()} {parts[0].strip()}"
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
