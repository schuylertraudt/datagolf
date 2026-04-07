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
from datetime import datetime

# Public API key embedded in pgatour.com's JavaScript bundle.
# If requests start failing with 401, inspect network traffic on pgatour.com/stats
# to find the updated key.
_API_URL = "https://orchestrator.pgatour.com/graphql"
_API_KEY  = "da2-gsrx5bibzbb4njvhl7t37wqyl4"

_STAT_QUERY = """
query StatDetails($tourCode: TourCode!, $statId: String!, $season: Int) {
  statDetails(tourCode: $tourCode, statId: $statId, season: $season) {
    tourCode
    year
    statId
    statTitle
    statEntries {
      playerId
      playerName
      rank
      total
      average
      statValues {
        statValue
        label
      }
    }
  }
}
"""


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

    def _fetch_stat(self, stat_id: str, season: int) -> list:
        """Fetch raw stat entries for one stat + season. Returns list of player dicts."""
        payload = {
            "query": _STAT_QUERY,
            "variables": {
                "tourCode": "R",   # "R" = PGA Tour
                "statId": stat_id,
                "season": season,
            },
        }
        resp = self.session.post(_API_URL, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        details = (
            data.get("data", {})
                .get("statDetails", {})
        )
        return details.get("statEntries") or [], details.get("statTitle", "")

    def get_combined_stat(
        self,
        stat_id: str,
        label: Optional[str] = None,
        current_weight: float = 0.6,
    ) -> pd.DataFrame:
        """
        Fetch a stat for the current AND previous season, then produce a single
        combined rank per player using a weighted average of their season percentiles.

        current_weight: 0–1. Weight given to current season (rest goes to previous).

        Returns a DataFrame with columns:
            player_name, stat_id, label, combined_rank,
            cur_rank, cur_value, prev_rank, prev_value
        """
        cur_year  = self._current_year
        prev_year = cur_year - 1

        cur_entries,  title = self._fetch_stat(stat_id, cur_year)
        prev_entries, _     = self._fetch_stat(stat_id, prev_year)

        cur_df  = _entries_to_df(cur_entries,  suffix="cur")
        prev_df = _entries_to_df(prev_entries, suffix="prev")

        # Merge on normalized player name
        df = cur_df.merge(prev_df, on="player_name", how="outer")

        n_cur  = df["cur_rank"].notna().sum()
        n_prev = df["prev_rank"].notna().sum()

        # Convert ranks to percentiles (lower rank = better = higher percentile)
        # Percentile = 1 - (rank - 1) / (n - 1), capped to [0, 1]
        if n_cur > 1:
            df["cur_pct"]  = 1 - (df["cur_rank"]  - 1) / (n_cur  - 1)
        else:
            df["cur_pct"]  = np.nan

        if n_prev > 1:
            df["prev_pct"] = 1 - (df["prev_rank"] - 1) / (n_prev - 1)
        else:
            df["prev_pct"] = np.nan

        prev_weight = 1.0 - current_weight

        # Blend percentiles; fall back to whichever season is available
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

        # Convert blended percentile back to a combined rank (1 = best)
        df = df.sort_values("blended_pct", ascending=False).reset_index(drop=True)
        df["combined_rank"] = df.index + 1

        df["stat_id"] = stat_id
        df["label"]   = label or title or stat_id

        return df[[
            "player_name", "stat_id", "label", "combined_rank",
            "cur_rank", "cur_value", "prev_rank", "prev_value",
        ]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entries_to_df(entries: list, suffix: str) -> pd.DataFrame:
    """Convert raw GraphQL stat entries to a normalized DataFrame."""
    rows = []
    for e in entries:
        name = _normalize_name(e.get("playerName") or "")
        if not name:
            continue
        rank = _to_int(e.get("rank"))
        # Primary value: try average, then total, then first statValue
        value = _to_float(e.get("average"))
        if value is None:
            value = _to_float(e.get("total"))
        if value is None:
            stat_vals = e.get("statValues") or []
            if stat_vals:
                value = _to_float(stat_vals[0].get("statValue"))
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
