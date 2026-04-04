"""
Ranking model that combines live SG stats with DataGolf win probabilities
into a composite betting score.
"""

from typing import Optional
import pandas as pd
import numpy as np

# Default weights (will be normalized to sum to 1.0 at runtime).
# Tune these to emphasize the metrics you care about most for a given course type.
DEFAULT_WEIGHTS: dict[str, float] = {
    "sg_total": 0.30,   # overall SG (correlated with others; captures variance not explained below)
    "win_prob": 0.30,   # DataGolf model win probability
    "sg_app": 0.15,     # approach (biggest differentiator on most PGA Tour courses)
    "sg_putt": 0.12,    # putting
    "sg_ott": 0.08,     # off-the-tee
    "sg_arg": 0.05,     # around-the-green
}

SG_STATS = ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_t2g", "sg_total"]


class RankingModel:
    """
    Builds a composite ranking from DataGolf live stats and in-play/pre-tournament
    win probabilities. Optionally surfaces value against market odds.

    Weights are normalized at construction time, so raw values don't need to sum to 1.
    """

    def __init__(self, weights: Optional[dict[str, float]] = None):
        raw = weights or DEFAULT_WEIGHTS
        total = sum(raw.values())
        self.weights = {k: v / total for k, v in raw.items() if v > 0}

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_live_stats(self, raw: dict) -> pd.DataFrame:
        """
        Parse the live-tournament-stats JSON response.
        Expects a top-level 'live_stats' list of player objects where SG values
        are either direct fields or nested under a 'stats' key.
        """
        players = raw.get("live_stats", [])
        rows = []
        for p in players:
            row: dict = {
                "player_name": p.get("player_name"),
                "dg_id": p.get("dg_id"),
                "position": p.get("current_pos"),
                "total": p.get("total"),
                "thru": p.get("thru"),
            }
            # Stats may be direct fields or a nested list of {stat, value} dicts
            if "stats" in p and isinstance(p["stats"], list):
                for s in p["stats"]:
                    name = s.get("stat") or s.get("stat_name")
                    val = s.get("value")
                    if name:
                        row[name] = _to_float(val)
            else:
                for stat in SG_STATS:
                    if stat in p:
                        row[stat] = _to_float(p[stat])
            rows.append(row)
        return pd.DataFrame(rows)

    def _parse_predictions(self, raw: dict, pre_tournament: bool = False) -> pd.DataFrame:
        """
        Parse in-play or pre-tournament predictions JSON.

        In-play structure:  {"data": [{dg_id, player_name, win, top_5, top_10, make_cut}]}
        Pre-tournament:     {"field": [{dg_id, player_name, baseline: {win, top_5, ...}}]}
        """
        rows = []
        if pre_tournament:
            for p in raw.get("field", []):
                baseline = p.get("baseline", p)
                rows.append({
                    "dg_id": p.get("dg_id"),
                    "player_name": p.get("player_name"),
                    "win_prob": _to_float(baseline.get("win")),
                    "top5_prob": _to_float(baseline.get("top_5")),
                    "top10_prob": _to_float(baseline.get("top_10")),
                    "make_cut_prob": _to_float(baseline.get("make_cut")),
                })
        else:
            for p in raw.get("data", []):
                rows.append({
                    "dg_id": p.get("dg_id"),
                    "player_name": p.get("player_name"),
                    "win_prob": _to_float(p.get("win")),
                    "top5_prob": _to_float(p.get("top_5")),
                    "top10_prob": _to_float(p.get("top_10")),
                    "make_cut_prob": _to_float(p.get("make_cut")),
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Core model
    # ------------------------------------------------------------------

    def build(
        self,
        live_raw: dict,
        predictions_raw: Optional[dict] = None,
        pre_tournament: bool = False,
        market_odds: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build the composite ranking DataFrame.

        Args:
            live_raw:         Response from get_live_tournament_stats().
            predictions_raw:  Response from get_in_play_predictions() or
                              get_pre_tournament_predictions(). Optional.
            pre_tournament:   Set True when predictions_raw is a pre-tournament response.
            market_odds:      DataFrame with columns [player_name, market_win_prob].
                              market_win_prob should be a decimal probability (0-1).
                              When provided, an 'edge' column is added.

        Returns:
            DataFrame sorted by composite_score descending, with a 'rank' column.
        """
        df = self._parse_live_stats(live_raw)

        if predictions_raw is not None:
            df_pred = self._parse_predictions(predictions_raw, pre_tournament)
            df = df.merge(
                df_pred[["dg_id", "win_prob", "top5_prob", "top10_prob", "make_cut_prob"]],
                on="dg_id",
                how="left",
            )
        else:
            for col in ("win_prob", "top5_prob", "top10_prob", "make_cut_prob"):
                df[col] = np.nan

        # Composite score: weighted sum of z-scored inputs
        score = pd.Series(0.0, index=df.index)
        for stat, weight in self.weights.items():
            if stat not in df.columns:
                continue
            col = df[stat].copy()
            # Fill missing with column median so absent players aren't penalized
            median = col.median()
            col = col.fillna(median if pd.notna(median) else 0.0)
            z = _zscore(col)
            score += weight * z

        df["composite_score"] = score
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        if market_odds is not None:
            df = self._add_edge(df, market_odds)

        return df

    def _add_edge(self, df: pd.DataFrame, market_odds: pd.DataFrame) -> pd.DataFrame:
        """
        Merge market implied probabilities and compute edge.
        market_odds must have columns: player_name, market_win_prob (decimal 0-1).
        Edge = DataGolf win_prob - market implied win_prob.
        Positive edge means DataGolf thinks the player is underpriced.
        """
        mo = market_odds[["player_name", "market_win_prob"]].copy()
        mo["player_name"] = mo["player_name"].str.strip()
        df = df.merge(mo, on="player_name", how="left")
        df["edge"] = df["win_prob"] - df["market_win_prob"]
        return df


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def american_to_prob(odds: float) -> float:
    """Convert American odds to implied win probability (0-1), no vig removed."""
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def decimal_to_prob(odds: float) -> float:
    """Convert decimal odds to implied win probability (0-1)."""
    return 1 / odds


def _to_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std
