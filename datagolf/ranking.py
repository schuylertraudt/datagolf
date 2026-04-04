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

    def _parse_skill_ratings(self, raw: dict) -> pd.DataFrame:
        """
        Parse the skill-ratings (or historical-raw-data/rounds) response into
        a DataFrame with the same SG columns used by _parse_live_stats.

        skill-ratings structure:
          {"last_updated": "...", "players": [{dg_id, player_name, sg_ott, ...}]}

        historical-raw-data/rounds structure (when available):
          {"last_updated": "...", "data": [{dg_id, player_name, sg_ott, ...}]}
        """
        players = raw.get("players") or raw.get("data", [])
        rows = []
        for p in players:
            name = (
                p.get("player_name")
                or p.get("player")
                or p.get("name")
                or p.get("full_name")
            )
            row: dict = {
                "player_name": name,
                "dg_id": p.get("dg_id"),
                "position": None,
                "total": None,
                "thru": None,
            }
            for stat in SG_STATS:
                row[stat] = _to_float(p.get(stat))
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
            # Response structure: {"event_name": "...", "baseline": [{dg_id, player_name,
            #   win, top_5, top_10, make_cut, ...}], "baseline_history_fit": [...], ...}
            players = raw.get("baseline") or raw.get("field", [])
            for p in players:
                rows.append({
                    "dg_id": p.get("dg_id"),
                    "player_name": p.get("player_name"),
                    "win_prob": _to_float(p.get("win")),
                    "top5_prob": _to_float(p.get("top_5")),
                    "top10_prob": _to_float(p.get("top_10")),
                    "make_cut_prob": _to_float(p.get("make_cut")),
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
            pred_cols = ["dg_id"] + [
                c for c in ("win_prob", "top5_prob", "top10_prob", "make_cut_prob")
                if c in df_pred.columns
            ]
            df = df.merge(
                df_pred[pred_cols],
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

    def build_pre_tournament(
        self,
        short_term_raw: dict,
        predictions_raw: Optional[dict] = None,
        market_odds: Optional[pd.DataFrame] = None,
        long_term_raw: Optional[dict] = None,
        long_term_weight: float = 0.40,
    ) -> pd.DataFrame:
        """
        Build a pre-tournament composite ranking using a blend of short-term
        and long-term rolling SG history.

        Args:
            short_term_raw:   Response from get_historical_sg_stats() with a smaller
                              n_rounds (e.g. 12) — captures recent form.
            predictions_raw:  Response from get_pre_tournament_predictions(). Optional.
            market_odds:      DataFrame with columns [player_name, market_win_prob].
            long_term_raw:    Response from get_historical_sg_stats() with a larger
                              n_rounds (e.g. 36) — captures sustained skill. When None,
                              short_term_raw is used for both (i.e. no blending).
            long_term_weight: Weight given to long-term stats (0–1). Short-term weight
                              is 1 − long_term_weight.

        Returns:
            DataFrame sorted by composite_score descending, with a 'rank' column.
        """
        short_df = self._parse_skill_ratings(short_term_raw)

        if long_term_raw is not None and long_term_weight > 0:
            long_df = self._parse_skill_ratings(long_term_raw)
            df = self._blend_sg_stats(short_df, long_df, long_term_weight)
        else:
            df = short_df

        if predictions_raw is not None:
            df_pred = self._parse_predictions(predictions_raw, pre_tournament=True)
            pred_cols = ["dg_id"] + [
                c for c in ("win_prob", "top5_prob", "top10_prob", "make_cut_prob")
                if c in df_pred.columns
            ]
            if len(pred_cols) > 1 and "dg_id" in df_pred.columns:
                # Inner join: keeps only players who are in the tournament field
                df = df.merge(df_pred[pred_cols], on="dg_id", how="inner")
            elif "dg_id" in df_pred.columns:
                # Predictions exist but have no useful stat columns — still filter to field
                df = df[df["dg_id"].isin(df_pred["dg_id"])]
                for col in ("win_prob", "top5_prob", "top10_prob", "make_cut_prob"):
                    df[col] = np.nan
            else:
                # Predictions response had no usable structure — show all, no win probs
                for col in ("win_prob", "top5_prob", "top10_prob", "make_cut_prob"):
                    df[col] = np.nan
        else:
            # No predictions available — can't filter to field, show all with a warning
            for col in ("win_prob", "top5_prob", "top10_prob", "make_cut_prob"):
                df[col] = np.nan

        # Composite score: same weighted z-score approach as live rankings
        score = pd.Series(0.0, index=df.index)
        for stat, weight in self.weights.items():
            if stat not in df.columns:
                continue
            col = df[stat].copy()
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

    def _blend_sg_stats(
        self,
        short_df: pd.DataFrame,
        long_df: pd.DataFrame,
        long_weight: float,
    ) -> pd.DataFrame:
        """
        Merge two skill-ratings DataFrames and produce a weighted average of each
        SG stat per player. Players present in only one set use that set's values.

        long_weight: 0–1. short_weight = 1 − long_weight.
        """
        short_weight = 1.0 - long_weight
        merged = short_df.merge(long_df, on="dg_id", how="outer", suffixes=("_s", "_l"))
        merged["player_name"] = merged["player_name_s"].fillna(merged["player_name_l"])

        for stat in SG_STATS:
            s_col, l_col = f"{stat}_s", f"{stat}_l"
            has_s = s_col in merged.columns
            has_l = l_col in merged.columns
            if has_s and has_l:
                # When a player is missing from one source, fall back to the other
                s_vals = merged[s_col].fillna(merged[l_col])
                l_vals = merged[l_col].fillna(merged[s_col])
                merged[stat] = short_weight * s_vals + long_weight * l_vals
            elif has_s:
                merged[stat] = merged[s_col]
            elif has_l:
                merged[stat] = merged[l_col]

        base_cols = ["player_name", "dg_id", "position", "total", "thru"]
        for col in ("position", "total", "thru"):
            if col not in merged.columns:
                merged[col] = None
        stat_cols = [s for s in SG_STATS if s in merged.columns]
        return merged[base_cols + stat_cols].copy()

    def blend_win_probs(
        self,
        df: pd.DataFrame,
        dg_weight: float = 0.5,
        temperature: float = 1.0,
    ) -> pd.DataFrame:
        """
        Derive adjusted win probabilities by blending DataGolf's win_prob with
        probabilities implied by the composite score.

        Composite scores (z-scores) are converted to a probability distribution
        via softmax, then blended with DataGolf probs:

            model_win_prob = dg_weight * dg_prob + (1 - dg_weight) * composite_prob

        Result is re-normalized to sum to 1 and stored in 'model_win_prob'.

        Args:
            dg_weight:   0 = ignore DataGolf entirely, 1 = use DataGolf only.
            temperature: Controls how spread out composite probabilities are.
                         Lower = more concentrated on top-ranked players (default 1.0).
        """
        # Softmax over composite scores
        scores = df["composite_score"].fillna(0.0)
        scaled = scores / temperature
        exp_s = np.exp(scaled - scaled.max())  # subtract max for numerical stability
        composite_prob = exp_s / exp_s.sum()

        # DataGolf probs — fill missing with equal share
        dg_prob = df["win_prob"].copy()
        dg_prob = dg_prob.fillna(1.0 / len(df))
        # Re-normalize DG probs in case they don't sum to 1
        dg_total = dg_prob.sum()
        if dg_total > 0:
            dg_prob = dg_prob / dg_total

        blended = dg_weight * dg_prob + (1.0 - dg_weight) * composite_prob
        blended = blended / blended.sum()  # ensure sums to 1

        df = df.copy()
        df["model_win_prob"] = blended
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
