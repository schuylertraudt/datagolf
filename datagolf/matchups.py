"""
Matchup edge calculator.

Compares our model's head-to-head probabilities (derived from SG totals
via a normal round-scoring model) against the market's implied probabilities
from DraftKings, FanDuel, and other books, as returned by the DataGolf
betting-tools/matchups endpoint.
"""

import math
from typing import Optional
import pandas as pd
import numpy as np


KNOWN_BOOKS = ["draftkings", "fanduel", "betmgm", "bovada", "bet365", "pointsbet"]

# Empirical PGA Tour single-round score standard deviation (strokes).
# A player's actual round score = expected_score + noise, where noise ~ N(0, ROUND_STD_DEV).
# For a head-to-head score differential, the combined std dev is sqrt(2) * ROUND_STD_DEV.
ROUND_STD_DEV = 2.9


def remove_vig(p1_raw: float, p2_raw: float) -> tuple[float, float]:
    """
    Remove the vig from two raw implied probabilities that sum to > 1.
    Uses the proportional (multiplicative) method.
    Returns (p1_fair, p2_fair) that sum to 1.
    """
    total = p1_raw + p2_raw
    if total <= 0:
        return 0.5, 0.5
    return p1_raw / total, p2_raw / total


def american_to_prob(odds: float) -> Optional[float]:
    """Convert American odds to raw implied probability (vig included)."""
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def prob_to_american(prob: float) -> str:
    """Convert probability (0-1) to American odds string."""
    if prob is None or np.isnan(prob) or prob <= 0 or prob >= 1:
        return "-"
    if prob >= 0.5:
        return str(int(round(-(prob / (1 - prob)) * 100)))
    return f"+{int(round(((1 - prob) / prob) * 100))}"


def round_matchup_prob(sg_a: float, sg_b: float) -> tuple[float, float]:
    """
    Compute head-to-head round matchup probability using a normal distribution
    over expected round score differential.

    sg_a, sg_b: strokes-gained totals (strokes per round relative to average field).
    P(A beats B) = Φ( (sg_a − sg_b) / (√2 × ROUND_STD_DEV) )

    Uses math.erfc from stdlib — no scipy required.
    Returns (p_a_wins, p_b_wins).
    """
    diff = sg_a - sg_b
    h2h_std = ROUND_STD_DEV * math.sqrt(2)  # combined std dev of score differential
    # Normal CDF: Φ(x) = 0.5 * erfc(-x / sqrt(2))
    p_a = 0.5 * math.erfc(-diff / (h2h_std * math.sqrt(2)))
    return p_a, 1.0 - p_a


def bradley_terry(p1_win_prob: float, p2_win_prob: float) -> tuple[float, float]:
    """
    Derive head-to-head matchup probabilities from overall win probabilities
    using the Bradley-Terry model: P(A beats B) = P_A / (P_A + P_B).

    NOTE: this is kept for reference but should NOT be used for round matchups —
    tournament win probabilities are a poor input for single-round H2H models.
    Use round_matchup_prob() instead.
    """
    total = p1_win_prob + p2_win_prob
    if total <= 0:
        return 0.5, 0.5
    return p1_win_prob / total, p2_win_prob / total


def _normalize_name(name: str) -> str:
    """
    Normalize player name to lowercase 'first last' for fuzzy matching.
    Handles both 'First Last' and 'Last, First' formats.
    """
    name = name.strip()
    if "," in name:
        parts = name.split(",", 1)
        name = f"{parts[1].strip()} {parts[0].strip()}"
    return name.lower()


def parse_matchups(raw: dict, model_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    matchups = raw.get("match_list") or raw.get("matchups") or raw.get("data", [])
    if not matchups:
        return pd.DataFrame()

    # Build SG lookup keyed by normalized name.
    # Prefer 'matchup_sg' (user-influenced blend) when present; fall back to 'sg_total'.
    # Both are in strokes/round units, directly usable in round_matchup_prob().
    sg_lookup: dict = {}
    if model_df is not None:
        sg_col = "matchup_sg" if "matchup_sg" in model_df.columns else "sg_total"
        for _, r in model_df.iterrows():
            name = (r.get("player_name") or "").strip()
            sg = r.get(sg_col)
            if name and sg is not None and not pd.isna(sg):
                sg_lookup[_normalize_name(name)] = float(sg)

    rows = []
    for m in matchups:
        p1_name = (m.get("p1_player_name") or m.get("p1") or "").strip()
        p2_name = (m.get("p2_player_name") or m.get("p2") or "").strip()

        # Odds section: {"bet365": {"p1": "+135", "p2": "-160"}, "datagolf": {...}, ...}
        odds_section = m.get("odds") or {}

        # Extract DataGolf's own probability from odds.datagolf (it's American odds)
        dg_odds = odds_section.get("datagolf") or {}
        p1_dg_raw = american_to_prob(_to_float(dg_odds.get("p1")))
        p2_dg_raw = american_to_prob(_to_float(dg_odds.get("p2")))
        if p1_dg_raw is not None and p2_dg_raw is not None:
            p1_dg_fair, p2_dg_fair = remove_vig(p1_dg_raw, p2_dg_raw)
        else:
            p1_dg_fair = p2_dg_fair = None

        # Our model's matchup probability via round-scoring normal distribution.
        # Uses sg_total (strokes gained/round) directly — more appropriate than
        # running tournament win probabilities through Bradley-Terry.
        p1_sg = sg_lookup.get(_normalize_name(p1_name))
        p2_sg = sg_lookup.get(_normalize_name(p2_name))
        if p1_sg is not None and p2_sg is not None:
            p1_our, p2_our = round_matchup_prob(p1_sg, p2_sg)
        else:
            p1_our = p2_our = None

        # Real sportsbooks (exclude 'datagolf' pseudo-book)
        books_found = [k for k in odds_section.keys() if k != "datagolf"]
        if not books_found:
            books_found = [None]

        for book in books_found:
            p1_odds_raw = p2_odds_raw = None

            if book and book in odds_section:
                book_data = odds_section[book]
                if isinstance(book_data, dict):
                    p1_odds_raw = _to_float(book_data.get("p1"))
                    p2_odds_raw = _to_float(book_data.get("p2"))

            # Convert American odds strings to fair probs (vig removed)
            if p1_odds_raw is not None and p2_odds_raw is not None:
                p1_mkt_raw = american_to_prob(p1_odds_raw)
                p2_mkt_raw = american_to_prob(p2_odds_raw)
                p1_mkt_fair, p2_mkt_fair = remove_vig(p1_mkt_raw, p2_mkt_raw)
            else:
                p1_mkt_fair = p2_mkt_fair = None
                p1_odds_raw = p2_odds_raw = None

            # Edge vs market
            p1_edge = (p1_our - p1_mkt_fair) if (p1_our is not None and p1_mkt_fair is not None) else None
            p2_edge = (p2_our - p2_mkt_fair) if (p2_our is not None and p2_mkt_fair is not None) else None

            rows.append({
                "p1_name":       p1_name,
                "p2_name":       p2_name,
                "book":          book or "",
                # Book lines
                "p1_book_odds":  p1_odds_raw,
                "p2_book_odds":  p2_odds_raw,
                # Market fair probs (vig removed)
                "p1_mkt_prob":   p1_mkt_fair,
                "p2_mkt_prob":   p2_mkt_fair,
                # DataGolf model probs
                "p1_dg_prob":    p1_dg_fair,
                "p2_dg_prob":    p2_dg_fair,
                # Our model probs
                "p1_our_prob":   p1_our,
                "p2_our_prob":   p2_our,
                # Edges
                "p1_edge":       p1_edge,
                "p2_edge":       p2_edge,
                "max_edge":      max(
                    p1_edge if p1_edge is not None else 0,
                    p2_edge if p2_edge is not None else 0,
                ),
            })

    df = pd.DataFrame(rows)
    if not df.empty and "max_edge" in df.columns:
        df = df.sort_values("max_edge", ascending=False).reset_index(drop=True)
    return df


def _to_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
