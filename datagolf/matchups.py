"""
Matchup edge calculator.

Compares our model's head-to-head probabilities (derived from model_win_prob
via Bradley-Terry) against the market's implied probabilities from DraftKings
and FanDuel, as returned by the DataGolf betting-tools/matchups endpoint.
"""

from typing import Optional
import pandas as pd
import numpy as np


KNOWN_BOOKS = ["draftkings", "fanduel", "betmgm", "bovada", "bet365", "pointsbet"]


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


def bradley_terry(p1_win_prob: float, p2_win_prob: float) -> tuple[float, float]:
    """
    Derive head-to-head matchup probabilities from overall win probabilities
    using the Bradley-Terry model: P(A beats B) = P_A / (P_A + P_B).
    """
    total = p1_win_prob + p2_win_prob
    if total <= 0:
        return 0.5, 0.5
    return p1_win_prob / total, p2_win_prob / total


def parse_matchups(raw: dict, model_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Parse the DataGolf betting-tools/matchups response and compute edges.

    For each matchup and each book present in the response:
      - Market implied probs (vig-removed) from the book's lines
      - DataGolf's own matchup probability
      - Our model's matchup probability (Bradley-Terry on model_win_prob)
      - Edge = our_prob - market_fair_prob

    Args:
        raw:        Response from client.get_matchups().
        model_df:   Rankings DataFrame with 'player_name' and 'model_win_prob'
                    columns. If None, only DG probs and market probs are shown.

    Returns:
        DataFrame with one row per matchup per book, sorted by |model_edge| desc.
    """
    matchups = raw.get("matchups", [])
    if not matchups:
        return pd.DataFrame()

    # Build model lookup: player_name -> model_win_prob
    model_lookup: dict = {}
    if model_df is not None and "model_win_prob" in model_df.columns:
        for _, r in model_df.iterrows():
            name = (r.get("player_name") or "").strip().lower()
            if name:
                model_lookup[name] = r["model_win_prob"]

    rows = []
    for m in matchups:
        p1_name = (m.get("p1_player_name") or m.get("p1") or "").strip()
        p2_name = (m.get("p2_player_name") or m.get("p2") or "").strip()
        p1_dg_id = m.get("p1_dg_id")
        p2_dg_id = m.get("p2_dg_id")

        # DataGolf's own matchup probability
        p1_dg = _to_float(m.get("p1_dg_win_prob") or m.get("p1_win_prob"))
        p2_dg = _to_float(m.get("p2_dg_win_prob") or m.get("p2_win_prob"))
        if p1_dg is not None and p2_dg is not None:
            p1_dg_fair, p2_dg_fair = remove_vig(p1_dg, p2_dg)
        else:
            p1_dg_fair = p2_dg_fair = None

        # Our model's matchup probability via Bradley-Terry
        p1_model = model_lookup.get(p1_name.lower())
        p2_model = model_lookup.get(p2_name.lower())
        if p1_model is not None and p2_model is not None:
            p1_our, p2_our = bradley_terry(p1_model, p2_model)
        else:
            p1_our = p2_our = None

        # Odds section — may be nested under "odds" or flat with book-prefixed keys
        odds_section = m.get("odds") or {}

        # Collect all books present
        books_found = set()
        if isinstance(odds_section, dict):
            books_found.update(odds_section.keys())
        # Also check flat keys like "draftkings_p1"
        for key in m.keys():
            for book in KNOWN_BOOKS:
                if key.startswith(book):
                    books_found.add(book)

        if not books_found:
            # No book odds — still emit one row with model/DG probs only
            books_found.add(None)

        for book in books_found:
            p1_odds_raw = p2_odds_raw = None

            if book and isinstance(odds_section, dict) and book in odds_section:
                book_data = odds_section[book]
                if isinstance(book_data, dict):
                    p1_odds_raw = _to_float(book_data.get("p1") or book_data.get("p1_odds"))
                    p2_odds_raw = _to_float(book_data.get("p2") or book_data.get("p2_odds"))
            elif book:
                # Try flat keys
                p1_odds_raw = _to_float(m.get(f"{book}_p1") or m.get(f"p1_{book}"))
                p2_odds_raw = _to_float(m.get(f"{book}_p2") or m.get(f"p2_{book}"))

            # Convert book American odds to fair probs
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
