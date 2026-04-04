#!/usr/bin/env python3
"""
DataGolf weekly betting ranking tool.

Usage examples:
  # Live rankings during a tournament (default)
  python main.py

  # Pre-tournament rankings (before play starts)
  python main.py --pre-tournament

  # Show only top 10, specific tour
  python main.py --top 10 --tour euro

  # Analyze a specific round only
  python main.py --round 2

  # Supply market odds to find value bets
  python main.py --odds-file odds.csv

  # Adjust weights (e.g. putt-heavy course)
  python main.py --weight-sg-putt 0.30 --weight-sg-app 0.10
"""

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.table import Table

from datagolf.client import DataGolfClient
from datagolf.ranking import DEFAULT_WEIGHTS, RankingModel, american_to_prob

console = Console()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DataGolf betting ranking model for weekly contests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--tour", default="pga",
                   help="Tour code: pga, euro, kft, opp, liv (default: pga)")
    p.add_argument("--round", default="event",
                   help="Round: 1-4 or 'event' for cumulative average (default: event)")
    p.add_argument("--top", type=int, default=20,
                   help="Number of players to display (default: 20)")
    p.add_argument("--pre-tournament", action="store_true",
                   help="Use pre-tournament predictions (use before first round starts)")
    p.add_argument("--no-predictions", action="store_true",
                   help="Skip prediction fetching; rank by SG stats only")
    p.add_argument("--odds-file",
                   help="Path to CSV with market odds. Required columns: player_name + one of "
                        "market_win_prob (decimal 0-1), market_odds_american, or market_odds_decimal")
    p.add_argument("--sort", default="composite",
                   choices=["composite", "win_prob", "sg_total", "edge"],
                   help="Column to sort by (default: composite)")
    p.add_argument("--all", action="store_true",
                   help="Show all players, not just top N")

    wg = p.add_argument_group(
        "Weight overrides",
        "Override default weights (values are normalized to sum to 1). "
        "Defaults: sg_total=0.30, win_prob=0.30, sg_app=0.15, sg_putt=0.12, "
        "sg_ott=0.08, sg_arg=0.05"
    )
    wg.add_argument("--weight-sg-total", type=float)
    wg.add_argument("--weight-win-prob", type=float)
    wg.add_argument("--weight-sg-app", type=float)
    wg.add_argument("--weight-sg-putt", type=float)
    wg.add_argument("--weight-sg-ott", type=float)
    wg.add_argument("--weight-sg-arg", type=float)
    wg.add_argument("--weight-sg-t2g", type=float)
    return p


def resolve_weights(args) -> dict:
    weights = dict(DEFAULT_WEIGHTS)
    overrides = {
        "sg_total": args.weight_sg_total,
        "win_prob": args.weight_win_prob,
        "sg_app": args.weight_sg_app,
        "sg_putt": args.weight_sg_putt,
        "sg_ott": args.weight_sg_ott,
        "sg_arg": args.weight_sg_arg,
        "sg_t2g": args.weight_sg_t2g,
    }
    for k, v in overrides.items():
        if v is not None:
            weights[k] = v
    return weights


# ---------------------------------------------------------------------------
# Market odds loading
# ---------------------------------------------------------------------------

def load_market_odds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    if "market_win_prob" not in df.columns:
        if "market_odds_american" in df.columns:
            df["market_win_prob"] = df["market_odds_american"].apply(american_to_prob)
        elif "market_odds_decimal" in df.columns:
            df["market_win_prob"] = 1 / df["market_odds_decimal"]
        else:
            raise ValueError(
                "odds CSV must have one of: market_win_prob, "
                "market_odds_american, or market_odds_decimal"
            )

    if "player_name" not in df.columns:
        raise ValueError("odds CSV must have a 'player_name' column")

    return df[["player_name", "market_win_prob"]]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def fmt(val, decimals: int = 2, signed: bool = False) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "[dim]-[/dim]"
    fmt_str = f"{'+' if signed else ''}.{decimals}f"
    return format(val, fmt_str)


def fmt_pct(val, signed: bool = False) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "[dim]-[/dim]"
    s = f"{val * 100:{'+' if signed else ''}.1f}%"
    if signed and val > 0:
        return f"[green]{s}[/green]"
    if signed and val < 0:
        return f"[red]{s}[/red]"
    return s


def fmt_pos(val) -> str:
    if not val or (isinstance(val, float) and pd.isna(val)):
        return "[dim]-[/dim]"
    return str(val)


def display_rankings(df: pd.DataFrame, top_n: int, show_edge: bool, sort_col: str):
    # Re-sort if needed
    sort_map = {
        "composite": "composite_score",
        "win_prob": "win_prob",
        "sg_total": "sg_total",
        "edge": "edge" if show_edge else "composite_score",
    }
    sort_by = sort_map.get(sort_col, "composite_score")
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

    display_df = df if top_n == 0 else df.head(top_n)

    t = Table(
        title=None,
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        pad_edge=False,
        min_width=80,
    )
    t.add_column("#", justify="right", width=4, style="dim")
    t.add_column("Player", min_width=22)
    t.add_column("Pos", justify="center", width=5)
    t.add_column("Thru", justify="center", width=5)
    t.add_column("SG:OTT", justify="right", width=7)
    t.add_column("SG:APP", justify="right", width=7)
    t.add_column("SG:ARG", justify="right", width=7)
    t.add_column("SG:PUT", justify="right", width=7)
    t.add_column("SG:TOT", justify="right", width=7)
    t.add_column("Win%", justify="right", width=7)
    t.add_column("Score", justify="right", width=7)
    if show_edge:
        t.add_column("Edge", justify="right", width=8)

    for _, row in display_df.iterrows():
        cells = [
            str(int(row["rank"])),
            str(row.get("player_name") or ""),
            fmt_pos(row.get("position")),
            fmt_pos(row.get("thru")),
            fmt(row.get("sg_ott"), signed=True),
            fmt(row.get("sg_app"), signed=True),
            fmt(row.get("sg_arg"), signed=True),
            fmt(row.get("sg_putt"), signed=True),
            fmt(row.get("sg_total"), signed=True),
            fmt_pct(row.get("win_prob")),
            fmt(row.get("composite_score"), signed=True),
        ]
        if show_edge:
            cells.append(fmt_pct(row.get("edge"), signed=True))
        t.add_row(*cells)

    console.print(t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()

    load_dotenv()
    api_key = os.getenv("DATAGOLF_API_KEY")
    if not api_key:
        console.print(
            "[red]Error:[/red] DATAGOLF_API_KEY not set.\n"
            "Copy [bold].env.example[/bold] to [bold].env[/bold] and add your key."
        )
        sys.exit(1)

    client = DataGolfClient(api_key)
    model = RankingModel(resolve_weights(args))

    # --- Fetch live stats ---
    with console.status("[cyan]Fetching live tournament stats…[/cyan]"):
        try:
            live_raw = client.get_live_tournament_stats(tour=args.tour, round=args.round)
        except Exception as exc:
            console.print(f"[red]Failed to fetch live stats:[/red] {exc}")
            sys.exit(1)

    # --- Fetch predictions ---
    predictions_raw = None
    if not args.no_predictions:
        label = "pre-tournament" if args.pre_tournament else "in-play"
        with console.status(f"[cyan]Fetching {label} predictions…[/cyan]"):
            try:
                if args.pre_tournament:
                    predictions_raw = client.get_pre_tournament_predictions(tour=args.tour)
                else:
                    predictions_raw = client.get_in_play_predictions(tour=args.tour)
            except Exception as exc:
                console.print(f"[yellow]Warning:[/yellow] Could not fetch predictions: {exc}")

    # --- Load market odds ---
    market_odds = None
    if args.odds_file:
        try:
            market_odds = load_market_odds(args.odds_file)
            console.print(
                f"[green]Loaded market odds for {len(market_odds)} players[/green]"
            )
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Could not load odds file: {exc}")

    # --- Build rankings ---
    df = model.build(
        live_raw,
        predictions_raw,
        pre_tournament=args.pre_tournament,
        market_odds=market_odds,
    )

    # --- Header ---
    event_name = live_raw.get("event_name") or live_raw.get("event") or "Current Event"
    last_updated = live_raw.get("last_updated", "")
    round_label = f"Round {args.round}" if args.round != "event" else "Cumulative"

    console.print()
    console.print(
        f"[bold cyan]{event_name}[/bold cyan]  "
        f"[dim]{round_label}[/dim]"
        + (f"  [dim]Updated: {last_updated}[/dim]" if last_updated else "")
    )
    console.print(
        "[dim]Weights:[/dim] "
        + "  ".join(f"{k}={v:.0%}" for k, v in model.weights.items())
    )
    console.print()

    top_n = 0 if args.all else args.top
    display_rankings(df, top_n, show_edge=market_odds is not None, sort_col=args.sort)

    # --- Value summary ---
    if market_odds is not None and "edge" in df.columns:
        value_df = df[df["edge"].notna() & (df["edge"] > 0)].head(5)
        if not value_df.empty:
            console.print("\n[bold green]Top value plays (DataGolf prob > market implied):[/bold green]")
            for _, row in value_df.iterrows():
                console.print(
                    f"  {row['player_name']:<24} "
                    f"DG: {row['win_prob']*100:.1f}%  "
                    f"Mkt: {row['market_win_prob']*100:.1f}%  "
                    f"Edge: [green]+{row['edge']*100:.1f}%[/green]"
                )


if __name__ == "__main__":
    main()
