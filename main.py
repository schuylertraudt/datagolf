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
from rich.prompt import FloatPrompt
from rich.rule import Rule
from rich.table import Table

from datagolf.client import DataGolfClient
from datagolf.matchups import parse_matchups, prob_to_american as mu_prob_to_american
from datagolf.pgatour import PGATourStats
from datagolf.ranking import DEFAULT_WEIGHTS, RankingModel, american_to_prob

WEEKLY_STATS_FILE = "weekly_stats.json"


def load_weekly_stats():
    """Load weekly PGA Tour stat config. Returns (stats, season_weight)."""
    from datagolf.pgatour import auto_season_weight
    auto_w, _ = auto_season_weight()
    import json as _json
    if not os.path.exists(WEEKLY_STATS_FILE):
        return [], auto_w
    with open(WEEKLY_STATS_FILE) as f:
        cfg = _json.load(f)
    weight = cfg.get("season_blend", {}).get("current_weight") or auto_w
    return cfg.get("stats", []), weight


def fetch_weekly_stats(stat_configs, debug: bool = False) -> dict[str, pd.DataFrame]:
    """
    Fetch each configured PGA Tour stat. Returns dict of label -> combined DataFrame.
    Silently skips any stat that fails to load.
    """
    if not stat_configs:
        return {}
    client = PGATourStats()
    results = {}
    for s in stat_configs:
        try:
            df = client.get_combined_stat(s["id"], label=s.get("label"))
            results[s.get("label", s["id"])] = df
            if debug:
                console.print(f"[dim]  {s.get('label', s['id'])}: {len(df)} players, "
                              f"sample names: {list(df['player_name'].head(3))}[/dim]")
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Could not fetch PGA Tour stat {s.get('label', s['id'])}: {exc}")
    return results


def merge_weekly_stats(rankings_df: pd.DataFrame, stat_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Left-join each weekly stat's combined_rank onto the rankings DataFrame.
    Adds one column per stat named '<label> Rk'.
    """
    from datagolf.pgatour import _normalize_name

    df = rankings_df.copy()
    # Pre-compute normalized names once (handles "Last, First" → "first last")
    normalized = df["player_name"].apply(lambda n: _normalize_name(str(n)) if pd.notna(n) else "")
    for label, stat_df in stat_dfs.items():
        col = f"{label} Rk"
        lookup = stat_df.set_index("player_name")["combined_rank"]
        df[col] = normalized.map(lookup)
        df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else None)
    return df


def run_setup_stats():
    """
    Interactive prompt to pick 5 PGA Tour stats for the week and save to weekly_stats.json.
    Fetches the stat catalog from the PGA Tour API; falls back to manual ID entry if unavailable.
    """
    import json as _json
    from rich.prompt import Prompt
    from rich.rule import Rule

    console.print()
    console.print(Rule("[bold]Weekly stats setup[/bold]"))
    console.print()
    from datagolf.pgatour import CURATED_STATS

    pga = PGATourStats()
    console.print("[cyan]Fetching PGA Tour stat catalog…[/cyan]")
    categories, err = pga.get_stat_categories()

    if not categories:
        console.print(f"[dim]API catalog unavailable ({err}) — using built-in list.[/dim]")
        categories = CURATED_STATS

    selected = []

    # Build a flat numbered list across all categories
    idx = 1
    idx_map = {}
    for cat in categories:
        console.print(f"\n[bold]{cat['category']}[/bold]")
        for s in cat["stats"]:
            console.print(f"  [dim]{idx:>3}[/dim]  {s['title']:<45} [dim]{s['id']}[/dim]")
            idx_map[idx] = {"id": s["id"], "label": s["title"]}
            idx += 1

    console.print()
    console.print("[dim]Enter 5 numbers from the list above, or type a stat ID directly.[/dim]")
    console.print()

    for i in range(1, 6):
        while True:
            raw = Prompt.ask(f"  Stat {i}", console=console).strip()
            if raw.isdigit() and int(raw) in idx_map:
                chosen = idx_map[int(raw)]
                label = Prompt.ask(
                    f"    Label for '{chosen['label']}'",
                    default=chosen["label"],
                    console=console,
                )
                selected.append({"id": chosen["id"], "label": label})
                break
            elif raw:
                # Treat as a raw stat ID typed directly
                label = Prompt.ask(f"    Label for stat {raw}", default=raw, console=console)
                selected.append({"id": raw, "label": label})
                break
            else:
                console.print("  [red]Please enter a number or stat ID.[/red]")

    console.print()
    from datagolf.pgatour import auto_season_weight
    auto_w, auto_desc = auto_season_weight()
    console.print("[bold]Season blend[/bold]")
    console.print(f"  [dim]Auto-calculated: {auto_desc}[/dim]")
    raw_w = FloatPrompt.ask(
        "  Current season weight (0 = all previous, 1 = all current)",
        default=auto_w,
        console=console,
    )
    season_weight = max(0.0, min(1.0, raw_w))

    cfg = {"season_blend": {"current_weight": season_weight}, "stats": selected}
    with open(WEEKLY_STATS_FILE, "w") as f:
        _json.dump(cfg, f, indent=2)

    console.print()
    console.print(f"[green]Saved {len(selected)} stats to {WEEKLY_STATS_FILE}[/green]")
    for s in selected:
        console.print(f"  {s['label']:<30} [dim]{s['id']}[/dim]")


def prob_to_american(prob: float) -> str:
    """Convert a win probability (0-1) to American odds string, e.g. '+350' or '-120'."""
    if not prob or pd.isna(prob) or prob <= 0 or prob >= 1:
        return "[dim]-[/dim]"
    if prob >= 0.5:
        odds = -(prob / (1 - prob)) * 100
        return f"{int(round(odds))}"
    else:
        odds = ((1 - prob) / prob) * 100
        return f"+{int(round(odds))}"

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
    p.add_argument("--no-prompt", action="store_true",
                   help="Skip interactive weight/blend prompts; use defaults or --weight-* flags")
    p.add_argument("--save-settings", action="store_true",
                   help="Save weights and blend config to settings.json after prompts (used by server.py)")
    p.add_argument("--setup-stats", action="store_true",
                   help="Interactively pick this week's 5 PGA Tour stats and save to weekly_stats.json")
    p.add_argument("--matchups", action="store_true",
                   help="Fetch DK/FD matchup lines and show edges vs your model odds")
    p.add_argument("--matchups-market", default="round_matchups",
                   choices=["round_matchups", "tournament_matchups"],
                   help="Matchup market to fetch (default: round_matchups)")
    p.add_argument("--matchups-min-edge", type=float, default=0.03,
                   help="Minimum edge to display a matchup (default: 0.03 = 3%%)")
    p.add_argument("--odds-file",
                   help="Path to CSV with market odds. Required columns: player_name + one of "
                        "market_win_prob (decimal 0-1), market_odds_american, or market_odds_decimal")
    p.add_argument("--sort", default="composite",
                   choices=["composite", "win_prob", "sg_total", "edge"],
                   help="Column to sort by (default: composite)")
    p.add_argument("--all", action="store_true",
                   help="Show all players, not just top N")
    p.add_argument("--compact", action="store_true",
                   help="Compact view: hide SG breakdown, show rank/player/score/aux stats only")
    p.add_argument("--debug-stats", action="store_true",
                   help="Print debug info about PGA Tour stat fetching (name matching)")

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
# Pre-tournament interactive configuration
# ---------------------------------------------------------------------------

# Default history blend config
DEFAULT_HISTORY = {"short_rounds": 12, "long_rounds": 60, "short_weight": 0.60}

# Labels for each weight key used in prompts
_WEIGHT_LABELS: dict[str, str] = {
    "sg_total": "Overall SG total",
    "win_prob": "DataGolf win probability",
    "sg_app":   "SG: Approach",
    "sg_putt":  "SG: Putting",
    "sg_ott":   "SG: Off the tee",
    "sg_arg":   "SG: Around the green",
    "sg_t2g":   "SG: Tee to green",
}


def prompt_pre_tournament_config(args) -> tuple[dict, dict]:
    """
    Interactively prompt for SG weights and short/long-term history blend.
    If a weight was already set via a --weight-* CLI flag it is shown but not re-asked.

    Returns:
        weights      — dict of raw (un-normalized) weights
        history_cfg  — dict with short_rounds, long_rounds, short_weight
    """
    console.print()
    console.print(Rule("[bold]Pre-tournament configuration[/bold]"))
    console.print()

    # --- SG weights ---
    console.print("[bold]SG component weights[/bold] [dim](normalized to sum to 1)[/dim]")

    cli_overrides = {
        "sg_total": args.weight_sg_total,
        "win_prob": args.weight_win_prob,
        "sg_app":   args.weight_sg_app,
        "sg_putt":  args.weight_sg_putt,
        "sg_ott":   args.weight_sg_ott,
        "sg_arg":   args.weight_sg_arg,
        "sg_t2g":   args.weight_sg_t2g,
    }

    weights = {}
    for key, default in DEFAULT_WEIGHTS.items():
        label = _WEIGHT_LABELS.get(key, key)
        cli_val = cli_overrides.get(key)
        if cli_val is not None:
            console.print(
                f"  {label:<30} [dim]{cli_val:.2f}  (from --weight-{key.replace('_', '-')})[/dim]"
            )
            weights[key] = cli_val
        else:
            val = FloatPrompt.ask(f"  {label:<30}", default=default, console=console)
            weights[key] = val

    console.print()

    # --- History blend ---
    console.print("[bold]Historical SG blend[/bold]")
    console.print(
        "[dim]Short-term captures recent form; long-term captures sustained skill.[/dim]"
    )
    short_rounds = int(
        FloatPrompt.ask("  Short-term window (recent rounds) ", default=DEFAULT_HISTORY["short_rounds"], console=console)
    )
    long_rounds = int(
        FloatPrompt.ask("  Long-term window  (recent rounds) ", default=DEFAULT_HISTORY["long_rounds"], console=console)
    )
    raw_sw = FloatPrompt.ask(
        "  Short-term weight (0 = all long-term, 1 = all short-term)",
        default=DEFAULT_HISTORY["short_weight"],
        console=console,
    )
    short_weight = max(0.0, min(1.0, raw_sw))

    console.print()

    console.print()

    # --- Model odds blend ---
    console.print("[bold]My odds blend[/bold]")
    console.print(
        "[dim]Blend DataGolf win probabilities with your composite score to derive custom odds.[/dim]"
    )
    raw_dg = FloatPrompt.ask(
        "  DataGolf weight (0 = my model only, 1 = DataGolf only)",
        default=0.5,
        console=console,
    )
    dg_weight = max(0.0, min(1.0, raw_dg))

    console.print()

    history_cfg = {
        "short_rounds": short_rounds,
        "long_rounds":  long_rounds,
        "short_weight": short_weight,
        "dg_weight":    dg_weight,
    }
    return weights, history_cfg


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


def display_rankings(
    df: pd.DataFrame,
    top_n: int,
    show_edge: bool,
    sort_col: str,
    pre_tournament: bool = False,
    compact: bool = False,
):
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

    extra_cols = [c for c in df.columns if c.endswith(" Rk")]

    t = Table(
        title=None,
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        pad_edge=False,
        expand=False,
    )
    t.add_column("#", justify="right", width=4, style="dim")
    t.add_column("Player", min_width=18 if compact else 24, no_wrap=True)

    if compact:
        # Compact: rank, player, my odds, score, aux stat ranks only
        if "model_win_prob" in df.columns:
            t.add_column("My Odds", justify="right", min_width=8)
        t.add_column("Score", justify="right", min_width=7)
        for col in extra_cols:
            t.add_column(col, justify="right", min_width=6)
    else:
        if not pre_tournament:
            t.add_column("Pos",  justify="center", width=5)
            t.add_column("Thru", justify="center", width=5)
        t.add_column("Win%",    justify="right", min_width=7)
        t.add_column("DG Odds", justify="right", min_width=8)
        if "model_win_prob" in df.columns:
            t.add_column("My Odds", justify="right", min_width=8)
        t.add_column("SG:OTT", justify="right", min_width=7)
        t.add_column("SG:APP", justify="right", min_width=7)
        t.add_column("SG:ARG", justify="right", min_width=7)
        t.add_column("SG:PUT", justify="right", min_width=7)
        t.add_column("SG:TOT", justify="right", min_width=7)
        t.add_column("Score",   justify="right", min_width=7)
        if show_edge:
            t.add_column("Edge", justify="right", min_width=8)
        for col in extra_cols:
            t.add_column(col, justify="right", min_width=6)

    for _, row in display_df.iterrows():
        cells = [str(int(row["rank"])), str(row.get("player_name") or "")]

        if compact:
            if "model_win_prob" in display_df.columns:
                cells.append(prob_to_american(row.get("model_win_prob")))
            cells.append(fmt(row.get("composite_score"), signed=True))
            for col in extra_cols:
                v = row.get(col)
                cells.append(str(int(v)) if v is not None else "[dim]-[/dim]")
        else:
            if not pre_tournament:
                cells += [fmt_pos(row.get("position")), fmt_pos(row.get("thru"))]
            cells += [
                fmt_pct(row.get("win_prob")),
                prob_to_american(row.get("win_prob")),
            ]
            if "model_win_prob" in display_df.columns:
                cells.append(prob_to_american(row.get("model_win_prob")))
            cells += [
                fmt(row.get("sg_ott"), signed=True),
                fmt(row.get("sg_app"), signed=True),
                fmt(row.get("sg_arg"), signed=True),
                fmt(row.get("sg_putt"), signed=True),
                fmt(row.get("sg_total"), signed=True),
                fmt(row.get("composite_score"), signed=True),
            ]
            if show_edge:
                cells.append(fmt_pct(row.get("edge"), signed=True))
            for col in extra_cols:
                v = row.get(col)
                cells.append(str(int(v)) if v is not None else "[dim]-[/dim]")

        t.add_row(*cells)

    console.print(t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()

    if args.setup_stats:
        run_setup_stats()
        return

    load_dotenv()
    api_key = os.getenv("DATAGOLF_API_KEY")
    if not api_key:
        console.print(
            "[red]Error:[/red] DATAGOLF_API_KEY not set.\n"
            "Copy [bold].env.example[/bold] to [bold].env[/bold] and add your key."
        )
        sys.exit(1)

    client = DataGolfClient(api_key)

    # --- Fetch stats and predictions ---
    if args.pre_tournament:
        # Prompt for weights and history blend (unless --no-prompt)
        if args.no_prompt:
            weights = resolve_weights(args)
            history_cfg = {**DEFAULT_HISTORY, "dg_weight": 0.5}
        else:
            weights, history_cfg = prompt_pre_tournament_config(args)

        if args.save_settings:
            import json as _json
            settings = {"weights": weights, "history": history_cfg, "tour": args.tour}
            with open("settings.json", "w") as f:
                _json.dump(settings, f, indent=2)
            console.print("[green]Settings saved to settings.json[/green]")

        model = RankingModel(weights)

        # Fetch short-term and long-term SG history in parallel would be ideal;
        # for simplicity we fetch sequentially.
        with console.status(
            f"[cyan]Fetching short-term SG history ({history_cfg['short_rounds']} rounds)…[/cyan]"
        ):
            try:
                short_raw = client.get_historical_sg_stats(
                    tour=args.tour, n_rounds=history_cfg["short_rounds"]
                )
            except Exception as exc:
                console.print(f"[red]Failed to fetch short-term SG history:[/red] {exc}")
                sys.exit(1)

        with console.status(
            f"[cyan]Fetching long-term SG history ({history_cfg['long_rounds']} rounds)…[/cyan]"
        ):
            try:
                long_raw = client.get_historical_sg_stats(
                    tour=args.tour, n_rounds=history_cfg["long_rounds"]
                )
            except Exception as exc:
                console.print(
                    f"[yellow]Warning:[/yellow] Could not fetch long-term SG history: {exc}. "
                    "Using short-term only."
                )
                long_raw = short_raw

        predictions_raw = None
        if not args.no_predictions:
            with console.status("[cyan]Fetching pre-tournament predictions…[/cyan]"):
                try:
                    predictions_raw = client.get_pre_tournament_predictions(tour=args.tour)
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

        df = model.build_pre_tournament(
            short_raw,
            predictions_raw,
            market_odds=market_odds,
            long_term_raw=long_raw,
            long_term_weight=1.0 - history_cfg["short_weight"],
        )
        df = model.blend_win_probs(df, dg_weight=history_cfg["dg_weight"])

        event_name = (
            (predictions_raw or {}).get("event_name")
            or (predictions_raw or {}).get("event")
            or short_raw.get("event_name")
            or "Pre-Tournament Rankings"
        )
        last_updated = short_raw.get("last_updated", "")
        sw = history_cfg["short_weight"]
        round_label = (
            f"SG blend: {sw:.0%} last {history_cfg['short_rounds']}r / "
            f"{1-sw:.0%} last {history_cfg['long_rounds']}r"
        )

    else:
        model = RankingModel(resolve_weights(args))

        # Live/in-play: use live tournament stats
        with console.status("[cyan]Fetching live tournament stats…[/cyan]"):
            try:
                stats_raw = client.get_live_tournament_stats(tour=args.tour, round=args.round)
            except Exception as exc:
                console.print(f"[red]Failed to fetch live stats:[/red] {exc}")
                sys.exit(1)

        predictions_raw = None
        if not args.no_predictions:
            with console.status("[cyan]Fetching in-play predictions…[/cyan]"):
                try:
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

        df = model.build(stats_raw, predictions_raw, market_odds=market_odds)

        event_name = stats_raw.get("event_name") or stats_raw.get("event") or "Current Event"
        last_updated = stats_raw.get("last_updated", "")
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

    # --- Weekly PGA Tour stats ---
    debug_stats = getattr(args, "debug_stats", False)
    stat_configs = []
    season_weight = 0.6
    weekly = load_weekly_stats()
    if weekly:
        stat_configs, season_weight = weekly
    if stat_configs:
        with console.status("[cyan]Fetching PGA Tour weekly stats…[/cyan]"):
            stat_dfs = fetch_weekly_stats(stat_configs, debug=debug_stats)
        if stat_dfs:
            if debug_stats:
                from datagolf.pgatour import _normalize_name
                sample_dg = [_normalize_name(str(n)) for n in df["player_name"].head(3) if pd.notna(n)]
                console.print(f"[dim]  DG normalized sample names: {sample_dg}[/dim]")
            df = merge_weekly_stats(df, stat_dfs)

    # --- DraftKings outright odds (TODO: parse once endpoint confirmed) ---
    outrights_raw = None

    top_n = 0 if args.all else args.top
    display_rankings(
        df, top_n,
        show_edge=market_odds is not None,
        sort_col=args.sort,
        pre_tournament=args.pre_tournament,
        compact=args.compact,
    )

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

    # --- Matchups ---
    if args.matchups:
        console.print()
        console.print(f"[bold cyan]Matchup edges — {args.matchups_market.replace('_', ' ').title()}[/bold cyan]")
        with console.status("[cyan]Fetching matchup lines…[/cyan]"):
            try:
                matchups_raw = client.get_matchups(tour=args.tour, market=args.matchups_market)
            except Exception as exc:
                console.print(f"[red]Failed to fetch matchups:[/red] {exc}")
                matchups_raw = None

        if matchups_raw:
            mu_df = parse_matchups(matchups_raw, model_df=df)
            display_matchups(mu_df, min_edge=args.matchups_min_edge)


def display_matchups(mu_df: pd.DataFrame, min_edge: float = 0.03):
    if mu_df.empty:
        console.print("[dim]No matchup data returned.[/dim]")
        return

    # Filter to rows with at least one side meeting the edge threshold
    has_model = "p1_our_prob" in mu_df.columns and mu_df["p1_our_prob"].notna().any()
    if has_model:
        edge_mask = (
            (mu_df["p1_edge"].fillna(0) >= min_edge) |
            (mu_df["p2_edge"].fillna(0) >= min_edge)
        )
        show_df = mu_df[edge_mask].copy()
    else:
        show_df = mu_df.copy()

    if show_df.empty:
        console.print(f"[dim]No matchups with edge ≥ {min_edge:.0%}. "
                      "Try --matchups-min-edge 0.01 to lower the threshold.[/dim]")
        return

    # Deduplicate: if multiple books, pick best edge per matchup
    show_df = (
        show_df.sort_values("max_edge", ascending=False)
        .drop_duplicates(subset=["p1_name", "p2_name"], keep="first")
        .reset_index(drop=True)
    )

    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold", pad_edge=False)
    t.add_column("Player A",    min_width=22, no_wrap=True)
    t.add_column("Player B",    min_width=22, no_wrap=True)
    t.add_column("Book",        width=12)
    t.add_column("Line A",      justify="right", min_width=7)
    t.add_column("Line B",      justify="right", min_width=7)
    if has_model:
        t.add_column("Our A",   justify="right", min_width=7)
        t.add_column("Our B",   justify="right", min_width=7)
        t.add_column("Edge A",  justify="right", min_width=7)
        t.add_column("Edge B",  justify="right", min_width=7)
        t.add_column("Bet",     min_width=22, no_wrap=True)

    for _, row in show_df.iterrows():
        p1_edge = row.get("p1_edge")
        p2_edge = row.get("p2_edge")

        # Highlight the side with positive edge
        if p1_edge is not None and p1_edge >= min_edge:
            p1_label = f"[green]{row['p1_name']}[/green]"
        else:
            p1_label = row["p1_name"]

        if p2_edge is not None and p2_edge >= min_edge:
            p2_label = f"[green]{row['p2_name']}[/green]"
        else:
            p2_label = row["p2_name"]

        def _fmt_odds(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return "[dim]-[/dim]"
            return f"+{int(v)}" if v > 0 else str(int(v))

        def _fmt_edge(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return "[dim]-[/dim]"
            s = f"{v*100:+.1f}%"
            return f"[green]{s}[/green]" if v >= min_edge else f"[dim]{s}[/dim]"

        # Best side to bet
        if has_model:
            if p1_edge is not None and p2_edge is not None:
                if p1_edge >= min_edge and p1_edge >= p2_edge:
                    bet = f"[green]→ {row['p1_name']}[/green]"
                elif p2_edge >= min_edge:
                    bet = f"[green]→ {row['p2_name']}[/green]"
                else:
                    bet = "[dim]no edge[/dim]"
            elif p1_edge is not None and p1_edge >= min_edge:
                bet = f"[green]→ {row['p1_name']}[/green]"
            elif p2_edge is not None and p2_edge >= min_edge:
                bet = f"[green]→ {row['p2_name']}[/green]"
            else:
                bet = "[dim]no edge[/dim]"

        cells = [
            p1_label,
            p2_label,
            str(row.get("book") or ""),
            _fmt_odds(row.get("p1_book_odds")),
            _fmt_odds(row.get("p2_book_odds")),
        ]
        if has_model:
            cells += [
                mu_prob_to_american(row.get("p1_our_prob")),
                mu_prob_to_american(row.get("p2_our_prob")),
                _fmt_edge(p1_edge),
                _fmt_edge(p2_edge),
                bet,
            ]
        t.add_row(*cells)

    console.print(t)


if __name__ == "__main__":
    main()
