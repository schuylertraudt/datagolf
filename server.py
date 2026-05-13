#!/usr/bin/env python3
"""
DataGolf rankings web server.

Setup:
  1. Run the CLI once to configure and save settings:
       python main.py --pre-tournament --save-settings
  2. Start the server:
       python server.py
  3. Open http://<your-server-ip>:8080 in a browser.

Pages:
  /           — Pre-tournament rankings
  /matchups   — Round matchup edges

The pages auto-refresh every 5 minutes. API results are cached server-side
so multiple browser loads don't hammer the DataGolf API.
"""

import json
import os
import time
import traceback
from datetime import datetime
from threading import Lock

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template_string, request

from datagolf.client import DataGolfClient
from datagolf.matchups import parse_matchups
from datagolf.pgatour import CURATED_STATS, PGATourStats
from datagolf.ranking import DEFAULT_WEIGHTS, RankingModel

WEEKLY_STATS_FILE = "weekly_stats.json"


def _load_weekly_stats():
    from datagolf.pgatour import auto_season_weight
    auto_w, _ = auto_season_weight()
    if not os.path.exists(WEEKLY_STATS_FILE):
        return [], auto_w
    with open(WEEKLY_STATS_FILE) as f:
        cfg = json.load(f)
    weight = cfg.get("season_blend", {}).get("current_weight") or auto_w
    return cfg.get("stats", []), weight

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SETTINGS_FILE = "settings.json"
CACHE_TTL = 300  # seconds between API refreshes (5 minutes)
DEFAULT_SETTINGS = {
    "weights": dict(DEFAULT_WEIGHTS),
    "history": {"short_rounds": 12, "long_rounds": 60, "short_weight": 0.60, "dg_weight": 0.50},
    "tour": "pga",
}


def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return DEFAULT_SETTINGS


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_cache: dict = {"data": None, "ts": 0}
_lock = Lock()


def get_data(force: bool = False) -> dict:
    """Return cached data, refreshing if stale or forced."""
    with _lock:
        now = time.time()
        if not force and _cache["data"] and (now - _cache["ts"]) < CACHE_TTL:
            return _cache["data"]

        settings = load_settings()
        try:
            data = _fetch(settings)
            data["error"] = None
        except Exception as exc:
            traceback.print_exc()
            data = _cache["data"] or {}
            data["error"] = str(exc)

        data["cached_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        _cache["data"] = data
        _cache["ts"] = time.time()
        return data


def _fetch(settings: dict) -> dict:
    load_dotenv()
    api_key = os.getenv("DATAGOLF_API_KEY")
    if not api_key:
        raise RuntimeError("DATAGOLF_API_KEY not set in .env")

    client = DataGolfClient(api_key)
    history = settings.get("history", DEFAULT_SETTINGS["history"])
    weights = settings.get("weights", DEFAULT_SETTINGS["weights"])
    tour = settings.get("tour", "pga")

    model = RankingModel(weights)

    short_raw = client.get_historical_sg_stats(tour=tour, n_rounds=history["short_rounds"])
    long_raw  = client.get_historical_sg_stats(tour=tour, n_rounds=history["long_rounds"])

    try:
        predictions_raw = client.get_pre_tournament_predictions(tour=tour)
    except Exception:
        predictions_raw = None

    df = model.build_pre_tournament(
        short_raw,
        predictions_raw,
        long_term_raw=long_raw,
        long_term_weight=1.0 - history["short_weight"],
    )
    df = model.blend_win_probs(df, dg_weight=history.get("dg_weight", 0.5))

    # Weekly PGA Tour stats
    stat_configs, season_weight = _load_weekly_stats()
    extra_cols = []
    if stat_configs:
        from datagolf.pgatour import _normalize_name
        import pandas as _pd
        pga = PGATourStats()
        normalized_names = df["player_name"].apply(
            lambda n: _normalize_name(str(n)) if not _pd.isna(n) else ""
        )
        for s in stat_configs:
            try:
                stat_df = pga.get_combined_stat(s["id"], label=s.get("label"), current_weight=season_weight)
                col = f"{s.get('label', s['id'])} Rk"
                lookup = stat_df.set_index("player_name")["combined_rank"]
                df[col] = normalized_names.map(lookup)
                df[col] = df[col].apply(lambda x: int(x) if not _pd.isna(x) else None)
                extra_cols.append(col)
            except Exception:
                pass

    # Compute user-influenced SG estimate for matchup model.
    # Re-weights the four SG components (ott/app/arg/putt) using the user's configured
    # weights, then blends it at MATCHUP_INFLUENCE into sg_total. On approach-heavy
    # courses where sg_app is upweighted, players who excel specifically at approach
    # (relative to their overall SG) receive a small boost.
    _SG_COMPONENTS = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    _MATCHUP_INFLUENCE = 0.08  # 8% — user weights, 92% raw sg_total

    _sg_comp_weights = {k: v for k, v in weights.items() if k in _SG_COMPONENTS}
    _total_sg_w = sum(_sg_comp_weights.values())
    if _total_sg_w > 0:
        import pandas as _pd
        _norm_w = {k: v / _total_sg_w for k, v in _sg_comp_weights.items()}
        _weighted_sg = sum(
            df[s].fillna(df[s].median()) * w
            for s, w in _norm_w.items()
            if s in df.columns
        )
        df["matchup_sg"] = (
            (1 - _MATCHUP_INFLUENCE) * df["sg_total"].fillna(0)
            + _MATCHUP_INFLUENCE * _weighted_sg
        )
    else:
        df["matchup_sg"] = df["sg_total"]

    # DraftKings outright odds
    dk_map = {}
    try:
        import math
        ou_raw = client.get_outrights(tour=tour)
        for p in (ou_raw.get("odds") or []):
            name = p.get("player_name")
            dk = p.get("draftkings")
            if name and dk is not None:
                try:
                    odds = float(dk)
                    prob = 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)
                    dk_map[name] = {"odds": odds, "prob": prob}
                except Exception:
                    pass
    except Exception:
        pass

    # Matchups — fetch first so matchup_round is available for the regression label
    matchups_html = ""
    matchup_round = ""
    matchup_no_data = False
    try:
        mu_raw = client.get_matchups(tour=tour)
        matchup_round = str(mu_raw.get("round_num") or mu_raw.get("round") or "")
    except Exception:
        matchup_no_data = True

    # Live tournament stats — putting regression signal
    # Identifies historically strong putters who are cold this week but striking it well.
    live_round = 0
    regression_html = ""
    _PUTT_REGRESSION_FACTOR = 0.10  # max ~0.20 stroke boost to matchup_sg
    try:
        import pandas as _pd3
        from datagolf.matchups import _normalize_name as _nn
        live_raw = client.get_live_tournament_stats(tour=tour, round="event", display="value")
        # stat_round returns "event" (the mode string), not a round number — use matchup_round - 1
        _sr = live_raw.get("round_num") or live_raw.get("round") or live_raw.get("stat_round") or 0
        try:
            live_round = int(_sr)
        except (ValueError, TypeError):
            live_round = 0

        # Parse live sg_putt and sg_t2g; key by normalized name to handle "Last, First" format
        live_sg: dict = {}
        for p in (live_raw.get("live_stats") or []):
            raw_name = (p.get("player_name") or "").strip()
            if not raw_name:
                continue
            norm = _nn(raw_name)  # handles "Burns, Sam" → "sam burns"
            if "stats" in p and isinstance(p["stats"], list):
                sg_map: dict = {}
                for s in p["stats"]:
                    k = s.get("stat") or s.get("stat_name") or ""
                    try:
                        sg_map[k] = float(s.get("value") or 0)
                    except Exception:
                        pass
                live_sg[norm] = {"live_sg_putt": sg_map.get("sg_putt"), "live_sg_t2g": sg_map.get("sg_t2g")}
            else:
                def _sf(v):
                    try: return float(v)
                    except Exception: return None
                live_sg[norm] = {"live_sg_putt": _sf(p.get("sg_putt")), "live_sg_t2g": _sf(p.get("sg_t2g"))}

        # Derive the completed round from matchup_round (next round) when API can't tell us
        if live_round == 0:
            try:
                live_round = max(int(matchup_round) - 1, 1) if matchup_round else (1 if live_sg else 0)
            except (ValueError, TypeError):
                live_round = 1 if live_sg else 0

        if live_sg:
            live_df_rows = [
                {"_norm": k, "live_sg_putt": v["live_sg_putt"], "live_sg_t2g": v["live_sg_t2g"]}
                for k, v in live_sg.items()
            ]
            live_merge = _pd3.DataFrame(live_df_rows)
            # Normalize model df names to match
            df["_norm"] = df["player_name"].apply(lambda n: _nn(str(n)) if _pd3.notna(n) else "")
            df = df.merge(live_merge, on="_norm", how="left").drop(columns=["_norm"])

            # putt_gap: positive means putting worse than historical average
            df["putt_gap_raw"] = df["sg_putt"] - df["live_sg_putt"]

            # Upward boost: historically good putter running cold, gated by live T2G.
            # They need to be creating birdie looks to capitalize on regression.
            # t2g_gate: 0→0, +0.5 SG T2G→1.0 (capped)
            t2g_gate = (df["live_sg_t2g"].fillna(0).clip(lower=0) / 0.5).clip(upper=1.0)
            upward_boost = _PUTT_REGRESSION_FACTOR * df["putt_gap_raw"].clip(lower=0) * t2g_gate

            # Downward penalty: player running hot on the putter, no T2G gate.
            # Putting regression happens regardless of T2G — and a hot putter with
            # poor T2G is converting a small number of looks at an unsustainable rate,
            # making regression even more likely.
            downward_penalty = _PUTT_REGRESSION_FACTOR * df["putt_gap_raw"].clip(upper=0)

            df["matchup_sg"] = df["matchup_sg"] + upward_boost.fillna(0) + downward_penalty.fillna(0)

            # Regression targets: historically good putters with a meaningful gap
            reg = df[
                (df["putt_gap_raw"].fillna(0) > 0.3) &       # meaningfully underperforming
                (df["sg_putt"].fillna(-99)    > 0.0)          # historically above-average putter
            ].copy()
            reg["regression_signal"] = (
                reg["putt_gap_raw"].clip(lower=0) *
                reg["live_sg_t2g"].fillna(0).clip(lower=0.1)  # small floor so gap alone registers
            )
            reg = reg.sort_values("regression_signal", ascending=False).head(10)
            if not reg.empty:
                regression_html = _regression_to_html(reg, live_round)
    except Exception:
        pass

    # Complete matchup parsing now that df has regression boosts applied
    try:
        if not matchup_no_data:
            mu_df = parse_matchups(mu_raw, model_df=df)
            if not mu_df.empty:
                matchups_html = _matchups_to_html(mu_df, matchup_round=matchup_round)
            else:
                matchup_no_data = True
    except Exception:
        matchup_no_data = True

    event_name = (
        (predictions_raw or {}).get("event_name")
        or short_raw.get("event_name")
        or "Pre-Tournament Rankings"
    )

    return {
        "event_name": event_name,
        "rankings_html": _rankings_to_html(df, extra_cols, dk_map),
        "extra_col_headers": extra_cols,
        "has_dk": bool(dk_map),
        "matchups_html": matchups_html,
        "matchup_round": matchup_round,
        "matchup_no_data": matchup_no_data,
        "regression_html": regression_html,
        "live_round": live_round,
        "weights": weights,
        "history": history,
    }


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _pct(v, signed=False):
    if v is None:
        return "<span class='dim'>-</span>"
    try:
        s = f"{v * 100:{'+' if signed else ''}.1f}%"
        if signed and v > 0:
            return f"<span class='pos'>{s}</span>"
        if signed and v < 0:
            return f"<span class='neg'>{s}</span>"
        return s
    except Exception:
        return "-"


def _num(v, signed=False):
    if v is None:
        return "<span class='dim'>-</span>"
    try:
        return f"{v:{'+' if signed else ''}.2f}"
    except Exception:
        return "-"


def _american(prob):
    if prob is None:
        return "<span class='dim'>-</span>"
    try:
        import math
        if math.isnan(prob) or prob <= 0 or prob >= 1:
            return "<span class='dim'>-</span>"
        if prob >= 0.5:
            return str(int(round(-(prob / (1 - prob)) * 100)))
        return f"+{int(round(((1 - prob) / prob) * 100))}"
    except Exception:
        return "-"


def _american_raw(odds):
    """Format raw American odds value."""
    try:
        v = float(odds)
        return f"+{int(v)}" if v > 0 else str(int(v))
    except Exception:
        return "<span class='dim'>-</span>"


def _rankings_to_html(df, extra_cols=None, dk_map=None) -> str:
    extra_cols = extra_cols or []
    dk_map = dk_map or {}
    rows = []
    for _, r in df.head(50).iterrows():
        name = r.get("player_name") or ""
        dk = dk_map.get(name, {})
        dk_odds_html = _american_raw(dk.get("odds")) if dk else "<span class='dim'>-</span>"

        # EV%: (my_prob / dk_implied - 1) * 100
        ev_html = "<span class='dim'>-</span>"
        my_p = r.get("model_win_prob")
        dk_p = dk.get("prob")
        if my_p and dk_p and dk_p > 0:
            import math
            if not (math.isnan(my_p) or math.isnan(dk_p)):
                ev = (my_p / dk_p - 1) * 100
                s = f"{ev:+.1f}%"
                ev_html = f"<span class='pos'>{s}</span>" if ev > 0 else f"<span class='dim'>{s}</span>"

        extra_cells = ""
        for col in extra_cols:
            v = r.get(col)
            import math as _math
            cell_val = "<span class='dim'>-</span>" if (v is None or (isinstance(v, float) and _math.isnan(v))) else str(int(v))
            extra_cells += f"<td>{cell_val}</td>"

        rows.append(f"""
        <tr>
          <td class="dim">{int(r['rank'])}</td>
          <td class="name">{name}</td>
          <td>{_pct(r.get('win_prob'))}</td>
          <td>{dk_odds_html}</td>
          <td>{_american(r.get('model_win_prob'))}</td>
          <td>{ev_html}</td>
          <td>{_num(r.get('sg_ott'), signed=True)}</td>
          <td>{_num(r.get('sg_app'), signed=True)}</td>
          <td>{_num(r.get('sg_arg'), signed=True)}</td>
          <td>{_num(r.get('sg_putt'), signed=True)}</td>
          <td>{_num(r.get('sg_total'), signed=True)}</td>
          <td>{_num(r.get('composite_score'), signed=True)}</td>
          {extra_cells}
        </tr>""")
    return "\n".join(rows)


def _fmt_odds(v):
    try:
        v = float(v)
        return f"+{int(v)}" if v > 0 else str(int(v))
    except Exception:
        return "-"


# Books to display and their labels (order matters for column order)
_BOOK_DISPLAY = {
    "draftkings": "DraftKings",
    "fanduel":    "FanDuel",
    "unibet":     "BetRivers",
    "caesars":    "Caesars",
    "betmgm":     "BetMGM",
}


def _regression_to_html(reg_df, live_round: int) -> str:
    """Render the putting regression targets table."""
    import math as _math
    rows = []
    for _, r in reg_df.iterrows():
        hist_p = r.get("sg_putt")
        live_p = r.get("live_sg_putt")
        gap    = r.get("putt_gap_raw", 0)
        t2g    = r.get("live_sg_t2g")

        def _safe(v):
            return v is not None and not (isinstance(v, float) and _math.isnan(v))

        live_p_html = (
            f"<span class='neg'>{live_p:+.2f}</span>"
            if _safe(live_p) and _safe(hist_p) and live_p < hist_p
            else _num(live_p, signed=True)
        )
        gap_html = f"<span class='pos'>+{gap:.2f}</span>" if gap and gap > 0 else _num(gap, signed=True)
        t2g_html = (
            f"<span class='pos'>{t2g:+.2f}</span>"
            if _safe(t2g) and t2g > 0
            else _num(t2g, signed=True)
        )

        rows.append(f"""
        <tr>
          <td class="name">{r['player_name']}</td>
          <td>{_num(hist_p, signed=True)}</td>
          <td>{live_p_html}</td>
          <td>{gap_html}</td>
          <td>{t2g_html}</td>
        </tr>""")

    return f"""
    <div class="reg-header">Putting Regression Targets — After Round {live_round}</div>
    <p class="reg-meta">Historically strong putters underperforming with the flat stick so far this week, while creating birdie looks tee-to-green. These players receive a boost in matchup edge calculations.</p>
    <table class="reg-table">
      <thead><tr>
        <th>Player</th><th>Hist SG:P</th><th>Live SG:P</th><th>Gap</th><th>Live T2G</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def _matchups_to_html(mu_df, min_edge: float = 0.05, matchup_round: str = "") -> str:
    import json as _json
    import math as _math
    import pandas as _pd
    round_label = f"Round {matchup_round}" if matchup_round else "this round"
    has_model = "p1_our_prob" in mu_df.columns and mu_df["p1_our_prob"].notna().any()

    def _safe(v):
        """Convert pandas NaN/NA to Python None so downstream math stays clean."""
        if v is None:
            return None
        try:
            if _pd.isna(v):
                return None
        except Exception:
            pass
        return v

    # ---------------------------------------------------------------------------
    # Phase 1: collect raw per-book data for each matchup pair
    # ---------------------------------------------------------------------------
    pair_data: dict = {}
    for _, r in mu_df.iterrows():
        key = (r["p1_name"], r["p2_name"])
        book = r.get("book") or ""
        is_displayable = bool(book and book in _BOOK_DISPLAY)

        if key not in pair_data:
            p1_pin = _safe(r.get("p1_pin_prob"))
            pair_data[key] = {
                "p1_our_prob": _safe(r.get("p1_our_prob")),
                "p2_our_prob": _safe(r.get("p2_our_prob")),
                "p1_pin_prob": p1_pin,
                "p2_pin_prob": _safe(r.get("p2_pin_prob")),
                "has_pin":     p1_pin is not None,
                "book_fairs":  {},   # {book: {p1, p2, p1_odds, p2_odds}}
                "book_edges":  {},   # populated in phase 2
                "max_edge":    0.0,
                "books":       {},
            }

        p1_mkt = _safe(r.get("p1_mkt_prob"))
        p2_mkt = _safe(r.get("p2_mkt_prob"))
        if is_displayable and p1_mkt is not None and p2_mkt is not None:
            pair_data[key]["book_fairs"][book] = {
                "p1":      float(p1_mkt),
                "p2":      float(p2_mkt),
                "p1_odds": _safe(r.get("p1_book_odds")),
                "p2_odds": _safe(r.get("p2_book_odds")),
            }

    # ---------------------------------------------------------------------------
    # Phase 2: per-book edge calcs + best-line + max_edge
    #
    # Three signals, all "soft book vs. some benchmark" (positive = value at soft book):
    #   vs_our  — our model prob minus soft-book fair prob
    #   vs_pin  — Pinnacle fair prob minus soft-book fair prob
    #   vs_mkt  — avg of OTHER books' fair probs minus soft-book fair prob
    # ---------------------------------------------------------------------------
    def _best_val(vals):
        valid = [x for x in vals if x is not None]
        return max(valid) if valid else None

    for v in pair_data.values():
        bf     = v["book_fairs"]
        p1_our = v["p1_our_prob"]
        p2_our = v["p2_our_prob"]
        p1_pin = v["p1_pin_prob"]
        p2_pin = v["p2_pin_prob"]

        book_edges: dict = {}
        for book, fairs in bf.items():
            bp1, bp2 = fairs["p1"], fairs["p2"]
            # Market consensus = average fair prob of ALL OTHER displayed books
            others_p1 = [f["p1"] for b, f in bf.items() if b != book]
            others_p2 = [f["p2"] for b, f in bf.items() if b != book]
            mkt_p1 = sum(others_p1) / len(others_p1) if others_p1 else None
            mkt_p2 = sum(others_p2) / len(others_p2) if others_p2 else None
            book_edges[book] = {
                "p1_vs_our": round(p1_our - bp1, 4) if p1_our is not None else None,
                "p2_vs_our": round(p2_our - bp2, 4) if p2_our is not None else None,
                "p1_vs_pin": round(p1_pin - bp1, 4) if p1_pin is not None else None,
                "p2_vs_pin": round(p2_pin - bp2, 4) if p2_pin is not None else None,
                "p1_vs_mkt": round(mkt_p1 - bp1, 4) if mkt_p1 is not None else None,
                "p2_vs_mkt": round(mkt_p2 - bp2, 4) if mkt_p2 is not None else None,
            }

        # "all" entry = best edge across books for each signal (shown when no book is selected)
        sigs = ("p1_vs_our", "p2_vs_our", "p1_vs_pin", "p2_vs_pin", "p1_vs_mkt", "p2_vs_mkt")
        book_edges["all"] = {s: _best_val([e[s] for e in book_edges.values()]) for s in sigs}
        v["book_edges"] = book_edges

        # Best line per player = book with highest American odds (best price for bettor)
        def _best_book(odds_key, _bf=bf):
            best_b, best_v = None, -9999
            for b, fairs in _bf.items():
                o = fairs.get(odds_key)
                if o is not None:
                    try:
                        ov = float(o)
                        if ov > best_v:
                            best_v, best_b = ov, b
                    except Exception:
                        pass
            return best_b

        v["p1_best_book"] = _best_book("p1_odds")
        v["p2_best_book"] = _best_book("p2_odds")
        for book, fairs in bf.items():
            v["books"][book] = {"p1_odds": fairs["p1_odds"], "p2_odds": fairs["p2_odds"]}

        ae = book_edges["all"]
        v["max_edge"] = max(
            (x if (x is not None and not _math.isnan(x)) else 0)
            for s in sigs
            for x in [ae.get(s)]
        )

    sorted_pairs = sorted(pair_data.items(), key=lambda x: x[1]["max_edge"], reverse=True)
    if has_model:
        sorted_pairs = [(k, v) for k, v in sorted_pairs if v["max_edge"] >= min_edge]
    sorted_pairs = [(k, v) for k, v in sorted_pairs if v["books"]]

    if not sorted_pairs:
        return f"<p class='dim'>No matchups with edge ≥ 5% for {round_label}. Lines may not be posted yet, or the market is well-priced.</p>"

    has_pin_data = any(v.get("has_pin") for _, v in sorted_pairs)
    has_mkt_data = any(
        v["book_edges"].get("all", {}).get("p1_vs_mkt") is not None
        or v["book_edges"].get("all", {}).get("p2_vs_mkt") is not None
        for _, v in sorted_pairs
    )

    seen_books: set = set()
    for _, v in sorted_pairs:
        seen_books.update(v["books"].keys())

    options = '<option value="all">All Books</option>\n'
    for b in _BOOK_DISPLAY:
        if b in seen_books:
            options += f'      <option value="{b}">{_BOOK_DISPLAY[b]}</option>\n'

    # JS: format edge value, and update edge columns + player highlights on book change
    filter_html = f"""<div class="filter-bar">
  <label for="bookFilter">Filter by book:</label>
  <select id="bookFilter" onchange="filterByBook(this.value)">
    {options}  </select>
</div>
<script>
function _fmtEdge(val, thr) {{
  if (val === null || val === undefined) return "<span class='dim'>-</span>";
  var pct = (val * 100).toFixed(1);
  var s = (val >= 0 ? '+' : '') + pct + '%';
  return val >= thr ? "<span class='pos'>" + s + "</span>"
                    : "<span class='dim neg'>" + s + "</span>";
}}
function filterByBook(book) {{
  document.querySelectorAll('.mu-card').forEach(function(card) {{
    var books = (card.dataset.books || '').split(' ');
    if (book !== 'all' && books.indexOf(book) < 0) {{
      card.style.display = 'none'; return;
    }}
    card.style.display = '';
    // Show/hide individual book columns
    card.querySelectorAll('.book-col[data-book]').forEach(function(col) {{
      col.style.display = (book === 'all' || col.dataset.book === book) ? '' : 'none';
    }});
    // Update edge columns from book-specific data
    var edges = JSON.parse(card.dataset.edges || '{{}}');
    var e = edges[book] || edges['all'] || {{}};
    card.querySelectorAll('[data-edge-type]').forEach(function(col) {{
      var type = col.dataset.edgeType;
      var thr  = parseFloat(col.dataset.threshold || '0.05');
      var p1el = col.querySelector('[data-player="p1"]');
      var p2el = col.querySelector('[data-player="p2"]');
      if (p1el) p1el.innerHTML = _fmtEdge(e['p1_' + type], thr);
      if (p2el) p2el.innerHTML = _fmtEdge(e['p2_' + type], thr);
    }});
    // Re-evaluate player name highlight for selected book
    card.querySelectorAll('.mu-player').forEach(function(el, idx) {{
      var pref = idx === 0 ? 'p1' : 'p2';
      var any  = ['vs_our','vs_pin','vs_mkt'].some(function(t) {{
        var v = e[pref + '_' + t]; return v !== null && v !== undefined && v >= 0.05;
      }});
      el.className = any ? 'mu-player player-edge' : 'mu-player';
    }});
  }});
}}
</script>"""

    def _fe(val, threshold=None):
        if val is None:
            return "<span class='dim'>-</span>"
        try:
            if _math.isnan(val):
                return "<span class='dim'>-</span>"
        except (TypeError, ValueError):
            pass
        t = threshold if threshold is not None else min_edge
        s = f"{val * 100:+.1f}%"
        return f"<span class='pos'>{s}</span>" if val >= t else f"<span class='dim neg'>{s}</span>"

    cards = []
    for (p1, p2), v in sorted_pairs:
        ae      = v["book_edges"].get("all", {})
        p1_best = v.get("p1_best_book")
        p2_best = v.get("p2_best_book")
        books_attr = " ".join(v["books"].keys())
        def _clean(obj):
            """Recursively replace NaN/Inf with None so JSON.parse never throws."""
            if isinstance(obj, dict):
                return {k: _clean(v2) for k, v2 in obj.items()}
            if isinstance(obj, float) and (_math.isnan(obj) or _math.isinf(obj)):
                return None
            return obj
        edges_json = _json.dumps(_clean(v["book_edges"]))

        p1_has_edge = any(ae.get(k) is not None and ae[k] >= min_edge
                          for k in ("p1_vs_our", "p1_vs_pin", "p1_vs_mkt"))
        p2_has_edge = any(ae.get(k) is not None and ae[k] >= min_edge
                          for k in ("p2_vs_our", "p2_vs_pin", "p2_vs_mkt"))

        # Book columns — best line for each player highlighted green
        book_cols_html = ""
        for book in _BOOK_DISPLAY:
            bdata = v["books"].get(book)
            if bdata is None:
                continue
            p1_cls = "book-line best-line" if book == p1_best else "book-line"
            p2_cls = "book-line best-line" if book == p2_best else "book-line"
            book_cols_html += f"""
      <div class="book-col" data-book="{book}">
        <div class="book-label">{_BOOK_DISPLAY[book]}</div>
        <div class="{p1_cls}">{_fmt_odds(bdata['p1_odds'])}</div>
        <div class="{p2_cls}">{_fmt_odds(bdata['p2_odds'])}</div>
      </div>"""

        # Analysis columns (our model odds + three edge signals)
        analysis_html = ""
        if has_model and v["p1_our_prob"] is not None:
            pin_col_html = ""
            if has_pin_data:
                pin_col_html = f"""
      <div class="book-col edge-col" data-edge-type="vs_pin" data-threshold="0.05">
        <div class="book-label">vs Pinnacle</div>
        <div class="book-line" data-player="p1">{_fe(ae.get('p1_vs_pin'))}</div>
        <div class="book-line" data-player="p2">{_fe(ae.get('p2_vs_pin'))}</div>
      </div>"""
            mkt_col_html = ""
            if has_mkt_data:
                mkt_col_html = f"""
      <div class="book-col edge-col mkt-col" data-edge-type="vs_mkt" data-threshold="0.03">
        <div class="book-label">vs Market</div>
        <div class="book-line" data-player="p1">{_fe(ae.get('p1_vs_mkt'), threshold=0.03)}</div>
        <div class="book-line" data-player="p2">{_fe(ae.get('p2_vs_mkt'), threshold=0.03)}</div>
      </div>"""
            analysis_html = f"""
      <div class="book-col model-col">
        <div class="book-label">Our Model</div>
        <div class="book-line">{_american(v['p1_our_prob'])}</div>
        <div class="book-line">{_american(v['p2_our_prob'])}</div>
      </div>
      <div class="book-col edge-col" data-edge-type="vs_our" data-threshold="0.05">
        <div class="book-label">vs Our Odds</div>
        <div class="book-line" data-player="p1">{_fe(ae.get('p1_vs_our'))}</div>
        <div class="book-line" data-player="p2">{_fe(ae.get('p2_vs_our'))}</div>
      </div>{pin_col_html}{mkt_col_html}"""

        p1_cls = "mu-player player-edge" if p1_has_edge else "mu-player"
        p2_cls = "mu-player player-edge" if p2_has_edge else "mu-player"

        cards.append(f"""
    <div class="mu-card" data-books="{books_attr}" data-edges='{edges_json}'>
      <div class="mu-players">
        <div class="{p1_cls}">{p1}</div>
        <div class="{p2_cls}">{p2}</div>
      </div>
      <div class="mu-books">{book_cols_html}
        <div class="edge-sep"></div>{analysis_html}
      </div>
    </div>""")

    return filter_html + '\n<div class="mu-grid">' + "".join(cards) + "\n</div>"


# ---------------------------------------------------------------------------
# Shared HTML pieces
# ---------------------------------------------------------------------------

_CSS = """
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f1117; color: #e2e8f0; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; padding: 24px; }
  h1 { font-size: 20px; color: #63b3ed; margin-bottom: 4px; }
  .meta { color: #718096; font-size: 12px; margin-bottom: 6px; }
  .weights { color: #718096; font-size: 11px; margin-bottom: 16px; }
  .error { background: #742a2a; color: #fed7d7; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px; }
  table { border-collapse: collapse; width: 100%; }
  th { text-align: right; color: #718096; padding: 6px 10px; border-bottom: 1px solid #2d3748; white-space: nowrap; }
  th:nth-child(2), th:first-child { text-align: left; }
  td { padding: 5px 10px; text-align: right; border-bottom: 1px solid #1a202c; white-space: nowrap; }
  td.name, td:first-child { text-align: left; }
  td.dim, span.dim { color: #4a5568; }
  span.pos, .pos { color: #68d391; }
  span.neg { color: #fc8181; }
  .bold { font-weight: 600; }
  tr:hover td { background: #1a202c; }
  nav { display: flex; gap: 8px; margin-bottom: 20px; }
  .nav-btn { padding: 6px 14px; background: #2d3748; color: #a0aec0; border-radius: 4px; text-decoration: none; font-size: 12px; }
  .nav-btn:hover { background: #4a5568; }
  .nav-btn.active { background: #2b6cb0; color: #bee3f8; }
  .round-badge { display: inline-block; background: #2b6cb0; color: #bee3f8; padding: 3px 12px; border-radius: 12px; font-size: 12px; margin-bottom: 16px; }
  .filter-bar { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
  .filter-bar label { color: #a0aec0; font-size: 12px; }
  .filter-bar select { background: #2d3748; color: #e2e8f0; border: 1px solid #4a5568; padding: 4px 10px; border-radius: 4px; font-size: 12px; cursor: pointer; }
  .mu-grid { display: flex; flex-direction: column; gap: 8px; }
  .mu-card { background: #1a202c; border: 1px solid #2d3748; border-radius: 8px; padding: 12px 16px; display: flex; align-items: center; gap: 20px; }
  .mu-card:hover { border-color: #4a5568; }
  .mu-players { display: flex; flex-direction: column; gap: 10px; min-width: 160px; flex-shrink: 0; }
  .mu-player { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 13px; }
  .player-edge { color: #68d391; font-weight: 600; }
  .mu-books { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }
  .book-col { display: flex; flex-direction: column; gap: 6px; min-width: 64px; }
  .book-label { color: #718096; font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; }
  .book-line { text-align: right; font-size: 13px; }
  .model-col .book-line { color: #90cdf4; }
  .edge-col .book-line { font-weight: 500; }
  .mkt-col .book-label { color: #f6ad55; }
  .mkt-col .book-line { font-weight: 500; color: #a0aec0; }
  .best-line { color: #68d391; font-weight: 600; }
  .edge-sep { width: 1px; background: #2d3748; align-self: stretch; margin: 0 8px; flex-shrink: 0; }
  span.neg { color: #4a5568; }
  .reg-header { font-size: 14px; color: #63b3ed; margin: 28px 0 4px; font-weight: 600; }
  .reg-meta { color: #718096; font-size: 11px; margin-bottom: 12px; }
  .reg-table { margin-bottom: 28px; width: auto; }
  .reg-table th, .reg-table td { padding: 5px 14px; }
  .reg-table th:first-child, .reg-table td:first-child { text-align: left; }
  /* Settings page */
  .settings-form { max-width: 640px; }
  .settings-section { margin-bottom: 28px; }
  .settings-section h2 { font-size: 14px; color: #63b3ed; margin-bottom: 10px; font-weight: 600; border-bottom: 1px solid #2d3748; padding-bottom: 6px; }
  .settings-hint { color: #718096; font-size: 11px; margin-bottom: 10px; }
  .settings-table { width: 100%; margin-bottom: 6px; }
  .settings-table th { color: #718096; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; padding: 5px 8px; font-weight: normal; text-align: left; border-bottom: 1px solid #2d3748; }
  .settings-table td { padding: 5px 8px; vertical-align: middle; border-bottom: 1px solid #1a202c; }
  .settings-table td.desc-col { color: #4a5568; font-size: 11px; }
  input.si { background: #2d3748; color: #e2e8f0; border: 1px solid #4a5568; padding: 5px 8px; border-radius: 4px; font-size: 12px; font-family: inherit; width: 72px; }
  select.si { background: #2d3748; color: #e2e8f0; border: 1px solid #4a5568; padding: 5px 8px; border-radius: 4px; font-size: 12px; font-family: inherit; }
  .stat-rows { display: flex; flex-direction: column; gap: 6px; margin-bottom: 8px; }
  .stat-row { display: flex; align-items: center; gap: 8px; }
  .stat-row .sid { background: #2d3748; color: #e2e8f0; border: 1px solid #4a5568; padding: 5px 8px; border-radius: 4px; font-size: 12px; font-family: inherit; width: 84px; }
  .stat-row .slbl { background: #2d3748; color: #e2e8f0; border: 1px solid #4a5568; padding: 5px 8px; border-radius: 4px; font-size: 12px; font-family: inherit; width: 180px; }
  .btn { padding: 7px 18px; border-radius: 4px; border: none; cursor: pointer; font-size: 12px; font-family: inherit; font-weight: 600; }
  .btn-primary { background: #2b6cb0; color: #bee3f8; }
  .btn-primary:hover { background: #2c5282; }
  .btn-danger { padding: 4px 8px; background: #742a2a; color: #fed7d7; border: none; border-radius: 3px; cursor: pointer; font-size: 11px; font-family: inherit; }
  .btn-add { padding: 5px 12px; background: #276749; color: #c6f6d5; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; font-family: inherit; }
"""

RANKINGS_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="300">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ event_name }}</title>
  <style>{{ css }}</style>
</head>
<body>
  {% if error %}<div class="error">⚠ {{ error }} — showing last cached result.</div>{% endif %}

  <h1>{{ event_name }}</h1>
  <div class="meta">Updated: {{ cached_at }}</div>
  <div class="weights">{{ weights_str }}</div>

  <nav>
    <a class="nav-btn active" href="/">Rankings</a>
    <a class="nav-btn" href="/matchups">Matchups</a>
    <a class="nav-btn" href="/settings">Settings</a>
    <a class="nav-btn" href="/refresh">&#x21bb; Refresh</a>
  </nav>

  <table>
    <thead><tr>
      <th>#</th><th>Player</th>
      <th>Win%</th><th>DK Odds</th><th>My Odds</th><th>EV%</th>
      <th>SG:OTT</th><th>SG:APP</th><th>SG:ARG</th><th>SG:PUT</th><th>SG:TOT</th>
      <th>Score</th>
      {% for col in extra_col_headers %}<th>{{ col }}</th>{% endfor %}
    </tr></thead>
    <tbody>{{ rankings_html | safe }}</tbody>
  </table>

  <p class="meta" style="margin-top:20px">Auto-refreshes every 5 minutes.</p>
</body>
</html>"""

MATCHUPS_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="300">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Matchups — {{ event_name }}</title>
  <style>{{ css }}</style>
</head>
<body>
  {% if error %}<div class="error">⚠ {{ error }} — showing last cached result.</div>{% endif %}

  <h1>{{ event_name }}</h1>
  <div class="meta">Updated: {{ cached_at }}</div>

  <nav>
    <a class="nav-btn" href="/">Rankings</a>
    <a class="nav-btn active" href="/matchups">Matchups</a>
    <a class="nav-btn" href="/settings">Settings</a>
    <a class="nav-btn" href="/matchups/refresh">&#x21bb; Refresh</a>
  </nav>

  {% if matchup_round %}
  <div class="round-badge">Round {{ matchup_round }}</div>
  {% endif %}

  {% if matchups_html %}
  {{ matchups_html | safe }}
  {% elif matchup_no_data and matchup_round == "1" %}
  <p class="meta">Round 1 is underway — Round 2 matchup lines haven't been posted yet. Check back after the round completes.</p>
  {% elif matchup_no_data %}
  <p class="meta">No matchup data available for this round yet. Try refreshing once the round starts.</p>
  {% endif %}

  {% if regression_html %}
  {{ regression_html | safe }}
  {% endif %}

  <p class="meta" style="margin-top:20px">Auto-refreshes every 5 minutes.</p>
</body>
</html>"""


SETTINGS_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Settings &#8212; DataGolf</title>
  <style>{{ css }}</style>
</head>
<body>
  <h1>Settings</h1>
  <div class="meta" style="margin-bottom:16px">Adjust model weights and stat overlays. Click Save &amp; Run to apply and re-fetch.</div>

  <nav>
    <a class="nav-btn" href="/">Rankings</a>
    <a class="nav-btn" href="/matchups">Matchups</a>
    <a class="nav-btn active" href="/settings">Settings</a>
  </nav>

  <form class="settings-form" method="POST" action="/settings">

    <div class="settings-section">
      <h2>SG Weights</h2>
      <p class="settings-hint">Relative values &#8212; normalized to 100% automatically. Tune per course type.</p>
      <table class="settings-table">
        <thead><tr><th>Component</th><th>Weight</th><th class="desc-col">Note</th></tr></thead>
        <tbody>
          <tr><td>SG: Total</td><td><input class="si" type="number" name="w_sg_total" value="{{ wv.sg_total }}" min="0" max="100" step="1"></td><td class="desc-col">Overall SG baseline</td></tr>
          <tr><td>Win Probability</td><td><input class="si" type="number" name="w_win_prob" value="{{ wv.win_prob }}" min="0" max="100" step="1"></td><td class="desc-col">DataGolf win model</td></tr>
          <tr><td>SG: Approach</td><td><input class="si" type="number" name="w_sg_app" value="{{ wv.sg_app }}" min="0" max="100" step="1"></td><td class="desc-col">Biggest course differentiator</td></tr>
          <tr><td>SG: Putting</td><td><input class="si" type="number" name="w_sg_putt" value="{{ wv.sg_putt }}" min="0" max="100" step="1"></td><td class="desc-col">Putting</td></tr>
          <tr><td>SG: Off-the-Tee</td><td><input class="si" type="number" name="w_sg_ott" value="{{ wv.sg_ott }}" min="0" max="100" step="1"></td><td class="desc-col">Driving</td></tr>
          <tr><td>SG: Around Green</td><td><input class="si" type="number" name="w_sg_arg" value="{{ wv.sg_arg }}" min="0" max="100" step="1"></td><td class="desc-col">Short game</td></tr>
        </tbody>
      </table>
    </div>

    <div class="settings-section">
      <h2>History Window</h2>
      <table class="settings-table">
        <tbody>
          <tr><td>Short window (rounds)</td><td><input class="si" type="number" name="short_rounds" value="{{ h.short_rounds }}" min="4" max="50" step="1"></td><td class="desc-col">Recent form window</td></tr>
          <tr><td>Long window (rounds)</td><td><input class="si" type="number" name="long_rounds" value="{{ h.long_rounds }}" min="20" max="200" step="1"></td><td class="desc-col">Long-term baseline</td></tr>
          <tr><td>Short-term weight</td><td><input class="si" type="number" name="short_weight" value="{{ h.short_weight }}" min="0" max="1" step="0.05"></td><td class="desc-col">0.60 = 60% recent, 40% long-term</td></tr>
          <tr><td>DG model blend</td><td><input class="si" type="number" name="dg_weight" value="{{ h.dg_weight }}" min="0" max="1" step="0.05"></td><td class="desc-col">0.50 = 50% DG win prob, 50% our model</td></tr>
        </tbody>
      </table>
    </div>

    <div class="settings-section">
      <h2>Tour</h2>
      <select class="si" name="tour">
        <option value="pga" {{ 'selected' if tour == 'pga' else '' }}>PGA Tour</option>
        <option value="euro" {{ 'selected' if tour == 'euro' else '' }}>European Tour</option>
      </select>
    </div>

    <div class="settings-section">
      <h2>Weekly Stats Overlay</h2>
      <p class="settings-hint">Choose stats from the catalog or enter a custom ID from pgatour.com/stats/detail/&lt;ID&gt;. Shown as extra columns on the rankings page.</p>

      <div style="display:flex;gap:8px;align-items:center;margin-bottom:12px">
        <select class="si" id="stat-catalog" style="width:260px">
          {% for group in catalog %}
          <optgroup label="{{ group.category }}">
            {% for stat in group.stats %}
            <option value="{{ stat.id }}">{{ stat.title }}</option>
            {% endfor %}
          </optgroup>
          {% endfor %}
        </select>
        <button type="button" class="btn-add" onclick="addFromCatalog()">+ Add</button>
      </div>

      <div class="stat-rows" id="stat-rows">
        {% for stat in ws_stats %}
        <div class="stat-row">
          <input type="text" class="sid" name="stat_id" placeholder="Stat ID" value="{{ stat.id }}">
          <input type="text" class="slbl" name="stat_label" placeholder="Label" value="{{ stat.label }}">
          <button type="button" class="btn-danger" onclick="this.parentElement.remove()">&#x2715;</button>
        </div>
        {% endfor %}
      </div>
      <button type="button" class="btn-add" onclick="addStatRow()" style="margin-bottom:14px">+ Custom stat</button>

      <div style="margin-top:6px;display:flex;align-items:center;gap:10px">
        <label style="color:#a0aec0;font-size:12px">Season blend weight:</label>
        <input class="si" type="number" name="season_weight" value="{{ ws_season_weight }}" min="0" max="1" step="0.05">
        <span class="settings-hint" style="margin:0">0.6 = 60% current season, 40% prior</span>
      </div>
    </div>

    <button type="submit" class="btn btn-primary">Save &amp; Run Model</button>
  </form>

  <script>
  function _makeStatRow(id, label) {
    var row = document.createElement('div');
    row.className = 'stat-row';
    var sid = document.createElement('input');
    sid.type = 'text'; sid.className = 'sid'; sid.name = 'stat_id';
    sid.placeholder = 'Stat ID'; sid.value = id || '';
    var slbl = document.createElement('input');
    slbl.type = 'text'; slbl.className = 'slbl'; slbl.name = 'stat_label';
    slbl.placeholder = 'Label'; slbl.value = label || '';
    var btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'btn-danger'; btn.textContent = '✕';
    btn.onclick = function() { this.parentElement.remove(); };
    row.appendChild(sid); row.appendChild(slbl); row.appendChild(btn);
    document.getElementById('stat-rows').appendChild(row);
  }
  function addStatRow() { _makeStatRow('', ''); }
  function addFromCatalog() {
    var sel = document.getElementById('stat-catalog');
    if (!sel || sel.selectedIndex < 0) return;
    var opt = sel.options[sel.selectedIndex];
    _makeStatRow(opt.value, opt.text);
  }
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    data = get_data()
    settings = load_settings()
    w = settings.get("weights", {})
    weights_str = "  ".join(f"{k}={v:.0%}" for k, v in w.items())
    return render_template_string(
        RANKINGS_TEMPLATE,
        css=_CSS,
        event_name=data.get("event_name", "DataGolf Rankings"),
        cached_at=data.get("cached_at", "—"),
        error=data.get("error"),
        rankings_html=data.get("rankings_html", ""),
        weights_str=weights_str,
        extra_col_headers=data.get("extra_col_headers", []),
    )


@app.route("/matchups")
def matchups():
    data = get_data()
    return render_template_string(
        MATCHUPS_TEMPLATE,
        css=_CSS,
        event_name=data.get("event_name", "DataGolf Rankings"),
        cached_at=data.get("cached_at", "—"),
        error=data.get("error"),
        matchups_html=data.get("matchups_html", ""),
        matchup_round=data.get("matchup_round", ""),
        matchup_no_data=data.get("matchup_no_data", False),
        regression_html=data.get("regression_html", ""),
    )


@app.route("/refresh")
def refresh():
    get_data(force=True)
    return redirect("/")


@app.route("/matchups/refresh")
def matchups_refresh():
    get_data(force=True)
    return redirect("/matchups")


@app.route("/api/data")
def api_data():
    data = get_data()
    return jsonify({
        "event_name": data.get("event_name"),
        "cached_at": data.get("cached_at"),
        "error": data.get("error"),
    })


@app.route("/debug/regression")
def debug_regression():
    """Diagnose the putting regression pipeline."""
    import traceback as _tb
    from dotenv import load_dotenv as _lde
    _lde()
    api_key = os.getenv("DATAGOLF_API_KEY")
    result = {}
    try:
        client = DataGolfClient(api_key)
        settings = load_settings()
        tour = settings.get("tour", "pga")

        # Step 1: live stats fetch
        try:
            live_raw = client.get_live_tournament_stats(tour=tour, round="event", display="value")
            result["live_round"] = live_raw.get("round_num") or live_raw.get("round")
            result["live_keys"] = list(live_raw.keys())
            players = live_raw.get("live_stats") or []
            result["live_player_count"] = len(players)
            if players:
                sample = players[0]
                result["live_sample_keys"] = list(sample.keys())
                result["live_sample_name"] = sample.get("player_name")
                result["live_sample_sg_putt"] = sample.get("sg_putt")
                result["live_sample_sg_t2g"] = sample.get("sg_t2g")
                # check nested stats
                if "stats" in sample and isinstance(sample["stats"], list):
                    result["live_stats_nested"] = True
                    result["live_stats_keys"] = [s.get("stat") or s.get("stat_name") for s in sample["stats"]]
                else:
                    result["live_stats_nested"] = False
        except Exception as e:
            result["live_fetch_error"] = str(e)

        # Step 2: model df
        try:
            data = get_data()
            result["matchup_round"] = data.get("matchup_round")
            result["live_round_cache"] = data.get("live_round")
            result["regression_html_len"] = len(data.get("regression_html") or "")
            result["regression_html_preview"] = (data.get("regression_html") or "")[:200]
        except Exception as e:
            result["cache_error"] = str(e)

    except Exception as e:
        result["error"] = _tb.format_exc()

    return jsonify(result)


@app.route("/debug/books")
def debug_books():
    """Show all book keys returned by the DataGolf matchups endpoint for a live matchup."""
    import traceback as _tb
    from dotenv import load_dotenv as _lde
    _lde()
    api_key = os.getenv("DATAGOLF_API_KEY")
    result = {}
    try:
        client = DataGolfClient(api_key)
        settings = load_settings()
        tour = settings.get("tour", "pga")
        mu_raw = client.get_matchups(tour=tour)
        matchups = mu_raw.get("match_list") or mu_raw.get("matchups") or mu_raw.get("data", [])
        result["matchup_count"] = len(matchups)
        if matchups:
            sample = matchups[0]
            odds_section = sample.get("odds") or {}
            result["books_available"] = sorted(odds_section.keys())
            result["has_pinnacle"] = "pinnacle" in odds_section
            result["sample_p1"] = sample.get("p1_player_name") or sample.get("p1")
            result["sample_p2"] = sample.get("p2_player_name") or sample.get("p2")
            # Show Pinnacle lines if present
            pin = odds_section.get("pinnacle") or {}
            result["pinnacle_lines"] = pin
            result["datagolf_lines"] = odds_section.get("datagolf") or {}
    except Exception:
        result["error"] = _tb.format_exc()
    return jsonify(result)


@app.route("/settings", methods=["GET"])
def settings_page():
    s = load_settings()
    w = dict(DEFAULT_WEIGHTS)
    w.update(s.get("weights", {}))
    wv = {k: int(round(v * 100)) for k, v in w.items()}

    h = dict(DEFAULT_SETTINGS["history"])
    h.update(s.get("history", {}))

    tour = s.get("tour", "pga")

    ws_stats = []
    ws_season_weight = 0.6
    if os.path.exists(WEEKLY_STATS_FILE):
        with open(WEEKLY_STATS_FILE) as f:
            wsc = json.load(f)
        ws_stats = wsc.get("stats", [])
        ws_season_weight = wsc.get("season_blend", {}).get("current_weight", 0.6)

    return render_template_string(
        SETTINGS_TEMPLATE,
        css=_CSS,
        wv=wv,
        h=h,
        tour=tour,
        ws_stats=ws_stats,
        ws_season_weight=ws_season_weight,
        catalog=CURATED_STATS,
    )


@app.route("/settings", methods=["POST"])
def settings_save():
    def _flt(key, default):
        try:
            return float(request.form.get(key, default))
        except (ValueError, TypeError):
            return float(default)

    def _int(key, default):
        try:
            return int(request.form.get(key, default))
        except (ValueError, TypeError):
            return int(default)

    weights = {
        "sg_total": _flt("w_sg_total", 30) / 100,
        "win_prob": _flt("w_win_prob", 30) / 100,
        "sg_app":   _flt("w_sg_app",   15) / 100,
        "sg_putt":  _flt("w_sg_putt",  12) / 100,
        "sg_ott":   _flt("w_sg_ott",    8) / 100,
        "sg_arg":   _flt("w_sg_arg",    5) / 100,
    }

    history = {
        "short_rounds": _int("short_rounds", 12),
        "long_rounds":  _int("long_rounds",  60),
        "short_weight": _flt("short_weight", 0.60),
        "dg_weight":    _flt("dg_weight",    0.50),
    }

    settings = {
        "weights": weights,
        "history": history,
        "tour": request.form.get("tour", "pga"),
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)

    stat_ids    = request.form.getlist("stat_id")
    stat_labels = request.form.getlist("stat_label")
    stats = [{"id": sid.strip(), "label": lbl.strip()}
             for sid, lbl in zip(stat_ids, stat_labels) if sid.strip()]
    weekly = {
        "season_blend": {"current_weight": _flt("season_weight", 0.6)},
        "stats": stats,
    }
    with open(WEEKLY_STATS_FILE, "w") as f:
        json.dump(weekly, f, indent=2)

    get_data(force=True)
    return redirect("/")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"Starting DataGolf rankings server on http://0.0.0.0:{port}")
    print(f"Settings loaded from: {SETTINGS_FILE if os.path.exists(SETTINGS_FILE) else 'defaults'}")
    app.run(host="0.0.0.0", port=port, debug=False)
