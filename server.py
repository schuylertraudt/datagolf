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
from flask import Flask, jsonify, redirect, render_template_string

from datagolf.client import DataGolfClient
from datagolf.matchups import parse_matchups
from datagolf.pgatour import PGATourStats
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

    # Matchups
    matchups_html = ""
    matchup_round = ""
    try:
        mu_raw = client.get_matchups(tour=tour)
        matchup_round = str(mu_raw.get("round_num") or mu_raw.get("round") or "")
        mu_df = parse_matchups(mu_raw, model_df=df)
        if not mu_df.empty:
            matchups_html = _matchups_to_html(mu_df)
    except Exception:
        pass

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


def _matchups_to_html(mu_df, min_edge: float = 0.03) -> str:
    has_model = "p1_our_prob" in mu_df.columns and mu_df["p1_our_prob"].notna().any()

    # Group rows by matchup pair, collecting all books
    pair_data: dict = {}
    for _, r in mu_df.iterrows():
        key = (r["p1_name"], r["p2_name"])
        if key not in pair_data:
            pair_data[key] = {
                "p1_our_prob": r.get("p1_our_prob"),
                "p2_our_prob": r.get("p2_our_prob"),
                "p1_edge": r.get("p1_edge"),
                "p2_edge": r.get("p2_edge"),
                "max_edge": float(r.get("max_edge") or 0),
                "books": {},
            }
        book = r.get("book") or ""
        if book:
            pair_data[key]["books"][book] = {
                "p1_odds": r.get("p1_book_odds"),
                "p2_odds": r.get("p2_book_odds"),
            }

    sorted_pairs = sorted(pair_data.items(), key=lambda x: x[1]["max_edge"], reverse=True)
    if has_model:
        sorted_pairs = [(k, v) for k, v in sorted_pairs if v["max_edge"] >= min_edge]

    if not sorted_pairs:
        return "<p class='dim'>No matchups with edge ≥ 3%. Try refreshing after round starts.</p>"

    # Collect books in order of first appearance
    all_books: list = []
    seen_books: set = set()
    for _, v in sorted_pairs:
        for b in v["books"]:
            if b not in seen_books:
                seen_books.add(b)
                all_books.append(b)

    # Book filter dropdown + JS
    options = '<option value="all">All Books</option>\n'
    for b in all_books:
        options += f'      <option value="{b}">{b.title()}</option>\n'

    filter_html = f"""<div class="filter-bar">
  <label for="bookFilter">Filter by book:</label>
  <select id="bookFilter" onchange="filterByBook(this.value)">
    {options}  </select>
</div>
<script>
function filterByBook(book) {{
  document.querySelectorAll('.mu-card').forEach(function(card) {{
    if (book === 'all') {{
      card.style.display = '';
    }} else {{
      var books = (card.dataset.books || '').split(' ');
      card.style.display = books.indexOf(book) >= 0 ? '' : 'none';
    }}
  }});
}}
</script>"""

    def _fe(v):
        if v is None:
            return "<span class='dim'>-</span>"
        s = f"{v * 100:+.1f}%"
        return f"<span class='pos'>{s}</span>" if v >= min_edge else f"<span class='dim neg'>{s}</span>"

    cards = []
    for (p1, p2), v in sorted_pairs:
        p1e = v["p1_edge"]
        p2e = v["p2_edge"]
        p1_has_edge = p1e is not None and p1e >= min_edge
        p2_has_edge = p2e is not None and p2e >= min_edge
        books_attr = " ".join(v["books"].keys())

        # Book columns
        book_cols_html = ""
        for book, bdata in v["books"].items():
            p1_line = _fmt_odds(bdata["p1_odds"])
            p2_line = _fmt_odds(bdata["p2_odds"])
            book_cols_html += f"""
      <div class="book-col" data-book="{book}">
        <div class="book-label">{book.title()}</div>
        <div class="book-line">{p1_line}</div>
        <div class="book-line">{p2_line}</div>
      </div>"""

        # Our model + edge columns
        model_html = ""
        if has_model and v["p1_our_prob"] is not None:
            model_html = f"""
      <div class="book-col model-col">
        <div class="book-label">Our Model</div>
        <div class="book-line">{_american(v['p1_our_prob'])}</div>
        <div class="book-line">{_american(v['p2_our_prob'])}</div>
      </div>
      <div class="book-col edge-col">
        <div class="book-label">Edge</div>
        <div class="book-line">{_fe(p1e)}</div>
        <div class="book-line">{_fe(p2e)}</div>
      </div>"""

        p1_cls = "mu-player player-edge" if p1_has_edge else "mu-player"
        p2_cls = "mu-player player-edge" if p2_has_edge else "mu-player"

        cards.append(f"""
    <div class="mu-card" data-books="{books_attr}">
      <div class="mu-players">
        <div class="{p1_cls}">{p1}</div>
        <div class="{p2_cls}">{p2}</div>
      </div>
      <div class="mu-books">{book_cols_html}{model_html}
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
  span.neg { color: #4a5568; }
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
    <a class="nav-btn" href="/refresh">↻ Refresh</a>
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
    <a class="nav-btn" href="/matchups/refresh">↻ Refresh</a>
  </nav>

  {% if matchup_round %}
  <div class="round-badge">Round {{ matchup_round }}</div>
  {% endif %}

  {% if matchups_html %}
  {{ matchups_html | safe }}
  {% else %}
  <p class="meta">No matchup data available — check back once the round starts.</p>
  {% endif %}

  <p class="meta" style="margin-top:20px">Auto-refreshes every 5 minutes.</p>
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"Starting DataGolf rankings server on http://0.0.0.0:{port}")
    print(f"Settings loaded from: {SETTINGS_FILE if os.path.exists(SETTINGS_FILE) else 'defaults'}")
    app.run(host="0.0.0.0", port=port, debug=False)
