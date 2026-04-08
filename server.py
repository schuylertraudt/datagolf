#!/usr/bin/env python3
"""
DataGolf rankings web server.

Setup:
  1. Run the CLI once to configure and save settings:
       python main.py --pre-tournament --save-settings
  2. Start the server:
       python server.py
  3. Open http://<your-server-ip>:8080 in a browser.

The page auto-refreshes every 5 minutes. API results are cached server-side
so multiple browser loads don't hammer the DataGolf API.
"""

import json
import os
import time
import traceback
from datetime import datetime
from threading import Lock

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string

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
    # If weight not explicitly set, use auto-calculated value
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


def get_rankings(force: bool = False) -> dict:
    """Return cached rankings, refreshing if stale or forced."""
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


def _parse_course_history_srv(raw: dict):
    """Parse historical-raw-data/event response into a rank DataFrame."""
    import pandas as pd

    records = raw.get("data") or raw.get("players") or []
    rows = []
    for p in records:
        name = (p.get("player_name") or p.get("player") or p.get("name") or "").strip()
        if not name:
            continue
        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        sg = _f(p.get("sg_total"))
        if sg is None:
            sg = _f(p.get("sg_t2g"))
        if sg is None:
            avg = _f(p.get("scoring_avg") or p.get("avg_score"))
            if avg is not None:
                sg = -avg
        rows.append({"player_name": name.lower(), "sg_val": sg})

    if not rows:
        return pd.DataFrame(columns=["player_name", "course_hist_rank"])

    df = pd.DataFrame(rows)
    df = df.groupby("player_name", as_index=False)["sg_val"].mean()
    df = df.dropna(subset=["sg_val"])
    if df.empty:
        return pd.DataFrame(columns=["player_name", "course_hist_rank"])

    df = df.sort_values("sg_val", ascending=False).reset_index(drop=True)
    df["course_hist_rank"] = df.index + 1
    return df[["player_name", "course_hist_rank"]]


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
        pga = PGATourStats()
        for s in stat_configs:
            try:
                stat_df = pga.get_combined_stat(s["id"], label=s.get("label"), current_weight=season_weight)
                col = f"{s.get('label', s['id'])} Rk"
                lookup = stat_df.set_index("player_name")["combined_rank"]
                df[col] = df["player_name"].str.strip().str.lower().map(lookup)
                df[col] = df[col].apply(lambda x: int(x) if not __import__("pandas").isna(x) else None)
                extra_cols.append(col)
            except Exception:
                pass

    # Course history
    try:
        import pandas as _pd
        _event_id = str(
            (predictions_raw or {}).get("event_id")
            or short_raw.get("event_id")
            or ""
        ).strip()
        ch_raw = client.get_course_history(tour=tour, event_id=_event_id)
        ch_df = _parse_course_history_srv(ch_raw)
        if not ch_df.empty:
            lookup = ch_df.set_index("player_name")["course_hist_rank"]
            df["Course Hist Rk"] = df["player_name"].str.strip().str.lower().map(lookup)
            df["Course Hist Rk"] = df["Course Hist Rk"].apply(
                lambda x: int(x) if not _pd.isna(x) else None
            )
            extra_cols.append("Course Hist Rk")
    except Exception:
        pass

    # Matchups
    matchups_html = ""
    try:
        mu_raw = client.get_matchups(tour=tour)
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
        "rankings_html": _rankings_to_html(df, extra_cols),
        "extra_col_headers": extra_cols,
        "matchups_html": matchups_html,
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


def _rankings_to_html(df, extra_cols=None) -> str:
    extra_cols = extra_cols or []
    rows = []
    for _, r in df.head(50).iterrows():
        extra_cells = ""
        for col in extra_cols:
            v = r.get(col)
            cell_val = "<span class='dim'>-</span>" if v is None else str(int(v))
            extra_cells += f"<td>{cell_val}</td>"
        rows.append(f"""
        <tr>
          <td class="dim">{int(r['rank'])}</td>
          <td class="name">{r.get('player_name') or ''}</td>
          <td>{_pct(r.get('win_prob'))}</td>
          <td>{_american(r.get('win_prob'))}</td>
          <td>{_american(r.get('model_win_prob'))}</td>
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

    edge_mask = (
        (mu_df["p1_edge"].fillna(0) >= min_edge) |
        (mu_df["p2_edge"].fillna(0) >= min_edge)
    )
    show_df = mu_df[edge_mask].copy() if has_model else mu_df.copy()

    if show_df.empty:
        return "<p class='dim'>No matchups with edge ≥ 3%. Try refreshing after round starts.</p>"

    show_df = (
        show_df.sort_values("max_edge", ascending=False)
        .drop_duplicates(subset=["p1_name", "p2_name"], keep="first")
    )

    rows = []
    for _, r in show_df.iterrows():
        p1e = r.get("p1_edge")
        p2e = r.get("p2_edge")

        p1_cls = " pos bold" if (p1e is not None and p1e >= min_edge) else ""
        p2_cls = " pos bold" if (p2e is not None and p2e >= min_edge) else ""

        if has_model:
            if p1e is not None and p2e is not None and p1e >= min_edge and p1e >= p2e:
                bet = f"<span class='pos bold'>→ {r['p1_name']}</span>"
            elif p2e is not None and p2e >= min_edge:
                bet = f"<span class='pos bold'>→ {r['p2_name']}</span>"
            else:
                bet = "<span class='dim'>—</span>"
        else:
            bet = ""

        def _fe(v):
            if v is None:
                return "<span class='dim'>-</span>"
            s = f"{v*100:+.1f}%"
            return f"<span class='pos'>{s}</span>" if v >= min_edge else f"<span class='dim'>{s}</span>"

        row = f"""
        <tr>
          <td class="name{p1_cls}">{r['p1_name']}</td>
          <td class="name{p2_cls}">{r['p2_name']}</td>
          <td>{r.get('book') or ''}</td>
          <td>{_fmt_odds(r.get('p1_book_odds'))}</td>
          <td>{_fmt_odds(r.get('p2_book_odds'))}</td>"""
        if has_model:
            row += f"""
          <td>{_american(r.get('p1_our_prob'))}</td>
          <td>{_american(r.get('p2_our_prob'))}</td>
          <td>{_fe(p1e)}</td>
          <td>{_fe(p2e)}</td>
          <td>{bet}</td>"""
        row += "</tr>"
        rows.append(row)

    extra_headers = """
      <th>Our A</th><th>Our B</th><th>Edge A</th><th>Edge B</th><th>Bet</th>""" if has_model else ""

    return f"""
    <table>
      <thead><tr>
        <th>Player A</th><th>Player B</th><th>Book</th>
        <th>Line A</th><th>Line B</th>{extra_headers}
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="300">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ event_name }}</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0f1117; color: #e2e8f0; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; padding: 24px; }
    h1 { font-size: 20px; color: #63b3ed; margin-bottom: 4px; }
    h2 { font-size: 15px; color: #63b3ed; margin: 32px 0 12px; }
    .meta { color: #718096; font-size: 12px; margin-bottom: 6px; }
    .weights { color: #718096; font-size: 11px; margin-bottom: 20px; }
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
    .refresh-btn { display: inline-block; margin-top: 8px; padding: 6px 14px; background: #2d3748; color: #a0aec0; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; text-decoration: none; }
    .refresh-btn:hover { background: #4a5568; }
  </style>
</head>
<body>
  {% if error %}
  <div class="error">⚠ Error fetching data: {{ error }} — showing last cached result.</div>
  {% endif %}

  <h1>{{ event_name }}</h1>
  <div class="meta">Updated: {{ cached_at }} &nbsp;·&nbsp; <a class="refresh-btn" href="/refresh">↻ Refresh now</a></div>
  <div class="weights">{{ weights_str }}</div>

  <table>
    <thead><tr>
      <th>#</th><th>Player</th>
      <th>Win%</th><th>DG Odds</th><th>My Odds</th>
      <th>SG:OTT</th><th>SG:APP</th><th>SG:ARG</th><th>SG:PUT</th><th>SG:TOT</th>
      <th>Score</th>
      {% for col in extra_col_headers %}<th>{{ col }}</th>{% endfor %}
    </tr></thead>
    <tbody>{{ rankings_html | safe }}</tbody>
  </table>

  {% if matchups_html %}
  <h2>Matchup Edges — Round</h2>
  {{ matchups_html | safe }}
  {% endif %}

  <p class="meta" style="margin-top:24px">Page auto-refreshes every 5 minutes.</p>
</body>
</html>"""


@app.route("/")
def index():
    data = get_rankings()
    settings = load_settings()
    w = settings.get("weights", {})
    weights_str = "  ".join(f"{k}={v:.0%}" for k, v in w.items())
    return render_template_string(
        HTML_TEMPLATE,
        event_name=data.get("event_name", "DataGolf Rankings"),
        cached_at=data.get("cached_at", "—"),
        error=data.get("error"),
        rankings_html=data.get("rankings_html", ""),
        matchups_html=data.get("matchups_html", ""),
        weights_str=weights_str,
        extra_col_headers=data.get("extra_col_headers", []),
    )


@app.route("/refresh")
def refresh():
    get_rankings(force=True)
    from flask import redirect
    return redirect("/")


@app.route("/api/data")
def api_data():
    data = get_rankings()
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
