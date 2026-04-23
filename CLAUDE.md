# DataGolf Rankings — Claude Code Guide

This is a Flask-based golf betting tool that pulls live DataGolf API data,
applies a proprietary matchup model, and serves a web UI with rankings and
matchup edge analysis.

---

## File Map

| File | Purpose |
|---|---|
| `server.py` | Flask server, all HTML rendering, fetch/cache logic |
| `datagolf/client.py` | DataGolf REST API client |
| `datagolf/matchups.py` | Matchup edge math (probability models) |
| `datagolf/ranking.py` | Pre-tournament ranking model (composite score) |
| `datagolf/pgatour.py` | PGA Tour GraphQL API for supplemental stats |
| `settings.json` | User weight config (SG component weights, history params) |
| `weekly_stats.json` | Optional PGA Tour stat overlays |
| `.env` | `DATAGOLF_API_KEY` — never commit this |

The server runs as a systemd service (`datagolf.service`). Restart after any
change with `systemctl restart datagolf`.

---

## Architecture: Matchup Pipeline

**Fetch order matters.** Matchups are fetched FIRST so `matchup_round` is
known before the live stats fetch. The live round is derived as
`int(matchup_round) - 1` because the DataGolf live stats endpoint returns
`stat_round = "event"` (a string, not a number) which is useless for this.
Do not swap the fetch order.

```
client.get_matchups()           → sets matchup_round
client.get_live_tournament_stats() → live sg_putt, sg_t2g
      ↓ putting regression applied to df["matchup_sg"]
parse_matchups(mu_raw, model_df=df) → mu_df with edge signals
_matchups_to_html(mu_df)        → HTML for /matchups page
```

---

## The Three Edge Signals

All three signals mean **positive = value at the soft book** (the book is
underpricing this player relative to the benchmark). They are displayed as
columns on each matchup card.

| Column | Formula | Green threshold |
|---|---|---|
| **vs Our Odds** | our model prob − soft-book fair prob | ≥ 5pp |
| **vs Pinnacle** | Pinnacle fair prob − soft-book fair prob | ≥ 5pp |
| **vs Market** | avg(other books' fair prob) − soft-book fair prob | ≥ 3pp |

When "All Books" is selected in the dropdown, each signal shows the **best
value across all displayed books**. When a specific book is selected, the
signal recalculates for that book only. This is driven entirely by JS reading
`data-edges` JSON embedded on each card — do not remove that attribute.

---

## Pinnacle: Reference Book, Not Display Book

Pinnacle lines come back in the DataGolf matchups response under the key
`"pinnacle"`. Pinnacle is intentionally **excluded from `_BOOK_DISPLAY`** —
it is used only as the sharp-market reference for the "vs Pinnacle" edge calc.

**Do not add `"pinnacle"` to `_BOOK_DISPLAY`** — it will break the edge
logic because "vs Pinnacle" is computed as `pinnacle_fair - soft_book_fair`,
which assumes Pinnacle is not itself one of the "soft books."

```python
# server.py
_BOOK_DISPLAY = {
    "draftkings": "DraftKings",
    "fanduel":    "FanDuel",
    "unibet":     "BetRivers",
    "caesars":    "Caesars",
    "betmgm":     "BetMGM",
}
```

In `matchups.py`, `"pinnacle"` and `"datagolf"` are excluded from the
per-book loop: `books_found = [k for k in odds_section.keys() if k not in ("datagolf", "pinnacle")]`

---

## Critical: NaN Handling

Pandas returns float `NaN` (not Python `None`) for missing DataFrame values.
`NaN is not None` is `True` in Python, so naive `is not None` guards fail.

**The `_safe()` helper** (defined inside `_matchups_to_html`) converts
pandas NaN/NA to Python `None`. It must be used when extracting any value
from a DataFrame row:

```python
def _safe(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v

# Correct:
p1_pin = _safe(r.get("p1_pin_prob"))
has_pin = p1_pin is not None   # safe — NaN already converted

# Wrong:
has_pin = r.get("p1_pin_prob") is not None  # True even when NaN!
```

**Why this matters for the JS filter:** `json.dumps` serializes Python `NaN`
as the bare token `NaN` (not `null`), which is invalid JSON. If NaN slips
into `data-edges`, `JSON.parse` throws in the browser, the column-hiding JS
aborts, and all books stay visible even when one is selected. Always sanitize
via `_safe()` before building `book_edges`, and run `_clean()` on the dict
before `json.dumps`.

---

## Putting Regression Signal

Applies only when live tournament stats are available (Rounds 2+).
Boosts/penalizes `matchup_sg` (the value used for H2H probability) based on
the gap between a player's historical SG:P and their current-week live SG:P.

- **Upward boost**: historically good putter running cold this week, gated by
  positive live SG:T2G. If a player isn't creating birdie looks, the putter
  regression doesn't help them.
- **Downward penalty**: player is hot on the putter this week — no T2G gate,
  because unsustainable putting hurts regardless of T2G skill.

`_PUTT_REGRESSION_FACTOR = 0.10` — max ~±0.2 stroke adjustment to `matchup_sg`.

The regression table (shown at bottom of `/matchups`) lists the top targets
with the largest upward boost signals. Players shown there **are already
receiving the boost** in the edge calculations.

---

## Matchup Probability Model

**Do not use Bradley-Terry or tournament win probabilities for round matchups.**
Tournament win probs are cumulative multi-round signals; they perform poorly
for single-round H2H.

The model uses a **normal distribution** over expected round score
differential:

```
P(A beats B) = Φ((sg_A − sg_B) / (√2 × 2.9))
```

`ROUND_STD_DEV = 2.9` is the empirical PGA Tour single-round score std dev.
Implemented in `datagolf/matchups.py::round_matchup_prob()` using
`math.erfc` (no scipy needed).

The input `sg` value used is `matchup_sg` (not raw `sg_total`). `matchup_sg`
is a blend:

```
matchup_sg = 0.92 × sg_total + 0.08 × user_weighted_sg_components
                              + putting_regression_adjustment
```

The 8% user-weight influence comes from `settings.json` weights on
`sg_ott/sg_app/sg_arg/sg_putt`. Changing `_MATCHUP_INFLUENCE` in `server.py`
adjusts this blend.

---

## Player Name Normalization

The DataGolf live stats endpoint returns names as `"Last, First"` (e.g.,
`"Burns, Sam"`). The rankings model uses `"First Last"` format. Merging these
requires `_normalize_name()` from `datagolf/matchups.py`, which converts
both formats to lowercase `"first last"`.

Always normalize both sides before any name-based merge or lookup. A direct
`df.merge(on="player_name")` without normalization will silently produce an
empty result — no error, just all NaN.

---

## Debug Routes

| Route | Purpose |
|---|---|
| `/debug/regression` | Diagnose putting regression pipeline (live round, player count, name format, regression HTML length) |
| `/debug/books` | Show all book keys the DataGolf matchups API is returning, including whether `"pinnacle"` is present and what its lines look like |

Use these before assuming a signal is broken — many issues trace back to the
API returning unexpected keys or name formats.

---

## What NOT to Touch

- **`_safe()` calls in Phase 1 of `_matchups_to_html`** — removing them causes NaN to propagate into JSON and break the book filter JS
- **Fetch order in `_fetch()`** — matchups must be fetched before live stats
- **`"pinnacle"` exclusion in `_BOOK_DISPLAY` and `books_found`** — see Pinnacle section above
- **`round_matchup_prob()` in `matchups.py`** — this is the core model; do not replace with Bradley-Terry
- **`data-edges` attribute on `.mu-card`** — the entire book filter + dynamic edge recalc depends on it
- **`data-edge-type` and `data-player` attributes on edge columns** — JS reads these to know which edge value to update
