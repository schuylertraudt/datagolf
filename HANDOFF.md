# Handoff Notes

## What was built

A Python CLI (`main.py`) that ranks golfers for weekly betting contests using the
DataGolf API. It combines live strokes-gained stats with in-play or pre-tournament
win probabilities into a single composite score, and can surface value bets against
market odds.

## Architecture

```
datagolf/
  client.py   — DataGolf API wrapper (live stats, predictions, skill ratings)
  ranking.py  — RankingModel: weighted z-score composite + edge calculation
  __init__.py — package exports
main.py       — Rich CLI (argument parsing, display, orchestration)
```

## Key design decisions

### Two ranking modes

**Live / in-play** (`python main.py`)
- Fetches `live-tournament-stats` for current-round or cumulative SG
- Fetches `in-play` predictions for win probabilities
- Uses `RankingModel.build()`

**Pre-tournament** (`python main.py --pre-tournament`)
- Uses `get_historical_sg_stats()` → tries `historical-raw-data/rounds`, falls back
  to `skill-ratings` (DataGolf rolling SG averages weighted by recency)
- Fetches `pre-tournament` predictions for win probabilities
- Uses `RankingModel.build_pre_tournament()`
- No position/thru columns since no live round is in progress

### Composite score

Weighted sum of z-scored inputs. Default weights:

| Metric    | Weight |
|-----------|--------|
| sg_total  | 30%    |
| win_prob  | 30%    |
| sg_app    | 15%    |
| sg_putt   | 12%    |
| sg_ott    | 8%     |
| sg_arg    | 5%     |

All weights are configurable via CLI flags and are normalized to sum to 1.

### Value bets

Pass `--odds-file odds.csv` with American, decimal, or implied-probability odds.
The `edge` column = DataGolf win_prob − market implied prob. Positive = underpriced.

## Setup

```bash
cp .env.example .env   # add DATAGOLF_API_KEY
pip install -r requirements.txt
python main.py --help
```

## What's next / known gaps

- The `historical-raw-data/rounds` endpoint may require specific `event_id`/`year`
  parameters depending on DataGolf API tier. If it errors, the code falls back to
  `skill-ratings` automatically.
- No caching: each run makes fresh API calls. Adding a `--cache` flag with a
  short TTL (e.g. 5 minutes) would speed up repeated runs during a round.
- Could add `--top5` / `--top10` probability columns to the display table.
- Consider adding a `--csv` flag to dump the ranked DataFrame to a file.
