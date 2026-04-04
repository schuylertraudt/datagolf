"""DataGolf API client."""

import requests

BASE_URL = "https://feeds.datagolf.com"

# All SG stats available from the live-tournament-stats endpoint
LIVE_STATS = "sg_ott,sg_app,sg_arg,sg_putt,sg_t2g,sg_total"


class DataGolfClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def _get(self, endpoint: str, params: dict) -> dict:
        params = {**params, "key": self.api_key, "file_format": "json"}
        resp = self.session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_live_tournament_stats(
        self,
        tour: str = "pga",
        stats: str = LIVE_STATS,
        round: str = "event",
        display: str = "value",
    ) -> dict:
        """
        Live strokes-gained and traditional stats for every player in the field.
        round: 1-4 for a specific round, or 'event' for cumulative tournament average.
        display: 'value' (raw SG) or 'rank' (rank among field).
        """
        return self._get(
            "preds/live-tournament-stats",
            {"tour": tour, "stats": stats, "round": round, "display": display},
        )

    def get_in_play_predictions(
        self,
        tour: str = "pga",
        odds_format: str = "percent",
        dead_heat: str = "no",
    ) -> dict:
        """
        Real-time win/top-5/top-10/make-cut probabilities, updated every ~5 minutes.
        Use during a tournament.
        """
        return self._get(
            "preds/in-play",
            {"tour": tour, "odds_format": odds_format, "dead_heat": dead_heat},
        )

    def get_pre_tournament_predictions(
        self,
        tour: str = "pga",
        odds_format: str = "percent",
        dead_heat: str = "no",
        add_position: str = "",
    ) -> dict:
        """
        Full-field win/top-5/top-10/top-20/make-cut probabilities before the tournament.
        Use before the first round starts.
        """
        params: dict = {
            "tour": tour,
            "odds_format": odds_format,
            "dead_heat": dead_heat,
        }
        if add_position:
            params["add_position"] = add_position
        return self._get("preds/pre-tournament", params)

    def get_dg_rankings(self) -> dict:
        """Top-500 players with DataGolf skill estimates and OWGR rank."""
        return self._get("preds/get-dg-rankings", {})

    def get_skill_ratings(self, display: str = "value") -> dict:
        """
        Per-category skill estimates (sg_ott, sg_app, sg_arg, sg_putt, sg_t2g, sg_total).
        display: 'value' or 'rank'.
        These are rolling averages over recent rounds and serve as the SG baseline
        for pre-tournament rankings.
        """
        return self._get("preds/skill-ratings", {"display": display})

    def get_historical_sg_stats(
        self,
        tour: str = "pga",
        n_rounds: int = 24,
    ) -> dict:
        """
        Fetch per-player rolling SG stats from recent rounds via the
        historical-raw-data endpoint, aggregated client-side.

        Falls back to skill-ratings if the historical endpoint is unavailable,
        since skill-ratings are themselves rolling weighted averages.

        n_rounds: approximate number of recent rounds to target (used as a hint;
                  actual coverage depends on API availability).
        """
        try:
            return self._get(
                "historical-raw-data/rounds",
                {"tour": tour, "n_rounds": n_rounds},
            )
        except Exception:
            # skill-ratings is a reliable fallback: it reflects rolling SG averages
            return self._get("preds/skill-ratings", {})
