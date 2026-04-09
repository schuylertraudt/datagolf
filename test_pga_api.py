#!/usr/bin/env python3
"""Quick diagnostic for PGA Tour API — run and share output."""
import json
import requests

_API_URL = "https://orchestrator.pgatour.com/graphql"
_API_KEY  = "da2-gsrx5bibzbb4njvhl7t37wqyl4"

_STAT_QUERY = """
query StatDetails($tourCode: TourCode!, $statId: String!, $year: Int) {
  statDetails(tourCode: $tourCode, statId: $statId, year: $year) {
    statId
    statTitle
    rows {
      ... on StatDetailsPlayer {
        playerName
        rank
        stats {
          statValue
          statId
        }
      }
    }
  }
}
"""

session = requests.Session()
session.headers.update({
    "x-api-key": _API_KEY,
    "x-amz-user-agent": "aws-amplify/3.8.21",
    "Content-Type": "application/json",
})

def fetch(stat_id, year=None):
    variables = {"tourCode": "R", "statId": stat_id}
    if year is not None:
        variables["year"] = year
    resp = session.post(_API_URL, json={"query": _STAT_QUERY, "variables": variables}, timeout=20)
    resp.raise_for_status()
    return resp.json()

stat_id = "02335"  # Par 5 Scoring Average
print("=== year=2026 ===")
d = fetch(stat_id, year=2026)
details = (d.get("data") or {}).get("statDetails") or {}
rows = details.get("rows") or []
print(f"statTitle: {details.get('statTitle')}")
print(f"total rows: {len(rows)}")
print(f"first 2 rows raw: {json.dumps(rows[:2], indent=2)}")

print()
print("=== no year (default) ===")
d2 = fetch(stat_id)
details2 = (d2.get("data") or {}).get("statDetails") or {}
rows2 = details2.get("rows") or []
print(f"statTitle: {details2.get('statTitle')}")
print(f"total rows: {len(rows2)}")
print(f"first 2 rows raw: {json.dumps(rows2[:2], indent=2)}")

print()
print("=== errors (year=2026) ===")
print(d.get("errors"))
