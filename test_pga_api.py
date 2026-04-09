#!/usr/bin/env python3
"""Introspect PGA Tour GraphQL API to find stat catalog query — run and share output."""
import json
import requests

_API_URL = "https://orchestrator.pgatour.com/graphql"
_API_KEY  = "da2-gsrx5bibzbb4njvhl7t37wqyl4"

session = requests.Session()
session.headers.update({
    "x-api-key": _API_KEY,
    "x-amz-user-agent": "aws-amplify/3.8.21",
    "Content-Type": "application/json",
})

def gql(query, variables=None):
    resp = session.post(_API_URL, json={"query": query, "variables": variables or {}}, timeout=20)
    resp.raise_for_status()
    return resp.json()

# 1. All Query-level fields
print("=== All Query fields ===")
d = gql("{ __type(name: \"Query\") { fields { name } } }")
fields = [f["name"] for f in (d.get("data") or {}).get("__type", {}).get("fields") or []]
for f in sorted(fields):
    print(" ", f)

# 2. Look for any field that sounds like a stat catalog
stat_fields = [f for f in fields if "stat" in f.lower() or "categor" in f.lower()]
print(f"\n=== Stat-related fields: {stat_fields} ===")

# 3. Try statLeaderboards or similar if present
for candidate in ["statLeaderboard", "statLeaderboards", "stats", "statList", "playerStats"]:
    if candidate in fields:
        print(f"\n=== Trying {candidate} ===")
        try:
            d2 = gql(f"{{ {candidate} {{ __typename }} }}")
            print(json.dumps(d2, indent=2)[:500])
        except Exception as e:
            print(f"  error: {e}")
