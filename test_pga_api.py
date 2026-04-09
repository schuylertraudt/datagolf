#!/usr/bin/env python3
"""Find stat catalog via statOverview.stats and StatLeaderCategory fields."""
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

def type_fields(type_name):
    d = gql(f'{{ __type(name: "{type_name}") {{ fields {{ name type {{ name kind ofType {{ name kind }} }} }} }} }}')
    return [(f["name"], f["type"]) for f in ((d.get("data") or {}).get("__type") or {}).get("fields") or []]

# 1. StatLeaderCategory actual fields
print("=== StatLeaderCategory fields ===")
for name, typ in type_fields("StatLeaderCategory"):
    print(f"  {name}: {typ}")

# 2. statOverview with stats + categories
print("\n=== statOverview(tourCode: R) with stats and categories — first 1000 chars ===")
try:
    d = gql('{ statOverview(tourCode: R) { stats { __typename } categories { __typename } } }')
    print(json.dumps(d, indent=2)[:500])
    # Get element types
    stats_list = ((d.get("data") or {}).get("statOverview") or {}).get("stats") or []
    cats_list  = ((d.get("data") or {}).get("statOverview") or {}).get("categories") or []
    print(f"  stats count: {len(stats_list)}, first typename: {stats_list[0] if stats_list else 'none'}")
    print(f"  categories count: {len(cats_list)}, first typename: {cats_list[0] if cats_list else 'none'}")
except Exception as e:
    print(f"error: {e}")

# 3. Try to get stat IDs from statOverview.stats
print("\n=== statOverview stats with id/title fields ===")
for field_combo in ["statId statTitle", "id title", "statId name", "id name"]:
    try:
        d = gql(f'{{ statOverview(tourCode: R) {{ stats {{ {field_combo} }} }} }}')
        if not d.get("errors"):
            stats = ((d.get("data") or {}).get("statOverview") or {}).get("stats") or []
            print(f"  Fields '{field_combo}' worked! {len(stats)} stats, first 3: {stats[:3]}")
            break
        else:
            print(f"  Fields '{field_combo}': {d['errors'][0]['message'][:80]}")
    except Exception as e:
        print(f"  Fields '{field_combo}': error {e}")

# 4. statLeaders with correct fields
print("\n=== StatLeaderCategory fields (for statLeaders) ===")
for name, typ in type_fields("StatLeaderCategory"):
    print(f"  {name}: {typ}")
