#!/usr/bin/env python3
"""Dig into OverviewStats and StatCategory to find stat catalog."""
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
    d = gql(f'{{ __type(name: "{type_name}") {{ fields {{ name type {{ name kind ofType {{ name kind }} }} }} enumValues {{ name }} }} }}')
    t = (d.get("data") or {}).get("__type") or {}
    return t.get("fields") or [], t.get("enumValues") or []

# 1. OverviewStats fields
print("=== OverviewStats fields ===")
fields, _ = type_fields("OverviewStats")
for f in fields:
    print(f"  {f['name']}: {f['type']}")

# 2. StatCategory enum values
print("\n=== StatCategory enum values ===")
_, enum_vals = type_fields("StatCategory")
categories = [v["name"] for v in enum_vals]
print(categories)

# 3. Call statOverview with tourCode=R
print("\n=== statOverview(tourCode: R) — first 800 chars ===")
try:
    d = gql('{ statOverview(tourCode: R) { __typename } }')
    # Now get actual fields
    if fields:
        first_field = fields[0]["name"]
        d2 = gql(f'{{ statOverview(tourCode: R) {{ {first_field} }} }}')
        print(json.dumps(d2, indent=2)[:800])
except Exception as e:
    print(f"error: {e}")

# 4. Try statLeaders with first category
if categories:
    cat = categories[0]
    print(f"\n=== statLeaders(tourCode: R, category: {cat}) — first 800 chars ===")
    try:
        d = gql(f'{{ statLeaders(tourCode: R, category: {cat}) {{ title stats {{ statId statTitle }} }} }}')
        print(json.dumps(d, indent=2)[:800])
    except Exception as e:
        print(f"error: {e}")
