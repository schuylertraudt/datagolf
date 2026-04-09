#!/usr/bin/env python3
"""Introspect promising stat catalog fields."""
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

def introspect_field(field_name):
    """Get args and return type for a Query field."""
    d = gql("""{ __type(name: "Query") { fields { name args { name type { name kind ofType { name kind } } } returnType: type { name kind ofType { name kind ofType { name kind } } } } } }""")
    fields = (d.get("data") or {}).get("__type", {}).get("fields") or []
    for f in fields:
        if f["name"] == field_name:
            return f
    return None

def introspect_type(type_name):
    """Get fields of a named type."""
    d = gql(f'{{ __type(name: "{type_name}") {{ fields {{ name type {{ name kind ofType {{ name kind }} }} }} }} }}')
    return (d.get("data") or {}).get("__type", {})

for candidate in ["statOverview", "allTimeRecordCategories", "statLeaders"]:
    print(f"\n=== {candidate} ===")
    info = introspect_field(candidate)
    if info:
        print(f"  args: {[(a['name'], a['type']) for a in info.get('args', [])]}")
        print(f"  returnType: {info.get('returnType')}")
    else:
        print("  not found")

# Try calling statOverview with no args to see what happens
print("\n=== statOverview call (no args) ===")
try:
    d = gql("{ statOverview { __typename } }")
    print(json.dumps(d, indent=2)[:600])
except Exception as e:
    print(f"error: {e}")

# Try allTimeRecordCategories
print("\n=== allTimeRecordCategories call ===")
try:
    d = gql("{ allTimeRecordCategories { __typename } }")
    print(json.dumps(d, indent=2)[:600])
except Exception as e:
    print(f"error: {e}")
