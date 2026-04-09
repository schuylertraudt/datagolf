#!/usr/bin/env python3
"""Introspect OverviewStat and StatCategoryConfig to find stat ID/title fields."""
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

for t in ["OverviewStat", "StatCategoryConfig"]:
    print(f"\n=== {t} fields ===")
    for name, typ in type_fields(t):
        print(f"  {name}: {typ}")

# Also check subCategories element type inside StatLeaderCategory
print("\n=== StatLeaderCategory.subCategories element type ===")
d = gql('{ __type(name: "StatLeaderCategory") { fields { name type { name kind ofType { name kind ofType { name kind } } } } } }')
fields = ((d.get("data") or {}).get("__type") or {}).get("fields") or []
for f in fields:
    if f["name"] in ("subCategories", "otherCategories"):
        print(f"  {f['name']}: {f['type']}")

# Try calling statLeaders with subCategories
print("\n=== statLeaders subCategories introspect ===")
sub_type = None
for f in fields:
    if f["name"] == "subCategories":
        t = f["type"]
        # Drill down through NON_NULL/LIST wrappers
        while t and not t.get("name"):
            t = t.get("ofType")
        sub_type = t.get("name") if t else None
        break

if sub_type:
    print(f"  subCategories element type: {sub_type}")
    for name, typ in type_fields(sub_type):
        print(f"    {name}: {typ}")
