#!/usr/bin/env python3
"""Get StatLeaderSubCategory.stats element type and build full catalog."""
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

# 1. Get stats element type inside StatLeaderSubCategory
print("=== statLeaders subCategories stats __typename ===")
d = gql("""
{
  statLeaders(tourCode: R, category: STROKES_GAINED) {
    subCategories { stats { __typename } }
  }
}
""")
subs = ((d.get("data") or {}).get("statLeaders") or {}).get("subCategories") or []
stat_items = (subs[0].get("stats") or []) if subs else []
print(f"stats count: {len(stat_items)}, first: {stat_items[0] if stat_items else 'none'}")

stat_typename = stat_items[0].get("__typename") if stat_items else None
if stat_typename:
    print(f"\n=== {stat_typename} fields ===")
    d2 = gql(f'{{ __type(name: "{stat_typename}") {{ fields {{ name type {{ name kind ofType {{ name kind }} }} }} }} }}')
    fields = ((d2.get("data") or {}).get("__type") or {}).get("fields") or []
    for f in fields:
        print(f"  {f['name']}: {f['type']}")

    # 2. Fetch with all field names
    all_field_names = " ".join(f[0] for f in [(f["name"], f["type"]) for f in fields])
    print(f"\n=== Full stat item (all fields) ===")
    try:
        d3 = gql(f"""
        {{
          statLeaders(tourCode: R, category: STROKES_GAINED) {{
            subCategories {{
              subCategoryName
              stats {{ {all_field_names} }}
            }}
          }}
        }}
        """)
        subs2 = ((d3.get("data") or {}).get("statLeaders") or {}).get("subCategories") or []
        for sub in subs2:
            print(f"\n  Subcategory: {sub.get('subCategoryName')}")
            for s in (sub.get("stats") or []):
                print(f"    {s}")
    except Exception as e:
        print(f"error: {e}")
        if d3.get("errors"):
            print(d3["errors"])
