#!/usr/bin/env python3
"""Build full stat catalog via statLeaders across all StatCategory enum values."""
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

CATEGORIES = [
    'APPROACH_GREEN', 'AROUND_GREEN', 'MONEY_FINISHES', 'OFF_TEE',
    'POINTS_RANKINGS', 'PUTTING', 'SCORING', 'STREAKS',
    'STROKES_GAINED', 'FACTS_AND_FIGURES', 'PACE_OF_PLAY'
]

# 1. Find subCategories element type
print("=== Introspect subCategories element type ===")
d = gql("""
{
  statLeaders(tourCode: R, category: STROKES_GAINED) {
    categoryHeader
    subCategories { __typename }
  }
}
""")
result = ((d.get("data") or {}).get("statLeaders") or {})
subs = result.get("subCategories") or []
print(f"categoryHeader: {result.get('categoryHeader')}")
print(f"subCategories count: {len(subs)}, first typename: {subs[0] if subs else 'none'}")

# 2. Get element type fields
sub_typename = subs[0].get("__typename") if subs else None
if sub_typename:
    print(f"\n=== {sub_typename} fields ===")
    d2 = gql(f'{{ __type(name: "{sub_typename}") {{ fields {{ name type {{ name kind ofType {{ name kind }} }} }} }} }}')
    fields = ((d2.get("data") or {}).get("__type") or {}).get("fields") or []
    for f in fields:
        print(f"  {f['name']}: {f['type']}")

    # 3. Try fetching with likely stat fields
    print(f"\n=== statLeaders subCategories with stat fields ===")
    for field_combo in ["statId statTitle", "statId title", "statId name", "statId displayTitle"]:
        try:
            d3 = gql(f"""
            {{
              statLeaders(tourCode: R, category: STROKES_GAINED) {{
                categoryHeader
                subCategories {{ {field_combo} }}
              }}
            }}
            """)
            if not d3.get("errors"):
                subs2 = ((d3.get("data") or {}).get("statLeaders") or {}).get("subCategories") or []
                print(f"  Fields '{field_combo}' worked! {len(subs2)} subcategories, first 3: {subs2[:3]}")
                break
            else:
                print(f"  '{field_combo}': {d3['errors'][0]['message'][:80]}")
        except Exception as e:
            print(f"  '{field_combo}': error {e}")
