#!/usr/bin/env python3
"""Generate random API keys for the courts and print the CLIENT_API_KEYS secret value.

  python scripts/issue_client_keys.py court-chennai court-madurai court-coimbatore

Paste the JSON line into the Space secret CLIENT_API_KEYS, and give each court ONLY its own key
(FL_API_KEY on that court's machine). To revoke a court, delete its entry and restart the Space.
"""
import json
import secrets
import sys

names = sys.argv[1:] or ["court-demo-1", "court-demo-2", "court-demo-3"]
keys = {n: secrets.token_urlsafe(32) for n in names}
print("CLIENT_API_KEYS=" + json.dumps(keys))
print("ADMIN_API_KEY=" + secrets.token_urlsafe(32))
