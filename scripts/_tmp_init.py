import json
import urllib.request

req = urllib.request.Request(
    "http://127.0.0.1:8080/api/init",
    data=json.dumps({"mode": "simulation"}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=180) as resp:
    print(resp.read().decode())
