import json
import urllib.request
from urllib.parse import urlencode

BASE = "http://127.0.0.1:8080"


def post(path, data=None, form=None):
    headers = {}
    body = None
    if form is not None:
        body = urlencode(form).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


key = "Temple Visit Narrative (\u0b95\u0bcb\u0baf\u0bbf\u0bb2\u0bcd \u0b9a\u0bbe\u0b9f\u0bcd\u0b9a\u0bbf\u0baf\u0bae\u0bcd)"
load = post("/api/load-sample", form={"sample_key": key})
print("load_ok", load.get("ok"))
tr = post("/api/transcribe")
print("transcribe_ok", tr.get("ok"), "elapsed", tr.get("elapsed_sec"), "draft_len", len(tr.get("draft") or ""))
