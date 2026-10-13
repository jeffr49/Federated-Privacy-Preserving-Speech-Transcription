# Federated Court ASR — deployment guide

Privacy-preserving, dialect-adaptive Tamil speech transcription. Court devices transcribe and learn
**locally**; only clipped LoRA-adapter deltas (a few MB) go to a hosted aggregation server; a CI gate
promotes a new adapter only if it doesn't regress on a held-out set.

```
 court device (GPU, local)                 Hugging Face                              GitHub
┌──────────────────────────┐     ┌────────────────────────────────┐      ┌──────────────────────────┐
│ Streamlit app            │     │ Space: aggregation server      │      │ Actions: promote-adapter │
│  mic/audio -> Whisper    │     │  (FastAPI, CPU, numpy-only)    │      │  1. eval candidate (CPU) │
│  + LoRA -> draft         │     │                                │      │  2. eval current (cached)│
│  clerk corrects          │     │  POST /submit_delta  <─────────┼──────┤  3. WER not worse?       │
│  local update            │ ──► │  validate + RE-CLIP + weighted │      │     yes -> promote main  │
│  end session:            │ Δ   │  average -> candidate branch   │ ───► │            + tag vN      │
│   clipped delta ─────────┼──►  │  fires repository_dispatch     │      │     no  -> record reject │
│   audio DISCARDED        │     │                                │      └──────────────────────────┘
│  GET /adapter  <─────────┼──── │ Hub repos (private):           │
└──────────────────────────┘     │  adapters   main = promoted    │
   audio never leaves here       │             candidate branch   │
                                 │  submissions pending deltas    │
                                 │  eval        held-out set      │
                                 └────────────────────────────────┘
```

## Repository layout

| Path | Purpose |
|---|---|
| `frontend/` | Premium Next.js App Router UI: live Whisper-LoRA transcription, clerk feedback loop, animated visual FL flow |
| `backend/api.py` | FastAPI backend service exposing REST endpoints for the Next.js frontend |
| `fl_client_sync.py` | HTTP client for court devices: pulls promoted adapter, submits clipped deltas to aggregation server |
| `demo_fl_rl_live.py` | CLI live demonstration: clerk correction SFT loop + simulated FedAvg rounds |
| `server/` | Aggregation service (FastAPI, Dockerfile + HF Space deployment card) |
| `ci/` | `evaluate_and_promote.py` (the gate), `make_eval_set.py`, test suite |
| `scripts/` | `bootstrap_hub.py`, `bootstrap_local.py`, `issue_client_keys.py`, `promote_local.py` |
| `training_scripts/` | Base fine-tuning and scratch federated training scripts (`train_whisper_lora_local.py`, `fl_train_from_scratch.py`) |
| `runs/whisper-small_lora_r32/final_adapter/` | Promoted fine-tuned weights for deployment |
| `.github/workflows/promote-adapter.yml` | The CI promotion pipeline |
| `.env.example` | Complete environment variable documentation |

## What has and hasn't been tested

| Tested (automated, passing) | NOT tested — verify first (needs GPU / accounts / network) |
|---|---|
| Server: auth, schema + NaN validation, stale-base rejection, server-side clipping vs a malicious client, weighted-average maths, raw-delta deletion (`server/test_server.py`, 9 tests) | `HubStore` against the real Hugging Face Hub |
| Client↔server over real HTTP: pull, 3 pushes, aggregation, stale rejection | The GitHub Actions run and `repository_dispatch` trigger |
| CI decision logic, WER, Tamil-safe text normalisation (`ci/test_ci.py`, 4 tests) | Whisper loading/eval in `evaluate_and_promote.py`, `make_eval_set.py` on the real manifest |
| Local-store bootstrap/promote helpers | Streamlit app + real LoRA key names end-to-end (the app preflights key names and says so clearly if they mismatch) |

Run the tests: `cd server && FL_SERVER_AUTOSTART=0 pytest -q` and `cd ci && pytest -q`.

## One-time setup

### 0. Accounts and tokens
- Hugging Face account. Create **two** fine-grained tokens (least privilege):
  - `server` token: write on the adapter repo and the submissions repo
  - `ci` token: write on the adapter repo, read on the eval repo
- GitHub repo (push this whole project to it). Create a fine-grained PAT for the *server* to trigger CI:
  this repo only, permission **Contents: Read and write** (that is what the `repository_dispatch` API requires).

### 1. Create the Hub repos and publish v0
```bash
pip install huggingface_hub safetensors numpy
export HF_TOKEN=hf_...          # write token
python scripts/bootstrap_hub.py \
  --adapter-dir ./runs/whisper-small_lora_r32/final_adapter \
  --adapter-repo  <user>/court-asr-adapters \
  --submissions-repo <user>/court-asr-submissions \
  --eval-repo <user>/court-asr-eval
```
It refuses adapters the server can't read (e.g. bf16) and prints the exact fix. It never overwrites an existing promoted model.

### 2. Build and upload the held-out eval set
```bash
python ci/make_eval_set.py --data-root D:/tamil_fl_data --out ./eval_set --n 100
huggingface-cli upload <user>/court-asr-eval ./eval_set . --repo-type dataset
```
Uses the **valid** split, spread across districts. Keep it fixed. Only use data you may redistribute to your own
private repo (AI4Bharat releases can be research-only). **Never real court recordings.**

### 3. Deploy the aggregation server (Hugging Face Space)
```bash
python scripts/issue_client_keys.py court-chennai court-madurai court-coimbatore   # prints secrets
# Create a Space: SDK = Gradio (Free), template = Blank, hardware = CPU basic (free). Then:
git clone https://huggingface.co/spaces/<user>/court-asr-aggregator
cp server/{README.md,requirements.txt,app.py,aggregator.py,store.py} court-asr-aggregator/
cd court-asr-aggregator && git add . && git commit -m "deploy" && git push
```
In the Space → Settings:
- **Secrets:** `HF_TOKEN` (server token), `CLIENT_API_KEYS`, `ADMIN_API_KEY`, `GITHUB_DISPATCH_TOKEN`
- **Variables:** `ADAPTER_REPO`, `SUBMISSIONS_REPO`, `GITHUB_REPO` (`owner/repo`), `MIN_CLIENTS` (3)

Open the Space URL: the dashboard should show `v0`. `GET /healthz` returns `{"ok": true}`.

### 4. Configure GitHub Actions
Repo → Settings → Secrets and variables → Actions:
- **Secret:** `HF_TOKEN` (the CI token)
- **Variables:** `ADAPTER_REPO`, `EVAL_REPO` (optional: `BASE_MODEL`, `EVAL_MAX_UTTERANCES`, `PROMOTE_TOLERANCE_WER`, `REQUIRE_IMPROVEMENT`)

### 5. Court clients
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
export FL_SERVER_URL=https://<user>-court-asr-aggregator.hf.space
export FL_CLIENT_ID=court-chennai
export FL_API_KEY=<that court's key>
python -m uvicorn backend.api:app --port 8080    # Starts backend API
cd frontend && npm run dev                       # Starts Next.js UI on http://localhost:3000
```
Each court uses **its own key**. Three courts on one machine works for a demo: run instances with different `FL_CLIENT_ID`/`FL_API_KEY`.

## Smoke test, in this order
1. **Local, no accounts** (proves the code on your machine before any cloud):
   ```bash
   python scripts/bootstrap_local.py --adapter-dir ./runs/.../final_adapter --store ./local_store
   cd server && STORE_BACKEND=local LOCAL_STORE_DIR=../local_store MIN_CLIENTS=2 \
     CLIENT_API_KEYS='{"c1":"k1","c2":"k2"}' ADMIN_API_KEY=adm uvicorn app:app --port 8000
   ```
   Point two Streamlit clients at `http://localhost:8000`, submit a session from each, then
   `python scripts/promote_local.py ./local_store` and click *Check for newer adapter*.
   **If a client says "adapter key mismatch", stop and fix that before touching the cloud.**
2. **Hub:** deploy the Space, submit from 3 clients, confirm a `candidate` branch appears on the adapter repo.
3. **CI:** Actions → *promote-adapter* → Run workflow with `dry_run` ticked (evaluates, writes nothing).
4. **Full loop:** untick dry run; confirm `main` moves to v1 and tag `v1` exists.

## Operations
- **Rollback:** every promoted version is tagged (`v0`, `v1`, ...). To roll back, download `adapter_model.safetensors`,
  `adapter_config.json` and `version.json` from tag `vN` (`hf_hub_download(..., revision="vN")`) and upload them to `main`.
  Clients pick it up via *Check for newer adapter* (it only fetches when the server version is **higher** than theirs, so
  after a rollback restart the client apps to reload).
- **Free Space sleeps after ~48 h idle** and its disk is ephemeral — all state is on the Hub, so a restart loses
  nothing. Open the dashboard a few minutes before any demo to wake it.
- **Revoke a court:** remove its entry from `CLIENT_API_KEYS`, restart the Space.
- **Force a round:** `curl -X POST "$URL/admin/aggregate?force=true" -H "X-API-Key: $ADMIN"`.
- **CI cost/time:** CPU runner, ~100 clips ≈ 5–10 min per adapter; the current model's score is cached after the first run.

## Privacy and security — what is actually guaranteed
- Audio and transcripts are never sent to the server; only clipped LoRA deltas are. The Streamlit app keeps session
  audio in memory and drops it on *End session* / *Discard*.
- The server **re-clips** every update, caps each court's weight (`MAX_WEIGHT_SAMPLES`), validates tensor names/shapes/finite
  values, and **deletes raw deltas** once aggregated. Candidate adapters never reach clients without passing the CI gate.
- **Differential privacy is NOT claimed.** There is clipping and an optional central-noise knob (`SERVER_NOISE_MULT`) but no privacy
  accountant, and with millions of LoRA parameters and only a few courts per round, noise large enough for a meaningful ε would
  swamp the update. Leave it at `0`, and do **not** quote an ε in a report or slide unless you add accounting
  (e.g. Opacus / dp-accounting) and measure the accuracy cost. Model updates can still leak information in principle.
- Do not put real or sensitive testimony through a public cloud demo. Use public-corpus clips or your own recordings
  and say the courts are simulated.
- No rate limiting beyond per-key auth and the upload size cap; put the Space behind a proxy if exposed beyond a demo.

## Troubleshooting
| Symptom | Likely cause / fix |
|---|---|
| Client: `adapter key mismatch` / server `422 tensor names do not match` | Local model's LoRA config differs from the published adapter (rank/targets). Rebuild from the same adapter. |
| `409 stale base_version` | A newer promoted adapter exists. *Check for newer adapter*, then redo the session. |
| `bootstrap_hub.py` says adapter unreadable | Adapter saved as bf16. Re-save as float32 (command printed by the script). |
| No CI run after a round | `GITHUB_REPO`/`GITHUB_DISPATCH_TOKEN` unset or wrong scope (needs Contents: write). Server log says "run the workflow manually" with the revision SHA. |
| CI says `STALE candidate` | `main` moved after that candidate was built. Harmless; the next round builds a fresh one. |
| `st.audio_input` records silence | Known browser/version quirk — use *Upload audio file*. |
| Round never triggers | Fewer than `MIN_CLIENTS` distinct keys have submitted for the current version (a resubmission replaces, it doesn't add). |
