---
title: Court ASR Aggregator
emoji: ⚖️
colorFrom: blue
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
---

# Federated Court ASR — aggregation server

Receives clipped LoRA-adapter deltas from court clients (never audio or transcripts), aggregates them
into a candidate adapter, and triggers the CI evaluation gate. See the project README for setup.

Required Space **secrets**: `HF_TOKEN`, `CLIENT_API_KEYS`, `ADMIN_API_KEY`, `GITHUB_DISPATCH_TOKEN`
Required Space **variables**: `ADAPTER_REPO`, `SUBMISSIONS_REPO`, `GITHUB_REPO`, `MIN_CLIENTS`
