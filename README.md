# Federated Privacy-Preserving Tamil Speech Transcription

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers%20%7C%20Hub-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14%20App%20Router-000000?style=flat&logo=next.js&logoColor=white)](https://nextjs.org/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20r%3D32-purple?style=flat)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research prototype for federated adaptation of Whisper-small to dialectal Tamil speech. Clients adapt the model locally with LoRA and share only clipped adapter updates. A hosted aggregator combines the updates, and a CI gate promotes a new adapter only if its word error rate does not regress.

Court proceedings are the motivating setting: recordings and transcripts there are confidential and cannot be pooled on a central server. The system is a research prototype for draft transcription with human verification. It is not validated for courtroom use and must not produce official records on its own.

The accompanying paper is "Federated LoRA Adaptation of Whisper for Dialectal Tamil Speech Recognition" (Jeffrey Rufus A, Rishwanth J, Varun R, and C. Christopher Columbus, Vellore Institute of Technology, Chennai).

## Table of Contents

- [Results](#results)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Repository Layout](#repository-layout)
- [Data](#data)
- [Installation](#installation)
- [Running the Platform](#running-the-platform)
- [Privacy and Security Specifications](#privacy-and-security-specifications)
- [Training and Evaluation Scripts](#training-and-evaluation-scripts)
- [Cloud Deployment and CI Gate](#cloud-deployment-and-ci-gate)
- [Testing](#testing)
- [License and Citation](#license-and-citation)

## Results

All scores use the `normalized` transcript field of the corpus as reference, with no further normalization. Decoding is greedy with Tamil transcription and a 395-token generation cap for every model.

| Model | Split | Utterances | WER (%) | CER (%) |
| --- | --- | --- | --- | --- |
| Zero-shot Whisper-small | Validation subset | 1,000 | 98.16 | 58.50 |
| Centralized LoRA | Validation subset | 1,000 | 41.55 | 13.09 |
| Centralized LoRA | Validation | 8,862 | 41.33 | 12.96 |
| Centralized LoRA | Test | 9,969 | 39.45 | 12.19 |
| Federated IFCA + FedProx | Validation | 8,862 | 63.42 | 22.41 |
| Federated IFCA + FedProx | Test | 9,969 | 60.87 | 21.31 |

- The rank-32 adapter on the query and value projections has 3,538,944 trainable parameters (1.44% of the 245.3 million parameter adapted model). A float32 adapter is about 14.2 MB.
- The federated run is preliminary and should not be read as a measure of a federation penalty. It uses 29% of the centralized training data and about 13% of its sample passes, the three IFCA clusters collapse to one from round 15 onward, and the clip bound of 1.0 is far below the raw client update norms of about 13 to 15, so every update is scaled down before aggregation.
- Each configuration was run once with seed 42, so there are no confidence intervals.
- Clipping is implemented, but the noise multiplier is 0 and there is no privacy accountant. The project makes no formal differential-privacy claim.

## System Architecture

```
                         COURT JURISDICTION NODES (ON-PREMISE)
+-----------------------------------------------------------------------------------+
|                                                                                   |
|   Microphone / Audio --> Whisper-Small + LoRA --> Initial Draft Transcript        |
|                                                         |                         |
|                                                         v                         |
|                                               Court Clerk Review / Edit           |
|                                                         |                         |
|                                                         v                         |
|                        Reward-Weighted SFT Local Update (Edit Distance)           |
|                                                         |                         |
|    [ SESSION PURGE ]                                    v                         |
|    Raw Audio & Transcripts --> RELEASED --> Extract Delta: dW = W_local - W_0     |
|                                                         |                         |
|                                                         v                         |
|                                            Client L2 Norm Clipping:               |
|                                            dW_clipped = dW * min(1, C / ||dW||_2) |
+---------------------------------------------------------+-------------------------+
                                                          |
                                           HTTPS Upload (only dW_clipped)
                                                          |
                                                          v
+-----------------------------------------------------------------------------------+
|                     FEDERATED AGGREGATION SERVER (FASTAPI / HF SPACE)             |
|                                                                                   |
|   - Server-side re-clipping and schema validation                                 |
|   - Sample-weighted federated averaging (FedAvg):                                 |
|          dW_global = sum_i [ min(s_i, S_max) / sum_j s_j ] * dW_clipped_i         |
|   - Raw submissions deleted after aggregation                                     |
|   - Aggregated weights published to a private Git candidate branch                |
+---------------------------------------------------------+-------------------------+
                                                          |
                                            Repository Dispatch Event
                                                          |
                                                          v
+-----------------------------------------------------------------------------------+
|                   CI PROMOTION GATE (GITHUB ACTIONS)                              |
|                                                                                   |
|   1. Evaluate the candidate adapter on a held-out multi-district validation set   |
|   2. Compare WER against the currently promoted adapter                           |
|   3. Gate verdict:                                                                |
|        WER_candidate <= WER_current + tolerance  -->  promote to main, tag v(N+1) |
|        WER_candidate >  WER_current + tolerance  -->  reject, log the regression  |
+---------------------------------------------------------+-------------------------+
                                                          |
                                      GET /adapter/download (version poll)
                                                          |
                                                          v
                                        All court nodes synchronize to v(N+1)
```

## Key Features

### On-premise transcription

- Whisper-small with a LoRA adapter (r = 32, alpha = 64, dropout 0.05) on the query and value projections, fine-tuned on dialectal Tamil field speech.
- Audio recordings and transcripts are held in memory during an active session and released when the session ends.

### Clerk-in-the-loop correction

- Based on the approach in "The Gift of Feedback" ([arXiv:2310.00141](https://arxiv.org/abs/2310.00141)): a clerk reviews each draft and corrects terminology such as names, statutory references, and colloquial phrases.
- The normalized edit distance between draft and correction scales a local supervised fine-tuning update.

### Norm-bounded adapter updates

- Training changes only the low-rank adapter matrices, and only the adapter delta leaves the client.
- The delta is clipped at the client (C = 1.0) and clipped again at the server:

  `dW_clipped = dW * min(1, C / ||dW||_2)`

### Web dashboard

- Next.js 14 App Router interface for the local workflow.
- An animated five-stage pipeline view covering simulated regional court nodes, privacy clipping, server aggregation, and CI promotion.
- Audio input from live microphone recording, WAV upload, or the bundled sample recordings.

## Repository Layout

```
.env.example                       Reference environment variables and secrets
.gitattributes                     Line-ending rules (LF for bash, CRLF for PowerShell)
setup.sh                           Linux/macOS/Git Bash bootstrap script
setup.ps1                          Windows PowerShell bootstrap script
requirements.txt                   Core Python dependencies
requirements_finetuning.txt        Fine-tuning dependencies
requirements_fl.txt                Federated learning dependencies

frontend/                          Next.js 14 App Router dashboard
  app/                             Pages, layouts, global styling
  components/                      Animated federated-learning canvas and UI components
  package.json                     Node dependencies

backend/
  api.py                           Local FastAPI service connecting the UI to PyTorch/Whisper

server/                            Hosted federated aggregator (FastAPI / HF Space)
  app.py                           Routes (/submit_delta, /adapter, /admin)
  aggregator.py                    FedAvg and L2 clipping logic
  store.py                         Storage abstraction (local disk or Hugging Face Hub)
  test_server.py                   Tests for authentication, aggregation math, and clipping

ci/                                Evaluation and promotion gate
  evaluate_and_promote.py          Candidate evaluation and Git branch promotion
  make_eval_set.py                 Builds the held-out district validation manifest
  test_ci.py                       Tests for evaluation and WER calculation

scripts/                           Operational utilities
  bootstrap_local.py               Initializes a local store with the base adapter
  bootstrap_hub.py                 Initializes Hugging Face repositories with the v0 adapter
  issue_client_keys.py             Generates API tokens for client courts
  promote_local.py                 Promotes candidate adapters offline

training_scripts/                  Model development and research
  train_whisper_lora_local.py      Centralized LoRA fine-tuning and evaluation
  fl_train_from_scratch.py         Simulated federated training (IFCA + FedProx)
  fl_eval_standalone.py            Evaluation of a saved federated checkpoint (no training)

sample_audio/                      Bundled Tamil sample recordings for the demo
runs/whisper-small_lora_r32/       Promoted deployment adapter
  final_adapter/                   adapter_model.safetensors and adapter_config.json

demo_fl_rl_live.py                 Terminal demonstration of the clerk feedback loop
fl_client_sync.py                  Client-side adapter synchronization
DEPLOYMENT_README.md               Production deployment manual
```

## Data

The training data is a district-organized Tamil subset of IndicVoices.

- Source corpus: [ai4bharat/IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) on Hugging Face, released under CC BY 4.0.
- Derived subset: [ajeffreyrufus/tamil-indicvoices](https://www.kaggle.com/datasets/ajeffreyrufus/tamil-indicvoices) on Kaggle. Cite the original corpus when you use it.
- Ten districts: Cuddalore, Dharmapuri, Erode, Krishnagiri, Mayiladuthurai, Nagapattinam, Namakkal, Sivaganga, Tiruvarur, and Viluppuram.
- Construction: clips between 1 and 30 seconds, at most 15 hours per district, stored as FLAC.
- Splits: speaker-disjoint within each district, with about 15% of speakers in validation, about 15% in test, and the rest in training (seed 42).
- After removing empty transcripts and 67 over-long examples, 62,850 utterances remain: 44,019 train, 8,862 validation (21.1 h), and 9,969 test (24.1 h).
- The federated run uses 12,863 training utterances, capped at about three hours per district.

The scripts expect a folder `tamil_fl_data/` containing `manifest.parquet` and an `audio/` directory. The manifest columns used are `audio_path`, `split`, `district`, `speaker_id`, `duration`, `text`, and `normalized`. Reference text is `normalized`, with `text` as the fallback when it is empty.

## Installation

### One-command setup

The setup scripts check the Python version, create `.venv`, detect CUDA or CPU for PyTorch, install dependencies, create `.env`, and build the frontend.

Linux, macOS, WSL, or Git Bash:

```bash
chmod +x setup.sh
./setup.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Open `.env` afterwards to review the settings. The defaults work for local simulation mode.

### Manual installation

Python 3.10 or newer is required for the platform. The training scripts pin recent library versions (see `requirements_fl.txt`), so Python 3.11 or newer is recommended for them.

1. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # Linux/macOS
   .\.venv\Scripts\Activate.ps1       # Windows PowerShell
   ```

2. Install PyTorch for your hardware, then the project dependencies:

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
   pip install torch                                                     # CPU only
   pip install -r requirements.txt
   ```

3. Copy the environment template:

   ```bash
   cp .env.example .env
   ```

4. Install the frontend dependencies:

   ```bash
   cd frontend
   npm install
   cd ..
   ```

## Running the Platform

### 1. Web application (Next.js and FastAPI)

Start the local backend:

```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8080 --reload
```

Start the frontend:

```bash
cd frontend
npm run dev
```

Open <http://localhost:3000>. The dashboard supports:

- **Initialize model:** loads `openai/whisper-small` with the local rank-32 LoRA weights onto CUDA or CPU.
- **Transcription:** pick a bundled sample clip, upload a WAV file, or record from the browser microphone.
- **Clerk correction:** review the draft in Tamil script, edit it, and submit to see the local LoRA loss and parameter update.
- **End session:** extract the norm-bounded adapter delta and confirm that the session audio is released from memory.
- **Pipeline view:** step through the five-stage federated pipeline with simulated court nodes (Chennai, Madurai, Coimbatore, Tiruchirappalli).

### 2. Command-line simulation

For presentations or headless servers:

```bash
# Interactive: type corrections in the terminal
python demo_fl_rl_live.py --adapter ./runs/whisper-small_lora_r32/final_adapter

# Automatic: uses the reference transcript as the correction
python demo_fl_rl_live.py --adapter ./runs/whisper-small_lora_r32/final_adapter --auto-clerk
```

### 3. Standalone aggregation server

```bash
# Initialize the local store with the v0 adapter
python scripts/bootstrap_local.py --adapter-dir ./runs/whisper-small_lora_r32/final_adapter --store ./local_store

# Start the aggregator
cd server
STORE_BACKEND=local LOCAL_STORE_DIR=../local_store MIN_CLIENTS=2 uvicorn app:app --port 8000
```

## Privacy and Security Specifications

| Mechanism | Implementation |
| --- | --- |
| Audio isolation | Audio stays on the court device. Transcription and local fine-tuning run on `localhost`. |
| Session memory release | Raw audio buffers and decoded tensors are dereferenced when a session closes. |
| Payload size | Only low-rank adapter deltas are sent: about 14.2 MB in float32 for the rank-32 adapter on the query and value projections. |
| Dual norm clipping | Deltas are clipped to an L2 norm of at most 1.0 at the client, and clipped again at the server. |
| Stale update defense | The server tracks `base_version` and rejects updates built on an outdated global version with `HTTP 409 Conflict`. |
| Submission deletion | The aggregation server deletes uploaded delta files after computing the FedAvg aggregate. |

**Differential privacy.** Clipping bounds each client's contribution, and the code supports an optional Gaussian noise multiplier (`SERVER_NOISE_MULT` on the server, `--dp-noise-mult` in simulation). The noise multiplier is 0 in every reported experiment and there is no privacy accountant, so the project makes no formal (epsilon, delta) differential-privacy guarantee. Keeping audio local and clipping updates reduces exposure but does not by itself prevent inference from model updates.

## Training and Evaluation Scripts

All commands below take `--data-root` pointing at the folder that holds `manifest.parquet` and `audio/`.

### Centralized LoRA fine-tuning

The reported centralized run uses ten epochs, AdamW, a peak learning rate of 1e-3, 100 warm-up steps, and seed 42.

```bash
python training_scripts/train_whisper_lora_local.py \
    --data-root /path/to/tamil_fl_data \
    --out-root ./runs \
    --run-name whisper-small_lora_r32 \
    --model openai/whisper-small \
    --epochs 10 \
    --lora-r 32 --lora-alpha 64 \
    --lr 1e-3
```

To evaluate an existing adapter on the validation and test splits without training:

```bash
python training_scripts/train_whisper_lora_local.py \
    --data-root /path/to/tamil_fl_data \
    --out-root ./runs --run-name whisper-small_lora_r32 \
    --eval-only --adapter ./runs/whisper-small_lora_r32/final_adapter
```

### Simulated federated training (IFCA + FedProx)

The defaults match the reported run: 40 rounds, 6 of 10 district clients per round, 60 local steps at batch size 4, 3 IFCA clusters reassigned every 3 rounds, FedProx mu = 0.01, clip norm 1.0, and noise multiplier 0.

```bash
python training_scripts/fl_train_from_scratch.py \
    --data-root /path/to/tamil_fl_data \
    --out-root ./runs \
    --rounds 40 \
    --clients-per-round 6 \
    --local-steps 60 \
    --clusters 3 \
    --fedprox-mu 0.01 \
    --dp-clip-norm 1.0
```

Training resumes automatically from `fl_state_latest.pt` in the run folder unless `--no-resume` is given.

### Evaluating a saved federated checkpoint

`fl_eval_standalone.py` needs only the checkpoint and the dataset. It decodes the validation and test splits with each district's assigned cluster adapter and writes predictions (with the cluster recorded per utterance), overall metrics, and per-district, per-duration, and per-cluster tables.

```bash
python training_scripts/fl_eval_standalone.py \
    --ckpt ./runs/fl_whisper-small_ifca_fedprox/fl_state_latest.pt \
    --data-root /path/to/tamil_fl_data \
    --out-dir ./fl_eval \
    --batch-size 16
```

## Cloud Deployment and CI Gate

### Deploying the aggregator to Hugging Face Spaces

1. Create a Hugging Face Space with the Gradio SDK (blank template, free CPU hardware).
2. Clone the Space and copy in the server files:

   ```bash
   git clone https://huggingface.co/spaces/<your-user>/court-asr-aggregator
   cp server/{README.md,requirements.txt,app.py,aggregator.py,store.py} court-asr-aggregator/
   cd court-asr-aggregator
   git add . && git commit -m "Deploy aggregator" && git push
   ```

3. Set these Space secrets:
   - `HF_TOKEN`: Hugging Face write token.
   - `CLIENT_API_KEYS`: JSON map of court API keys, for example `{"court-chennai":"key1","court-madurai":"key2"}`.
   - `ADMIN_API_KEY`: secret for manual admin operations.
   - `GITHUB_DISPATCH_TOKEN`: fine-grained GitHub token with Contents read and write access.
4. Set these Space variables:
   - `ADAPTER_REPO`: `<your-user>/court-asr-adapters`
   - `SUBMISSIONS_REPO`: `<your-user>/court-asr-submissions`
   - `GITHUB_REPO`: `<your-github-user>/Federated-Privacy-Preserving-Speech-Transcription`
   - `MIN_CLIENTS`: `3`

### CI promotion gate (`.github/workflows/promote-adapter.yml`)

When the aggregation server has merged updates from enough clients, it triggers GitHub Actions through `repository_dispatch`. The runner:

1. Loads the candidate adapter and the currently promoted adapter.
2. Computes WER on the held-out Tamil validation manifest.
3. Merges the candidate into `main` and tags a new version (`v1`, `v2`, and so on) only if WER does not regress beyond the configured tolerance.

The gate checks aggregate WER on a fixed evaluation set. It does not guard against a regression on a district or vocabulary that the set does not cover.

## Testing

```bash
# Aggregation server: authentication, L2 clipping, stale rejection, FedAvg math
cd server
FL_SERVER_AUTOSTART=0 pytest -v test_server.py
cd ..

# CI gate: Tamil text normalizer, WER thresholds, promotion verdicts
cd ci
pytest -v test_ci.py
cd ..
```

## License and Citation

This project is released under the MIT License. See [LICENSE](LICENSE).

If you use this work, please cite the paper and the repository:

```bibtex
@misc{rufus2026federated,
  author = {Rufus A, Jeffrey and J, Rishwanth and R, Varun and Columbus, C. Christopher},
  title  = {Federated {LoRA} Adaptation of {Whisper} for Dialectal {Tamil} Speech Recognition},
  year   = {2026},
  url    = {https://github.com/jeffr49/Federated-Privacy-Preserving-Speech-Transcription}
}
```
