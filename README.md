# ⚖️ Federated Privacy-Preserving Tamil Speech Transcription

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces%20%7C%20Hub-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14%20App%20Router-000000?style=flat&logo=next.js&logoColor=white)](https://nextjs.org/)
[![PEFT / LoRA](https://img.shields.io/badge/PEFT-LoRA%20r%3D32-purple?style=flat)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, privacy-first **Federated Automatic Speech Recognition (ASR)** system custom-built for high-stakes **Tamil courtroom proceedings**. 

Legal courtrooms require speech transcription models to continuously adapt to regional Tamil accents, formal legal registers, magistrate orders, and witness testimonies. However, statutory confidentiality and client-attorney privilege strictly prohibit court recordings from leaving the courthouse.

This platform resolves the tension between **dialect adaptation** and **statutory data privacy** through **Decentralized On-Premise Adaptation**, **Clerk-in-the-Loop Feedback**, **Differential Privacy L2-Norm Bounding**, **Zero-Retention In-Memory Processing**, and **Automated CI/CD Quality Promotion Gates**.

---

## 📑 Table of Contents

- [System Architecture](#-system-architecture)
- [Key Features & Privacy Guarantees](#-key-features--privacy-guarantees)
- [Repository Layout](#-repository-layout)
- [Quickstart (One-Click Setup)](#-quickstart-one-click-setup)
- [Manual Installation](#-manual-installation)
- [Running the Platform](#-running-the-platform)
  - [1. Full-Stack Web Application (Next.js + FastAPI)](#1-full-stack-web-application-nextjs--fastapi)
  - [2. Interactive CLI Simulation](#2-interactive-cli-simulation)
  - [3. Standalone Aggregation Server](#3-standalone-aggregation-server)
- [Privacy & Security Technical Specifications](#-privacy--security-technical-specifications)
- [Model Training & Research Scripts](#-model-training--research-scripts)
- [Cloud Deployment & CI/CD Gate](#-cloud-deployment--cicd-gate)
- [Testing & Quality Verification](#-testing--quality-verification)
- [License & Citation](#-license--citation)

---

## 🏛️ System Architecture

```
                          COURT JURISDICTION NODES (ON-PREMISE)
 ┌───────────────────────────────────────────────────────────────────────────────────┐
 │                                                                                   │
 │   Microphone / Audio ──► Whisper-Small + LoRA ──► Initial Draft Transcript        │
 │                                                         │                         │
 │                                                         ▼                         │
 │                                               Court Clerk Review / Edit           │
 │                                                         │                         │
 │                                                         ▼                         │
 │                        Reward-Weighted SFT Local Update (Edit Distance)           │
 │                                                         │                         │
 │    [ ZERO-RETENTION PURGE ]                             ▼                         │
 │    Raw Audio & Transcripts ──► DESTROYED ──►  Extract Delta: ΔW = W_local - W_0   │
 │                                                         │                         │
 │                                                         ▼                         │
 │                                            Client L2 Norm Clipping:               │
 │                                            ΔW_clipped = ΔW * min(1, C / ||ΔW||_2) │
 └─────────────────────────────────────────────────────────┼─────────────────────────┘
                                                           │
                                            HTTPS Upload (Only ΔW_clipped)
                                                           │
                                                           ▼
 ┌───────────────────────────────────────────────────────────────────────────────────┐
 │                     SECURE FEDERATED AGGREGATION SERVER (FASTAPI / HF SPACE)      │
 │                                                                                   │
 │   • Defense-in-depth: Server-side Re-clipping & Schema Validation                 │
 │   • Sample-weighted Federated Averaging (FedAvg):                                 │
 │          ΔW_global = Σ [ min(s_i, S_max) / Σ s_j ] * ΔW_clipped_i                  │
 │   • Immediate Deletion of Raw Submissions upon Aggregation                        │
 │   • Publishes Aggregated Weights to Private Git Candidate Branch                  │
 └─────────────────────────────────────────────────────────┼─────────────────────────┘
                                                           │
                                             Repository Dispatch Event
                                                           │
                                                           ▼
 ┌───────────────────────────────────────────────────────────────────────────────────┐
 │                   CI/CD ZERO-REGRESSION PROMOTION GATE (GITHUB ACTIONS)           │
 │                                                                                   │
 │   1. Evaluates Candidate Adapter on Held-out Multi-District Validation Set        │
 │   2. Compares Word Error Rate (WER) vs. Current Promoted Adapter                  │
 │   3. Automated Gate Verdict:                                                      │
 │        WER_candidate <= WER_current + Tolerance  ──► PROMOTE TO MAIN & TAG v(N+1) │
 │        WER_candidate >  WER_current + Tolerance  ──► REJECT & LOG REGRESSION     │
 └─────────────────────────────────────────────────────────┼─────────────────────────┘
                                                           │
                                       GET /adapter/download (Version Poll)
                                                           │
                                                           ▼
                                         All Court Nodes Synchronize to v(N+1)
```

---

## 🌟 Key Features & Privacy Guarantees

### 1. Zero-Retention On-Premise Transcription
- **Whisper-Small + LoRA ($r=32$)**: Optimized low-latency speech inference fine-tuned on spoken Tamil dialects.
- **Ephemerality**: Audio recordings and text transcripts are retained solely in volatile RAM during active courtroom sessions and **permanently purged** from memory upon session termination.

### 2. "The Gift of Feedback" (Clerk-in-the-Loop)
- Inspired by *The Gift of Feedback* ([arXiv:2310.00141](https://arxiv.org/abs/2310.00141)), court clerks review model drafts and correct courtroom terminology (accused names, statutory IPC sections, colloquial phrases).
- The difference between draft and correction calculates a normalized edit-distance scalar that dynamically scales a local **Supervised Fine-Tuning (SFT)** parameter nudge.

### 3. Differential Privacy Bounding
- Parameter updates are strictly isolated to low-rank adapter matrices ($\Delta W$).
- **Dual-Layer $L_2$ Norm Clipping**:
  $$\Delta W_{\text{clipped}} = \Delta W \cdot \min\left(1, \frac{C}{\|\Delta W\|_2}\right)$$
  Enforced both on client devices ($C = 1.0$) and re-verified on the server to prevent adversarial poison attacks.

### 4. Interactive Simulation & Visual Architecture Flow
- **Next.js 14 App Router UI**: Clean cyber-judicial dark interface.
- **Dynamic 5-Stage Animated Pipeline**: Visualizes data flows from local court nodes (Chennai, Madurai, Coimbatore, Tiruchirappalli) through privacy clipping shields, server FedAvg aggregation, and CI promotion.
- **Multi-Source Audio**: Supports live microphone recording, user WAV upload, or built-in authentic judicial Tamil audio recordings.

---

## 📂 Repository Layout

```plaintext
├── .env.example                     # Reference environment variables & secrets documentation
├── .gitattributes                   # Line ending enforcement (LF for bash, CRLF for powershell)
├── setup.sh                         # Linux/macOS/Git Bash one-click bootstrap script
├── setup.ps1                        # Windows PowerShell one-click bootstrap script
├── requirements.txt                 # Core Python production dependencies
├── requirements_finetuning.txt      # Model fine-tuning requirements (accelerate, datasets)
├── requirements_fl.txt              # Federated learning dependencies
│
├── frontend/                        # Next.js 14 App Router Dashboard
│   ├── app/                         # Pages, layouts, and global styling
│   ├── components/                  # Animated FL Canvas & UI components
│   └── package.json                 # Node dependencies (Lucide icons, Tailwind-free vanilla CSS)
│
├── backend/                         # Local Courtroom REST API
│   └── api.py                       # FastAPI service connecting Next.js UI to PyTorch/Whisper
│
├── server/                          # Hosted Federated Aggregator (FastAPI / HF Space)
│   ├── app.py                       # Aggregator routes (/submit_delta, /adapter, /admin)
│   ├── aggregator.py                # Mathematical FedAvg and L2 clipping logic
│   ├── store.py                     # Storage abstraction (Local disk store or Hugging Face Hub)
│   └── test_server.py               # Unit tests verifying authentication, math, and clipping
│
├── ci/                              # Continuous Integration & Zero-Regression Gate
│   ├── evaluate_and_promote.py      # Automated benchmark evaluator & Git branch promoter
│   ├── make_eval_set.py             # Curates held-out district validation manifests
│   └── test_ci.py                   # Automated tests for evaluation and WER calculations
│
├── scripts/                         # Operational & Bootstrap Utilities
│   ├── bootstrap_local.py           # Initializes local storage directory with base adapter
│   ├── bootstrap_hub.py             # Initializes Hugging Face repositories with v0 adapter
│   ├── issue_client_keys.py         # Generates cryptographic tokens for client courts
│   └── promote_local.py             # Promotes candidate adapters in offline environments
│
├── training_scripts/                # Model Development & Research
│   ├── train_whisper_lora_local.py  # Supervised LoRA fine-tuning on regional Tamil corpus
│   └── fl_train_from_scratch.py     # Multi-client federated training from scratch simulation
│
├── sample_audio/                    # Bundled Authentic Tamil Judicial Audio Samples
│   ├── court_witness_testimony_1.wav# சாட்சியம் (Witness testimony)
│   ├── court_magistrate_order_2.wav # நீதிமன்ற உத்தரவு (Magistrate hearing)
│   └── court_police_statement_3.wav # காவல்துறை அறிக்கை (Police statement)
│
├── runs/whisper-small_lora_r32/     # Promoted Deployment Weights
│   └── final_adapter/               # adapter_model.safetensors & adapter_config.json
│
├── demo_fl_rl_live.py               # Terminal CLI interactive demonstration script
└── DEPLOYMENT_README.md             # In-depth production deployment operations manual
```

---

## ⚡ Quickstart (One-Click Setup)

The repository provides automated setup scripts that verify Python 3.9+, instantiate `.venv`, auto-detect CUDA 11/12 or CPU for PyTorch, install all dependencies, configure `.env`, and build the frontend:

### Linux / macOS / WSL / Git Bash:
```bash
chmod +x setup.sh
./setup.sh
```

### Windows (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Once complete, open your `.env` file to verify or supply your credentials (defaults work out-of-the-box in simulation mode).

---

## 🛠️ Manual Installation

If you prefer to configure your environment manually:

### 1. Python Virtual Environment
```bash
python3 -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

### 2. PyTorch & Python Dependencies
Install PyTorch tailored to your hardware:
```bash
# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CPU-only:
pip install torch

# Install core project dependencies
pip install -r requirements.txt
```

### 3. Environment Secrets
Copy the reference template:
```bash
cp .env.example .env
```
*(In self-contained simulation mode, default placeholder keys are accepted automatically).*

### 4. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

---

## 🚀 Running the Platform

### 1. Full-Stack Web Application (Next.js + FastAPI)

The platform provides a dual-service architecture:

#### Start the Python Court Backend:
```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8080 --reload
```

#### Start the Next.js Frontend:
```bash
cd frontend
npm run dev
```

Visit **[http://localhost:3000](http://localhost:3000)** in your browser.

#### What you can do in the Web Dashboard:
- **Initialize Model**: Loads `openai/whisper-small` with local rank-32 LoRA weights onto CUDA or CPU.
- **Speech Transcription**: Choose from pre-bundled Tamil judicial audio clips, upload custom WAV recordings, or record real-time audio through your browser microphone.
- **Clerk Correction Loop**: Review the initial draft in Tamil script, modify legal terms, and submit to observe the local LoRA loss and parameter nudge.
- **End Session & Purge**: Extract the $L_2$-bounded parameter delta ($\Delta W$) and confirm in-memory deletion of the raw audio.
- **Visual Pipeline Canvas**: Step through the animated 5-stage federated pipeline with simulated regional court nodes (Chennai, Madurai, Coimbatore, Tiruchirappalli).

---

### 2. Interactive CLI Simulation

For presentations, academic defenses, or headless server environments, run the interactive terminal-based feedback loop:

```bash
# Interactive mode (type corrections directly in terminal)
python demo_fl_rl_live.py --adapter ./runs/whisper-small_lora_r32/final_adapter

# Auto-clerk demonstration mode (uses reference transcript as correction)
python demo_fl_rl_live.py --adapter ./runs/whisper-small_lora_r32/final_adapter --auto-clerk
```

---

### 3. Standalone Aggregation Server

To run the central aggregation server locally:

```bash
# 1. Initialize local model store with v0 adapter
python scripts/bootstrap_local.py --adapter-dir ./runs/whisper-small_lora_r32/final_adapter --store ./local_store

# 2. Launch FastAPI aggregator service
cd server
STORE_BACKEND=local LOCAL_STORE_DIR=../local_store MIN_CLIENTS=2 uvicorn app:app --port 8000
```

---

## 🔒 Privacy & Security Technical Specifications

| Protection Mechanism | Implementation Details |
|---|---|
| **Audio Isolation** | Audio recordings never leave courtroom device hardware; transcription and SFT occur strictly on `localhost`. |
| **Volatile RAM Purge** | Raw byte buffers and decoded tensors are zeroed and de-referenced at session close via Python garbage collector. |
| **Payload Minimization** | Only low-rank delta matrices ($\Delta W$) are transmitted (~3.2 MB for $r=32$), eliminating inverse waveform reconstruction risks. |
| **Dual L2 Norm Bounding** | Parameter deltas clipped to $\|\Delta W\|_2 \le 1.0$ at both client and server boundaries to enforce bounded sensitivity. |
| **Replay & Stale Defense** | Server tracks `base_version`; client updates derived from outdated global versions are strictly rejected (`HTTP 409 Conflict`). |
| **Raw Delta Deletion** | The aggregation server deletes uploaded delta files from storage immediately upon computing the FedAvg aggregate. |

> [!NOTE]
> **Differential Privacy Note:** This architecture enforces strict sensitivity bounding ($L_2$ clipping) and supports central Gaussian perturbation (`SERVER_NOISE_MULT`). However, formal DP guarantees ($\epsilon, \delta$) require accounting tools (e.g., Opacus). In default production configurations, noise is set to `0.0` to preserve domain transcription accuracy.

---

## 🧪 Model Training & Research Scripts

### Supervised LoRA Fine-Tuning
To reproduce the base Tamil courtroom adapter from raw audio manifests:
```bash
python training_scripts/train_whisper_lora_local.py \
    --model_id openai/whisper-small \
    --data_root /path/to/tamil_court_dataset \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 1e-4 \
    --output_dir ./runs/whisper-small_lora_r32
```

### Multi-Client Federated Training from Scratch
To run simulated federated training rounds across non-IID district partitions:
```bash
python training_scripts/fl_train_from_scratch.py \
    --model openai/whisper-small \
    --clients 4 \
    --rounds 10 \
    --local-epochs 1 \
    --clip-norm 1.0
```

---

## ☁️ Cloud Deployment & CI/CD Gate

### Deploying Aggregator to Hugging Face Spaces
1. Create a Hugging Face Space with **Gradio SDK** (blank template, free CPU hardware).
2. Clone the Space repository and deploy the server files:
   ```bash
   git clone https://huggingface.co/spaces/<your-user>/court-asr-aggregator
   cp server/{README.md,requirements.txt,app.py,aggregator.py,store.py} court-asr-aggregator/
   cd court-asr-aggregator
   git add . && git commit -m "Deploy aggregator" && git push
   ```
3. Set Space Secrets in Settings:
   - `HF_TOKEN`: Hugging Face write token.
   - `CLIENT_API_KEYS`: JSON string of court API keys: `{"court-chennai":"key1","court-madurai":"key2"}`.
   - `ADMIN_API_KEY`: Secret string for manual admin operations.
   - `GITHUB_DISPATCH_TOKEN`: Fine-grained GitHub PAT with **Contents: Read and write**.
4. Set Space Variables:
   - `ADAPTER_REPO`: `<your-user>/court-asr-adapters`
   - `SUBMISSIONS_REPO`: `<your-user>/court-asr-submissions`
   - `GITHUB_REPO`: `<your-github-user>/Federated-Privacy-Preserving-Speech-Transcription`
   - `MIN_CLIENTS`: `3`

### CI/CD Zero-Regression Gate (`.github/workflows/promote-adapter.yml`)
When the aggregation server successfully merges updates from sufficient court jurisdictions, it triggers GitHub Actions via `repository_dispatch`. The CI runner:
1. Loads the candidate adapter and the currently promoted baseline.
2. Computes the Word Error Rate (WER) across the standardized Tamil validation manifest.
3. Automatically merges candidate weights into the `main` branch and stamps a new Git version tag (`v1`, `v2`, etc.) only if performance does not regress.

---

## 🔬 Testing & Quality Verification

Run the comprehensive unit test suites for the aggregation server and CI evaluator:

```bash
# 1. Test Aggregation Server logic (Auth, L2 Clipping, Stale Rejection, FedAvg math)
cd server
FL_SERVER_AUTOSTART=0 pytest -v test_server.py
cd ..

# 2. Test Continuous Integration Gate (Tamil text normalizer, WER thresholds, promotion verdicts)
cd ci
pytest -v test_ci.py
cd ..
```

---

## 📜 License & Citation

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details.

### Academic Citation
If you utilize this architecture, simulation framework, or Tamil courtroom adapters in your research, please cite:

```bibtex
@software{tamil_federated_court_asr_2026,
  author = {Jeffrey and Collaborators},
  title = {Federated Privacy-Preserving Speech Transcription for Tamil Courtrooms},
  year = {2026},
  url = {https://github.com/jeffr49/Federated-Privacy-Preserving-Speech-Transcription}
}
```
