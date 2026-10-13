'use client';

import { useState, useEffect, useRef } from 'react';
import { api } from '../lib/api';
import FLPipelineCanvas from '../components/FLPipeline';

export default function Home() {
  // Session & Status
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('live'); // 'live' | 'visual' | 'analytics'
  const [toast, setToast] = useState(null);

  // Init form
  const [mode, setMode] = useState('simulation');
  const [adapterDir, setAdapterDir] = useState('./runs/whisper-small_lora_r32/final_adapter');
  const [localLr, setLocalLr] = useState(0.0005);
  const [dpClipNorm, setDpClipNorm] = useState(1.0);
  const [nPeers, setNPeers] = useState(3);
  const [peerNoise, setPeerNoise] = useState(0.005);
  const [serverUrl, setServerUrl] = useState('http://localhost:8000');
  const [clientId, setClientId] = useState('court-chennai-01');
  const [apiKey, setApiKey] = useState('');
  const [initLoading, setInitLoading] = useState(false);

  // Tab 1: Live Transcription & Audio
  const [audioSource, setAudioSource] = useState('sample'); // 'sample' | 'mic' | 'upload'
  const [selectedSample, setSelectedSample] = useState('Witness Testimony (சாட்சியம் 1)');
  const [audioUrl, setAudioUrl] = useState('');
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [draftText, setDraftText] = useState('');
  const [clerkCorrection, setClerkCorrection] = useState('');
  const [isConfirming, setIsConfirming] = useState(false);
  const [isEndingSession, setIsEndingSession] = useState(false);

  // Mic recording
  const [isRecording, setIsRecording] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);

  // Tab 2: Visual Pipeline
  const [isSteppingPipeline, setIsSteppingPipeline] = useState(false);
  const [isResettingPipeline, setIsResettingPipeline] = useState(false);
  const [isRunningFL, setIsRunningFL] = useState(false);
  const [peerData, setPeerData] = useState([]);

  // Tab 3: History
  const [history, setHistory] = useState([]);

  // Show Toast helper
  const showToast = (message, type = 'info') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 4000);
  };

  // Fetch status on load
  const refreshStatus = async () => {
    try {
      const data = await api.status();
      setStatus(data);
      if (data.draft) {
        setDraftText(data.draft);
        if (!clerkCorrection) setClerkCorrection(data.draft);
      }
      return data;
    } catch (err) {
      console.warn('API connection offline:', err.message);
    } finally {
      setLoading(false);
    }
  };

  const refreshPeers = async () => {
    try {
      const res = await api.peers();
      setPeerData(res.peers || []);
    } catch (err) {
      console.warn('Error fetching peers:', err);
    }
  };

  const refreshHistory = async () => {
    try {
      const res = await api.history();
      setHistory(res.history || []);
    } catch (err) {
      console.warn('Error fetching history:', err);
    }
  };

  useEffect(() => {
    refreshStatus();
    const interval = setInterval(refreshStatus, 6000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (status?.loaded) {
      refreshPeers();
      refreshHistory();
    }
  }, [status?.loaded, status?.fed_round_no, status?.utterances_this_session]);

  // Handle Init
  const handleInit = async (e) => {
    e.preventDefault();
    setInitLoading(true);
    try {
      const res = await api.init({
        mode,
        adapter_dir: adapterDir,
        local_lr: parseFloat(localLr),
        dp_clip_norm: parseFloat(dpClipNorm),
        n_peers: parseInt(nPeers, 10),
        peer_noise: parseFloat(peerNoise),
        server_url: serverUrl,
        client_id: clientId,
        api_key: apiKey,
      });
      showToast(`Model successfully initialized on ${res.device?.toUpperCase() || 'CPU'}!`, 'success');
      await refreshStatus();
    } catch (err) {
      showToast(`Initialization failed: ${err.message}`, 'error');
    } finally {
      setInitLoading(false);
    }
  };

  // Handle Sample Load
  const handleLoadSample = async () => {
    try {
      const res = await api.loadSample(selectedSample);
      setAudioUrl(api.getAudioUrl());
      setDraftText('');
      setClerkCorrection('');
      showToast(`Sample loaded: "${selectedSample}"`, 'success');
      await refreshStatus();
    } catch (err) {
      showToast(`Failed to load sample: ${err.message}`, 'error');
    }
  };

  // Handle File Upload
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      await api.uploadAudio(file);
      setAudioUrl(api.getAudioUrl());
      setDraftText('');
      setClerkCorrection('');
      showToast(`Uploaded: ${file.name}`, 'success');
      await refreshStatus();
    } catch (err) {
      showToast(`Audio upload failed: ${err.message}`, 'error');
    }
  };

  // Handle Microphone Recording
  const startRecording = async () => {
    audioChunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const file = new File([audioBlob], `mic_utterance_${Date.now()}.wav`, { type: 'audio/wav' });
        try {
          await api.uploadAudio(file);
          setAudioUrl(URL.createObjectURL(audioBlob));
          setDraftText('');
          setClerkCorrection('');
          showToast('Microphone audio captured and stored locally.', 'success');
          await refreshStatus();
        } catch (err) {
          showToast(`Mic upload error: ${err.message}`, 'error');
        }
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordDuration(0);
      timerRef.current = setInterval(() => {
        setRecordDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      showToast(`Microphone permission denied: ${err.message}`, 'error');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(timerRef.current);
    }
  };

  // Handle Transcribe
  const handleTranscribe = async () => {
    setIsTranscribing(true);
    try {
      const res = await api.transcribe();
      setDraftText(res.draft);
      setClerkCorrection(res.draft);
      showToast(`Transcribed in ${res.elapsed_sec}s!`, 'success');
      await refreshStatus();
    } catch (err) {
      showToast(`Transcription error: ${err.message}`, 'error');
    } finally {
      setIsTranscribing(false);
    }
  };

  // Handle Confirm Clerk Correction
  const handleConfirm = async () => {
    if (!clerkCorrection.trim()) {
      showToast('Please enter clerk text to confirm.', 'error');
      return;
    }
    setIsConfirming(true);
    try {
      const res = await api.confirm(clerkCorrection, localLr);
      showToast(`Confirmed! WER: ${res.wer}% · Reward: ${res.reward}${res.loss !== null ? ` · Loss: ${res.loss.toFixed(4)}` : ''}`, 'success');
      setDraftText('');
      setClerkCorrection('');
      setAudioUrl('');
      await refreshStatus();
      await refreshHistory();
    } catch (err) {
      showToast(`Confirmation error: ${err.message}`, 'error');
    } finally {
      setIsConfirming(false);
    }
  };

  // Handle End Session
  const handleEndSession = async () => {
    setIsEndingSession(false);
    try {
      const res = await api.endSession();
      showToast(`Session committed! Delta clipped: ${res.norm_after_clip || 0}. Raw audio deleted from RAM.`, 'success');
      setDraftText('');
      setClerkCorrection('');
      setAudioUrl('');
      await refreshStatus();
    } catch (err) {
      showToast(`End session error: ${err.message}`, 'error');
    }
  };

  // Tab 2: FL Pipeline Controls
  const handleStepPipeline = async () => {
    setIsSteppingPipeline(true);
    try {
      const res = await api.stepPipeline();
      showToast(`Advanced to Visual Stage ${res.active_visual_stage}`, 'info');
      await refreshStatus();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setIsSteppingPipeline(false);
    }
  };

  const handleResetPipeline = async () => {
    setIsResettingPipeline(true);
    try {
      await api.resetPipeline();
      showToast('Visual pipeline reset to Stage 1', 'info');
      await refreshStatus();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setIsResettingPipeline(false);
    }
  };

  const handleRunFLRound = async () => {
    setIsRunningFL(true);
    try {
      const res = await api.runFLRound();
      showToast(`🎉 Round ${res.fed_round_no} finished! Global Adapter promoted to v${res.adapter_version}`, 'success');
      await refreshStatus();
      await refreshPeers();
    } catch (err) {
      showToast(`FL Round error: ${err.message}`, 'error');
    } finally {
      setIsRunningFL(false);
    }
  };

  // Simple Word Error Rate preview calculation
  const calculateWerPreview = (ref, hyp) => {
    if (!ref || !hyp) return 0;
    const rWords = ref.trim().split(/\s+/);
    const hWords = hyp.trim().split(/\s+/);
    if (!rWords.length) return hWords.length ? 100 : 0;
    let diff = 0;
    const maxLen = Math.max(rWords.length, hWords.length);
    for (let i = 0; i < maxLen; i++) {
      if (rWords[i] !== hWords[i]) diff++;
    }
    return Math.min(100, Math.round((diff / rWords.length) * 100));
  };

  const liveWerEstimate = draftText && clerkCorrection ? calculateWerPreview(clerkCorrection, draftText) : 0;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Toast Alert */}
      {toast && (
        <div
          style={{
            position: 'fixed',
            top: '1.25rem',
            right: '1.25rem',
            zIndex: 9999,
            padding: '0.85rem 1.4rem',
            borderRadius: '10px',
            background: toast.type === 'error' ? '#EF4444' : toast.type === 'success' ? '#10B981' : '#62B8E8',
            color: '#060B18',
            fontWeight: 600,
            boxShadow: '0 8px 30px rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            animation: 'slideUp 0.25s ease',
          }}
        >
          <span>{toast.type === 'error' ? '⚠️' : toast.type === 'success' ? '✅' : 'ℹ️'}</span>
          <span>{toast.message}</span>
        </div>
      )}

      {/* Top Navigation Bar */}
      <header
        style={{
          borderBottom: '1px solid var(--border-subtle)',
          background: 'rgba(12, 21, 37, 0.85)',
          backdropFilter: 'blur(12px)',
          position: 'sticky',
          top: 0,
          zIndex: 1000,
        }}
      >
        <div
          style={{
            maxWidth: '1380px',
            margin: '0 auto',
            padding: '0.875rem 1.5rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
            flexWrap: 'wrap',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span style={{ fontSize: '1.6rem', filter: 'drop-shadow(0 2px 8px rgba(98,184,232,0.4))' }}>⚖️</span>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <h1 style={{ fontSize: '1.25rem', fontWeight: 700, letterSpacing: '-0.02em', margin: 0 }}>
                  Federated Tamil Court ASR
                </h1>
                <span className="badge badge-ice" style={{ fontSize: '0.7rem' }}>
                  {status?.mode === 'hosted' ? 'Hosted Federation' : 'Simulation Mode'}
                </span>
                {status?.have_torch && (
                  <span className="badge badge-gold" style={{ fontSize: '0.7rem' }}>
                    {status.device?.toUpperCase() || 'GPU'}
                  </span>
                )}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Differential Privacy · Whisper-LoRA On-Device · Zero-Leakage SFT
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              <span
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: status ? '#10B981' : '#EF4444',
                  boxShadow: status ? '0 0 8px #10B981' : 'none',
                }}
              />
              <span>{status ? 'API Connected' : 'Connecting...'}</span>
            </div>

            {status?.loaded && (
              <button
                className="btn btn-ghost btn-sm"
                onClick={async () => {
                  try {
                    await api.resetPipeline();
                    await refreshStatus();
                    showToast('State refreshed.', 'info');
                  } catch (e) {
                    showToast(e.message, 'error');
                  }
                }}
              >
                🔄 Refresh
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content Container */}
      <main style={{ maxWidth: '1380px', margin: '0 auto', padding: '1.5rem', flex: 1, width: '100%' }}>
        {/* Loading State */}
        {loading && (
          <div style={{ textAlign: 'center', padding: '5rem 0' }}>
            <div className="spinner" style={{ width: '2.5rem', height: '2.5rem', borderWidth: '3px' }} />
            <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>Connecting to Federated Court ASR Service...</p>
          </div>
        )}

        {/* State 1: Model NOT Initialized Yet */}
        {!loading && status && !status.loaded && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
            {/* Welcome & Architecture Highlights Hero */}
            <div className="card" style={{ background: 'linear-gradient(145deg, #0C1525, #111D35)', border: '1px solid var(--border-medium)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1.5rem' }}>
                <div style={{ maxWidth: '780px' }}>
                  <span className="badge badge-ice" style={{ marginBottom: '0.75rem' }}>
                    🏛️ Official Court Transcription System
                  </span>
                  <h2 style={{ fontSize: '1.85rem', fontWeight: 700, marginBottom: '0.75rem', lineHeight: 1.25 }}>
                    Federated Privacy-Preserving Tamil Speech Transcription
                  </h2>
                  <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
                    Designed specifically for Indian judicial chambers and sessions courts. Audio testimony is transcribed locally
                    with Whisper-Small + LoRA adapter. Clerk corrections trigger on-device supervised fine-tuning, strictly preserving witness privacy through L2 Differential Privacy bounds and zero-data retention.
                  </p>
                </div>
                <div
                  style={{
                    background: 'rgba(98,184,232,0.06)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-md)',
                    padding: '1.25rem',
                    minWidth: '260px',
                  }}
                >
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--ice-400)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>
                    System Specifications
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                    <div>Model: <strong style={{ color: '#fff' }}>Whisper-Small (244M)</strong></div>
                    <div>LoRA: <strong style={{ color: '#fff' }}>r=32, α=64 (q_proj, v_proj)</strong></div>
                    <div>Sampling Rate: <strong style={{ color: '#fff' }}>16,000 Hz Mono</strong></div>
                    <div>Target Language: <strong style={{ color: '#fff' }}>Tamil (தமிழ்)</strong></div>
                    <div>Privacy Bound: <strong style={{ color: '#10B981' }}>Differential Privacy L2</strong></div>
                  </div>
                </div>
              </div>

              {/* Architecture highlights grid */}
              <div className="grid-3" style={{ marginTop: '1.75rem' }}>
                <div style={{ background: 'rgba(6,11,24,0.6)', padding: '1.2rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>🎙️ Local Inference</div>
                  <h4 style={{ fontSize: '0.95rem', marginBottom: '0.35rem', color: 'var(--text-primary)' }}>Live On-Device ASR</h4>
                  <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                    Transcribes spoken Tamil witness testimony instantly in courtroom RAM without transmitting voice packets over public networks.
                  </p>
                </div>

                <div style={{ background: 'rgba(6,11,24,0.6)', padding: '1.2rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>🛡️ Zero Audio Leakage</div>
                  <h4 style={{ fontSize: '0.95rem', marginBottom: '0.35rem', color: 'var(--violet-300)' }}>Differential Privacy</h4>
                  <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                    Raw audio and transcripts are irreversibly wiped from RAM upon session closure. Only clipped weight deltas (ΔW) leave the court.
                  </p>
                </div>

                <div style={{ background: 'rgba(6,11,24,0.6)', padding: '1.2rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>⚡ Regional FedAvg</div>
                  <h4 style={{ fontSize: '0.95rem', marginBottom: '0.35rem', color: 'var(--gold-300)' }}>Secure Model Aggregation</h4>
                  <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                    Aggregates parameter updates across Chennai, Madurai, and Coimbatore courts subject to continuous CI WER regression gates.
                  </p>
                </div>
              </div>
            </div>

            {/* Initialization Configuration Panel */}
            <div className="card" style={{ maxWidth: '850px', margin: '0 auto', width: '100%' }}>
              <div className="section-header">
                <div>
                  <h3 style={{ fontSize: '1.2rem' }}>Deployment Configuration & Model Loading</h3>
                  <p style={{ fontSize: '0.825rem' }}>Select operational mode and hyperparameters for local session execution.</p>
                </div>
                <span className="badge badge-neutral">Ready to Initialize</span>
              </div>

              <form onSubmit={handleInit} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                <div>
                  <label className="label">Operational Environment Mode</label>
                  <div className="radio-group">
                    <label className={`radio-option ${mode === 'simulation' ? 'selected' : ''}`}>
                      <input
                        type="radio"
                        name="mode"
                        value="simulation"
                        checked={mode === 'simulation'}
                        onChange={(e) => setMode(e.target.value)}
                      />
                      <span>🏛️ Interactive Multi-Court Simulation (Offline & Review Panel Demo)</span>
                    </label>
                    <label className={`radio-option ${mode === 'hosted' ? 'selected' : ''}`}>
                      <input
                        type="radio"
                        name="mode"
                        value="hosted"
                        checked={mode === 'hosted'}
                        onChange={(e) => setMode(e.target.value)}
                      />
                      <span>🌐 Hosted Federation (Connect to Central Aggregation Server)</span>
                    </label>
                  </div>
                </div>

                <div className="grid-2">
                  <div>
                    <label className="label">LoRA Adapter Directory / Path</label>
                    <input
                      type="text"
                      className="input input-mono"
                      value={adapterDir}
                      onChange={(e) => setAdapterDir(e.target.value)}
                      required
                    />
                  </div>
                  <div>
                    <label className="label">Clerk SFT Learning Rate (Local Nudge)</label>
                    <input
                      type="number"
                      step="0.0001"
                      className="input input-mono"
                      value={localLr}
                      onChange={(e) => setLocalLr(e.target.value)}
                      required
                    />
                  </div>
                </div>

                <div className="grid-3">
                  <div>
                    <label className="label">Differential Privacy L2 Bound</label>
                    <input
                      type="number"
                      step="0.1"
                      className="input input-mono"
                      value={dpClipNorm}
                      onChange={(e) => setDpClipNorm(e.target.value)}
                      required
                    />
                  </div>
                  <div>
                    <label className="label">Simulated Regional Peers</label>
                    <input
                      type="number"
                      min="1"
                      max="10"
                      className="input input-mono"
                      value={nPeers}
                      onChange={(e) => setNPeers(e.target.value)}
                      disabled={mode === 'hosted'}
                    />
                  </div>
                  <div>
                    <label className="label">Peer Delta Variance (Noise Std)</label>
                    <input
                      type="number"
                      step="0.001"
                      className="input input-mono"
                      value={peerNoise}
                      onChange={(e) => setPeerNoise(e.target.value)}
                      disabled={mode === 'hosted'}
                    />
                  </div>
                </div>

                {mode === 'hosted' && (
                  <div className="grid-3 animate-fade-in" style={{ background: 'rgba(255,255,255,0.02)', padding: '1rem', borderRadius: 'var(--radius-sm)' }}>
                    <div>
                      <label className="label">Server Aggregation URL</label>
                      <input
                        type="text"
                        className="input input-mono"
                        value={serverUrl}
                        onChange={(e) => setServerUrl(e.target.value)}
                        required
                      />
                    </div>
                    <div>
                      <label className="label">Court Jurisdiction ID</label>
                      <input
                        type="text"
                        className="input"
                        value={clientId}
                        onChange={(e) => setClientId(e.target.value)}
                        required
                      />
                    </div>
                    <div>
                      <label className="label">Federation API Key</label>
                      <input
                        type="password"
                        className="input"
                        placeholder="court_sec_..."
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                      />
                    </div>
                  </div>
                )}

                <button
                  type="submit"
                  className="btn btn-primary btn-lg btn-full"
                  disabled={initLoading}
                  style={{ marginTop: '0.5rem' }}
                >
                  {initLoading ? (
                    <>
                      <span className="spinner" /> Loading Whisper Model & LoRA Adapter Weights...
                    </>
                  ) : (
                    '🚀 Initialize & Spin Up Local Court Model'
                  )}
                </button>
              </form>
            </div>
          </div>
        )}

        {/* State 2: Model IS Initialized */}
        {!loading && status && status.loaded && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {/* Privacy Guarantee Banner */}
            <div className="privacy-banner">
              <span className="shield-icon">🛡️</span>
              <div>
                <strong>Courtroom Privacy Guarantee Active:</strong> Voice recordings and transcripts are processed strictly in volatile local memory
                and immediately destroyed upon session closure. Only L2-clipped weight deltas ($L_2 \le {status.dp_clip_norm}$) participate in regional aggregation.
              </div>
            </div>

            {/* Top 5 Metrics Strip */}
            <div className="grid-5">
              <div className="stat-chip">
                <span className="stat-label">GLOBAL ADAPTER</span>
                <span className="stat-value text-ice">v{status.adapter_version}</span>
              </div>
              <div className="stat-chip">
                <span className="stat-label">SESSION BUFFER</span>
                <span className="stat-value">{status.utterances_this_session} <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>utterances</span></span>
              </div>
              <div className="stat-chip">
                <span className="stat-label">ROLLING WER (LAST 5)</span>
                <span className="stat-value" style={{ color: status.rolling_wer !== null && status.rolling_wer <= 15 ? 'var(--success)' : 'var(--text-primary)' }}>
                  {status.rolling_wer !== null ? `${status.rolling_wer}%` : '—'}
                </span>
              </div>
              <div className="stat-chip">
                <span className="stat-label">FL ROUNDS COMPLETED</span>
                <span className="stat-value text-gold">{status.fed_round_no}</span>
              </div>
              <div className="stat-chip">
                <span className="stat-label">PRIVACY L2 CAP</span>
                <span className="stat-value text-violet">≤ {status.dp_clip_norm}</span>
              </div>
            </div>

            {/* Tab Bar */}
            <div className="tab-bar">
              <button
                className={`tab-item ${activeTab === 'live' ? 'active' : ''}`}
                onClick={() => setActiveTab('live')}
              >
                🎙️ Live Courtroom Transcription & Clerk Feedback
              </button>
              <button
                className={`tab-item ${activeTab === 'visual' ? 'active' : ''}`}
                onClick={() => setActiveTab('visual')}
              >
                ⚡ Visual Federated Flow & Training Cycle
              </button>
              <button
                className={`tab-item ${activeTab === 'analytics' ? 'active' : ''}`}
                onClick={() => setActiveTab('analytics')}
              >
                📈 Performance Trends & Session Audit Log ({history.length})
              </button>
            </div>

            {/* TAB 1: Live Courtroom Transcription & Clerk Feedback */}
            {activeTab === 'live' && (
              <div className="grid-2 animate-fade-in" style={{ alignItems: 'start' }}>
                {/* Left Column: Audio Source & Audio Player */}
                <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  <div className="section-header" style={{ marginBottom: 0 }}>
                    <h3 style={{ fontSize: '1.05rem' }}>1. Courtroom Witness Audio Source</h3>
                    <span className="badge badge-neutral">Audio Input</span>
                  </div>

                  {/* Audio Source Options */}
                  <div className="radio-group">
                    <label className={`radio-option ${audioSource === 'sample' ? 'selected' : ''}`}>
                      <input
                        type="radio"
                        name="audioSource"
                        value="sample"
                        checked={audioSource === 'sample'}
                        onChange={() => setAudioSource('sample')}
                      />
                      <span>Built-in Legal Clips</span>
                    </label>
                    <label className={`radio-option ${audioSource === 'mic' ? 'selected' : ''}`}>
                      <input
                        type="radio"
                        name="audioSource"
                        value="mic"
                        checked={audioSource === 'mic'}
                        onChange={() => setAudioSource('mic')}
                      />
                      <span>Microphone (Live)</span>
                    </label>
                    <label className={`radio-option ${audioSource === 'upload' ? 'selected' : ''}`}>
                      <input
                        type="radio"
                        name="audioSource"
                        value="upload"
                        checked={audioSource === 'upload'}
                        onChange={() => setAudioSource('upload')}
                      />
                      <span>Upload File (.wav, .mp3)</span>
                    </label>
                  </div>

                  {/* Source Sub-Interface: Built-in Samples */}
                  {audioSource === 'sample' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                      <label className="label">Select Pre-Bundled Tamil Court Utterance</label>
                      <select
                        className="input"
                        value={selectedSample}
                        onChange={(e) => setSelectedSample(e.target.value)}
                      >
                        {status.sample_utterances &&
                          Object.keys(status.sample_utterances).map((k) => (
                            <option key={k} value={k}>
                              {k}
                            </option>
                          ))}
                      </select>

                      {status.sample_utterances?.[selectedSample] && (
                        <div
                          style={{
                            background: 'rgba(98,184,232,0.06)',
                            border: '1px solid var(--border-subtle)',
                            borderRadius: 'var(--radius-sm)',
                            padding: '0.75rem',
                            fontSize: '0.8rem',
                          }}
                        >
                          <div>📍 <strong>Origin:</strong> {status.sample_utterances[selectedSample].court}</div>
                          <div>⚖️ <strong>Context:</strong> {status.sample_utterances[selectedSample].description}</div>
                        </div>
                      )}

                      <button
                        className="btn btn-secondary btn-full"
                        onClick={handleLoadSample}
                      >
                        📥 Load Selected Statement Clip into RAM
                      </button>
                    </div>
                  )}

                  {/* Source Sub-Interface: Microphone */}
                  {audioSource === 'mic' && (
                    <div
                      style={{
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        padding: '1.5rem',
                        background: 'rgba(255,255,255,0.02)',
                        borderRadius: 'var(--radius-md)',
                        border: '1px dashed var(--border-medium)',
                        gap: '1rem',
                      }}
                    >
                      <div
                        style={{
                          width: '64px',
                          height: '64px',
                          borderRadius: '50%',
                          background: isRecording ? 'rgba(239, 68, 68, 0.2)' : 'rgba(98, 184, 232, 0.1)',
                          border: isRecording ? '2px solid #EF4444' : '2px solid var(--ice-400)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '1.75rem',
                          animation: isRecording ? 'pulse 1s infinite' : 'none',
                        }}
                      >
                        🎙️
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontWeight: 600, color: isRecording ? '#EF4444' : 'var(--text-primary)' }}>
                          {isRecording ? `Recording Witness Statement (${recordDuration}s)` : 'Microphone Ready'}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                          Live 16kHz audio captured securely in browser memory
                        </div>
                      </div>

                      {!isRecording ? (
                        <button className="btn btn-primary" onClick={startRecording}>
                          🔴 Start Recording
                        </button>
                      ) : (
                        <button className="btn btn-danger-outline" onClick={stopRecording}>
                          ⏹️ Stop & Save Utterance
                        </button>
                      )}
                    </div>
                  )}

                  {/* Source Sub-Interface: File Upload */}
                  {audioSource === 'upload' && (
                    <label className="drop-zone">
                      <input
                        type="file"
                        accept="audio/wav,audio/mp3,audio/flac,audio/ogg"
                        onChange={handleFileUpload}
                        style={{ display: 'none' }}
                      />
                      <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📁</div>
                      <div style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
                        Click or drag & drop courtroom audio file
                      </div>
                      <div style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>
                        WAV, MP3, FLAC (automatically resampled to 16kHz mono)
                      </div>
                    </label>
                  )}

                  {/* Audio Player */}
                  {audioUrl && (
                    <div style={{ marginTop: '0.5rem' }}>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                        Loaded Audio Stream:
                      </div>
                      <audio controls src={audioUrl} />
                    </div>
                  )}

                  {/* Ground Truth Reference Transcript */}
                  {status.current_ref_text && (
                    <details
                      style={{
                        background: 'rgba(255,255,255,0.02)',
                        border: '1px solid var(--border-subtle)',
                        borderRadius: 'var(--radius-sm)',
                        padding: '0.65rem 0.85rem',
                        fontSize: '0.8125rem',
                      }}
                    >
                      <summary style={{ cursor: 'pointer', color: 'var(--ice-400)', fontWeight: 500 }}>
                        📄 Ground Truth Reference Transcript (Court Record)
                      </summary>
                      <p style={{ marginTop: '0.5rem', color: 'var(--text-primary)', fontStyle: 'italic' }}>
                        "{status.current_ref_text}"
                      </p>
                    </details>
                  )}

                  {/* Transcribe Button */}
                  <div style={{ marginTop: '0.5rem' }}>
                    <button
                      className="btn btn-primary btn-lg btn-full"
                      disabled={!status.has_audio || isTranscribing}
                      onClick={handleTranscribe}
                    >
                      {isTranscribing ? (
                        <>
                          <span className="spinner" /> Transcribing with Whisper-LoRA...
                        </>
                      ) : (
                        '📝 Transcribe Audio (Local Whisper + LoRA)'
                      )}
                    </button>
                    {!status.has_audio && (
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '0.35rem' }}>
                        Load a sample utterance or upload audio first
                      </div>
                    )}
                  </div>
                </div>

                {/* Right Column: Model Draft & Clerk Verification */}
                <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  <div className="section-header" style={{ marginBottom: 0 }}>
                    <h3 style={{ fontSize: '1.05rem' }}>2. Model Draft & Clerk Verification</h3>
                    <span className="badge badge-gold">Human-in-the-Loop</span>
                  </div>

                  {/* Draft Transcript */}
                  <div>
                    <label className="label">Local Model Output (Whisper Inference)</label>
                    <div
                      style={{
                        background: 'var(--surface-2)',
                        border: '1px solid var(--border-subtle)',
                        borderRadius: 'var(--radius-sm)',
                        padding: '0.85rem 1rem',
                        minHeight: '85px',
                        fontSize: '0.9rem',
                        color: draftText ? 'var(--text-primary)' : 'var(--text-muted)',
                        fontStyle: draftText ? 'normal' : 'italic',
                        whiteSpace: 'pre-wrap',
                      }}
                    >
                      {draftText || 'Awaiting transcription execution from courtroom audio...'}
                    </div>
                  </div>

                  {/* Clerk Correction Station */}
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.375rem' }}>
                      <label className="label" style={{ margin: 0 }}>
                        3. Clerk Correction Station (Supervised Feedback)
                      </label>
                      {draftText && clerkCorrection && (
                        <span className={`badge ${liveWerEstimate > 0 ? 'badge-warning' : 'badge-success'}`} style={{ fontSize: '0.72rem' }}>
                          Edit Distance: ~{liveWerEstimate}% WER
                        </span>
                      )}
                    </div>
                    <textarea
                      className="textarea"
                      placeholder="Review or edit errors (legal terminology, regional phrasing, names)..."
                      value={clerkCorrection}
                      onChange={(e) => setClerkCorrection(e.target.value)}
                      style={{ minHeight: '110px' }}
                    />
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                      💡 Edits directly provide the supervision signal for on-device parameter adaptation (SFT nudge).
                    </div>
                  </div>

                  {/* Action Buttons: Confirm & End Session */}
                  <div className="grid-2" style={{ marginTop: '0.5rem' }}>
                    <button
                      className="btn btn-secondary btn-full"
                      disabled={!draftText || isConfirming}
                      onClick={handleConfirm}
                    >
                      {isConfirming ? (
                        <>
                          <span className="spinner" /> Adapting Local LoRA...
                        </>
                      ) : (
                        '✅ Confirm & Nudge Local LoRA'
                      )}
                    </button>

                    <button
                      className="btn btn-primary btn-full"
                      disabled={status.utterances_this_session === 0 || isEndingSession}
                      onClick={handleEndSession}
                    >
                      {isEndingSession ? (
                        <>
                          <span className="spinner" /> Purging Audio & Clipping...
                        </>
                      ) : (
                        `🏁 End Session (${status.utterances_this_session})`
                      )}
                    </button>
                  </div>

                  {/* Privacy assurance pill inside card */}
                  <div
                    style={{
                      background: 'rgba(124,58,237,0.06)',
                      border: '1px solid rgba(124,58,237,0.15)',
                      borderRadius: 'var(--radius-sm)',
                      padding: '0.65rem 0.85rem',
                      fontSize: '0.75rem',
                      color: 'var(--violet-300)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                    }}
                  >
                    <span>🔒</span>
                    <span>
                      "End Session" immediately zeroes out all witness audio buffers from memory, calculates parameter delta ΔW, and clips it by L2 norm cap.
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* TAB 2: Visual Federated Flow & Training Cycle */}
            {activeTab === 'visual' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }} className="animate-fade-in">
                {/* Interactive Animated Pipeline Canvas */}
                <FLPipelineCanvas
                  activeStage={status.active_visual_stage}
                  metrics={status.last_round_metrics}
                  adapterVersion={status.adapter_version}
                />

                {/* Training Cycle Controls */}
                <div className="card">
                  <div className="section-header">
                    <div>
                      <h3 style={{ fontSize: '1.05rem' }}>⚡ Federated Training Cycle Controls</h3>
                      <p style={{ fontSize: '0.825rem' }}>Step through architectural stages or trigger multi-court FedAvg aggregation.</p>
                    </div>
                  </div>

                  <div className="grid-3">
                    <button
                      className="btn btn-secondary btn-full"
                      disabled={isSteppingPipeline}
                      onClick={handleStepPipeline}
                    >
                      ▶️ Step Stage Animation (Now: {status.active_visual_stage}/5)
                    </button>

                    <button
                      className="btn btn-ghost btn-full"
                      disabled={isResettingPipeline}
                      onClick={handleResetPipeline}
                    >
                      🔄 Reset Visual Stage to 1
                    </button>

                    <button
                      className="btn btn-primary btn-full"
                      disabled={isRunningFL}
                      onClick={handleRunFLRound}
                    >
                      {isRunningFL ? (
                        <>
                          <span className="spinner" /> Running FedAvg...
                        </>
                      ) : (
                        '⚡ Run Multi-Court Federated Round Now'
                      )}
                    </button>
                  </div>
                </div>

                {/* Mathematical Aggregation & Privacy Inspection */}
                <div className="grid-2">
                  {/* Peer Courts Table */}
                  <div className="card">
                    <div className="section-header">
                      <h3 style={{ fontSize: '1rem' }}>Participating Regional Courts in Round</h3>
                      <span className="badge badge-ice">Regional Mesh</span>
                    </div>

                    <div style={{ overflowX: 'auto' }}>
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Court Jurisdiction</th>
                            <th>Samples</th>
                            <th>Weight</th>
                            <th>Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {peerData.map((peer, i) => (
                            <tr key={i}>
                              <td>{peer.court}</td>
                              <td><span className="mono">{peer.samples}</span></td>
                              <td><span className="mono">{peer.weight}</span></td>
                              <td>
                                <span className={`badge ${i === 0 ? 'badge-ice' : 'badge-neutral'}`} style={{ fontSize: '0.7rem' }}>
                                  {peer.status}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Privacy & Gate Verification Summary */}
                  <div className="card">
                    <div className="section-header">
                      <h3 style={{ fontSize: '1rem' }}>Privacy & Quality Gate Verification</h3>
                      <span className="badge badge-success">CI Verified</span>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.85rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--border-subtle)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>LoRA Parameter Count:</span>
                        <strong className="mono">3,538,944 float32 weights</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--border-subtle)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Raw Audio Transmitted:</span>
                        <strong style={{ color: 'var(--success)' }}>0 Bytes (100% Local Retention)</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--border-subtle)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Differential Privacy Bound:</span>
                        <strong className="mono" style={{ color: 'var(--violet-300)' }}>L2 ≤ {status.dp_clip_norm}</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--border-subtle)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Continuous Integration Gate:</span>
                        <strong style={{ color: 'var(--success)' }}>{status.ci_verdict}</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Promoted Global Adapter:</span>
                        <strong className="mono text-ice">v{status.adapter_version}</strong>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* TAB 3: Performance Trends & Session Audit Log */}
            {activeTab === 'analytics' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }} className="animate-fade-in">
                {/* WER Progression Visualization */}
                <div className="card">
                  <div className="section-header">
                    <div>
                      <h3 style={{ fontSize: '1.05rem' }}>Word Error Rate (WER) Progression Across Utterances</h3>
                      <p style={{ fontSize: '0.825rem' }}>Track adaptation accuracy improvements across courtroom clerk verification cycles.</p>
                    </div>
                    {history.length > 0 && (
                      <span className="badge badge-ice">
                        Avg WER: {(history.reduce((acc, h) => acc + h.wer, 0) / history.length).toFixed(1)}%
                      </span>
                    )}
                  </div>

                  {history.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '2.5rem 0', color: 'var(--text-muted)' }}>
                      No utterances transcribed yet. Execute transcriptions in Tab 1 to view error rate curves.
                    </div>
                  ) : (
                    <div className="wer-bar-container">
                      {history.map((h) => (
                        <div key={h.utterance} className="wer-bar-row">
                          <span className="wer-bar-label">#{h.utterance}</span>
                          <div className="wer-bar-track">
                            <div
                              className={`wer-bar-fill ${h.wer > 20 ? 'high' : ''}`}
                              style={{ width: `${Math.min(100, Math.max(5, h.wer))}%` }}
                            />
                          </div>
                          <span className="wer-bar-value">{h.wer}%</span>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', width: '120px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {h.source}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Session Utterance Audit Log */}
                <div className="card">
                  <div className="section-header">
                    <div>
                      <h3 style={{ fontSize: '1.05rem' }}>Session Utterance Audit Log</h3>
                      <p style={{ fontSize: '0.825rem' }}>Immutable record of draft outputs, clerk corrections, and error metrics.</p>
                    </div>
                    <span className="badge badge-neutral">{history.length} records</span>
                  </div>

                  {history.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '2rem 0', color: 'var(--text-muted)' }}>
                      No session history recorded yet.
                    </div>
                  ) : (
                    <div style={{ overflowX: 'auto' }}>
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>#</th>
                            <th>Audio Source</th>
                            <th>Model Draft Output</th>
                            <th>Clerk Supervised Correction</th>
                            <th>WER %</th>
                            <th>FL Round</th>
                          </tr>
                        </thead>
                        <tbody>
                          {history.map((item) => (
                            <tr key={item.utterance}>
                              <td><span className="mono">#{item.utterance}</span></td>
                              <td><span style={{ fontSize: '0.8rem' }}>{item.source}</span></td>
                              <td style={{ maxWidth: '280px', whiteSpace: 'normal', fontSize: '0.825rem' }}>{item.draft}</td>
                              <td style={{ maxWidth: '280px', whiteSpace: 'normal', color: 'var(--ice-300)', fontSize: '0.825rem' }}>{item.correction}</td>
                              <td>
                                <span className={`badge ${item.wer <= 10 ? 'badge-success' : item.wer <= 20 ? 'badge-gold' : 'badge-danger'}`}>
                                  {item.wer}%
                                </span>
                              </td>
                              <td><span className="mono">Round {item.round}</span></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: '1px solid var(--border-subtle)',
          padding: '1.25rem 1.5rem',
          textAlign: 'center',
          fontSize: '0.8rem',
          color: 'var(--text-muted)',
          background: 'var(--navy-950)',
          marginTop: 'auto',
        }}
      >
        <div style={{ maxWidth: '1380px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            ⚖️ <strong>Federated Tamil Court ASR System</strong> — Powered by Whisper-Small + LoRA + Differential Privacy FedAvg
          </div>
          <div>
            Hardware Device: <strong style={{ color: 'var(--text-secondary)' }}>{status?.device?.toUpperCase() || 'CPU'}</strong> · Zero Audio Retention Standard
          </div>
        </div>
      </footer>
    </div>
  );
}
