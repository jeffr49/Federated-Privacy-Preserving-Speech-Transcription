'use client';

import { useState, useEffect, useRef } from 'react';
import { api } from '../lib/api';
import FLPipelineCanvas from '../components/FLPipeline';
import CourtAnimation from '../components/CourtAnimation';

// Lightweight pure-JS 16kHz mono WAV encoder for uncompressed audio transmission
function encodeWavPcm16(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // Byte rate
  view.setUint16(32, 2, true); // Block align
  view.setUint16(34, 16, true); // 16 bits
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

export default function Home() {
  // Session & Status
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('live'); // 'live' | 'visual' | 'analytics'
  const [toast, setToast] = useState(null);
  const [showGuideModal, setShowGuideModal] = useState(false);

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
  const [selectedSample, setSelectedSample] = useState('Temple Visit Narrative (கோயில் சாட்சியம்)');
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
  const audioCtxRef = useRef(null);
  const micStreamRef = useRef(null);
  const audioProcessorRef = useRef(null);
  const pcmSamplesRef = useRef([]);

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
      await api.loadSample(selectedSample);
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

  // Handle Microphone Recording via Web Audio API (true 16kHz PCM WAV)
  const startRecording = async () => {
    pcmSamplesRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      micStreamRef.current = stream;

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioContextClass();
      audioCtxRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      // Buffer size 4096, 1 input channel, 1 output channel
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      audioProcessorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        if (ctx.sampleRate === 16000) {
          for (let i = 0; i < input.length; i++) {
            pcmSamplesRef.current.push(input[i]);
          }
        } else {
          // Downsample / resample linearly to 16000Hz for Whisper
          const ratio = ctx.sampleRate / 16000;
          const newLength = Math.round(input.length / ratio);
          for (let i = 0; i < newLength; i++) {
            const index = Math.min(input.length - 1, Math.floor(i * ratio));
            pcmSamplesRef.current.push(input[index]);
          }
        }
      };

      source.connect(processor);
      processor.connect(ctx.destination);

      setIsRecording(true);
      setRecordDuration(0);
      timerRef.current = setInterval(() => {
        setRecordDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      showToast(`Microphone permission denied: ${err.message}`, 'error');
    }
  };

  const stopRecording = async () => {
    setIsRecording(false);
    clearInterval(timerRef.current);

    if (audioProcessorRef.current) {
      audioProcessorRef.current.disconnect();
      audioProcessorRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((track) => track.stop());
      micStreamRef.current = null;
    }
    if (audioCtxRef.current) {
      try { audioCtxRef.current.close(); } catch (_) {}
      audioCtxRef.current = null;
    }

    const rawSamples = new Float32Array(pcmSamplesRef.current);
    if (rawSamples.length < 1600) {
      showToast('Recording too short (please speak for at least 1 second).', 'error');
      return;
    }

    const wavBlob = encodeWavPcm16(rawSamples, 16000);
    const file = new File([wavBlob], `mic_utterance_${Date.now()}.wav`, { type: 'audio/wav' });

    try {
      await api.uploadAudio(file);
      setAudioUrl(URL.createObjectURL(wavBlob));
      setDraftText('');
      setClerkCorrection('');
      showToast(`Audio (${(rawSamples.length / 16000).toFixed(1)}s) captured in 16kHz WAV format & ready for transcription!`, 'success');
      await refreshStatus();
    } catch (err) {
      showToast(`Mic upload error: ${err.message}`, 'error');
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
    setIsEndingSession(true);
    try {
      const res = await api.endSession();
      showToast(`Session committed! Delta clipped: ${res.norm_after_clip || 0}. Raw witness audio wiped from RAM.`, 'success');
      setDraftText('');
      setClerkCorrection('');
      setAudioUrl('');
      await refreshStatus();
    } catch (err) {
      showToast(`End session error: ${err.message}`, 'error');
    } finally {
      setIsEndingSession(false);
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

  // Word Error Rate preview calculation
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
            background: toast.type === 'error' ? '#B91C1C' : toast.type === 'success' ? '#15803D' : '#1D4ED8',
            color: '#FFFFFF',
            fontWeight: 600,
            boxShadow: '0 10px 25px -4px rgba(15, 23, 42, 0.25)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            animation: 'slideUp 0.25s ease',
          }}
        >
          <span>{toast.type === 'error' ? '⚠️' : toast.type === 'success' ? '⚖️' : 'ℹ️'}</span>
          <span>{toast.message}</span>
        </div>
      )}

      {/* Review Panel Guide / Viva Modal */}
      {showGuideModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(15, 23, 42, 0.65)',
            backdropFilter: 'blur(6px)',
            zIndex: 10000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '1.5rem',
          }}
          onClick={() => setShowGuideModal(false)}
        >
          <div
            className="card"
            style={{
              maxWidth: '820px',
              width: '100%',
              maxHeight: '90vh',
              overflowY: 'auto',
              background: '#FFFFFF',
              boxShadow: '0 25px 50px -12px rgba(15, 23, 42, 0.25)',
              border: '2px solid rgba(217, 119, 6, 0.4)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #E2E8F0', paddingBottom: '0.75rem', marginBottom: '1.25rem' }}>
              <div>
                <span className="badge badge-gold" style={{ marginBottom: '0.3rem' }}>Review Panel Briefing</span>
                <h3 className="court-heading" style={{ fontSize: '1.25rem', color: 'var(--navy-900)', margin: 0 }}>
                  Deployment Modes & Parameter Reference Guide
                </h3>
              </div>
              <button className="btn btn-ghost btn-sm" onClick={() => setShowGuideModal(false)}>✕ Close</button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
              <div style={{ background: '#FFFBEB', border: '1px solid #FDE68A', padding: '1rem', borderRadius: '8px' }}>
                <strong style={{ color: '#92400E' }}>🏛️ Interactive Multi-Court Simulation (Offline Mode)</strong>
                <p style={{ marginTop: '0.35rem', color: '#78350F', fontSize: '0.825rem' }}>
                  <strong>When to choose:</strong> Viva defense, professor evaluations, and offline demonstrations. Runs self-contained in local memory with virtual regional peer courts (Chennai, Madurai, Coimbatore).
                </p>
                <div style={{ marginTop: '0.5rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  <span className="badge badge-neutral">SFT Learning Rate: 0.0005</span>
                  <span className="badge badge-neutral">L₂ Bound: 1.0</span>
                  <span className="badge badge-neutral">Peers: 3 courts</span>
                </div>
              </div>

              <div style={{ background: '#EFF6FF', border: '1px solid #BFDBFE', padding: '1rem', borderRadius: '8px' }}>
                <strong style={{ color: '#1E40AF' }}>🌐 Hosted Federation (Central Aggregator)</strong>
                <p style={{ marginTop: '0.35rem', color: '#1E3A8A', fontSize: '0.825rem' }}>
                  <strong>When to choose:</strong> Multi-laptop demonstrations or cloud deployments. Authenticates with a remote FastAPI aggregation server via API keys and synchronizes model versions.
                </p>
                <div style={{ marginTop: '0.5rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  <span className="badge badge-neutral">Requires: Server URL & Client Key</span>
                  <span className="badge badge-neutral">Rejects Stale Versions (vN)</span>
                </div>
              </div>

              <div>
                <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#0F172A', marginBottom: '0.5rem' }}>
                  Key Mathematical Parameters & System Behavior
                </h4>
                <ul style={{ paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.825rem' }}>
                  <li><strong>Differential Privacy L₂ Bound (<code>dp_clip_norm = 1.0</code>):</strong> Mathematically limits the Euclidean length of the parameter shift. Ensures sensitive witness vocal qualities cannot be reconstructed via gradient inversion.</li>
                  <li><strong>Clerk SFT Learning Rate (<code>local_lr = 0.0005</code>):</strong> Step size for on-device AdamW adaptation. Weighted by clerk edit distance to prevent catastrophic forgetting.</li>
                  <li><strong>Peer Delta Noise (<code>peer_noise = 0.005</code>):</strong> Injects Gaussian variance N(0, σ²) into virtual peers to represent Non-IID regional dialect variation across district courts.</li>
                </ul>
              </div>

              <div style={{ background: '#F1F5F9', padding: '0.75rem', borderRadius: '6px', fontSize: '0.8rem', color: '#475569' }}>
                📁 Complete comprehensive explanation is permanently documented in <code>UI_Journey/home_page.md</code>.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Top Judicial Header Bar */}
      <header
        style={{
          borderBottom: '2px solid rgba(217, 119, 6, 0.35)',
          background: '#FFFFFF',
          boxShadow: '0 2px 12px rgba(15, 23, 42, 0.05)',
          position: 'sticky',
          top: 0,
          zIndex: 1000,
        }}
      >
        <div
          style={{
            maxWidth: '1380px',
            margin: '0 auto',
            padding: '0.85rem 1.5rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
            flexWrap: 'wrap',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <div
              style={{
                width: '42px',
                height: '42px',
                borderRadius: '8px',
                background: 'linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%)',
                border: '1px solid #D97706',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.5rem',
                boxShadow: '0 2px 6px rgba(180, 83, 9, 0.15)',
              }}
            >
              ⚖️
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                <h1 className="court-heading" style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--navy-900)', margin: 0 }}>
                  FEDERATED TAMIL COURT ASR
                </h1>
                <span className="badge badge-gold" style={{ fontSize: '0.7rem' }}>
                  {status?.mode === 'hosted' ? 'Hosted Federation' : 'Simulation Mode'}
                </span>
                {status?.have_torch && (
                  <span className="badge badge-blue" style={{ fontSize: '0.7rem' }}>
                    {status.device?.toUpperCase() || 'GPU'}
                  </span>
                )}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                High Court of Judicature · On-Device Whisper-LoRA · Differential Privacy Standard
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => setShowGuideModal(true)}
              title="Open Review Panel & Parameter Guide"
            >
              📖 Panel Guide
            </button>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              <span
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: status ? '#15803D' : '#B91C1C',
                  boxShadow: status ? '0 0 6px #15803D' : 'none',
                }}
              />
              <span style={{ fontWeight: 500 }}>{status ? 'Court Engine Active' : 'Connecting...'}</span>
            </div>

            {status?.loaded && (
              <button
                className="btn btn-ghost btn-sm"
                onClick={async () => {
                  try {
                    await api.resetPipeline();
                    await refreshStatus();
                    showToast('Court state synchronized.', 'info');
                  } catch (e) {
                    showToast(e.message, 'error');
                  }
                }}
              >
                🔄 Sync State
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Judicial Chamber Content */}
      <main style={{ maxWidth: '1380px', margin: '0 auto', padding: '1.5rem', flex: 1, width: '100%' }}>
        {/* Loading State */}
        {loading && (
          <div style={{ textAlign: 'center', padding: '6rem 0' }}>
            <div className="spinner" style={{ width: '2.5rem', height: '2.5rem', borderWidth: '3px', color: 'var(--gold-500)' }} />
            <p style={{ marginTop: '1rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
              Connecting to Judicial Court ASR Engine...
            </p>
          </div>
        )}

        {/* State 1: Model NOT Initialized Yet (The Rich Home Page Gateway) */}
        {!loading && status && !status.loaded && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            {/* Judicial Hero Section with Animated Scales of Justice */}
            <div className="card card-court-hero" style={{ padding: '2rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '2rem', alignItems: 'center' }}>
                {/* Left Side: Court Authority & Architecture Brief */}
                <div>
                  <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.85rem' }}>
                    <span className="badge badge-gold" style={{ fontSize: '0.78rem', textTransform: 'uppercase' }}>
                      🏛️ Official Judicial Station Gateway
                    </span>
                    <span className="badge badge-blue" style={{ fontSize: '0.78rem' }}>
                      Tamil Nadu High Court Standard
                    </span>
                  </div>

                  <h2
                    className="court-heading"
                    style={{
                      fontSize: 'clamp(1.75rem, 2.8vw, 2.4rem)',
                      fontWeight: 800,
                      color: 'var(--navy-950)',
                      lineHeight: 1.2,
                      marginBottom: '1rem',
                    }}
                  >
                    Privacy-Preserving Speech Transcription
                  </h2>

                  <p style={{ fontSize: '0.975rem', color: 'var(--text-secondary)', lineHeight: 1.65, marginBottom: '1.35rem' }}>
                    Engineered specifically for Indian courtrooms and sessions benches. Witness testimony is transcribed locally
                    using <strong>Whisper-Small + LoRA</strong>. Supervised clerk corrections adapt the local model on-device.
                    To satisfy the <em>Indian Evidence Act</em>, <strong>audio and transcripts are permanently destroyed</strong> upon session completion—only mathematically bounded L₂ weight deltas participate in regional federated learning.
                  </p>

                  {/* 3 Core Judicial Pillars */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.85rem', marginBottom: '1.5rem' }}>
                    <div style={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: '8px', padding: '0.85rem' }}>
                      <div style={{ fontSize: '1.25rem', marginBottom: '0.25rem' }}>🎙️</div>
                      <div style={{ fontWeight: 700, fontSize: '0.825rem', color: 'var(--navy-900)' }}>Local Inference</div>
                      <div style={{ fontSize: '0.725rem', color: 'var(--text-muted)' }}>Zero audio transmitted outside court RAM</div>
                    </div>

                    <div style={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: '8px', padding: '0.85rem' }}>
                      <div style={{ fontSize: '1.25rem', marginBottom: '0.25rem' }}>🛡️</div>
                      <div style={{ fontWeight: 700, fontSize: '0.825rem', color: 'var(--violet-700)' }}>Differential Privacy</div>
                      <div style={{ fontSize: '0.725rem', color: 'var(--text-muted)' }}>L₂ ≤ 1.0 clipping against gradient attacks</div>
                    </div>

                    <div style={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: '8px', padding: '0.85rem' }}>
                      <div style={{ fontSize: '1.25rem', marginBottom: '0.25rem' }}>⚡</div>
                      <div style={{ fontWeight: 700, fontSize: '0.825rem', color: 'var(--gold-600)' }}>Regional FedAvg</div>
                      <div style={{ fontSize: '0.725rem', color: 'var(--text-muted)' }}>Chennai, Madurai, Coimbatore mesh</div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
                    <button
                      className="btn btn-gold"
                      onClick={() => {
                        const configElem = document.getElementById('init-configuration-panel');
                        configElem?.scrollIntoView({ behavior: 'smooth' });
                      }}
                    >
                      ⚙️ Configure & Mount Station
                    </button>
                    <button
                      className="btn btn-secondary"
                      onClick={() => setShowGuideModal(true)}
                    >
                      📖 Review Panel Parameter Guide
                    </button>
                  </div>
                </div>

                {/* Right Side: Impressive Courtroom Scales of Justice Animation */}
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                  <CourtAnimation />
                </div>
              </div>
            </div>

            {/* Initialization Configuration Panel */}
            <div id="init-configuration-panel" className="card" style={{ maxWidth: '920px', margin: '0 auto', width: '100%', background: '#FFFFFF' }}>
              <div className="section-header" style={{ borderBottom: '1px solid #E2E8F0', paddingBottom: '1rem', marginBottom: '1.25rem' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <h3 className="court-heading" style={{ fontSize: '1.2rem', color: 'var(--navy-900)', margin: 0 }}>
                      Judicial Station Deployment & Model Initialization
                    </h3>
                  </div>
                  <p style={{ fontSize: '0.825rem', marginTop: '0.25rem' }}>
                    Choose the operational architecture and adjust differential privacy bounds for daily courtroom proceedings.
                  </p>
                </div>
                <span className="badge badge-gold">Pre-Flight Config</span>
              </div>

              <form onSubmit={handleInit} style={{ display: 'flex', flexDirection: 'column', gap: '1.35rem' }}>
                {/* Mode Selector Cards */}
                <div>
                  <label className="label" style={{ fontSize: '0.85rem', marginBottom: '0.5rem' }}>
                    Select Operational Environment Mode
                  </label>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <label
                      className="card"
                      style={{
                        padding: '1rem',
                        cursor: 'pointer',
                        border: mode === 'simulation' ? '2px solid #D97706' : '1px solid #E2E8F0',
                        background: mode === 'simulation' ? '#FFFBEB' : '#FFFFFF',
                        transition: 'all 0.2s ease',
                      }}
                    >
                      <input
                        type="radio"
                        name="mode"
                        value="simulation"
                        checked={mode === 'simulation'}
                        onChange={(e) => setMode(e.target.value)}
                        style={{ display: 'none' }}
                      />
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem' }}>
                        <span style={{ fontSize: '1.2rem' }}>🏛️</span>
                        <strong style={{ color: mode === 'simulation' ? '#92400E' : 'var(--navy-900)', fontSize: '0.9rem' }}>
                          Interactive Multi-Court Simulation
                        </strong>
                      </div>
                      <div style={{ fontSize: '0.78rem', color: mode === 'simulation' ? '#78350F' : 'var(--text-muted)', lineHeight: 1.45 }}>
                        Self-contained offline execution with simulated peers (Chennai, Madurai, Coimbatore). <strong>Recommended for Review Panel & Viva.</strong>
                      </div>
                    </label>

                    <label
                      className="card"
                      style={{
                        padding: '1rem',
                        cursor: 'pointer',
                        border: mode === 'hosted' ? '2px solid #1D4ED8' : '1px solid #E2E8F0',
                        background: mode === 'hosted' ? '#EFF6FF' : '#FFFFFF',
                        transition: 'all 0.2s ease',
                      }}
                    >
                      <input
                        type="radio"
                        name="mode"
                        value="hosted"
                        checked={mode === 'hosted'}
                        onChange={(e) => setMode(e.target.value)}
                        style={{ display: 'none' }}
                      />
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem' }}>
                        <span style={{ fontSize: '1.2rem' }}>🌐</span>
                        <strong style={{ color: mode === 'hosted' ? '#1E40AF' : 'var(--navy-900)', fontSize: '0.9rem' }}>
                          Hosted Federation
                        </strong>
                      </div>
                      <div style={{ fontSize: '0.78rem', color: mode === 'hosted' ? '#1E3A8A' : 'var(--text-muted)', lineHeight: 1.45 }}>
                        Connects over secure REST to a central FastAPI aggregation server. Recommended for multi-laptop cloud deployment.
                      </div>
                    </label>
                  </div>
                </div>

                {/* Hyperparameters */}
                <div className="grid-2">
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <label className="label">LoRA Adapter Directory / Path</label>
                      <span className="badge badge-neutral" style={{ fontSize: '0.65rem' }}>r=32, α=64</span>
                    </div>
                    <input
                      type="text"
                      className="input input-mono"
                      value={adapterDir}
                      onChange={(e) => setAdapterDir(e.target.value)}
                      required
                    />
                  </div>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <label className="label">Clerk SFT Learning Rate (Local Nudge)</label>
                      <span className="badge badge-neutral" style={{ fontSize: '0.65rem' }}>Recommended: 0.0005</span>
                    </div>
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
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <label className="label">Differential Privacy L₂ Cap</label>
                      <span className="badge badge-gold" style={{ fontSize: '0.65rem' }}>C = 1.0</span>
                    </div>
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
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <label className="label">Simulated Regional Peers</label>
                      <span className="badge badge-neutral" style={{ fontSize: '0.65rem' }}>Courts</span>
                    </div>
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
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <label className="label">Peer Noise Variance</label>
                      <span className="badge badge-neutral" style={{ fontSize: '0.65rem' }}>Non-IID</span>
                    </div>
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

                {/* Hosted Credentials */}
                {mode === 'hosted' && (
                  <div className="grid-3 animate-fade-in" style={{ background: '#EFF6FF', border: '1px solid #BFDBFE', padding: '1.1rem', borderRadius: 'var(--radius-md)' }}>
                    <div>
                      <label className="label" style={{ color: '#1E40AF' }}>Server Aggregation URL</label>
                      <input
                        type="text"
                        className="input input-mono"
                        value={serverUrl}
                        onChange={(e) => setServerUrl(e.target.value)}
                        required
                      />
                    </div>
                    <div>
                      <label className="label" style={{ color: '#1E40AF' }}>Court Jurisdiction ID</label>
                      <input
                        type="text"
                        className="input"
                        value={clientId}
                        onChange={(e) => setClientId(e.target.value)}
                        required
                      />
                    </div>
                    <div>
                      <label className="label" style={{ color: '#1E40AF' }}>Federation API Key</label>
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
                  className="btn btn-gold btn-lg btn-full"
                  disabled={initLoading}
                  style={{ marginTop: '0.5rem', fontSize: '1rem', padding: '1rem' }}
                >
                  {initLoading ? (
                    <>
                      <span className="spinner" /> Loading Whisper Model & Mounting LoRA Adapter Weights...
                    </>
                  ) : (
                    '🚀 Initialize & Mount Courtroom Transcription Engine'
                  )}
                </button>
              </form>
            </div>
          </div>
        )}

        {/* State 2: Model IS Initialized (Active Judicial Session) */}
        {!loading && status && status.loaded && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.35rem' }}>
            {/* Privacy Guarantee Banner */}
            <div className="privacy-banner">
              <span className="shield-icon">🛡️</span>
              <div>
                <strong>Active Courtroom Privacy Standard:</strong> Audio statements and clerk transcripts are retained strictly in volatile RAM
                and wiped upon session closure. Only clipped LoRA updates (L₂ ≤ {status.dp_clip_norm}) participate in federation.
              </div>
            </div>

            {/* Top 5 Metrics Strip */}
            <div className="grid-5">
              <div className="stat-chip">
                <span className="stat-label">GLOBAL ADAPTER</span>
                <span className="stat-value text-blue">v{status.adapter_version}</span>
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
                <span className="stat-label">PRIVACY L₂ CAP</span>
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
                    <h3 style={{ fontSize: '1.05rem', color: 'var(--navy-900)' }}>1. Courtroom Witness Audio Source</h3>
                    <span className="badge badge-neutral">Input Stream</span>
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
                      <label className="label">Select Pre-Bundled Tamil Speech Utterance</label>
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
                            background: '#F8FAFC',
                            border: '1px solid #E2E8F0',
                            borderRadius: 'var(--radius-sm)',
                            padding: '0.75rem',
                            fontSize: '0.8rem',
                          }}
                        >
                          <div>📍 <strong>Origin:</strong> {status.sample_utterances[selectedSample].court}</div>
                          <div style={{ marginTop: '0.2rem' }}>⚖️ <strong>Context:</strong> {status.sample_utterances[selectedSample].description}</div>
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
                        background: '#F8FAFC',
                        borderRadius: 'var(--radius-md)',
                        border: '1px dashed #CBD5E1',
                        gap: '1rem',
                      }}
                    >
                      <div
                        style={{
                          width: '64px',
                          height: '64px',
                          borderRadius: '50%',
                          background: isRecording ? '#FEE2E2' : '#EFF6FF',
                          border: isRecording ? '2px solid #EF4444' : '2px solid var(--court-blue)',
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
                        <div style={{ fontWeight: 600, color: isRecording ? '#B91C1C' : 'var(--text-primary)' }}>
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
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                        Click or drag & drop courtroom audio file
                      </div>
                      <div style={{ fontSize: '0.75rem', marginTop: '0.25rem', color: 'var(--text-muted)' }}>
                        WAV, MP3, FLAC (automatically resampled to 16kHz mono)
                      </div>
                    </label>
                  )}

                  {/* Audio Player */}
                  {audioUrl && (
                    <div style={{ marginTop: '0.5rem', background: '#F8FAFC', padding: '0.75rem', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600, marginBottom: '0.25rem' }}>
                        Loaded Court Audio Stream:
                      </div>
                      <audio controls src={audioUrl} />
                    </div>
                  )}

                  {/* Ground Truth Reference Transcript */}
                  {status.current_ref_text && (
                    <details
                      style={{
                        background: '#FFFBEB',
                        border: '1px solid #FDE68A',
                        borderRadius: 'var(--radius-sm)',
                        padding: '0.65rem 0.85rem',
                        fontSize: '0.8125rem',
                      }}
                    >
                      <summary style={{ cursor: 'pointer', color: '#92400E', fontWeight: 600 }}>
                        📄 Ground Truth Reference Transcript (Court Record)
                      </summary>
                      <p style={{ marginTop: '0.5rem', color: '#78350F', fontStyle: 'italic' }}>
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
                    <h3 style={{ fontSize: '1.05rem', color: 'var(--navy-900)' }}>2. Model Draft & Clerk Verification</h3>
                    <span className="badge badge-gold">Human-in-the-Loop</span>
                  </div>

                  {/* Draft Transcript */}
                  <div>
                    <label className="label">Local Model Output (Whisper Inference)</label>
                    <div
                      style={{
                        background: '#F8FAFC',
                        border: '1px solid #E2E8F0',
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
                      background: 'var(--violet-subtle)',
                      border: '1px solid var(--violet-border)',
                      borderRadius: 'var(--radius-sm)',
                      padding: '0.65rem 0.85rem',
                      fontSize: '0.75rem',
                      color: 'var(--violet-700)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                    }}
                  >
                    <span>🔒</span>
                    <span>
                      "End Session" immediately zeroes out all witness audio buffers from memory, calculates parameter delta ΔW, and clips it by L₂ norm cap.
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
                      <h3 style={{ fontSize: '1.05rem', color: 'var(--navy-900)' }}>⚡ Federated Training Cycle Controls</h3>
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
                      className="btn btn-gold btn-full"
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
                      <h3 style={{ fontSize: '1rem', color: 'var(--navy-900)' }}>Participating Regional Courts in Round</h3>
                      <span className="badge badge-blue">Regional Mesh</span>
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
                                <span className={`badge ${i === 0 ? 'badge-blue' : 'badge-neutral'}`} style={{ fontSize: '0.7rem' }}>
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
                      <h3 style={{ fontSize: '1rem', color: 'var(--navy-900)' }}>Privacy & Quality Gate Verification</h3>
                      <span className="badge badge-success">CI Verified</span>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.85rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--border-subtle)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>LoRA Parameter Count:</span>
                        <strong className="mono" style={{ color: 'var(--navy-900)' }}>3,538,944 float32 weights</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--border-subtle)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Raw Audio Transmitted:</span>
                        <strong style={{ color: 'var(--success)' }}>0 Bytes (100% Local Retention)</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--border-subtle)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Differential Privacy Bound:</span>
                        <strong className="mono" style={{ color: 'var(--violet-700)' }}>L₂ ≤ {status.dp_clip_norm}</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--border-subtle)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Continuous Integration Gate:</span>
                        <strong style={{ color: 'var(--success)' }}>{status.ci_verdict}</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Promoted Global Adapter:</span>
                        <strong className="mono text-blue">v{status.adapter_version}</strong>
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
                      <h3 style={{ fontSize: '1.05rem', color: 'var(--navy-900)' }}>Word Error Rate (WER) Progression Across Utterances</h3>
                      <p style={{ fontSize: '0.825rem' }}>Track adaptation accuracy improvements across courtroom clerk verification cycles.</p>
                    </div>
                    {history.length > 0 && (
                      <span className="badge badge-blue">
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
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', width: '140px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
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
                      <h3 style={{ fontSize: '1.05rem', color: 'var(--navy-900)' }}>Session Utterance Audit Log</h3>
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
                              <td style={{ maxWidth: '280px', whiteSpace: 'normal', color: 'var(--court-blue)', fontWeight: 500, fontSize: '0.825rem' }}>{item.correction}</td>
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

      {/* Judicial Chamber Footer */}
      <footer
        style={{
          borderTop: '1px solid #E2E8F0',
          padding: '1.25rem 1.5rem',
          textAlign: 'center',
          fontSize: '0.8rem',
          color: 'var(--text-muted)',
          background: '#FFFFFF',
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
