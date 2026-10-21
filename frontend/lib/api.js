/**
 * lib/api.js — typed API client for the FastAPI backend
 */

const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

async function request(method, path, body, isForm = false) {
  const opts = { method, headers: {} };
  if (body) {
    if (isForm) {
      opts.body = body; // FormData
    } else {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
  }
  const res = await fetch(`${BASE}${path}`, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

export const api = {
  status: () => request('GET', '/api/status'),
  init: (payload) => request('POST', '/api/init', payload),
  loadSample: (sampleKey) => {
    const fd = new FormData();
    fd.append('sample_key', sampleKey);
    return request('POST', '/api/load-sample', fd, true);
  },
  uploadAudio: (file) => {
    const fd = new FormData();
    fd.append('file', file);
    return request('POST', '/api/upload-audio', fd, true);
  },
  transcribe: () => request('POST', '/api/transcribe'),
  transcribeChunk: (file) => {
    const fd = new FormData();
    fd.append('file', file);
    return request('POST', '/api/transcribe-chunk', fd, true);
  },
  setDraft: (draft) => request('POST', '/api/set-draft', { draft }),
  confirm: (correction, localLr) => request('POST', '/api/confirm', { correction, local_lr: localLr }),
  endSession: () => request('POST', '/api/end-session'),
  runFLRound: () => request('POST', '/api/run-fl-round'),
  history: () => request('GET', '/api/history'),
  peers: () => request('GET', '/api/peers'),
  stepPipeline: () => request('POST', '/api/step-pipeline'),
  resetPipeline: () => request('POST', '/api/reset-pipeline'),
  resetSession: () => request('POST', '/api/reset-session'),
  getAudioUrl: () => `${BASE}/api/audio?t=${Date.now()}`,
};
