// frontend/src/api.js
import axios from 'axios'

export const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000
})

// helpers
export async function uploadAudio(file) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/upload_audio', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return data // { job_id }
}

export async function getJob(jobId) {
  const { data } = await api.get(`/jobs/${jobId}`)
  return data
}

export async function chat({ session_id, job_id, message }) {
  const { data } = await api.post('/chat', { session_id, job_id, message })
  return data // { answer, citations }
}

// NEW: streaming chat (token-by-token)
export async function streamChat({ session_id, job_id, message }) {
  const res = await fetch(`${API_BASE}/chat_stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id, job_id, message }),
  })
  if (!res.ok || !res.body) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let done = false

  async function* chunks() {
    while (!done) {
      const { value, done: doneReading } = await reader.read()
      if (doneReading) { done = true; break }
      const chunk = decoder.decode(value, { stream: true })
      yield chunk
    }
  }

  return chunks()
}

export function getOrCreateSessionId() {
  let sid = localStorage.getItem('session_id')
  if (!sid) {
    sid = crypto.randomUUID()
    localStorage.setItem('session_id', sid)
  }
  return sid
}
