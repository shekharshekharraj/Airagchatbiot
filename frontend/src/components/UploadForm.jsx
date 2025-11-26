import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadAudio } from '../api'

export default function UploadForm() {
  const [file, setFile] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  async function onSubmit(e) {
    e.preventDefault()
    setError('')
    if (!file) { setError('Please choose an audio file.'); return }
    try {
      setBusy(true)
      const { job_id } = await uploadAudio(file)
      navigate(`/chat?job_id=${encodeURIComponent(job_id)}`)
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Upload failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={onSubmit} className="card">
      <h2>Upload audio</h2>
      <p className="status">Accepted: mp3, wav, m4a, ogg, webm</p>
      <div className="row" style={{ marginTop: 10 }}>
        <input className="input" type="file" accept="audio/*"
               onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <button className="btn primary" type="submit" disabled={busy}>
          {busy ? 'Uploading…' : 'Upload & Continue'}
        </button>
      </div>
      {error && <p style={{ color: 'var(--danger)', marginTop: 10 }}>{error}</p>}
    </form>
  )
}
