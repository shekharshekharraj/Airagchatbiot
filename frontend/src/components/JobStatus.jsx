import { useEffect, useState } from 'react'
import { getJob } from '../api'
import usePolling from '../hooks/usePolling'

export default function JobStatus({ jobId }) {
  const [job, setJob] = useState(null)
  const [error, setError] = useState('')

  async function fetchJob() {
    try {
      const data = await getJob(jobId)
      setJob(data)
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    }
  }

  useEffect(() => { if (jobId) fetchJob() }, [jobId])
  usePolling(fetchJob, job?.status === 'done' || job?.status === 'indexed' ? null : 1500, [jobId, job?.status])

  const status = job?.status || 'unknown'
  return (
    <div className="card">
      <div className="row">
        <div className="pill">Job ID: {jobId}</div>
        <div className="pill">Status: {status}</div>
      </div>
      {error && <p style={{ color: 'var(--danger)' }}>{error}</p>}
      {job?.error && <p style={{ color: 'var(--danger)' }}>{job.error}</p>}
      {job?.summary && (
        <>
          <hr/>
        </>
      )}
    </div>
  )
}
