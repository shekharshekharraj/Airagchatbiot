import { useSearchParams } from 'react-router-dom'
import JobStatus from '../components/JobStatus.jsx'
import ChatPanel from '../components/ChatPanel.jsx'

export default function Chat() {
  const [params] = useSearchParams()
  const jobId = params.get('job_id') || ''

  return (
    <div className="grid" style={{ display: 'grid', gap: 16 }}>
      {jobId && <JobStatus jobId={jobId} />}
      <ChatPanel jobId={jobId} />
    </div>
  )
}
