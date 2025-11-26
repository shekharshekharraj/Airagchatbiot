// frontend/src/components/ChatPanel.jsx
import { useEffect, useMemo, useRef, useState } from 'react'
import { chat, streamChat, getOrCreateSessionId } from '../api'
import Citation from './Citation.jsx'
import Message from './Message.jsx'

export default function ChatPanel({ jobId }) {
  const sessionId = useMemo(() => getOrCreateSessionId(), [])
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const scrollerRef = useRef(null)

  // Seed helper text
  useEffect(() => {
    const welcome = jobId
      ? 'Ask about your audio using RAG (e.g., “What did the speaker say about the budget?”).\nFull transcript: “Give me the full transcript”.\nEmail: “Send email” then provide the address when asked.\nFor general questions, just chat normally; I will browse the web automatically when fresh info is needed.'
      : 'No audio linked yet. You can still chat normally.\nI will browse the web automatically when your question needs fresh info (e.g., current weather/rates/news).\nUpload audio anytime to enable RAG Q&A and transcripts.'
    setMessages([{ role: 'bot', content: welcome }])
  }, [jobId])

  useEffect(() => {
    if (!scrollerRef.current) return
    scrollerRef.current.scrollTop = scrollerRef.current.scrollHeight
  }, [messages])

  async function send() {
    const text = input.trim()
    if (!text) return
    setMessages(prev => [...prev, { role: 'user', content: text }])
    setInput('')

    // Create a placeholder assistant message to fill as chunks arrive
    const assistantIndex = messages.length + 1
    setMessages(prev => [...prev, { role: 'bot', content: '' }])

    setBusy(true)
    setStreaming(true)
    try {
      const payload = { session_id: sessionId, message: text }
      if (jobId) payload.job_id = jobId

      // Use streaming endpoint by default
      const gen = await streamChat(payload)

      let accumulated = ''
      for await (const chunk of gen) {
        accumulated += chunk
        setMessages(prev => {
          const copy = [...prev]
          copy[assistantIndex] = { ...copy[assistantIndex], role: 'bot', content: accumulated }
          return copy
        })
      }
    } catch (e) {
      const msg = e?.message || 'Streaming failed'
      setMessages(prev => {
        const copy = [...prev]
        copy[assistantIndex] = { role: 'bot', content: `Error: ${msg}` }
        return copy
      })
    } finally {
      setStreaming(false)
      setBusy(false)
    }
  }

  function onKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="card">
      <h2>Chat {streaming ? '· streaming…' : ''}</h2>
      <div
        ref={scrollerRef}
        className="list"
        style={{ marginTop: 12, maxHeight: 400, overflow: 'auto', whiteSpace: 'pre-wrap' }}
      >
        {messages.map((m, i) => (
          <div key={i}>
            <Message role={m.role} text={m.content} />
            {!!m.citations?.length && (
              <div style={{ marginTop: 6 }}>
                {m.citations.map((c, j) => <Citation key={j} c={c} />)}
              </div>
            )}
          </div>
        ))}
      </div>

      <hr />
      <div className="row">
        <textarea
          className="input"
          style={{ flex: 1, minHeight: 60 }}
          placeholder="Type your message…"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKey}
        />
        <button className="btn primary" onClick={send} disabled={busy || !input.trim()}>
          {busy ? 'Sending…' : 'Send'}
        </button>
      </div>
    </div>
  )
}
