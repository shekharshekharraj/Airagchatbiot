
export default function Message({ role, text }) {
  return (
    <div className="message">
      <div className="role">{role === 'bot' ? 'Assistant' : 'You'}</div>
      <div className="text">{text}</div>
    </div>
  )
}
