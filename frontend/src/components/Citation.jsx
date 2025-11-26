
export default function Citation({ c }) {
  const sp = c?.metadata?.speaker ?? c?.speaker ?? 'Speaker'
  const start = c?.metadata?.start ?? c?.start
  const end = c?.metadata?.end ?? c?.end
  const snip = c?.text ?? c?.snippet ?? ''
  return (
    <div className="citation">
      <div><strong>{sp}</strong> [{start}-{end}]</div>
      <div>{snip}</div>
    </div>
  )
}
