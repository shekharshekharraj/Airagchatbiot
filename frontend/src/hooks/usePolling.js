import { useEffect, useRef } from 'react'

export default function usePolling(callback, delayMs, deps = []) {
  const saved = useRef(callback)
  useEffect(() => { saved.current = callback }, [callback])
  useEffect(() => {
    if (delayMs == null) return
    const id = setInterval(() => saved.current?.(), delayMs)
    return () => clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [delayMs, ...deps])
}
