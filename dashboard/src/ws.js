import { useEffect, useRef, useState } from 'react'
import { getJSON } from './api'

const WS_URL = 'ws://localhost:8000/ws'
const BUFFER_MAX = 320 // ~105s at 3 Hz

// Live telemetry + event stream over WebSocket, with polling fallback.
export function useLiveData({ onEvent }) {
  const [telemetry, setTelemetry] = useState([])
  const [connected, setConnected] = useState(false)
  const onEventRef = useRef(onEvent)
  onEventRef.current = onEvent

  useEffect(() => {
    let ws
    let pollTimer
    let keepalive
    let closed = false

    getJSON('/api/telemetry?window=110')
      .then((docs) => setTelemetry(docs.slice(-BUFFER_MAX)))
      .catch(() => {})

    function connect() {
      if (closed) return
      ws = new WebSocket(WS_URL)
      ws.onopen = () => {
        setConnected(true)
        clearInterval(pollTimer)
        keepalive = setInterval(() => ws.readyState === 1 && ws.send('ping'), 15000)
      }
      ws.onmessage = (msg) => {
        const data = JSON.parse(msg.data)
        if (data.type === 'telemetry') {
          setTelemetry((buf) => [...buf.slice(-(BUFFER_MAX - 1)), data.sample])
        } else if (data.type === 'event') {
          onEventRef.current?.(data.event)
        }
      }
      ws.onclose = () => {
        clearInterval(keepalive)
        if (closed) return // effect torn down (StrictMode remount / page nav)
        setConnected(false)
        // Fallback: poll while reconnecting so the chart keeps moving.
        pollTimer = setInterval(async () => {
          try {
            const docs = await getJSON('/api/telemetry?window=110')
            setTelemetry(docs.slice(-BUFFER_MAX))
          } catch { /* server down; keep trying */ }
        }, 2000)
        setTimeout(connect, 3000)
      }
      ws.onerror = () => ws.close()
    }
    connect()

    return () => {
      closed = true
      clearInterval(pollTimer)
      clearInterval(keepalive)
      ws?.close()
    }
  }, [])

  return { telemetry, connected }
}
