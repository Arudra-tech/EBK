import { useCallback, useEffect, useRef, useState } from 'react'
import { getJSON } from './api'
import { useLiveData } from './ws'
import LatencyChart from './components/LatencyChart'
import SloCard from './components/SloCard'
import AgentPanel from './components/AgentPanel'
import ExperimentsTable from './components/ExperimentsTable'
import EventLog from './components/EventLog'
import Toasts from './components/Toasts'
import SummaryStrip from './components/SummaryStrip'

const TOAST_EVENTS = {
  spike_detected: { color: 'red', title: '⚠ Latency spike detected' },
  spike_resolved: { color: 'green', title: '✓ Spike resolved' },
  slo_met: { color: 'green', title: '✓ SLO satisfied' },
  config_applied: { color: 'indigo', title: 'Configuration applied' },
}

// Events that change server-side state worth refetching.
const REFRESH_EVENTS = new Set([
  'agent_activated', 'baseline_measured', 'benchmark_done', 'candidate_rejected',
  'config_applied', 'slo_met', 'slo_not_met', 'run_failed', 'spike_resolved',
])

export default function App() {
  const [appState, setAppState] = useState(null)
  const [events, setEvents] = useState([])
  const [experiments, setExperiments] = useState([])
  const [toasts, setToasts] = useState([])
  const toastId = useRef(0)

  const refresh = useCallback(async () => {
    try {
      const [state, exps] = await Promise.all([
        getJSON('/api/state'),
        getJSON('/api/experiments?run_id=latest&limit=12'),
      ])
      setAppState(state)
      setExperiments(exps)
    } catch { /* server restarting */ }
  }, [])

  useEffect(() => {
    refresh()
    getJSON('/api/events?limit=60').then(setEvents).catch(() => {})
  }, [refresh])

  const pushToast = useCallback((event) => {
    const spec = TOAST_EVENTS[event.type]
    if (!spec) return
    const id = ++toastId.current
    setToasts((ts) => [...ts, { id, color: spec.color, title: spec.title, body: event.message }])
    setTimeout(() => setToasts((ts) => ts.filter((t) => t.id !== id)), 6000)
    if (typeof Notification !== 'undefined' && Notification.permission === 'granted') {
      new Notification(spec.title, { body: event.message })
    }
  }, [])

  const onEvent = useCallback(
    (event) => {
      setEvents((evs) => [event, ...evs].slice(0, 150))
      pushToast(event)
      if (REFRESH_EVENTS.has(event.type)) refresh()
    },
    [pushToast, refresh],
  )

  const { telemetry, connected } = useLiveData({ onEvent })

  // A WebSocket can stay "connected" while the harness itself is dead — the
  // server just stops having anything to broadcast. Track the age of the
  // newest sample (by its own server timestamp) so the badge can tell
  // "healthy and quiet" apart from "frozen on the last good frame".
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [])

  const sloTarget = appState?.slo?.target_latency_ms
  const last = telemetry[telemetry.length - 1]
  const lastSampleAgeMs = last?.ts ? now - new Date(last.ts).getTime() : null
  const stale = connected && lastSampleAgeMs !== null && lastSampleAgeMs > 4000
  const breaching = !!(last && sloTarget && last.latency_ms > sloTarget)
  const appliedMarkers = events
    .filter((e) => e.type === 'config_applied')
    .map((e) => ({ ts: e.ts, config: e.payload?.config }))

  return (
    <div className="mx-auto max-w-6xl space-y-4 p-4">
      <Toasts toasts={toasts} />
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-xl font-bold text-zinc-100">
            EBK <span className="font-normal text-zinc-500">· deployment optimizer</span>
          </h1>
          <p className="text-xs text-zinc-600">The LLM proposes. The hardware decides.</p>
        </div>
        <div className="flex items-center gap-3 text-xs text-zinc-500">
          <button
            onClick={() => typeof Notification !== 'undefined' && Notification.requestPermission()}
            className="rounded-lg border border-zinc-800 px-2 py-1 hover:bg-zinc-800"
            title="Enable browser notifications"
          >
            🔔 Enable pings
          </button>
          <span
            className={`flex items-center gap-1.5 ${
              !connected ? 'text-red-500' : stale ? 'text-amber-500' : 'text-emerald-500'
            }`}
            title={stale ? `No new telemetry for ${Math.round(lastSampleAgeMs / 1000)}s — is the harness still up?` : undefined}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                !connected ? 'bg-red-500' : stale ? 'bg-amber-500' : 'bg-emerald-500'
              }`}
            />
            {!connected ? 'reconnecting' : stale ? 'no data' : 'live'}
          </span>
        </div>
      </header>

      <SummaryStrip latestRun={appState?.latest_run} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <LatencyChart telemetry={telemetry} sloTarget={sloTarget} appliedMarkers={appliedMarkers} />
        </div>
        <SloCard appState={appState} breaching={breaching} onSaved={refresh} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <AgentPanel appState={appState} events={events} onActivated={refresh} />
        <ExperimentsTable experiments={experiments} latestRun={appState?.latest_run} />
      </div>

      <EventLog events={events} latestRun={appState?.latest_run} />
    </div>
  )
}
