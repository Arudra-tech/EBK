import { API } from '../api'

const DOT = {
  spike_detected: 'bg-red-400',
  spike_resolved: 'bg-emerald-400',
  candidate_rejected: 'bg-red-400',
  benchmark_done: 'bg-emerald-400',
  config_applied: 'bg-indigo-400',
  slo_met: 'bg-emerald-400',
  slo_not_met: 'bg-amber-400',
  agent_activated: 'bg-indigo-400',
  run_failed: 'bg-red-400',
}

export default function EventLog({ events, latestRun }) {
  const reportUrl = latestRun && latestRun.status === 'done'
    ? `${API}/api/report/${latestRun.run_id}`
    : null
  return (
    <div className="card flex h-80 flex-col">
      <div className="flex items-center justify-between pb-2">
        <h2 className="text-sm font-semibold tracking-wide text-zinc-400 uppercase">Change log</h2>
        {reportUrl && (
          <a
            href={reportUrl}
            target="_blank"
            rel="noreferrer"
            className="rounded-lg border border-zinc-700 px-3 py-1 text-xs font-medium text-zinc-300 hover:bg-zinc-800"
          >
            ⬇ Download run report
          </a>
        )}
      </div>
      <div className="flex-1 space-y-1.5 overflow-y-auto text-xs">
        {events.map((e, i) => (
          <div key={`${e.ts}-${i}`} className="flex gap-2">
            <span className={`mt-1 h-1.5 w-1.5 shrink-0 rounded-full ${DOT[e.type] ?? 'bg-zinc-600'}`} />
            <div>
              <span className="mono mr-1.5 text-zinc-600">{new Date(e.ts).toLocaleTimeString()}</span>
              <span className="mr-1.5 text-zinc-500">{e.type}</span>
              <span className="text-zinc-300">{e.message}</span>
            </div>
          </div>
        ))}
        {events.length === 0 && <p className="pt-6 text-center text-zinc-600">No events yet.</p>}
      </div>
    </div>
  )
}
