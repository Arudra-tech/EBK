import { postJSON } from '../api'

const RUN_EVENT_STYLE = {
  agent_activated: 'text-indigo-300',
  baseline_measured: 'text-zinc-300',
  agent_reasoning: 'text-indigo-300 italic',
  candidate_proposed: 'text-sky-300',
  benchmark_started: 'text-zinc-400',
  benchmark_done: 'text-emerald-300',
  candidate_rejected: 'text-red-400',
  config_applied: 'text-emerald-300 font-semibold',
  slo_met: 'text-emerald-300 font-semibold',
  slo_not_met: 'text-amber-300 font-semibold',
  spike_resolved: 'text-emerald-300',
  run_failed: 'text-red-400 font-semibold',
}

export default function AgentPanel({ appState, events, onActivated }) {
  const activeRunId = appState?.active_run_id
  const latestRunId = activeRunId ?? appState?.latest_run?.run_id
  const runEvents = events.filter((e) => e.run_id && e.run_id === latestRunId)
  const benchmarking = runEvents[0]?.type === 'benchmark_started'

  async function activate() {
    try {
      await postJSON('/api/agent/activate')
      onActivated()
    } catch {
      /* 409: already running */
    }
  }

  return (
    <div className="card flex h-72 flex-col">
      <div className="flex items-center justify-between pb-2">
        <h2 className="text-sm font-semibold tracking-wide text-zinc-400 uppercase">
          Optimization agent
        </h2>
        <button
          onClick={activate}
          disabled={!!activeRunId}
          className={`rounded-lg px-4 py-1.5 text-sm font-bold transition ${
            activeRunId
              ? 'cursor-not-allowed bg-zinc-800 text-zinc-500'
              : 'bg-indigo-600 text-white hover:bg-indigo-500'
          }`}
        >
          {activeRunId ? 'Agent running…' : 'Activate Agent'}
        </button>
      </div>
      <div className="flex-1 space-y-1.5 overflow-y-auto text-xs leading-relaxed">
        {runEvents.length === 0 && (
          <p className="pt-6 text-center text-zinc-600">
            No run yet. Set an SLO and activate the agent — or let the watcher
            catch a spike on its own.
          </p>
        )}
        {[...runEvents].reverse().map((e, i) => (
          <div key={`${e.ts}-${i}`} className={RUN_EVENT_STYLE[e.type] ?? 'text-zinc-400'}>
            <span className="mono mr-1 text-zinc-600">
              {new Date(e.ts).toLocaleTimeString()}
            </span>
            {e.message}
            {e.type === 'benchmark_started' && benchmarking && i === runEvents.length - 1 && (
              <span className="ml-1 inline-block animate-spin">◌</span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
