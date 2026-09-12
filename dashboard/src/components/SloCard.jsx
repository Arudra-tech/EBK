import { useEffect, useState } from 'react'
import { postJSON, configLabel } from '../api'

export default function SloCard({ appState, breaching, onSaved }) {
  const [target, setTarget] = useState('')
  const [budget, setBudget] = useState('')

  useEffect(() => {
    if (appState?.slo) {
      setTarget(String(appState.slo.target_latency_ms))
      setBudget(String(appState.slo.max_accuracy_loss_pp))
    }
  }, [appState?.slo])

  async function save() {
    await postJSON('/api/slo', {
      target_latency_ms: parseFloat(target),
      max_accuracy_loss_pp: parseFloat(budget),
    })
    onSaved()
  }

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-wide text-zinc-400 uppercase">Objective (SLO)</h2>
        <span
          className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
            breaching ? 'bg-red-500/15 text-red-400' : 'bg-emerald-500/15 text-emerald-400'
          }`}
        >
          {breaching ? 'BREACH' : 'MEETING SLO'}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <label className="text-xs text-zinc-500">
          Target latency (ms)
          <input
            className="mono mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
          />
        </label>
        <label className="text-xs text-zinc-500">
          Max accuracy loss (pp)
          <input
            className="mono mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100"
            value={budget}
            onChange={(e) => setBudget(e.target.value)}
          />
        </label>
      </div>
      <button
        onClick={save}
        className="w-full rounded-lg border border-zinc-700 bg-zinc-800 py-1.5 text-sm font-medium hover:bg-zinc-700"
      >
        Update SLO
      </button>
      <div className="border-t border-zinc-800 pt-2 text-xs text-zinc-500">
        Current config:{' '}
        <span className="mono text-zinc-300">{configLabel(appState?.current_config)}</span>
        <span className="ml-2 text-zinc-600">· device {appState?.device_profile ?? '—'}</span>
      </div>
    </div>
  )
}
