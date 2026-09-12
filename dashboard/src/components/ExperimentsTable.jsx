import { configLabel } from '../api'

export default function ExperimentsTable({ experiments, latestRun }) {
  const base = latestRun?.baseline
  return (
    <div className="card">
      <h2 className="pb-2 text-sm font-semibold tracking-wide text-zinc-400 uppercase">
        Experiments
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-zinc-800 text-zinc-500">
              <th className="py-1.5 pr-3 font-medium">Configuration</th>
              <th className="py-1.5 pr-3 font-medium">Latency</th>
              <th className="py-1.5 pr-3 font-medium">Accuracy</th>
              <th className="py-1.5 pr-3 font-medium">Δ acc</th>
              <th className="py-1.5 font-medium">Verdict</th>
            </tr>
          </thead>
          <tbody className="mono">
            {base && (
              <tr className="border-b border-zinc-800/60 text-zinc-400">
                <td className="py-1.5 pr-3">{configLabel(base.config)} <span className="text-zinc-600">(baseline)</span></td>
                <td className="py-1.5 pr-3">{base.median_latency_ms.toFixed(1)} ms</td>
                <td className="py-1.5 pr-3">{(base.accuracy * 100).toFixed(1)}%</td>
                <td className="py-1.5 pr-3">—</td>
                <td className="py-1.5">reference</td>
              </tr>
            )}
            {experiments.map((e, i) => (
              <tr
                key={i}
                className={`border-b border-zinc-800/60 ${e.valid ? 'text-zinc-200' : 'text-red-400/90'}`}
              >
                <td className="py-1.5 pr-3">{configLabel(e.config)}</td>
                <td className="py-1.5 pr-3">{e.median_latency_ms.toFixed(1)} ms</td>
                <td className="py-1.5 pr-3">{(e.accuracy * 100).toFixed(1)}%</td>
                <td className="py-1.5 pr-3">{e.accuracy_delta_pp > 0 ? '+' : ''}{e.accuracy_delta_pp.toFixed(1)} pp</td>
                <td className="py-1.5">{e.valid ? '✅ valid' : '❌ rejected'}</td>
              </tr>
            ))}
            {experiments.length === 0 && !base && (
              <tr><td colSpan={5} className="py-4 text-center text-zinc-600">No experiments yet.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
