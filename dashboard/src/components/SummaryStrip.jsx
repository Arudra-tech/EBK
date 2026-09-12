import { configLabel } from '../api'

export default function SummaryStrip({ latestRun }) {
  if (!latestRun || latestRun.status !== 'done' || !latestRun.winner) return null
  const b = latestRun.baseline
  const w = latestRun.winner
  const Stat = ({ label, children }) => (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">{label}</div>
      <div className="mono text-lg font-bold text-zinc-100">{children}</div>
    </div>
  )
  return (
    <div className="card flex flex-wrap items-center gap-x-8 gap-y-3 border-emerald-800/40 bg-emerald-950/20">
      <Stat label="Recommended">{configLabel(w.config)}</Stat>
      <Stat label="Latency">
        {b.median_latency_ms.toFixed(0)} ms → <span className="text-emerald-300">{w.median_latency_ms.toFixed(1)} ms</span>
      </Stat>
      <Stat label="Speedup"><span className="text-emerald-300">{latestRun.speedup?.toFixed(2)}×</span></Stat>
      <Stat label="Accuracy">
        {(b.accuracy * 100).toFixed(1)}% → {(w.accuracy * 100).toFixed(1)}%{' '}
        <span className="text-sm text-zinc-400">({w.accuracy_delta_pp > 0 ? '+' : ''}{w.accuracy_delta_pp.toFixed(1)} pp)</span>
      </Stat>
      <Stat label="SLO">
        {latestRun.slo_met
          ? <span className="text-emerald-300">satisfied</span>
          : <span className="text-amber-300">not met — fastest valid applied</span>}
      </Stat>
    </div>
  )
}
