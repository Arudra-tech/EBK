import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { configLabel } from '../api'

export default function LatencyChart({ telemetry, sloTarget, appliedMarkers }) {
  const data = telemetry.map((s) => ({
    ts: s.ts,
    t: new Date(s.ts).getTime(),
    latency: s.latency_ms,
  }))
  const last = data[data.length - 1]
  // Recharts picks "nice" epoch-ms ticks that land outside the window; hand it
  // evenly spaced ticks from the actual data instead.
  const ticks = data.length > 1
    ? [0, 0.25, 0.5, 0.75, 1].map((f) => data[Math.floor(f * (data.length - 1))].t)
    : undefined
  const breaching = last && sloTarget && last.latency > sloTarget
  const color = breaching ? '#f87171' : '#34d399'
  const yMax = Math.max(sloTarget * 1.4 || 0, ...data.map((d) => d.latency), 10)

  return (
    <div className="card h-72 flex flex-col">
      <div className="flex items-baseline justify-between pb-2">
        <div className="flex items-center gap-2">
          <span className={`live-dot h-2 w-2 rounded-full ${breaching ? 'bg-red-400' : 'bg-emerald-400'}`} />
          <h2 className="text-sm font-semibold tracking-wide text-zinc-400 uppercase">
            Live inference latency
          </h2>
        </div>
        <div className="mono text-2xl font-bold" style={{ color }}>
          {last ? `${last.latency.toFixed(1)} ms` : '—'}
        </div>
      </div>
      <div className={`flex-1 ${breaching ? 'trace-glow-bad' : 'trace-glow-ok'}`}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -18 }}>
            <defs>
              <linearGradient id="latFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.32} />
                <stop offset="100%" stopColor={color} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#27272a" strokeDasharray="3 6" vertical={false} />
            <XAxis
              dataKey="t"
              type="number"
              domain={['dataMin', 'dataMax']}
              ticks={ticks}
              tickFormatter={(t) => new Date(t).toLocaleTimeString([], { hour12: false })}
              stroke="#3f3f46"
              tick={{ fill: '#71717a', fontSize: 10 }}
            />
            <YAxis
              domain={[0, Math.ceil(yMax / 10) * 10]}
              stroke="#3f3f46"
              tick={{ fill: '#71717a', fontSize: 10 }}
            />
            <Tooltip
              contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', borderRadius: 8, fontSize: 12 }}
              labelFormatter={(t) => new Date(t).toLocaleTimeString()}
              formatter={(v) => [`${v.toFixed(1)} ms`, 'latency']}
            />
            {sloTarget && (
              <ReferenceLine
                y={sloTarget}
                stroke="#fbbf24"
                strokeDasharray="6 4"
                label={{ value: `SLO ${sloTarget} ms`, fill: '#fbbf24', fontSize: 10, position: 'insideTopRight' }}
              />
            )}
            {appliedMarkers.map((m) => (
              <ReferenceLine
                key={m.ts}
                x={new Date(m.ts).getTime()}
                stroke="#818cf8"
                strokeDasharray="2 4"
                ifOverflow="discard"
                label={{ value: `⚙ ${configLabel(m.config)}`, fill: '#818cf8', fontSize: 9, position: 'insideTopLeft' }}
              />
            ))}
            <Area
              type="monotone"
              dataKey="latency"
              stroke={color}
              strokeWidth={2}
              fill="url(#latFill)"
              isAnimationActive={false}
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
