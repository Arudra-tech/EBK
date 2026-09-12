const STYLE = {
  red: 'border-red-500/40 bg-red-950/80 text-red-200',
  green: 'border-emerald-500/40 bg-emerald-950/80 text-emerald-200',
  indigo: 'border-indigo-500/40 bg-indigo-950/80 text-indigo-200',
}

export default function Toasts({ toasts }) {
  return (
    <div className="pointer-events-none fixed right-4 top-4 z-50 flex w-96 flex-col gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`toast rounded-xl border px-4 py-3 text-sm shadow-xl backdrop-blur ${STYLE[t.color]}`}
        >
          <div className="font-bold">{t.title}</div>
          <div className="text-xs opacity-90">{t.body}</div>
        </div>
      ))}
    </div>
  )
}
