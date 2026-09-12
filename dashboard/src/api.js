export const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export async function getJSON(path) {
  const r = await fetch(`${API}${path}`)
  if (!r.ok) throw new Error(`${path}: ${r.status}`)
  return r.json()
}

export async function postJSON(path, body) {
  const r = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) throw new Error(`${path}: ${r.status}`)
  return r.json()
}

export function configLabel(c) {
  if (!c) return '—'
  const rt = c.runtime === 'tensorrt' ? 'TensorRT' : 'PyTorch'
  return `${c.precision.toUpperCase()}+${rt} @${c.resolution} b${c.batch_size}`
}
