#!/usr/bin/env bash
# Best-effort clock locking so benchmarks are repeatable. Run with sudo.
#   Jetson: max power mode + jetson_clocks (pins CPU/GPU/EMC at max, fan on)
#   GB10:   try nvidia-smi -lgc at max SM clock; if unsupported, we just log clocks per benchmark.
# Never fake "edge-lo" by under-clocking the GB10 when a real Jetson is available.
set -u

if [ -f /etc/nv_tegra_release ]; then
  echo "== Jetson"
  echo "-- current power mode:"; nvpmodel -q 2>/dev/null || echo "   (nvpmodel needs sudo)"
  # Mode 0 is MAXN on most Orin boards (MAXN SUPER on Orin Nano Super). Check /etc/nvpmodel.conf if unsure.
  MODE="${NVP_MODE:-0}"
  echo "-- nvpmodel -m $MODE"; nvpmodel -m "$MODE" || echo "   nvpmodel failed (need sudo?)"
  echo "-- jetson_clocks"; jetson_clocks || echo "   jetson_clocks failed (need sudo?)"
  jetson_clocks --show 2>/dev/null | head -20
  exit 0
fi

echo "== dGPU / GB10"
if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi not found"; exit 1; fi
MAX_SM=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
echo "-- max SM clock: ${MAX_SM:-?} MHz"
if [ -n "${MAX_SM:-}" ]; then
  if nvidia-smi -lgc "$MAX_SM,$MAX_SM" >/dev/null 2>&1; then
    echo "-- locked SM clock to $MAX_SM MHz (undo: sudo nvidia-smi -rgc)"
  else
    echo "-- nvidia-smi -lgc not supported on this device; clocks will be logged per benchmark instead"
  fi
fi
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv
