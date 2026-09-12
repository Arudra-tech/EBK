#!/usr/bin/env bash
# Run the exact NemoClaw command server/proposer.py::call_agent_model() runs, with a
# trivial prompt, and print stdout / stderr / exit code verbatim. Use it to capture the
# raw wrapper shape on a device before touching the parser.
#
#   NEMOCLAW_SANDBOX=my-assistant bash tools/nemoclaw_probe.sh
#
# Env (same names as the server):
#   NEMOCLAW_SANDBOX     sandbox name          (default: ebk-agent)
#   NEMOCLAW_SESSION_ID  OpenClaw session id   (default: ebk-probe, so it never pollutes ebk-optimizer)
#   NEMOCLAW_TIMEOUT_S   agent timeout, seconds (default: 60)
set -u
SANDBOX="${NEMOCLAW_SANDBOX:-ebk-agent}"
SESSION="${NEMOCLAW_SESSION_ID:-ebk-probe}"
TIMEOUT="${NEMOCLAW_TIMEOUT_S:-60}"
PROMPT='Reply with JSON only: {"reasoning":"hi","candidates":[]}'

command -v nemoclaw >/dev/null || { echo "nemoclaw not on PATH" >&2; exit 127; }

echo "== nemoclaw $SANDBOX status =="
nemoclaw "$SANDBOX" status 2>&1 | head -20
echo
echo "== command =="
echo "nemoclaw $SANDBOX agent --session-id $SESSION -m '<prompt>' --json --timeout $TIMEOUT"
echo
OUT=$(mktemp) ERR=$(mktemp)
nemoclaw "$SANDBOX" agent --session-id "$SESSION" -m "$PROMPT" --json --timeout "$TIMEOUT" >"$OUT" 2>"$ERR"
RC=$?
echo "== exit code: $RC =="
echo "== stdout ($(wc -c <"$OUT") bytes) =="; cat "$OUT"; echo
echo "== stderr ($(wc -c <"$ERR") bytes) =="; cat "$ERR"; echo
echo "== parsed by server.proposer.extract_agent_text =="
REPO="$(cd "$(dirname "$0")/.." && pwd)"
(cd "$REPO" && python3 -c "
import sys
from server.proposer import extract_agent_text
print(extract_agent_text(open(sys.argv[1]).read()))
" "$OUT") 2>&1 || true
rm -f "$OUT" "$ERR"
exit $RC
