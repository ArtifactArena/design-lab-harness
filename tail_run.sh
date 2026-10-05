#!/usr/bin/env bash
# Pretty live-tail of a run: what the agent is doing, step by step.
#
#   ./tail_run.sh                  # follow the latest run's journal
#   ./tail_run.sh runs/2026...-ab  # follow a specific run
#   ./tail_run.sh --full           # show full replies + tool results (transcript)
#   ./tail_run.sh --full runs/...  # full view of a specific run
#
# Journal view: one line per step  ->  [step] tool args -> outcome   // note
# Full view: the model's complete reasoning + emitted call + the full tool result
# (incl. exact run_bash/run_python commands and their stdout/stderr).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL=0; RUN=""
for a in "$@"; do
  case "$a" in --full|-f) FULL=1 ;; *) RUN="$a" ;; esac
done
[ -n "$RUN" ] || RUN="$(ls -td "$HERE"/runs/*/ 2>/dev/null | head -1 || true)"
[ -n "$RUN" ] || { echo "no runs yet under $HERE/runs/" >&2; exit 1; }
RUN="${RUN%/}"

if [ "$FULL" = "1" ]; then
  echo "tailing (full) $RUN/transcript.jsonl"
  tail -n +1 -f "$RUN/transcript.jsonl" | python -c '
import sys, json
for line in sys.stdin:
    try: o = json.loads(line)
    except Exception: continue
    role = o.get("role"); step = o.get("step", "")
    print(f"\n===== step {step} [{role}] =====")
    print(o.get("content", ""))
'
else
  echo "tailing $RUN/journal.jsonl  (use --full for replies + results)"
  tail -n +1 -f "$RUN/journal.jsonl" | python -c '
import sys, json
for line in sys.stdin:
    try: o = json.loads(line)
    except Exception: continue
    step = o.get("step"); tool = o.get("tool"); outcome = o.get("outcome", "")
    note = o.get("note", "")
    args = json.dumps(o.get("args", {}))
    if len(args) > 200: args = args[:200] + "...}"
    tail = f"   // {note}" if note else ""
    print(f"[{step}] {tool} {args} -> {outcome}{tail}")
'
fi
