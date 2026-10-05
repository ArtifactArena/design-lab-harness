#!/usr/bin/env bash
# Run the harness across models, each in its OWN git worktree — one isolated copy
# per agent/model, so their git state + run outputs never interfere.
#
#   ./run_experiment.sh google/gemini-3-flash-medium
#   ./run_experiment.sh google/gemini-3-flash-medium openai/gpt-5.5
#
# A MODEL is a path under configs/models/ (without .yaml). For each model this
# creates an isolated worktree off BASE_BRANCH and runs the agent RUNS_PER_MODEL
# times at MAX_STEPS. Results land under <worktree>/runs/.
#
# Env overrides: RUNS_PER_MODEL (default 5), MAX_STEPS (default 10),
#                BASE_BRANCH (default: the current branch).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
RUNS_PER_MODEL="${RUNS_PER_MODEL:-5}"
MAX_STEPS="${MAX_STEPS:-10}"
BASE_BRANCH="${BASE_BRANCH:-$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)}"

[ $# -ge 1 ] || { echo "usage: $0 MODEL [MODEL...]   (e.g. google/gemini-3-flash-medium)" >&2; exit 2; }

for model in "$@"; do
  safe="$(echo "$model" | tr '/' '-')"
  wt="$ROOT/../open-ended-run-$safe"            # sibling worktree dir, one per model
  branch="run/$safe"
  if [ ! -d "$wt" ]; then
    echo ">>> creating worktree $wt (branch $branch off $BASE_BRANCH)"
    git -C "$ROOT" worktree add -B "$branch" "$wt" "$BASE_BRANCH"
  fi
  for i in $(seq 1 "$RUNS_PER_MODEL"); do
    echo ">>> $model — run $i/$RUNS_PER_MODEL (max-steps $MAX_STEPS)"
    ( cd "$wt" && ./run_agent.sh --model "$model" --max-steps "$MAX_STEPS" )
  done
  echo ">>> $model done — runs under $wt/runs/"
done
