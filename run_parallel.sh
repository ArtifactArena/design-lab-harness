#!/usr/bin/env bash
# Parallel, fully-isolated runner: ONE git worktree per (model, iteration) off main,
# all agents launched concurrently. No two agents ever share a directory — each model
# at each iteration gets its own checkout, because the agent has full read/write over
# its workspace and must not collide with another.
#
#   ./run_parallel.sh                 # iteration 1, the 24-model roster, all parallel
#   ITER_START=2 ITER_END=5 ./run_parallel.sh     # 4 more iterations (2..5), all parallel
#   ./run_parallel.sh openai/gpt-5.5 ...          # just these models
#
# Env: ITER_START (default 1), ITER_END (default ITER_START), MAX_STEPS (default 10),
#      MAX_PARALLEL (default 0 = no cap), BASE_BRANCH (default main).
#
# Worktrees:  ../open-ended-run-<safe-model>-i<iter>   (branch run/<safe-model>-i<iter>)
# Per-run log: parallel_logs/<safe-model>-i<iter>.log
# Manifest:    parallel_logs/manifest-<ITER_START>_<ITER_END>.tsv  (model iter wt log pid)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
MAX_STEPS="${MAX_STEPS:-10}"
BASE_BRANCH="${BASE_BRANCH:-main}"
ITER_START="${ITER_START:-1}"
ITER_END="${ITER_END:-$ITER_START}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"   # 0 = launch everything at once

# The September-2026 roster of record (configs/models/run-roster-2026-09.yaml);
# tests/test_big_experiment_roster.py keeps it equal. Override by passing models as args.
DEFAULT_MODELS=(
  anthropic/claude-fable-5-1-high
  anthropic/claude-fable-5-high
  anthropic/claude-opus-5-high
  anthropic/claude-sonnet-5-high
  anthropic/claude-opus-4-7-high
  anthropic/claude-opus-4-8-high
  openai/gpt-6-astra
  openai/gpt-5.6-luna
  openai/gpt-5.6-sol
  openai/gpt-5.6-terra
  openai/gpt-5.5
  openai/gpt-5.4
  openai/gpt-5.3-codex-high
  google/gemini-3-1-pro-high
  google/gemini-3-8-flash-high
  xai/grok-4.6-high
  xai/grok-4.5-high
  xai/grok-4.20-reasoning
  deepseek/deepseek-v4-pro-thinking
  kimi/kimi-k3-high
  minimax/minimax-m3-high
  qwen/qwen3.8-2.4t-a95b-thinking
  zai/glm-5.3-high
  zai/glm-5.2-high
)
MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || MODELS=("${DEFAULT_MODELS[@]}")

LOGDIR="$ROOT/parallel_logs"
mkdir -p "$LOGDIR"
MANIFEST="$LOGDIR/manifest-${ITER_START}_${ITER_END}.tsv"
: > "$MANIFEST"

echo ">>> parallel run: ${#MODELS[@]} models x iterations ${ITER_START}..${ITER_END}, ${MAX_STEPS} steps each"
echo ">>> base branch: $BASE_BRANCH | max_parallel: $MAX_PARALLEL (0=unlimited)"

# Phase 1 — create every worktree SERIALLY off BASE_BRANCH (concurrent `git worktree
# add` would race on the shared .git). Fast: each is a tiny checkout.
declare -a JOB_MODEL JOB_ITER JOB_WT JOB_LOG
for iter in $(seq "$ITER_START" "$ITER_END"); do
  for model in "${MODELS[@]}"; do
    safe="$(echo "$model" | tr '/' '-')"
    wt="$ROOT/../open-ended-run-${safe}-i${iter}"
    branch="run/${safe}-i${iter}"
    log="$LOGDIR/${safe}-i${iter}.log"
    if [ ! -d "$wt" ]; then
      echo "    + worktree $wt (branch $branch off $BASE_BRANCH)"
      git -C "$ROOT" worktree add -B "$branch" "$wt" "$BASE_BRANCH" >/dev/null
    else
      echo "    = worktree exists, reusing $wt"
    fi
    JOB_MODEL+=("$model"); JOB_ITER+=("$iter"); JOB_WT+=("$wt"); JOB_LOG+=("$log")
  done
done

# Phase 2 — launch every agent concurrently (optionally throttled to MAX_PARALLEL).
declare -a PIDS
n=${#JOB_MODEL[@]}
echo ">>> launching $n agents in parallel..."
for ((j=0; j<n; j++)); do
  model="${JOB_MODEL[$j]}"; iter="${JOB_ITER[$j]}"; wt="${JOB_WT[$j]}"; log="${JOB_LOG[$j]}"
  # Throttle: if a cap is set, wait until fewer than MAX_PARALLEL jobs are running.
  if [ "$MAX_PARALLEL" -gt 0 ]; then
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 2; done
  fi
  ( cd "$wt" && ./run_agent.sh --model "$model" --max-steps "$MAX_STEPS" ) >"$log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  printf '%s\t%s\t%s\t%s\t%s\n' "$model" "$iter" "$wt" "$log" "$pid" >> "$MANIFEST"
  echo "    launched pid $pid : $model (iter $iter) -> $log"
done

echo ">>> all $n agents launched. manifest: $MANIFEST"
echo ">>> waiting for completion (logs stream to $LOGDIR/)..."

# Phase 3 — wait for all, tally exit codes.
fail=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then fail=$((fail+1)); fi
done
echo ">>> done. $((n-fail))/$n agents exited 0, $fail nonzero."
exit 0
