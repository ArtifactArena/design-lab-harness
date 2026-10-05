#!/usr/bin/env bash
# Main big run — the study roster, 5 runs per model, 10 model calls (steps) per run.
#
#   ./run_big_experiment.sh                       # all models below, 5 x 10 each
#   ./run_big_experiment.sh openai/gpt-5.5 ...    # just these models, still 5 x 10
#   RUNS_PER_MODEL=3 MAX_STEPS=80 ./run_big_experiment.sh   # override the defaults
#
# Each (model, run) gets its OWN git worktree off the current branch; every run lands
# in ../open-ended-run-<model>/runs/<ts>/ (workspace + journal + transcript + final/).
# A "step" == one model call, so MAX_STEPS=10 == up to 10 model calls per run.
#
# Runs are SEQUENTIAL (one model, one run at a time): 24 models x 5 runs x 10 turns is
# long and uses real API budget across every provider — launch under tmux/nohup.
# On macOS prepend `MUJOCO_GL=` (physics-only); on a Linux/EGL box leave it unset.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The September-2026 study roster — mirrors configs/models/run-roster-2026-09.yaml
# (the SINGLE list shared with the ARH and SH runs; edit the YAML first, then this).
# tests/test_big_experiment_roster.py asserts the two lists are identical.
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

export RUNS_PER_MODEL="${RUNS_PER_MODEL:-5}"   # 5 runs per model
export MAX_STEPS="${MAX_STEPS:-10}"            # 10 model calls per run

echo ">>> big run: ${#MODELS[@]} models x ${RUNS_PER_MODEL} runs x ${MAX_STEPS} steps"
printf '    %s\n' "${MODELS[@]}"
exec "$HERE/run_experiment.sh" "${MODELS[@]}"
