#!/usr/bin/env bash
# Run a model on the harness IN THIS CHECKOUT (no worktrees needed).
#
# Each run writes to its own runs/<timestamp>/ dir, so sequential runs never clobber
# each other. Use this for a single run or a handful of sequential runs. Use
# run_experiment.sh instead only when you want an isolated git worktree/branch per
# model (e.g. heavy parallelism or per-model archival).
#
#   ./run_model.sh --list                        # list available models
#   ./run_model.sh google/gemini-2-5-flash       # 1 run, 10 turns
#   ./run_model.sh xai/grok-3-mini 5             # 5 runs, 10 turns
#   ./run_model.sh openai/gpt-5-nano 3 30        # 3 runs, 30 steps
#   ./run_model.sh --all 2                       # every bundled model, 2 runs each
#
# Needs the model's API key in the env (or ./.env): GEMINI_API_KEY (google),
# XAI_API_KEY (xai), OPENAI_API_KEY (openai), TOGETHER_API_KEY (qwen/deepseek/kimi),
# ANTHROPIC_API_KEY (anthropic). On macOS prepend MUJOCO_GL= for physics-only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$HERE/configs/models"

list_models() {
  find "$MODELS_DIR" -name '*.yaml' | sed "s|$MODELS_DIR/||; s|\.yaml\$||" | sort
}

usage() {
  echo "Available models:"; list_models | sed 's/^/  /'
  echo
  echo "Usage: $0 MODEL [N_RUNS=1] [MAX_STEPS=10]"
  echo "       $0 --all [N_RUNS=1] [MAX_STEPS=10]"
}

run_one() {  # model  max_steps
  local model="$1" max="$2"
  if [ ! -f "$MODELS_DIR/$model.yaml" ]; then
    echo "  ERROR: unknown model '$model' (try --list)" >&2; return 1
  fi
  ( cd "$HERE" && ./run_agent.sh --model "$model" --max-steps "$max" )
  local last; last="$(ls -td "$HERE"/runs/*/ 2>/dev/null | head -1)"
  if [ -n "$last" ] && [ -f "$last/summary.json" ]; then
    python -c "import json,sys; s=json.load(open(sys.argv[1])); \
print(f\"  -> {sys.argv[2]}  steps={s['steps']} finished={s['finished']} deliverables={s['deliverables_present']}\")" \
      "$last/summary.json" "$last"
  fi
}

case "${1:-}" in
  ""|--list|-l|-h|--help) usage; exit 0 ;;
esac

if [ "$1" = "--all" ]; then
  shift; N="${1:-1}"; MAX="${2:-10}"
  for m in $(list_models); do
    for i in $(seq 1 "$N"); do echo ">>> $m — run $i/$N (max-steps $MAX)"; run_one "$m" "$MAX"; done
  done
else
  model="$1"; N="${2:-1}"; MAX="${3:-10}"
  for i in $(seq 1 "$N"); do echo ">>> $model — run $i/$N (max-steps $MAX)"; run_one "$model" "$MAX"; done
fi
