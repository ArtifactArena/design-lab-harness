#!/usr/bin/env bash
# Launch the state-refresh bot-building agent.
#
# Loads .env (API keys), sets a headless GL backend, and runs the loop against the
# SELF-CONTAINED bundled mjarena (design-lab-harness/mjarena). The driving model is the
# bundled Gemini Flash by default; override with `--model <name>` (a path under
# configs/models/, e.g. google/gemini-2-5-flash) or by setting MH_MODEL_CONFIG.
#
#   ./run_agent.sh --max-steps 10
#   ./run_agent.sh --model openai/gpt-5.5 --max-steps 10
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # design-lab-harness/ (self-contained root)

# .env (optional) — e.g. GEMINI_API_KEY=... ; checked here then one level up.
for envf in "$HERE/.env" "$HERE/../.env"; do
  if [ -f "$envf" ]; then set -a; . "$envf"; set +a; break; fi
done

# --model <name> -> MH_MODEL_CONFIG; pass the rest through to agent.py.
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --model) export MH_MODEL_CONFIG="configs/models/$2.yaml"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

export MUJOCO_GL="${MUJOCO_GL-egl}"   # unset -> egl; explicitly empty -> physics-only
export PYTHONPATH="$HERE:${PYTHONPATH:-}"   # bundled mjarena/ is importable
export ARENA_REPO_ROOT="$HERE"              # resolve configs/assets from the bundle

exec python "$HERE/agent.py" ${ARGS[@]+"${ARGS[@]}"}
