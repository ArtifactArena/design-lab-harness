# Running models on the harness

Model runs require the Linux namespace sandbox described in README.md. All entry points use it; worktrees separate Git changes but are not the security boundary.

## 1. `run_model.sh` — run a model in this checkout (no worktrees)

The everyday way. Each run writes to its own `runs/<timestamp>/`, so sequential runs
never clobber each other.

```bash
./run_model.sh --list                      # list available models
./run_model.sh google/gemini-2-5-flash     # 1 run, 10 turns
./run_model.sh xai/grok-3-mini 5           # 5 runs, 10 turns
./run_model.sh openai/gpt-5-nano 3 30      # 3 runs, 30 steps
./run_model.sh --all 2                     # every bundled model, 2 runs each
```

After each run it prints the run dir + `steps / finished / deliverables`.

## 2. `run_agent.sh` — one run, lowest level

```bash
./run_agent.sh --model xai/grok-3-mini --max-steps 10
./run_agent.sh --max-steps 10              # default model (gemini-2.5-flash)
```

## 3. `run_experiment.sh` — worktree per model (isolation / parallelism)

```bash
./run_experiment.sh google/gemini-2-5-flash xai/grok-3-mini
# env: RUNS_PER_MODEL=5  MAX_STEPS=10  BASE_BRANCH=<current branch>
```

## Do I need worktrees?

**No — not for a single run or a few sequential runs.** `run_agent.sh` /
`run_model.sh` are enough; every run already lands in a unique `runs/<ts>/` dir.

Use `run_experiment.sh` (worktrees) only when you want:
- to run several models **in parallel** without their git state interfering, or
- a **separate branch per model** for clean archival/comparison.

A git worktree is a second working copy of the repo on its own branch, sharing the
same `.git`. `run_experiment.sh` makes one per model under `../open-ended-run-<model>/` so
each model's runs + branch are isolated.

## Models & API keys

**Study roster (September 2026, 24 models):** `configs/models/run-roster-2026-09.yaml`,
identical to harness-public's. `./run_big_experiment.sh` and `./run_parallel.sh`
launch it by default; `tests/test_big_experiment_roster.py` keeps their lists equal to
the YAML.

**Smoke configs:** the cheaper models used to check that a provider drives the loop
(`google/gemini-2-5-flash` — the default when no `--model` is given —
`google/gemini-2-5-flash-lite`, `google/gemini-3-flash-{medium,high}`,
`xai/grok-3-mini`, `openai/gpt-5-nano`) and the June-2026 roster's configs stay
bundled. Add more under `configs/models/<provider>/<name>.yaml`.

### Reasoning effort & token caps are provider-specific — read this before comparing

> These settings apply to `run_agent.sh` and the launchers here, which call providers
> through DSPy/litellm. The final tournament's open-ended runs used the native-API
> driver in `tournament/` with per-run `config.json` settings instead (see README).


Configs are read by `configure_lm` and forwarded to litellm, which maps each field to
the provider's real mechanism (`litellm.drop_params=False`, so a param the provider
doesn't accept **raises** — it is never silently dropped). Verified mappings:

| provider | `max_tokens` →  | `reasoning_effort` → |
| --- | --- | --- |
| google (gemini 3) | `max_output_tokens` | `thinkingConfig.thinkingLevel: <effort>` |
| google (gemini 2.5) | `max_output_tokens` | `thinkingConfig.thinkingBudget` (token budget) |
| openai (gpt-5) | `max_completion_tokens` | `reasoning_effort` (native) |
| xai (grok) | `max_tokens` | `reasoning_effort` (native) |
| together (qwen) | `max_tokens` | not supported → use `extra_body.chat_template_kwargs` |
| **anthropic (claude)** | `max_tokens` | ⚠️ **do NOT use `reasoning_effort`** |

**Anthropic gotcha:** in litellm 1.80.11 (the version pinned in `requirements.txt`),
`reasoning_effort: high` on *any* Claude
model maps to a fixed, weak `thinking.budget_tokens: 4096` — far below real high
effort. Use Anthropic's **native** budget instead, which passes through correctly:

```yaml
model: anthropic/claude-opus-4-7
max_tokens: 128000
output_config:
  effort: high          # provider-managed effort — NOT reasoning_effort
```

The bundled `anthropic/*-high` configs already do this. Across providers, "high" is
each provider's own tier (e.g. Gemini-2.5 medium = a 2048-token budget, Gemini-3
medium = a level), so reasoning effort is matched by *tier*, not by identical budgets.

Each provider needs its key in the env or `./.env`:

| provider | env var |
| --- | --- |
| google (gemini) | `GEMINI_API_KEY` |
| xai (grok) | `XAI_API_KEY` |
| openai | `OPENAI_API_KEY` |
| qwen / deepseek / kimi (Together) | `TOGETHER_API_KEY` |
| anthropic (claude) | `ANTHROPIC_API_KEY` |

Smoke-check that a model drives the loop before a long run:

```bash
MUJOCO_GL= STEPS=3 python scripts/smoke_models.py xai/grok-3-mini
```

> **`MUJOCO_GL`:** leave unset on a Linux/EGL box (the wrapper picks `egl`); on
> macOS prepend `MUJOCO_GL=` (empty) for physics-only.

## Where results land

```
runs/<timestamp>-<id>/
  final/                  # the RESOLVED submission: robot.xml + controller.py + bot_artifact.json
                          #   (round-robin winner among saved candidates)
  bots/<name>/            # saved designs (robot.xml, controller.py, bot_artifact.json w/ design intent)
  matches/<match_id>/     # match_data.json (full replay) + match_result.json
  journal.jsonl           # one line per step (action → outcome)
  transcript.jsonl        # full per-step prompts + replies + tool results
  summary.json            # steps, finished, deliverables_present, submission{how,name,ranking}
```
