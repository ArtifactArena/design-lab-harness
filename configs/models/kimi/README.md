# Kimi K3 on the Open-Ended Harness — config reference & operating manual

Single source of truth for running `kimi/kimi-k3-high` here. Written
2026-08-10 after the full SH condition (50 samples + 1081-matchup RR) and
~20h of ARH refinement runs on this exact config, so every claim below is
empirical, not guessed.

---

## 1. The config, line by line (`kimi-k3-high.yaml`)

```yaml
model: together_ai/moonshotai/Kimi-K3
max_tokens: 1048576
num_retries: 10
extra_body:
  reasoning_effort: high
```

| Field | Value | EXACTLY why |
|---|---|---|
| `model` | `together_ai/moonshotai/Kimi-K3` | Together AI serving of Moonshot's Kimi K3 (2.8T MoE, 1M context, thinking always on). litellm provider prefix `together_ai/` + Together's endpoint id `moonshotai/Kimi-K3` (verified against Together's live `/v1/models`). Needs `TOGETHER_API_KEY`. |
| `max_tokens` | `1048576` | K3's **native max output** (1M). Roster convention pins each model at its native max (Claude/GPT configs sit at their 128000). Probe-verified 2026-08-09: Together accepts 1048576 without error or hang. It is a **cap, not a target** — cost does not scale with it. Do NOT leave it null: litellm has no model_info entry for K3 and null raises at setup. **K3's thinking tokens count against this cap** (like Anthropic models), another reason it's set at max. |
| `num_retries` | `10` | Roster standard. litellm re-sends on transient API failures. NOTE: retries fire only when a call *errors* — they do NOT rescue silent hangs (see §3). |
| `extra_body.reasoning_effort` | `high` | K3 levels are low/high/max (API default: max). `high` matches the study convention of each provider's "high" tier (same choice as kimi-k2.7-code-high, glm-5.2-high). Must go via `extra_body` — litellm rejects `reasoning_effort` as a top-level param for the together provider (same as GLM/K2.7). Passed through raw; verified end-to-end: responses carry `reasoning_content`. |

### Fields that are deliberately ABSENT

| Missing field | Why it must stay missing |
|---|---|
| `temperature`, `top_p` | **K3's sampling params are fixed server-side** (Together: "The sampling parameters are fixed and you should omit them from requests"). Every other kimi/qwen/glm config sets them — K3 is the exception. Do not "fix" this by copying them in. Sample diversity across identical prompts comes from server-side sampling and is confirmed real (50 distinct designs from 50 identical zero-shot prompts). |
| `timeout` | Not set in the yaml on purpose: the harness default is 100 min (`NO_LLM_TIMEOUT_S` in `mjarena/dspy_core.py`, the same in every arena harness). Legitimate K3 calls run up to ~80 min (see §2), so do not set anything tighter; at 100 min a hung connection is aborted and retried by litellm, which is what makes unattended runs self-heal. |
| `model_type`, `output_config` | Anthropic/OpenAI-specific knobs; not applicable to together provider. |

### Cost (Together, as of 2026-08)

$3.00/M input ($0.30/M cache-hit) · **$15.00/M output — thinking bills as
output.** A 50-step OEH run ≈ $15-40; budget accordingly.

---

## 2. Empirical call-time profile (n≈350 calls, reasoning_effort=high)

| Statistic | Value |
|---|---|
| median | **7.2 min** |
| mean | 14.6 min |
| min / max observed | 2.5 min / **~80 min (completed successfully)** |

Heavily right-skewed: most calls land in 3-10 min, but a long-thinking tail
to 80+ min is NORMAL. **Do not kill a call just because it is slow.** In the
OEH loop this means single steps can stall the journal for over an hour
while still being healthy.

## 3. The hang failure mode (rare — ~1 in ~350 calls)

Signature: Together's server side dies but leaves the TCP socket
ESTABLISHED. The client blocks on read **forever** — no error, so
`num_retries` never triggers.

Detect: run log **silent > 2 h** AND the python process at **0.0% CPU** AND
one ESTABLISHED socket to Together (`lsof -p <pid> | grep TCP`).

Remedy, in order:
1. The 100-minute harness default aborts and retries the hung call by itself — nothing to do unless you tightened `timeout`.
2. Drop just the socket (`sudo tcpdrop <src-ip> <src-port> <dst-ip> 443`):
   client sees a reset, litellm retries, **all in-process state survives**.
3. Kill the process and re-run that unit of work (for OEH: that run;
   completed runs are unaffected).

## 4. Known K3-on-arena behavior (from the SH study, same config)

- **~6% of zero-shot generations forfeit** (invalid/failed output) — the
  harness writes `<!-- FORFEIT -->` placeholders; treat those as failed
  samples, never as bots.
- **~22% of its bots' games end in physics instability (qacc)** — K3 likes
  aggressive/marginal morphologies; expect verifier pushback in the loop.
- **Draws are a non-issue** (0.4% of 5405 RR games) — its designs engage
  (78% of games end in ring-out).
- Refinement works: ARH lineages climbed from ~2.x to 4.1-4.3 qualification
  score within ~15 commits, with coherent iterative design ("Mk.XVIII →
  Mk.XIX → Mk.XX" naming).

## 5. Running it on THIS harness

```bash
# once
echo "TOGETHER_API_KEY=sk-..." >> .env
# macOS laptop: physics-only  |  Linux/EGL box: leave MUJOCO_GL unset/egl
./run_model.sh --list                    # confirm kimi/kimi-k3-high resolves
./run_model.sh kimi/kimi-k3-high 5 50    # study standard: 5 runs x 50 steps
```

- Each run lands in `runs/<timestamp>/` (journal.jsonl + workspace +
  final/). Runs are independent — a killed run never contaminates others.
- The loop is state-refreshed (journal-based), so one slow/hung step wastes
  that step's call only.
- Expect a 50-step run to take **6-12 h wall-clock** at this call profile.
- The vendored `mjarena.dspy_core.configure_lm` reads this YAML directly, the same
  loader the ArtifactArena harness uses.
