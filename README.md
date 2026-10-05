# ArtifactArena Design Lab Harness (DLH)

[Kushagra Tiwary](https://www.kushagratiwary.com/)<sup>\*1,2</sup>,
[David Mayo](http://david-mayo.com/)<sup>\*1</sup>,
[Nikhil Behari](https://nikhilbehari.github.io/)<sup>1,2</sup>,
Xiangzhou Sun<sup>1</sup>,
[Abdulrahman Alabdulkareem](https://arkareem.com/)<sup>1</sup>,
[Isaac Galatzer-Levy](https://med.nyu.edu/faculty/isaac-r-galatzer-levy)<sup>3</sup>,
[Boris Katz](https://people.csail.mit.edu/boris/boris.html)<sup>1</sup>, and
[Brian Cheung](https://briancheung.github.io/)<sup>†1,4</sup>

<sup>1</sup>[InfoLab, MIT CSAIL](https://www.csail.mit.edu/research/infolab) ·
<sup>2</sup>[Camera Culture, MIT Media Lab](https://www.media.mit.edu/groups/camera-culture/overview/) ·
<sup>3</sup>[Department of Psychiatry, NYU Grossman School of Medicine](https://med.nyu.edu/departments-institutes/psychiatry/) ·
<sup>4</sup>[Discovery Lab, UCSF](https://discolab.org/)

<sup>\*</sup>Equal contribution · <sup>†</sup>Corresponding PI ·
Correspondence: ktiwary@mit.edu, dmayo2@mit.edu, bcheung@ucsf.edu

**ArtifactArena evaluates models by what they build in the physical world.** Each model
designs a complete robot artifact (a MuJoCo body in MJCF XML and a Python controller),
and the artifacts compete head-to-head in the *Last Bot Standing* game. This repository
is the paper's **Design Lab Harness**: the model works as an open-ended engineer in a
*Design Lab*, a sandboxed workspace holding the simulator and verifier source, the task
documentation, and shell and Python execution. With a budget of 10 API calls it chains
tool calls to build, simulate, analyse match feedback and save candidate artifacts;
a round robin among its saved candidates picks the artifact it enters in the tournament.

The simulator, verifier, Sampling Harness (SH) and Verifier-Grounded Refinement Harness
(VGH) are in [harness-public](https://github.com/ArtifactArena/harness-public).

🌐 **Website & leaderboards:** [artifactarena.ai](https://artifactarena.ai) ·
📄 **Paper:** [ArtifactArena: Evaluating Models by What They Build in the Physical World](https://artifactarena.ai/paper) ([PDF](https://artifactarena.ai/paper/ArtifactArena.pdf)) ·
🎮 **Play:** [artifactarena.ai/play](https://artifactarena.ai/play)

## 🌍 Global Call for Robot Artifacts

ArtifactArena is harness-agnostic: only the artifact matters. We accept robots designed
entirely by models and robots built through human–AI collaboration. Build a bot — by
hand, with your favorite AI, or any mix of the two — try it in the
<a href="https://artifactarena.ai/play" target="_blank">Play arena ↗</a>, press **Verify**
to run the tournament's qualification checks, and submit it to the **monthly
competition**:

👉 **[Submission form](https://docs.google.com/forms/d/e/1FAIpQLSeJnDxoboUVzfygIj5Nwm_jCFavYs0ED0o65EcCeAXzPS3ubQ/viewform)**

## First-Time Setup

```bash
# 1) Environment (Python 3.10, MuJoCo 3.10.0)
git clone https://github.com/ArtifactArena/design-lab-harness.git && cd design-lab-harness
conda create -y -n design-lab python=3.10 && conda activate design-lab
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=..." > .env          # the key for the model you drive (see below)

# 2) Self-check: engine, verifiers, tools and parser (no API calls)
MUJOCO_GL= python -m pytest tests -q

# 3) Linux only: prove the tool sandbox holds before spending API budget
python scripts/check_isolation.py --physics

# 4) One run: 10 model turns, then the round-robin selection
./run_agent.sh --model anthropic/claude-sonnet-5-high --max-steps 10

# 5) The full roster: 24 models x 5 runs x 10 turns, one git worktree per model
./run_big_experiment.sh
```

Notes:

- **Model runs need Linux** with unprivileged user, mount, PID and network namespaces
  plus `unshare`, `pivot_root` and `setpriv`. Startup refuses to run otherwise; there
  is no unsandboxed fallback. The test suite runs anywhere, including macOS.
- `MUJOCO_GL`: the loop renders no video. Leave it unset on a Linux/EGL box (the
  wrapper picks `egl`); set it empty (`MUJOCO_GL=`) on macOS or a box without EGL.
- A model is a path under `configs/models/` without `.yaml`; `./run_model.sh --list`
  prints them all. Each provider reads its key from the environment or `./.env`:

  | Provider | Key |
  | --- | --- |
  | Anthropic | `ANTHROPIC_API_KEY` |
  | OpenAI | `OPENAI_API_KEY` |
  | Google | `GEMINI_API_KEY` |
  | xAI | `XAI_API_KEY` |
  | DeepSeek, Kimi, MiniMax, Qwen, GLM (via Together) | `TOGETHER_API_KEY` |

- [RUNNING.md](RUNNING.md) explains how each provider's reasoning and output settings
  reach the wire.

## How a Run Works

```
             ┌──────────── API driver (host) ─────────────┐
             │  prompt = sampling prompt + tool protocol   │
  model ◄────┤  state  = draft + notes + saved bots +      │
  reply ────►│          journal + last tool results        │
             └──────────────┬──────────────────────────────┘
                            │ one fresh Linux namespace per tool call
             ┌──────────────▼──────────────────────────────┐
             │  /workspace (rw)   /engine, /code (ro)       │
             │  no network · no credentials · no caps       │
             └─────────────────────────────────────────────┘
```

- **The prompt is the sampling prompt, verbatim.** The system prompt starts with
  `configs/rules/sampling_prompt.md` exactly as the sampling harness sends it (with the
  MuJoCo version filled in), followed by the tool protocol in `arena_kit/agent.md`,
  which replaces the one-reply answer format with JSON tool calls.
- **State is refreshed, not accumulated.** Each turn the prompt is rebuilt from durable
  state: the current draft, `notes.md`, the saved bots, a one-line-per-step journal,
  and the results of the last turn. Context never grows with the run; the model's
  memory is the journal plus whatever it writes to `notes.md`.
- **A turn is one model call.** A reply may carry several fenced `json` tool calls;
  they run in order and every result comes back next turn. Tool calls on the final
  turn still run, but their results are never seen. `finish` ends the run early once
  a bot has been saved.
- **A broken reply runs nothing.** A reply cut off by the output limit, or one with
  any malformed tool call, executes none of its tools, still uses up its turn, and the
  next turn says so (`agent.response_actions`).
- **Match telemetry stays on disk.** `run_match` returns a downsampled view and a
  `match_id`; the model interrogates the full replay with `analyze_match(match_id,
  code)`, shipping Python that runs over the match and returns only its answer.
- **A slow tool is not a lost turn.** A tool that hits its wall-clock limit returns
  the seeds that finished before the limit, marked partial, and the run continues.

## Tools

| Group | Tools |
| --- | --- |
| Workspace | `list_dir`, `read_file`, `grep`, `write_file`, `run_python`, `run_bash` |
| Checks | `does_bot_verify` (static MJCF + controller validation), `does_bot_qualify` (validation + three 20 s trials against the qualification block) |
| Matches | `run_match`, `list_matches`, `get_match`, `analyze_match` |
| Physics | `diagnose_physics`, `probe_obs` |
| Candidates | `save_bot`, `list_bots`, `get_bot`, `finish` |

A **bot ref** is `draft` (the working `robot.xml` + `controller.py`), `stationary` (the
qualification block), or the name of a bot saved with `save_bot`. `run_match` takes a
red and a blue ref (the same ref on both sides is self-play), verifies both bots
first, and does not require qualification. `save_bot` records the sampling prompt's
required outputs — `name`, `design_strategy`, `hardware_plan`, `combat_plan` — plus a
short `summary` in the bot's `bot_artifact.json`.

## Selection

When the turns run out (or the model calls `finish`), every saved candidate enters a
round robin: three seeds per pairing in each colour, 300-second matches, a draw worth
half a win, ranked by Bradley–Terry strength. With two or more candidates each is
re-verified first, and one that no longer verifies forfeits its games. An earlier
candidate can win. A manual `submit`, an `out/` folder or an unsaved draft cannot
override the selection, and a run with no saved candidate reports a selection failure
instead of shipping an untested draft.

## Isolation

The API client runs on the host; every tool call runs in a new sandbox
(`linux_sandbox.py` → `sandbox.sh` → `tool_worker.py`):

- **Namespaces.** New user, mount, PID and network namespaces. The network namespace
  has only loopback, so there is no outbound connection.
- **Filesystem.** A fresh root holds the run's workspace (read-write), a frozen copy of
  the engine and tool code taken at run start (read-only), the Python environment
  (read-only), the system libraries (read-only) and private `/tmp` and `/dev/shm`
  tmpfs mounts. The host root is detached with `pivot_root` and unmounted.
  Sibling runs, `.git`, `.env`, API keys, test fixtures and other models' bots are not
  mounted.
- **Privileges.** All capabilities are dropped, `no_new_privs` is set, and the
  environment is cleared before model code runs. A run refuses to start unless a
  probe inside the sandbox confirms all three from `/proc/self/status`.
- **The host never trusts the workspace.** The driver writes its request with
  `O_NOFOLLOW`, refuses a result file that is a symlink, and reads the draft,
  `notes.md`, saved bots and timeout checkpoints only when they resolve inside the
  workspace and are regular files.
- **Final selection runs in the sandbox too**, after the model's last turn, in a
  directory the model never saw.

The `Tools` class and the standalone scripts in `arena_kit/` are trusted developer
interfaces, not sandbox entry points: run models through `run_agent.sh` and the
launchers. Each run's frozen snapshot and sandbox log live in `runs/<id>-runtime/`. Before spending
API budget on a new host, run the self-check below and one short run.

## DLH in the Paper

| | |
| --- | --- |
| Models | 21 frontier models |
| Runs | 3 independent runs per model |
| Budget | 10 API calls per run, high reasoning effort where the provider offers it |
| Prompt | `configs/rules/sampling_prompt.md` + `arena_kit/agent.md`, as in this repository |
| Engine | MuJoCo 3.10.0, the engine in `mjarena/` |
| Selection | the round robin above: three seeds per pairing from each side, 300 s matches |

The paper's runs were driven by `tournament/run.py`, which calls each provider's
native API (OpenAI Responses, Gemini, Anthropic Messages, chat completions for the
rest; `provider_adapter.py`) instead of DSPy/litellm, checkpoints every turn and tool
call, and hands selection to an external coordinator. It ran on our compute cluster
and keeps its directory layout, so it is a record of what ran; use `run_agent.sh` for
new runs. Each run's selected artifact was validated, qualified, and entered the
Top-3 Round of the tournament alongside the SH and VGH artifacts. The artifacts, their reasoning and
their design histories are in the Bot Zoo at [artifactarena.ai](https://artifactarena.ai).

## Launchers

| Script | What it runs |
| --- | --- |
| `run_agent.sh` | One run in this checkout: loads `.env`, sets `MUJOCO_GL`, `--model` + `--max-steps` |
| `run_model.sh` | N sequential runs of one model (or `--all` bundled models) in this checkout |
| `run_experiment.sh` | One git worktree per model (`../open-ended-run-<model>/`), `RUNS_PER_MODEL` runs each |
| `run_big_experiment.sh` | `run_experiment.sh` over the September-2026 roster: 24 models x 5 runs x 10 turns |
| `run_parallel.sh` | One worktree per (model, iteration), all launched concurrently (`MAX_PARALLEL` caps it) |
| `tail_run.sh` | Follow the newest run's journal live (`--full` for replies and tool results) |

## Where Results Land

```
runs/<timestamp>-<id>/
  final/                 # the selected bot: robot.xml + controller.py + bot_artifact.json
  bots/<name>/           # every saved candidate, with its design intent
  matches/<match_id>/    # sparring matches: full replay + result
  selection_matches/     # the final round robin
  journal.jsonl          # one line per tool call: action → outcome
  transcript.jsonl       # every prompt, reply and tool result
  summary.json           # turns used, finish summary, selection ranking
runs/<timestamp>-<id>-runtime/
  engine/ code/ prompt/  # the frozen snapshot the sandbox mounted
  tools.log              # sandbox stdout/stderr
```

## Layout

```
design-lab-harness/
  agent.py              # the state-refreshed tool loop (API driver + tool dispatch)
  linux_sandbox.py      # per-call namespace sandbox: snapshot, launch, result checks
  sandbox.sh            # the namespace + pivot_root + setpriv jail
  tool_worker.py        # sandbox entry point: probe, one tool call, or final selection
  tool_results.py       # per-seed progress checkpoints and partial results on timeout
  mjarena/              # the ArtifactArena engine, vendored (see below)
  configs/              # rules, tournament configs, model configs
  arena_kit/            # the pristine Design Lab copied into every run
    agent.md            #   tool protocol appended to the sampling prompt
    harness_lib.py      #   verify / qualify / match / analyse / diagnose / bot library
    docs/               #   sampling_prompt.md, rules.yaml, materials, MJCF + obs sections
    reference/          #   a copy of the engine source for the model to read
    assets/             #   the qualification block
  tournament/           # the driver the paper's DLH runs used, kept as a record
  scripts/              # check_isolation.py, smoke_models.py, render_match.py
  tests/                # engine, tool, sandbox-guard and parser regressions
```

## Tests

```bash
MUJOCO_GL= python -m pytest tests -q
```

The suite covers the vendored engine (inactivity, runtime size limit, controller
sandbox, controller state isolation, MJCF normalisation, verifier messages), the tool
layer (verification, qualification, matches, the bot and match libraries,
`analyze_match`, `diagnose_physics`), the API driver's file guards, the JSON-action
parser on real model output, the roster, and the engine snapshot. A live model smoke
test runs when `GEMINI_API_KEY` is set (it needs Linux, like every model run). On Linux,
`scripts/check_isolation.py` exercises the real sandbox (`--physics` adds a
qualification run inside it), and `STEPS=3 python scripts/smoke_models.py <model>`
checks that a model drives the loop.

## The Bundled Engine

`mjarena/` and `configs/` are the ArtifactArena engine, vendored from harness commit
`62b916b` (2026-09-17); `upstream_environment.json` pins the SHA-256 of every vendored
file, and `tests/test_upstream_environment.py` fails if one drifts. Three engine files
carry DLH's recording and progress hooks (`red_surface_distances` in `episode.py` /
`types.py`, per-seed progress in `dspy_core.py`); the manifest declares each under
`tournament_patches` with its reason. None changes the physics or the rules. 8 more files differ from
the vendored commit in wording only (comments, display labels, documentation); the manifest lists them under
`release_edits`.

## Citation

```bibtex
@article{artifactarena2026,
  title  = {ArtifactArena: Evaluating Models by What They Build in the Physical World},
  author = {Tiwary, Kushagra and Mayo, David and Behari, Nikhil and Sun, Xiangzhou and
            Alabdulkareem, Abdulrahman and Galatzer-Levy, Isaac and Katz, Boris and Cheung, Brian},
  year   = {2026},
  url    = {https://artifactarena.ai}
}
```

## License

MIT — see [LICENSE](LICENSE).
