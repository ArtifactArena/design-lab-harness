{{sampling_prompt}}

## Open-ended design lab

You have access to a design lab where you can build robots, run experiments, inspect the results, and revise your designs. You decide how to use the lab to create the strongest robot you can for the tournament described above.

You may read the simulation and design-lab source code, write and edit robot XML and controller Python, validate designs, run qualification matches, run matches between bots you create, inspect recorded observations and motion, and write your own analysis code. You may improve an existing design, explore alternatives, or start over.

The rules, robot requirements, controller interface, and observation descriptions above apply to every bot you build. The tool-call response format below replaces the sampling prompt's six-section response format. Always use the tool-call format on every turn, including the final turn. Supply the required name and design explanations as arguments to `save_bot`, and write the required robot XML and controller Python to workspace files through tools, as described below.

## Model-turn budget

You have {{model_turn_budget}} model turns to build and improve your robot.
Current model turn: {{model_turn_number}} of {{model_turn_budget}}.
Model turns remaining after this turn: {{model_turns_remaining_after_this}}.

A model turn is one response from you, corresponding to one model API call. You may request multiple tool calls in a turn. A turn does not have to produce a new bot. Use your turns to explore designs, run experiments, inspect results, and improve your robot.

## What you receive each turn

Each turn starts with a fresh prompt containing the instructions above and your current workspace state:

- Your current draft: `robot.xml` and `controller.py`.
- Your notes in `notes.md`.
- The latest validation and qualification status, including whether the draft has changed since it was checked.
- The bots in your library, with their names, status, and summaries.
- A journal summarizing your previous tool calls, their outcomes, and your notes about them.
- The results of the tool calls from your previous turn.

You do not receive the full prior conversation. Keep useful findings, hypotheses, and plans in `notes.md`. Long files and results may be shortened to fit the prompt, and older journal entries may be omitted. Full files and match recordings remain in your workspace; use the tools to inspect them when needed.

On the first turn, there is no previous bot or experiment feedback. Start by creating a design or investigating the lab.

## How to call tools

End each response with one or more fenced JSON tool calls, including when you deliver a complete bot or finish your work. You may explain your intended actions before the calls, but a six-section robot answer does not replace the required tool calls. Each call contains `tool`, `args`, and a short `note` explaining its purpose. For example:

```json
{"tool":"list_dir","args":{"path":"reference"},"note":"Locate the simulation source"}
```
```json
{"tool":"read_file","args":{"path":"notes.md","offset":0,"limit":100},"note":"Review recorded findings"}
```

Multiple calls in one response execute in the order you write them. Their results are returned together on the next model turn. You can write `robot.xml`, write `controller.py`, and then call `does_bot_qualify` in a single response, because those calls can be specified without first seeing their results.

Tool calls on the final turn still execute, but you will not receive their feedback.

Use a later turn when deciding a call requires an earlier call's result. For example, you can run a match on one turn, then use its returned `match_id` and outcomes to decide what analysis to run on the next. Similarly, you can revise your bot using the evidence returned from the previous turn and request new tests for that revision in the same response.

A model turn is one API call. One API call may request any number of tool calls, and those tool calls do not count against the {{model_turn_budget}}-call budget. Reading and reacting to their results requires another API call.

## Working draft and complete bot outputs

Build and edit your working draft in `robot.xml` and `controller.py` at the workspace root. The XML and Python must follow the requirements above and work together as one bot.

When you output a complete bot, call `save_bot` with `source: "draft"` to capture its XML and controller together under a distinct name. This tool records a complete version of your bot so later edits to the draft do not replace it.

Put the sampling prompt's required `name`, `design_strategy`, `hardware_plan`, and `combat_plan` outputs in the corresponding arguments of `save_bot`. Each field must satisfy its full requirements in the sampling prompt above, including the requested explanations, justifications, calculations, and hardware-to-strategy mapping. Also include a short `summary` of the bot and what you intended to improve. Explanations outside the tool call do not replace these required arguments.

The sampling prompt's `robot_xml` and `controller_code` outputs belong in the workspace files `robot.xml` and `controller.py`. Write their complete contents through tools before calling `save_bot`; `save_bot` captures those files from the draft. All XML and Python requirements in the sampling prompt still apply.

Use a new name for each complete revision so every bot you output remains available for the selection tournament. You may leave work in progress in the draft while you investigate or edit; you do not need to output a complete bot on every turn. Describing a bot in prose alone does not create its XML and controller files.

You can use `get_bot` to load an earlier version into the draft and develop another branch. Your bot library lets you compare alternatives and return to earlier designs.

## Bot references

Tools that take a bot reference accept:

- `draft`: your current `robot.xml` and `controller.py`.
- `stationary`: the stationary baseline block, the only built-in opponent.
- A bot name: a complete version you previously recorded with `save_bot`.

You can run matches between your designs or against the stationary block. A match with the same bot on both sides is self-play.

## Available tools

You may use multiple tools per model turn.

| Tool | Arguments | What it does |
| --- | --- | --- |
| `list_dir` | `{"path":"."}` | List a directory in your workspace, including bundled source directories. |
| `read_file` | `{"path":"robot.xml","offset":0,"limit":200}` | Read a workspace file. `offset` and `limit` are measured in lines and let you page through large files. |
| `grep` | `{"pattern":"freejoint","path":"reference"}` | Search files for a pattern. |
| `write_file` | `{"path":"notes.md","content":"..."}` | Write a workspace file, including your robot, controller, notes, or experiment scripts. This replaces the file's contents. |
| `does_bot_verify` | `{"ref":"draft"}` | Validate morphology and perform static controller checks without running a match against the block. Returns `verification_passed`, `morphology_passed`, `controller_passed`, and `feedback`. |
| `does_bot_qualify` | `{"ref":"draft"}` | Validate the bot and run its qualification trials against the block. Returns `validation_passed`, `qualification_passed`, and `feedback`. |
| `run_match` | `{"red":"draft","blue":"stationary","n_seeds":3}` | Verify both bots, then run matches. Qualification is not required. If verification fails, return the errors without running matches. Otherwise return results for every seed under `seeds`, a sampled replay under `replay`, and a unique `match_id`. Full recordings remain on disk. |
| `list_matches` | `{}` | List matches you have run. |
| `get_match` | `{"match_id":"..."}` | Retrieve a previous match's per-seed results and sampled replay without running it again. |
| `analyze_match` | `{"match_id":"...","code":"def analyze(match_data):\n    return {key: seed['winner'] for key, seed in match_data.items() if key.startswith('seed_')}"}` | Run your own Python analysis over a match's full recording and return the function's result. `match_id` can be `"latest"`. See the analysis instructions below. |
| `diagnose_physics` | `{"ref":"draft"}` | Inspect the bot's gear, inertia, traction, and mass breakdown to investigate physical behavior or instability. |
| `probe_obs` | `{"ref":"draft"}` | Inspect the actual observation dictionary received by `policy_step`, including keys, shapes, and sample values. |
| `save_bot` | `{"name":"wedge_v1","source":"draft","summary":"low wedge","design_strategy":"...","hardware_plan":"...","combat_plan":"..."}` | Capture a complete bot in the library, along with validation/qualification results and your design explanations. |
| `list_bots` | `{}` | List the bots in your library, their status, and their summaries. |
| `get_bot` | `{"name":"wedge_v1"}` | Copy a library bot's XML and controller into the working draft, replacing the draft. |
| `run_python` | `{"code":"..."}` or `{"path":"experiment.py"}` | Run Python in your workspace. The simulation package `mjarena` is importable. |
| `run_bash` | `{"cmd":"ls -la"}` | Run a shell command in your workspace. |
| `finish` | `{"summary":"..."}` | End your work early if you are finished. Bot selection still uses the round robin described below. |

The `...` strings in tool examples stand for content you supply. Actual robot XML, controller Python, and executable analysis code must be complete.

## Chaining tools into experiments

Choose your own workflow. For example, you might write a draft, validate it, run qualification trials, inspect a physics issue, record a complete version, and compare it with another design. You can then analyze the recorded match, write your findings into `notes.md`, and revise either the morphology or controller.

Bot names and match identifiers let you connect experiments across turns. A bot recorded with `save_bot` can later be used in `run_match` or restored with `get_bot`. The `match_id` returned by `run_match` can later be used with `get_match` or `analyze_match`.

You decide which experiments to run and which evidence is useful. You may write your own scripts and compute your own measurements instead of following a prescribed optimization procedure.

## Writing match analysis

`analyze_match` runs Python over a full match recording stored on disk. Define a function taking exactly one argument, `match_data`. The harness calls the function and returns its return value to you. The full recording does not need to enter your prompt.

`match_data` contains one entry per seed, such as `seed_0`, `seed_1`, and `seed_2`. Each seed includes `winner`, `termination_reason`, `num_steps`, and `ring_radius`, together with per-step data such as:

- `red_positions` and `blue_positions`.
- `red_velocities` and `blue_velocities`.
- `red_edge_distances` and `blue_edge_distances`; positive values indicate being inside the ring.
- `contacts` and `contact_forces`.
- `red_tipping` and `blue_tipping`, with values from 0 to 1.
- `red_actions` and `blue_actions`.
- `qpos`.

You define the analysis and any derived metrics yourself. For example, this function returns the recorded outcome for each seed:

```python
def analyze(match_data):
    return {
        key: {
            "winner": seed["winner"],
            "termination_reason": seed["termination_reason"],
        }
        for key, seed in match_data.items()
        if key.startswith("seed_")
    }
```

Your analysis code runs as a Python module and may import packages such as NumPy. Analysis code runs in the lab; the robot controller still follows the controller restrictions in the sampling instructions above.

## Workspace and source access

Your workspace starts fresh for this run. The only supplied opponent is the stationary baseline block. You may use bots you create during this run; bots, controllers, match recordings, and results from other runs or models are not provided. Network access is disabled, and tools cannot access other run workspaces or the host's files outside the mounted lab environment.

Everything you need is available inside your run workspace:

- `robot.xml` and `controller.py`: your working draft.
- `notes.md`: your persistent notes.
- `docs/`: reference documentation.
- `reference/`: simulation and validation source code.
- `config.py` and `harness_lib.py`: lab configuration and tool implementation.
- `assets/`: baseline assets.
- `bots/` and `matches/`: bot versions and recorded experiments you create.

You may inspect these files with `read_file`, `list_dir`, and `grep`, and use `run_python` and `run_bash` for experiments. Work within your run workspace. Access to other runs or other models' creations is not provided. Paths that escape the workspace and destructive operations such as deleting files, formatting disks, or changing system privileges are refused.

## Final-bot selection

After all {{model_turn_budget}} model turns are complete, or you end your work with `finish`, all complete bots you output during those turns will compete in a round-robin selection tournament. Each complete bot consists of robot XML and its controller Python, recorded together through `save_bot`. Each bot will face every other bot you output during the {{model_turn_budget}} turns. The winner of that round robin will be selected to compete against bots created by other models in the final tournament described above.

You do not choose a single final entry with `submit`; the selection round robin chooses it. If you output only one complete bot, that bot is the only selection candidate. Make sure you output at least one complete bot before your work ends.

Your ultimate goal is for the selected bot to beat the bots created by other models in the final tournament.
