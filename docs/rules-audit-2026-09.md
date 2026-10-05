# Rules audit — September 2026

**Question this document answers:** does the prompt the models read state exactly
the rules the simulator and the verifier enforce — nothing more, nothing less?

**Deliverable of Task 13.** No behaviour was changed by the audit itself. Every
disagreement is reported, not fixed. The one exception was a wrong comment in an
asset file (`stationary_block_3d.xml` claimed 86.4 kg for what was then a 36 kg
block), corrected in the audit commit. Task 14 acted on one flag: the
baseline block was rebuilt from the material palette and is now a 342 kg plastic
slab, stated in the prompt (F-N11, row B39, `7890ccee`).

**Task 15 (2026-09-16) then acted on the rest.** Every flag listed under "New and
open" below has been decided and shipped, under one principle the user stated:
*the code has exactly the checks the prompt lists, nothing more, nothing less; one
big prompt per harness.* Rows changed by that work carry their new verdict, and each
flag line ends with **DECIDED and SHIPPED (Task 15, item X)**. The modular prompt
files are retired: `game.md`, `game-zeroshot.md`, `mjcf_syntax.yaml` and
`obs_schema.yaml` are deleted, their loaders with them, and `autoresearch_prompt_path`
/ `zero_shot_prompt_path` are required config keys — so §H1–§H4 are moot and marked
*retired*. Nothing else in this document has been re-audited since.

---

## Scope and method

Three rule sets were extracted and compared row by row:

| Source | Files |
|---|---|
| **Prompt** (what the model is told) | `configs/rules/autoresearch_prompt.md` (ARH, iterative), `configs/rules/sampling_prompt.md` (SH, zero-shot) |
| **Verifier** (what a submission must satisfy before it ever runs) | `mjarena/design_shop/rules/mj_validators.py`, `mjarena/design_shop/rules/hardware_rules.py`, `mjarena/design_shop/rules/software_rules.py`, `mjarena/design_shop/utils.py`, `mjarena/design_shop/policy_base.py`, `mjarena/design_shop/pipelines/{morphology,controller,unified}_*.py` |
| **Simulator** (what ends a round and who wins) | `mjarena/envs/sumo.py`, `mjarena/envs/utils.py`, `mjarena/runner/episode.py`, `mjarena/agents/runtime.py`, `mjarena/agents/policy_runtime.py` |

The two consolidated prompts are the **only** source of what the model is told:
the runs send them (`BuildConfig.autoresearch_prompt_path` /
`zero_shot_prompt_path`, now required keys). At audit time a modular fallback
(`game.md`, `mjcf_syntax.yaml`, `obs_schema.yaml`) still existed and its
divergences were recorded in **§H**; Task 15 (item R) deleted it. `rules.yaml`
stays: it is the validator's numbers, quoted verbatim into both prompts.

**Prompt line citations** are written `ARH:n / SH:m`. The two prompts' *rule*
text is byte-identical — a full diff shows only the iterative-development
narrative, the section numbering (eight sections vs six) and two non-rule
sentences (§H4). Line numbers diverge by a constant 25 from the Qualification
Round heading onward.

Reproduce the extraction with:

```bash
grep -nE "[0-9]+(\.[0-9]+)? ?(m|kg|s|seconds|N|rad)\b|must|lose|forbidden|not allowed|ignored|removed|fixed at|reject" \
  configs/rules/autoresearch_prompt.md configs/rules/sampling_prompt.md
grep -n "^def validate_\|errors.append" mjarena/design_shop/rules/mj_validators.py \
  mjarena/design_shop/utils.py mjarena/design_shop/rules/hardware_rules.py \
  mjarena/design_shop/rules/software_rules.py mjarena/design_shop/policy_base.py
grep -n 'termination_reason"\] =\|raise ValueError\|_inactivity\|_size_limits\|condim\|priority' \
  mjarena/envs/sumo.py mjarena/envs/utils.py mjarena/runner/episode.py
```

`tests/test_rules_audit_is_current.py` fails if a validator, a `rules.yaml`
check string or a `termination_reason` literal is added without a row here.

### Verdicts

- `MATCH` — the prompt states the rule and the code enforces it as stated.
- `SIM-ONLY (prompt silent)` — enforced, never stated to the model.
- `PROMPT-ONLY (not enforced)` — stated to the model, nothing checks it.
- `NUMBERS DIFFER` — same rule, different number.
- `WORDING DIFFERS` — same rule and numbers, but the prompt (or `rules.yaml`)
  describes the mechanism inaccurately or incompletely.

§E and §F additionally use the bookkeeping label `DEAD (enforces nothing)` for
code that never runs. That is **not** one of the five verdicts — a validator
that never executes imposes no rule and so cannot disagree with the prompt — but
every such entry is listed in **§Flags**.

### Out of scope (deliberately no row)

- The **100-minute LLM call timeout** (`tests/test_llm_timeout.py`) as a *game*
  rule: it is a build-harness budget and the model is never subject to it
  mid-match. It does get a row (C10) because the SH prompt asserts it does not
  exist.
- The **qualification combat score** (engagement / dominant_contact /
  displacement / destabilization / self_stability, `mjarena/eval/metrics.py`):
  the prompt states it is "a development measure, not tournament win rate"
  (ARH:32), which is exactly what it is.
- The **round robin among a model's own qualified commits**: documented in the
  retired `game.md:31` but implemented in neither repository; with that file
  deleted (Task 15, item R) no document now promises it.
- The **WASM runtime**, the **Bot Builder IDE**, and the `cube_predictor_v1`
  qualification score mode.
- Field-by-field observation semantics beyond the spot checks in **§D**: the
  prompt's `obs_schema` section is now the only observation contract
  (`obs_schema.yaml` was deleted in Task 15, item R); before that deletion the two
  texts were diffed programmatically and every field key and description string
  was identical.

---

## Verdict summary

Counts **as audited (Task 13)**, before Task 15 acted on the flags:

| Verdict | Count |
|---|---|
| `MATCH` | 71 |
| `SIM-ONLY (prompt silent)` | 11 |
| `PROMPT-ONLY (not enforced)` | 4 |
| `NUMBERS DIFFER` | 1 |
| `WORDING DIFFERS` | 5 |
| **Total rule rows** | **92** |
| `DEAD (enforces nothing)` — bookkeeping, §E/§F only | 6 |

**After Task 15:** three rows were removed with the code they described (B36
gravity backstop, B37 friction caps, and the `rules.yaml` check string Y11 that
advertised B36), and `M11, M12, M25, B8, B20, B25, B27, B30, B35, C6, C8, C9,
C10` and `O3` became `MATCH`, each by stating the rule in both prompts or by
deleting a check that was never stated. Of the six `DEAD` pieces, five are
deleted and one (`validate_model_mass`) is kept solely because the upstream suite
imports it.

**Four rule rows are still not `MATCH`**, and each is deliberate:

| Row | Verdict | Why it stands |
|---|---|---|
| M5 | `SIM-ONLY (prompt silent)` | the soft-contact artifact filter on floor contacts — an implementation detail of the floor rule, not a rule of its own |
| M26 | `SIM-ONLY (prompt silent)` | the authoring structure checks re-run inside `SumoEnv.__init__`; the rules themselves are stated (B16, B17, B21, B22) |
| C7 | `SIM-ONLY (prompt silent)` | AST hardening beyond the stated allowlist — naming the bypasses in the prompt would advertise them |
| B34 | `PROMPT-ONLY (not enforced)` | `description=` is required of the model but not checked, by the user's own ruling |

Counts **as the tables now stand** (recounted from §A–§D; §E and §F are
inventories whose rows map onto those, so they add none):

| Verdict | Count |
|---|---|
| `MATCH` | 85 |
| `SIM-ONLY (prompt silent)` | 3 |
| `PROMPT-ONLY (not enforced)` | 1 |
| `NUMBERS DIFFER` | 0 |
| `WORDING DIFFERS` | 0 |
| **Total rule rows** | **89** |
| `DEAD (enforces nothing)` — bookkeeping, §E only | 1 |

Per section: §A match rules 28 rows (26 `MATCH`, 2 `SIM-ONLY`); §B build rules
42 rows (41 `MATCH`, 1 `PROMPT-ONLY`); §C controller rules 10 rows (9 `MATCH`,
1 `SIM-ONLY`); §D observation spot checks 9 rows (all `MATCH`). §E's validator
inventory is 21 rows (20 `MATCH`, 1 `DEAD`) and §F's `rules.yaml` check strings
are 17 rows (all `MATCH`); every one of them points at a rule row above.

Every non-`MATCH` row appears in **§Flags for the user**, one line each, with the
smallest change that would make it `MATCH` and a recommendation.

---

## §A — Match rules: what ends a round and who wins

| # | Rule | Prompt (file:line, quote) | Verifier (function, message) | Simulator (function, `termination_reason`) | Verdict |
|---|---|---|---|---|---|
| M1 | Elevated circular platform, 15 m diameter (radius 7.5 m) | ARH:7 / SH:7 "an elevated circular platform, 15 m in diameter"; ARH:108 / SH:83 `radius_m: 7.5  # diameter 15m` | — | `sumo_ring` geom `size="7.5 1.0"` in `mjarena/assets/sumo_ring_env_{cinematic,studio}_3d.xml`; `SumoEnv.__init__` reads `self.ring_radius` from that geom | `MATCH` |
| M2 | Platform 2 m thick (top surface z = 2.0, floor z = 0) | ARH:109 / SH:84 `thickness_z_m: 2.0` | — | ring cylinder half-height 1.0 at `pos="0 0 1.0"`; `SumoEnv.ring_top_z = 2.0` | `MATCH` |
| M3 | Randomized starting positions, qualification and tournament | ARH:7, ARH:94, ARH:100 / SH:7, SH:69, SH:75 "with randomized robot starting positions" | — | `compose_sumo_model(randomize_spawn_3d=True)` from `unified_builder.py:213` (qualification) and `tournament/tournament.py:216` (tournament); `SumoEnv._set_start_positions(seed)` | `MATCH` |
| M4 | Any part touching the floor below the platform = loss | ARH:7 / SH:7 "A robot loses if any part of it touches the floor below the platform." | — | `SumoEnv.step` scans `data.contact` for the `outside_floor` geom and attributes it by body ancestry (not geom names) → `ring_out`; `Match._is_match_done` awards the round to the other side | `MATCH` |
| M5 | Floor contacts from geoms that penetrated *through* the ring are ignored | — (prompt silent) | — | `sumo.py:1336` — a contact is skipped when the geom is inside `ring_radius − 0.1` **and** above `ring_top_z − 0.05` (soft-contact artifact filter) | `SIM-ONLY (prompt silent)` |
| M6 | From 10 s, lose if the two furthest COM positions in the trailing 10 s are < 0.5 m apart (XYZ) | ARH:7 / SH:7 "Starting 10 seconds into the round … less than 0.5 m apart, measured in XYZ." | — | `SumoEnv.check_inactivity` — window `_inactivity_window_steps = 10.0 / 0.01 = 1000`, `_inactivity_min_displacement = 0.5`, history seeded at reset after settling → `inactivity` | `MATCH` |
| M7 | Simultaneous inactivity: the robot whose COM is more than 1 cm higher wins, else tie | ARH:7 / SH:7 "the robot whose center of mass is more than 1 cm higher wins; otherwise, the round is a tie" | — | `check_inactivity` — `red_z - blue_z > 0.01` → `"blue"` loses, symmetric, else `"both"` → tie | `MATCH` |
| M8 | The qualification block is exempt from inactivity loss | ARH:94 / SH:69 "It is exempt from inactivity loss."; ARH:537 / SH:512 "Always 0 for the qualification block." | — | `inactivity_exempt_prefixes=["blue_"]` (`unified_builder.py:231`); `run_match` otherwise exempts any contender with `action_dim == 0`; `check_inactivity` zeroes the timer for exempt sides | `MATCH` |
| M9 | Physics instability (NaN/Inf in forces) terminates the match as a loss | ARH:7 / SH:7 "If your robot causes the physics simulation to become unstable (NaN/Inf in forces), the match terminates and counts as a loss." | — | `SumoEnv.step` reads `data.warning[mjWARN_BADQACC].number` (MuJoCo auto-recovers, so `isnan(qacc)` never fires) and blames the body owning the bad DOF → `qacc`; `"both"` when unattributable | `MATCH` |
| M10 | A controller error counts as a loss | ARH:9 / SH:9 "A controller error counts as a loss." | `compile_policy`, `execute_policy`, `validate_actions` reject a broken controller before the match | `Match.single_match_step` catches any exception from `bot.act`, substitutes a zero action, sets `self._forfeit_color`; the loop forces the result and sets `termination_reason = "forfeit_crash"` (`episode.py:935/996`) | `MATCH` |
| M11 | Loss priority: controller error > floor contact > size violation > inactivity > instability | ARH:9 / SH:9 "controller error, floor contact, size violation, inactivity, then physics instability" (Task 15, item B) | — | Actual order: forfeit override (`episode.py:932`) > `ring_out` (set last in `sumo.py:1443`) > `size_violation` > `inactivity` > `qacc` > `timeout` | `MATCH` |
| M12 | Root-frame extents over 2.44 × 2.44 × 3.05 m end the match as a loss for that robot | Stated (Task 15, item B): "During a match the limits are also checked on every control step in your robot's own root-body frame: if your robot's extents along its root axes exceed 2.44 × 2.44 × 3.05 m at any step, you lose the round." | `validate_size_constraints` checks the *initial pose*, in BOTH frames (row B5): `initial_root_frame_extents` runs the same `root_frame_extents` helper the match runs, so a root body carrying `euler=`/`quat=` can no longer pass authoring and then lose on step 1 (2026-09-16 review, C-1) | `SumoEnv.robot_extents` (root body frame, so yaw does not inflate the box) + `check_size_limit` (tolerance 1e-6) → `size_violation`; armed in qualification, the tournament runner, the two-stage runners and both viewer paths | `MATCH` |
| M13 | If both robots lose under the same condition the round is a tie | ARH:9 / SH:9 "If both robots lose under that condition, the match is a tie, except for the inactivity height tiebreaker" | — | `"both"` from `check_size_limit`, `check_inactivity`, `_qacc_loser` and `_forfeit_color` all map to `"tie"` in `_is_match_done`; simultaneous floor contact also ties | `MATCH` |
| M14 | Loss conditions take precedence over time expiration; otherwise the match is a draw | ARH:7, ARH:9 / SH:7, SH:9 "If neither robot loses before time expires, the match is a draw." / "Loss conditions take precedence over time expiration." | — | `timeout` is assigned only when `termination_reason is None` (`sumo.py:1447`); `MatchResult.termination_reason` is `"draw"` when there is no winner | `MATCH` |
| M15 | Qualification rounds are 20 s (2 000 control steps) | ARH:94 / SH:69 "three 20-second rounds"; ARH:542 / SH:517 "2000 for qualification (20 seconds)" | — | `configs/tournaments/base.yaml` → `rules.controller.match_time: 20`; `run_match` resolves `max_steps = 20 / 0.01 = 2000` and passes the same cap to the env and the `Match` wrapper | `MATCH` |
| M16 | Tournament rounds last up to 300 s (30 000 control steps) | ARH:100 / SH:75 "rounds lasting up to 300 seconds"; ARH:542 / SH:517 "30000 for the tournament (300 seconds)" | — | `base.yaml` → `tournament.match.match_time: 300`; `two_stage/match_config.py` reads the same key (`req(match, "match_time", "tournament.match")`) | `MATCH` |
| M17 | Controller called every 0.01 s of simulated time, 40 physics steps between calls | ARH:13 / SH:13 "called every 0.01 seconds of simulated time, with 40 physics simulation steps between calls" | — | `SumoEnv.HIGH_FIDELITY_CONTACT` = `{model_timestep: 0.00025, control_timestep: 0.01}`; `step()` computes `n_substeps = round(0.01 / 0.00025) = 40` | `MATCH` |
| M18 | Gravity 9.81 m/s² along −Z | ARH:13 / SH:13 "with gravity of 9.81 m/s²" | — the robot cannot set gravity: `<option>` is stripped before validation (row B29). `validate_simulation_settings`, the old post-compile backstop, was deleted in Task 15, item M (row B36) | arena XML `<option gravity="0 0 -9.81"/>` | `MATCH` |
| M19 | Before the first controller call: 50 physics steps (0.0125 s) with zero motor commands, then all joint velocities zeroed | ARH:300 / SH:275 "the environment simulates 50 physics steps (0.0125 seconds) with zero motor commands, then sets all joint velocities to zero. Joint positions may change during settling." | — | `sumo.py` `SETTLE_STEPS = 50`; `reset()` forces `data.ctrl[:] = 0` each step, then `data.qvel[:] = 0.0` and `mj_forward`; 50 × 0.00025 s = 0.0125 s | `MATCH` |
| M20 | Qualification: must **not lose** any of three 20-second rounds against the block; a draw passes; winning all three is the goal, not the bar | ARH:94 / SH:69, verbatim paragraph | `qualify_round` — `qualified = n_rollouts > 0 and matchup.n_losses() == 0`; failure text "qualification requires not losing any round… Winning every round is the goal." | three seeds via `run_seeds`, `n_rollouts = 3` (`build_config.py:20`) | `MATCH` |
| M21 | The block has no motors and can freely translate, rotate and topple | ARH:94 / SH:69 "The block has no motors and can freely translate, rotate, and topple." | — | `mjarena/core/assets/stationary_block_3d.xml` — a single `<body name="block">` with `<freejoint name="root"/>` and an empty `<actuator>` section; driven by `PolicySpec.zero` | `MATCH` |
| M22 | Failing qualification forfeits the tournament | ARH:94 / SH:69 "If your robot fails to qualify, it forfeits and loses the tournament." | `validate_controller` marks the commit not-qualified; `refinement_state` never promotes it | two-stage rosters only admit qualified artifacts | `MATCH` |
| M23 | Standings are Bradley–Terry with draws counted as half a win | ARH:100 / SH:75 "Bradley–Terry ratings, with draws counted as half a win for each robot" | — | `mjarena/elo/core.py::bradley_terry_ratings` — "draws count as half a win to each side", MM algorithm, one virtual draw per pair | `MATCH` |
| M24 | Robot geoms get `contype=1 conaffinity=1 condim=6 priority=2 solmix=1 solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" margin=0 gap=0`; torsional 0.005, rolling 0.0001; sliding from the material | ARH:148, ARH:150 / SH:123, SH:125 | `normalize_robot_collision_settings` writes `ROBOT_COLLISION_SETTINGS` onto every `<geom>` (including `<default><geom>`), records each override in `Ignored environment-owned collision settings: …`, and re-writes friction as `"<sliding> 0.005 0.0001"` | `enforce_contact_settings(model)` at `SumoEnv.__init__` re-applies `geom_condim[:] = 6` and `geom_friction[:, 1:] = [0.005, 0.0001]` to the **whole composed model** (arena geoms included) and rewrites any explicit contact pair to `dim 6` | `MATCH` |
| M25 | The ring surface itself has sliding friction 2.0 and contact priority 1 | Stated (Task 15, item F): "The platform's own friction never applies to a robot–platform contact: robot geoms have contact priority 2 and the platform priority 1, so your material's friction is your traction." | — | arena XML `sumo_ring` `friction="2.00 0.005 0.0001" priority="1"`; because robot geoms are priority 2, the *robot's* material friction wins every robot-ring contact | `MATCH` |
| M26 | The actuation and single-root-free-joint rules are re-checked at match construction; a violation aborts the match | — (prompt silent) | `validate_internal_actuation`, `validate_single_root_free_joint` (authoring) | `SumoEnv.__init__` raises `ValueError("Invalid robot actuation: …")` / `ValueError("Invalid robot structure: …")` — an infrastructure abort, **not** a graded loss | `SIM-ONLY (prompt silent)` |
| M27 | No per-call controller time limit | ARH:13 / SH:13 "There is no per-call time limit; the simulation waits for the controller to return." | — | no timer anywhere in `Match.single_match_step` / `BotRuntime.act` | `MATCH` |
| M28 | Controller state (including globals) persists within a match; every match starts fresh | ARH:80 / SH:59 "Controller state, including globals, persists between `policy_step` calls within a match. Every qualification and tournament match starts with a fresh controller state." | `compile_policy_function` execs into one namespace per compile | `run_match` calls `ActuatorPolicyAdapter.new_match()` for both sides, re-executing the source in a fresh namespace per match | `MATCH` |

---

## §B — Build rules: what the verifier requires of a submission

| # | Rule | Prompt (file:line, quote) | Verifier (function, message) | Simulator | Verdict |
|---|---|---|---|---|---|
| B1 | Total mass 25–800 kg, geometry + motor mass | ARH:118–120 / SH:93–95 `min_total_kg: 25.0`, `max_total_kg: 800.0` | `validate_mass_constraints` — "Total mass {…} below minimum 25.0" / "exceeds maximum 800.0"; `validate_motor_mass` reports the geometry/motor split and repeats the max check with a fix hint | `obs["my_mass"]` reports the compiled total | `MATCH` |
| B2 | No per-body mass cap | prompt's `rules` block lists only `min_total_kg` / `max_total_kg` | none — `max_single_body` / `min_geom_mass_kg` were removed in Task 1 and exist nowhere in the tree | — | `MATCH` |
| B3 | Motor mass = Σ\|gear\| × 0.01 kg; gear magnitude is `abs(gear)` for a scalar, `sqrt(Σ component²)` for a vector | ARH:114–116, ARH:124 / SH:89–91, SH:99 | `inject_motor_mass` uses `np.linalg.norm(gear_vals)`; `validate_motor_mass` recomputes the same norm for the breakdown | motor mass is baked into `mass=` before compilation, so `body_mass` already includes it | `MATCH` |
| B4 | Motor mass lands on the first geom of the mounting body (joint body / `site` body / `slidersite` body / root for tendon motors), searching rigidly attached descendants depth-first in XML order, skipping jointed bodies | ARH:353 / SH:328 | `inject_motor_mass` + `_first_rigidly_attached_geom`; raises "Cannot place motor '…' mass: body '…' has no geom in itself or its rigidly attached descendants." | — | `MATCH` |
| B5 | Size box 2.44 × 2.44 × 3.05 m, measured on the XML initial pose — along world X/Y/Z, and (since the 2026-09-16 review) along the root body's own axes too, which is what the match measures ("Validation measures the initial pose in the root-body frame as well and rejects a robot outside the box") | ARH:139–143, ARH:146 / SH:114–118, SH:121 | `validate_size_constraints` + `compute_robot_aabb` (SDF geoms use compiled mesh vertices, not a rotated local AABB) — "X-span {…}m exceeds 2.44m limit. Fix: reduce geom size= values…"; and + `initial_root_frame_extents` — "X-span {…}m in the robot's own root-body frame exceeds 2.44m limit…" (row M12) | — (runtime limit is row M12) | `MATCH` |
| B6 | At most 2 000 bodies, 2 000 geoms, 2 000 actuators; at least 1 actuator | ARH:133–137 / SH:108–112 | `validate_structural_constraints` — "Body count {…} exceeds maximum 2000", "Actuator count {…} below minimum 1"; `validate_torque_budget` repeats the zero-actuator case ("Robot has no actuators (min_actuators=1 required)") | — | `MATCH` |
| B7 | Motor commands range from −1 to 1 | ARH:82, ARH:351 / SH:61, SH:326 "a float in `[-1, 1]`" / "Motor commands range from −1 to 1." | `validate_actions` / `_coerce_action_dict` accept the shape; values are not range-checked | `clip_actuator_actions` clips to `[-1, 1]` (a literal, not the config key); `BotRuntime.apply_action` clips again before writing `data.ctrl` | `MATCH` |
| B8 | Motor commands are clipped to [-1, 1] | ARH/SH "Motor commands range from −1 to 1" and "a float in `[-1, 1]`". The `max_ctrl_magnitude` key is **removed** from both prompts and from `rules.yaml` (Task 15, item I) | the clip in `clip_actuator_actions` is the literal `-1.0, 1.0`; `ModelValidationConfig` no longer carries the field | — | `MATCH` |
| B9 | `gear_clamp_ratio: 0` — 0 disables gear clamping; when enabled, max gear = `body_inertia_min × ratio` | ARH:128–130 / SH:103–105 | `ModelValidationConfig.gear_clamp_ratio = ctrl_cfg["gear_clamp_ratio"]` (required read, single source `rules.yaml`); `morphology_pipeline` only calls `clamp_actuator_gears` when `> 0` | `run_match(gear_clamp_ratio=…)` likewise clamps only when `> 0`; the value threaded from `rules.yaml` is 0, and every signature default is also `0.0` | `MATCH` (rule inert by configuration) |
| B10 | Internal joint/tendon damping 0–600, no mass cost | ARH:125, ARH:331, ARH:372 / SH:100, SH:306, SH:347 | `validate_control_constraints` — "DOF damping {…} exceeds maximum 600"; `validate_passive_mechanisms` — "Tendon damping must be finite and between 0 and 600" | root DOFs are pinned to 0 by sanitization, so `model.dof_damping` is effectively the internal set | `MATCH` |
| B11 | Internal joint/tendon frictionloss 0–600, finite | ARH:126, ARH:333 / SH:101, SH:308 | `validate_passive_mechanisms` — "Joint frictionloss must be finite and between 0 and 600", same for tendons | — | `MATCH` |
| B12 | Spring stiffness finite, ≥ 0, no upper cap; springs must start unloaded in the XML pose (`springref` = `ref`; tendon initial length inside `springlength`) | ARH:335 / SH:310 | `validate_passive_mechanisms` — "Joint stiffness must be finite and nonnegative", "Joint spring '…' must start unloaded: set springref equal to ref", "Tendon spring '…' must start unloaded: omit springlength or include the initial length in its rest interval". Checked at `qpos0`, per extension, so opposing preloaded springs cannot cancel | — | `MATCH` |
| B13 | Root-joint damping, frictionloss and stiffness are fixed at 0 | ARH:127, ARH:331 / SH:102, SH:306 "The environment sets root-joint damping, frictionloss, and stiffness to 0, regardless of your XML." | `sanitize_robot_xml_report` step 4 — sets all four attributes to `"0"` on the root body's joints (removes them outright from a `<freejoint>`, which ignores `<default>`), adds `springdamper="0 0"`, and reports "ROOT joint '…': damping=, … removed automatically" | — | `MATCH` |
| B14 | All armature fixed at 0 (joints, tendons, actuators, including via `<default>`) | ARH:335 / SH:310 "The environment fixes joint, tendon, and actuator armature at 0." | `sanitize_robot_xml_report` step 3 — rewrites `armature="0"` on every `joint` / `motor` / `general` / `spatial` / `fixed` element anywhere in the tree (so `<default>` blocks too) and notes "armature= ignored (fixed at 0)" | — | `MATCH` |
| B15 | Only scalar damping and stiffness (no polynomial coefficients) | ARH:331, ARH:335, ARH:372 / SH:306, SH:310, SH:347 — "scalar `damping=\"d\"`", "scalar `stiffness=\"k\"`", "scalar damping from 0 to 600" | `validate_control_constraints` — "Polynomial (nonlinear) joint damping is not allowed; use scalar damping="; `validate_passive_mechanisms` — "must be scalar; nonlinear spring coefficients must be zero", "Use scalar linear tendon damping" | — | `MATCH` |
| B16 | Exactly one root `<body>` directly inside `<worldbody>`, every other body nested beneath it; the root MUST carry a `<freejoint/>` | ARH:199, ARH:204 / SH:174, SH:179 | `validate_root_freejoint` — "[3D MODE] Root body must be mounted on a freejoint (6 DOF…)"; `validate_single_root_free_joint` — "A robot must have exactly one top-level body directly under &lt;worldbody&gt; -- its root…" (never guesses the root from body id order) | `SumoEnv.__init__` re-runs `validate_single_root_free_joint` per prefix (row M26) | `MATCH` |
| B17 | Exactly one free joint per robot, on the root — no detachable parts | ARH:325 / SH:300 "**free:** 6-DOF (translate + rotate). Only for the root body of the robot." | `validate_single_root_free_joint` — "Body '…' has a free joint '…' but is not a top-level body." Mocap `…beacon` bodies are excluded, matching `SumoEnv._body_belongs_to_contender` | same check at match construction | `MATCH` |
| B18 | Each body with a joint or freejoint needs a geom in itself or a rigidly attached descendant; descendants beyond another joint do not count | ARH:203 / SH:178 | `validate_moving_bodies_have_geoms` — "Body '…' has joint(s) […] but no geom in itself or its rigidly attached descendants. … geoms beyond another joint or freejoint do not count." | — | `MATCH` |
| B19 | Only `<motor>` actuators are allowed | ARH:341 / SH:316 "Only `<motor>` actuators are allowed." | `validate_actuator_tags` — "Only &lt;motor&gt; actuators are allowed; found …. Motor mass … is only charged for &lt;motor&gt; elements, so other actuator tags would get gear force for free." Runs pre-compile at XML level | — | `MATCH` |
| B20 | Every actuator must have a `name=` | Stated (Task 15, item D): "Every `<motor>` must have a unique `name`: your controller returns a dictionary keyed by motor name, so an unnamed motor fails validation." | `validate_actuator_names` — "Actuators at indices […] have no name= attribute. Every &lt;motor&gt; must have a unique name so the controller can reference it." | — | `MATCH` |
| B21 | Motor forces must act between parts of the same robot: no root-freejoint actuation, no site motor without `refsite`, no world or cross-robot anchor | ARH:344, ARH:348, ARH:349 / SH:319, SH:323, SH:324 | `validate_internal_actuation` (via `validate_control_constraints`) — "Actuator '…' drives root/world joint '…'. Root-joint actuation is forbidden.", "must have both sites within the same robot: site= requires refsite=; cranksite= requires slidersite=", "uses an unsupported transmission" | re-checked at `SumoEnv.__init__` (row M26) | `MATCH` |
| B22 | Tendons must stay entirely within the robot; world anchors and cross-robot links are forbidden | ARH:374 / SH:349 "Tendons must remain entirely within the robot; world anchors and connections to other robots are forbidden." | `validate_internal_actuation` tendon loop — "Tendon '…' must connect only joints, sites, and wrapping geoms within one robot" (passive tendons included: a world-anchored spring is the same external wrench) | — | `MATCH` |
| B23 | Every geom takes a material from the palette; mass = volume × material density; a geom with no `material=` is auto-assigned `foam` | ARH:225, ARH:407, ARH:419–421 / SH:200, SH:382, SH:394–396 | `validate_material_attributes` — "Geom '…' missing material= attribute", "has unknown material='…'"; `apply_material_properties` computes `mass = volume × density`, writes friction and `priority`, injects the visual `<material>` assets, and reports "Geoms without material= attribute defaulted to 'foam'" | — | `MATCH` |
| B24 | No `mass=`, `density=`, body `mass=` or `<inertial>` | ARH:407 / SH:382 "Do not use `mass=`, `density=`, body `mass=`, or `<inertial>`." | `validate_material_attributes` — "Geom '…' has mass= attribute.", "has density= attribute.", "Forbidden: mass= on body.", "Forbidden: &lt;inertial&gt; element." | — | `MATCH` |
| B25 | Supported geom types: `box`, `cylinder`, `sphere`, `capsule`, `ellipsoid`, `sdf`; `type="mesh"` is rejected | ARH/SH, after the supported-types list: "`type=\"mesh\"` is not accepted; any mesh can be expressed as an `sdf` geom from the same vertices and faces." (Task 15, item J2) | `validate_material_attributes` — "Geom '…': type=\"mesh\" is not accepted; express the same shape as an sdf geom from the same vertices and faces." Other unsupported types still fail indirectly ("Cannot compute volume for geom '…' (type='…')") | a `mesh` geom would have collided as its convex hull; the rejection makes SDF-vs-convex a deliberate design choice | `MATCH` |
| B26 | Inline meshes must be closed, consistently outward-facing and enclose nonzero volume; vertices in metres, faces as zero-based triples; `scale` multiplies dimensions | ARH:239, ARH:241 / SH:214, SH:216 | `_inline_mesh_volume` — "mesh requires inline vertex and face data defining a closed surface", "mesh face index is outside the vertex array", "mesh must be closed with consistently oriented faces (each edge shared twice)", "mesh must have positive enclosed volume and outward-facing triangles", "mesh scale must be nonzero on every axis". Signed-tetrahedron integration, so concavities are subtracted rather than charged as a convex hull | `sanitize` forces `<mesh inertia="exact">` | `MATCH` |
| B27 | Mesh surfaces must be closed and outward-facing | ARH/SH "The surface must be closed and outward-facing (every edge shared by exactly two triangles, positive enclosed volume)." — the self-intersection requirement was dropped (Task 15, item N) | `_inline_mesh_volume` — "mesh must be closed with consistently oriented faces (each edge shared twice)" and "mesh must have positive enclosed volume and outward-facing triangles", exactly the two properties now stated. No self-intersection code was added | — | `MATCH` |
| B28 | External mesh files and custom SDF functions are not supported | ARH:245 / SH:220 | `sanitize_robot_xml_report` keeps only `<mesh>` children of `<asset>`; a `<mesh file="…">` with no inline `vertex`/`face` fails `_inline_mesh_volume` | — | `MATCH` |
| B29 | `<option>` is removed before validation and reported | ARH:186 / SH:161 "any `<option>` in your XML is removed before validation and reported" | `FORBIDDEN_ROOT_TAGS = {"option"}` → `sanitize_robot_xml` reports "Removed &lt;option&gt; block(s): … the arena sets the physics options.". The post-compile backstop `validate_simulation_settings` is deleted (Task 15, item M, row B36): stripping is the whole enforcement | — | `MATCH` |
| B30 | Non-`<mesh>` children of `<asset>` (materials, textures, …) are stripped and reported | Stated (Task 15, item E), after the existing asset sentence: "Any other child of `<asset>` (materials, textures) is removed before validation and reported." | `sanitize_robot_xml_report` step 2 — `"<asset><{tag}> removed (only inline <mesh> definitions are allowed)"` | — | `MATCH` |
| B31 | `<contact>` blocks, contact pairs and collision exclusions are ignored and reported | ARH:150 / SH:125 "Do not define `<contact>` blocks, contact pairs, or collision exclusions. The validator reports and ignores these overrides." | `normalize_robot_collision_settings` removes `contact` / `pair` / `exclude` elements and lists them in the ignored set | `enforce_contact_settings` additionally rewrites any surviving pair to `dim 6` with palette-consistent friction | `MATCH` |
| B32 | The arena sets `compiler angle="radian"` when validating and composing | ARH:190 / SH:165 | `sanitize_robot_xml_report` creates or edits `<compiler angle="radian">` | arena XML declares `<compiler angle="radian"/>` | `MATCH` |
| B33 | Optional `<default>` blocks may define shared joint, tendon and motor settings | ARH:191 / SH:166 | honoured: `inject_motor_mass` resolves inherited `gear` through nested `<default>` classes and materialises it; sanitization reaches `<default>` children for armature and collision settings | composition preserves scoped defaults | `MATCH` |
| B34 | `description="…"` is **REQUIRED** on every body, geom, internal joint, motor, mesh, site and tendon | ARH:72, ARH:205–207 / SH:51, SH:180–182 "`description=\"...\"` — REQUIRED." | `sanitize_robot_xml_report` deletes every `description=` attribute (MuJoCo rejects unknown attributes); **no validator checks presence** | — | `PROMPT-ONLY (not enforced)` — user-decided 2026-09-16: strip, no check |
| B35 | Every geom must have a unique name, set by its `name` attribute | ARH/SH "Every geom must have a unique name, set by its `name` attribute. A geom without a name fails validation." (Task 15, item J) | `validate_geom_names` (check group "Geom Names", next to "Actuator Names") — "Geom #N in body 'X' has no name; every geom needs a unique name=". MuJoCo's compiler still owns the *uniqueness* half | — | `MATCH` |
| B36 | ~~Post-compile gravity backstop~~ | — | **removed** (Task 15, item M). `validate_simulation_settings`, its check group, `rules.yaml` `robot.expected_physics` and the "Timestep and gravity within valid ranges" check string are all deleted: `<option>` is stripped before validation (row B29), so the backstop could only fire on a model the arena itself had composed | — | *row removed* |
| B37 | ~~Rolling friction ≤ 5.0 and torsional friction ≤ 5.0~~ | — | **removed** (Task 15, item O). `validate_material_properties` is deleted with its check group and the `robot.material` block in `rules.yaml`: sanitization pins both frictions before compilation, so the caps could never fire, and the function's sliding branch read a `max_sliding_friction` attribute that never existed | — | *row removed* |
| B38 | Motors have no speed, power, energy or thermal limit; available torque does not fall off with speed | ARH:355 / SH:330 | nothing imposes one | plain MuJoCo `<motor>`: force = gear × ctrl | `MATCH` |
| B39 | The baseline block is a 1.2 × 1.2 × 0.25 m plastic slab of about 342 kg with the palette's plastic friction | ARH:96 / SH:71 — one sentence after the dictated qualification paragraph, stating exactly those dimensions, that mass and that the friction is the palette's | not validated, and it does not need to be: the block carries `material="plastic"` and `mjarena/core/qualification_block.py` runs it through `apply_material_properties` with the run's palette before composition, the same step a robot goes through. 0.36 m³ × 950 kg/m³ = 342 kg, sliding friction 0.25. `mass=` is additionally baked into the asset so a raw compile weighs the same slab; `tests/test_baseline_block.py` fails if the baked number drifts from the palette product | composition (`normalize_robot_collision_settings`) owns condim 6, priority 2, solref, solimp, margin and gap on the block as on any robot; a path that composes the asset raw gets composition's 1.0 sliding-friction fallback instead of the palette's 0.25. The file's `<default><joint armature="0.0"/>` has no effect either way — MuJoCo freejoints ignore `<default><joint>`, and all six compiled DOFs have armature 0 | `MATCH` |
| B40 | One complete, self-contained design per reply: the MJCF in one fenced `xml` block, the controller in one fenced `python` block, with no placeholders, ellipses, omitted helpers or external-file dependencies | ARH:38, ARH:68, ARH:76 / SH:21, SH:47, SH:55 | `extract_code_block` (through `strip_wrappers` and `_strip_code_fences`) accepts exactly one closed fence and raises `ValueError` otherwise, so the build loop never guesses which of several designs was meant. "No external files" is enforced concretely by B28 (inline meshes only) and C3 (import allowlist); "no placeholders / ellipses / omitted helpers" is enforced indirectly — an ellipsis fails XML or Python parsing and a missing helper raises in `execute_policy` | — | `MATCH` |
| B41 | A body with a ball joint may not also carry a hinge or a second ball joint; slide joints are allowed | ARH:323 / SH:298 "A body with a ball joint cannot also have a hinge or another ball joint. It may have slide joints." | MuJoCo's compiler, through `mujoco_compile` — verified: ball + hinge gives "Error: ball followed by rotation in body '…'", ball + slide compiles | — | `MATCH` |
| B42 | An enabled motor `ctrl_range` further limits the command; commands must still be in `[-1, 1]` | ARH:612 / SH:587 "Commands must still be in `[-1, 1]`; an enabled `ctrl_range` further limits the command used by that motor." | nothing strips `ctrl_range` / `ctrl_limited`; both are reported back in the `motors` records of the observation | `clip_actuator_actions` applies `[-1, 1]` first, then MuJoCo clamps `data.ctrl` to `ctrl_range` when `ctrl_limited` is set | `MATCH` |
| B43 | Geoms on independently moving branches of the same robot can collide with each other | ARH:294 / SH:269 "Geoms on independently moving branches of your robot can collide with each other. Check that moving parts can complete their intended motion without unintended collisions." | a robot cannot suppress it: `normalize_robot_collision_settings` removes `<contact><exclude>` (row B31) and pins `contype=1 conaffinity=1` on every geom | MuJoCo's default parent-child contact filtering applies, so sibling branches collide and welded parent/child pairs do not. the `enable_selfcollision` parameter of `compose_sumo_model`, which no call site passed and whose branch wrote an `<option><flag selfcollision>` that is not a MuJoCo 3.x flag, was deleted in Task 15 (item O) | `MATCH` |
| B44 | `size=` does not resize an SDF geom; its dimensions come from the mesh vertices multiplied by `<mesh scale>` | ARH:241 / SH:216 "Do not use geom `size` to resize an SDF." | `_geom_volume` routes `sdf` to `_inline_mesh_volume`, which reads vertices and `scale` and never reads `size`; `_geom_local_half_extents` uses the compiled `geom_aabb` for SDF geoms rather than `geom_size[0]` | `SumoEnv.robot_extents` measures SDF geoms from compiled mesh vertices | `MATCH` |

---

## §C — Controller rules and the sandbox

| # | Rule | Prompt (file:line, quote) | Verifier (function, message) | Simulator | Verdict |
|---|---|---|---|---|---|
| C1 | Implement `def policy_step(obs) -> dict[str, float]` | ARH:78 / SH:57 | `compile_policy_function` — "policy_step(obs) function not found in policy code"; `compile_policy` reports syntax errors with the offending line ("… at line 16 of controller_code:") | — | `MATCH` |
| C2 | Return every actuator name in `robot_xml`, names matching the XML exactly | ARH:82, ARH:84 / SH:61, SH:63 | `validate_actions` / `_coerce_action_dict` — "Missing actuator keys: […]. Must include all: […]" | `clip_actuator_actions` raises `ValueError("Actuator key mismatch — missing: […], unexpected: […]")` mid-match, which the forfeit path turns into a loss (row M10) | `MATCH` |
| C3 | Available libraries: `math`, `numpy` (also bare `np`), `typing`, `random`, `collections`; no other imports | ARH:592–597 / SH:567–572 | `_ALLOWED_IMPORT_ROOTS = {'typing', 'numpy', 'math', 'random', 'collections'}` (`collections` is row C6); `_safe_import` and the AST walk both reject anything else with "Blocked import in policy: {name} (line N)"; relative imports (`level > 0`) are never allowed | namespace pre-injects `math`, `np`, `numpy` | `MATCH` |
| C4 | All builtins available except `open`, `exec`, `eval`, `compile`, `breakpoint`, `exit`, `quit`, `input`, `globals`, `locals`, `vars` | ARH:598 / SH:573 | `_DENIED_BUILTINS` is that exact set; `_restricted_builtins` removes them from the namespace **and** `FORBIDDEN_CALLS` rejects them at validation with a line number ("Disallowed construct in policy: call to open() (line N)") — a compile-time failure the model can fix, rather than a mid-match `NameError` that costs a seed | — | `MATCH` |
| C5 | Forbidden: `os`, `subprocess`, `socket`, `open()`, `exec()`, `eval()`, `pickle`, `threading` | ARH:599 / SH:574 | `FORBIDDEN_MODULES` denies a **superset**: also `sys`, `requests`, `urllib`, `shutil`, `psutil`, `ctypes` (+ `ctypeslib`, `_ctypes`), `multiprocessing`, `pexpect`, `pty`, `dill`, `importlib`, `builtins`, `signal`, `asyncio`, `http`, `ftplib`, `pathlib`. Covered by the prompt's "no other imports allowed" — the allowlist is the operative rule and the denylist only changes the error message | — | `MATCH` |
| C6 | `collections` is importable and pre-injected | ARH/SH `controller_environment`: "    import collections  # deque, defaultdict, Counter" (Task 15, item A) | `compile_policy_function` puts `collections` in the namespace **and** `collections` is in `_ALLOWED_IMPORT_ROOTS`, so both `collections.deque(...)` and `import collections` work | — | `MATCH` |
| C7 | AST hardening: dunder attribute access, `__import__` / `__builtins__` (including `__dict__['…']` and literal `getattr(obj, '…')` lookups), and `np.load(..., allow_pickle=True)` are rejected | — (implied by "forbidden: … eval(), pickle", never stated) | `_validate_policy_code_safety` — "Disallowed construct in policy: dunder attribute access .__class__ (line N)", "allow_pickle". Names are compared as identifier paths, so a local variable called `requests` and a docstring mentioning `os.system` are fine | — | `SIM-ONLY (prompt silent)` |
| C8 | Before qualification the controller is executed once on the **real** initial observation of its own processed robot | Stated (Task 15, item G), in the `controller_code` section: "Before qualification, `policy_step` is called once on a real observation of your robot; an exception or a bad return there is reported as a validation failure." | `initial_controller_observation` composes the robot against the block and runs a 1-step match to capture a genuine observation (then pins `max_t = 2000`, `time_remaining = 20.0`); `execute_policy` — "policy_step(obs) raised: {exc}" | — | `MATCH` |
| C9 | A branch-exercise suite (`default`, `edge_recovery`, `engage_contact`, `hold_center_far`, `anti_inactivity`) is run before qualification | — (prompt silent; the probe itself is row C8) | `exercise_policy` / `_build_execution_cases`. On the normal path (a real `base_observation`) only the `default` case exists, and the pass message now says so: "policy_step(obs) executed successfully on the initial observation of your robot."; otherwise it names the cases that ran (Task 15, item K) | — | `MATCH` |
| C10 | Every LLM call carries a 100-minute wall-clock limit | Both prompts (Task 15, item C): "Each API request has a 100-minute wall-clock limit and a maximum output length; a request that exceeds either ends the run." The SH claim "There is no thinking time limit" is deleted | — | every LLM call carries an explicit per-call timeout of `NO_LLM_TIMEOUT_S = 6000` s = 100 min (`dspy_core.py:441`, set at `:509`; `litellm` coerces `timeout=None` back to 600 s, so an explicit number is required). A call that exceeds it is classified `llm_timeout` (`utils/llm_errors.py`) and, under the retained abort-on-infrastructure-error behaviour, ends that model's build run | `MATCH` |

---

## §D — Observation contract (spot checks)

The prompt's `obs_schema` section and the retired `configs/rules/obs_schema.yaml` were diffed
field by field: every key and every description string is identical, so the two
documents cannot disagree. These rows check the *values* the runtime actually
supplies.

| # | Rule | Prompt (file:line) | Runtime | Verdict |
|---|---|---|---|---|
| O1 | `max_t` = 2000 for qualification, 30000 for the tournament | ARH:542 / SH:517 | `Match.max_steps` from `match_time / control_timestep` = 20/0.01 and 300/0.01 | `MATCH` |
| O2 | `control_dt` = 0.01 | ARH:565 / SH:540 | `SumoEnv.control_timestep` from the high-fidelity preset | `MATCH` |
| O3 | `ring_radius` = arena radius | ARH:549 / SH:524 | real matches: `SumoEnv.ring_radius = 7.5`, read from the `sumo_ring` geom. the fallback dummy observation now reports the same numbers: `BotObservation.ring_radius = 7.5`, and the dummy dict's `control_dt = 0.01`, `proximity_cutoff = SURFACE_CUTOFF` (2.0) and `proximity_limit = SURFACE_LIMIT` (32), imported from `mjarena/envs/detailed_observations.py` (Task 15, item L) | `MATCH` |
| O4 | `obs_history`: last 20 dicts, each with `my_pos`, `opponent_pos`, `my_yaw`, `my_velocity`, `opponent_velocity`, `distance_to_opponent`, `my_edge_distance`, `opponent_edge_distance`, `t`, `my_actuator_velocity`, `game`; empty at t=0 | ARH:560 / SH:535 | `BotRuntime.act` appends exactly those eleven keys to a `deque(maxlen=obs_lookback)`; `obs_lookback` default 20, `base.yaml` `max_obs_lookback: 20` | `MATCH` |
| O5 | `action_history`: last 20 action dicts, recorded before motor-specific `ctrl_range` limits | ARH:561 / SH:536 | `BotRuntime.act` records the adapter's output — already clipped to `[-1,1]`, not yet subject to MuJoCo's `ctrl_range` | `MATCH` |
| O6 | `proximity_cutoff` = 2 m, `proximity_limit` = 32 | ARH:584–585 / SH:559–560 | `detailed_observations.py` `SURFACE_CUTOFF = 2.0`, `SURFACE_LIMIT = 32` | `MATCH` |
| O7 | `game` is `{}` in every match | ARH:557 / SH:532 | `Match.build_bot_observation` filters `__`-prefixed keys out of `game_context`, and no shipped call site passes a non-`__` key | `MATCH` |
| O8 | `opponent_inactivity_timer` is always 0 against the block | ARH:537 / SH:512 | `check_inactivity` zeroes the timer for exempt prefixes | `MATCH` |
| O9 | `arena_grid` cells are 0.25 m | ARH:552 / SH:527 | `base.yaml` `tournament.agents.observation.arena_grid_cell_size: 0.25` | `MATCH` |

---

## §E — Validator inventory

Every `validate_*` defined in `mjarena/design_shop/rules/mj_validators.py`, plus the
pre-compile validators in `mjarena/design_shop/utils.py` and the software gates.
`Runs?` is for the shipped 3D configuration (`physics_mode: 3d`, palette active).

| Validator | Runs? | Enforces | Row | Disposition |
|---|---|---|---|---|
| `validate_mass_constraints` | yes (post-compile) | total mass 25–800 kg | B1 | `MATCH` |
| `validate_size_constraints` | yes | initial-pose spans ≤ 2.44/2.44/3.05 m, world axes **and** root-body frame | B5, M12 | `MATCH` |
| `validate_material_properties` | — | **deleted** (Task 15, item O): both friction caps were unreachable and the sliding branch read an attribute that did not exist | B37 | *removed* |
| `validate_control_constraints` | yes | DOF damping ≤ 600, finite, non-negative; no polynomial damping; delegates to `validate_passive_mechanisms` and `validate_internal_actuation` | B10, B11, B12, B15, B21, B22 | `MATCH` |
| `validate_simulation_settings` | — | **deleted** (Task 15, item M): `<option>` is stripped before validation, so the backstop was unreachable | B36 | *removed* |
| `validate_torque_budget` | yes | `min_actuators` when `nu == 0`, and reports total gear. The legacy `total_gear_budget` / `min_single_gear` / `max_single_gear` branches and their config fields are deleted (Task 15, item O) | B6 | `MATCH` |
| `validate_motor_mass` | yes | reports the geometry/motor mass split; repeats the 800 kg cap with an actionable fix | B1, B3 | `MATCH` |
| `validate_structural_constraints` | yes | 2000/2000/2000 caps, `min_actuators` | B6 | `MATCH` |
| `validate_root_freejoint` | yes (3D branch) | root body carries a freejoint; delegates to `validate_single_root_free_joint` | B16, B17 | `MATCH` |
| `validate_2d_physics` | — | **deleted** (Task 15, item O): never selected in 3D, and it read `config.enforce_2d_physics` / `config.warn_2d_only`, which `ModelValidationConfig` never defined | — | *removed* |
| `validate_density_constraints` | — | **deleted** (Task 15, item O): never in `CHECK_GROUPS` | — | *removed* |
| `validate_actuator_names` | yes | every actuator has `name=` | B20 | `MATCH` |
| `validate_geom_names` | yes (check group "Geom Names") | every robot geom has `name=` | B35 | `MATCH` |
| `validate_actuator_tags` | yes (pre-compile, XML level) | only `<motor>` elements inside `<actuator>` | B19 | `MATCH` |
| `validate_material_attributes` (utils) | yes | palette material per geom; no `mass=` / `density=` / body `mass=` / `<inertial>`; rejects `type="mesh"` (Task 15, item J2) | B23, B24, B25 | `MATCH` |
| `validate_moving_bodies_have_geoms` (utils) | yes | a geom in each jointed body's rigid assembly | B18 | `MATCH` |
| `validate_passive_mechanisms` (utils) | yes | damping/frictionloss 0–600, stiffness finite ≥ 0, springs unloaded at `qpos0` | B10–B12, B15 | `MATCH` |
| `validate_internal_actuation` (utils) | yes (authoring **and** `SumoEnv.__init__`) | no root/world actuation, site needs refsite, cranksite needs slidersite, tendons inside one robot | B21, B22, M26 | `MATCH` |
| `validate_single_root_free_joint` (utils) | yes (authoring **and** `SumoEnv.__init__`) | one top-level body = root; exactly one free joint, on it | B16, B17, M26 | `MATCH` |
| `validate_model_mass` (utils) | **no** | boolean total-mass helper, in no check group. Kept only because `tests_environment/test_joint_damping_rules.py` imports it; its `max_total_mass=1000.0` default was removed so no caller can inherit a cap nobody configured (Task 15, item O) | — | `DEAD (enforces nothing)`; kept for `tests_environment` import |
| `validate_model_size` (utils) | — | **deleted** (Task 15, item O): boolean span helper, never called, carrying the stale `max_span=10.0` | — | *removed* |
| `compile_policy` / `compile_policy_function` (software) | yes | syntax, sandbox, `policy_step` present | C1, C3, C4, C7 | `MATCH` |
| `execute_policy` (software) | yes | one call on the real initial observation | C8 | `MATCH` |
| `validate_actions` (software) | yes | return type, exact actuator key set | C2 | `MATCH` |
| `exercise_policy` (software) | yes | branch suite (degenerate on the real-observation path, and its message now says which cases ran) | C9 | `MATCH` |
| `qualify_round` (match tools) | yes | no losses over three 20 s seeds | M20 | `MATCH` |

---

## §F — `rules.yaml` `validation.checks` inventory

`configs/rules/rules.yaml` advertises seventeen checks. Each is quoted verbatim
below with the code that implements it. Task 15 deleted Y11 and added Y16–Y18;
the numbers are assigned in the order the rows were audited, not in file order
(the three new strings sit next to Y7 in `rules.yaml`).

| # | Check string | Implemented by | Row | Verdict |
|---|---|---|---|---|
| Y1 | `XML compiles in MuJoCo` | `mujoco_compile` | B26/B33 | `MATCH` |
| Y2 | `Total mass (geometry_mass + sum(\|gear\|) × motor_mass_per_gear) under limit` | `validate_mass_constraints`, `validate_motor_mass` | B1, B3 | `MATCH` |
| Y3 | `Actuator count within min-max range` | `validate_structural_constraints`, `validate_torque_budget` | B6 | `MATCH` |
| Y4 | `Only <motor> actuator tags allowed (other actuator types are rejected)` | `validate_actuator_tags` | B19 | `MATCH` |
| Y5 | `Root body has freejoint (6 DOF for 3D)` | `validate_root_freejoint` | B16 | `MATCH` |
| Y6 | `Exactly one free joint per robot, on the root body (no detachable parts)` | `validate_single_root_free_joint` | B17 | `MATCH` |
| Y7 | `Every geom has a valid material= from the palette` | `validate_material_attributes` | B23 | `MATCH` |
| Y8 | `No mass=, density=, body mass=, or <inertial> on geoms` | `validate_material_attributes` | B24 | `MATCH` |
| Y9 | `Size constraints (AABB spans within limits)` | `validate_size_constraints` (world-axis AABB and root-frame spans) | B5, M12 | `MATCH` |
| Y10 | `Collision settings are environment-owned (condim 6, priority 2, fixed solref/solimp; author overrides ignored and reported)` | `normalize_robot_collision_settings`, `enforce_contact_settings` | M24, B31 | `MATCH` |
| ~~Y11~~ | ~~`Timestep and gravity within valid ranges`~~ | **check string removed** with `validate_simulation_settings` (Task 15, item M) | B36 | *removed* |
| Y12 | `Gear per actuator <= body_inertia_min × gear_clamp_ratio (clamped if exceeded, 0 = disabled)` | `clamp_actuator_gears`, gated on `gear_clamp_ratio > 0`; the shipped value is 0 | B9 | `MATCH` (inert) |
| Y13 | `Internal joint/tendon damping and frictionloss finite and within 0–600; stiffness finite and >= 0; springs unloaded in the initial pose` | `validate_passive_mechanisms`, `validate_control_constraints` | B10–B12 | `MATCH` |
| Y14 | `Inactivity (qualification and tournament): from 10 s, a robot loses when its two furthest COM positions in the trailing 10 s are < 0.5 m apart (XYZ)` | `SumoEnv.check_inactivity` | M6, M7 | `MATCH` |
| Y15 | `Runtime size limit (every control step, qualification and tournament): a robot whose root-frame extents exceed max_{x,y,z}_span_m loses on that step` | `SumoEnv.check_size_limit` | M12 | `MATCH` (the prompt states it too since Task 15, item B) |
| Y16 | `Every <motor> has a name= (unnamed actuators are rejected)` | `validate_actuator_names` | B20 | `MATCH` |
| Y17 | `Every geom has a name= (unnamed geoms are rejected)` | `validate_geom_names` | B35 | `MATCH` |
| Y18 | `Geom type= is one of box, cylinder, sphere, capsule, ellipsoid, sdf (type="mesh" is rejected)` | `validate_material_attributes` | B25 | `MATCH` |

`rules.yaml` carries one size block, `robot.size`, read by `ModelValidationConfig`.
The duplicate `robot.bounding_box_3d` was removed in Task 15 (item P) together with
its only non-test reader, the retired `dspy_core.load_constraints_as_string`.

---

## §G — `termination_reason` inventory

Literals assigned in `mjarena/envs/sumo.py`, plus the two set in
`mjarena/runner/episode.py`.

| Reason | Set where | Meaning | Row |
|---|---|---|---|
| `qacc` | `sumo.py:1419` | MuJoCo `mjWARN_BADQACC` fired; blamed on the body owning the bad DOF, `"both"` when unattributable | M9 |
| `size_violation` | `sumo.py:1430` | root-frame extents exceeded the size box this step | M12 |
| `inactivity` | `sumo.py:1438` | trailing-10 s COM diameter below 0.5 m for a full window | M6, M7 |
| `ring_out` | `sumo.py:1443` | a robot geom contacted the `outside_floor` plane; overrides `qacc`, `size_violation` and `inactivity` | M4 |
| `timeout` | `sumo.py:1447` | step cap reached with no loss condition | M14 |
| `forfeit_crash` | `episode.py:935`, `episode.py:996` | a controller raised; overrides every reason above | M10 |
| `draw` | `episode.py:1149` | reported by `MatchResult` when the match ended with no winner | M14 |

---

## §H — Modular rule files vs the consolidated prompts — **RETIRED**

**Retired by Task 15 (item R).** There is no fallback prompt path any more:
`configs/rules/game.md`, `game-zeroshot.md`, `mjcf_syntax.yaml` and
`obs_schema.yaml` are deleted, their loaders (`load_game_spec`,
`load_constraints_as_string`, `load_mjcf_syntax`, `load_obs_schema`,
`make_morphology_task_spec*`, `get_task_spec`) are deleted from
`mjarena/dspy_core.py`, the four static inputs `task_spec` / `rules` /
`mjcf_syntax` / `obs_schema` are gone from both baseline signatures, and
`BuildConfig.__post_init__` refuses a run that does not name its one prompt.
Items 1–4 below are therefore moot and kept only as a record of what the retired
files said; item 5 was settled by Task 15 items C and H.

1. **`game.md:5`** — "You win the battle if the opponent robot falls off the ring
   and hits the ground before you." This names only the ring-out condition and
   contradicts `game.md:19` fourteen lines below, which correctly lists
   controller error, floor contact, inactivity and physics instability. The
   consolidated prompts have no such sentence. *(Modular file only.)*
2. **`game.md:31`, `game.md:45`, `game.md:78`** describe a round robin among a
   model's own qualified commits deciding the tournament entry. Not implemented
   in either repository (documented out of scope). The consolidated prompts do
   not promise it. *(Modular file only.)*
3. **`mjcf_syntax.yaml`** is otherwise identical to the prompt's MJCF section but
   (a) omits the `<option>` sentence (ARH:186 / SH:161) and (b) hardcodes
   "MuJoCo 3.10.0" instead of the `{mujoco_version}` placeholder.
4. **`obs_schema.yaml`**: every `fields` key and description was byte-identical to
   the prompts, but its `controller_environment` block omitted `import random` and
   the denied-builtins sentence. *(File deleted.)*
5. The two consolidated prompts differed from each other in exactly two
   rule-adjacent sentences, both fixed in Task 15: (a) SH's "Primitive distances
   use MuJoCo" was replaced by ARH's "Primitive proximity uses convex
   surface-distance queries" (item H); (b) SH's "There is no thinking time limit"
   was replaced, in both documents, by the 100-minute sentence (item C, row C10).
   The two documents now describe one engine identically.

---

## Provenance

| Module | Source |
|---|---|
| `mjarena/design_shop/utils.py`, `rules/mj_validators.py`, `rules/hardware_rules.py`, `rules/software_rules.py`, `policy_base.py` | ported from the upstream `arena-main` tip **`16726431`** ("Add cube-only tournament predictor to AutoResearch") |
| `mjarena/envs/sumo.py`, `mjarena/envs/utils.py`, `mjarena/runner/episode.py`, `mjarena/agents/runtime.py` | ported from `arena-main` **`16726431`**, with the user's own rules layered on: 3D furthest-pair inactivity with the 1 cm tiebreak (`6c73846e`), the runtime size limit (`6ae6a5df`, `56c4dfaa`), one free joint per robot (`a5235e6d`, `adbba831`) |
| `mjarena/envs/surface_distance.py`, `mjarena/envs/detailed_observations.py`, `mjarena/design_shop/controller_observation.py` | verbatim from the upstream commit **`c7cb3814`** ("Fix robot observations and add native and WASM stress coverage"), not from `arena-main`: `arena-main`'s tip predates his observation fix, and its `mj_geomDistance`-based proximity cannot meet the acceptance suite's tolerances on MuJoCo 3.10.0. `tests_environment/README.md` names `c7cb3814` as its verified baseline (commit `dbd52faf`) |
| `configs/rules/autoresearch_prompt.md`, `configs/rules/sampling_prompt.md` | the upstream consolidated prompts (`b07ec43a`), with the user's rulings applied in `d49f22c8` / `76a8276e` (3.05 m height, no-loss qualification paragraph verbatim, 300 s rounds, `random`, the `<option>` sentence, single-source `gear_clamp_ratio`) and in Task 15 (runtime size limit, loss priority, 100-minute call limit, motor and geom names, no `type="mesh"`, `<asset>` stripping, ring friction, the execution probe, the mesh wording, `collections`, no `max_ctrl_magnitude`). Since Task 15 they are the **only** prompt documents: `game.md`, `game-zeroshot.md`, `mjcf_syntax.yaml` and `obs_schema.yaml` are deleted |
| `tests_environment/` | the upstream acceptance suite, unmodified. Current state **334 passed, 9 failed** (300 s rounds deselected) — see below |

### Record fields this port redefined

Two fields of every `GameRecord` mean something slightly different after the
port, so records written before it are not directly comparable:

- **`num_steps`** is now the FRAME COUNT (`len(qpos_log)`, `episode.py:1099`),
  which includes the initial post-settle frame — one more than the step count
  the old records carried. Any per-step mean taken over a whole match (the
  engagement-style combat metrics) shifts by at most 1/N.
- **`control_dt`** is now derived, not assumed: `physics_dt × substeps ×
  apply_n_repeated_actions` (`detailed_observations.py:65`). In every shipped
  config that evaluates to the same 0.01 s the old constant asserted, so no
  shipped record changes value — but a config that repeated actions used to
  report a control interval it was not running.

Neither field changes who won a match; both change what a downstream reader
should assume about one.

The nine expected upstream failures, all by design and all reported upstream rather than
worked around:

- 3 height cases that assume a 10.0 m z-limit where the user's rule is 3.05 m:
  `test_constraint_boundaries.py::test_size_limit_boundaries[-0.0001-True-2-10.0]`,
  `…[0-True-2-10.0]`,
  `test_sdf_bounds.py::test_size_validator_accepts_and_rejects_elongated_sdfs[2-9.9-True]`.
- 5 `test_qualification.py` cases that encode the upstream three-wins qualification
  rule against the user's no-losses rule:
  `test_qualify_round_draw_fails[winners0]`, `[winners1]`,
  `test_stationary_block_rule_requires_all_wins[winners0-True]`,
  `[winners1-False]`, `[winners2-False]`.
- 1 known MuJoCo 3.10.0 SDF sign-field defect, kept as a failing regression by
  The upstream comment:
  `test_sdf_contacts.py::test_sdf_scoop_does_not_generate_contacts_across_verified_air_gap`.

`test_match_diagnostics.py::test_match_duration_overrides_stale_environment_step_cap`,
listed as a known failure when this plan was written, now passes.

---

## Flags for the user

### Already ruled on — status after this audit

| # | Flag | Your decision | Status now |
|---|---|---|---|
| 1 | Height: prompt 3.05 m; the upstream `rules.yaml` + 3 test cases assume 10.0 | 3.05 everywhere | **Done.** 3.05 in `rules.yaml` (`size`; the duplicate `bounding_box_3d` is gone, item P) and in both prompts. 3 of the upstream cases stay red by design — tell upstream. `validate_model_size`, which carried the last 10.0, is deleted (item O). |
| 2 | Qualification: upstream = win all three; user = must not lose any, draw passes | user's paragraph verbatim | **Done.** `qualify_round`: `n_losses() == 0`. Paragraph verbatim at ARH:94 / SH:69. 5 of the upstream cases stay red by design. |
| 3 | Tournament length: prompt 300 s; `base.yaml` 120 s | 5 minutes | **Done.** `tournament.match.match_time: 300`; qualification `rules.controller.match_time: 20`. Rows M15, M16. |
| 4 | Runtime size limit vs the prompt's "not checked during motion" | explanation requested | **Done** (Task 15, item B). Both prompts state the root-frame check and the loss, and "size violation" is in the priority list. Rows M11, M12 are `MATCH`. |
| 5 | `description=` required in the prompt, not checked | strip, no check | **Resolved as intended.** Row B34: the sanitizer deletes it, nothing validates it, prompt wording stays. |
| 6 | `max_ctrl_magnitude: 1.05` read by nothing | fix to 1, then (Task 15) delete it | **Done.** The key is deleted from `rules.yaml`, `ModelValidationConfig` and both prompts; the `[-1, 1]` clip sentence is the rule. Row B8 is `MATCH`. |
| 7 | `random` importable but absent from the prompt | put `random` in the prompt | **Done** in both consolidated prompts (row C3). `obs_schema.yaml` is deleted (Task 15, item R), so there is no second list to drift. |
| 8 | `<option>` stripped, prompt silent | tell the model | **Done.** ARH:186 / SH:161, row B29. `mjcf_syntax.yaml` is deleted (Task 15, item R). |
| 9 | `gear_clamp_ratio` in two config sources | single source | **Done.** `rules.yaml` only, required read (no `.get` default). Row B9, Y12. |
| 10 | Baseline block `<default><joint armature="0.01"/>` | check | **Checked: no effect**, and the attribute is gone. MuJoCo freejoints ignore `<default><joint>`; all six compiled DOFs have armature 0, before and after. Task 14 (`7890ccee`) set it to `0.0` with the rest of the block rewrite. Row B39. The file's stale "86.4 kg" comment was fixed in the audit commit; the block is now 342 kg. |
| 11 | Sandbox denies more modules than the prompt names | — | **`MATCH` in effect.** Row C5: the allowlist is the operative rule and the prompt says "no other imports allowed". |
| 12 | Qualification spawn randomization | — | **`MATCH`.** Row M3: `randomize_spawn_3d=True` in the block factory and the tournament runner. |
| 13 | Missing/extra actuator keys → controller error | — | **`MATCH`.** Row C2: `clip_actuator_actions` raises; the forfeit path turns it into a loss. |
| 14 | Controller per-call time limit: prompt says none | — | **`MATCH`.** Row M27: no timer exists. |
| 15 | Size limit not armed in the two-stage runners or the viewer | — | **Done** (`56c4dfaa`). `MatchConfig.size_limits` is a required field; every match-starting call site verified. Row M12. |
| 16 | Size limit measures all geoms; a detached projectile would inflate its launcher | strict: no flying parts | **Done** (`a5235e6d`, `adbba831`). Row B17: exactly one free joint per robot, on the root, enforced at authoring **and** at match construction. |
| 17 | `globals()`/`locals()`/`vars()` allowed to compile, failed mid-match | — | **Resolved.** Row C4: rejected at validation with a line number. |
| 18 | Uncaught simulator exception aborts the whole build run | keep the upstream behaviour | **Kept.** No change. Follow-up outside this plan: make `scripts/arh_launch.sh` resume a lineage from its last completed commit. |

### New and open — one line per non-`MATCH` row

Each line gives the smallest change that would make the row `MATCH`, and which
side I recommend changing. A flag decided and shipped since the audit keeps its
line here, marked **DECIDED and SHIPPED** with the commit that did it; the row it
names is already `MATCH` in the tables above.

- **F-N0 (B34, `PROMPT-ONLY`) — DECIDED, no action.** `description="…"` is marked
  REQUIRED in the prompt and the sanitizer strips it without ever checking it.
  You ruled on 2026-09-16: strip, no check, prompt wording stays (it drives the
  website's design-intent rendering). Recorded here only so no non-`MATCH` row is
  missing from this list.
- **F-N1 (M12, `SIM-ONLY`, user-flagged).** The runtime size limit ends a match
  as a loss, but ARH:146 / SH:121 says "These limits are not checked during
  motion; mechanisms may extend beyond them after the round begins." *Smallest
  fix:* replace that sentence with one stating that the XML-pose check uses world
  axes while the match check uses the root body's own frame, and that exceeding
  it ends the round as a loss. **Recommend changing the prompt** — the rule is
  yours and the engine implements it exactly; only the sentence is stale. Yours
  to word.
  **DECIDED and SHIPPED (Task 15, item B).** The sentence now states the runtime check in the root-body frame and the loss; row M12 is `MATCH`.
- **F-N2 (M11, `WORDING DIFFERS`).** ARH:9 / SH:9's priority list omits the size
  violation, which sits between floor contact and inactivity. *Smallest fix:*
  insert "size violation" into that one list. **Recommend changing the prompt**
  (same edit as F-N1; they should ship together).
  **DECIDED and SHIPPED (Task 15, item B).** "size violation" sits between floor contact and inactivity in both prompts; row M11 is `MATCH`.
- **F-N3 (B8, `PROMPT-ONLY`).** `max_ctrl_magnitude: 1.0` is shown to the model
  and loaded into `ModelValidationConfig`, but read nowhere; the real clip is the
  literal `-1.0, 1.0` in `clip_actuator_actions`. *Smallest fix:* have
  `clip_actuator_actions` take the configured magnitude. **Recommend changing the
  code** — a dead key that the prompt presents as a rule is exactly the failure
  mode this audit exists to catch. (Behaviour is unchanged at 1.0, so this is
  safe to defer.)
  **DECIDED and SHIPPED (Task 15, item I).** The key was deleted from `rules.yaml`, from `ModelValidationConfig` and from both prompts' rules block; the `[-1, 1]` clip sentence stays. Row B8 is `MATCH`.
- **F-N4 (B20, `SIM-ONLY`).** `validate_actuator_names` rejects an unnamed
  `<motor>`; the prompt never says `name=` is required. *Smallest fix:* add
  "Every `<motor>` must have a unique `name`" to the MOTOR section.
  **Recommend changing the prompt** — the check is right and the model currently
  learns the rule only by failing.
  **DECIDED and SHIPPED (Task 15, item D).** The MOTOR section now states it; row B20 is `MATCH`.
- **F-N5 (B25, `SIM-ONLY`).** `type="mesh"` geoms are accepted end to end
  (volume, bounds, runtime extents), but ARH:224 / SH:199 lists only `sdf`.
  A `mesh` geom collides as a convex hull, an `sdf` geom preserves concavity —
  a real competitive difference. *Smallest fix:* reject `type="mesh"` in
  `validate_material_attributes`. **Recommend changing the code** — the prompt's
  list is the intended rule and SDF-vs-convex is a design decision the model
  should make knowingly, not stumble into.
  **DECIDED and SHIPPED (Task 15, item J2).** `validate_material_attributes` rejects `type="mesh"` and both prompts say so; row B25 is `MATCH`.
- **F-N6 (B27, `PROMPT-ONLY`).** The prompt requires meshes to be
  "non-self-intersecting"; `_inline_mesh_volume` checks closedness, orientation
  and positive volume, never self-intersection. *Smallest fix:* soften the prompt
  to the three properties actually checked. **Recommend changing the prompt** —
  a robust self-intersection test is expensive and the signed-volume integral
  already prices concave geometry correctly.
  **DECIDED and SHIPPED (Task 15, item N).** The prompt now states only closedness and outward orientation, in the code's own terms; no self-intersection code was added. Row B27 is `MATCH`.
- **F-N7 (B30, `WORDING DIFFERS`).** Non-`<mesh>` children of `<asset>` are
  stripped and reported, but the prompt only says not to define them — unlike
  `<option>`, which explicitly says "removed before validation and reported".
  *Smallest fix:* extend the `<asset>` sentence the same way. **Recommend
  changing the prompt.**
  **DECIDED and SHIPPED (Task 15, item E).** Row B30 is `MATCH`.
- **F-N8 (B35, `PROMPT-ONLY`).** "Every geom must have a unique name" —
  uniqueness is enforced by MuJoCo, presence is not checked. *Smallest fix:*
  either add a presence check next to `validate_actuator_names`, or soften the
  sentence. **Recommend changing the code**: unnamed geoms degrade the
  observation records (`geoms` keyed by generated names) and the design-intent
  rendering, so the prompt's rule is the one worth keeping.
  **DECIDED and SHIPPED (Task 15, item J).** `validate_geom_names` runs next to `validate_actuator_names` and the prompt says an unnamed geom fails; row B35 is `MATCH`.
- **F-N9 (B36, `SIM-ONLY`).** The post-compile gravity backstop is not mentioned
  to the model. *Smallest fix:* none needed. **Recommend no change** — it fires
  only on a submission that already violated the stated `<option>` rule, and
  telling the model about a backstop invites probing it. Recorded for
  completeness.
  **DECIDED and SHIPPED (Task 15, item M) — the other way.** Rather than describe a backstop, the backstop was deleted: `<option>` is stripped before validation, so `validate_simulation_settings` could never fire. Row B36 and check string Y11 are removed.
- **F-N10 (B37, `SIM-ONLY`, unreachable).** `validate_material_properties` caps
  rolling and torsional friction at 5.0, but sanitization has already pinned them
  to 0.0001 / 0.005, so the check can never fire. *Smallest fix:* delete the two
  caps and the `material` block from `rules.yaml`. **Recommend changing the
  code** — dead validation that reads config keys is how stale rules survive.
  **DECIDED and SHIPPED (Task 15, item O).** The caps, the `robot.material` block and `validate_material_properties` are deleted; row B37 is removed.
- **F-N11 (B39) — DECIDED and SHIPPED, row is now `MATCH`.** The block was built
  with `density="50"`, not the palette's `foam` (200 kg/m³): 36 kg where the
  palette would give 144 kg, with its `friction`/`priority` written directly. It
  is an environment asset, so no validator sees it, but it is the opponent every
  qualification result is measured against. Your decision of 2026-09-16: rebuild
  it from the palette at roughly a third to a half of the 800 kg cap, **and** state
  it in the prompt. Shipped in Task 14 (`7890ccee`): palette plastic
  (950 kg/m³, sliding friction 0.25), half-extents `0.60 0.60 0.125` — a
  1.2 × 1.2 × 0.25 m slab of 342 kg — resolved through `apply_material_properties`
  on the qualification path, with one sentence added after the dictated
  qualification paragraph in all four prompt documents. Every qualification
  outcome recorded before that commit was measured against the old 36 kg block.
- **F-N12 (C6, `SIM-ONLY`, internally inconsistent).** `collections` is
  pre-injected into the controller namespace, so bare `collections.deque(...)`
  works, but `import collections` is rejected with "Blocked import in policy".
  The prompt lists neither. *Smallest fix:* add `collections` to
  `_ALLOWED_IMPORT_ROOTS` **and** to the prompt's library list, or drop it from
  the namespace. **Recommend changing the code to allow the import and listing it
  in the prompt** — a model that writes the conventional `import collections`
  currently loses a commit to a rule it was never told, while a model that
  guesses the pre-injection is rewarded.
  **DECIDED and SHIPPED (Task 15, item A).** `collections` is in `_ALLOWED_IMPORT_ROOTS` and in both prompts' library list; row C6 is `MATCH`.
- **F-N13 (C7, `SIM-ONLY`).** The AST hardening (dunder attributes,
  `__builtins__` / `__import__` paths, `allow_pickle=True`) is not described to
  the model. *Smallest fix:* none needed. **Recommend no change** — these are
  sandbox-escape guards, not design rules; describing them is an invitation.
- **F-N14 (C8, `SIM-ONLY`).** The pre-qualification execution probe on the real
  initial observation is not described. *Smallest fix:* one sentence in the
  `controller_code` section. **Recommend changing the prompt** — it is feedback
  the model receives anyway, and knowing it exists encourages defensive
  first-call handling.
  **DECIDED and SHIPPED (Task 15, item G).** The `controller_code` section states the probe; row C8 is `MATCH`.
- **F-N15 (C9, `WORDING DIFFERS`).** On the normal path (a real robot
  observation) `exercise_policy` runs only the `default` case, yet reports
  "executed successfully on representative edge/contact/inactivity cases."
  *Smallest fix:* make the message reflect which cases ran. **Recommend changing
  the code** — it is a verifier message the model reads and reasons from, and it
  currently claims coverage that does not exist. (Whether the four synthetic
  cases *should* also run against the real observation is a separate design
  question; if you want them to, that is a behaviour change, not a wording fix.)
  **DECIDED and SHIPPED (Task 15, item K).** The pass message names the cases that ran ("on the initial observation of your robot" when only the default case exists); row C9 is `MATCH`. The separate question — whether the four synthetic cases should also run against the real observation — is still open and is a behaviour change, not a wording fix.
- **F-N16 (O3, `NUMBERS DIFFER`).** The fallback dummy observation reports
  `ring_radius = 5.0` while its own `platform.radius` is 7.5 and every real match
  is 7.5; `control_dt`, `proximity_cutoff` and `proximity_limit` are 0.0/0.0/0
  there too. *Smallest fix:* set `BotObservation.ring_radius` to 7.5 and give the
  dummy the real constants. **Recommend changing the code** — no shipped run
  reaches this path today (C8 supplies a real observation), but a controller that
  branches on `ring_radius` would be exercised against a ring that does not
  exist.
  **DECIDED and SHIPPED (Task 15, item L).** `BotObservation.ring_radius` is 7.5 and the dummy observation reports `control_dt` 0.01, `proximity_cutoff` `SURFACE_CUTOFF` and `proximity_limit` `SURFACE_LIMIT`; row O3 is `MATCH`.
- **F-N17 (Y11, `WORDING DIFFERS`).** `rules.yaml` advertises "Timestep and
  gravity within valid ranges"; `validate_simulation_settings` checks gravity
  only. *Smallest fix:* either add a `model.opt.timestep` comparison or drop
  "Timestep and" from the check string. **Recommend changing the code** — the
  arena owns the timestep and a smuggled one would change every physics result,
  so the advertised check is the one worth having.
  **DECIDED and SHIPPED (Task 15, item M) — the other way.** The check string was dropped with the validator it advertised.
- **F-N18 (M5, `SIM-ONLY`).** Floor contacts from geoms that penetrated through
  the ring are ignored (inside `radius − 0.1` and above `top_z − 0.05`).
  **Recommend no change** — a soft-contact artifact filter, not a rule; stating
  it would invite exploitation of the tolerance.
- **F-N19 (M25, `SIM-ONLY`).** The ring's own sliding friction is 2.0 at
  priority 1, so robot geoms (priority 2) always supply the friction of a
  robot-ring contact. *Smallest fix:* one sentence after ARH:148 / SH:123.
  **Recommend changing the prompt** — traction against the ring is a first-order
  design input, and today the model can only infer it.
  **DECIDED and SHIPPED (Task 15, item F).** Both prompts state it in a paragraph
  after the collision paragraph ("The platform's own friction never applies to a
  robot–platform contact: robot geoms have contact priority 2 and the platform
  priority 1, so your material's friction is your traction."); row M25 is `MATCH`.
- **F-N20 (M26, `SIM-ONLY`).** `SumoEnv.__init__` re-runs the actuation and
  single-free-joint checks and raises `ValueError`, which aborts the match (and,
  under the upstream abort-on-infra-error behaviour, the model's whole build run)
  rather than scoring a loss. A submission that passed authoring validation
  cannot reach it, so this is defence in depth. *Smallest fix:* none needed.
  **Recommend no change**, but noted because the failure mode is an infra abort,
  not a graded loss.
- **F-N21 (§H1, modular file).** `game.md:5` says you win only when the opponent
  "falls off the ring and hits the ground", contradicting `game.md:19` below it.
  *Smallest fix:* delete the second half of that sentence. **Recommend changing
  `game.md`** — it is a fallback path, but a contradiction inside one document is
  worse than an omission.
  **RETIRED (Task 15, item R).** `game.md` is deleted; there is no fallback prompt path.
- **F-N22 (§H3, modular file).** `mjcf_syntax.yaml` omits the `<option>` sentence
  and hardcodes "MuJoCo 3.10.0". **Recommend changing `mjcf_syntax.yaml`** to
  match the prompts (or retiring the modular path).
  **RETIRED (Task 15, item R).** `mjcf_syntax.yaml` is deleted; the MJCF section lives only in the two prompts.
- **F-N23 (§H4, modular file).** `obs_schema.yaml`'s `controller_environment`
  omits `import random` and the denied-builtins sentence, so the modular path
  still ships the pre-Task-10 library list. **Recommend changing
  `obs_schema.yaml`** to match the prompts.
  **RETIRED (Task 15, item R).** `obs_schema.yaml` is deleted; the observation schema lives only in the two prompts.
- **F-N24 (§H5).** ARH:615 and SH:590 describe the same proximity code
  differently ("convex surface-distance queries" vs "Primitive distances use
  MuJoCo"). *Smallest fix:* copy the ARH sentence into SH. **Recommend changing
  the SH prompt** — the two documents should describe one engine identically.
  **DECIDED and SHIPPED (Task 15, item H).** The ARH sentence was copied over the SH one.
- **F-N25 (`DEAD` code, §E).** Six pieces of validation never run and impose no
  rule: `validate_2d_physics` (never selected in 3D, and reads
  `config.enforce_2d_physics` / `config.warn_2d_only`, which
  `ModelValidationConfig` does not define — it would raise `AttributeError` if it
  ever did run), `validate_density_constraints` (absent from
  `validate_mujoco_model`'s `CHECK_GROUPS`), `validate_model_mass` and
  `validate_model_size` (imported by `mj_validators` and never called; the latter
  still carries the stale `max_span=10.0`), the legacy `total_gear_budget` /
  `min_single_gear` / `max_single_gear` branches of `validate_torque_budget`, and
  the sliding-friction branch of `validate_material_properties`. One more piece
  of dead code sits in the environment: `compose_sumo_model(enable_selfcollision=…)`
  defaults to `False`, no call site passes `True`, and the branch would write an
  `<option><flag selfcollision>` that is not a MuJoCo 3.x flag (row B43).
  *Smallest fix:* delete them. **Recommend changing the code** — none of it can
  fire, and each piece reads config keys that a future editor could reasonably
  believe are live. Deliberately not done in this task (Task 13 is documentation).
  **DECIDED and SHIPPED (Task 15, item O).** All of it is deleted, with one
  exception: `validate_model_mass` stays importable because the upstream
  `tests_environment/test_joint_damping_rules.py` imports it, with its
  `max_total_mass=1000.0` default removed so no caller can inherit a cap nobody
  configured. `compose_sumo_model`'s `enable_selfcollision` parameter and its
  branch are gone.
- **F-N27 (C10, `WORDING DIFFERS`).** The SH prompt tells the model "There is no
  thinking time limit; the only cap is your maximum output length budget", but
  every LLM call carries a 100-minute timeout, and a call that trips it ends the
  build run. *Smallest fix:* replace that half-sentence with the ARH wording
  ("Use the available reasoning and output budget"), or state the 100-minute cap.
  **Recommend changing the SH prompt** — a model that believes it has unbounded
  thinking time can spend a whole zero-shot sample discovering otherwise. This is
  the one claim in either prompt that is affirmatively false.
  **DECIDED and SHIPPED (Task 15, item C).** Both prompts now state the 100-minute wall-clock limit; the false SH claim is gone and row C10 is `MATCH`.
- **F-N26 (bookkeeping).** `rules.yaml` carries both `robot.size` and
  `robot.bounding_box_3d` with the same 2.44 / 2.44 / 3.05 values, read by
  different consumers (`ModelValidationConfig` vs `dspy_core.py:663`). Not a
  mismatch today; a single source would prevent one from drifting.
  **Recommend changing the code** when convenient.
  **DECIDED and SHIPPED (Task 15, item P).** `robot.bounding_box_3d` is deleted;
  every reader uses `robot.size`. Its only non-test reader,
  `dspy_core.load_constraints_as_string`, was retired with the modular path.
