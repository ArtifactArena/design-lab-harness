"""Unified Engineer field template — the feedback fields one commit carries.

Inputs: the design ledger and two designs — the best commit so far (the anchor to
refine from) and the last commit (the design that the verifier feedback and replay
describe). The call is stateless, so everything the model knows about earlier commits
is what these fields carry. The run's rules, syntax and observation schema are not
fields: they are the consolidated prompt, which is the signature's instruction
(``signatures/autoresearch.py`` deep-copies the fields below under it).
"""
import dspy


class BaselineUnifiedEngineer(dspy.Signature):
    """Design a combat robot: hardware (MJCF XML) and controller (Python)."""

    # --- Where this build is in its budget (first thing the model reads) ---
    commit_status: str = dspy.InputField(
        desc="Which commit this is and how many remain, e.g. 'Commit 12 of 50; 38 remaining after this one.'")

    # --- Dynamic context (changes each commit) ---
    design_ledger: str = dspy.InputField(
        desc="Design ledger: one entry per past commit — commit number, bot name, "
             "FAIL / QUALIFIED: NO / QUALIFIED with a [NEW BEST], [CURRENT BEST] or [REGRESSION] "
             "marker, the qualification score, and that commit's change summary. Ends with "
             "which commit is the current best. Empty on commit 0.",
        default="")
    best_commit_robot_xml: str = dspy.InputField(
        desc="MJCF XML of the BEST commit so far (highest-ranked: qualified beats unqualified, "
             "then qualification score). This is the design to refine from; the tournament entry "
             "is chosen later by a round robin among your qualified commits. Until a commit "
             "passes validation it is the latest attempt. Empty on commit 0.",
        default="")
    best_commit_controller_code: str = dspy.InputField(
        desc="Python policy_step code of that same best commit. Empty on commit 0.",
        default="")
    last_commit_robot_xml: str = dspy.InputField(
        desc="MJCF XML of your LAST commit — the design that last_commit_verifier_feedback and "
             "last_commit_match_replay describe. When the last commit is also the best commit "
             "this reads 'same as best_commit (commit N)' instead of repeating the XML. "
             "Empty on commit 0.",
        default="")
    last_commit_controller_code: str = dspy.InputField(
        desc="Python policy_step code of that same last commit, with the same "
             "'same as best_commit' shortcut. Empty on commit 0.",
        default="")
    last_commit_verifier_feedback: str = dspy.InputField(
        desc="Validation and qualification report for the LAST commit: the morphology checks, "
             "the controller checks, then the qualification matches — each check tagged PASS/FAIL "
             "with exact numbers, each seed's outcome and why it ended, and the score with its "
             "parts. Empty on commit 0.",
        default="")
    last_commit_match_replay: str = dspy.InputField(
        desc="JSON qualification replay of the LAST commit, downsampled: per-seed positions, "
             "actions, combat metrics and termination reason for the lost seeds (or the "
             "lowest-scoring seed if none were lost). Use it to diagnose why the last commit "
             "won/lost/drew against the stationary qualification block. Empty on commit 0.",
        default="")

    # --- Outputs ---
    name: str = dspy.OutputField(
        prefix="Bot Name:",
        desc="A memorable name for your bot design that hints at your strategy.")
    design_strategy: str = dspy.OutputField(
        prefix="Design Strategy:",
        desc="Your overall concept: what kind of bot, what's the win condition, "
             "how hardware and controller work together. This is your elevator pitch.")
    hardware_plan: str = dspy.OutputField(
        prefix="Hardware Plan:",
        desc="Take your design_strategy and work out the physics to make it real. "
             "This is the bridge between strategy and XML — show the engineering math "
             "that justifies your implementation. Include: mass budget (volume x density "
             "per geom), motor mass (sum|gear| x motor_mass_per_gear), gear force vs "
             "traction, mechanism sizing.\n\n"
             "IMPORTANT: End with a section that explicitly maps each physical mechanism "
             "to the attack, movement, and defense strategies from your design_strategy.")
    combat_plan: str = dspy.OutputField(
        prefix="Combat Plan:",
        desc="How the controller fights. Navigation from any spawn position (heading error, "
             "steering), engagement strategy, edge awareness, mode switching. Include concrete "
             "distance/angle thresholds. Stress-test with scenarios: 'if I spawn facing away "
             "at (-3, 2), does my code turn toward the opponent or drive off the edge?'")
    robot_xml: str = dspy.OutputField(
        prefix="MJCF XML:",
        desc="Complete MJCF XML for the robot. Must follow the design instructions exactly. "
             "Every tag MUST include a brief description=\"...\" attribute explaining "
             "what it does and why it exists in context of other parts. "
             "MuJoCo ignores this attribute; it is used for reasoning and visualization.")
    controller_code: str = dspy.OutputField(
        prefix="Python Controller Code:",
        desc="def policy_step(obs) -> dict[str, float]. Must use EXACT actuator names from "
             "robot_xml. All values in [-1, 1]. "
             "Make sure you comment your code to explain what it does.")
    change_summary: str = dspy.OutputField(
        prefix="Change Summary:",
        desc="What you changed and why. This goes into the design ledger alongside your "
             "pass/fail status and combat score — future rounds see the full history. "
             "If the previous commit failed or regressed, ANALYZE WHY based on the verifier "
             "feedback so future rounds avoid the same mistake. "
             "Commit 0: 'Initial design: <concept>'. "
             "Later commits: 'Changed X because Y — previous attempt failed/regressed because Z'.")
