"""Sampling (zero-shot) engineer: the consolidated prompt is the whole call — no inputs.

The performance-first prompt asks for six named sections. ``change_summary`` is kept as
a seventh output because the harness pipeline records it with the submission (the design
ledger, the build summary and the bot artifact all read it).
"""
import dspy

# The six sections of the performance-first prompt, in its order.
OUTPUTS = ["name", "design_strategy", "hardware_plan", "combat_plan", "robot_xml", "controller_code"]


def make_sampling_signature(prompt: str) -> type[dspy.Signature]:
    if not prompt.strip():
        raise ValueError("Sampling prompt must not be empty")
    fields = {
        name: (str, dspy.OutputField(desc=f"The {name} section specified in the design instructions."))
        for name in OUTPUTS
    }
    fields["change_summary"] = (str, dspy.OutputField(
        desc="One or two sentences summarising this design. Not one of the sections above; "
             "the harness records it with the submission."))
    return dspy.make_signature(fields, instructions=prompt, signature_name="SamplingEngineer")
