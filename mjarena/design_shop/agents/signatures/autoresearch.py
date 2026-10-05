"""AutoResearch engineer: the consolidated prompt is the instruction; inputs are the harness feedback fields."""
from copy import deepcopy

import dspy

from .baseline_unified import BaselineUnifiedEngineer

INPUTS = ["commit_status", "design_ledger", "best_commit_robot_xml", "best_commit_controller_code",
          "last_commit_robot_xml", "last_commit_controller_code",
          "last_commit_verifier_feedback", "last_commit_match_replay"]
OUTPUTS = ["name", "improvement_plan", "design_strategy", "hardware_plan", "combat_plan",
           "robot_xml", "controller_code", "change_summary"]

# Selection rules belong to the consolidated document, which says the harness keeps the
# best design when a revision regresses. The modular signature's description promises a
# round robin among qualified commits, which neither harness runs — drop it here only.
_UNIMPLEMENTED_SELECTION_CLAUSE = (
    "; the tournament entry "
    "is chosen later by a round robin among your qualified commits")


def make_autoresearch_signature(prompt: str) -> type[dspy.Signature]:
    if not prompt.strip():
        raise ValueError("AutoResearch prompt must not be empty")
    fields = {}
    for name in INPUTS:
        source = BaselineUnifiedEngineer.input_fields[name]
        field = deepcopy(source)
        desc = field.json_schema_extra.get("desc", "")
        if _UNIMPLEMENTED_SELECTION_CLAUSE in desc:
            field.json_schema_extra["desc"] = desc.replace(_UNIMPLEMENTED_SELECTION_CLAUSE, "")
        fields[name] = (source.annotation, field)
    for name in OUTPUTS:
        fields[name] = (str, dspy.OutputField(desc=f"The {name} section specified in the design instructions."))
    return dspy.make_signature(fields, instructions=prompt, signature_name="AutoResearchEngineer")
