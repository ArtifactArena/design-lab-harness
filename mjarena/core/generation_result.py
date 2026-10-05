"""GenerationResult — return type for the unified builder.

`unified_builder.generate_unified()` returns this dataclass so that
`bot_builder.build_bot()` can run shared post-processing (debug JSON,
prompt dumps, video concat, artifact assembly).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mjarena.agents.types import VerificationResult


@dataclass
class GenerationResult:
    """Common result from both split and unified generation paths."""

    # Core outputs
    robot_xml: str  # Raw LLM-produced XML (pre-pipeline)
    processed_xml: str  # Post-pipeline XML (materials, mass injected)
    controller_code: str
    actuator_names: List[str]

    # Morphology design fields
    overview: str = ""
    design_calculations: str = ""
    movement: str = ""
    attack: str = ""
    defense: str = ""
    bot_design: str = ""  # Assembled from above + robot_xml

    # Controller design fields
    control_calculations: str = ""
    strategy_plan: str = ""
    movement_logic: str = ""
    attack_logic: str = ""
    defense_logic: str = ""
    actuator_plan: str = ""

    # Scores
    morphology_score: float = 0.0
    controller_score: float = 0.0

    # Verification
    morphology_verification: Optional[VerificationResult] = None
    controller_verification: Optional[VerificationResult] = None

    # Timing
    morphology_generation_time_sec: float = 0.0
    controller_generation_time_sec: float = 0.0

    # Debug (LM history + reasoning for each phase)
    morph_lm_history: List[Any] = field(default_factory=list)
    ctrl_lm_history: List[Any] = field(default_factory=list)
    morph_reasoning: str = ""
    ctrl_reasoning: str = ""

    # Per-round Creator/Critic history (for gameplay/ output)
    morph_round_history: List[Dict[str, Any]] = field(default_factory=list)
    ctrl_round_history: List[Dict[str, Any]] = field(default_factory=list)

    # Which path produced this result
    generation_mode: str = "split"  # "split" or "unified"

    # Controller feedback text (from verifier, for prompt dumps)
    controller_feedback: str = ""

    @property
    def total_generation_time_sec(self) -> float:
        return self.morphology_generation_time_sec + self.controller_generation_time_sec
