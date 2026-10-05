"""Shared result types for the verification and refinement pipeline.

Single source of truth for VerifierResult, ControllerResult, RoundEntry,
BuildMorphologyResult, BuildControllerResult, and GameRecord.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VerifierResult:
    """Result from a single validation step."""

    passed: bool
    label: str
    message: str
    skipped: bool = False
    quiet: bool = False                  # passed with nothing worth telling the model; not printed
    score: Optional[float] = None        # numeric score (match tools only, None for static checks)
    data: Dict[str, Any] = field(default_factory=dict)  # lightweight metadata from the step
    name: str = ""                       # prefixed type: "verifier:compile", "tool:stationary", etc.
    elapsed_sec: float = 0.0             # wall-clock time for this step
    matchup: Optional[MatchupResult] = None  # full matchup result (forward ref, defined below)


@dataclass
class ControllerResult:
    """Result from the full controller validation pipeline. Parallel to MorphologyResult."""

    passed: bool
    score: float              # mean across match tool scores, -1 static fail, 0 no matches
    feedback: str             # concatenated step messages (same format as morphology)
    step_results: List[VerifierResult] = field(default_factory=list)
    qualification: Optional[VerifierResult] = None  # VerifierResult from qualify_round


@dataclass
class UnifiedEnvResult:
    """Full result from the unified verifier pipeline for one commit."""

    feedback: str
    score: float = -1.0  # qualification / controller score
    qualification: Optional[VerifierResult] = None


@dataclass
class CommitEntry:
    """One commit in the refinement journal."""

    commit_num: int
    score: float
    is_best: bool  # True if this became the new best
    best_at_time: int  # which commit was best when this ran
    creator_summary: str  # 1-2 sentence overview of what was tried
    diff_from_best: str  # what changed vs best (empty if this IS best)
    verifier_feedback: str  # formatted verification results
    validation_passed: bool = True  # Morphology + controller static checks passed
    qualification_passed: bool = False  # Bot passes qualification round
    bot_name: str = ""  # Bot name from Engineer output
    scores: Dict[str, Any] = field(default_factory=dict)  # Structured scores from qualification
    # Engineer output fields (from BaselineUnifiedEngineer signature)
    reasoning: str = ""  # Chain-of-thought reasoning
    improvement_plan: str = ""  # Feedback diagnosis and intended changes for this commit
    design_strategy: str = ""  # Overall concept / elevator pitch
    hardware_plan: str = ""  # Physics engineering / mass budget
    combat_plan: str = ""  # Controller fighting strategy


# Backwards compat alias
RoundEntry = CommitEntry


@dataclass
class BuildMorphologyResult:
    """Full result from morphology refinement (one candidate lineage)."""

    passed: bool
    score: float
    robot_xml: str  # raw XML from Creator
    processed_xml: str  # post-pipeline XML
    actuator_names: List[str]

    # Design fields
    overview: str = ""
    design_calculations: str = ""
    movement: str = ""
    attack: str = ""
    defense: str = ""
    bot_design: str = ""  # assembled for controller handoff

    # Verification
    verifier_feedback: str = ""  # feedback for the winning design
    step_results: List[VerifierResult] = field(default_factory=list)

    # Refinement
    rounds_run: int = 0
    best_round: int = 0
    journal: List[RoundEntry] = field(default_factory=list)

    # Meta
    elapsed_sec: float = 0.0
    reasoning: str = ""

    # Legacy compat — round_history for gameplay saving
    round_history: List[Dict[str, Any]] = field(default_factory=list)
    lm_history: List[Any] = field(default_factory=list)


@dataclass
class BuildControllerResult:
    """Full result from controller refinement (one candidate lineage)."""

    passed: bool
    score: float
    controller_code: str

    # Strategy fields
    strategy_plan: str = ""
    movement_logic: str = ""
    attack_logic: str = ""
    defense_logic: str = ""

    # Verification
    verifier_feedback: str = ""
    match_data: Dict[str, Any] = field(default_factory=dict)  # per-verifier per-seed replay

    # Refinement
    rounds_run: int = 0
    best_round: int = 0
    journal: List[RoundEntry] = field(default_factory=list)

    # Meta
    elapsed_sec: float = 0.0
    reasoning: str = ""

    # Legacy compat
    round_history: List[Dict[str, Any]] = field(default_factory=list)
    lm_history: List[Any] = field(default_factory=list)


@dataclass
class GameRecord:
    """Complete recording of one match seed.

    Contains everything needed to reconstruct a match visually and compute
    any scoring metric. NOT a score — it's the raw recording.
    """

    # Match outcome
    seed: int                            # actual random seed used for this match
    winner: str                          # "red", "blue", "tie"
    num_steps: int
    winner_step: Optional[int]
    progress: float                      # fraction of initial distance closed (0-1)

    # Arena
    ring_radius: float

    # Bot static properties
    red_mass: float
    blue_mass: float
    red_bounding_radius: float
    blue_bounding_radius: float
    red_actuator_names: List[str]
    blue_actuator_names: List[str]

    # Per-step timeseries (Lists, length = num_steps)
    red_positions: List[List[float]]     # [[x,y,z], ...]
    blue_positions: List[List[float]]
    red_orientations: List[Dict[str, float]]   # [{"yaw":, "pitch":, "roll":}, ...]
    blue_orientations: List[Dict[str, float]]
    red_velocities: List[List[float]]
    blue_velocities: List[List[float]]
    red_angular_velocities: List[List[float]]
    blue_angular_velocities: List[List[float]]
    distances_to_opponent: List[float]
    red_edge_distances: List[float]
    blue_edge_distances: List[float]
    contacts: List[bool]
    contact_forces: List[float]
    red_tipping: List[float]             # 0=upright, 1=fallen
    blue_tipping: List[float]
    red_ground_contacts: List[bool]
    blue_ground_contacts: List[bool]
    blue_displacements: List[float]      # meters moved per step
    red_actions: List[Dict[str, float]]
    blue_actions: List[Dict[str, float]]

    # Physics stability
    physics_unstable: bool
    qacc_warning_steps: List[int]

    # Initial state
    initial_red_pos: List[float]         # [x, y, z]
    initial_blue_pos: List[float]
    initial_distance: float

    # Termination
    # "ring_out" | "size_violation" | "inactivity" | "qacc" | "forfeit_crash"
    # | "timeout" | "draw"
    termination_reason: str = "unknown"


    # Precomputed combat metrics (convenience)
    combat_metrics: Optional[Dict[str, float]] = None

    # QACC / gear diagnostics from physics checks
    qacc_gear_diagnostics: List[Dict] = field(default_factory=list)
    qacc_loser: Optional[str] = None          # "red", "blue", "both"
    qacc_dof_index: Optional[int] = None      # DOF MuJoCo blamed
    qacc_body_name: Optional[str] = None      # body that owns that DOF

    # Inactivity timer timeseries (per-step)
    red_inactivity_timers: List[float] = field(default_factory=list)
    blue_inactivity_timers: List[float] = field(default_factory=list)

    # Full joint state per frame (for Blender re-rendering)
    # Each entry is the full qpos array as a list of floats.
    # Not sent to LLMs (excluded from downsample).
    qpos: List[List[float]] = field(default_factory=list)
    initial_qpos: List[float] = field(default_factory=list)

    # Control timestep in seconds (time between qpos frames)
    control_dt: float = 0.01

    # First controller runtime exception observed for each bot, if any.
    # Entries are JSON-safe dictionaries with color, step, type, and message.
    controller_errors: List[Dict[str, Any]] = field(default_factory=list)
    red_surface_distances: List[float] = field(default_factory=list)

    @property
    def result(self) -> str:
        """Match result from red bot's perspective: 'win', 'loss', or 'draw'."""
        if self.winner == "red":
            return "win"
        elif self.winner == "blue":
            return "loss"
        return "draw"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all fields to a JSON-compatible dict."""
        return dataclasses.asdict(self)

    # Series worth showing the model. The opponent's orientation/velocity/tipping/
    # action/inactivity series are dropped (trivial for the stationary block), as
    # are qpos, control_dt and diagnostics that are empty unless something failed.
    _PROMPT_SERIES = (
        "red_positions", "red_orientations", "red_velocities", "red_angular_velocities",
        "red_edge_distances", "red_tipping", "red_ground_contacts", "red_inactivity_timers",
        "red_actions", "blue_positions", "blue_edge_distances", "blue_displacements",
        "distances_to_opponent", "contacts", "contact_forces",
    )

    def to_prompt_dict(self) -> Dict[str, Any]:
        """The replay as pasted into the Engineer prompt: informative fields only."""
        out: Dict[str, Any] = {
            "seed": self.seed,
            "winner": self.winner,
            "termination_reason": self.termination_reason,
            "num_steps": self.num_steps,
            "winner_step": self.winner_step,
            "ring_radius": self.ring_radius,
            "red_mass": self.red_mass,
            "blue_mass": self.blue_mass,
            "red_bounding_radius": self.red_bounding_radius,
            "blue_bounding_radius": self.blue_bounding_radius,
            "red_actuator_names": list(self.red_actuator_names),
            "initial_red_pos": self.initial_red_pos,
            "initial_blue_pos": self.initial_blue_pos,
            "initial_distance": self.initial_distance,
            "combat_metrics": self.combat_metrics,
        }
        for name in self._PROMPT_SERIES:
            out[name] = getattr(self, name)
        if self.physics_unstable:
            out["physics_unstable"] = True
            out["qacc_warning_steps"] = self.qacc_warning_steps
            out["qacc_gear_diagnostics"] = self.qacc_gear_diagnostics
        return out

    def downsample(self, step_interval: int) -> Dict[str, Any]:
        """Downsample timeseries and return a serializable dict.

        Args:
            step_interval: Take every Nth step (e.g., 50 for 2Hz at 100Hz control).

        Returns:
            Dict with downsampled timeseries + match metadata.
        """
        step_interval = max(1, step_interval)
        indices = list(range(0, self.num_steps, step_interval))
        # Always include the last step
        if indices and indices[-1] != self.num_steps - 1:
            indices.append(self.num_steps - 1)

        def _sample(series):
            return [series[i] for i in indices if i < len(series)]

        return {
            "result": self.result,
            "winner": self.winner,
            "termination_reason": self.termination_reason,
            "num_steps": self.num_steps,
            "sampled_steps": len(indices),
            "step_interval": step_interval,
            "combat_metrics": self.combat_metrics,
            "physics_unstable": self.physics_unstable,
            "qacc_warning_steps": self.qacc_warning_steps,
            "qacc_gear_diagnostics": self.qacc_gear_diagnostics,
            "qacc_loser": self.qacc_loser,
            "qacc_dof_index": self.qacc_dof_index,
            "qacc_body_name": self.qacc_body_name,
            "controller_errors": self.controller_errors,
            "red_positions": _sample(self.red_positions),
            "blue_positions": _sample(self.blue_positions),
            "red_orientations": _sample(self.red_orientations),
            "blue_orientations": _sample(self.blue_orientations),
            "red_velocities": _sample(self.red_velocities),
            "blue_velocities": _sample(self.blue_velocities),
            "distances_to_opponent": _sample(self.distances_to_opponent),
            "red_edge_distances": _sample(self.red_edge_distances),
            "blue_edge_distances": _sample(self.blue_edge_distances),
            "contacts": _sample(self.contacts),
            "contact_forces": _sample(self.contact_forces),
            "red_tipping": _sample(self.red_tipping),
            "blue_tipping": _sample(self.blue_tipping),
            "red_actions": _sample(self.red_actions),
            "blue_actions": _sample(self.blue_actions),
            "red_inactivity_timers": _sample(self.red_inactivity_timers),
            "blue_inactivity_timers": _sample(self.blue_inactivity_timers),
        }

    @staticmethod
    def select_for_replay(
        records: List["GameRecord"],
        combat_scores: List[float],
        replay_hz: float = 2.0,
        control_hz: float = 100.0,
    ) -> Dict[str, Any]:
        """Select the most informative seeds and downsample for LLM replay.

        Selection logic:
        - Include all losses (bot needs to learn from failures)
        - If no losses, include the seed with the lowest combat score

        Args:
            records: GameRecords from qualification (one per seed).
            combat_scores: Per-seed combat scores (same order as records).
            replay_hz: Replay sample rate in Hz.
            control_hz: Physics control rate in Hz.

        Returns:
            Dict with selected seeds, each downsampled.
        """
        step_interval = max(1, round(control_hz / replay_hz))

        # Single seed: just downsample it directly without loss-selection logic
        if len(records) == 1:
            entry = records[0].downsample(step_interval)
            entry["seed"] = 0
            entry["combat_score"] = round(combat_scores[0], 2)
            return {
                "n_total_seeds": 1,
                "n_selected_seeds": 1,
                "selection_reason": "single_seed",
                "replay_hz": replay_hz,
                "seeds": [entry],
            }

        # Find losses
        selected = [
            (i, r, s) for i, (r, s) in enumerate(zip(records, combat_scores))
            if r.result == "loss"
        ]

        # If no losses, pick the seed with the lowest combat score
        if not selected:
            worst_idx = min(range(len(combat_scores)), key=lambda i: combat_scores[i])
            selected = [(worst_idx, records[worst_idx], combat_scores[worst_idx])]

        seeds = []
        for seed_idx, record, score in selected:
            entry = record.downsample(step_interval)
            entry["seed"] = seed_idx
            entry["combat_score"] = round(score, 2)
            seeds.append(entry)

        return {
            "n_total_seeds": len(records),
            "n_selected_seeds": len(seeds),
            "selection_reason": "losses" if any(r.result == "loss" for _, r, _ in selected) else "lowest_score",
            "replay_hz": replay_hz,
            "seeds": seeds,
        }


@dataclass
class MatchupResult:
    """Result of running N seeds between two bots. Pure data — no policy decisions."""
    game_records: List[GameRecord]       # one per seed, full timeseries
    combat_scores: Any                   # List[CombatScore] — Any avoids circular import
    seed_scores: List[float]             # combined score per seed (0-6 range)
    elapsed_sec: float                   # wall-clock time for all seeds
    video_fps: Optional[int] = None      # from first seed that recorded video

    @property
    def avg_score(self) -> float:
        return round(sum(self.seed_scores) / len(self.seed_scores), 2) if self.seed_scores else 0.0

    @property
    def winners(self) -> List[str]:
        return [r.winner for r in self.game_records]

    @property
    def n_seeds(self) -> int:
        return len(self.game_records)

    def n_wins(self, perspective: str = "red") -> int:
        return sum(1 for r in self.game_records if r.winner == perspective)

    def n_losses(self, perspective: str = "red") -> int:
        opponent = "blue" if perspective == "red" else "red"
        return sum(1 for r in self.game_records if r.winner == opponent)

    @property
    def n_ties(self) -> int:
        return sum(1 for r in self.game_records if r.winner not in ("red", "blue"))

    @property
    def qacc_failures(self) -> List[GameRecord]:
        return [r for r in self.game_records if r.physics_unstable]

    def seed_summary_lines(self, perspective: str = "red") -> List[str]:
        lines = []
        for i, (record, cs) in enumerate(zip(self.game_records, self.combat_scores)):
            if record.winner == perspective:
                outcome = "WIN"
            elif record.winner not in ("red", "blue"):
                outcome = "DRAW"
            else:
                outcome = "LOSS"
            composite = cs.composite if hasattr(cs, "composite") else 0.0
            dt = float(getattr(record, "control_dt", 0.01) or 0.01)
            reason = str(getattr(record, "termination_reason", "") or "")
            me = perspective
            if reason == "ring_out":
                why = "you fell off the ring" if record.winner != me else "opponent fell off the ring"
            elif reason == "inactivity":
                why = ("you tripped the inactivity rule (stalled 10 s)" if record.winner != me
                       else "opponent tripped the inactivity rule")
            elif reason == "size_violation":
                why = ("your robot extended past the size box (root-frame extents "
                       "over 2.44 × 2.44 × 3.05 m)" if record.winner != me
                       else "opponent extended past the size box")
            elif reason == "qacc":
                why = "physics instability"
            elif reason == "timeout":
                why = "time ran out"
            else:
                why = reason or "unknown"
            lines.append(
                f"  Seed {i}: {outcome} — {why} at {record.num_steps * dt:.1f}s "
                f"(combat={composite:.2f}/5)"
            )
        return lines

    def combat_feedback_lines(self) -> List[str]:
        return [
            f"[Seed {i}] {cs.feedback_text}"
            for i, cs in enumerate(self.combat_scores)
        ]

    def to_prompt_dict(self, replay_hz: float = 2.0) -> Dict[str, Any]:
        """Prompt-facing replay: per-seed GameRecord.to_prompt_dict() + seed scores."""
        return {
            "replay_hz": replay_hz,
            "n_seeds_shown": len(self.game_records),
            "seed_scores": list(self.seed_scores),
            "combat_scores": [
                dataclasses.asdict(cs) if dataclasses.is_dataclass(cs) else cs
                for cs in self.combat_scores
            ],
            "games": [r.to_prompt_dict() for r in self.game_records],
        }

    def downsample(self, replay_hz: float = 2.0, control_hz: float = 100.0) -> "MatchupResult":
        """Return a new MatchupResult filtered to informative seeds and downsampled.

        Steps:
        1. If 1 seed: keep it. If N seeds: pick all losses; if no losses, pick lowest score.
        2. Downsample each selected GameRecord (take every Nth step).
        3. Filter combat_scores and seed_scores to match selected seeds.

        Returns a new MatchupResult with fewer seeds and shorter timeseries.
        """
        step_interval = max(1, round(control_hz / replay_hz))

        if len(self.game_records) <= 1:
            selected_indices = list(range(len(self.game_records)))
        else:
            # Pick all losses
            selected_indices = [i for i, r in enumerate(self.game_records) if r.winner == "blue"]
            if not selected_indices:
                # No losses - pick lowest seed score
                selected_indices = [min(range(len(self.seed_scores)), key=lambda i: self.seed_scores[i])]

        # Create new GameRecords with downsampled timeseries
        new_records = []
        for idx in selected_indices:
            record = self.game_records[idx]
            indices = list(range(0, record.num_steps, step_interval))
            if indices and indices[-1] != record.num_steps - 1:
                indices.append(record.num_steps - 1)

            def _sample(series, _indices=indices):
                return [series[i] for i in _indices if i < len(series)]

            new_record = GameRecord(
                winner=record.winner,
                num_steps=record.num_steps,  # keep original for context
                winner_step=record.winner_step,
                progress=record.progress,
                ring_radius=record.ring_radius,
                red_mass=record.red_mass,
                blue_mass=record.blue_mass,
                red_bounding_radius=record.red_bounding_radius,
                blue_bounding_radius=record.blue_bounding_radius,
                red_actuator_names=record.red_actuator_names,
                blue_actuator_names=record.blue_actuator_names,
                red_positions=_sample(record.red_positions),
                blue_positions=_sample(record.blue_positions),
                red_orientations=_sample(record.red_orientations),
                blue_orientations=_sample(record.blue_orientations),
                red_velocities=_sample(record.red_velocities),
                blue_velocities=_sample(record.blue_velocities),
                red_angular_velocities=_sample(record.red_angular_velocities),
                blue_angular_velocities=_sample(record.blue_angular_velocities),
                distances_to_opponent=_sample(record.distances_to_opponent),
                red_edge_distances=_sample(record.red_edge_distances),
                blue_edge_distances=_sample(record.blue_edge_distances),
                contacts=_sample(record.contacts),
                contact_forces=_sample(record.contact_forces),
                red_tipping=_sample(record.red_tipping),
                blue_tipping=_sample(record.blue_tipping),
                red_ground_contacts=_sample(record.red_ground_contacts),
                blue_ground_contacts=_sample(record.blue_ground_contacts),
                blue_displacements=_sample(record.blue_displacements),
                red_actions=_sample(record.red_actions),
                blue_actions=_sample(record.blue_actions),
                physics_unstable=record.physics_unstable,
                qacc_warning_steps=record.qacc_warning_steps,
                initial_red_pos=record.initial_red_pos,
                initial_blue_pos=record.initial_blue_pos,
                initial_distance=record.initial_distance,
                termination_reason=record.termination_reason,
                combat_metrics=record.combat_metrics,
                qacc_gear_diagnostics=record.qacc_gear_diagnostics,
                red_inactivity_timers=_sample(record.red_inactivity_timers),
                blue_inactivity_timers=_sample(record.blue_inactivity_timers),
                seed=record.seed,
                controller_errors=list(record.controller_errors),
            )
            new_records.append(new_record)

        new_combat_scores = [self.combat_scores[i] for i in selected_indices]
        new_seed_scores = [self.seed_scores[i] for i in selected_indices]

        return MatchupResult(
            game_records=new_records,
            combat_scores=new_combat_scores,
            seed_scores=new_seed_scores,
            elapsed_sec=self.elapsed_sec,
            video_fps=self.video_fps,
        )
