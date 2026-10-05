"""
DSPy metrics and reward functions for arena evaluation.

Provides:
- CombatScore / compute_combat_score / compute_seed_score: Combat-aware scoring
- make_reward_fn: Thin wrapper around validate_controller for BestOfN/Refine
- run_rollouts: Simple multi-seed rollout runner for tournaments (returns raw results)
- arena_metric: DSPy metric for evaluator/optimizer use
"""
from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from mjarena.eval.runtime import build_policy_callable


def _get_validate_controller():
    from mjarena.design_shop.pipelines.controller_pipeline import validate_controller
    return validate_controller

logger = logging.getLogger(__name__)


# =============================================================================
# Combat Scoring
# =============================================================================


@dataclass
class CombatScore:
    """Combat metrics from a single match obs_log.

    Each metric is 0.0–1.0. Composite range: 0.0–5.0.

    Metrics:
        engagement: Fraction of steps with opponent contact.
        dominant_contact: Fraction of contact steps where blue (opponent) displaced.
        displacement: Cumulative blue displacement during contact / ring_diameter, clamped [0,1].
        destabilization: max(blue_is_tipping) across all steps.
        self_stability: 1.0 - max(red_is_tipping) across all steps.
        composite: Sum of all 5 metrics (0.0–5.0).
        feedback_text: Human-readable summary of combat performance.
    """
    engagement: float
    dominant_contact: float
    displacement: float
    destabilization: float
    self_stability: float
    composite: float
    feedback_text: str


@dataclass
class CombatV2Score:
    """Post-hoc combat score fitted to tournament point rate.

    This metric is computed only from the red/current bot's qualification match
    replay against the stationary baseline cube. The coefficients were fitted
    post hoc on existing Stage1/Stage2 bots to predict tournament point rate.
    """

    score: float
    predicted_point_rate: float
    raw_predicted_point_rate: float
    features: Dict[str, float]
    feedback_text: str


_COMBATV2_SCORE_COEFFICIENTS = {
    "intercept": 0.352272,
    "blue_tip_max": (0.093818, 0.337048, 0.332157),
    "outward_mean": (-0.050783, 0.760818, 1.033325),
    "signed_action_change": (0.066879, 0.029160, 0.051055),
    "final_closure": (0.047672, 0.238768, 0.349764),
    "signed_steps": (-0.030125, 238.317073, 238.647214),
    "signed_red_speed": (-0.060090, 1.938636, 2.214461),
    "signed_finish": (0.031805, 0.657264, 0.157677),
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _mean_numeric(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(clean) / len(clean) if clean else 0.0


def _xy(point: Any) -> Optional[tuple[float, float]]:
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    return None


def _xy_speed(vector: Any) -> float:
    if isinstance(vector, (list, tuple)) and len(vector) >= 2:
        return math.hypot(float(vector[0]), float(vector[1]))
    return 0.0


def _record_outcome_from_red(record: Any) -> int:
    if getattr(record, "winner", None) == "red":
        return 1
    if getattr(record, "winner", None) == "blue":
        return -1
    return 0


def _action_change_mean(actions: Sequence[Mapping[str, float]]) -> float:
    changes: List[float] = []
    previous: Optional[List[float]] = None
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        values = [
            float(value)
            for value in action.values()
            if isinstance(value, (int, float))
        ]
        if previous and len(previous) == len(values):
            changes.append(
                sum(abs(value - prev) for value, prev in zip(values, previous))
                / len(values)
            )
        if values:
            previous = values
    return _mean_numeric(changes)


def _outward_velocity_mean(
    positions: Sequence[Sequence[float]],
    velocities: Sequence[Sequence[float]],
) -> float:
    radial_velocities: List[float] = []
    for position, velocity in zip(positions, velocities):
        pos_xy = _xy(position)
        if pos_xy is None or not isinstance(velocity, (list, tuple)) or len(velocity) < 2:
            continue
        radius = math.hypot(pos_xy[0], pos_xy[1]) or 1.0
        radial_velocities.append(
            (pos_xy[0] * float(velocity[0]) + pos_xy[1] * float(velocity[1]))
            / radius
        )
    return _mean_numeric(radial_velocities)


def compute_combatv2_score(records: Sequence[Any]) -> CombatV2Score:
    """Compute the CombatV2 score.

    Args:
        records: Full qualification GameRecord objects. The red/current bot is
            assumed to be the candidate, and the blue bot is the baseline cube.

    Returns:
        CombatV2Score on the same 0-6 scale used by build scores.
    """
    per_seed: List[Dict[str, float]] = []

    for record in records:
        outcome = _record_outcome_from_red(record)
        steps = float(getattr(record, "num_steps", 1000) or 1000.0)
        signed_finish_raw = outcome * _clamp(1.0 - steps / 1000.0)

        distances = list(getattr(record, "distances_to_opponent", []) or [])
        initial_distance = float(
            getattr(record, "initial_distance", 0.0)
            or (distances[0] if distances else 0.0)
            or 0.0
        )
        final_distance = float(distances[-1]) if distances else initial_distance
        final_closure = (
            (initial_distance - final_distance) / max(initial_distance, 1e-6)
            if initial_distance
            else 0.0
        )

        red_positions = list(getattr(record, "red_positions", []) or [])
        red_velocities = list(getattr(record, "red_velocities", []) or [])
        blue_tipping = list(getattr(record, "blue_tipping", []) or [])
        red_actions = list(getattr(record, "red_actions", []) or [])

        per_seed.append({
            "outcome": float(outcome),
            "signed_finish_raw": signed_finish_raw,
            "steps": steps,
            "blue_tip_max": max(
                [float(v) for v in blue_tipping if isinstance(v, (int, float))],
                default=0.0,
            ),
            "outward_mean": _outward_velocity_mean(red_positions, red_velocities),
            "action_change": _action_change_mean(red_actions),
            "red_speed": _mean_numeric([_xy_speed(v) for v in red_velocities]),
            "final_closure": final_closure,
        })

    if not per_seed:
        features = {
            "signed_finish": 0.5,
            "blue_tip_max": 0.0,
            "outward_mean": 0.0,
            "signed_action_change": 0.0,
            "final_closure": 0.0,
            "signed_steps": 0.0,
            "signed_red_speed": 0.0,
        }
    else:
        features = {
            "signed_finish": (
                1.0 + _mean_numeric([p["signed_finish_raw"] for p in per_seed])
            ) / 2.0,
            "blue_tip_max": _mean_numeric([p["blue_tip_max"] for p in per_seed]),
            "outward_mean": _mean_numeric([p["outward_mean"] for p in per_seed]),
            "signed_action_change": _mean_numeric([
                p["outcome"] * p["action_change"] for p in per_seed
            ]),
            "final_closure": _mean_numeric([p["final_closure"] for p in per_seed]),
            "signed_steps": _mean_numeric([
                p["outcome"] * p["steps"] for p in per_seed
            ]),
            "signed_red_speed": _mean_numeric([
                p["outcome"] * p["red_speed"] for p in per_seed
            ]),
        }

    raw_predicted = float(_COMBATV2_SCORE_COEFFICIENTS["intercept"])
    for name in (
        "blue_tip_max",
        "outward_mean",
        "signed_action_change",
        "final_closure",
        "signed_steps",
        "signed_red_speed",
        "signed_finish",
    ):
        coef, mean, std = _COMBATV2_SCORE_COEFFICIENTS[name]
        raw_predicted += coef * ((features[name] - mean) / std)

    predicted = _clamp(raw_predicted)
    score = 6.0 * predicted
    feedback_lines = [
        (
            "CombatV2 score: "
            f"{score:.2f}/6.00 (predicted tournament point rate={predicted:.3f})"
        ),
        "  Uses qualification replay only: current bot vs baseline cube.",
        f"  blue_tip_max:          {features['blue_tip_max']:.3f}",
        f"  outward_mean:          {features['outward_mean']:.3f}",
        f"  signed_action_change:  {features['signed_action_change']:.3f}",
        f"  final_closure:         {features['final_closure']:.3f}",
        f"  signed_steps:          {features['signed_steps']:.1f}",
        f"  signed_red_speed:      {features['signed_red_speed']:.3f}",
        f"  signed_finish:         {features['signed_finish']:.3f}",
    ]

    return CombatV2Score(
        score=score,
        predicted_point_rate=predicted,
        raw_predicted_point_rate=raw_predicted,
        features=features,
        feedback_text="\n".join(feedback_lines),
    )


QualificationMechanicsScore = CombatV2Score
CubeReplayMetricScore = CombatV2Score


def compute_qualification_mechanics_score(records: Sequence[Any]) -> CombatV2Score:
    """Backward-compatible alias for compute_combatv2_score."""
    return compute_combatv2_score(records)


def compute_cube_replay_metric(records: Sequence[Any]) -> CombatV2Score:
    """Backward-compatible alias for compute_combatv2_score."""
    return compute_combatv2_score(records)


def compute_combat_score(source, flip: bool = False) -> CombatScore:
    """Compute combat metrics from an obs_log or GameRecord.

    Args:
        source: Either a List[Dict] obs_log (legacy) or a GameRecord (columnar).
        flip: score the BLUE bot instead of RED (GameRecord path only). Lets the
            caller get a side-neutral pair of scores (model_a AND model_b) from
            one match. Ignored for the legacy obs_log path (red-only).

    Returns:
        CombatScore with all 5 metrics computed.
    """
    from mjarena.design_shop.types import GameRecord

    if isinstance(source, GameRecord):
        return _combat_score_from_game_record(source, flip=flip)

    obs_log = source
    if not obs_log:
        return CombatScore(
            engagement=0.0, dominant_contact=0.0, displacement=0.0,
            destabilization=0.0, self_stability=1.0, composite=1.0,
            feedback_text="No obs_log data.",
        )

    total_steps = len(obs_log)
    ring_radius = obs_log[0].get("ring_radius", 5.0)
    ring_diameter = ring_radius * 2.0

    # Metric 1: Contact engagement
    contact_steps = sum(1 for o in obs_log if o.get("contact", False))
    engagement = contact_steps / total_steps if total_steps > 0 else 0.0

    # Metric 2: Dominant contact (contact steps where blue moved)
    contact_steps_blue_moved = 0
    for o in obs_log:
        if o.get("contact", False) and o.get("blue_displacement", 0.0) > 1e-4:
            contact_steps_blue_moved += 1
    dominant_contact = (
        contact_steps_blue_moved / contact_steps if contact_steps > 0 else 0.0
    )

    # Metric 3: Contact displacement
    total_blue_disp_during_contact = 0.0
    for o in obs_log:
        if o.get("contact", False):
            total_blue_disp_during_contact += o.get("blue_displacement", 0.0)
    displacement = min(1.0, total_blue_disp_during_contact / ring_diameter) if ring_diameter > 0 else 0.0

    # Metric 4: Opponent destabilization
    destabilization = max((o.get("blue_is_tipping", 0.0) for o in obs_log), default=0.0)

    # Metric 5: Self stability
    max_red_tipping = max((o.get("red_is_tipping", 0.0) for o in obs_log), default=0.0)
    self_stability = 1.0 - max_red_tipping

    composite = engagement + dominant_contact + displacement + destabilization + self_stability

    # Build feedback text
    lines = [
        f"Combat score: {composite:.2f}/5.00",
        f"  engagement:      {engagement:.2f} ({contact_steps}/{total_steps} steps with contact)",
        f"  dominant_contact: {dominant_contact:.2f} ({contact_steps_blue_moved}/{contact_steps} contact steps with blue displacement)",
        f"  displacement:    {displacement:.2f} (blue displaced {total_blue_disp_during_contact:.2f}m during contact, ring_diam={ring_diameter:.1f}m)",
        f"  destabilization: {destabilization:.2f} (max blue tipping)",
        f"  self_stability:  {self_stability:.2f} (1 - max red tipping {max_red_tipping:.2f})",
    ]
    feedback_text = "\n".join(lines)

    return CombatScore(
        engagement=engagement,
        dominant_contact=dominant_contact,
        displacement=displacement,
        destabilization=destabilization,
        self_stability=self_stability,
        composite=composite,
        feedback_text=feedback_text,
    )


def _xy_step_disp(positions) -> list:
    """Per-step horizontal (xy) distance travelled, length == len(positions)
    (leading 0.0 so it aligns index-for-index with the contacts/steps arrays).
    Used to reconstruct the RED bot's per-step displacement (only blue's is
    stored as a column) so combat metrics can be scored from EITHER side."""
    out = [0.0]
    for a, b in zip(positions, positions[1:]):
        out.append(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5)
    return out


def _combat_score_from_game_record(record, flip: bool = False) -> CombatScore:
    """Compute combat metrics directly from GameRecord columnar fields.

    flip=False -> score the RED bot (candidate; original behaviour): opponent is
    blue. flip=True -> score the BLUE bot symmetrically: opponent is red. The
    record has no red_displacements column, so red's per-step displacement is
    reconstructed from red_positions. Everything else is a straight red<->blue
    swap. engagement is symmetric. This makes match_result.json side-neutral:
    the same match yields a combat score for BOTH bots (model_a and model_b),
    not just red — so behavioural stats are unbiased regardless of red/blue slot."""
    total_steps = record.num_steps
    if total_steps == 0:
        return CombatScore(
            engagement=0.0, dominant_contact=0.0, displacement=0.0,
            destabilization=0.0, self_stability=1.0, composite=1.0,
            feedback_text="No data (0 steps).",
        )

    ring_diameter = record.ring_radius * 2.0

    # opponent displacement column + tipping arrays, chosen by which side we score
    if not flip:
        opp_disp = record.blue_displacements          # blue is the opponent
        opp_tipping = record.blue_tipping
        self_tipping = record.red_tipping
    else:
        opp_disp = _xy_step_disp(record.red_positions)  # red is the opponent
        opp_tipping = record.red_tipping
        self_tipping = record.blue_tipping

    contact_steps = sum(1 for c in record.contacts if c)
    engagement = contact_steps / total_steps

    contact_steps_blue_moved = sum(
        1 for c, d in zip(record.contacts, opp_disp)
        if c and d > 1e-4
    )
    dominant_contact = contact_steps_blue_moved / contact_steps if contact_steps > 0 else 0.0

    total_blue_disp_during_contact = sum(
        d for c, d in zip(record.contacts, opp_disp) if c
    )
    displacement = min(1.0, total_blue_disp_during_contact / ring_diameter) if ring_diameter > 0 else 0.0

    destabilization = max(opp_tipping) if opp_tipping else 0.0
    max_red_tipping = max(self_tipping) if self_tipping else 0.0
    self_stability = 1.0 - max_red_tipping

    composite = engagement + dominant_contact + displacement + destabilization + self_stability

    lines = [
        f"Combat score: {composite:.2f}/5.00",
        f"  engagement:      {engagement:.2f} ({contact_steps}/{total_steps} steps with contact)",
        f"  dominant_contact: {dominant_contact:.2f} ({contact_steps_blue_moved}/{contact_steps} contact steps with blue displacement)",
        f"  displacement:    {displacement:.2f} (blue displaced {total_blue_disp_during_contact:.2f}m during contact, ring_diam={ring_diameter:.1f}m)",
        f"  destabilization: {destabilization:.2f} (max blue tipping)",
        f"  self_stability:  {self_stability:.2f} (1 - max red tipping {max_red_tipping:.2f})",
    ]

    return CombatScore(
        engagement=engagement,
        dominant_contact=dominant_contact,
        displacement=displacement,
        destabilization=destabilization,
        self_stability=self_stability,
        composite=composite,
        feedback_text="\n".join(lines),
    )


def compute_seed_score(winner: str, combat_score: CombatScore) -> float:
    """Compute per-seed score from winner and combat metrics.

    Args:
        winner: "red" (our bot won), "blue" (opponent won), or "tie".
        combat_score: CombatScore from compute_combat_score.

    Returns:
        Float score: 1.0 + composite if red won, else 0.0 + composite.
        Range: 0.0–6.0 (win) or 0.0–5.0 (loss/draw).
    """
    if winner == "red":
        return 1.0 + combat_score.composite
    return combat_score.composite


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class RolloutResult:
    """Result from running match rollouts.

    Attributes:
        winners: List of winners per seed ("red", "blue", "tie").
        progress_values: List of progress values per seed (from episode.py).
        step_counts: List of step counts per seed.
        action_summaries: List of action summaries per seed.
        match_results: Raw result dicts from each seed (includes obs_log,
            action_log, combat_metrics).
    """
    winners: List[str]
    progress_values: List[float]
    step_counts: List[int]
    action_summaries: List[Dict[str, Any]]
    match_results: List[Dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.match_results is None:
            self.match_results = []


# =============================================================================
# Rollout Runner (returns raw results, no scoring)
# =============================================================================


def _run_single_seed(
    run_match_fn: Callable,
    red_policy_callable: Callable,
    i: int,
    seed: int,
) -> Dict[str, Any]:
    """Run a single seed rollout. Returns a dict with parsed fields or error marker."""
    from mjarena.design_shop.types import GameRecord
    try:
        result = run_match_fn(red_policy_callable, seed=seed)
        # Support both GameRecord (new) and dict (legacy) return types
        if isinstance(result, GameRecord):
            return {
                "idx": i,
                "winner": result.winner,
                "progress": result.progress,
                "steps": result.winner_step or result.num_steps,
                "action_summary": {"red": {}, "blue": {}},
                "raw": result,
            }
        return {
            "idx": i,
            "winner": result.get("winner", "tie"),
            "progress": result.get("progress", 0.0),
            "steps": result.get("winner_step") or result.get("num_steps", 0),
            "action_summary": {
                "red": result.get("red_action_summary", {}),
                "blue": result.get("blue_action_summary", {}),
            },
            "raw": result,
        }
    except Exception as exc:
        logger.warning("Rollout %d (seed=%d) failed: %s", i, seed, exc)
        return {
            "idx": i,
            "winner": "error",
            "progress": 0.0,
            "steps": 0,
            "action_summary": {"red": {}, "blue": {}},
            "raw": {},
        }


def run_rollouts(
    red_policy_callable: Callable,
    run_match_fn: Callable,
    n_seeds: int = 1,
    seeds: Optional[List[int]] = None,
    n_parallel_seeds: int = 1,
) -> RolloutResult:
    """
    Run match rollouts and return raw results.

    This is a simple rollout runner - no scoring logic here.
    Scoring is done by validate_controller or make_reward_fn.

    Args:
        red_policy_callable: Policy function for red robot.
        run_match_fn: Function that runs a single match, returns dict with
            'winner', 'progress', 'winner_step'/'num_steps'.
        n_seeds: Number of rollouts to run.
        seeds: Optional list of specific seeds to use.
        n_parallel_seeds: Number of seeds to run in parallel (1 = sequential).

    Returns:
        RolloutResult with raw results from each match.
    """
    if seeds is None:
        seeds = list(range(n_seeds))

    use_parallel = n_parallel_seeds > 1 and len(seeds) > 1

    if use_parallel:
        workers = min(n_parallel_seeds, len(seeds))
        collected: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _run_single_seed, run_match_fn, red_policy_callable, i, seed
                ): i
                for i, seed in enumerate(seeds)
            }
            for fut in as_completed(futures):
                res = fut.result()
                collected[res["idx"]] = res

        # Assemble in original seed order
        ordered = [collected[i] for i in range(len(seeds))]
    else:
        ordered = [
            _run_single_seed(run_match_fn, red_policy_callable, i, seed)
            for i, seed in enumerate(seeds)
        ]

    return RolloutResult(
        winners=[r["winner"] for r in ordered],
        progress_values=[r["progress"] for r in ordered],
        step_counts=[r["steps"] for r in ordered],
        action_summaries=[r["action_summary"] for r in ordered],
        match_results=[r["raw"] for r in ordered],
    )


# =============================================================================
# Arena Metric (for DSPy Evaluator)
# =============================================================================


def arena_metric(
    example,
    pred,
    trace=None,
    *,
    morphology_ref: str = "",
    actuator_names: Sequence[str] = (),
    obs_schema: Mapping[str, object] = None,
    run_match_fn: Callable = None,
    n_rollouts: int = 1,
    invalid_score: float = -1.0,
) -> Dict[str, Any]:
    """
    DSPy metric for arena evaluation.

    This metric:
    1. Extracts red_controller_code from prediction
    2. Verifies the controller
    3. If valid, runs rollouts and computes score
    4. Returns {"score": float, "feedback": str}

    For GEPA compatibility, always returns both score and feedback.

    Args:
        example: DSPy Example with scenario specification.
        pred: DSPy Prediction with red_controller_code.
        trace: Optional trace for debugging.
        morphology_ref: Path to robot morphology XML.
        actuator_names: List of actuator names.
        obs_schema: Observation schema.
        run_match_fn: Function to run matches (takes policy callable, returns result).
        n_rollouts: Number of rollouts to run.
        invalid_score: Score for invalid controllers.

    Returns:
        Dict with "score" (float) and "feedback" (str).
    """
    # Extract controller code from prediction
    code = None
    for key in ("red_controller_code", "policy_code", "code"):
        if hasattr(pred, key):
            code = getattr(pred, key)
            if isinstance(code, str):
                break

    if not code:
        return {
            "score": invalid_score,
            "feedback": "No controller code found in prediction.",
        }

    # Get actuator names and obs_schema from example if not provided
    if not actuator_names and hasattr(example, "actuator_names"):
        actuator_names = example.actuator_names
    if obs_schema is None and hasattr(example, "obs_schema"):
        obs_schema = example.obs_schema
    if not morphology_ref and hasattr(example, "morphology_ref"):
        morphology_ref = example.morphology_ref

    # Verify the controller (static checks only for arena_metric)
    verification = _get_validate_controller()(
        policy_code=code,
        actuator_names=list(actuator_names),
        obs_schema=obs_schema or {},
        processed_robot_xml=Path(morphology_ref).read_text() if morphology_ref else None,
    )

    if not verification.passed:
        return {
            "score": invalid_score,
            "feedback": f"Verification failed: {verification.feedback}",
        }

    # If no match function provided, return verification-only score
    if run_match_fn is None:
        return {
            "score": 0.0,
            "feedback": "Controller verified. No rollouts performed.",
        }

    # Build policy callable and run matches
    try:
        from mjarena.eval.match_runner import run_seeds

        policy_callable = build_policy_callable(
            controller_code=code,
            actuator_names=list(actuator_names),
        )

        matchup = run_seeds(
            run_match_fn=run_match_fn,
            policy_callable=policy_callable,
            n_seeds=n_rollouts,
        )

        avg_score = matchup.avg_score
        red_wins = matchup.n_wins()
        feedback = f"Ran {matchup.n_seeds} matches. Red won {red_wins}. Score: {avg_score:.2f}"

        return {
            "score": avg_score,
            "feedback": feedback,
        }

    except Exception as exc:
        return {
            "score": invalid_score,
            "feedback": f"Rollout execution failed: {exc}",
        }


# =============================================================================
# Reward Function for BestOfN/Refine
# =============================================================================


def make_reward_fn(
    *,
    actuator_names: Sequence[str],
    obs_schema: Mapping[str, object] = None,
    run_match_fn: Callable = None,
    n_rollouts: int = 1,
    evaluation_log: Optional[List[Dict]] = None,
) -> Callable[[Dict, Any], float]:
    """
    Create a reward function for dspy.BestOfN or dspy.Refine.

    Thin wrapper around validate_controller — the verifier is the single
    source of truth for both feedback AND score.

    Args:
        actuator_names: List of actuator names.
        obs_schema: Observation schema.
        run_match_fn: Function to run matches (for threshold tests).
        n_rollouts: Number of rollouts to run and average (default 1).
        evaluation_log: Optional list to track evaluations for verbose output.

    Returns:
        Callable reward function for DSPy optimization.
    """
    def reward_fn(args_dict: Dict, prediction) -> float:
        # Check if ctrl_verifier_score exists (from VerifiedActuatorControllerWriter)
        if hasattr(prediction, "ctrl_verifier_score"):
            score = prediction.ctrl_verifier_score
            if evaluation_log is not None:
                evaluation_log.append({
                    "score": score,
                    "passed": prediction.ctrl_verifier_passed,
                    "feedback": getattr(prediction, "ctrl_verifier_feedback", ""),
                })
            return score

        code = getattr(prediction, "controller_code", None)
        if not code:
            if evaluation_log is not None:
                evaluation_log.append({"score": -1.0, "error": "no_code"})
            return -1.0

        result = _get_validate_controller()(
            policy_code=code,
            actuator_names=list(actuator_names),
            obs_schema=obs_schema or {},
            qualification_match_fn=run_match_fn,
            n_rollouts=n_rollouts,
        )

        if evaluation_log is not None:
            evaluation_log.append({
                "score": result.score,
                "passed": result.passed,
                "feedback": result.feedback,
            })

        return result.score

    return reward_fn
