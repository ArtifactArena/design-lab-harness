"""Match tools — run matches via run_seeds(), return VerifierResult with typed matchup.

Core functions:
- qualify_round(): Runs the seeds against the stationary block via run_seeds() and
  passes the bot when it loses none of them, per the prompt. Returns VerifierResult.
- evaluate_vs_stationary/pusher/self(): LLM-facing ReAct tools. n_rollouts=1 default. Return VerifierResult.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional, Sequence

from mjarena.core.build_progress import BuildProgressReporter, make_seed_progress_listener
from mjarena.design_shop.types import MatchupResult, VerifierResult
from mjarena.eval.match_runner import run_seeds, save_matchup_to_disk
from mjarena.policy_spec import PolicySpec

logger = logging.getLogger(__name__)


_QUALIFICATION_SCORE_MODE_ALIASES = {
    "combat": "combat",
    "default": "combat",
    "original": "combat",
    "original_combat": "combat",
    "combatv2_score": "combatv2_score",
    "combat_v2_score": "combatv2_score",
    "combatv2": "combatv2_score",
    "qualification_mechanics_score": "combatv2_score",
    "cube_replay_metric": "combatv2_score",
    "cube_replay_mechanics": "combatv2_score",
    "qualification_cube_replay_metric": "combatv2_score",
}


def _score_qualification_matchup(matchup) -> tuple[float, dict]:
    """Qualification score = mean over seeds of (combat composite + 1.0 win bonus)."""
    combat_avg_score = matchup.avg_score
    data = {
        "qualification_score": combat_avg_score,
        "combat_avg_score": combat_avg_score,
        "wins": matchup.n_wins(),
        "n_seeds": len(matchup.game_records),
    }
    parts = ("engagement", "dominant_contact", "displacement", "destabilization", "self_stability")
    scores = [cs for cs in matchup.combat_scores if hasattr(cs, "composite")]
    if scores:
        for name in parts:
            data[f"mean_{name}"] = sum(float(getattr(cs, name, 0.0)) for cs in scores) / len(scores)
    return combat_avg_score, data


def _format_qualification_score_feedback(qualification_score: float) -> str:
    return (
        f"Qualification score: {qualification_score:.2f}/6.0 "
        "(mean over seeds of combat composite 0-5, plus 1.0 for each seed won)"
    )


def _format_qacc_feedback(qacc_failures) -> str:
    """Physics-instability feedback: the diagnosis once, then one line per failed seed."""
    parts = ["Physics instability detected — match terminated as loss."]
    for record in qacc_failures:
        first_step = record.qacc_warning_steps[0] if record.qacc_warning_steps else 0
        dt = float(getattr(record, "control_dt", 0.01) or 0.01)
        parts.append(
            f"  Seed {record.seed}: MuJoCo BADQACC (NaN/inf joint acceleration) at control step "
            f"{first_step} (t={first_step * dt:.2f}s)."
        )
    diagnostics = next((r.qacc_gear_diagnostics for r in qacc_failures if r.qacc_gear_diagnostics), None)
    parts.append(
        "\nCore equation: angular_acceleration = (gear x ctrl) / body_inertia. "
        "If gear/inertia is too large, even small ctrl values cause explosive acceleration."
    )
    if diagnostics:
        parts.append("Actuator gear-to-inertia ratios:")
        for d in diagnostics:
            safe_gear = 5000.0 * float(d["body_inertia_min"])
            parts.append(
                f"  {d['actuator']}: gear={d['gear']}, "
                f"body_inertia_min={d['body_inertia_min']}, "
                f"ratio={d['ratio']:.0f} -> with this body, gear <= {safe_gear:.0f} keeps the ratio under 5000"
            )
        parts.append(
            "Ratios above ~5000 are dangerous, and light wheels under a heavy chassis blow up even below "
            "that. Fixes, most reliable first:\n"
            "- bigger / heavier wheels (rubber, radius >= 0.12 m, thickness >= 0.06 m) so each wheel "
            "carries real inertia\n"
            "- armature=\"0.02\" on each wheel <joint> (allowed; adds rotor inertia)\n"
            "- lower gear to the value above, or increase the wheel body's mass\n"
            "- controller slew-rate limiting (+/-0.1 per step), no sudden full-torque reversals"
        )
    return "\n".join(parts)


# =============================================================================
# Internal helpers (play_against_*)
# =============================================================================


def qualify_round(
    policy_callable: Callable,
    run_match_fn: Callable,
    policy_spec: Optional[PolicySpec] = None,
    n_rollouts: int = 3,
    n_parallel: int = 1,
    parallel_backend: str = "thread",
    output_dir: Optional[Path] = None,
    commit_num: int = 0,
    progress_reporter: Optional[BuildProgressReporter] = None,
    progress_key: Optional[str] = None,
    progress_message: str = "",
) -> VerifierResult:
    """Qualification round: run against stationary, check bot doesn't lose any seed.

    Per the prompt: the bot passes if it loses none of the seeds (no ring-out, no
    instability, no inactivity). A draw passes; winning every round is the goal,
    not the bar. An empty run (n_rollouts == 0) never qualifies.

    Returns:
        VerifierResult with matchup= field containing downsampled match data.
    """
    qual_dir = None
    if output_dir is not None:
        qual_dir = Path(output_dir) / "refinement" / f"commit_{commit_num}" / "qualification"
        qual_dir.mkdir(parents=True, exist_ok=True)

    if progress_reporter and progress_key:
        progress_reporter.update_subtask(
            progress_key,
            phase="controller",
            stage="qualification",
            wait="simulation",
            message=progress_message or f"commit {commit_num + 1}: qualification",
        )
    simulation_progress_listener = None
    if progress_reporter:
        simulation_progress_listener = make_seed_progress_listener(
            progress_reporter,
            phase="controller",
            stage="qualification",
            total_seeds=n_rollouts,
            progress_key=progress_key,
        )

    matchup = run_seeds(
        run_match_fn=run_match_fn,
        policy_callable=policy_callable,
        policy_spec=policy_spec,
        n_seeds=n_rollouts,
        n_parallel=n_parallel,
        parallel_backend=parallel_backend,
        simulation_progress_listener=simulation_progress_listener,
    )

    # QACC hard failure
    if matchup.qacc_failures:
        qacc_data = {
            "qualification_score": -1.0,
            "combat_avg_score": matchup.avg_score,
        }
        if qual_dir:
            save_matchup_to_disk(
                matchup,
                qual_dir,
                tool_name="stationary",
                extra_match_result_fields=qacc_data,
            )
        message = _format_qacc_feedback(matchup.qacc_failures)
        return VerifierResult(
            passed=False,
            label="Qualification",
            message=message,
            score=-1.0,
            data=qacc_data,
            matchup=matchup.downsample(),
            name="qualification",
            elapsed_sec=matchup.elapsed_sec,
        )

    qualification_score, score_data = _score_qualification_matchup(matchup)
    if qual_dir:
        save_matchup_to_disk(
            matchup,
            qual_dir,
            tool_name="stationary",
            extra_match_result_fields=score_data,
        )

    qualified = n_rollouts > 0 and matchup.n_losses() == 0
    seed_lines = matchup.seed_summary_lines()
    combat_lines = matchup.combat_feedback_lines()
    downsampled = matchup.downsample()

    if not qualified:
        message = (
            f"Lost {matchup.n_losses()}/{n_rollouts} matches; qualification requires not losing any round.\n"
            f"Won {matchup.n_wins()}/{n_rollouts}; drew {matchup.n_ties}. Winning every round is the goal.\n"
            + "\n".join(seed_lines) + "\n\n"
            + "\n".join(combat_lines) + "\n\n"
            + _format_qualification_score_feedback(qualification_score) + "\n\n"
            + f"Detailed replay attached for {downsampled.n_seeds} seed(s) "
            f"(losses only). Examine red_positions, red_edge_distances, "
            f"contacts, and red_actions step by step."
        )
    else:
        message = (
            f"Won {matchup.n_wins()}/{n_rollouts}, drew {matchup.n_ties}/{n_rollouts}; "
            f"qualified (no losses). Winning every round is the goal.\n"
            + "\n".join(seed_lines) + "\n\n" + "\n".join(combat_lines)
            + "\n\n" + _format_qualification_score_feedback(qualification_score)
        )

    return VerifierResult(
        passed=qualified,
        label="Qualification",
        message=message,
        score=qualification_score,
        data=score_data,
        matchup=downsampled,
        name="qualification",
        elapsed_sec=matchup.elapsed_sec,
    )


