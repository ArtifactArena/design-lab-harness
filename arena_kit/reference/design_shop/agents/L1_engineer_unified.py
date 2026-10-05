"""L1 Engineer — unified morphology + controller generation with single LLM.

Replaces the Creator→Critic two-call pattern. The Engineer sees verifier
feedback + match replay + design ledger directly, and self-corrects.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import dspy

from mjarena.core.build_progress import BuildProgressReporter
from mjarena.design_shop.refinement_state import RefinementState
from mjarena.dspy_core import _strip_markdown_code_block, _strip_mjcf_tags, USAGE_CTX
from mjarena.utils.llm_errors import classify_llm_error

logger = logging.getLogger(__name__)


def run_baseline_engineer_unified(
    *,
    lm: dspy.LM,
    consolidated_prompt: str,
    run_env_fn: Callable,  # (robot_xml, controller_code) -> UnifiedEnvResult
    refine_n: int = 10,
    output_dir: Optional[Path] = None,
    tournament_data: str = "",
    verbose: bool = True,
    dump_prompt: bool = False,
    generator: str = "",
    progress_reporter: Optional[BuildProgressReporter] = None,
    zero_shot_mode: bool = False,
    seed_robot_xml: str = "",
    seed_controller_code: str = "",
    seed_qualification_passed: bool = False,
    seed_qualification_score: float = float("-inf"),
    seed_bot_name: str = "",
) -> RefinementState:
    """Run the unified Engineer loop for refine_n commits.

    Args:
        lm: DSPy language model to use.
        consolidated_prompt: The run's one design document — the signature's whole
            instruction. Required: there is no modular prompt path any more.
        run_env_fn: Callable that validates XML + code, runs matches, returns feedback.
            Signature: (robot_xml: str, controller_code: str) -> UnifiedEnvResult
                       (feedback, score, qualification with the downsampled replay)
        refine_n: Number of refinement commits.
        output_dir: Directory for checkpoints and ledger.
        tournament_data: Cross-iteration tournament history (empty on first iteration).
        verbose: Log progress.
        dump_prompt: Save prompt/response text to each commit directory.
        generator: Model/generator name for progress display.

    Returns:
        RefinementState with best design tracked.
    """
    if not consolidated_prompt.strip():
        raise ValueError("consolidated_prompt is required: one big prompt per harness")
    if zero_shot_mode:
        from mjarena.design_shop.agents.signatures.sampling import make_sampling_signature
        engineer_signature = make_sampling_signature(consolidated_prompt)
    else:
        from mjarena.design_shop.agents.signatures.autoresearch import make_autoresearch_signature
        engineer_signature = make_autoresearch_signature(consolidated_prompt)
    engineer = dspy.ChainOfThought(engineer_signature)
    state = RefinementState(mode="unified", output_dir=output_dir)
    label = generator or "bot"
    if seed_robot_xml:
        state.latest_content = seed_robot_xml
    if seed_controller_code:
        state.latest_controller_code = seed_controller_code
    if seed_robot_xml and seed_controller_code and seed_qualification_passed:
        state.best_content = seed_robot_xml
        state.best_controller_code = seed_controller_code
        state.best_qualification_passed = True
        state.best_score = seed_qualification_score
        state.best_commit = -1
        state.best_scores = {
            "qualification_score": seed_qualification_score,
            "score": seed_qualification_score,
        }
        state.best_creator_outputs = {"name": seed_bot_name}
        n_qualified = 1
    else:
        n_qualified = 0

    prev_feedback = ""
    prev_replay = ""

    _CYAN = "\033[36m"
    _GREEN = "\033[32m"
    _RED = "\033[91m"
    _YELLOW = "\033[33m"
    _RESET = "\033[0m"

    for commit_num in range(refine_n):
        best_marker = _format_best_marker(state)
        progress_key = f"unified:{commit_num}"
        progress_message = f"commit {commit_num + 1}/{refine_n}"
        if progress_reporter:
            progress_reporter.update(
                phase="unified",
                stage="engineer_loop",
                wait="",
                message=progress_message,
            )
            progress_reporter.update_subtask(
                progress_key,
                label=f"commit {commit_num + 1}",
                phase="unified",
                stage="create",
                wait="api",
                message=progress_message,
            )
        if verbose:
            print(f"  {_CYAN}[{label}] commit {commit_num + 1}/{refine_n}{_RESET} | "
                  f"qualified: {n_qualified} | best: {best_marker}",
                  flush=True)

        # --- Call the Engineer ---
        history_before = len(lm.history)
        _usage_token = USAGE_CTX.set({
            "bot": output_dir.name if output_dir else "unknown",
            "commit": commit_num,
            "role": "engineer",
            "output_dir": str(output_dir) if output_dir else "",
        })
        try:
            try:
                with dspy.context(lm=lm):
                    if zero_shot_mode:
                        iteration_inputs = {}
                    else:
                        remaining = refine_n - commit_num - 1
                        iteration_inputs = {
                            "commit_status": (
                                f"Commit {commit_num + 1} of {refine_n}; {remaining} remaining after this one."
                            ),
                            "design_ledger": state.format_journal_for_creator(),
                            "best_commit_robot_xml": state.current_best_for_creator,
                            "best_commit_controller_code": state.current_best_controller_for_creator,
                            "last_commit_robot_xml": state.last_commit_xml_for_creator,
                            "last_commit_controller_code": state.last_commit_controller_for_creator,
                            "last_commit_verifier_feedback": prev_feedback,
                            "last_commit_match_replay": prev_replay,
                        }
                    # The document is the whole instruction: send only the fields it names.
                    iteration_inputs = {k: v for k, v in iteration_inputs.items()
                                        if k in engineer_signature.input_fields}
                    result = engineer(**iteration_inputs)
            finally:
                USAGE_CTX.reset(_usage_token)
        except Exception as exc:
            error_category, error_detail = classify_llm_error(exc)

            # Auth errors and internal bugs must propagate
            if error_category in ("llm_auth_error", "internal_error"):
                logger.error(f"[Engineer] {error_category} — aborting: {error_detail}")
                raise

            # Transient LLM error — record failed commit and continue to next
            logger.warning(f"[Engineer] Commit {commit_num} failed ({error_category}): {error_detail}")
            if progress_reporter:
                progress_reporter.fail_subtask(progress_key, message=f"{progress_message}: {error_category}")
            state.update(
                round_num=commit_num,
                score=-1.0,
                content="",
                verifier_feedback=f"{error_category.upper()} — {error_detail}",
                creator_summary=f"LLM call failed ({error_category}).",
                validation_passed=False,
            )
            if verbose:
                print(f"  {_RED}[{label}]{_RESET} commit {commit_num + 1}/{refine_n}: "
                      f"{error_category.upper()} — skipping to next commit", flush=True)
            continue

        # Capture LM history for this commit (prompt + response)
        new_entries = lm.history[history_before:]
        state.lm_history.extend(new_entries)

        # Dump prompt/response to commit directory
        if dump_prompt and output_dir and new_entries:
            _dump_prompt_to_commit(output_dir, commit_num, new_entries)

        robot_xml = _strip_markdown_code_block(getattr(result, "robot_xml", ""))
        robot_xml = _strip_mjcf_tags(robot_xml)
        controller_code = _strip_markdown_code_block(getattr(result, "controller_code", ""))
        change_summary = getattr(result, "change_summary", "")
        bot_name = getattr(result, "name", "")

        # --- Validate + run matches ---
        try:
            if progress_reporter:
                progress_reporter.update_subtask(
                    progress_key,
                    label=f"commit {commit_num + 1}",
                    phase="unified",
                    stage="validate",
                    wait="",
                    message=progress_message,
                )
            env_result = run_env_fn(
                robot_xml,
                controller_code,
                progress_reporter=progress_reporter,
                progress_key=progress_key,
                progress_message=progress_message,
            )
            feedback = env_result.feedback
            score = env_result.score
            qualification = env_result.qualification
        except Exception as exc:
            logger.warning(f"[Engineer] Env failed commit {commit_num}: {exc}")
            # Invalid designs return normal validation results. An uncaught
            # evaluator exception is an infrastructure failure, not feedback
            # to score against the robot or spend another commit fixing.
            raise RuntimeError(
                f"Consolidated-prompt evaluation failed at commit {commit_num + 1}") from exc

        # --- Update ledger ---
        validation_passed = score != -1.0
        qual_passed = qualification.passed if qualification else False
        qual_scores: Dict[str, Any] = {}
        if qualification and qualification.data:
            qual_scores.update(qualification.data)
        qual_scores["qualification_score"] = qualification.score if qualification else score
        qual_scores["score"] = qualification.score if qualification else score

        if qual_passed:
            n_qualified += 1

        is_best = state.update(
            round_num=commit_num,
            score=score,
            content=robot_xml,
            verifier_feedback=feedback,
            creator_summary=change_summary,
            controller_code=controller_code,
            creator_outputs={
                "name": bot_name,
                "reasoning": getattr(result, "reasoning", ""),
                "improvement_plan": getattr(result, "improvement_plan", ""),
                "design_strategy": getattr(result, "design_strategy", ""),
                "hardware_plan": getattr(result, "hardware_plan", ""),
                "combat_plan": getattr(result, "combat_plan", ""),
            },
            validation_passed=validation_passed,
            qualification_passed=qual_passed,
            scores=qual_scores,
        )

        if output_dir:
            state.save_ledger_text()

        # Per-commit result line
        if not validation_passed:
            # Extract first FAILED reason: the line after "── ... ── FAILED"
            fail_reason = ""
            lines = feedback.splitlines()
            for i, line in enumerate(lines):
                if "FAILED" in line and line.strip().startswith("──"):
                    # Reason is on the next non-empty line
                    for next_line in lines[i + 1:]:
                        stripped = next_line.strip()
                        if stripped:
                            fail_reason = stripped[:80]
                            break
                    break
            reason_suffix = f": {fail_reason}" if fail_reason else ""
            status = f"{_RED}FAILED{reason_suffix}{_RESET}"
        elif qual_passed:
            status = f"{_GREEN}QUALIFIED { _format_commit_status(score, qual_scores) }{_RESET}"
        else:
            status = f"{_YELLOW}score={score:.2f}{_RESET}"
        marker = f" {_GREEN}[NEW BEST]{_RESET}" if is_best else ""
        if progress_reporter:
            progress_reporter.complete_subtask(
                progress_key,
                message=f"{progress_message}: score={score:.2f}",
            )
        if verbose:
            print(f"  {_CYAN}[{label}]{_RESET} commit {commit_num + 1}/{refine_n} done: "
                  f"{bot_name} | {status}{marker}",
                  flush=True)

        # --- Set up feedback for next commit ---
        prev_feedback = feedback
        # qualification.matchup is already downsampled
        if qualification and qualification.matchup:
            prev_replay = json.dumps(qualification.matchup.to_prompt_dict(), default=str)
        else:
            prev_replay = ""

    q_tag = " [Q]" if state.best_qualification_passed else ""
    if progress_reporter:
        progress_reporter.update(
            phase="unified",
            stage="select_best",
            wait="",
            message=f"best commit {state.best_round}",
            clear_subtasks=True,
        )
    if verbose:
        print(f"  {_CYAN}[{label}]{_RESET} done: best=commit {state.best_round} "
              f"score={state.best_score:.2f}{q_tag} qualified={n_qualified}/{refine_n}",
              flush=True)

    return state


def _format_best_marker(state: RefinementState) -> str:
    """Compact best-so-far marker for progress output."""

    best_marker = f"{state.best_score:.2f}"
    if state.best_qualification_passed:
        best_marker += " [Q]"
    return best_marker


def _format_commit_status(score: float, scores: Dict[str, Any]) -> str:
    """Compact per-commit status for progress output."""
    return f"qual={score:.2f}"

def _dump_prompt_to_commit(output_dir: Path, commit_num: int, lm_entries: list) -> None:
    """Write prompt + response as readable text into the commit directory."""
    commit_dir = output_dir / "refinement" / f"commit_{commit_num}"
    commit_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for entry in lm_entries:
        if not isinstance(entry, dict):
            continue
        for msg in entry.get("messages", []):
            if isinstance(msg, dict):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                # Skip non-text content
                if isinstance(content, list):
                    text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                    content = "\n".join(text_parts) if text_parts else "(multimodal content)"
                lines.append(f"=== {role} ===")
                lines.append(content)
                lines.append("")

        # Response
        response = entry.get("outputs", entry.get("response", []))
        if isinstance(response, list):
            for item in response:
                if isinstance(item, dict):
                    # model_type: responses — DSPy hands back {"text": ..., "reasoning_content": ...}
                    item = item.get("text", "")
                if isinstance(item, str):
                    lines.append("=== RESPONSE ===")
                    lines.append(item)
                    lines.append("")
        elif isinstance(response, str):
            lines.append("=== RESPONSE ===")
            lines.append(response)
            lines.append("")

    (commit_dir / "prompt.txt").write_text("\n".join(lines), encoding="utf-8")
