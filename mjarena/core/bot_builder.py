"""End-to-end bot builder for LLM-generated robots.

This module provides the complete pipeline for generating a bot (morphology + controller)
using a specified LLM. It delegates to unified_builder for DSPy generation, then runs
shared post-processing (debug JSON, prompt dumps, video concat, artifact assembly).
"""
from __future__ import annotations

import datetime
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from mjarena.design_shop.utils import _thread_log
from mjarena.agents.types import BotArtifact
from mjarena.core.build_config import BuildConfig
from mjarena.core.build_progress import BuildProgressReporter
from mjarena.core.generation_result import GenerationResult
from mjarena.dspy_core import configure_lm
from mjarena.elo.display import _get_display_name
from mjarena.utils.file import ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers (used by debug/prompt/candidate saving)
# ---------------------------------------------------------------------------


def _extract_response_text(response: Any) -> str:
    """Extract readable text from a DSPy/litellm response object."""
    if response is None:
        return ""
    if hasattr(response, "choices"):
        texts = []
        for choice in response.choices:
            msg = getattr(choice, "message", None)
            if msg:
                content = getattr(msg, "content", None)
                if content:
                    texts.append(content)
        if texts:
            return "\n".join(texts)
    if isinstance(response, list):
        return "\n---\n".join(str(item) for item in response)
    return str(response)


def _serialize_usage(usage: Any) -> Optional[Dict]:
    """Extract token usage dict from a litellm Usage object or dict."""
    if usage is None:
        return None

    def _to_serializable(obj: Any) -> Any:
        if obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items() if v is not None}
        if hasattr(obj, "__dict__"):
            return {k: _to_serializable(v) for k, v in obj.__dict__.items()
                    if not k.startswith("_") and v is not None}
        return str(obj)

    return _to_serializable(usage)


def _parse_lm_call(entry: dict) -> Dict[str, Any]:
    """Parse a single DSPy LM history entry into a clean dict for JSON."""
    call: Dict[str, Any] = {}

    messages = entry.get("messages", [])
    if isinstance(messages, list) and messages:
        call["messages"] = []
        for msg in messages:
            if isinstance(msg, dict):
                call["messages"].append({
                    "role": msg.get("role", "unknown"),
                    "content": msg.get("content", str(msg)),
                })
            else:
                call["messages"].append({"role": "unknown", "content": str(msg)})

    response = entry.get("response")
    response_text = _extract_response_text(response)
    if not response_text:
        raw_outputs = entry.get("outputs")
        if raw_outputs:
            response_text = _extract_response_text(raw_outputs)
    call["response_text"] = response_text

    usage = _serialize_usage(entry.get("usage"))
    if usage:
        call["usage"] = usage
    cost = entry.get("cost")
    if cost is not None:
        call["cost"] = round(cost, 6)
    timestamp = entry.get("timestamp")
    if timestamp:
        call["timestamp"] = str(timestamp)
    uuid_val = entry.get("uuid")
    if uuid_val:
        call["uuid"] = str(uuid_val)

    return call


def load_lm_config(config_path: Path) -> Dict[str, Any]:
    """Load LLM configuration from YAML file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Prompt text saving
# ---------------------------------------------------------------------------


def _save_prompt_text(debug_dir: Path, phase: str, lm_history: list) -> None:
    """Save LLM prompt and response as human-readable text files."""
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not lm_history:
        return

    use_suffix = len(lm_history) > 1

    for idx, raw_entry in enumerate(lm_history):
        entry = raw_entry if isinstance(raw_entry, dict) else {}
        suffix = f"_{idx}" if use_suffix else ""

        messages = entry.get("messages", [])
        prompt_lines = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                prompt_lines.append(f"=== {role} ===")
                prompt_lines.append(content)
                prompt_lines.append("")
            else:
                prompt_lines.append(str(msg))
                prompt_lines.append("")

        (debug_dir / f"{phase}_prompt{suffix}.txt").write_text(
            "\n".join(prompt_lines), encoding="utf-8"
        )

        response_text = _extract_response_text(entry.get("response"))
        if not response_text:
            response_text = _extract_response_text(entry.get("outputs"))
        (debug_dir / f"{phase}_response{suffix}.txt").write_text(
            response_text or "(no response)", encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Candidate saving
# ---------------------------------------------------------------------------


def _save_morphology_candidates(candidates_dir: Path, lm_history: list) -> None:
    """Save all morphology candidate XMLs from LM history."""
    import re
    for i, entry in enumerate(lm_history):
        if not isinstance(entry, dict):
            continue
        response_text = _extract_response_text(entry.get("response"))
        if not response_text:
            response_text = _extract_response_text(entry.get("outputs"))
        if not response_text:
            continue
        xml_match = re.search(r'(<mujoco[\s\S]*?</mujoco>)', response_text)
        if xml_match:
            xml_content = xml_match.group(1)
            filename = f"candidate_{i}.xml"
            (candidates_dir / filename).write_text(xml_content, encoding="utf-8")


def _save_controller_candidates(candidates_dir: Path, lm_history: list) -> None:
    """Save all controller candidate code from LM history."""
    import re
    for i, entry in enumerate(lm_history):
        if not isinstance(entry, dict):
            continue
        response_text = _extract_response_text(entry.get("response"))
        if not response_text:
            response_text = _extract_response_text(entry.get("outputs"))
        if not response_text:
            continue
        code_match = re.search(r'(def policy_step[\s\S]*?)(?:\n```|\Z)', response_text)
        if code_match:
            code_content = code_match.group(1).rstrip()
            filename = f"candidate_{i}.py"
            (candidates_dir / filename).write_text(code_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Shared post-processing helpers
# ---------------------------------------------------------------------------



def _save_all_prompts(
    output_dir: Path,
    gen_result: GenerationResult,
    verbose: bool,
) -> None:
    """Save prompt/response text files for all generation phases."""
    debug_dir = output_dir / "debug"
    ensure_dir(debug_dir)

    if gen_result.generation_mode == "unified":
        _save_prompt_text(debug_dir, "unified", gen_result.morph_lm_history)
    else:
        _save_prompt_text(debug_dir, "morphology", gen_result.morph_lm_history)
        _save_prompt_text(debug_dir, "controller", gen_result.ctrl_lm_history)
        # Save controller verifier feedback if available
        feedback = gen_result.controller_feedback
        if feedback and feedback != "Controller verified.":
            (debug_dir / "controller_verifier_details.txt").write_text(
                feedback, encoding="utf-8"
            )

    if verbose:
        print(f"  [dump-prompt] Saved prompt/response to {debug_dir}")


def _save_all_candidates(
    output_dir: Path,
    gen_result: GenerationResult,
    verbose: bool,
) -> None:
    """Save intermediate BestOfN candidates."""
    if gen_result.generation_mode == "unified":
        # Unified candidates contain both morphology + controller
        if gen_result.morph_lm_history:
            candidates_dir = output_dir / "candidates" / "unified"
            ensure_dir(candidates_dir)
            _save_morphology_candidates(candidates_dir, gen_result.morph_lm_history)
            _save_controller_candidates(candidates_dir, gen_result.morph_lm_history)
            if verbose:
                print(f"  [candidates] Saved unified candidates to {candidates_dir}")
    else:
        if gen_result.morph_lm_history:
            candidates_dir = output_dir / "candidates" / "morphology"
            ensure_dir(candidates_dir)
            _save_morphology_candidates(candidates_dir, gen_result.morph_lm_history)
            if verbose:
                print(f"  [candidates] Saved morphology candidates to {candidates_dir}")
        if gen_result.ctrl_lm_history:
            candidates_dir = output_dir / "candidates" / "controller"
            ensure_dir(candidates_dir)
            _save_controller_candidates(candidates_dir, gen_result.ctrl_lm_history)
            if verbose:
                print(f"  [candidates] Saved controller candidates to {candidates_dir}")



def _assemble_artifact(
    gen_result: GenerationResult,
    generator: str,
    output_dir: Path,
    lm_config_path: Path,
    lm_config: Dict[str, Any],
    constraints_path: Path,
    cfg: BuildConfig,
    match_feedback: str,
    season_id: str,
    tournament_id: str,
) -> BotArtifact:
    """Assemble BotArtifact with COMPLETE metadata for both split and unified modes."""
    bot_name = generator

    morphology_path = output_dir / "robot.xml"
    controller_path = output_dir / "controller.py"

    morphology_verification = gen_result.morphology_verification
    morphology_verified = morphology_verification.passed if morphology_verification else False
    morphology_errors = morphology_verification.errors if morphology_verification else []

    controller_verification = gen_result.controller_verification
    controller_verified = controller_verification.passed if controller_verification else False
    controller_errors = controller_verification.errors if controller_verification else []

    # Resolve paths to refinement history

    # Find journal path — split mode saves per-phase, unified saves in candidate dirs
    journal_path = None
    for candidate in ("refinement/journal.json",):
        jp = output_dir / candidate
        if jp.exists():
            journal_path = jp
            break
    # Also check candidate subdirs (unified parallel candidates)
    if journal_path is None:
        for cand_dir in sorted(output_dir.glob("candidate_*/refinement/journal.json")):
            journal_path = cand_dir
            break

    return BotArtifact(
        name=bot_name,
        generator=generator,
        morphology_xml=morphology_path,
        controller_code=controller_path,
        actuator_names=gen_result.actuator_names,
        morphology_score=gen_result.morphology_score,
        morphology_verified=morphology_verified,
        morphology_errors=morphology_errors,
        controller_score=gen_result.controller_score,
        controller_verified=controller_verified,
        controller_errors=controller_errors,
        journal_path=journal_path,
        metadata={
            "generated_at": datetime.datetime.now().isoformat(),
            "lm_config_path": str(lm_config_path),
            "constraints_path": str(constraints_path),
            "generation_mode": gen_result.generation_mode,
            # Generation timing
            "morphology_generation_time_sec": round(gen_result.morphology_generation_time_sec, 2),
            "controller_generation_time_sec": round(gen_result.controller_generation_time_sec, 2),
            "total_generation_time_sec": round(gen_result.total_generation_time_sec, 2),
            # BestOfN / Refine params
            "morphology_k": cfg.morphology_k,
            "controller_k": cfg.controller_k,
            "n_rollouts": cfg.controller_validation_params.n_rollouts,
            "commit_budget": cfg.commit_budget,
            # LLM params from config (full YAML for post-hoc analysis)
            "lm_params": {
                k: v for k, v in lm_config.items()
                if k not in ("api_key", "api_base")
            },
            # Other
            "match_feedback_provided": bool(match_feedback),
            "use_generator_name_in_prompt": cfg.use_generator_name_in_prompt,
            "use_dspy_cache": cfg.use_dspy_cache,
            # Best commit's engineer outputs
            "reasoning": gen_result.morph_reasoning,
            "design_strategy": gen_result.overview,
            "hardware_plan": gen_result.design_calculations,
            "combat_plan": gen_result.control_calculations,
        }
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_bot(
    generator: str,
    lm_config_path: Path,
    constraints_path: Path,
    output_dir: Path,
    cfg: BuildConfig,
    *,
    arena_xml: Optional[Path] = None,
    opponent_xml: Optional[Path] = None,
    eval_dir: Optional[Path] = None,
    match_feedback: str = "",
    env: str = "DEBUG",
    season_id: str = "season_00",
    tournament_id: str = "tournament_00",
    iteration: int = 0,
    verbose: bool = True,
    progress_reporter: Optional[BuildProgressReporter] = None,
) -> BotArtifact:
    """Generate complete bot (morphology + controller) using specified LLM.

    This is the main entry point for bot generation. It:
    1. Configures the LLM
    2. Dispatches to split or unified builder
    3. Runs shared post-processing (save files, debug, video, artifact)

    Args:
        generator: Name of the generator LLM
        lm_config_path: Path to LLM configuration YAML
        constraints_path: Path to BOT_CONSTRAINTS.yaml
        output_dir: Directory to save bot artifacts
        cfg: BuildConfig with all build-phase tuneable parameters.
        arena_xml: Path to arena XML (for controller evaluation)
        opponent_xml: Path to opponent robot XML (defaults to self)
        eval_dir: Directory for evaluation rollouts
        match_feedback: Optional feedback from previous matches
        env: Environment identifier
        season_id: Season identifier
        tournament_id: Tournament identifier
        iteration: Iteration number
        verbose: If True, print progress information

    Returns:
        BotArtifact with paths to generated files and metadata
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Building bot with generator: {generator}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")

    # Step 1: Configure LLM
    if progress_reporter:
        progress_reporter.update(
            phase="setup",
            stage="configure_lm",
            wait="",
            message="configuring LM",
        )
    if verbose:
        print(f"[1/4] Configuring LLM from {lm_config_path}...")
    lm = configure_lm(str(lm_config_path), use_cache=cfg.use_dspy_cache)
    lm_config = load_lm_config(lm_config_path)

    # Step 2-3: Generate bot (unified pipeline)
    from mjarena.core.unified_builder import generate_unified
    gen_result = generate_unified(
        generator=generator,
        lm=lm,
        constraints_path=constraints_path,
        output_dir=output_dir,
        cfg=cfg,
        arena_xml=arena_xml,
        opponent_xml=opponent_xml,
        eval_dir=eval_dir,
        match_feedback=match_feedback,
        season_id=season_id,
        tournament_id=tournament_id,
        verbose=verbose,
        progress_reporter=progress_reporter,
    )

    # ── Shared post-processing ───────────────────────────────────────────

    # Save robot.xml + controller.py (write both unconditionally so the
    # for arena composition; unified may not have saved yet — write both
    # unconditionally so the final output_dir is always complete)
    morphology_path = output_dir / "robot.xml"
    morphology_path.write_text(gen_result.processed_xml, encoding="utf-8")

    controller_path = output_dir / "controller.py"
    controller_path.write_text(gen_result.controller_code, encoding="utf-8")

    # Prompt dumps
    if cfg.dump_prompt:
        _save_all_prompts(output_dir, gen_result, verbose)

    # Intermediate candidates
    if cfg.save_intermediate_candidates:
        _save_all_candidates(output_dir, gen_result, verbose)

    # Step 4: Create and save artifact
    if verbose:
        print(f"\n[4/4] Saving bot artifact...")
    if progress_reporter:
        progress_reporter.update(
            phase="artifact",
            stage="save_artifact",
            wait="",
            message="writing robot/controller/artifact files",
            clear_subtasks=True,
        )

    artifact = _assemble_artifact(
        gen_result=gen_result,
        generator=generator,
        output_dir=output_dir,
        lm_config_path=lm_config_path,
        lm_config=lm_config,
        constraints_path=constraints_path,
        cfg=cfg,
        match_feedback=match_feedback,
        season_id=season_id,
        tournament_id=tournament_id,
    )
    artifact.save(output_dir)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Bot generation complete!")
        print(f"  Morphology score: {gen_result.morphology_score:.2f}")
        print(f"  Controller score: {gen_result.controller_score:.2f}")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}\n")

    return artifact


def main():
    """CLI entry point for bot_builder."""
    import argparse

    parser = argparse.ArgumentParser(description="Build a bot using an LLM")
    parser.add_argument("--generator", required=True, help="Name of the generator (e.g., gpt4o, claude)")
    parser.add_argument("--lm-config", required=True, help="Path to LLM config YAML")
    parser.add_argument("--constraints", default="configs/rules/rules.yaml",
                       help="Path to constraints YAML")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--arena", help="Path to arena XML (optional)")
    parser.add_argument("--opponent", help="Path to opponent XML (optional)")
    parser.add_argument("--morphology-k", type=int, default=3,
                       help="Number of morphology candidates")
    parser.add_argument("--controller-k", type=int, default=5,
                       help="Number of controller candidates")
    parser.add_argument("--n-rollouts", type=int, default=3,
                       help="Number of rollouts for evaluation")
    parser.add_argument("--match-feedback", default="", help="Feedback from previous matches")
    parser.add_argument("--refine-morphology", action="store_true",
                       help="Use dspy.Refine for morphology improvement")
    parser.add_argument("--refine-controller", action="store_true",
                       help="Use dspy.Refine for controller improvement")
    parser.add_argument("--commit-budget", type=int, default=50,
                        help="Commits per build (engineer loop bound; also what the prompt states)")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    cfg = BuildConfig(
        morphology_k=args.morphology_k,
        controller_k=args.controller_k,
        commit_budget=args.commit_budget,
        unified_generation=True,
        autoresearch_prompt_path="configs/rules/autoresearch_prompt.md",
    )

    artifact = build_bot(
        generator=args.generator,
        lm_config_path=Path(args.lm_config),
        constraints_path=Path(args.constraints),
        output_dir=Path(args.output),
        cfg=cfg,
        arena_xml=Path(args.arena) if args.arena else None,
        opponent_xml=Path(args.opponent) if args.opponent else None,
        match_feedback=args.match_feedback,
        verbose=not args.quiet,
    )

    print(f"Bot artifact saved to: {args.output}")
    print(f"  Morphology: {artifact.morphology_xml}")
    print(f"  Controller: {artifact.controller_code}")


if __name__ == "__main__":
    main()
