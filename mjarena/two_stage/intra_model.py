"""Stage 2 — intra-model RR.

For each model, take top-N qualified commits, run a full pairwise round-robin,
compute artifact-level Elo, pick top K_max, and write `top_{K_max}_bots.json`.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mjarena.agents.types import BotArtifact
from mjarena.elo.core import MatchOutcome, compute_standings
from mjarena.tournament.tournament import run_round_robin
from mjarena.two_stage.artifact_loader import (
    TopKBotRecord,
    load_commit_artifact,
    make_artifact_id,
    write_top_k_bots,
)
from mjarena.two_stage.match_config import TwoStageMatchConfig

logger = logging.getLogger(__name__)


def _read_match_outcomes(matches_dir: Path) -> List[MatchOutcome]:
    """Read every `match_result.json` under `matches_dir` and convert to MatchOutcomes.

    Each rollout (seed) becomes one MatchOutcome; ties become draws.
    """
    outcomes: List[MatchOutcome] = []
    for match_json in sorted(matches_dir.rglob("match_result.json")):
        try:
            data = json.loads(match_json.read_text())
        except json.JSONDecodeError:
            logger.warning("skipping unreadable %s", match_json)
            continue
        red = data.get("red_bot")
        blue = data.get("blue_bot")
        if not red or not blue:
            continue
        for m in data.get("matches", []):
            winner = m.get("winner")
            if winner == "red":
                outcomes.append(MatchOutcome(player_a=red, player_b=blue, winner=red))
            elif winner == "blue":
                outcomes.append(MatchOutcome(player_a=red, player_b=blue, winner=blue))
            else:
                outcomes.append(
                    MatchOutcome(player_a=red, player_b=blue, winner="draw")
                )
    return outcomes


def _select_top_n_qualified(
    qualification_path: Path,
    model: str,
    top_n: int,
    repo_root: Path,
) -> List[Dict[str, Any]]:
    """Read qualification_ranking.json, return top-N records for a model."""
    payload = json.loads(qualification_path.read_text())
    model_records = payload.get("models", {}).get(model, [])
    return model_records[:top_n]


def _build_artifacts_for_model(
    top_n_records: List[Dict[str, Any]],
    model: str,
    repo_root: Path,
    *,
    clean_cache_root: Optional[Path] = None,
    constraints_path: Optional[Path] = None,
    physics_mode: str = "3d",
) -> Dict[str, BotArtifact]:
    """Build a {artifact_id: BotArtifact} map from top-N qualification records."""
    bots: Dict[str, BotArtifact] = {}
    for rec in top_n_records:
        commit_dir = repo_root / rec["commit_dir"]
        artifact = load_commit_artifact(
            commit_dir=commit_dir,
            model_name=model,
            commit_idx=int(rec["commit_idx"]),
            tournament_idx=int(rec["tournament_idx"]),
            clean_cache_root=clean_cache_root,
            constraints_path=constraints_path,
            physics_mode=physics_mode,
        )
        bots[artifact.name] = artifact
    return bots


def run_intra_rr_for_model(
    model: str,
    top_n_records: List[Dict[str, Any]],
    output_root: Path,
    repo_root: Path,
    match_cfg: TwoStageMatchConfig,
    *,
    k_max: int = 5,
    n_rollouts: Optional[int] = None,
    n_parallel_matches: int = 8,
    n_parallel_seeds: int = 1,
    clean_cache_root: Optional[Path] = None,
    elo_k_factor: float = 32.0,
    elo_initial_rating: float = 1000.0,
    qualification_ranking_path: Optional[Path] = None,
) -> Path:
    """Run intra-model RR for one model, write per-model `top_{k_max}_bots.json`.

    Returns the path to the written top-K file.
    """
    if k_max > len(top_n_records):
        logger.warning(
            "%s: only %d qualified records but k_max=%d; capping",
            model,
            len(top_n_records),
            k_max,
        )

    if n_rollouts is None:
        n_rollouts = match_cfg.n_rollouts

    model_dir = output_root / model
    model_dir.mkdir(parents=True, exist_ok=True)

    top_n_path = model_dir / "top_n_qualified.json"
    top_n_path.write_text(
        json.dumps(
            {
                "model": model,
                "n": len(top_n_records),
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "records": top_n_records,
            },
            indent=2,
        )
    )

    bots = _build_artifacts_for_model(
        top_n_records,
        model,
        repo_root,
        clean_cache_root=clean_cache_root,
        constraints_path=match_cfg.constraints_path,
        physics_mode=match_cfg.physics_mode,
    )
    if len(bots) < 2:
        raise RuntimeError(
            f"{model}: need at least 2 qualified bots for intra-RR, got {len(bots)}"
        )

    arena_path = (
        match_cfg.arena_xml
        if match_cfg.arena_xml.is_absolute()
        else repo_root / match_cfg.arena_xml
    )

    logger.info(
        "Stage 2 [%s]: intra-RR over %d artifacts, %d rollouts, save_video=%s",
        model,
        len(bots),
        n_rollouts,
        match_cfg.save_video,
    )

    run_round_robin(
        bots=bots,
        arena_xml=arena_path,
        out_dir=model_dir,
        n_rollouts=n_rollouts,
        record_video=match_cfg.save_video,
        save_all_videos=match_cfg.save_all_videos,
        highres=match_cfg.highres,
        verbose=True,
        trace_progress=False,
        physics_mode=match_cfg.physics_mode,
        score_function=match_cfg.score_function,
        match_time=match_cfg.match_time,
        contact_fidelity=match_cfg.contact_fidelity,
        inactivity_timeout_seconds=match_cfg.inactivity_timeout,
        inactivity_min_displacement=match_cfg.inactivity_min_displacement,
        size_limits=match_cfg.size_limits,
        n_parallel_matches=n_parallel_matches,
        n_parallel_seeds=n_parallel_seeds,
        camera_mode=match_cfg.camera,
        rendering_flags=match_cfg.rendering_flags or None,
        season_id=f"two_stage_intra/{model}",
        tournament_id="intra_model",
    )

    matches_dir = model_dir / "matches"
    outcomes = _read_match_outcomes(matches_dir)
    standings = compute_standings(outcomes)

    elo_payload = {
        "model": model,
        "method": "bt",
        "n_rollouts": n_rollouts,
        "standings": standings,
    }
    (model_dir / "elo.json").write_text(json.dumps(elo_payload, indent=2))

    qual_lookup = {
        make_artifact_id(model, int(r["tournament_idx"]), int(r["commit_idx"])): r
        for r in top_n_records
    }

    sorted_artifacts = sorted(
        standings.items(), key=lambda kv: kv[1]["elo"], reverse=True
    )
    top_k = sorted_artifacts[:k_max]

    rows: List[Dict[str, Any]] = []
    for rank, (artifact_id, stats) in enumerate(top_k, start=1):
        qual_rec = qual_lookup.get(artifact_id)
        if qual_rec is None:
            logger.warning("%s: no qualification record for %s", model, artifact_id)
            continue
        commit_dir_rel = qual_rec["commit_dir"]
        rows.append(
            TopKBotRecord(
                rank=rank,
                artifact_id=artifact_id,
                model=model,
                kind="refinement_commit",
                tournament_idx=int(qual_rec["tournament_idx"]),
                commit_idx=int(qual_rec["commit_idx"]),
                robot_xml=f"{commit_dir_rel}/robot.xml",
                controller_py=f"{commit_dir_rel}/controller.py",
                qualification_score=float(qual_rec["qualification_score"]),
                intra_model_elo=float(stats["elo"]),
                intra_model_wld={
                    "wins": int(stats["wins"]),
                    "losses": int(stats["losses"]),
                    "draws": int(stats["draws"]),
                },
                cross_model_elo={"k1": None, "k3": None, "k5": None},
                cross_model_wld={"k1": None, "k3": None, "k5": None},
            ).to_dict()
        )

    top_k_path = model_dir / f"top_{k_max}_bots.json"
    write_top_k_bots(
        top_k_path,
        k=k_max,
        scope="intra_model",
        model=model,
        source={
            "logs_root": None,
            "qualification_ranking": str(qualification_ranking_path)
            if qualification_ranking_path is not None
            else None,
            "intra_model_elo": str((model_dir / "elo.json").relative_to(repo_root))
            if _is_subpath(model_dir / "elo.json", repo_root)
            else str(model_dir / "elo.json"),
            "top_n_qualified": len(top_n_records),
            "n_rollouts": n_rollouts,
            "elo_method": "bt",
        },
        bots=rows,
    )
    logger.info("Stage 2 [%s]: wrote %s (top-%d)", model, top_k_path, len(rows))
    return top_k_path


def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def run_stage2(
    qualification_path: Path,
    output_root: Path,
    repo_root: Path,
    match_cfg: TwoStageMatchConfig,
    *,
    top_n_qualified: int = 15,
    k_max: int = 5,
    n_rollouts: Optional[int] = None,
    n_parallel_matches: int = 8,
    models: Optional[List[str]] = None,
    clean_cache_root: Optional[Path] = None,
) -> Dict[str, Path]:
    """Stage 2 entry point: run intra-RR for each model.

    Returns {model: path_to_top_k_file}.
    """
    payload = json.loads(qualification_path.read_text())
    all_models = list(payload.get("models", {}).keys())
    target_models = models if models else all_models

    output_root.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    for model in target_models:
        records = _select_top_n_qualified(
            qualification_path, model, top_n_qualified, repo_root
        )
        if len(records) < 2:
            logger.warning("Skipping %s (only %d qualified)", model, len(records))
            continue
        try:
            top_k_path = run_intra_rr_for_model(
                model=model,
                top_n_records=records,
                output_root=output_root,
                repo_root=repo_root,
                match_cfg=match_cfg,
                k_max=k_max,
                n_rollouts=n_rollouts,
                n_parallel_matches=n_parallel_matches,
                clean_cache_root=clean_cache_root,
                qualification_ranking_path=qualification_path.relative_to(repo_root)
                if _is_subpath(qualification_path, repo_root)
                else qualification_path,
            )
            results[model] = top_k_path
        except Exception:
            logger.exception("Stage 2 [%s]: failed", model)
            continue

    return results
