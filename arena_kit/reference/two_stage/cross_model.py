"""Stage 3 — cross-model RR.

Pool every model's top-K + baselines, run a full pairwise round-robin, compute
artifact-level Elo, then back-fill `cross_model_elo.{k1,k3,k5}` and
`cross_model_wld.{k1,k3,k5}` into the pool's `top_{K_max}_bots.json`.
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
    load_baseline_artifact,
    load_commit_artifact,
    make_baseline_artifact_id,
    read_top_k_bots,
    write_top_k_bots,
)
from mjarena.two_stage.intra_model import _read_match_outcomes
from mjarena.two_stage.match_config import TwoStageMatchConfig

logger = logging.getLogger(__name__)

DEFAULT_BASELINES = [
    "baseline-pusher",  # human-coded
    "baseline-static",  # Box
    "baseline-tetrahedron",
    "baseline-hexahedron",
    "baseline-octahedron",
    "baseline-dodecahedron",
    "baseline-icosahedron",
]


def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _build_pool_rows(
    inter_model_dir: Path,
    baselines_dir: Path,
    repo_root: Path,
    *,
    k_max: int,
    baselines: List[str],
    k_values: List[int],
) -> List[Dict[str, Any]]:
    """Concatenate all models' top-K rows + baseline rows. Empty cross_model_*."""
    rows: List[Dict[str, Any]] = []

    for model_dir in sorted(inter_model_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        top_k_path = model_dir / f"top_{k_max}_bots.json"
        if not top_k_path.exists():
            logger.warning("missing %s; skipping model %s", top_k_path, model_dir.name)
            continue
        payload = read_top_k_bots(top_k_path)
        for row in payload["bots"]:
            row = dict(row)
            row["cross_model_elo"] = {f"k{k}": None for k in k_values}
            row["cross_model_wld"] = {f"k{k}": None for k in k_values}
            rows.append(row)

    null_cross = {f"k{k}": None for k in k_values}
    for baseline_name in baselines:
        baseline_dir = baselines_dir / baseline_name
        if not baseline_dir.is_dir():
            logger.warning("baseline dir missing: %s", baseline_dir)
            continue
        artifact = load_baseline_artifact(baseline_dir)
        artifact_id = make_baseline_artifact_id(artifact.generator)
        robot_rel = (
            str(artifact.morphology_xml.relative_to(repo_root))
            if _is_subpath(artifact.morphology_xml, repo_root)
            else str(artifact.morphology_xml)
        )
        ctrl_rel = (
            str(artifact.controller_code.relative_to(repo_root))
            if _is_subpath(artifact.controller_code, repo_root)
            else str(artifact.controller_code)
        )
        rows.append(
            TopKBotRecord(
                rank=1,
                artifact_id=artifact_id,
                model=artifact.generator,
                kind="baseline",
                tournament_idx=None,
                commit_idx=None,
                robot_xml=robot_rel,
                controller_py=ctrl_rel,
                qualification_score=None,
                intra_model_elo=None,
                intra_model_wld=None,
                cross_model_elo=dict(null_cross),
                cross_model_wld=dict(null_cross),
            ).to_dict()
        )

    return rows


def _build_artifacts_from_pool(
    rows: List[Dict[str, Any]],
    repo_root: Path,
    baselines_dir: Path,
    *,
    clean_cache_root: Optional[Path] = None,
    constraints_path: Optional[Path] = None,
    physics_mode: str = "3d",
) -> Dict[str, BotArtifact]:
    """Construct {bot_name: BotArtifact} keyed by artifact_id (baselines re-keyed)."""
    bots: Dict[str, BotArtifact] = {}
    for row in rows:
        kind = row["kind"]
        if kind == "refinement_commit":
            commit_dir = (repo_root / row["robot_xml"]).parent
            artifact = load_commit_artifact(
                commit_dir=commit_dir,
                model_name=row["model"],
                commit_idx=int(row["commit_idx"]),
                tournament_idx=int(row["tournament_idx"]),
                clean_cache_root=clean_cache_root,
                constraints_path=constraints_path,
                physics_mode=physics_mode,
            )
            bots[artifact.name] = artifact
        elif kind == "baseline":
            artifact = load_baseline_artifact((repo_root / row["robot_xml"]).parent)
            artifact_id = make_baseline_artifact_id(artifact.generator)
            artifact.name = artifact_id
            artifact.generator = artifact_id
            bots[artifact_id] = artifact
        else:
            raise ValueError(f"unknown kind in pool: {kind!r}")
    return bots


def _filter_outcomes_for_subset(
    outcomes: List[MatchOutcome],
    allowed_ids: set[str],
) -> List[MatchOutcome]:
    return [o for o in outcomes if o.player_a in allowed_ids and o.player_b in allowed_ids]


def _allowed_artifact_ids(
    rows: List[Dict[str, Any]],
    k_prime: int,
) -> set[str]:
    """All baseline artifact_ids + non-baseline rows with rank <= k_prime."""
    allowed: set[str] = set()
    for row in rows:
        if row["kind"] == "baseline":
            allowed.add(row["artifact_id"])
        elif int(row["rank"]) <= k_prime:
            allowed.add(row["artifact_id"])
    return allowed


def _backfill_cross_model_fields(
    rows: List[Dict[str, Any]],
    outcomes: List[MatchOutcome],
    *,
    k_values: List[int],
    elo_k_factor: float,
    elo_initial_rating: float,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Populate cross_model_elo/wld[k1/k3/...] on each row in place.

    Returns the per-K' standings dicts so callers can persist them too:
      {"k1": {artifact_id: {"elo": ..., "wins": ...}}, "k3": {...}, ...}
    """
    per_k_standings: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for k_prime in k_values:
        key = f"k{k_prime}"
        allowed = _allowed_artifact_ids(rows, k_prime)
        filtered = _filter_outcomes_for_subset(outcomes, allowed)
        standings = compute_standings(filtered)
        per_k_standings[key] = standings

        for row in rows:
            aid = row["artifact_id"]
            if aid in standings:
                stat = standings[aid]
                row["cross_model_elo"][key] = float(stat["elo"])
                row["cross_model_wld"][key] = {
                    "wins": int(stat["wins"]),
                    "losses": int(stat["losses"]),
                    "draws": int(stat["draws"]),
                }
            else:
                row["cross_model_elo"][key] = None
                row["cross_model_wld"][key] = None

    return per_k_standings


def run_stage3(
    inter_model_dir: Path,
    output_root: Path,
    baselines_dir: Path,
    repo_root: Path,
    match_cfg: TwoStageMatchConfig,
    *,
    k_max: int = 5,
    n_rollouts: Optional[int] = None,
    n_parallel_matches: int = 8,
    n_parallel_seeds: int = 1,
    baselines: Optional[List[str]] = None,
    k_values: Optional[List[int]] = None,
    clean_cache_root: Optional[Path] = None,
    elo_k_factor: float = 32.0,
    elo_initial_rating: float = 1000.0,
) -> Path:
    """Stage 3 entry point.

    Writes:
      cross_model/top_{k_max}_matches/top_{k_max}_bots.json
      cross_model/top_{k_max}_matches/elo_artifact.json
      cross_model/top_{k_max}_matches/matches/<a>_vs_<b>/match_result.json
    Returns path to the pool file.
    """
    if baselines is None:
        baselines = list(DEFAULT_BASELINES)
    if k_values is None:
        k_values = sorted({k for k in (1, 3, k_max) if k <= k_max})
    if n_rollouts is None:
        n_rollouts = match_cfg.n_rollouts

    cross_dir = output_root / f"top_{k_max}_matches"
    cross_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_pool_rows(
        inter_model_dir=inter_model_dir,
        baselines_dir=baselines_dir,
        repo_root=repo_root,
        k_max=k_max,
        baselines=baselines,
        k_values=k_values,
    )
    if len(rows) < 2:
        raise RuntimeError(f"Stage 3: pool has {len(rows)} rows; need >=2")

    pool_path = cross_dir / f"top_{k_max}_bots.json"
    write_top_k_bots(
        pool_path,
        k=k_max,
        scope="cross_model_pool",
        model=None,
        source={
            "logs_root": None,
            "inter_model_dir": str(inter_model_dir.relative_to(repo_root))
            if _is_subpath(inter_model_dir, repo_root)
            else str(inter_model_dir),
            "baselines_dir": str(baselines_dir.relative_to(repo_root))
            if _is_subpath(baselines_dir, repo_root)
            else str(baselines_dir),
            "match_config": str(match_cfg.source_path) if match_cfg.source_path else None,
            "n_rollouts": n_rollouts,
            "match_time": match_cfg.match_time,
            "save_video": match_cfg.save_video,
            "highres": match_cfg.highres,
            "elo_k_factor": elo_k_factor,
            "elo_initial_rating": elo_initial_rating,
            "k_values": k_values,
        },
        bots=rows,
    )

    bots = _build_artifacts_from_pool(
        rows,
        repo_root,
        baselines_dir,
        clean_cache_root=clean_cache_root,
        constraints_path=match_cfg.constraints_path,
        physics_mode=match_cfg.physics_mode,
    )
    arena_path = (
        match_cfg.arena_xml
        if match_cfg.arena_xml.is_absolute()
        else repo_root / match_cfg.arena_xml
    )

    logger.info(
        "Stage 3: cross-model RR over %d artifacts, %d rollouts, save_video=%s, highres=%s",
        len(bots),
        n_rollouts,
        match_cfg.save_video,
        match_cfg.highres,
    )

    run_round_robin(
        bots=bots,
        arena_xml=arena_path,
        out_dir=cross_dir,
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
        season_id="two_stage_cross",
        tournament_id="cross_model",
    )

    matches_dir = cross_dir / "matches"
    outcomes = _read_match_outcomes(matches_dir)

    standings_full = compute_standings(outcomes)
    (cross_dir / "elo_artifact.json").write_text(
        json.dumps(
            {
                "method": "bt",
                "n_rollouts": n_rollouts,
                "standings": standings_full,
            },
            indent=2,
        )
    )

    per_k_standings = _backfill_cross_model_fields(
        rows,
        outcomes,
        k_values=k_values,
        elo_k_factor=elo_k_factor,
        elo_initial_rating=elo_initial_rating,
    )

    pool_payload = read_top_k_bots(pool_path)
    pool_payload["bots"] = rows
    pool_payload["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pool_path.write_text(json.dumps(pool_payload, indent=2))

    for k_key, standings in per_k_standings.items():
        (cross_dir / f"elo_artifact_{k_key}.json").write_text(
            json.dumps(
                {
                    "k_filter": k_key,
                    "k_factor": elo_k_factor,
                    "initial_rating": elo_initial_rating,
                    "standings": standings,
                },
                indent=2,
            )
        )

    logger.info("Stage 3: wrote %s", pool_path)
    return pool_path
