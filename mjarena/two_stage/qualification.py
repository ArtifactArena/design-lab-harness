"""Stage 1 — qualification ranking.

Walks the LOGS tree, reads each refinement commit's `qualification/match_result.json`,
computes the mean composite combat metric across qualification seeds, and writes a
per-model ranked list to `qualification_ranking.json`.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TOURNAMENT_DIR_RE = re.compile(r"^tournament_(\d+)$")
COMMIT_DIR_RE = re.compile(r"^commit_(\d+)$")


@dataclass
class CommitRecord:
    model: str
    tournament_idx: int
    commit_idx: int
    commit_dir: Path  # absolute path
    commit_dir_rel: str  # repo-relative
    qualification_score: float  # mean composite over qual seeds
    n_qual_seeds: int


def _read_qualification_score(commit_dir: Path) -> Optional[CommitRecord]:
    """Return (score, n_seeds) for a commit, or None if missing/invalid."""
    qual_json = commit_dir / "qualification" / "match_result.json"
    if not qual_json.exists():
        return None
    try:
        data = json.loads(qual_json.read_text())
    except json.JSONDecodeError:
        return None
    matches = data.get("matches", [])
    composites: List[float] = []
    for m in matches:
        cm = (m.get("combat_metrics") or {}).get("composite")
        if cm is None:
            continue
        try:
            composites.append(float(cm))
        except (TypeError, ValueError):
            continue
    if not composites:
        return None
    return CommitRecord(
        model="",  # filled by caller
        tournament_idx=-1,
        commit_idx=-1,
        commit_dir=commit_dir,
        commit_dir_rel="",
        qualification_score=sum(composites) / len(composites),
        n_qual_seeds=len(composites),
    )


def collect_qualification_records(
    logs_root: Path,
    repo_root: Path,
) -> Dict[str, List[CommitRecord]]:
    """Walk LOGS, return ranked records per model.

    Skips:
      - directories starting with "baseline-" (baselines have no refinement chain)
      - commits without a usable qualification/match_result.json
    """
    by_model: Dict[str, List[CommitRecord]] = {}
    skip_count = 0

    for tournament_dir in sorted(logs_root.iterdir()):
        if not tournament_dir.is_dir():
            continue
        m = TOURNAMENT_DIR_RE.match(tournament_dir.name)
        if not m:
            continue
        tournament_idx = int(m.group(1))

        bots_dir = tournament_dir / "round_robin_match" / "bots"
        if not bots_dir.is_dir():
            continue

        for model_dir in sorted(bots_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_dir.name.startswith("baseline-"):
                continue
            model_name = model_dir.name

            refinement_dir = model_dir / "refinement"
            if not refinement_dir.is_dir():
                continue

            for commit_dir in sorted(refinement_dir.iterdir()):
                if not commit_dir.is_dir():
                    continue
                cm = COMMIT_DIR_RE.match(commit_dir.name)
                if not cm:
                    continue
                commit_idx = int(cm.group(1))

                rec = _read_qualification_score(commit_dir)
                if rec is None:
                    skip_count += 1
                    continue
                rec.model = model_name
                rec.tournament_idx = tournament_idx
                rec.commit_idx = commit_idx
                try:
                    rec.commit_dir_rel = str(commit_dir.relative_to(repo_root))
                except ValueError:
                    rec.commit_dir_rel = str(commit_dir)

                by_model.setdefault(model_name, []).append(rec)

    for model_name, recs in by_model.items():
        recs.sort(key=lambda r: r.qualification_score, reverse=True)

    if skip_count:
        logger.info("Skipped %d commits without qualification scores", skip_count)
    return by_model


def write_qualification_ranking(
    by_model: Dict[str, List[CommitRecord]],
    output_path: Path,
    logs_root: Path,
    repo_root: Path,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "logs_root": str(logs_root.relative_to(repo_root))
        if logs_root.is_absolute() and _is_subpath(logs_root, repo_root)
        else str(logs_root),
        "models": {
            model: [
                {
                    "rank": i + 1,
                    "tournament_idx": r.tournament_idx,
                    "commit_idx": r.commit_idx,
                    "qualification_score": r.qualification_score,
                    "n_qual_seeds": r.n_qual_seeds,
                    "commit_dir": r.commit_dir_rel,
                }
                for i, r in enumerate(recs)
            ]
            for model, recs in sorted(by_model.items())
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def run_stage1(
    logs_root: Path,
    output_path: Path,
    repo_root: Path,
) -> Dict[str, List[CommitRecord]]:
    """Stage 1 entry point: collect + write."""
    logger.info("Stage 1: scanning %s", logs_root)
    by_model = collect_qualification_records(logs_root, repo_root)
    for model, recs in sorted(by_model.items()):
        logger.info("  %s: %d commits with qualification scores", model, len(recs))
    write_qualification_ranking(by_model, output_path, logs_root, repo_root)
    logger.info("Stage 1: wrote %s", output_path)
    return by_model
