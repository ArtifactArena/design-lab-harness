"""Loaders + writers for the two-stage Elo pipeline.

The canonical data structure is `top_{k}_bots.json`, written per-model after
intra-model RR (Stage 2) and again as a pooled file for cross-model RR (Stage 3).
Schema details: see `docs/plans/2026-04-26-two-stage-elo.md`.

Important convention:
  Two-stage `BotArtifact`s use `name = generator = artifact_id` so that the
  match-folder layout in `tournament.run_round_robin` (which uses `generator`
  as the discriminator) yields unique directories within a single intra-model
  run. The conceptual model identity (e.g. "claude-opus-4-6-high") is recovered
  from the `top_{k}_bots.json` row's `model` field, not from the BotArtifact.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mjarena.agents.types import BotArtifact
from mjarena.design_shop.rules.hardware_rules import (
    apply_material_properties as _apply_material_properties,
    inject_motor_mass as _inject_motor_mass,
    sanitize_robot_xml as _sanitize_robot_xml,
    strip_wrappers as _strip_wrappers,
)
from mjarena.design_shop.rules.mj_validators import ModelValidationConfig

ARTIFACT_ID_PATTERN = re.compile(
    r"^(?P<model>.+)__t(?P<tournament>\d{2})_c(?P<commit>\d{3})$"
)

_DEFAULT_CONSTRAINTS_PATH = Path("configs/rules/rules.yaml")


def _resolve_constraints_path(constraints_path: Optional[Path]) -> Path:
    if constraints_path is not None:
        return constraints_path
    return _DEFAULT_CONSTRAINTS_PATH


# Cache the built ModelValidationConfig so we don't re-parse YAML per artifact.
_VALIDATION_CONFIG_CACHE: Dict[str, ModelValidationConfig] = {}


def _get_validation_config(
    constraints_path: Optional[Path], physics_mode: str = "3d"
) -> ModelValidationConfig:
    path = _resolve_constraints_path(constraints_path)
    key = f"{path}:{physics_mode}"
    cached = _VALIDATION_CONFIG_CACHE.get(key)
    if cached is None:
        cached = ModelValidationConfig(
            constraints_yaml_path=Path(path), physics_mode=physics_mode
        )
        _VALIDATION_CONFIG_CACHE[key] = cached
    return cached


def _sanitize_to_cache(
    raw_xml_path: Path,
    cache_path: Path,
    *,
    constraints_path: Optional[Path] = None,
    physics_mode: str = "3d",
) -> Path:
    """Run the build pipeline's pre-compile XML transforms on a raw refinement XML.

    Refinement commit robot.xmls were saved before the build pipeline's
    sanitize/material/motor steps, so MuJoCo rejects them on multiple grounds:
      * `description=` LLM attributes (sanitize_robot_xml strips these)
      * material= names without <material> defs (apply_material_properties
        injects defs, computes mass=vol*density, sets friction)
      * missing motor masses (inject_motor_mass adds gear-proportional mass to
        each actuated body's first geom)

    The build pipeline calls these in order before MuJoCo compile. We mirror
    that here and cache the result so subsequent runs skip the work.
    """
    if cache_path.exists():
        return cache_path

    config = _get_validation_config(constraints_path, physics_mode=physics_mode)
    raw = raw_xml_path.read_text()
    xml, _ = _strip_wrappers(raw)
    xml, _ = _sanitize_robot_xml(xml)
    xml, _ = _apply_material_properties(xml, config)
    xml, _ = _inject_motor_mass(xml, config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(xml)
    return cache_path


def make_artifact_id(model: str, tournament_idx: int, commit_idx: int) -> str:
    return f"{model}__t{tournament_idx:02d}_c{commit_idx:03d}"


def make_baseline_artifact_id(name: str) -> str:
    return f"baseline__{name}"


def parse_artifact_id(artifact_id: str) -> Optional[Dict[str, Any]]:
    """Parse a non-baseline artifact_id back into (model, tournament_idx, commit_idx)."""
    m = ARTIFACT_ID_PATTERN.match(artifact_id)
    if not m:
        return None
    return {
        "model": m.group("model"),
        "tournament_idx": int(m.group("tournament")),
        "commit_idx": int(m.group("commit")),
    }


def _extract_actuator_names(robot_xml: Path) -> List[str]:
    """Pull <motor name="..."> entries from a robot XML.

    Falls back to `<actuator>` children with name attributes if `<motor>` is
    not used.
    """
    text = robot_xml.read_text()
    names = re.findall(r'<motor[^>]*\bname\s*=\s*"([^"]+)"', text)
    if names:
        return names
    return re.findall(
        r'<actuator>.*?<\w+\s[^>]*\bname\s*=\s*"([^"]+)"',
        text,
        flags=re.DOTALL,
    )


def load_commit_artifact(
    commit_dir: Path,
    model_name: str,
    commit_idx: int,
    tournament_idx: int,
    *,
    clean_cache_root: Optional[Path] = None,
    constraints_path: Optional[Path] = None,
    physics_mode: str = "3d",
) -> BotArtifact:
    """Build a BotArtifact from a refinement/commit_N/ directory.

    The commit dir contains only `robot.xml` + `controller.py` (no per-commit
    bot_artifact.json). The robot.xml is sanitized (strip_wrappers +
    sanitize_robot_xml + apply_material_properties + inject_motor_mass) and
    cached, then `morphology_xml` points at the cleaned copy so MuJoCo can
    compile it. `actuator_names` is parsed from the cleaned XML. `name` and
    `generator` are both set to the artifact_id so downstream tournament code
    generates unique match-folder paths per artifact.
    """
    robot_xml = commit_dir / "robot.xml"
    controller_py = commit_dir / "controller.py"
    if not robot_xml.exists():
        raise FileNotFoundError(f"robot.xml not found in {commit_dir}")
    if not controller_py.exists():
        raise FileNotFoundError(f"controller.py not found in {commit_dir}")

    artifact_id = make_artifact_id(model_name, tournament_idx, commit_idx)

    if clean_cache_root is None:
        clean_cache_root = Path("two_stage_tournaments") / "_clean_artifacts"
    cache_path = clean_cache_root / artifact_id / "robot.xml"
    sanitized_xml = _sanitize_to_cache(
        robot_xml,
        cache_path,
        constraints_path=constraints_path,
        physics_mode=physics_mode,
    )

    return BotArtifact(
        name=artifact_id,
        generator=artifact_id,
        morphology_xml=sanitized_xml,
        controller_code=controller_py,
        actuator_names=_extract_actuator_names(sanitized_xml),
        morphology_score=1.0,
        morphology_verified=True,
        controller_score=0.0,
        controller_verified=True,
        metadata={
            "two_stage_kind": "refinement_commit",
            "model": model_name,
            "tournament_idx": tournament_idx,
            "commit_idx": commit_idx,
        },
    )


def load_baseline_artifact(baseline_dir: Path) -> BotArtifact:
    """Build a BotArtifact from a `baseline-<name>/` directory.

    Reads bot_artifact.json for the canonical name/generator/actuator_names.
    Resolves relative `morphology_xml` / `controller_code` paths against the
    baseline_dir.
    """
    artifact_json = baseline_dir / "bot_artifact.json"
    if not artifact_json.exists():
        raise FileNotFoundError(f"bot_artifact.json not found in {baseline_dir}")
    data = json.loads(artifact_json.read_text())

    name = data["name"]
    generator = data["generator"]

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else baseline_dir / path

    return BotArtifact(
        name=name,
        generator=generator,
        morphology_xml=_resolve(data["morphology_xml"]),
        controller_code=_resolve(data["controller_code"]),
        actuator_names=list(data.get("actuator_names", [])),
        morphology_score=float(data.get("morphology_score", 1.0)),
        morphology_verified=bool(data.get("morphology_verified", True)),
        controller_score=float(data.get("controller_score", 0.0)),
        controller_verified=bool(data.get("controller_verified", True)),
        metadata={
            "two_stage_kind": "baseline",
            "model": "baseline",
            "baseline_dir": str(baseline_dir),
        },
    )


@dataclass
class TopKBotRecord:
    """One row in `top_{k}_bots.json`."""

    rank: int
    artifact_id: str
    model: str
    kind: str  # "refinement_commit" | "baseline" | "final_artifact"
    tournament_idx: Optional[int]
    commit_idx: Optional[int]
    robot_xml: str  # repo-relative path string
    controller_py: str  # repo-relative path string
    qualification_score: Optional[float]
    intra_model_elo: Optional[float]
    intra_model_wld: Optional[Dict[str, int]]
    cross_model_elo: Dict[str, Optional[float]]  # {"k1": ..., "k3": ..., "k5": ...}
    cross_model_wld: Dict[str, Optional[Dict[str, int]]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "artifact_id": self.artifact_id,
            "model": self.model,
            "kind": self.kind,
            "tournament_idx": self.tournament_idx,
            "commit_idx": self.commit_idx,
            "robot_xml": self.robot_xml,
            "controller_py": self.controller_py,
            "qualification_score": self.qualification_score,
            "intra_model_elo": self.intra_model_elo,
            "intra_model_wld": self.intra_model_wld,
            "cross_model_elo": self.cross_model_elo,
            "cross_model_wld": self.cross_model_wld,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TopKBotRecord":
        return cls(
            rank=int(d["rank"]),
            artifact_id=str(d["artifact_id"]),
            model=str(d["model"]),
            kind=str(d["kind"]),
            tournament_idx=d.get("tournament_idx"),
            commit_idx=d.get("commit_idx"),
            robot_xml=str(d["robot_xml"]),
            controller_py=str(d["controller_py"]),
            qualification_score=d.get("qualification_score"),
            intra_model_elo=d.get("intra_model_elo"),
            intra_model_wld=d.get("intra_model_wld"),
            cross_model_elo=d.get(
                "cross_model_elo", {"k1": None, "k3": None, "k5": None}
            ),
            cross_model_wld=d.get(
                "cross_model_wld", {"k1": None, "k3": None, "k5": None}
            ),
        )


def empty_cross_model_fields(k_values: List[int]) -> Dict[str, Dict[str, None]]:
    """Initial null cross_model_elo/wld dicts keyed by 'k1', 'k3', ..."""
    keys = [f"k{k}" for k in k_values]
    return {
        "elo": {k: None for k in keys},
        "wld": {k: None for k in keys},
    }


def write_top_k_bots(
    path: Path,
    *,
    k: int,
    scope: str,  # "intra_model" | "cross_model_pool"
    model: Optional[str],
    source: Dict[str, Any],
    bots: List[Dict[str, Any]],
) -> None:
    """Write `top_{k}_bots.json` with consistent key ordering.

    `bots` should be a list of dicts in TopKBotRecord shape (use
    TopKBotRecord(...).to_dict()).
    """
    payload = {
        "k": k,
        "scope": scope,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "bots": bots,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def read_top_k_bots(path: Path) -> Dict[str, Any]:
    """Read raw `top_{k}_bots.json` payload."""
    return json.loads(path.read_text())


def load_bots_from_top_k(path: Path, repo_root: Path) -> List[BotArtifact]:
    """Read `top_{k}_bots.json` and construct BotArtifacts ready for run_round_robin.

    For refinement_commit rows, builds via load_commit_artifact() so the
    artifact_id wiring (name=generator=artifact_id) is consistent.
    For baseline rows, builds via load_baseline_artifact() (name/generator
    come from the baseline's own bot_artifact.json).
    """
    payload = read_top_k_bots(path)
    artifacts: List[BotArtifact] = []
    for row in payload["bots"]:
        kind = row["kind"]
        robot_path = repo_root / row["robot_xml"]
        if kind == "refinement_commit":
            commit_dir = robot_path.parent
            artifacts.append(
                load_commit_artifact(
                    commit_dir=commit_dir,
                    model_name=row["model"],
                    commit_idx=int(row["commit_idx"]),
                    tournament_idx=int(row["tournament_idx"]),
                )
            )
        elif kind == "baseline":
            baseline_dir = robot_path.parent
            artifacts.append(load_baseline_artifact(baseline_dir))
        else:
            raise ValueError(f"unknown bot kind: {kind!r}")
    return artifacts
