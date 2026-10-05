"""
Core type definitions for arena agents.

ACTUATOR format only - controllers output dict[str, float].
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import numpy as np


# ACTUATOR is the only supported format
ControllerFormat = Literal["ACTUATOR"]


@dataclass
class TaskSpec:
    """Stable task description consumed by DSPy programs and metrics."""

    task_id: str
    arena_id: str
    arena_params: Dict[str, Any] = field(default_factory=dict)
    base_seed: int = 0
    n_rollouts: int = 1
    time_limit: Optional[float] = None
    dt: Optional[float] = None
    max_steps: Optional[int] = None
    obs_schema: Dict[str, Any] = field(default_factory=dict)
    action_interface: ControllerFormat = "ACTUATOR"
    constraints: Dict[str, Any] = field(default_factory=dict)
    opponent_id: Optional[str] = None
    opponent_config: Optional[Dict[str, Any]] = None


@dataclass
class BotObservation:
    """Standardized observation for bots in 3D sumo environments.

    Full 3D observation schema with position, orientation, velocity, distances,
    contact, stability, and time fields.
    """

    # Position
    my_pos: np.ndarray  # shape (3,), [x, y, z] COM. x/y = ground plane, z = height up
    opponent_pos: np.ndarray  # shape (3,), [x, y, z] opponent COM

    # Orientation (scalars, radians)
    my_yaw: float  # heading (0 = facing +X)
    my_pitch: float  # forward/back tilt (0 = level)
    my_roll: float  # left/right tilt (0 = level)
    opponent_yaw: float  # opponent heading
    opponent_pitch: float  # opponent forward/back tilt
    opponent_roll: float  # opponent left/right tilt

    # Velocity
    my_velocity: np.ndarray  # shape (3,), [vx, vy, vz] world-frame linear velocity
    my_angular_velocity: np.ndarray  # shape (3,), [wx, wy, wz] world-frame angular velocity
    opponent_velocity: np.ndarray  # shape (3,), [vx, vy, vz] opponent linear velocity
    opponent_angular_velocity: np.ndarray  # shape (3,), [wx, wy, wz] opponent angular velocity

    # Distances
    distance_to_opponent: float  # euclidean distance between COMs
    my_edge_distance: float  # signed distance to ring (positive=inside, negative=outside)
    opponent_edge_distance: float  # opponent's distance to ring edge

    # Contact
    opponent_contact: bool  # am I touching opponent?
    opponent_contact_force: float  # total contact force magnitude (N)
    ground_contact: bool  # any of my bodies touching arena ground?

    # Stability
    is_tipping: float  # 0.0=upright, 1.0=fallen

    # Time
    t: int  # current timestep
    max_t: int  # max timesteps in match (t/max_t → fraction of match elapsed)

    # Optional controller feedback and task context
    my_actuator_velocity: Dict[str, float] = field(default_factory=dict)
    opponent_actuator_velocity: Dict[str, float] = field(default_factory=dict)
    game: Dict[str, Any] = field(default_factory=dict)

    # Robot properties (static per match)
    my_bounding_radius: float = 0.5  # max XY extent from COM (meters)
    opponent_bounding_radius: float = 0.5  # opponent's bounding radius
    ring_radius: float = 7.5  # arena radius (meters), rules.yaml arena.radius_m

    # Spatial grids (optional, populated when ObservationConfig enables them)
    arena_grid: Optional[np.ndarray] = None           # (N,N) int8
    arena_mass_grid: Optional[np.ndarray] = None       # (N,N) float32
    edge_distance_grid: Optional[np.ndarray] = None    # (N,N) float32

    # Inactivity
    my_inactivity_timer: float = 0.0      # seconds since COM was last >= 0.5 m from its current spot (loss at 10)
    opponent_inactivity_timer: float = 0.0 # same for the opponent

    # Additional named physical state and interaction fields.
    details: Dict[str, Any] = field(default_factory=dict)

    # Legacy alias for backward compatibility
    @property
    def my_heading(self) -> float:
        return self.my_yaw

    @property
    def my_closest_distance_to_ring(self) -> float:
        return self.my_edge_distance

    @staticmethod
    def get_dummy_bot_obs() -> "BotObservation":
        """Create dummy observation for testing purposes."""
        # Dummy grids: 41x41 (cell_size=0.25)
        grid_size = 41
        dummy_int_grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        dummy_float_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        return BotObservation(
            my_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            opponent_pos=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            my_yaw=0.0,
            my_pitch=0.0,
            my_roll=0.0,
            opponent_yaw=0.0,
            opponent_pitch=0.0,
            opponent_roll=0.0,
            my_velocity=np.zeros(3, dtype=np.float32),
            my_angular_velocity=np.zeros(3, dtype=np.float32),
            opponent_velocity=np.zeros(3, dtype=np.float32),
            opponent_angular_velocity=np.zeros(3, dtype=np.float32),
            distance_to_opponent=1.0,
            my_edge_distance=5.0,
            opponent_edge_distance=5.0,
            opponent_contact=False,
            opponent_contact_force=0.0,
            ground_contact=True,
            is_tipping=0.0,
            t=0,
            max_t=1000,
            my_actuator_velocity={},
            opponent_actuator_velocity={},
            game={},
            my_bounding_radius=0.5,
            opponent_bounding_radius=0.5,
            ring_radius=7.5,
            arena_grid=dummy_int_grid.copy(),
            arena_mass_grid=dummy_float_grid.copy(),
            edge_distance_grid=dummy_float_grid.copy(),
            my_inactivity_timer=0.0,
            opponent_inactivity_timer=0.0,
        )


@dataclass
class VerificationResult:
    """Result from controller verification."""

    passed: bool
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    processed_xml: Optional[str] = None  # Post-pipeline XML (material assets injected, mass computed)
    warnings: List[str] = field(default_factory=list)  # Non-fatal info for refine (e.g., "geom X defaulted to foam")


@dataclass
class BotArtifact:
    """Complete bot artifact including morphology and controller.

    Attributes:
        name: Human-readable name for the bot
        generator: Who created this bot ("claude", "gemini", "gpt4o", "human", etc.)
        morphology_xml: Path to the robot XML file
        controller_code: Path to the controller Python file
        actuator_names: List of actuator names in the robot

        # Scores and verification status
        morphology_score: +1 if compiled, -1 if failed
        morphology_verified: True if morphology passes all checks
        morphology_errors: List of error messages if morphology verification failed
        controller_score: 1.0-2.0 if won rollouts, -1 if lost/timed out
        controller_verified: True if passes all checks, False if broken/threshold failed
        controller_errors: List of error messages if verification failed

        # Metadata
        metadata: Additional metadata (timestamps, config, etc.)
    """
    name: str
    generator: str
    morphology_xml: Path
    controller_code: Path
    actuator_names: List[str]

    # Scores and verification status
    morphology_score: float = 0.0
    morphology_verified: bool = False
    morphology_errors: List[str] = field(default_factory=list)
    controller_score: float = 0.0
    controller_verified: bool = False  # True = passes all checks, False = broken/threshold failed
    controller_errors: List[str] = field(default_factory=list)  # Error messages if verification failed

    # Forfeit status (when bot generation completely fails)
    forfeit: bool = False  # True if bot failed to generate entirely
    forfeit_stage: str = ""  # "morphology" or "controller" - where generation failed
    forfeit_error: str = ""  # The error message for debugging

    # Refinement history
    journal_path: Optional[Path] = None  # refinement/journal.json (score progression, best commit)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "generator": self.generator,
            "morphology_xml": str(self.morphology_xml),
            "controller_code": str(self.controller_code),
            "actuator_names": self.actuator_names,
            "morphology_score": self.morphology_score,
            "morphology_verified": self.morphology_verified,
            "morphology_errors": self.morphology_errors,
            "controller_score": self.controller_score,
            "controller_verified": self.controller_verified,
            "controller_errors": self.controller_errors,
            "forfeit": self.forfeit,
            "forfeit_stage": self.forfeit_stage,
            "forfeit_error": self.forfeit_error,
            "journal_path": str(self.journal_path) if self.journal_path else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotArtifact":
        """Create BotArtifact from dictionary.

        Backward compatible: ignores old morphology_outputs/actuator_outputs blocks.
        """
        return cls(
            name=data["name"],
            generator=data.get("generator", data.get("llm_name", "unknown")),
            morphology_xml=Path(data["morphology_xml"]),
            controller_code=Path(data["controller_code"]),
            actuator_names=data.get("actuator_names", []),
            morphology_score=data.get("morphology_score", 0.0),
            morphology_verified=data.get("morphology_verified", False),
            morphology_errors=data.get("morphology_errors", []),
            controller_score=data.get("controller_score", 0.0),
            controller_verified=data.get("controller_verified", False),
            controller_errors=data.get("controller_errors", []),
            forfeit=data.get("forfeit", False),
            forfeit_stage=data.get("forfeit_stage", ""),
            forfeit_error=data.get("forfeit_error", ""),
            journal_path=Path(data["journal_path"]) if data.get("journal_path") else None,
            metadata=data.get("metadata", {}),
        )

    def save(self, output_dir: Path) -> Path:
        """Save bot artifact metadata to JSON file."""
        from mjarena.utils.file import ensure_dir
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        metadata_path = output_dir / "bot_artifact.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return metadata_path

    @classmethod
    def load(cls, path: Path) -> "BotArtifact":
        """Load bot artifact from JSON file."""
        path = Path(path)
        if path.is_dir():
            path = path / "bot_artifact.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        artifact = cls.from_dict(data)
        base_dir = path.parent

        # Older artifacts, especially copied baseline bots, may store paths
        # relative to their own artifact directory (e.g. "robot.xml").
        # Only rebase relative paths that do not already resolve as written;
        # newer artifacts often store valid repo-relative paths like logs/... .
        for field_name in (
            "morphology_xml",
            "controller_code",
            "journal_path",
        ):
            value = getattr(artifact, field_name)
            if value is not None and not value.is_absolute() and not value.exists():
                setattr(artifact, field_name, base_dir / value)

        return artifact
