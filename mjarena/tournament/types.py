"""
Type definitions for tournament module.

Contains:
- MatchResult: Result of a single match between two bots
- MatchupResult: Result of running multiple seeds between two bots
- TournamentResult: Complete results from a round-robin tournament
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mjarena.design_shop.types import MatchupResult as DSMatchupResult


@dataclass
class MatchResult:
    """Result of a single match between two bots.

    Attributes:
        red_bot: Name of red (first) bot
        blue_bot: Name of blue (second) bot
        winner: "red", "blue", or "tie"
        red_score: Score from red's perspective
        seed: Random seed used for this match
        steps: Number of simulation steps
        video_path: Path to video file (if recorded)
        details: Additional match details
    """
    red_bot: str
    blue_bot: str
    winner: str
    red_score: float
    seed: int
    steps: int = 0
    video_path: Optional[Path] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "red_bot": self.red_bot,
            "blue_bot": self.blue_bot,
            "winner": self.winner,
            "red_score": self.red_score,
            "seed": self.seed,
            "steps": self.steps,
            "video_path": str(self.video_path) if self.video_path else None,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatchResult":
        """Create from dict."""
        video_path = data.get("video_path")
        if video_path:
            video_path = Path(video_path)
        return cls(
            red_bot=data["red_bot"],
            blue_bot=data["blue_bot"],
            winner=data["winner"],
            red_score=data["red_score"],
            seed=data["seed"],
            steps=data.get("steps", 0),
            video_path=video_path,
            details=data.get("details", {}),
        )


@dataclass
class MatchupResult:
    """Result of running multiple seeds between two bots.

    Attributes:
        bot_a: Name of first bot
        bot_b: Name of second bot
        matches: Individual match results
        _ds_matchup: Typed MatchupResult from design_shop.types (carries GameRecords)
        wins_a: Number of wins for bot_a
        wins_b: Number of wins for bot_b
        ties: Number of ties
    """
    bot_a: str
    bot_b: str
    matches: List[MatchResult]
    _ds_matchup: Optional["DSMatchupResult"] = None
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0


@dataclass
class TournamentResult:
    """Complete results from a round-robin tournament.

    Attributes:
        matches: List of all match results
        standings: Dict mapping bot name to stats (wins, losses, draws, elo)
        elo_history: History of Elo ratings (for plotting)
        metadata: Tournament configuration and timing info
    """
    matches: List[MatchResult]
    standings: Dict[str, Dict]
    elo_history: List[Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matches": [m.to_dict() for m in self.matches],
            "standings": self.standings,
            "elo_history": self.elo_history,
            "metadata": self.metadata,
        }

    def save(self, output_path: Path) -> None:
        """Save tournament results to JSON file."""
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TournamentResult":
        """Load tournament results from JSON file."""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            matches=[MatchResult.from_dict(m) for m in data["matches"]],
            standings=data["standings"],
            elo_history=data["elo_history"],
            metadata=data.get("metadata", {}),
        )
