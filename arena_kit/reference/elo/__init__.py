"""Rating system: Bradley-Terry ratings, methods, and visualization."""
from .core import (
    MatchOutcome,
    bradley_terry_ratings,
    compute_standings,
    elo_rank_players,
)

__all__ = [
    # Core
    "MatchOutcome",
    "bradley_terry_ratings",
    "compute_standings",
    "elo_rank_players",
]
