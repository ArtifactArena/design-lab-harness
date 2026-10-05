"""Tournament module for bot competitions."""
from mjarena.elo.core import (
    MatchOutcome,
    bradley_terry_ratings,
    compute_standings,
    elo_rank_players,
)
from .types import (
    MatchResult,
    MatchupResult,
    TournamentResult,
)
from .tournament import (
    run_matchup,
    run_round_robin,
    get_win_rate_matrix,
)

__all__ = [
    # Ratings
    "MatchOutcome",
    "bradley_terry_ratings",
    "compute_standings",
    "elo_rank_players",
    # Tournament
    "MatchResult",
    "MatchupResult",
    "TournamentResult",
    "run_matchup",
    "run_round_robin",
    "get_win_rate_matrix",
]
