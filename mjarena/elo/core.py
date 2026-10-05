"""
Rating system for bot tournaments.

Ratings are computed with the Bradley-Terry MLE (minorization-maximization),
the same order-independent estimator used by LMSYS Chatbot Arena. There is no
sequential/K-factor Elo in this codebase any more — every standing is BT.
Strengths are reported on an Elo-like scale via ``400 * log10(gamma) + 1000``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class MatchOutcome:
    """Single match outcome for rating calculation.

    Attributes:
        player_a: Name/ID of first player
        player_b: Name/ID of second player
        winner: Name/ID of winner, or "draw" for tie
    """
    player_a: str
    player_b: str
    winner: str  # player_a, player_b, or "draw"

    @property
    def is_draw(self) -> bool:
        return self.winner == "draw"


def bradley_terry_ratings(
    matches: List[MatchOutcome],
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """Bradley-Terry MLE via MM (minorization-maximization) algorithm.

    Order-independent, same family as LMSYS Chatbot Arena. Uses 1 virtual draw
    per pair for regularization (handles 0-win players). This is the single
    canonical rating implementation; every standings function routes through it.

    Args:
        matches: List of match outcomes (one per seed/rollout).
        max_iter: Maximum MM iterations.
        tol: Convergence tolerance.

    Returns:
        {player: elo_rating} (converted from BT strengths via 400*log10 + 1000).
    """
    players: List[str] = sorted(
        {o.player_a for o in matches} | {o.player_b for o in matches}
    )
    if len(players) < 2:
        return {p: 1000.0 for p in players}

    idx = {p: i for i, p in enumerate(players)}
    n = len(players)

    # Build win matrix (draws count as half a win to each side).
    W = [[0.0] * n for _ in range(n)]
    for o in matches:
        i, j = idx[o.player_a], idx[o.player_b]
        if o.is_draw:
            W[i][j] += 0.5
            W[j][i] += 0.5
        elif o.winner == o.player_a:
            W[i][j] += 1.0
        elif o.winner == o.player_b:
            W[j][i] += 1.0
        else:
            # Unknown winner id -> treat as a draw.
            W[i][j] += 0.5
            W[j][i] += 0.5

    # Regularization: add 1 virtual draw per pair.
    for i in range(n):
        for j in range(i + 1, n):
            W[i][j] += 0.5
            W[j][i] += 0.5

    # MM algorithm.
    gamma = [1.0] * n
    for _ in range(max_iter):
        gamma_new = [0.0] * n
        for i in range(n):
            num = 0.0
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                n_ij = W[i][j] + W[j][i]
                if n_ij == 0:
                    continue
                num += W[i][j]
                denom += n_ij / (gamma[i] + gamma[j])
            gamma_new[i] = num / denom if denom > 0 else gamma[i]

        # Normalize so geometric mean = 1.
        log_mean = sum(math.log(max(g, 1e-15)) for g in gamma_new) / n
        gamma_new = [g / math.exp(log_mean) for g in gamma_new]

        max_diff = max(abs(gamma_new[i] - gamma[i]) for i in range(n))
        gamma = gamma_new
        if max_diff < tol:
            break

    return {
        p: 400.0 * math.log10(max(gamma[idx[p]], 1e-15)) + 1000.0
        for p in players
    }


def compute_standings(matches: List[MatchOutcome]) -> Dict[str, Dict]:
    """Compute full standings including wins, losses, draws, and BT Elo.

    Args:
        matches: List of match outcomes (one per seed/rollout).

    Returns:
        Dictionary mapping player names to stats dict with:
        - wins: int
        - losses: int
        - draws: int
        - elo: float  (Bradley-Terry MLE on the Elo-like scale)
        - games: int
    """
    stats: Dict[str, Dict] = {}

    def ensure_player(name: str):
        if name not in stats:
            stats[name] = {
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "elo": 1000.0,
                "games": 0,
            }

    for match in matches:
        ensure_player(match.player_a)
        ensure_player(match.player_b)

        stats[match.player_a]["games"] += 1
        stats[match.player_b]["games"] += 1

        if match.is_draw:
            stats[match.player_a]["draws"] += 1
            stats[match.player_b]["draws"] += 1
        elif match.winner == match.player_a:
            stats[match.player_a]["wins"] += 1
            stats[match.player_b]["losses"] += 1
        elif match.winner == match.player_b:
            stats[match.player_b]["wins"] += 1
            stats[match.player_a]["losses"] += 1

    ratings = bradley_terry_ratings(matches)
    for name, elo in ratings.items():
        if name in stats:
            stats[name]["elo"] = elo

    return stats


def elo_rank_players(ratings: Dict[str, float]) -> List[Tuple[str, float]]:
    """Rank players by rating (highest first).

    Args:
        ratings: Dictionary of player name to rating

    Returns:
        List of (player_name, rating) sorted by rating descending
    """
    return sorted(ratings.items(), key=lambda x: x[1], reverse=True)


__all__ = [
    "MatchOutcome",
    "bradley_terry_ratings",
    "compute_standings",
    "elo_rank_players",
]
