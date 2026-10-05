"""
Rating computation module: loads match JSONs, computes ratings via multiple
strategies and methods, caches to CSV for fast re-reads.

Strategies (which matches to include):
  - cumulative: All matches across all iterations.
  - topk: Top K bots per model by win rate, only their matches.

Ratings are always Bradley-Terry MLE (order-independent, same as LMSYS).
Both strategies support bot-level and model-level aggregation.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

from .core import MatchOutcome, bradley_terry_ratings, compute_standings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A match row is a flat dict with all fields from the JSON (one row per seed).
MatchRow = Dict[str, Any]

# ---------------------------------------------------------------------------
# CSV column order (stable for reproducibility)
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "red_bot", "blue_bot", "red_model", "blue_model",
    "red_id", "blue_id", "iteration", "timestamp",
    "seed", "winner", "red_score",
    "num_steps", "winner_step", "progress",
    "initial_distance",
    "initial_red_com", "initial_blue_com",
    "forfeit",
    "physics_unstable",
    "red_artifact", "blue_artifact",
    "red_debug_dir", "blue_debug_dir",
    "matchup_video",
]


# ---------------------------------------------------------------------------
# Loading & caching
# ---------------------------------------------------------------------------


def _walk_match_jsons(season_dir: Path) -> List[MatchRow]:
    """Walk season directory tree and load all match_result.json files.

    Returns one MatchRow per seed (i.e. per individual match).
    """
    rows: List[MatchRow] = []
    pattern = "tournament_*/round_robin_match/matches/*/match_result.json"

    for json_path in sorted(season_dir.glob(pattern)):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read %s: %s", json_path, exc)
            continue

        # Infer iteration from directory name (tournament_XX) if not in JSON
        iteration = data.get("iteration")
        if iteration is None:
            m = re.search(r"tournament_(\d+)", str(json_path))
            iteration = int(m.group(1)) if m else 0

        # Top-level fields shared across all seeds in this matchup
        top = {
            "red_bot": data.get("red_bot", ""),
            "blue_bot": data.get("blue_bot", ""),
            "red_model": data.get("red_model", ""),
            "blue_model": data.get("blue_model", ""),
            "red_id": data.get("red_id", ""),
            "blue_id": data.get("blue_id", ""),
            "iteration": iteration,
            "timestamp": data.get("timestamp", ""),
            "red_artifact": data.get("red_artifact", ""),
            "blue_artifact": data.get("blue_artifact", ""),
            "red_debug_dir": data.get("red_debug_dir", ""),
            "blue_debug_dir": data.get("blue_debug_dir", ""),
            "matchup_video": data.get("matchup_video", ""),
        }

        for m in data.get("matches", []):
            row = dict(top)
            row["seed"] = m.get("seed", 0)
            row["winner"] = m.get("winner", "tie")
            row["red_score"] = m.get("red_score", 0.0)
            row["num_steps"] = m.get("num_steps", 0)
            row["winner_step"] = m.get("winner_step", 0)
            row["progress"] = m.get("progress", 0.0)
            row["initial_distance"] = m.get("initial_distance", 0.0)
            row["initial_red_com"] = json.dumps(m.get("initial_red_com", []))
            row["initial_blue_com"] = json.dumps(m.get("initial_blue_com", []))
            row["forfeit"] = m.get("forfeit", False)
            row["physics_unstable"] = m.get("physics_unstable", False)
            rows.append(row)

    return rows


def _write_csv(rows: List[MatchRow], csv_path: Path) -> None:
    """Write match rows to CSV."""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("Wrote %d rows to %s", len(rows), csv_path)


def _read_csv(csv_path: Path) -> List[MatchRow]:
    """Read match rows from CSV."""
    rows: List[MatchRow] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ("red_id", "blue_id", "iteration", "seed",
                        "num_steps", "winner_step"):
                if key in row and row[key]:
                    try:
                        row[key] = int(row[key])
                    except (ValueError, TypeError):
                        row[key] = 0
            for key in ("red_score", "progress", "initial_distance"):
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        row[key] = 0.0
            # Convert booleans back from string
            if "forfeit" in row:
                row["forfeit"] = row["forfeit"] in ("True", "true", "1", True)
            if "physics_unstable" in row:
                row["physics_unstable"] = row["physics_unstable"] in ("True", "true", "1", True)
            rows.append(row)
    return rows


def load_matches(
    season_dir: Path,
    *,
    force: bool = False,
    iteration_range: Optional[Tuple[int, int]] = None,
    models: Optional[List[str]] = None,
    exclude_forfeits: bool = False,
    exclude_unstable: bool = False,
) -> List[MatchRow]:
    """Load all match data from a season directory.

    Checks for cached ``match_log.csv``; if missing or *force=True*, walks
    all ``match_result.json`` files and writes the CSV.

    Args:
        season_dir: Root season output directory.
        force: Re-scan JSONs even if CSV exists.
        iteration_range: (start, end) inclusive range of iterations to keep.
        models: If set, keep only matches where at least one bot's model
            is in this list.
        exclude_forfeits: Drop forfeit matches.
        exclude_unstable: Drop matches with physics instability.

    Returns:
        List of MatchRow dicts (one per individual seed/match).
    """
    season_dir = Path(season_dir)
    csv_path = season_dir / "match_log.csv"

    if csv_path.exists() and not force:
        logger.info("Loading cached match log from %s", csv_path)
        rows = _read_csv(csv_path)
    else:
        logger.info("Scanning match JSONs in %s ...", season_dir)
        rows = _walk_match_jsons(season_dir)
        if rows:
            _write_csv(rows, csv_path)
        else:
            logger.warning("No match results found in %s", season_dir)

    # Apply filters
    if iteration_range is not None:
        lo, hi = iteration_range
        rows = [r for r in rows if lo <= int(r.get("iteration", 0)) <= hi]

    if models is not None:
        model_set = set(models)
        rows = [r for r in rows
                if r.get("red_model") in model_set or r.get("blue_model") in model_set]

    if exclude_forfeits:
        rows = [r for r in rows if not r.get("forfeit", False)]

    if exclude_unstable:
        rows = [r for r in rows if not r.get("physics_unstable", False)]

    return rows


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def strategy_cumulative(matches: List[MatchRow]) -> List[MatchRow]:
    """Return all matches sorted by iteration then seed."""
    return sorted(matches, key=lambda r: (int(r.get("iteration", 0)), int(r.get("seed", 0))))


def strategy_topk(matches: List[MatchRow], k: int = 3) -> List[MatchRow]:
    """Keep only matches involving the top K bots per model (by win rate).

    Args:
        matches: All match rows.
        k: Number of top bots to keep per model.

    Returns:
        Filtered match rows.
    """
    # Compute win rates per bot
    bot_wins: Dict[str, int] = {}
    bot_games: Dict[str, int] = {}
    bot_model: Dict[str, str] = {}

    for m in matches:
        for side, bot_key, model_key in [("red", "red_bot", "red_model"),
                                          ("blue", "blue_bot", "blue_model")]:
            bot = m.get(bot_key, "")
            model = m.get(model_key, "")
            if not bot:
                continue
            bot_model[bot] = model
            bot_games[bot] = bot_games.get(bot, 0) + 1
            if m.get("winner") == side:
                bot_wins[bot] = bot_wins.get(bot, 0) + 1

    # Group bots by model, sort by win rate
    model_bots: Dict[str, List[Tuple[str, float]]] = {}
    for bot, games in bot_games.items():
        model = bot_model.get(bot, "")
        wr = bot_wins.get(bot, 0) / games if games > 0 else 0.0
        model_bots.setdefault(model, []).append((bot, wr))

    # Pick top K per model
    keep_bots = set()
    for model, bots in model_bots.items():
        bots.sort(key=lambda x: x[1], reverse=True)
        for bot_name, _ in bots[:k]:
            keep_bots.add(bot_name)

    return [m for m in matches
            if m.get("red_bot") in keep_bots and m.get("blue_bot") in keep_bots]


# ---------------------------------------------------------------------------
# Helper: rows → MatchOutcome list
# ---------------------------------------------------------------------------


def _rows_to_outcomes(
    rows: List[MatchRow],
    level: str = "bot",
) -> List[MatchOutcome]:
    """Convert flat match rows to MatchOutcome list.

    Args:
        rows: Match rows.
        level: "bot" uses red_bot/blue_bot, "model" uses red_model/blue_model.

    Returns:
        List of MatchOutcome.
    """
    outcomes: List[MatchOutcome] = []
    for r in rows:
        if level == "model":
            pa = r.get("red_model", r.get("red_bot", ""))
            pb = r.get("blue_model", r.get("blue_bot", ""))
        else:
            pa = r.get("red_bot", "")
            pb = r.get("blue_bot", "")

        winner_side = r.get("winner", "tie")
        if winner_side == "red":
            winner = pa
        elif winner_side == "blue":
            winner = pb
        else:
            winner = "draw"

        outcomes.append(MatchOutcome(player_a=pa, player_b=pb, winner=winner))
    return outcomes


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def compute_bradley_terry(
    matches: List[MatchRow],
    level: str = "bot",
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """Bradley-Terry MLE via MM (minorization-maximization) algorithm.

    Order-independent, same family as LMSYS Chatbot Arena.
    Uses 1 virtual draw per pair for regularization (handles 0-win players).

    Args:
        matches: Match rows.
        level: "bot" or "model".
        max_iter: Maximum MM iterations.
        tol: Convergence tolerance.

    Returns:
        {player: elo_rating} (converted from BT strengths via 400*log10 + 1000).
    """
    outcomes = _rows_to_outcomes(matches, level=level)
    return bradley_terry_ratings(outcomes, max_iter=max_iter, tol=tol)


def bootstrap_ratings(
    matches: List[MatchRow],
    compute_fn: Callable[[List[MatchRow]], Dict[str, float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Bootstrap confidence intervals for ratings.

    Resamples matches N times, computes ratings each time, reports
    (median, lo_2.5%, hi_97.5%) per player.

    Args:
        matches: Match rows.
        compute_fn: Rating function (e.g., partial of compute_bradley_terry).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        {player: (median, lo_2.5, hi_97.5)}
    """
    rng = random.Random(seed)
    n = len(matches)
    if n == 0:
        return {}

    all_ratings: Dict[str, List[float]] = {}

    for _ in range(n_bootstrap):
        sample = [matches[rng.randint(0, n - 1)] for _ in range(n)]
        ratings = compute_fn(sample)
        for player, rating in ratings.items():
            all_ratings.setdefault(player, []).append(rating)

    result: Dict[str, Tuple[float, float, float]] = {}
    for player, samples in all_ratings.items():
        samples.sort()
        lo_idx = max(0, int(0.025 * len(samples)))
        hi_idx = min(len(samples) - 1, int(0.975 * len(samples)))
        median_idx = len(samples) // 2
        result[player] = (samples[median_idx], samples[lo_idx], samples[hi_idx])

    return result


def compute_win_loss_draw(
    matches: List[MatchRow],
    level: str = "bot",
) -> Dict[str, Dict[str, int]]:
    """Compute win/loss/draw counts per player.

    Args:
        matches: Match rows.
        level: "bot" or "model".

    Returns:
        {player: {"wins": int, "losses": int, "draws": int, "games": int}}
    """
    stats: Dict[str, Dict[str, int]] = {}

    def ensure(name: str):
        if name not in stats:
            stats[name] = {"wins": 0, "losses": 0, "draws": 0, "games": 0}

    for r in matches:
        if level == "model":
            pa = r.get("red_model", r.get("red_bot", ""))
            pb = r.get("blue_model", r.get("blue_bot", ""))
        else:
            pa = r.get("red_bot", "")
            pb = r.get("blue_bot", "")

        ensure(pa)
        ensure(pb)
        stats[pa]["games"] += 1
        stats[pb]["games"] += 1

        winner_side = r.get("winner", "tie")
        if winner_side == "red":
            stats[pa]["wins"] += 1
            stats[pb]["losses"] += 1
        elif winner_side == "blue":
            stats[pb]["wins"] += 1
            stats[pa]["losses"] += 1
        else:
            stats[pa]["draws"] += 1
            stats[pb]["draws"] += 1

    return stats


# ---------------------------------------------------------------------------
# High-level: compute everything and save results
# ---------------------------------------------------------------------------


def compute_ratings(
    season_dir: Path,
    *,
    level: str = "model",
    strategy: str = "cumulative",
    top_k: int = 3,
    bootstrap: int = 0,
    exclude_forfeits: bool = False,
    exclude_unstable: bool = False,
    force: bool = True,
    iteration_range: Optional[Tuple[int, int]] = None,
    models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute ratings end-to-end: load, filter, rate.

    Ratings are always Bradley-Terry MLE (order-independent, same as LMSYS).

    Args:
        season_dir: Season output directory.
        level: "bot" or "model".
        strategy: "cumulative" or "topk".
        top_k: K for topk strategy.
        bootstrap: Number of bootstrap resamples (0 = off).
        exclude_forfeits: Drop forfeit matches.
        exclude_unstable: Drop matches with physics instability.
        force: Re-scan JSONs (skip CSV cache).
        iteration_range: (lo, hi) inclusive iteration filter.
        models: Model name filter.

    Returns:
        Dict with keys: ratings, confidence (or None), wld, matches,
        metadata (level, strategy, etc.).
    """
    matches = load_matches(
        season_dir,
        force=force,
        iteration_range=iteration_range,
        models=models,
        exclude_forfeits=exclude_forfeits,
        exclude_unstable=exclude_unstable,
    )

    # Apply strategy
    if strategy == "topk":
        matches = strategy_topk(matches, k=top_k)
    else:
        matches = strategy_cumulative(matches)

    # Compute ratings (Bradley-Terry MLE).
    ratings = compute_bradley_terry(matches, level=level)

    # Bootstrap CIs
    confidence = None
    if bootstrap > 0:
        from functools import partial
        fn = partial(compute_bradley_terry, level=level)
        confidence = bootstrap_ratings(matches, fn, n_bootstrap=bootstrap)

    # W/L/D
    wld = compute_win_loss_draw(matches, level=level)

    return {
        "ratings": ratings,
        "confidence": confidence,
        "wld": wld,
        "matches": matches,
        "metadata": {
            "method": "bt",
            "level": level,
            "strategy": strategy,
            "top_k": top_k if strategy == "topk" else None,
            "bootstrap": bootstrap,
            "exclude_forfeits": exclude_forfeits,
            "exclude_unstable": exclude_unstable,
            "total_matches": len(matches),
            "timestamp": datetime.now().isoformat(),
        },
    }
