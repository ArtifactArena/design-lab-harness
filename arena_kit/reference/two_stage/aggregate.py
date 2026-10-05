"""Stage 4 — model Elo aggregation + plot.

Reads the cross-model pool file + saved match results; never simulates a match.
For each K' in k_values, filters matches to participants in the top-K' subset
(baselines always included), reruns Elo, aggregates per-model, and writes
`elo_model_top{K'}.json`. Renders a grouped bar chart.

Also writes app-compatible `elo_results_top{K'}.json` files (matching the
`mjarena/elo/display.py` schema with `ratings` + `wld` + `metadata`) and
per-K family-shaded leaderboards via `plot_leaderboard`, so the existing
app/api/leaderboard route can serve them with a `?top_k=` selector.
"""
from __future__ import annotations

import json
import logging
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mjarena.elo.core import MatchOutcome, bradley_terry_ratings, compute_standings
from mjarena.two_stage.artifact_loader import read_top_k_bots
from mjarena.two_stage.cross_model import _allowed_artifact_ids, _filter_outcomes_for_subset
from mjarena.two_stage.intra_model import _read_match_outcomes

logger = logging.getLogger(__name__)


def _compute_bt_ratings(outcomes: List[MatchOutcome]) -> Dict[str, float]:
    """Bradley-Terry MLE on a list of MatchOutcomes (player keys: artifact ids)."""
    return bradley_terry_ratings(outcomes)


def _compute_bt_standings(outcomes: List[MatchOutcome]) -> Dict[str, Dict[str, Any]]:
    """Standings (W/L/D + BT Elo). Thin alias over ``compute_standings``, which
    is itself Bradley-Terry; kept so existing call sites stay drop-in."""
    return compute_standings(outcomes)


def _aggregate_model_elo_mean(
    rows: List[Dict[str, Any]],
    standings: Dict[str, Dict[str, Any]],
    k_prime: int,
) -> Dict[str, Dict[str, Any]]:
    """For each model: mean Elo over its surviving artifacts (rank<=k_prime).

    Baselines: one Elo per baseline (their single artifact).
    Returns {model: {"elo": float, "n_artifacts": int, "artifact_elos": [...]}}.
    """
    by_model: Dict[str, List[float]] = {}
    for row in rows:
        aid = row["artifact_id"]
        if aid not in standings:
            continue
        if row["kind"] == "baseline":
            model = row["model"]
        else:
            if int(row["rank"]) > k_prime:
                continue
            model = row["model"]
        by_model.setdefault(model, []).append(float(standings[aid]["elo"]))

    out: Dict[str, Dict[str, Any]] = {}
    for model, elos in by_model.items():
        out[model] = {
            "elo": statistics.mean(elos),
            "elo_min": min(elos),
            "elo_max": max(elos),
            "n_artifacts": len(elos),
            "artifact_elos": elos,
        }
    return out


def _aggregate_model_elo_median(
    rows: List[Dict[str, Any]],
    standings: Dict[str, Dict[str, Any]],
    k_prime: int,
) -> Dict[str, Dict[str, Any]]:
    """For each model: MEDIAN Elo over its surviving artifacts (rank<=k_prime).

    Same shape as `_aggregate_model_elo_mean` but uses the median, which is robust
    to a single tanking artifact dragging the central tendency (a weak bot that
    lost almost everything gets an extreme-low BT Elo and would skew the mean).
    Baselines: one Elo per baseline (their single artifact).
    """
    by_model: Dict[str, List[float]] = {}
    for row in rows:
        aid = row["artifact_id"]
        if aid not in standings:
            continue
        if row["kind"] == "baseline":
            model = row["model"]
        else:
            if int(row["rank"]) > k_prime:
                continue
            model = row["model"]
        by_model.setdefault(model, []).append(float(standings[aid]["elo"]))

    out: Dict[str, Dict[str, Any]] = {}
    for model, elos in by_model.items():
        out[model] = {
            "elo": statistics.median(elos),
            "elo_min": min(elos),
            "elo_max": max(elos),
            "n_artifacts": len(elos),
            "artifact_elos": elos,
        }
    return out


def _aggregate_model_elo_pooled(
    rows: List[Dict[str, Any]],
    outcomes: List[MatchOutcome],
    k_prime: int,
    *,
    elo_k_factor: float,
    elo_initial_rating: float,
) -> Dict[str, Dict[str, Any]]:
    """Treat all of a model's surviving artifacts as one player; recompute Elo.

    Maps every artifact_id to its model name (or baseline name) and re-runs
    compute_standings on the model-keyed match list.
    """
    aid_to_model: Dict[str, str] = {}
    for row in rows:
        if row["kind"] == "baseline":
            aid_to_model[row["artifact_id"]] = row["model"]
        elif int(row["rank"]) <= k_prime:
            aid_to_model[row["artifact_id"]] = row["model"]
    allowed = set(aid_to_model.keys())
    filtered = [o for o in outcomes if o.player_a in allowed and o.player_b in allowed]
    relabelled: List[MatchOutcome] = []
    for o in filtered:
        a = aid_to_model[o.player_a]
        b = aid_to_model[o.player_b]
        if a == b:
            continue
        if o.winner == "draw":
            relabelled.append(MatchOutcome(player_a=a, player_b=b, winner="draw"))
        else:
            winner_model = aid_to_model.get(o.winner, o.winner)
            relabelled.append(MatchOutcome(player_a=a, player_b=b, winner=winner_model))
    return _compute_bt_standings(relabelled)


def _aggregate_model_elo_max(
    rows: List[Dict[str, Any]],
    standings: Dict[str, Dict[str, Any]],
    k_prime: int,
) -> Dict[str, Dict[str, Any]]:
    """Per-model max: Elo of the model's single best artifact (peak capability).
    With max, top@1 ≥ top@3 ≥ top@5 always — metric never drops as K grows."""
    by_model: Dict[str, List[float]] = {}
    for row in rows:
        aid = row["artifact_id"]
        if aid not in standings:
            continue
        if row["kind"] == "baseline":
            model = row["model"]
        else:
            if int(row["rank"]) > k_prime:
                continue
            model = row["model"]
        by_model.setdefault(model, []).append(float(standings[aid]["elo"]))

    out: Dict[str, Dict[str, Any]] = {}
    for model, elos in by_model.items():
        out[model] = {
            "elo": max(elos),
            "elo_min": min(elos),
            "elo_max": max(elos),
            "n_artifacts": len(elos),
            "artifact_elos": elos,
        }
    return out


_AGGREGATION_FNS: Dict[str, Any] = {
    "mean": lambda elos: statistics.mean(elos),
    "max": lambda elos: max(elos),
    "min": lambda elos: min(elos),
}

# Baselines with no actuators (actuator_names == []) — can't push each other,
# so any stationary-vs-stationary match runs to inactivity timeout and is
# always a draw. Those matches contribute zero information to BT MLE; we
# filter them out of all aggregations so they don't inflate game counts or
# pollute the H2H matrix. Add a model name here if you introduce new
# stationary baselines.
_STATIONARY_BASELINES: frozenset = frozenset({
    "tetrahedron", "hexahedron", "octahedron",
    "dodecahedron", "icosahedron", "box",
})


def _apply_anchor(
    ratings: Dict[str, Dict[str, Any]],
    *,
    anchor: Optional[str],
    anchor_elo: float = 1000.0,
) -> Dict[str, Dict[str, Any]]:
    """Shift every artifact's Elo so the anchor sits at `anchor_elo`.

    `anchor` is matched against artifact_id case-insensitively (substring),
    so "box" matches "baseline__Box", "tetrahedron" matches
    "baseline__tetrahedron", etc. Pass `None` (or "none") to skip pinning.

    BT MLE itself is invariant under uniform shifts of all log-strengths,
    so applying a shift after MLE preserves all relative differences and
    just renames the absolute scale.
    """
    if not anchor or anchor.lower() == "none":
        return ratings
    needle = anchor.lower()
    match: Optional[str] = None
    for aid in ratings:
        if needle in aid.lower():
            match = aid
            break
    if match is None:
        logger.warning(
            "anchor=%r not found among artifact ids; skipping pin",
            anchor,
        )
        return ratings
    shift = anchor_elo - ratings[match]["elo"]
    if abs(shift) < 1e-9:
        return ratings
    out: Dict[str, Dict[str, Any]] = {}
    for aid, stats in ratings.items():
        out[aid] = dict(stats)
        out[aid]["elo"] = stats["elo"] + shift
    logger.info(
        "Anchored %s to %.1f (shift=%+.2f)", match, anchor_elo, shift
    )
    return out


def _build_app_elo_results(
    rows: List[Dict[str, Any]],
    outcomes: List[MatchOutcome],
    k_prime: int,
    *,
    elo_k_factor: float,
    elo_initial_rating: float,
    aggregation: str = "mean",
    full_artifact_standings: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the {ratings, wld, metadata} schema that mjarena/elo/display.py and
    the app's /api/leaderboard route both consume.

    Single-BT-pass mode (preferred): pass `full_artifact_standings` computed
    once on the full T_max match list (already anchor-shifted if desired).
    This call then just filters by k_prime and aggregates the frozen Elos.
    Result: top@1 / top@3 / top@5 share a scale and `max` is monotone in K.

    Per-K-pass mode (legacy, still supported when `full_artifact_standings`
    is None): re-runs BT MLE on the K-filtered match subset, which uses a
    different normalization per K. top@1/top@3/top@5 numbers are NOT comparable
    across K in this mode.
    """
    aid_to_model: Dict[str, str] = {}
    aid_is_eligible: Dict[str, bool] = {}
    for row in rows:
        aid = row["artifact_id"]
        aid_to_model[aid] = row["model"]
        if row["kind"] == "baseline":
            aid_is_eligible[aid] = True
        else:
            aid_is_eligible[aid] = int(row["rank"]) <= k_prime

    allowed = {aid for aid, ok in aid_is_eligible.items() if ok}

    if full_artifact_standings is None:
        filtered = [
            o for o in outcomes if o.player_a in allowed and o.player_b in allowed
        ]
        artifact_standings = _compute_bt_standings(filtered)
        n_matches_used = len(filtered)
        single_pass = False
    else:
        # Pull only the eligible artifacts' frozen ratings out of the
        # already-computed full standings.
        artifact_standings = {
            aid: full_artifact_standings[aid]
            for aid in allowed
            if aid in full_artifact_standings
        }
        n_matches_used = sum(s.get("games", 0) for s in artifact_standings.values()) // 2
        single_pass = True

    # Aggregate to model-level
    ratings: Dict[str, float] = {}
    wld: Dict[str, Dict[str, int]] = {}
    elos_by_model: Dict[str, List[float]] = {}
    wld_by_model: Dict[str, Dict[str, int]] = {}

    for aid, stats in artifact_standings.items():
        model = aid_to_model.get(aid, aid)
        elos_by_model.setdefault(model, []).append(float(stats["elo"]))
        agg = wld_by_model.setdefault(
            model, {"wins": 0, "losses": 0, "draws": 0, "games": 0}
        )
        agg["wins"] += int(stats["wins"])
        agg["losses"] += int(stats["losses"])
        agg["draws"] += int(stats["draws"])
        agg["games"] += int(stats["games"])

    agg_fn = _AGGREGATION_FNS.get(aggregation, _AGGREGATION_FNS["mean"])
    for model, elos in elos_by_model.items():
        ratings[model] = round(agg_fn(elos), 2)
        wld[model] = wld_by_model[model]

    return {
        "ratings": ratings,
        "wld": wld,
        "confidence": None,
        "metadata": {
            "method": "bt",
            "single_bt_pass": single_pass,
            "top_k": k_prime,
            "aggregation": aggregation,
            "total_matches": n_matches_used,
            "n_models_or_baselines": len(ratings),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "two_stage_tournaments",
        },
    }


def _render_family_leaderboard(
    ratings: Dict[str, float],
    output_path: Path,
    *,
    k_prime: int,
    method: str = "BT",
) -> None:
    """Render with `plot_leaderboard` from mjarena.elo.display so the style
    matches the existing per-model Elo leaderboards (family-shaded bars,
    rounded tops, etc.)."""
    try:
        from mjarena.elo.display import plot_leaderboard
    except ImportError:
        logger.warning("mjarena.elo.display not importable; skipping family plot")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_leaderboard(
        ratings,
        method=method,
        title=f"Two-Stage Model Elo (top-{k_prime})",
        subtitle=f"K={k_prime} artifacts per model · cross-model RR · {method}",
        output_path=output_path,
    )


def _render_leaderboard_table_vertical(
    ratings: Dict[str, float],
    wld: Dict[str, Dict[str, int]],
    output_path: Path,
    *,
    title: str = "",
    method: str = "BT",
) -> None:
    """Compact leaderboard table with models on the Y-axis (one row each) and
    stats as columns: Rank · Model · Family · Elo · Win % · W-L-D · Games.

    Sorted by Elo descending. Family color shows as a small left-edge swatch
    on each row. Drop-in alternative to the horizontal `plot_leaderboard_table`
    when there are many models — much easier to scan vertically.
    """
    try:
        import matplotlib.pyplot as plt
        from mjarena.elo.display import (
            _DEFAULT_BASE,
            _FAMILY_BASE_COLORS,
            _detect_family,
            _get_display_name,
            _setup_style,
        )
    except ImportError:
        logger.warning("matplotlib unavailable; skipping vertical table")
        return
    _setup_style()

    if not ratings:
        return

    ranked = sorted(ratings.items(), key=lambda kv: -kv[1])
    n = len(ranked)

    # Column layout (in axes data coords). Model column is sized to the
    # longest display name so we don't waste horizontal space.
    display_for_sizing = [_get_display_name(m) for m, _ in ranked]
    longest_chars = max((len(s) for s in display_for_sizing), default=12)
    # Empirical: ~0.13 axes-unit per character at fontsize 10, plus padding.
    model_col_w = max(2.0, longest_chars * 0.13 + 0.3)

    cols = [
        ("Rank", 0.45),
        ("Model", model_col_w),
        ("Family", 1.4),
        ("Elo", 0.9),
        ("Win %", 0.9),
        ("W / L / D", 2.4),
        ("Games", 0.9),
    ]
    col_widths = [w for _label, w in cols]
    col_x: List[float] = [0.0]
    for w in col_widths:
        col_x.append(col_x[-1] + w)
    total_w = col_x[-1]

    # Figure size: width tracks total column width; height grows with rows.
    fig_w = max(7.0, total_w * 0.88)
    fig_h = max(2.5, 0.40 * n + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, n + 1)  # +1 for header row
    ax.invert_yaxis()
    ax.axis("off")

    # Header row at y = 0.5
    header_y = 0.5
    for i, (label, _w) in enumerate(cols):
        cx = (col_x[i] + col_x[i + 1]) / 2
        ax.text(
            cx, header_y, label,
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="#444444",
        )
    # Header underline
    ax.plot([0.1, total_w - 0.1], [1.0, 1.0],
            color="#888888", linewidth=1.2)

    # Data rows
    for row_idx, (model, elo) in enumerate(ranked):
        y = row_idx + 1.5  # below header
        # Alternating row tint (very subtle)
        if row_idx % 2 == 1:
            ax.add_patch(plt.Rectangle(
                (0, y - 0.45), total_w, 0.9,
                facecolor="#F2EFE9", edgecolor="none", zorder=0,
            ))

        family = _detect_family(model)
        base_rgb = _FAMILY_BASE_COLORS.get(family, _DEFAULT_BASE)
        color = "#%02x%02x%02x" % (
            int(base_rgb[0] * 255),
            int(base_rgb[1] * 255),
            int(base_rgb[2] * 255),
        )

        w = wld.get(model, {}) if wld else {}
        wins = int(w.get("wins", 0))
        losses = int(w.get("losses", 0))
        draws = int(w.get("draws", 0))
        games = int(w.get("games", wins + losses + draws))
        win_pct = (wins / games * 100) if games else 0.0

        # Family color swatch on the left edge of the row
        ax.add_patch(plt.Rectangle(
            (0.0, y - 0.35), 0.18, 0.7,
            facecolor=color, edgecolor="white", linewidth=0.6, zorder=2,
        ))

        cells = [
            f"#{row_idx + 1}",
            _get_display_name(model),
            family,
            f"{elo:.0f}",
            f"{win_pct:.0f}%",
            f"{wins}W / {losses}L / {draws}D",
            f"{games}",
        ]
        for i, val in enumerate(cells):
            cx = (col_x[i] + col_x[i + 1]) / 2
            # Left-align "Model" so long names are readable
            if cols[i][0] == "Model":
                ax.text(col_x[i] + 0.35, y, val,
                        ha="left", va="center",
                        fontsize=10, color="#222222")
            else:
                ax.text(cx, y, val,
                        ha="center", va="center",
                        fontsize=10, color="#333333")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold",
                     color="#222222", y=0.995)

    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="#FAFAF8")
    plt.close()
    logger.info("Saved vertical leaderboard table to %s", output_path)


def _render_h2h_heatmap(
    matches_dir: Path,
    rows: List[Dict[str, Any]],
    output_path: Path,
    *,
    k_prime: int,
    suppressed_models: set,
) -> None:
    """Win-rate heatmap aggregated by MODEL, restricted to matches where both
    participants are cross-model rank ≤ k_prime (or are baselines).

    Reuses plot_win_rate_heatmap from mjarena.elo.display (the same function
    the per-model Elo pipeline uses). Each cell is row's win rate vs column.
    """
    try:
        from mjarena.elo.display import plot_win_rate_heatmap
    except ImportError:
        logger.warning("mjarena.elo.display unavailable; skipping h2h heatmap")
        return

    # aid → model + cross_model_rank lookup
    aid_to_model: Dict[str, str] = {}
    aid_to_rank: Dict[str, int] = {}
    for r in rows:
        aid_to_model[r["artifact_id"]] = r["model"]
        if r["kind"] == "baseline":
            aid_to_rank[r["artifact_id"]] = 1
        else:
            aid_to_rank[r["artifact_id"]] = int(r.get("rank") or 99)

    # Build match rows pivoted to model-level, filtered by K'.
    rows_for_plot: List[Dict[str, Any]] = []
    for match_dir in matches_dir.iterdir():
        mr = match_dir / "match_result.json"
        if not mr.exists():
            continue
        try:
            data = json.loads(mr.read_text())
        except json.JSONDecodeError:
            continue
        red_aid = data.get("red_bot")
        blue_aid = data.get("blue_bot")
        if red_aid not in aid_to_rank or blue_aid not in aid_to_rank:
            continue
        if aid_to_rank[red_aid] > k_prime or aid_to_rank[blue_aid] > k_prime:
            continue
        red_model = aid_to_model[red_aid]
        blue_model = aid_to_model[blue_aid]
        if red_model in suppressed_models or blue_model in suppressed_models:
            continue
        if red_model == blue_model:
            continue  # intra-model fights aren't useful in a model-level heatmap
        for m in data.get("matches", []):
            winner = m.get("winner", "tie")
            rows_for_plot.append(
                {
                    "red_model": red_model,
                    "blue_model": blue_model,
                    "winner": winner if winner in ("red", "blue") else "tie",
                }
            )

    if not rows_for_plot:
        logger.warning("h2h_top%d: no eligible matches; skipping", k_prime)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_win_rate_heatmap(
        rows_for_plot,
        level="model",
        title=f"Two-Stage H2H Win Rate · top-{k_prime} per model",
        output_path=output_path,
    )


def _render_depth_strip(
    rows: List[Dict[str, Any]],
    full_artifact_standings: Dict[str, Dict[str, Any]],
    output_path: Path,
    *,
    suppressed_models: set,
    anchor_elo: float = 1000.0,
    strip_reasoning_suffix: bool = True,
    display_name_fn=None,
    label_fontsize: float = 25,
    row_height: float = 0.75,
    top_n_models: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    human_elo: Optional[float] = None,
) -> None:
    """Per-model "depth strip": dots for each artifact's cross-model Elo with
    a horizontal range line connecting min↔max. Models stacked top-to-bottom
    by mean Elo (highest at top). Vertical dotted line at anchor (Elo = 1000).
    Family-shaded dots match the leaderboard palette.

    Communicates: peak vs depth (tight cluster = consistent; wide range = peaky).
    """
    try:
        import matplotlib.pyplot as plt
        from mjarena.elo.display import (
            _DEFAULT_BASE,
            _FAMILY_BASE_COLORS,
            _detect_family,
            _get_display_name,
            _setup_style,
        )
    except ImportError:
        logger.warning("matplotlib unavailable; skipping depth strip")
        return

    _setup_style()

    # Group artifacts by model. Skip suppressed models.
    by_model: Dict[str, List[float]] = {}
    for row in rows:
        if row["model"] in suppressed_models:
            continue
        aid = row["artifact_id"]
        elo = full_artifact_standings.get(aid, {}).get("elo")
        if elo is None:
            continue
        by_model.setdefault(row["model"], []).append(float(elo))

    # Sort models by mean Elo descending (best at top).
    sorted_models = sorted(by_model.items(), key=lambda kv: -statistics.mean(kv[1]))
    if top_n_models is not None:
        sorted_models = sorted_models[:top_n_models]

    # Resolve per-model display labels. A caller-supplied display_name_fn
    # (e.g. combined-pool "<base>\nR=<n>" labels) takes precedence; otherwise
    # fall back to the shared _get_display_name, optionally dropping the
    # trailing reasoning-effort abbreviation (" (H)", " (M)", ...).
    def _label(model: str) -> str:
        if display_name_fn is not None:
            return display_name_fn(model)
        name = _get_display_name(model)
        if strip_reasoning_suffix:
            name = re.sub(r"\s*\([A-Z]\)$", "", name)
        return name

    n_rows = len(sorted_models)
    if n_rows == 0:
        logger.warning("no data to plot for depth strip")
        return

    # Figure sizing: ~0.75in per row, min 12in tall × 20in wide so the dots,
    # range lines, and stat labels all read clearly at paper-PDF scale.
    if figsize is None:
        fig_h = max(12.0, n_rows * row_height + 2.0)
        figsize = (20, fig_h)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_elos: List[float] = []
    for _model, elos in sorted_models:
        all_elos.extend(elos)
    elo_min = min(all_elos)
    elo_max = max(all_elos)
    pad = (elo_max - elo_min) * 0.05 if elo_max > elo_min else 50
    x_lo, x_hi = elo_min - pad, elo_max + pad

    # Vertical anchor line at Elo = 1000 — drawn first, behind the dots.
    ax.axvline(
        x=anchor_elo,
        color="#444444",
        linestyle=(0, (3, 3)),
        linewidth=1.0,
        alpha=0.85,
        zorder=1,
    )

    # Human reference line — distinct warm color + longer dash.
    if human_elo is not None:
        ax.axvline(
            x=human_elo,
            color="#B23A48",
            linestyle=(0, (6, 2)),
            linewidth=1.2,
            alpha=0.9,
            zorder=1,
        )

    families_seen: Dict[str, str] = {}  # family → color (for legend)
    for row_idx, (model, elos) in enumerate(sorted_models):
        # Display y from top to bottom (highest Elo on top → row_idx=0 at top)
        y = n_rows - row_idx - 1
        family = _detect_family(model)
        base_rgb = _FAMILY_BASE_COLORS.get(family, _DEFAULT_BASE)
        color = "#%02x%02x%02x" % (
            int(base_rgb[0] * 255), int(base_rgb[1] * 255), int(base_rgb[2] * 255)
        )
        families_seen[family] = color

        # Range line min↔max (skipped for single-artifact models = baselines).
        # Higher alpha so the family color reads at glance.
        if len(elos) > 1:
            ax.plot(
                [min(elos), max(elos)], [y, y],
                color=color, linewidth=3.0, alpha=0.7,
                solid_capstyle="round", zorder=2,
            )

        # Dots for each artifact — bigger, full saturation.
        ax.scatter(
            elos, [y] * len(elos),
            s=200, color=color, edgecolors="white", linewidths=1.8,
            zorder=3,
        )

        # Mean marker — small black tick
        m = statistics.mean(elos)
        ax.plot(
            [m, m], [y - 0.2, y + 0.2],
            color="#222222", linewidth=1.4, alpha=0.7, zorder=4,
        )

        # ABOVE each row: mean Elo + standard deviation, so the row's
        # statistical summary is right where the eye is. Centered on the
        # mean marker, sitting just above the dot/range line. Single-artifact
        # rows (baselines) just show the Elo (σ would be 0 by definition).
        if len(elos) > 1:
            try:
                std = statistics.stdev(elos)
            except statistics.StatisticsError:
                std = 0.0
            stat_label = f"{m:.0f}  (σ {std:.0f})"
        else:
            stat_label = f"{elos[0]:.0f}"
        ax.text(
            m, y + 0.32, stat_label,
            ha="center", va="bottom",
            fontsize=21, color="#111111", fontweight="bold",
            zorder=5,
        )

    # Y-axis: model names BACK on the left (we previously had them on the
    # right; the new above-row stat labels make a separate Elo column
    # redundant, so the right margin is freed and names return to their
    # natural place on the left).
    span = x_hi - x_lo
    ax.set_yticks([n_rows - i - 1 for i in range(n_rows)])
    ax.set_yticklabels(
        [_label(m) for m, _ in sorted_models],
        fontsize=label_fontsize, fontweight="bold", color="#111111",
    )
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    # X-tick labels (the Elo numbers themselves): bumped 1.5x from 15 → 23 and
    # set to bold/near-black so the Elo scale reads clearly across the strip.
    ax.tick_params(axis="x", labelsize=23, colors="#111111")
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")

    # X-axis: Elo, INVERTED so highest is on the left.
    pad_left = span * 0.10
    ax.set_xlim(x_hi + pad_left, x_lo)   # high → low (left → right)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_xlabel(f"Cross-model Elo (single BT pass · Box at {anchor_elo:.0f})",
                  fontsize=22, color="#444444", fontweight="bold")

    # Anchor label near the top of the line — kept just to the right of the
    # anchor in DATA coords. With the x-axis inverted, "anchor + tiny" is
    # visually to the LEFT of the dotted line; flip the offset sign so the
    # label sits to the right of the line as before.
    ax.text(
        anchor_elo - span * 0.005, n_rows - 0.6,
        f"Box Baseline\n(Elo = {anchor_elo:.0f})",
        ha="left", va="top", fontsize=22, color="#444444",
        style="italic", fontweight="bold",
    )
    if human_elo is not None:
        ax.text(
            human_elo - span * 0.005, n_rows - 0.6,
            f"Human\n(Elo = {human_elo:.0f})",
            ha="left", va="top", fontsize=22, color="#B23A48",
            style="italic", fontweight="bold",
        )

    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_title(
        "Top-K Artifact Depth · cross-model Elo per artifact",
        fontsize=22, fontweight="bold", color="#111111", pad=18,
    )

    # Family legend — show colored swatches so the rows can be read by family
    # at a glance. Order families by the order they first appear (= sorted by
    # mean Elo, since rows are sorted that way).
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color, edgecolor="white", linewidth=0.5, label=fam)
        for fam, color in families_seen.items()
    ]
    # Anchor legend below the chart (centered, single row) so it never
    # collides with the right-edge "peak (Δ range)" labels.
    # Match the leaderboard.png legend sizing so the two figures read at the
    # same scale: entries 16pt, 'Model Family' title 18pt.
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        fontsize=16,
        title="Model Family",
        title_fontsize=18,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        handlelength=1.0,
        handleheight=0.9,
        labelspacing=0.4,
        columnspacing=1.4,
        borderpad=0.6,
        ncol=len(legend_handles),
        frameon=False,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved depth strip to %s", output_path)


def _render_grouped_family_leaderboard(
    ratings_by_k: Dict[int, Dict[str, float]],
    output_path: Path,
    *,
    k_values: List[int],
    aggregation: str = "mean",
    sort_descending: bool = True,
    box_baseline_right: bool = False,
    title: Optional[str] = None,
    y_label: str = "Artifact Arena ELO Score",
    box_baseline_elo: Optional[float] = None,
    human_baseline_elo: Optional[float] = None,
    display_name_fn: Optional[Any] = None,
    show_legend: bool = True,
    dpi: int = 200,
    xtick_rotation: float = 0.0,
    strip_reasoning_suffix: bool = False,
    color_override_fn: Optional[Any] = None,
) -> None:
    """Family-shaded grouped bar chart: one model = `len(k_values)` thin bars.

    Same family palette as plot_leaderboard (Anthropic / Google / OpenAI / xAI
    / Other base colors with intra-family shading by K). No legend; each bar
    is labeled "top@{K}\\n<elo>" directly above it. Bars are thin and grouped
    tightly per model; small gap between models within a family; bigger gap
    between families.

    Models (and the families they belong to) are sorted by **top@1 Elo** —
    single-best-bot strength — as the default ranking metric. Family order on
    the chart is implied: each family appears at the position of its
    best-top@1 member, and within a family, members are sorted by top@1 too.

    sort_descending=True (default) puts highest-Elo model on the left;
    sort_descending=False reverses the order (lowest-Elo on the left).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.path import Path as MplPath
        from matplotlib.patches import PathPatch
        from mjarena.elo.display import (
            _DEFAULT_BASE,
            _FAMILY_BASE_COLORS,
            _detect_family,
            _get_display_name,
            _make_shades,
            _setup_style,
            _wrap_name,
        )
    except ImportError:
        logger.warning("matplotlib or mjarena.elo.display unavailable; skipping plot")
        return

    from collections import OrderedDict

    _setup_style()

    # No family grouping: every (model, K) pair is its own bar, and ALL bars
    # are sorted globally by Elo. Sort key is each bar's own Elo; within a
    # single model, top@1 / top@3 / top@5 can sit far apart in the ordering.
    if not any(ratings_by_k.get(k) for k in k_values):
        logger.warning("no ratings to plot")
        return

    # Pre-compute family color shades per model so each bar gets the family
    # palette + the K-specific shade.
    all_models: set = set()
    for k in k_values:
        all_models.update(ratings_by_k.get(k, {}).keys())
    shade_by_model_k: Dict[Tuple[str, int], str] = {}
    color_for_single: Dict[str, str] = {}
    family_color: Dict[str, str] = {}   # family → hex color, for the legend
    family_of_model: Dict[str, str] = {}
    for m in all_models:
        fam = _detect_family(m)
        family_of_model[m] = fam
        base_rgb = _FAMILY_BASE_COLORS.get(fam, _DEFAULT_BASE)
        shades = _make_shades(base_rgb, len(k_values))
        for ki, k in enumerate(k_values):
            shade_by_model_k[(m, k)] = shades[ki]
        hex_color = "#%02x%02x%02x" % (
            int(base_rgb[0] * 255),
            int(base_rgb[1] * 255),
            int(base_rgb[2] * 255),
        )
        color_for_single[m] = hex_color
        family_color.setdefault(fam, hex_color)

    # Optional per-bar color override (e.g. shade-by-condition instead of
    # family hue). Returns a hex color for a model tag, or None to keep the
    # family color. Applied to both the single-bar and per-K shade maps.
    if color_override_fn is not None:
        for m in all_models:
            ov = color_override_fn(m)
            if ov:
                color_for_single[m] = ov
                for k in k_values:
                    shade_by_model_k[(m, k)] = ov

    # Build the flat bar list. Single-artifact models (top@1 == top@3 == top@5
    # by construction) collapse to ONE bar tagged k=-1, so we don't emit three
    # visually identical bars for a baseline.
    bars: List[Tuple[float, str, str, int]] = []  # (elo, color, model, k_or_-1)
    for m in all_models:
        elos = [ratings_by_k[k].get(m, 0.0) for k in k_values]
        is_single = len(set(round(e, 2) for e in elos)) == 1
        if is_single:
            bars.append((elos[0], color_for_single[m], m, -1))
        else:
            for k, elo in zip(k_values, elos):
                bars.append((elo, shade_by_model_k[(m, k)], m, k))

    bars.sort(key=lambda b: (-b[0] if sort_descending else b[0]))

    BAR_WIDTH = 0.42   # slimmer bars
    BAR_GAP = 0.38     # widened so long labels (e.g. 'Qwen3.5 397B (T)') fit
    SINGLE_BAR_W = BAR_WIDTH

    # Place bars sequentially with uniform spacing — no family grouping.
    placements: List[Tuple[float, float, str, str, int]] = []  # (x, elo, color, model, k_or_-1)
    bar_centers: List[float] = []
    bar_labels: List[str] = []

    x = 0.0
    for elo, color, m, k in bars:
        bar_w = SINGLE_BAR_W if k == -1 else BAR_WIDTH
        placements.append((x, elo, color, m, k))
        bar_centers.append(x + bar_w / 2)
        # X-tick label: full display name including reasoning-effort suffix.
        # max_chars=7 forces aggressive wrapping so 'Qwen3.5 397B (T)' breaks
        # onto 3 lines ('Qwen3.5' / '397B' / '(T)') instead of overflowing into
        # the next bar's slot. Shorter labels still wrap cleanly to 2 lines.
        if display_name_fn is not None:
            display = display_name_fn(m)
        else:
            display = _get_display_name(m)
            if strip_reasoning_suffix:
                display = re.sub(r"\s*\([^)]*\)\s*$", "", display).strip()
        # Slanted labels read cleanly on a single line; only wrap to short
        # stacked lines when the labels are horizontal (rotation == 0).
        bar_labels.append(
            display if xtick_rotation else _wrap_name(display, max_chars=7)
        )
        x += bar_w + BAR_GAP

    max_elo = max(p[1] for p in placements) if placements else 1000
    min_elo = min((p[1] for p in placements if p[1] > 0), default=max_elo)
    # Floor the y-axis a bit below the minimum bar so we use vertical space
    # for actual variance rather than empty 0→min territory. Round down to
    # the nearest 50 with ~50-Elo padding.
    y_lo = max(0.0, math.floor((min_elo - 50.0) / 50.0) * 50.0)
    # Paper-PDF-friendly geometry: width-to-bars ratio is ~1.15 in/bar so the
    # two-line "top@K / <elo>" value labels never collide horizontally. With a
    # 3rd condition (Sampling/AutoResearch/Laboratory) a model spans 3 bars, so
    # ~39 bars need a wider canvas — cap raised to 30 in. Fonts are fixed-pt so
    # a wider figure just spreads the bars (and their labels) farther apart.
    n_bars = len(placements)
    fig_w = max(11, min(24, x * 1.15 + 2))
    # Value-label font shrinks as the bar count climbs so adjacent numbers keep
    # clear air between them even at the width cap.
    _vfont = 14 if n_bars <= 20 else (12 if n_bars <= 30 else 10)
    fig, ax = plt.subplots(figsize=(fig_w, 7.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    r_y = max_elo * 0.025
    for xp, elo, color, _model, _k in placements:
        # Single-bar baselines use SINGLE_BAR_W (~2x BAR_WIDTH); K-specific
        # bars use BAR_WIDTH.
        bar_w = SINGLE_BAR_W if _k == -1 else BAR_WIDTH
        rx = bar_w * 0.20
        ry = min(r_y, elo / 2) if elo > 0 else 0
        verts = [
            (xp, 0),
            (xp, elo - ry),
            (xp, elo),
            (xp + rx, elo),
            (xp + bar_w - rx, elo),
            (xp + bar_w, elo),
            (xp + bar_w, elo - ry),
            (xp + bar_w, 0),
            (xp, 0),
        ]
        codes = [
            MplPath.MOVETO, MplPath.LINETO,
            MplPath.CURVE3, MplPath.CURVE3,
            MplPath.LINETO,
            MplPath.CURVE3, MplPath.CURVE3,
            MplPath.LINETO, MplPath.CLOSEPOLY,
        ]
        path = MplPath(verts, codes)
        # Laboratory condition keeps its family color but is overlaid with
        # diagonal hatch lines so it's unmistakable next to Sampling/AutoResearch.
        _lab_hatch = "////" if "__laboratory" in _model else None
        _patch = PathPatch(
            path, facecolor=color,
            edgecolor=("#2b2b2b" if _lab_hatch else "white"),
            linewidth=(0.7 if _lab_hatch else 0.4),
        )
        if _lab_hatch:
            _patch.set_hatch(_lab_hatch)
        ax.add_patch(_patch)

        # Two-line label above each bar: top@K (bigger, prominent) on top,
        # Elo number underneath. Single-artifact baselines just show the Elo.
        if _k == -1:
            ax.text(
                xp + bar_w / 2,
                elo + max_elo * 0.0065,
                f"{elo:.0f}",
                ha="center", va="bottom",
                fontsize=_vfont + 1, color="#222222", fontweight="bold",
            )
        else:
            ax.text(
                xp + bar_w / 2,
                elo + max_elo * 0.0065,
                f"top@{_k}\n{elo:.0f}",
                ha="center", va="bottom",
                fontsize=_vfont, color="#222222",
                fontweight="bold",
                linespacing=1.05,
            )

    # X-axis: one tick per BAR (model + K spec), since family grouping is gone.
    # Horizontal labels centered under each bar. Names are pre-wrapped to
    # max_chars=12 so multi-word display names break onto two short lines
    # rather than running long horizontally.
    ax.set_xticks(bar_centers)
    # fontweight='black' is the heaviest weight matplotlib offers (heavier than
    # 'bold'); combined with a near-black color it makes the model names pop
    # against the gray default xtick color from _setup_style().
    xtick_labels = ax.set_xticklabels(
        bar_labels, fontsize=12, rotation=xtick_rotation,
        ha=("right" if xtick_rotation else "center"),
        rotation_mode="anchor",
        fontweight="black", color="#111111",
    )
    for lbl in xtick_labels:
        lbl.set_fontweight("black")
    ax.tick_params(axis="y", labelsize=12)

    ax.set_xlim(-0.3, x + 0.1)
    ax.set_ylim(y_lo, max_elo + (max_elo - y_lo) * 0.18)
    ax.set_ylabel(
        y_label, fontsize=15, fontweight="bold", color="#444444", labelpad=12
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(8, integer=True))
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Box baseline reference line, drawn at its ACTUAL Elo position. If no
    # explicit position was supplied, auto-detect the Box bar and use its Elo —
    # this keeps the dotted reference line correct under anchor=none (where Box
    # floats with the field rather than sitting at 1000).
    if box_baseline_elo is None:
        for _e, _c, _m, _k in bars:
            if (_m.lower() == "box" or "baseline-static" in _m.lower()
                    or _get_display_name(_m).lower() == "box"):
                box_baseline_elo = _e
                break
    # Auto-detect Box's Elo from its bar if no explicit position was supplied
    # (keeps the dotted reference correct under anchor=none, where Box floats).
    if box_baseline_elo is None:
        for _e, _c, _m, _k in bars:
            if (_m.lower() == "box" or "baseline-static" in _m.lower()
                    or _get_display_name(_m).lower() == "box"):
                box_baseline_elo = _e
                break

    # Reference-line label placement. Bars are sorted descending, and the
    # family legend sits in the upper-right corner — so neither edge is safe
    # for a label near the top of the chart. Instead we park each reference
    # label in the clear band ABOVE the first bar that is SHORTER than the
    # line (by construction every bar there is below the line, and the legend
    # is far to the right). Falls back to the configured edge if no bar dips
    # below the line.
    def _clear_label_x(line_elo: float) -> Tuple[float, str]:
        # When the caller explicitly wants the labels on the right, pin them to
        # the right edge: in a descending leaderboard the bars there are the
        # shortest, so the band above them at the line height is empty and the
        # label never collides with a bar's value number.
        if box_baseline_right:
            return ax.get_xlim()[1] - 0.05, "right"
        # Otherwise left-align at the first sub-threshold bar so the label
        # extends RIGHT over the (even shorter) bars to its right — never over
        # the taller bar to its left, whose value number sits at the line height.
        for (_x, _elo, _c, _m, _k) in placements:
            if _elo < line_elo:
                return _x, "left"
        return ax.get_xlim()[0] + 0.05, "left"

    if box_baseline_elo is not None:
        ax.axhline(
            y=box_baseline_elo,
            color="#444444",
            linestyle=(0, (3, 3)),
            linewidth=1.1,
            alpha=0.85,
            zorder=10,
        )
        # Box label lives in the legend (top-right), not as on-plot text.

    # Human reference line — the baseline is the hand-written pusher bot, the
    # "human-engineered" reference. Distinct warm color + longer dash.
    if human_baseline_elo is not None:
        ax.axhline(
            y=human_baseline_elo,
            color="#B23A48",
            linestyle=(0, (6, 2)),
            linewidth=1.3,
            alpha=0.9,
            zorder=10,
        )
        # Pusher label lives in the legend (top-right), not as on-plot text.

    if title is None:
        ks_str = " / ".join(f"top-{k}" for k in k_values)
        title = (
            f"Two-Stage Model Elo · {ks_str} artifacts per model "
            f"(BT · {aggregation})"
        )
    ax.set_title(
        title,
        pad=14,
        fontsize=18,
        fontweight="bold",
        color="#111111",
    )

    # Family legend in the top-right corner — colored swatches for every
    # family that has at least one bar in the chart. Order families by their
    # best member's Elo (descending) so the legend reads top-down by strength.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    families_in_chart = list(dict.fromkeys(
        family_of_model[m] for _elo, _c, m, _k in bars
    ))
    family_best_elo = {fam: float("-inf") for fam in families_in_chart}
    for elo, _c, m, _k in bars:
        fam = family_of_model[m]
        if elo > family_best_elo[fam]:
            family_best_elo[fam] = elo
    families_ordered = sorted(families_in_chart, key=lambda f: -family_best_elo[f])
    legend_handles = [
        Patch(facecolor=family_color[f], edgecolor="white", linewidth=0.6, label=f)
        for f in families_ordered
    ]
    # Condition key: the bars encode condition by SHADE/HATCH (Sampling =
    # lighter tint, AutoResearch = full color, Laboratory = full + diagonal
    # hatch); the family hues live in the separate model_legend.png. Show a
    # neutral-gray key for every condition present so all three are explained
    # inline, not just Laboratory.
    _COND_LABELS = {"apionly": "Sampling", "r50": "AutoResearch",
                    "laboratory": "Laboratory"}
    conds_present = []
    for _e, _c, m, _k in bars:
        suf = m.split("__")[-1] if "__" in m else None
        if suf in _COND_LABELS and suf not in conds_present:
            conds_present.append(suf)
    cond_handles = []
    if len(conds_present) > 1 or "laboratory" in conds_present:
        _SLATE_FULL, _SLATE_LIGHT = "#8d8d8d", "#d2d2d2"
        for c in [c for c in ("apionly", "r50", "laboratory") if c in conds_present]:
            if c == "apionly":
                cond_handles.append(Patch(facecolor=_SLATE_LIGHT, edgecolor="white",
                                          linewidth=0.6, label="Sampling (lighter)"))
            elif c == "r50":
                cond_handles.append(Patch(facecolor=_SLATE_FULL, edgecolor="white",
                                          linewidth=0.6, label="AutoResearch (full)"))
            else:
                cond_handles.append(Patch(facecolor=_SLATE_FULL, edgecolor="#2b2b2b",
                                          linewidth=0.6, hatch="////",
                                          label="Laboratory (hatched)"))
    baseline_handles = []
    if box_baseline_elo is not None:
        baseline_handles.append(Line2D(
            [0], [0], color="#444444", linestyle=(0, (3, 3)), linewidth=1.1,
            label=f"Baseline (Box) — {box_baseline_elo:.0f}",
        ))
    if human_baseline_elo is not None:
        baseline_handles.append(Line2D(
            [0], [0], color="#B23A48", linestyle=(0, (6, 2)), linewidth=1.3,
            label=f"Baseline (Pusher) — {human_baseline_elo:.0f}",
        ))
    # When the family legend is on, append the baselines to it (2 columns keeps
    # the box compact). When it's off (e.g. the combined pool, whose family +
    # condition legend is a separate PNG), still show a compact baselines-only
    # legend so the reference lines are labeled — top-right, stacked vertically.
    if show_legend:
        ax.legend(
            handles=legend_handles + cond_handles + baseline_handles,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            ncol=2,
            fontsize=16,
            framealpha=0.95,
            edgecolor="#CCCCCC",
            facecolor="white",
            title="Model Family",
            title_fontsize=18,
            handlelength=1.4,
            handleheight=1.2,
            labelspacing=0.45,
            columnspacing=1.3,
            borderpad=0.7,
        )
    elif cond_handles or baseline_handles:
        ax.legend(
            handles=cond_handles + baseline_handles,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            ncol=1,
            fontsize=14,
            framealpha=0.95,
            edgecolor="#CCCCCC",
            facecolor="white",
            title="Condition & Baselines",
            title_fontsize=15,
            handlelength=1.8,
            labelspacing=0.5,
            borderpad=0.7,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved grouped family leaderboard to %s", output_path)


def run_stage4(
    cross_model_dir: Path,
    *,
    k_values: Optional[List[int]] = None,
    output_plot: Optional[Path] = None,
    app_elo_dir: Optional[Path] = None,
    elo_k_factor: float = 32.0,
    elo_initial_rating: float = 1000.0,
    aggregations: Tuple[str, ...] = ("mean",),
    render_per_k: bool = False,
    single_bt_pass: bool = True,
    anchor: Optional[str] = "none",
    anchor_elo: float = 1000.0,
    ignore: Optional[List[str]] = None,
    sort_descending: bool = True,
    box_baseline_right: bool = False,
    title: Optional[str] = None,
    y_label: str = "Artifact Arena ELO Score",
) -> Dict[str, Any]:
    """Stage 4 entry point.

    Args:
        cross_model_dir: e.g. `two_stage_tournaments/cross_model/top_5_matches`
        k_values: list of K' to evaluate (defaults to [1, 3, k_max] from pool).
        output_plot: optional path for the grouped bar chart.
        aggregations: which model-Elo aggregation modes to write.
    """
    pool_path = next(cross_model_dir.glob("top_*_bots.json"), None)
    if pool_path is None:
        raise FileNotFoundError(f"no top_*_bots.json in {cross_model_dir}")
    payload = read_top_k_bots(pool_path)
    rows = payload["bots"]
    pool_k_max = int(payload["k"])

    if k_values is None:
        # Default scheme: top@1, top@3, top@5 (clipped to whatever the pool
        # supports — e.g. if pool_k_max=3 we drop the 5). Override via
        # --k-values for a different scheme like 1 5 10.
        k_values = sorted({k for k in (1, 3, 5) if k <= pool_k_max})
        if not k_values:
            k_values = [pool_k_max]

    matches_dir = cross_model_dir / "matches"
    outcomes = _read_match_outcomes(matches_dir)
    logger.info(
        "Stage 4: %d outcomes loaded from %s", len(outcomes), matches_dir
    )

    # Drop stationary-vs-stationary baseline matches (Box vs polyhedra,
    # polyhedra vs polyhedra). These run to inactivity timeout and are
    # always draws — zero information for BT, just inflates game counts.
    stationary_aids = {
        r["artifact_id"]
        for r in rows
        if r["kind"] == "baseline" and r["model"].lower() in _STATIONARY_BASELINES
    }
    if stationary_aids:
        before = len(outcomes)
        outcomes = [
            o for o in outcomes
            if not (o.player_a in stationary_aids and o.player_b in stationary_aids)
        ]
        dropped = before - len(outcomes)
        if dropped:
            logger.info(
                "Stage 4: dropped %d stationary-vs-stationary baseline matches "
                "(all draws, no BT signal)",
                dropped,
            )

    # Build the set of artifact_ids to suppress from leaderboard outputs.
    # BT MLE still includes their matches (so calibration / anchor are
    # consistent), but they're dropped before writing JSON or plotting.
    ignore_terms = [t.lower() for t in (ignore or [])]
    suppressed_aids: set = set()
    suppressed_models: set = set()
    if ignore_terms:
        for row in rows:
            aid_low = row["artifact_id"].lower()
            model_low = row["model"].lower()
            if any(t in aid_low or t in model_low for t in ignore_terms):
                suppressed_aids.add(row["artifact_id"])
                suppressed_models.add(row["model"])
        if suppressed_models:
            logger.info(
                "Stage 4: --ignore suppressing models from outputs: %s",
                sorted(suppressed_models),
            )

    # Only draw the "Box Baseline" reference line when a baseline is actually
    # pinned (anchor != none); under anchor=none Box floats with the field, so
    # a line at a fixed Elo would be misleading.
    _box_baseline_line_elo: Optional[float] = (
        anchor_elo if (anchor and str(anchor).lower() != "none") else None
    )

    # Single BT pass on the full T_max field, anchored once. Every K' aggregation
    # downstream filters from these frozen Elos.
    full_artifact_standings: Optional[Dict[str, Dict[str, Any]]] = None
    if single_bt_pass:
        logger.info("Stage 4: running single BT pass over %d outcomes", len(outcomes))
        full_artifact_standings = _compute_bt_standings(outcomes)
        full_artifact_standings = _apply_anchor(
            full_artifact_standings, anchor=anchor, anchor_elo=anchor_elo
        )

        # Re-rank each model's artifacts by their cross-model BT Elo (not the
        # intra-model RR rank that Stage 2 produced). The intra-model rank was
        # only a candidate-selection heuristic; once we have cross-model data,
        # "top-K" means "K best by cross-model Elo." This makes mean strictly
        # monotone non-increasing in K and lines up with the actual headline
        # metric. The original intra-RR rank stays in the on-disk pool file
        # for traceability.
        by_model: Dict[str, List[Tuple[str, float]]] = {}
        for row in rows:
            if row["kind"] == "baseline":
                continue
            aid = row["artifact_id"]
            elo = full_artifact_standings.get(aid, {}).get("elo", 0.0)
            by_model.setdefault(row["model"], []).append((aid, elo))
        cross_model_ranks: Dict[str, int] = {}
        for model, aid_elos in by_model.items():
            aid_elos.sort(key=lambda x: -x[1])
            for new_rank, (aid, _elo) in enumerate(aid_elos, start=1):
                cross_model_ranks[aid] = new_rank
        n_reranked = 0
        for row in rows:
            if row["kind"] == "baseline":
                continue
            new_rank = cross_model_ranks.get(row["artifact_id"])
            if new_rank is not None and row.get("rank") != new_rank:
                row["intra_model_rank"] = row.get("rank")
                row["rank"] = new_rank
                n_reranked += 1
        if n_reranked:
            logger.info(
                "Stage 4: re-ranked %d artifacts by cross-model Elo "
                "(intra-RR rank preserved in 'intra_model_rank')",
                n_reranked,
            )

    summary: Dict[str, Any] = {
        "k_values": k_values,
        "aggregations": list(aggregations),
        "single_bt_pass": single_bt_pass,
        "anchor": anchor,
        "anchor_elo": anchor_elo,
        "per_k": {},
    }

    # Track per-K artifact standings so we can back-fill the pool file later.
    per_k_artifact_standings: Dict[int, Dict[str, Dict[str, Any]]] = {}

    for k_prime in k_values:
        allowed = _allowed_artifact_ids(rows, k_prime)
        filtered = _filter_outcomes_for_subset(outcomes, allowed)
        if single_bt_pass and full_artifact_standings is not None:
            # Pull frozen Elos for the eligible artifacts. W/L/D from the
            # filtered subset (since which matches count depends on K').
            wld_only = compute_standings(filtered)
            artifact_standings = {}
            for aid in allowed:
                if aid not in full_artifact_standings:
                    continue
                # Use frozen Elo, K-filtered W/L/D
                wld = wld_only.get(aid, {"wins": 0, "losses": 0, "draws": 0, "games": 0})
                artifact_standings[aid] = {
                    "elo": full_artifact_standings[aid]["elo"],
                    "wins": wld["wins"],
                    "losses": wld["losses"],
                    "draws": wld["draws"],
                    "games": wld["games"],
                }
        else:
            artifact_standings = _compute_bt_standings(filtered)

        per_k_artifact_standings[k_prime] = artifact_standings
        section: Dict[str, Any] = {"n_outcomes": len(filtered), "n_artifacts": len(allowed)}

        if "mean" in aggregations:
            mean_agg = _aggregate_model_elo_mean(rows, artifact_standings, k_prime)
            section["mean"] = mean_agg

        if "max" in aggregations:
            max_agg = _aggregate_model_elo_max(rows, artifact_standings, k_prime)
            section["max"] = max_agg

        if "pooled" in aggregations:
            pooled = _aggregate_model_elo_pooled(
                rows,
                outcomes,
                k_prime,
                elo_k_factor=elo_k_factor,
                elo_initial_rating=elo_initial_rating,
            )
            section["pooled"] = pooled

        out_path = cross_model_dir / f"elo_model_top{k_prime}.json"
        out_path.write_text(json.dumps(section, indent=2))
        logger.info("Stage 4: wrote %s", out_path)

        summary["per_k"][f"k{k_prime}"] = section

    if output_plot is not None:
        _render_leaderboard(rows, summary, output_plot, k_values=k_values)

    # Back-fill the pool file (top_K_bots.json) and elo_artifact.json with
    # the up-to-date single-pass anchored Elos so on-disk JSONs stay in sync
    # with the leaderboard charts. Stage 3 wrote stale per-K BT values; we
    # overwrite them here with the single-pass numbers + cross-model rank.
    if single_bt_pass and full_artifact_standings is not None:
        # Refresh elo_artifact.json (artifact-level full-pool standings).
        full_for_disk: Dict[str, Dict[str, Any]] = {}
        for aid, stats in full_artifact_standings.items():
            full_for_disk[aid] = {
                "elo": float(stats["elo"]),
                "wins": int(stats["wins"]),
                "losses": int(stats["losses"]),
                "draws": int(stats["draws"]),
                "games": int(stats["games"]),
            }
        (cross_model_dir / "elo_artifact.json").write_text(
            json.dumps(
                {
                    "method": "bt",
                    "single_bt_pass": True,
                    "anchor": anchor,
                    "anchor_elo": anchor_elo,
                    "n_outcomes": len(outcomes),
                    "standings": full_for_disk,
                },
                indent=2,
            )
        )
        logger.info(
            "Stage 4: refreshed %s with single-pass + anchor=%s",
            cross_model_dir / "elo_artifact.json", anchor,
        )

        # Back-fill the pool file's per-row cross_model_elo / wld / rank.
        pool_payload = read_top_k_bots(pool_path)
        for row in pool_payload["bots"]:
            aid = row["artifact_id"]
            full = full_artifact_standings.get(aid)
            # cross_model_rank: 1 for baselines (single artifact), else from re-rank.
            if row["kind"] == "baseline":
                row["cross_model_rank"] = 1
            else:
                row["cross_model_rank"] = cross_model_ranks.get(aid)
                # Preserve the original intra-model rank for traceability.
                if "intra_model_rank" not in row:
                    row["intra_model_rank"] = row.get("rank")

            row["cross_model_elo"] = {}
            row["cross_model_wld"] = {}
            for k in k_values:
                key = f"k{k}"
                stat = per_k_artifact_standings.get(k, {}).get(aid)
                if stat is None or full is None:
                    row["cross_model_elo"][key] = None
                    row["cross_model_wld"][key] = None
                else:
                    # Single-pass: Elo is the frozen full-pool value.
                    row["cross_model_elo"][key] = float(full["elo"])
                    row["cross_model_wld"][key] = {
                        "wins": int(stat["wins"]),
                        "losses": int(stat["losses"]),
                        "draws": int(stat["draws"]),
                    }

        pool_payload.setdefault("source", {})
        pool_payload["source"]["single_bt_pass"] = True
        pool_payload["source"]["anchor"] = anchor
        pool_payload["source"]["anchor_elo"] = anchor_elo
        pool_payload["source"]["ignored"] = sorted(suppressed_models)
        pool_payload["generated_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        pool_path.write_text(json.dumps(pool_payload, indent=2))
        logger.info(
            "Stage 4: refreshed %s (anchored, cross-model rank, %d ignored)",
            pool_path, len(suppressed_models),
        )

    # App-compatible elo_results_topK.json + family-shaded per-K leaderboard.
    # Lives alongside the existing per-model elo_results.json so the app's
    # /api/leaderboard route can serve it via ?top_k=N.
    if app_elo_dir is None:
        # Default: <two_stage_root>/elo/  (e.g. .../two_stage_tournaments/elo/)
        app_elo_dir = cross_model_dir.parent.parent / "elo"
    app_elo_dir.mkdir(parents=True, exist_ok=True)

    # Pick which aggregations to write to the app-elo dir. "pooled" doesn't
    # round-trip into the {ratings, wld} schema cleanly because it relabels
    # players to model names mid-flight, so we keep it out of the app files
    # and only emit mean/max (and any other simple-aggregation modes).
    app_aggs = [a for a in aggregations if a in _AGGREGATION_FNS]
    if not app_aggs:
        app_aggs = ["mean"]
    primary_agg = app_aggs[0]  # "mean" by default; first in `aggregations` wins

    def _suppress(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Drop ignored models from the {ratings, wld} dicts in place."""
        if not suppressed_models:
            return payload
        payload["ratings"] = {
            m: v for m, v in payload["ratings"].items() if m not in suppressed_models
        }
        payload["wld"] = {
            m: v for m, v in payload["wld"].items() if m not in suppressed_models
        }
        payload["metadata"]["ignored_models"] = sorted(suppressed_models)
        payload["metadata"]["n_models_or_baselines"] = len(payload["ratings"])
        return payload

    # Always write the JSONs (cheap, app needs them for the dropdown).
    # Only write per-K family-shaded PNGs when `render_per_k=True` — the
    # default plot output is the two grouped charts (one per aggregation).
    for agg in app_aggs:
        for k_prime in k_values:
            app_payload = _suppress(_build_app_elo_results(
                rows,
                outcomes,
                k_prime,
                elo_k_factor=elo_k_factor,
                elo_initial_rating=elo_initial_rating,
                aggregation=agg,
                full_artifact_standings=full_artifact_standings,
            ))
            # File name for the primary aggregation stays as the canonical
            # `elo_results_topK.json`; other aggregations get a suffix
            # (`elo_results_topK_max.json`, etc.) so the app can pick.
            suffix = "" if agg == primary_agg else f"_{agg}"
            out_json = app_elo_dir / f"elo_results_top{k_prime}{suffix}.json"
            out_json.write_text(json.dumps(app_payload, indent=2))
            logger.info(
                "Stage 4: wrote %s (%d entries, agg=%s)",
                out_json, len(app_payload["ratings"]), agg,
            )

            if render_per_k:
                leaderboard_png = app_elo_dir / f"leaderboard_top{k_prime}{suffix}.png"
                # Per-K plot: same flat-descending style as the main grouped
                # plot, but with a single-K dict so each model gets exactly
                # one bar (no top@K split — the whole chart is for one K).
                _render_grouped_family_leaderboard(
                    {k_prime: app_payload["ratings"]},
                    leaderboard_png,
                    k_values=[k_prime],
                    aggregation=agg,
                    sort_descending=sort_descending,
                    box_baseline_right=box_baseline_right,
                    title=title,
                    y_label=y_label,
                    box_baseline_elo=_box_baseline_line_elo,
                )

    # Default `elo_results.json` (no suffix) points at top-K_max with the
    # primary aggregation, so legacy consumers get the most permissive view.
    largest_k = max(k_values)
    default_payload = _suppress(_build_app_elo_results(
        rows, outcomes, largest_k,
        elo_k_factor=elo_k_factor, elo_initial_rating=elo_initial_rating,
        aggregation=primary_agg,
        full_artifact_standings=full_artifact_standings,
    ))
    (app_elo_dir / "elo_results.json").write_text(json.dumps(default_payload, indent=2))

    # Main leaderboard: one grouped 3-bar chart for the PRIMARY aggregation
    # only. Under cross-model ranking, `max` is the same value for every K
    # (the model's cross-model peak doesn't change as K grows), so the max
    # plot would be visually degenerate. `mean` is the canonical depth view.
    # If you want a max plot too, list `max` first in --aggregations and
    # it'll become the primary.
    ratings_by_k: Dict[int, Dict[str, float]] = {}
    for k_prime in k_values:
        payload = _suppress(_build_app_elo_results(
            rows, outcomes, k_prime,
            elo_k_factor=elo_k_factor, elo_initial_rating=elo_initial_rating,
            aggregation=primary_agg,
            full_artifact_standings=full_artifact_standings,
        ))
        ratings_by_k[k_prime] = payload["ratings"]
    _render_grouped_family_leaderboard(
        ratings_by_k,
        app_elo_dir / f"leaderboard_{primary_agg}.png",
        k_values=k_values,
        aggregation=primary_agg,
        sort_descending=sort_descending,
        box_baseline_right=box_baseline_right,
        title=title,
        y_label=y_label,
        box_baseline_elo=_box_baseline_line_elo,
    )

    # Compact leaderboard table — Elo / Win % / W-L-D / Games per model.
    # Models on the Y-axis (one row each) so it scans like a normal stats
    # table. Built from the top-K_max default payload so totals reflect all
    # matches feeding the headline ratings.
    _render_leaderboard_table_vertical(
        default_payload["ratings"],
        default_payload["wld"],
        app_elo_dir / "leaderboard_table.png",
        title=f"Two-Stage Leaderboard · top-{largest_k} {primary_agg}",
        method="BT",
    )

    # Depth strip plot — one row per model, dots for each artifact's
    # cross-model Elo + range line min↔max. Reveals depth vs. peakiness
    # that the bar chart hides.
    if single_bt_pass and full_artifact_standings is not None:
        _render_depth_strip(
            rows,
            full_artifact_standings,
            app_elo_dir / "depth_strip.png",
            suppressed_models=suppressed_models,
            anchor_elo=anchor_elo,
        )

    # Per-K H2H model-level win-rate heatmap. Same color palette + style as
    # the existing mjarena/elo/display.py heatmap so it matches the rest of
    # the dashboard. One PNG per K' in k_values.
    for k_prime in k_values:
        _render_h2h_heatmap(
            matches_dir,
            rows,
            app_elo_dir / f"h2h_top{k_prime}.png",
            k_prime=k_prime,
            suppressed_models=suppressed_models,
        )

    return summary


def _render_leaderboard(
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_plot: Path,
    *,
    k_values: List[int],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available; skipping plot")
        return

    baseline_models = sorted(
        {row["model"] for row in rows if row["kind"] == "baseline"}
    )
    model_models = sorted(
        {row["model"] for row in rows if row["kind"] != "baseline"}
    )

    largest_k = max(k_values)
    largest_section = summary["per_k"][f"k{largest_k}"].get("mean", {})
    model_models.sort(
        key=lambda m: largest_section.get(m, {}).get("elo", 0.0),
        reverse=True,
    )

    all_models = model_models + baseline_models
    n_models = len(all_models)
    if n_models == 0:
        return

    fig_width = max(8.0, 0.55 * n_models + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))

    bar_width = 0.8 / len(k_values)
    x = np.arange(n_models)

    for i, k in enumerate(k_values):
        section = summary["per_k"][f"k{k}"].get("mean", {})
        elos = [section.get(m, {}).get("elo", math.nan) for m in all_models]
        offset = (i - (len(k_values) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            elos,
            width=bar_width,
            label=f"top-{k}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, rotation=45, ha="right")
    ax.set_ylabel("Model Elo (mean of artifact Elos)")
    ax.set_title("Two-Stage Model Elo: top-K artifacts per model")
    ax.axvline(len(model_models) - 0.5, color="grey", linestyle=":", linewidth=1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=120)
    plt.close(fig)
    logger.info("Stage 4: rendered %s", output_plot)
