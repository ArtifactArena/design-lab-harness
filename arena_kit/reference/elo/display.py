"""
Visualization for Elo ratings.

Generates individual publication-quality figures: leaderboard bars, win-rate
heatmap, Elo progression over iterations.  Earthy muted color palette inspired
by Anthropic benchmark charts.

Saves each plot as a separate file inside ``{season_dir}/elo/``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Display name mapping
# ---------------------------------------------------------------------------

# Reasoning effort suffixes → display label.
# Appended to the base display name when not "none".
_REASONING_SUFFIXES = {
    "-none": "",
    "-low": " (L)",
    "-medium": " (M)",
    "-high": " (H)",
}

# Base model stem (without reasoning suffix) → pretty display name.
_BASE_DISPLAY_NAMES: Dict[str, str] = {
    # OpenAI
    "gpt-5": "GPT-5",
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5-nano": "GPT-5 Nano",
    "gpt-5-pro": "GPT-5 Pro",
    "gpt-5.1": "GPT-5.1",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.2-pro": "GPT-5.2 Pro",
    "gpt-5.3": "GPT-5.3",
    "gpt-5-codex": "GPT-5 Codex",
    "gpt-4o": "GPT-4o",
    "gpt4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4-turbo": "GPT-4 Turbo",
    "gpt-4.1": "GPT-4.1",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gpt-4.1-nano": "GPT-4.1 Nano",
    "openai-mini": "OpenAI Mini",
    "o1": "o1",
    "o1-pro": "o1 Pro",
    "o3": "o3",
    "o3-mini": "o3 Mini",
    "o3-pro": "o3 Pro",
    "o4-mini": "o4 Mini",
    # Anthropic
    "claude-sonnet-4": "Sonnet 4",
    "claude-3-5-sonnet": "Sonnet 3.5",
    "claude-3-opus": "Opus 3",
    "claude-opus-4-6": "Opus 4.6",
    "claude-opus-4-5": "Opus 4.5",
    "claude-sonnet-4-5": "Sonnet 4.5",
    "claude-haiku-4-5": "Haiku 4.5",
    "claude-opus-4-6": "Opus 4.6",
    "claude-sonnet-4-6": "Sonnet 4.6",
    # Google
    "gemini-2-5-pro": "Gemini 2.5 Pro",
    "gemini-3-flash": "Gemini 3 Flash",
    "gemini-3-pro": "Gemini 3 Pro",
    "gemini-3-1-pro": "Gemini 3 Pro",
    "gemini-3-1-flash": "Gemini 3 Flash",
    "gemini-2-0-flash": "Gemini 2 Flash",
    "gemini-1-5-pro": "Gemini 1.5 Pro",
    "gemini-2-5-flash": "Gemini 2.5 Flash",
    "gemma-4-31b": "Gemma 4 31B",
    "gemma-4-26b-a4b": "Gemma 4 26B A4B",
    # Meta
    "llama-3-3-70b": "Llama 3.3 70B",
    "llama-3-1-405b": "Llama 3.1 405B",
    # DeepSeek (full names — wrap to 2 lines on bar charts via _wrap_name)
    "deepseek-reasoner": "DeepSeek R1",
    "deepseek-chat": "DeepSeek Chat",
    "deepseek-r1": "DeepSeek R1",
    "deepseek-v3-1-thinking": "DeepSeek V3.1",
    # xAI
    "grok-2": "Grok 2",
    "grok-2-mini": "Grok 2 Mini",
    "grok-3": "Grok 3",
    "grok-4-1": "Grok 4.1",
    "grok-4-1-fast-reasoning": "Grok 4.1 Fast",
    "grok-4.20-reasoning": "Grok 4.20",
    "grok-code-fast-1": "Grok CodeFast",
    # OpenAI extras
    "gpt-5.4": "GPT-5.4",
    "gpt-5.3-codex": "GPT-5.3 Codex",
    # Moonshot
    "kimi-k2.5-thinking": "Kimi K2.5",
    # Qwen (full names — wrap to 2 lines on bar charts via _wrap_name)
    # Keep parameter counts visible (Coder: 480B; Thinking: 397B (T)).
    "qwen3-coder-480b": "Qwen3 Coder 480B",
    "qwen3.5-397b-thinking": "Qwen3.5 397B (T)",
    # Baselines (no provider suffix)
    "tetrahedron": "Tetra",
    "octahedron": "Octa",
    "icosahedron": "Icosa",
    "dodecahedron": "Dodec",
    "hexahedron": "Box",
    "box": "Box",
    "pusher": "Human",
}


def _get_display_name(raw_name: str) -> str:
    """Convert a raw bot/model name to a clean display name.

    Handles full bot names (``season_00.tournament.gpt-5-medium``) and
    bare generator stems (``gpt-5-medium``).

    Reasoning effort suffixes (``-none``, ``-low``, ``-medium``, ``-high``)
    are detected and appended as abbreviations so duplicate base models
    are distinguishable — e.g. ``gpt-5.1-medium`` → ``"GPT-5.1 (M)"``.
    """
    # Extract generator stem from full bot name
    parts = raw_name.split(".")
    stem = ".".join(parts[2:]) if len(parts) >= 3 else raw_name

    # Detect reasoning effort suffix
    suffix_label = ""
    base_stem = stem
    for suffix, label in _REASONING_SUFFIXES.items():
        if stem.endswith(suffix):
            suffix_label = label
            base_stem = stem[: -len(suffix)]
            break

    # Try base stem (e.g. "gpt-5.1" → "GPT-5.1", then append "(M)")
    if base_stem in _BASE_DISPLAY_NAMES:
        return _BASE_DISPLAY_NAMES[base_stem] + suffix_label

    # Try full stem for names without reasoning suffix (e.g. "grok-2")
    if stem in _BASE_DISPLAY_NAMES:
        return _BASE_DISPLAY_NAMES[stem]

    # Fallback: capitalize and clean up
    display = base_stem.replace("-", " ").replace("_", " ").title()
    return display + suffix_label


# ---------------------------------------------------------------------------
# Family detection & color palette
# ---------------------------------------------------------------------------

_FAMILY_PATTERNS = [
    ("gpt", "OpenAI"),
    ("codex", "OpenAI"),
    ("o1", "OpenAI"),
    ("o3", "OpenAI"),
    ("o4", "OpenAI"),
    ("openai", "OpenAI"),
    ("claude", "Anthropic"),
    ("sonnet", "Anthropic"),
    ("opus", "Anthropic"),
    ("haiku", "Anthropic"),
    ("gemini", "Google"),
    ("gemma", "Google"),
    ("llama", "Meta"),
    ("mistral", "Mistral"),
    ("deepseek", "DeepSeek"),
    ("grok", "xAI"),
    ("qwen", "Qwen"),
    ("kimi", "Moonshot"),
    # Hand-coded algorithmic baselines: platonic solids + 'box'.
    ("tetrahedron", "Platonic Solids"),
    ("hexahedron", "Platonic Solids"),
    ("octahedron", "Platonic Solids"),
    ("dodecahedron", "Platonic Solids"),
    ("icosahedron", "Platonic Solids"),
    ("box", "Platonic Solids"),
    # Compact display names used by the leaderboard model field.
    ("tetra", "Platonic Solids"),
    ("octa", "Platonic Solids"),
    ("icosa", "Platonic Solids"),
    ("dodec", "Platonic Solids"),
    # Human-coded reference bot (kept separate from algorithmic baselines).
    ("pusher", "Human"),
]

# Base color per family. Orange = Anthropic, Teal = OpenAI (per user request).
_FAMILY_BASE_COLORS: Dict[str, Tuple[float, float, float]] = {
    "Anthropic": (0.816, 0.482, 0.353),   # #D07B5A coral orange
    "OpenAI":    (0.353, 0.678, 0.541),    # #5AAD8A seafoam green
    "Google":    (0.243, 0.494, 0.690),    # #3E7EB0 steel blue
    "Meta":      (0.773, 0.722, 0.596),    # #C5B898 warm beige
    "Mistral":   (0.769, 0.545, 0.624),    # #C48B9F dusty rose
    "DeepSeek":  (0.486, 0.651, 0.651),    # #7CA6A6 teal sage
    "xAI":       (0.608, 0.557, 0.769),    # #9B8EC4 muted lavender
    "Qwen":      (0.831, 0.655, 0.416),    # #D4A76A golden tan
    "Moonshot":  (0.341, 0.349, 0.490),    # #57597D dusk indigo (Kimi)
    "Platonic Solids":
                 (0.784, 0.769, 0.722),    # #C8C4B8 warm gray (algorithmic baselines)
    "Human":     (0.651, 0.353, 0.286),    # #A65A49 rusty brick (human-coded)
}
_DEFAULT_BASE = (0.784, 0.769, 0.722)      # #C8C4B8 warm gray

# Fallback ordered palette for progression chart / non-grouped contexts
_PALETTE = [
    "#D07B5A", "#5AAD8A", "#3E7EB0", "#C5B898", "#C8C4B8",
    "#C48B9F", "#7CA6A6", "#9B8EC4", "#D4A76A", "#8FBC8F",
]


def _detect_family(name: str) -> str:
    """Detect model family from a raw or display name."""
    lower = name.lower()
    for pattern, family in _FAMILY_PATTERNS:
        if pattern in lower:
            return family
    return "Other"


def _make_shades(base_rgb: Tuple[float, float, float], n: int) -> List[str]:
    """Generate *n* shades from a base RGB color.

    For 1 model: return the base color.
    For 2+: spread from slightly lighter to slightly darker.
    """
    if n <= 0:
        return []
    if n == 1:
        r, g, b = base_rgb
        return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"]

    shades = []
    for i in range(n):
        # t goes from -0.15 (lighter) to +0.15 (darker)
        t = -0.15 + 0.3 * i / (n - 1)
        r = max(0.0, min(1.0, base_rgb[0] - t))
        g = max(0.0, min(1.0, base_rgb[1] - t))
        b = max(0.0, min(1.0, base_rgb[2] - t))
        shades.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return shades


def _assign_family_colors(raw_names: List[str]) -> Dict[str, str]:
    """Assign colors based on family, with shades for models within a family.

    Returns a dict mapping raw_name → hex color.
    """
    # Group names by family (preserving order within each family)
    from collections import OrderedDict
    family_members: Dict[str, List[str]] = OrderedDict()
    for name in raw_names:
        fam = _detect_family(name)
        family_members.setdefault(fam, []).append(name)

    mapping: Dict[str, str] = {}
    for fam, members in family_members.items():
        base = _FAMILY_BASE_COLORS.get(fam, _DEFAULT_BASE)
        shades = _make_shades(base, len(members))
        for name, shade in zip(members, shades):
            mapping[name] = shade
    return mapping


def _get_color(name: str) -> str:
    """Return a hex color for a single model name based on its family."""
    family = _detect_family(name)
    base = _FAMILY_BASE_COLORS.get(family, _DEFAULT_BASE)
    r, g, b = base
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _assign_colors(names: list) -> Dict[str, str]:
    """Assign palette colors to a list of player names (in order).

    Returns a dict mapping name → hex color.  Uses family-based shading.
    """
    return _assign_family_colors(names)


# ---------------------------------------------------------------------------
# Shared style setup
# ---------------------------------------------------------------------------


def _setup_style():
    """Apply a clean publication-quality matplotlib style."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Use a clean base
    plt.style.use("seaborn-v0_8-whitegrid")

    # Override for our aesthetic
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelcolor": "#333333",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "grid.alpha": 0.25,
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.5,
        "figure.facecolor": "#FAFAF8",
        "axes.facecolor": "#FAFAF8",
        "savefig.facecolor": "#FAFAF8",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.titlesize": 20,
        "figure.titleweight": "bold",
    })


# ---------------------------------------------------------------------------
# Text wrapping & smart label placement
# ---------------------------------------------------------------------------


def _wrap_name(name: str, max_chars: int = 14) -> str:
    """Wrap a display name into multiple lines at word boundaries."""
    if len(name) <= max_chars:
        return name
    words = name.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return "\n".join(lines)


def _smart_annotate(ax, raw_names, xs, ys, *, fontsize=9, max_chars=14):
    """Annotate scatter points with labels pushed away from nearby dots.

    For each point, computes a repulsion vector from all other points and
    places the label in the direction with the most free space.  If the
    ``adjustText`` package is installed, it is used for final refinement.
    """
    import numpy as np

    n = len(raw_names)
    if n == 0:
        return []

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    x_range = max(xs.max() - xs.min(), 1e-10)
    y_range = max(ys.max() - ys.min(), 1e-10)

    texts = []
    for i in range(n):
        display = _get_display_name(raw_names[i])
        wrapped = _wrap_name(display, max_chars=max_chars)

        # Net repulsion direction from other points
        dx_sum, dy_sum = 0.0, 0.0
        for j in range(n):
            if i == j:
                continue
            dx = (xs[i] - xs[j]) / x_range
            dy = (ys[i] - ys[j]) / y_range
            dist_sq = max(dx ** 2 + dy ** 2, 1e-10)
            dx_sum += dx / dist_sq
            dy_sum += dy / dist_sq

        mag = max((dx_sum ** 2 + dy_sum ** 2) ** 0.5, 1e-10)
        offset_x = dx_sum / mag * 14
        offset_y = dy_sum / mag * 10

        # Ensure minimum clearance from the point itself
        if abs(offset_x) < 8:
            offset_x = 8 if offset_x >= 0 else -8
        if abs(offset_y) < 6:
            offset_y = 6 if offset_y >= 0 else -6

        ha = "left" if offset_x >= 0 else "right"
        va = "bottom" if offset_y >= 0 else "top"

        t = ax.annotate(
            wrapped, (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            fontsize=fontsize, color="#333333", fontweight="medium",
            ha=ha, va=va,
        )
        texts.append(t)

    # Refine with adjustText if available
    try:
        from adjustText import adjust_text
        adjust_text(
            texts, x=xs.tolist(), y=ys.tolist(), ax=ax,
            arrowprops=dict(arrowstyle="-", color="#CCCCCC", lw=0.5),
        )
    except (ImportError, Exception):
        pass

    return texts


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_leaderboard(
    ratings: Dict[str, float],
    *,
    confidence: Optional[Dict[str, Tuple[float, float, float]]] = None,
    method: str = "BT",
    title: str = "",
    subtitle: str = "",
    output_path: Optional[Path] = None,
    ax=None,
):
    """Vertical bar chart of Elo ratings, grouped by model family.

    Bars within the same family are joined (no gap). A gap separates
    different families. Each family gets a base color with shades for
    individual models.

    Args:
        ratings: {player: elo_rating}.
        confidence: {player: (median, lo, hi)} from bootstrap.
        method: Rating method label (shown in title).
        title: Plot title (overrides default).
        subtitle: Subtitle below title.
        output_path: If set, save figure to this path.
        ax: Optional matplotlib Axes.

    Returns:
        The matplotlib Axes.
    """
    import matplotlib.pyplot as plt
    _setup_style()

    # Sort by Elo descending, then group by family
    ranked = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    # Build family groups: within each family, keep Elo-descending order.
    # Family order = order of first appearance in the ranked list.
    from collections import OrderedDict
    family_groups: Dict[str, List[Tuple[str, float]]] = OrderedDict()
    for raw_name, elo in ranked:
        fam = _detect_family(raw_name)
        family_groups.setdefault(fam, []).append((raw_name, elo))

    # Flatten into display order: families separated by gaps
    BAR_WIDTH = 0.72
    FAMILY_GAP = 0.45  # extra space between families

    ordered_raw: List[str] = []
    ordered_elos: List[float] = []
    x_positions: List[float] = []
    family_spans: List[Tuple[float, float, str]] = []  # (x_start, x_end, family)

    x = 0.0
    for fam_idx, (fam, members) in enumerate(family_groups.items()):
        if fam_idx > 0:
            x += FAMILY_GAP  # gap before this family
        fam_start = x
        for raw_name, elo in members:
            ordered_raw.append(raw_name)
            ordered_elos.append(elo)
            x_positions.append(x)
            x += BAR_WIDTH  # bars touch within family
        fam_end = x - BAR_WIDTH
        family_spans.append((fam_start, fam_end, fam))

    # Assign family-based colors (shades within each family)
    cmap = _assign_family_colors(ordered_raw)
    colors = [cmap[n] for n in ordered_raw]
    display_names = [_get_display_name(n) for n in ordered_raw]

    own_fig = ax is None
    if own_fig:
        total_width = x_positions[-1] + BAR_WIDTH if x_positions else 6
        fig, ax = plt.subplots(figsize=(max(6, total_width * 1.3 + 2), 7))

    # Error bars from bootstrap CIs
    yerr = None
    if confidence:
        lo_err = []
        hi_err = []
        for raw_name in ordered_raw:
            if raw_name in confidence:
                med, lo, hi = confidence[raw_name]
                lo_err.append(ratings[raw_name] - lo)
                hi_err.append(hi - ratings[raw_name])
            else:
                lo_err.append(0)
                hi_err.append(0)
        yerr = [lo_err, hi_err]

    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    # Draw bars with rounded top corners.
    # x/y radii are computed separately so corners look round despite
    # the very different axis scales (x ~ 0-3, y ~ 0-1200).
    max_elo = max(ordered_elos) if ordered_elos else 1000
    r_x = BAR_WIDTH * 0.22
    r_y = max_elo * 0.035

    for i, (xp, elo, color) in enumerate(zip(x_positions, ordered_elos, colors)):
        rx = min(r_x, BAR_WIDTH / 2)
        ry = min(r_y, elo / 2) if elo > 0 else 0
        verts = [
            (xp, 0),                         # bottom-left
            (xp, elo - ry),                   # left edge before curve
            (xp, elo),                        # control point (top-left)
            (xp + rx, elo),                   # after top-left curve
            (xp + BAR_WIDTH - rx, elo),       # before top-right curve
            (xp + BAR_WIDTH, elo),            # control point (top-right)
            (xp + BAR_WIDTH, elo - ry),       # after top-right curve
            (xp + BAR_WIDTH, 0),              # bottom-right
            (xp, 0),                          # close
        ]
        codes = [
            MplPath.MOVETO, MplPath.LINETO,
            MplPath.CURVE3, MplPath.CURVE3,
            MplPath.LINETO,
            MplPath.CURVE3, MplPath.CURVE3,
            MplPath.LINETO, MplPath.CLOSEPOLY,
        ]
        path = MplPath(verts, codes)
        patch = PathPatch(path, facecolor=color, edgecolor="white",
                          linewidth=0.5)
        ax.add_patch(patch)

    # Error bars (drawn separately since we use patches, not ax.bar)
    if yerr:
        tick_positions = [xp + BAR_WIDTH / 2 for xp in x_positions]
        ax.errorbar(tick_positions, ordered_elos,
                    yerr=yerr, fmt="none", capsize=4, ecolor="#666666",
                    elinewidth=1.2, capthick=1.2)
    else:
        tick_positions = [xp + BAR_WIDTH / 2 for xp in x_positions]

    # Must set xlim/ylim manually since add_patch doesn't auto-scale
    max_elo = max(ordered_elos) if ordered_elos else 1000
    ax.set_xlim(-0.3, x_positions[-1] + BAR_WIDTH + 0.3 if x_positions else 2)
    ax.set_ylim(0, max_elo * 1.12)

    # X-axis: tick at center of each bar (wrapped names to avoid overlap)
    wrapped_names = [_wrap_name(n, max_chars=12) for n in display_names]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(wrapped_names, fontsize=10, color="#333333",
                       rotation=35, ha="right", rotation_mode="anchor")

    ax.set_ylabel("ELO RATING", fontsize=11, fontweight="bold", color="#555555",
                   labelpad=10)

    ax.yaxis.set_major_locator(plt.MaxNLocator(8, integer=True))

    # Only horizontal grid lines
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels above bars
    for i, elo in enumerate(ordered_elos):
        ax.text(tick_positions[i], elo + max_elo * 0.015, f"{elo:.0f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
                color="#333333")

    # Color legend (top-right)
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, edgecolor="white", linewidth=0.5, label=d)
                      for c, d in zip(colors, display_names)]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
              framealpha=0.9, edgecolor="#CCCCCC", handlelength=1.2,
              handleheight=0.9, labelspacing=0.4, borderpad=0.6)

    # Title
    final_title = title or f"Elo Leaderboard ({method})"
    ax.set_title(final_title, pad=20 if subtitle else 12, fontsize=18,
                 fontweight="bold", color="#222222")
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=12, color="#777777")

    if own_fig:
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches="tight",
                        facecolor="#FAFAF8")
            logger.info("Saved leaderboard to %s", output_path)
        plt.close()

    return ax


def plot_leaderboard_table(
    ratings: Dict[str, float],
    *,
    confidence: Optional[Dict[str, Tuple[float, float, float]]] = None,
    wld: Optional[Dict[str, Dict[str, int]]] = None,
    method: str = "BT",
    title: str = "",
    output_path: Optional[Path] = None,
):
    """Render a publication-quality leaderboard table as an image.

    Styled after Anthropic benchmark tables: clean rows, model columns
    grouped by family color, best values highlighted, rounded border
    on the top-ranked column.

    Args:
        ratings: {player: elo}.
        confidence: {player: (median, lo, hi)}.
        wld: {player: {wins, losses, draws, games}}.
        method: Rating method label.
        title: Table title.
        output_path: Save path.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    _setup_style()

    if not ratings:
        return

    # Sort by Elo descending
    ranked = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    raw_names = [r[0] for r in ranked]
    n_models = len(raw_names)

    # Get family colors
    cmap = _assign_family_colors(raw_names)
    display_names = [_get_display_name(n) for n in raw_names]
    families = [_detect_family(n) for n in raw_names]

    # Build row data: (label, subtitle, values[], format_str, higher_is_better)
    rows: List[Tuple[str, str, List[str], List[bool]]] = []

    # -- Elo Rating
    elo_vals = [ratings[n] for n in raw_names]
    best_elo = max(elo_vals)
    elo_strs = []
    elo_bests = []
    for i, n in enumerate(raw_names):
        elo = ratings[n]
        s = f"{elo:.0f}"
        if confidence and n in confidence:
            _, lo, hi = confidence[n]
            s += f"\n[{lo:.0f}, {hi:.0f}]"
        elo_strs.append(s)
        elo_bests.append(abs(elo - best_elo) < 0.1)
    rows.append(("Elo Rating", method.upper(), elo_strs, elo_bests))

    # -- Win Rate
    if wld:
        wr_strs = []
        wr_bests = []
        best_wr = -1.0
        wr_vals = []
        for n in raw_names:
            w = wld.get(n, {})
            games = w.get("games", 0)
            wins = w.get("wins", 0)
            wr = wins / games if games > 0 else 0.0
            wr_vals.append(wr)
            if wr > best_wr:
                best_wr = wr
        for i, n in enumerate(raw_names):
            wr_strs.append(f"{wr_vals[i]*100:.0f}%")
            wr_bests.append(abs(wr_vals[i] - best_wr) < 0.001)
        rows.append(("Win Rate", "", wr_strs, wr_bests))

        # -- W / L / D
        wld_strs = []
        for n in raw_names:
            w = wld.get(n, {})
            wld_strs.append(f"{w.get('wins',0)}W / {w.get('losses',0)}L / {w.get('draws',0)}D")
        rows.append(("Record", "W / L / D", wld_strs, [False] * n_models))

        # -- Total Games
        game_strs = [str(wld.get(n, {}).get("games", 0)) for n in raw_names]
        rows.append(("Games", "", game_strs, [False] * n_models))

    # --- Layout constants ---
    COL_WIDTH = 2.2
    ROW_LABEL_WIDTH = 2.0
    ROW_HEIGHT = 0.7
    HEADER_HEIGHT = 1.0
    n_rows = len(rows)

    fig_w = ROW_LABEL_WIDTH + n_models * COL_WIDTH + 0.4
    fig_h = HEADER_HEIGHT + n_rows * ROW_HEIGHT + 0.6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAF8")

    # Y positions (top to bottom)
    top_y = fig_h - 0.3

    # --- Header row: model names with family color backgrounds ---
    for col, raw_name in enumerate(raw_names):
        x_left = ROW_LABEL_WIDTH + col * COL_WIDTH
        color = cmap[raw_name]
        # Light tint of family color for header background
        import matplotlib.colors as mcolors
        r, g, b = mcolors.to_rgb(color)
        bg = (r * 0.3 + 0.7, g * 0.3 + 0.7, b * 0.3 + 0.7)  # lighten

        header_box = FancyBboxPatch(
            (x_left + 0.05, top_y - HEADER_HEIGHT),
            COL_WIDTH - 0.1, HEADER_HEIGHT - 0.05,
            boxstyle="round,pad=0,rounding_size=0.15",
            facecolor=bg, edgecolor=color if col == 0 else "none",
            linewidth=2.5 if col == 0 else 0,
        )
        ax.add_patch(header_box)

        # Model display name (wrapped to fit column width)
        wrapped = _wrap_name(display_names[col], max_chars=12)
        ax.text(x_left + COL_WIDTH / 2, top_y - HEADER_HEIGHT / 2.8,
                wrapped,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="#222222", linespacing=1.1)
        # Family subtitle
        ax.text(x_left + COL_WIDTH / 2, top_y - HEADER_HEIGHT / 1.3,
                families[col],
                ha="center", va="center", fontsize=9, color="#888888")

    # --- Data rows ---
    for row_idx, (label, sub, values, bests) in enumerate(rows):
        y_top = top_y - HEADER_HEIGHT - row_idx * ROW_HEIGHT
        y_mid = y_top - ROW_HEIGHT / 2

        # Alternating row background
        if row_idx % 2 == 0:
            bg_rect = plt.Rectangle(
                (0, y_top - ROW_HEIGHT), fig_w, ROW_HEIGHT,
                facecolor="#F5F4F0", edgecolor="none", zorder=0,
            )
            ax.add_patch(bg_rect)

        # Row label
        ax.text(0.15, y_mid + 0.08, label,
                ha="left", va="center", fontsize=11, fontweight="bold",
                color="#333333")
        if sub:
            ax.text(0.15, y_mid - 0.18, sub,
                    ha="left", va="center", fontsize=8, color="#999999")

        # Values
        for col, (val_str, is_best) in enumerate(zip(values, bests)):
            x_center = ROW_LABEL_WIDTH + col * COL_WIDTH + COL_WIDTH / 2

            # Highlight best value
            if is_best:
                hl = FancyBboxPatch(
                    (x_center - COL_WIDTH / 2 + 0.15, y_top - ROW_HEIGHT + 0.05),
                    COL_WIDTH - 0.3, ROW_HEIGHT - 0.1,
                    boxstyle="round,pad=0,rounding_size=0.1",
                    facecolor="#F0EDE5", edgecolor="none",
                )
                ax.add_patch(hl)

            weight = "bold" if is_best else "normal"
            ax.text(x_center, y_mid,
                    val_str,
                    ha="center", va="center", fontsize=11, fontweight=weight,
                    color="#222222" if is_best else "#444444",
                    linespacing=1.4)

    # Thin separator lines between rows
    for row_idx in range(n_rows + 1):
        y = top_y - HEADER_HEIGHT - row_idx * ROW_HEIGHT
        ax.axhline(y=y, xmin=0, xmax=1, color="#E0DDD5", linewidth=0.5)

    # Title
    final_title = title or f"Leaderboard ({method.upper()})"
    ax.text(fig_w / 2, top_y + 0.15, final_title,
            ha="center", va="bottom", fontsize=16, fontweight="bold",
            color="#222222")

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight",
                    facecolor="#FAFAF8")
        logger.info("Saved leaderboard table to %s", output_path)
    plt.close()


def plot_win_rate_heatmap(
    matches: list,
    level: str = "model",
    *,
    title: str = "",
    output_path: Optional[Path] = None,
    ax=None,
):
    """Win rate matrix as a heatmap with annotations.

    Args:
        matches: List of MatchRow dicts.
        level: "bot" or "model".
        title: Plot title.
        output_path: If set, save figure.
        ax: Optional Axes for embedding.

    Returns:
        The matplotlib Axes.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    # Build win matrix
    if level == "model":
        key_a, key_b = "red_model", "blue_model"
    else:
        key_a, key_b = "red_bot", "blue_bot"

    raw_players = sorted({m.get(key_a, "") for m in matches}
                         | {m.get(key_b, "") for m in matches})
    raw_players = [p for p in raw_players if p]  # drop empty
    if not raw_players:
        return ax

    display_players = [_wrap_name(_get_display_name(p), max_chars=12)
                       for p in raw_players]
    idx = {p: i for i, p in enumerate(raw_players)}
    n = len(raw_players)
    wins = np.zeros((n, n))
    total = np.zeros((n, n))

    for m in matches:
        a, b = m.get(key_a, ""), m.get(key_b, "")
        if a not in idx or b not in idx or a == b:
            continue
        i, j = idx[a], idx[b]
        total[i][j] += 1
        total[j][i] += 1
        winner = m.get("winner", "tie")
        if winner == "red":
            wins[i][j] += 1
        elif winner == "blue":
            wins[j][i] += 1
        else:
            wins[i][j] += 0.5
            wins[j][i] += 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        wr = np.where(total > 0, wins / total, 0.5)
    np.fill_diagonal(wr, 0.5)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(6, n * 1.2 + 2), max(5, n * 1.0 + 1)))

    try:
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
        # Earthy diverging: terracotta (loss) → cream → sage (win)
        cmap = LinearSegmentedColormap.from_list(
            "earthy", ["#C0856B", "#FAF3E8", "#7BAE7F"])
        sns.heatmap(wr, annot=True, fmt=".2f", cmap=cmap,
                    xticklabels=display_players, yticklabels=display_players,
                    vmin=0, vmax=1, ax=ax, linewidths=1, linecolor="white",
                    cbar_kws={"label": "Win Rate", "shrink": 0.8},
                    annot_kws={"fontsize": 12, "fontweight": "bold"})
    except ImportError:
        ax.imshow(wr, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(display_players, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(display_players)

    ax.set_title(title or "Win Rate Matrix", fontsize=18, fontweight="bold",
                 color="#222222", pad=12)

    # Remove default axis labels — names are self-explanatory
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Style ticks
    ax.tick_params(axis="both", which="both", length=0)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    if own_fig:
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches="tight",
                        facecolor="#FAFAF8")
            logger.info("Saved heatmap to %s", output_path)
        plt.close()

    return ax


def plot_elo_progression(
    season_dir: Path,
    level: str = "model",
    *,
    title: str = "",
    output_path: Optional[Path] = None,
    ax=None,
):
    """Line chart of Bradley-Terry rating over iterations (one line per player).

    Args:
        season_dir: Season output directory.
        level: "bot" or "model".
        title: Plot title.
        output_path: If set, save figure.
        ax: Optional Axes for embedding.

    Returns:
        The matplotlib Axes.
    """
    import matplotlib.pyplot as plt
    from .ratings import load_matches, compute_bradley_terry
    _setup_style()

    all_matches = load_matches(season_dir, force=False)
    if not all_matches:
        return ax

    # Group by iteration
    max_iter = max(int(m.get("iteration", 0)) for m in all_matches)
    iterations = list(range(max_iter + 1))

    compute_fn = compute_bradley_terry

    # Compute cumulative ratings at each iteration
    player_history: Dict[str, List[float]] = {}
    for end_iter in iterations:
        subset = [m for m in all_matches if int(m.get("iteration", 0)) <= end_iter]
        if not subset:
            continue
        ratings = compute_fn(subset, level=level)
        for player, rating in ratings.items():
            player_history.setdefault(player, []).append(rating)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 6))

    cmap = _assign_colors(sorted(player_history.keys()))
    for player, history in sorted(player_history.items()):
        color = cmap.get(player, _PALETTE[0])
        display = _get_display_name(player)
        iters = list(range(len(history)))
        ax.plot(iters, history, marker="o", markersize=5, label=display,
                color=color, linewidth=2.5)

    ax.set_xlabel("Iteration", fontsize=12, fontweight="bold", color="#555555")
    ax.set_ylabel("ELO RATING", fontsize=11, fontweight="bold", color="#555555")
    ax.set_title(title or "Elo Over Time (BT)", fontsize=18,
                 fontweight="bold", color="#222222", pad=12)
    ax.legend(loc="best", fontsize=10, framealpha=0.9, edgecolor="#CCCCCC")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if own_fig:
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches="tight",
                        facecolor="#FAFAF8")
            logger.info("Saved progression to %s", output_path)
        plt.close()

    return ax


# ---------------------------------------------------------------------------
# Time / Cost vs Elo scatter plots
# ---------------------------------------------------------------------------


def _match_build_stats_to_ratings(
    ratings: Dict[str, float],
    build_stats: Dict[str, Dict],
) -> List[Tuple[str, float, Dict]]:
    """Match rating keys to build_stats keys (by generator stem).

    Returns [(raw_name, elo, stats_dict), ...] for matched models.
    """
    matched = []
    for raw_name, elo in ratings.items():
        # Extract generator stem the same way as _get_display_name
        parts = raw_name.split(".")
        stem = ".".join(parts[2:]) if len(parts) >= 3 else raw_name

        if stem in build_stats:
            matched.append((raw_name, elo, build_stats[stem]))
    return matched


def plot_time_vs_elo(
    ratings: Dict[str, float],
    build_stats: Dict[str, Dict],
    *,
    phase: str = "morphology",
    title: str = "",
    output_path: Optional[Path] = None,
):
    """Scatter plot: Generation Time (X, inverted) vs Elo (Y).

    Args:
        ratings: {player: elo}.
        build_stats: Output of ``load_build_stats()``.
        phase: ``"morphology"`` or ``"controller"``.
        title: Plot title.
        output_path: Save path.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _setup_style()

    matched = _match_build_stats_to_ratings(ratings, build_stats)
    time_key = f"{phase}_time_avg"

    # Filter to models with timing data
    points = [(n, elo, s[time_key]) for n, elo, s in matched if s.get(time_key, 0) > 0]
    if len(points) < 2:
        logger.info("Not enough timing data for %s time vs Elo plot (%d points)", phase, len(points))
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    raw_names = [p[0] for p in points]
    elos = np.array([p[1] for p in points])
    times = np.array([p[2] for p in points])

    # Color by family
    family_plotted: Dict[str, bool] = {}
    for i, name in enumerate(raw_names):
        family = _detect_family(name)
        base = _FAMILY_BASE_COLORS.get(family, _DEFAULT_BASE)
        color = f"#{int(base[0]*255):02x}{int(base[1]*255):02x}{int(base[2]*255):02x}"
        label = family if family not in family_plotted else None
        family_plotted[family] = True
        ax.scatter(times[i], elos[i], c=color, s=140, edgecolors="white",
                   linewidth=1.5, zorder=5, label=label)

    # Smart label placement (avoids dots, wraps long names)
    _smart_annotate(ax, raw_names, times, elos)

    # Regression line
    if len(points) >= 3:
        z = np.polyfit(times, elos, 1)
        p = np.poly1d(z)
        x_line = np.linspace(times.min() * 0.9, times.max() * 1.1, 100)
        ax.plot(x_line, p(x_line), "--", color="#AAAAAA", alpha=0.7, linewidth=1)

        # Correlation
        n = len(times)
        mean_x, mean_y = times.mean(), elos.mean()
        cov = np.sum((times - mean_x) * (elos - mean_y))
        var_x = np.sum((times - mean_x) ** 2)
        var_y = np.sum((elos - mean_y) ** 2)
        denom = (var_x * var_y) ** 0.5
        corr = cov / denom if denom > 0 else 0
        corr_str = f"r = {corr:.3f}"
    else:
        corr_str = ""

    phase_label = phase.replace("_", " ").title()
    ax.set_xlabel(f"{phase_label} Generation Time (seconds)", fontsize=13,
                  fontweight="bold", color="#555555")
    ax.set_ylabel("Elo Score", fontsize=13, fontweight="bold", color="#555555")
    default_title = f"{phase_label} Generation Time vs Elo"
    if corr_str:
        default_title += f" ({corr_str})"
    ax.set_title(title or default_title, fontsize=18, fontweight="bold",
                 color="#222222", pad=14)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=10, framealpha=0.9, edgecolor="#CCCCCC",
              title="Family", title_fontsize=11)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#FAFAF8")
        logger.info("Saved time vs Elo plot to %s", output_path)
    plt.close()


def plot_cost_vs_elo(
    ratings: Dict[str, float],
    build_stats: Dict[str, Dict],
    *,
    phase: str = "morphology",
    title: str = "",
    output_path: Optional[Path] = None,
):
    """Scatter plot: Generation Cost (X) vs Elo (Y).

    Args:
        ratings: {player: elo}.
        build_stats: Output of ``load_build_stats()``.
        phase: ``"morphology"`` or ``"controller"``.
        title: Plot title.
        output_path: Save path.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _setup_style()

    matched = _match_build_stats_to_ratings(ratings, build_stats)
    cost_key = f"{phase}_cost_avg"

    # Filter to models with cost data
    points = [(n, elo, s[cost_key]) for n, elo, s in matched if s.get(cost_key, 0) > 0]
    if len(points) < 2:
        logger.info("Not enough cost data for %s cost vs Elo plot (%d points)", phase, len(points))
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    raw_names = [p[0] for p in points]
    elos = np.array([p[1] for p in points])
    costs = np.array([p[2] for p in points])

    # Color by family
    family_plotted: Dict[str, bool] = {}
    for i, name in enumerate(raw_names):
        family = _detect_family(name)
        base = _FAMILY_BASE_COLORS.get(family, _DEFAULT_BASE)
        color = f"#{int(base[0]*255):02x}{int(base[1]*255):02x}{int(base[2]*255):02x}"
        label = family if family not in family_plotted else None
        family_plotted[family] = True
        ax.scatter(costs[i], elos[i], c=color, s=140, edgecolors="white",
                   linewidth=1.5, zorder=5, label=label)

    # Smart label placement (avoids dots, wraps long names)
    _smart_annotate(ax, raw_names, costs, elos)

    # Regression line
    if len(points) >= 3:
        z = np.polyfit(costs, elos, 1)
        p = np.poly1d(z)
        x_line = np.linspace(costs.min() * 0.9, costs.max() * 1.1, 100)
        ax.plot(x_line, p(x_line), "--", color="#AAAAAA", alpha=0.7, linewidth=1)

        # Correlation
        n = len(costs)
        mean_x, mean_y = costs.mean(), elos.mean()
        cov = np.sum((costs - mean_x) * (elos - mean_y))
        var_x = np.sum((costs - mean_x) ** 2)
        var_y = np.sum((elos - mean_y) ** 2)
        denom = (var_x * var_y) ** 0.5
        corr = cov / denom if denom > 0 else 0
        corr_str = f"r = {corr:.3f}"
    else:
        corr_str = ""

    phase_label = phase.replace("_", " ").title()
    ax.set_xlabel(f"{phase_label} Generation Cost (USD)", fontsize=13,
                  fontweight="bold", color="#555555")
    ax.set_ylabel("Elo Score", fontsize=13, fontweight="bold", color="#555555")
    default_title = f"{phase_label} Generation Cost vs Elo"
    if corr_str:
        default_title += f" ({corr_str})"
    ax.set_title(title or default_title, fontsize=18, fontweight="bold",
                 color="#222222", pad=14)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=10, framealpha=0.9, edgecolor="#CCCCCC",
              title="Family", title_fontsize=11)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#FAFAF8")
        logger.info("Saved cost vs Elo plot to %s", output_path)
    plt.close()


def plot_num_calls_vs_elo(
    ratings: Dict[str, float],
    build_stats: Dict[str, Dict],
    *,
    phase: str = "morphology",
    title: str = "",
    output_path: Optional[Path] = None,
):
    """Scatter plot: LLM Calls (X) vs Elo (Y).

    Args:
        ratings: {player: elo}.
        build_stats: Output of ``load_build_stats()``.
        phase: ``"morphology"`` or ``"controller"``.
        title: Plot title.
        output_path: Save path.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _setup_style()

    matched = _match_build_stats_to_ratings(ratings, build_stats)
    calls_key = f"{phase}_num_calls_avg"

    # Filter to models with call data
    points = [(n, elo, s[calls_key]) for n, elo, s in matched if s.get(calls_key, 0) > 0]
    if len(points) < 2:
        logger.info("Not enough call data for %s num_calls vs Elo plot (%d points)", phase, len(points))
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    raw_names = [p[0] for p in points]
    elos = np.array([p[1] for p in points])
    calls = np.array([p[2] for p in points])

    # Color by family
    family_plotted: Dict[str, bool] = {}
    for i, name in enumerate(raw_names):
        family = _detect_family(name)
        base = _FAMILY_BASE_COLORS.get(family, _DEFAULT_BASE)
        color = f"#{int(base[0]*255):02x}{int(base[1]*255):02x}{int(base[2]*255):02x}"
        label = family if family not in family_plotted else None
        family_plotted[family] = True
        ax.scatter(calls[i], elos[i], c=color, s=140, edgecolors="white",
                   linewidth=1.5, zorder=5, label=label)

    # Smart label placement (avoids dots, wraps long names)
    _smart_annotate(ax, raw_names, calls, elos)

    # Regression line
    if len(points) >= 3:
        z = np.polyfit(calls, elos, 1)
        p = np.poly1d(z)
        x_line = np.linspace(calls.min() * 0.9, calls.max() * 1.1, 100)
        ax.plot(x_line, p(x_line), "--", color="#AAAAAA", alpha=0.7, linewidth=1)

        # Correlation
        mean_x, mean_y = calls.mean(), elos.mean()
        cov = np.sum((calls - mean_x) * (elos - mean_y))
        var_x = np.sum((calls - mean_x) ** 2)
        var_y = np.sum((elos - mean_y) ** 2)
        denom = (var_x * var_y) ** 0.5
        corr = cov / denom if denom > 0 else 0
        corr_str = f"r = {corr:.3f}"
    else:
        corr_str = ""

    phase_label = phase.replace("_", " ").title()
    ax.set_xlabel(f"{phase_label} LLM Calls", fontsize=13,
                  fontweight="bold", color="#555555")
    ax.set_ylabel("Elo Score", fontsize=13, fontweight="bold", color="#555555")
    default_title = f"{phase_label} LLM Calls vs Elo"
    if corr_str:
        default_title += f" ({corr_str})"
    ax.set_title(title or default_title, fontsize=18, fontweight="bold",
                 color="#222222", pad=14)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=10, framealpha=0.9, edgecolor="#CCCCCC",
              title="Family", title_fontsize=11)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#FAFAF8")
        logger.info("Saved num_calls vs Elo plot to %s", output_path)
    plt.close()


# ---------------------------------------------------------------------------
# Orchestrator: saves 3 separate files into elo/ folder
# ---------------------------------------------------------------------------


def plot_dashboard(
    ratings: Dict[str, float],
    matches: list,
    season_dir: Path,
    level: str = "model",
    *,
    method: str = "BT",
    confidence: Optional[Dict[str, Tuple[float, float, float]]] = None,
    wld: Optional[Dict[str, Dict[str, int]]] = None,
    build_stats: Optional[Dict[str, Dict]] = None,
    title: str = "",
    output_path: Optional[Path] = None,
):
    """Generate plot files inside ``{season_dir}/elo/``.

    Files created:
      - ``elo/leaderboard.png``
      - ``elo/leaderboard_table.png``
      - ``elo/win_rate_heatmap.png``
      - ``elo/elo_progression.png``
      - ``elo/morphology_time_vs_elo.png`` (if build_stats provided)
      - ``elo/controller_time_vs_elo.png`` (if build_stats provided)
      - ``elo/morphology_cost_vs_elo.png`` (if build_stats has cost data)
      - ``elo/controller_cost_vs_elo.png`` (if build_stats has cost data)

    Args:
        ratings: {player: elo}.
        matches: MatchRow list.
        season_dir: Season output directory.
        level: "bot" or "model".
        method: Rating method label.
        confidence: Bootstrap CIs.
        wld: Win/loss/draw counts.
        build_stats: Output of ``load_build_stats()``. If provided, generates
            time-vs-elo and cost-vs-elo scatter plots.
        title: Title prefix.
        output_path: Ignored (kept for backward compat).
    """
    elo_dir = Path(season_dir) / "elo"
    elo_dir.mkdir(parents=True, exist_ok=True)

    method_label = method.upper() if method else "BT"

    # 1. Leaderboard bar chart
    plot_leaderboard(
        ratings,
        confidence=confidence,
        method=method_label,
        title=title or None,
        output_path=elo_dir / "leaderboard.png",
    )

    # 2. Leaderboard table
    plot_leaderboard_table(
        ratings,
        confidence=confidence,
        wld=wld,
        method=method_label,
        title=title or None,
        output_path=elo_dir / "leaderboard_table.png",
    )

    # 3. Win rate heatmap
    plot_win_rate_heatmap(
        matches,
        level=level,
        title="Win Rate Matrix",
        output_path=elo_dir / "win_rate_heatmap.png",
    )

    # 4. Elo progression
    plot_elo_progression(
        season_dir,
        level=level,
        title="Elo Over Time",
        output_path=elo_dir / "elo_progression.png",
    )

    # 5. Time / Cost vs Elo (if build_stats provided)
    if build_stats:
        for phase in ("morphology", "controller"):
            plot_time_vs_elo(
                ratings, build_stats,
                phase=phase,
                output_path=elo_dir / f"{phase}_time_vs_elo.png",
            )
            plot_cost_vs_elo(
                ratings, build_stats,
                phase=phase,
                output_path=elo_dir / f"{phase}_cost_vs_elo.png",
            )
            plot_num_calls_vs_elo(
                ratings, build_stats,
                phase=phase,
                output_path=elo_dir / f"{phase}_num_calls_vs_elo.png",
            )

    logger.info("Saved plots to %s", elo_dir)


# ---------------------------------------------------------------------------
# JSON result output
# ---------------------------------------------------------------------------


def save_results(
    path: Path,
    ratings: Dict[str, float],
    *,
    confidence: Optional[Dict[str, Tuple[float, float, float]]] = None,
    wld: Optional[Dict[str, Dict[str, int]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save ratings to JSON for programmatic use (e.g., web app).

    Output schema::

        {
            "ratings": {"player": 1234.5, ...},
            "confidence": {"player": [median, lo, hi], ...} or null,
            "wld": {"player": {"wins": N, "losses": N, "draws": N}, ...},
            "metadata": {...}
        }
    """
    data: Dict[str, Any] = {
        "ratings": {k: round(v, 1) for k, v in ratings.items()},
    }
    if confidence:
        data["confidence"] = {
            k: [round(v[0], 1), round(v[1], 1), round(v[2], 1)]
            for k, v in confidence.items()
        }
    else:
        data["confidence"] = None

    data["wld"] = wld
    data["metadata"] = metadata or {}

    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved results JSON to %s", path)
