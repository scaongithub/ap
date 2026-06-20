"""
Visualization Module
=====================

Premium-quality visualizations for football match score predictions.
Renders heatmaps, outcome bar charts, top-score charts, and rich
terminal summaries with a cohesive dark-themed aesthetic.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    import matplotlib.figure

# Attempt to import rich — graceful degradation if absent
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False

# ── Local imports ──────────────────────────────────────────────────────
from src.poisson_matrix import get_outcome_probabilities, get_top_scores

# ── Shared style constants ─────────────────────────────────────────────
_BG_DARK = "#1a1a2e"
_BG_AXES = "#16213e"
_TEXT_COLOR = "#e0e0e0"
_ACCENT_HOME = "#2ecc71"  # emerald green
_ACCENT_DRAW = "#f39c12"  # amber
_ACCENT_AWAY = "#e74c3c"  # coral red
_FONT_FAMILY = "Segoe UI"

# Pre-configure matplotlib defaults for a premium dark feel
matplotlib.rcParams.update(
    {
        "font.family": _FONT_FAMILY,
        "text.color": _TEXT_COLOR,
        "axes.labelcolor": _TEXT_COLOR,
        "xtick.color": _TEXT_COLOR,
        "ytick.color": _TEXT_COLOR,
        "figure.facecolor": _BG_DARK,
        "axes.facecolor": _BG_AXES,
        "savefig.facecolor": _BG_DARK,
        "savefig.edgecolor": _BG_DARK,
    }
)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1. SCORE PROBABILITY HEATMAP                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


def plot_score_heatmap(
    matrix: np.ndarray,
    home_team: str,
    away_team: str,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Render a premium heatmap of the score probability matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Score probability matrix (e.g. 8×8 from :func:`build_score_matrix`).
    home_team : str
        Display name of the home team.
    away_team : str
        Display name of the away team.
    save_path : str, optional
        If provided, save the figure as a PNG at this path (200 DPI).

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib Figure.
    """
    plt.style.use("dark_background")
    n = matrix.shape[0]

    # Custom colormap: dark navy → teal/cyan → vibrant gold/amber
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "navy_teal_gold",
        ["#0d1b2a", "#1b3a4b", "#1f8a8a", "#45b7a0", "#f0c27f", "#f5a623"],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_AXES)

    # Build annotation array (percentages)
    annot = np.empty_like(matrix, dtype=object)
    pct = matrix * 100
    for i in range(n):
        for j in range(n):
            val = pct[i, j]
            if val < 0.1:
                annot[i, j] = ""
            elif val < 1.0:
                annot[i, j] = f"{val:.1f}%"
            else:
                annot[i, j] = f"{val:.1f}%"

    sns.heatmap(
        matrix,
        ax=ax,
        annot=annot,
        fmt="",
        cmap=cmap,
        linewidths=1.5,
        linecolor="#0a0f1a",
        square=True,
        cbar_kws={
            "label": "Probability",
            "shrink": 0.75,
            "aspect": 30,
            "pad": 0.02,
        },
        annot_kws={"size": 10, "weight": "bold", "color": _TEXT_COLOR},
        xticklabels=range(n),
        yticklabels=range(n),
    )

    # Add subtle text glow on high-prob cells
    for text_obj in ax.texts:
        text_obj.set_path_effects(
            [pe.withStroke(linewidth=2, foreground="#0d1b2a")]
        )

    ax.set_xlabel(f"Away Goals ({away_team})", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_ylabel(f"Home Goals ({home_team})", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_title(
        f"Score Probability Matrix: {home_team} vs {away_team}",
        fontsize=16,
        fontweight="bold",
        pad=20,
        color="#ffffff",
    )
    ax.tick_params(axis="both", labelsize=12)

    # Style the colour-bar text
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(_TEXT_COLOR)
    cbar.ax.yaxis.label.set_fontsize(11)
    cbar.ax.tick_params(colors=_TEXT_COLOR)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2. OUTCOME PROBABILITY BARS                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


def plot_outcome_bars(
    outcomes: dict,
    home_team: str,
    away_team: str,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Render horizontal bars for Home Win / Draw / Away Win probabilities.

    Parameters
    ----------
    outcomes : dict
        Dictionary with ``'home_win'``, ``'draw'``, ``'away_win'`` keys.
    home_team : str
        Display name of the home team.
    away_team : str
        Display name of the away team.
    save_path : str, optional
        If provided, save the figure as a PNG at 200 DPI.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib Figure.
    """
    plt.style.use("dark_background")

    labels = [f"Home Win ({home_team})", "Draw", f"Away Win ({away_team})"]
    values = [outcomes["home_win"], outcomes["draw"], outcomes["away_win"]]
    colors = [_ACCENT_HOME, _ACCENT_DRAW, _ACCENT_AWAY]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_DARK)

    bars = ax.barh(
        labels,
        values,
        color=colors,
        edgecolor="#ffffff22",
        height=0.55,
        zorder=3,
    )

    # Rounded-edge effect via FancyBboxPatch isn't natively supported by
    # barh, so we approximate with a subtle shadow + rounded bar cap.
    for bar, val, col in zip(bars, values, colors):
        bar.set_linewidth(0)
        # Add glow / soft shadow
        bar.set_path_effects(
            [pe.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace=col, alpha=0.25)]
        )
        # Percentage annotation
        pct_text = f"  {val * 100:.1f}%"
        ax.text(
            val + 0.005,
            bar.get_y() + bar.get_height() / 2,
            pct_text,
            va="center",
            ha="left",
            fontsize=14,
            fontweight="bold",
            color="#ffffff",
            path_effects=[pe.withStroke(linewidth=2, foreground=_BG_DARK)],
        )

    ax.set_xlim(0, max(values) * 1.25)
    ax.set_title(
        "Match Outcome Probabilities",
        fontsize=16,
        fontweight="bold",
        pad=16,
        color="#ffffff",
    )
    ax.xaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("#ffffff33")
    ax.tick_params(axis="y", labelsize=13, colors=_TEXT_COLOR)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  3. TOP SCORELINES BAR CHART                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


def plot_top_scores(
    top_scores: list[tuple[int, int, float]],
    home_team: str,
    away_team: str,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Render a gradient-coloured horizontal bar chart of top scorelines.

    Parameters
    ----------
    top_scores : list[tuple[int, int, float]]
        Sorted list of ``(home_goals, away_goals, probability)`` tuples.
    home_team : str
        Display name of the home team.
    away_team : str
        Display name of the away team.
    save_path : str, optional
        If provided, save the figure as a PNG at 200 DPI.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib Figure.
    """
    plt.style.use("dark_background")

    labels = [f"{h} - {a}" for h, a, _ in top_scores]
    probs = [p for _, _, p in top_scores]

    # Gradient from brightest (most likely) to faded
    max_p = max(probs) if probs else 1.0
    gradient_cmap = mcolors.LinearSegmentedColormap.from_list(
        "score_grad", ["#f5a623", "#45b7a0", "#1b3a4b"]
    )
    norm = plt.Normalize(vmin=0, vmax=max_p)
    bar_colors = [gradient_cmap(norm(p)) for p in probs]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_DARK)

    # Reverse so highest probability is at the top
    labels = labels[::-1]
    probs = probs[::-1]
    bar_colors = bar_colors[::-1]

    bars = ax.barh(
        labels,
        probs,
        color=bar_colors,
        edgecolor="#ffffff11",
        height=0.6,
        zorder=3,
    )

    for bar, val in zip(bars, probs):
        bar.set_path_effects(
            [pe.withSimplePatchShadow(offset=(1, -1), shadow_rgbFace="#000000", alpha=0.3)]
        )
        ax.text(
            val + max_p * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color="#ffffff",
        )

    ax.set_xlim(0, max_p * 1.25)
    ax.set_title(
        f"Top Predicted Scorelines: {home_team} vs {away_team}",
        fontsize=15,
        fontweight="bold",
        pad=16,
        color="#ffffff",
    )
    ax.xaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("#ffffff33")
    ax.tick_params(axis="y", labelsize=12, colors=_TEXT_COLOR)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  4. COMBINED PLOT + SAVE                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝


def plot_all(
    matrix: np.ndarray,
    home_team: str,
    away_team: str,
    output_dir: str = "output",
) -> None:
    """Generate and save all visualizations, then display them.

    Creates ``output_dir`` if it does not already exist and saves:

    * ``heatmap.png`` — score probability heatmap
    * ``outcomes.png`` — outcome probability bars
    * ``top_scores.png`` — top predicted scorelines

    Parameters
    ----------
    matrix : np.ndarray
        Score probability matrix.
    home_team : str
        Display name of the home team.
    away_team : str
        Display name of the away team.
    output_dir : str, optional
        Directory in which to save the PNGs (default ``'output'``).
    """
    os.makedirs(output_dir, exist_ok=True)

    outcomes = get_outcome_probabilities(matrix)
    top_scores = get_top_scores(matrix, n=10)

    plot_score_heatmap(
        matrix, home_team, away_team,
        save_path=os.path.join(output_dir, "heatmap.png"),
    )
    plot_outcome_bars(
        outcomes, home_team, away_team,
        save_path=os.path.join(output_dir, "outcomes.png"),
    )
    plot_top_scores(
        top_scores, home_team, away_team,
        save_path=os.path.join(output_dir, "top_scores.png"),
    )

    plt.show()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  5. RICH TERMINAL SUMMARY                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝


def render_rich_summary(
    matrix: np.ndarray,
    outcomes: dict,
    top_scores: list[tuple[int, int, float]],
    home_team: str,
    away_team: str,
    model_info: dict | None = None,
) -> None:
    """Print a beautifully-formatted terminal summary using the *rich* library.

    Parameters
    ----------
    matrix : np.ndarray
        Score probability matrix.
    outcomes : dict
        Outcome probabilities (``home_win``, ``draw``, ``away_win``).
    top_scores : list[tuple[int, int, float]]
        Sorted list of ``(home_goals, away_goals, probability)`` tuples.
    home_team : str
        Display name of the home team.
    away_team : str
        Display name of the away team.
    model_info : dict, optional
        If provided, a mapping of ``{model_name: prediction_dict}`` that
        will be rendered in an additional panel.
    """
    if not _HAS_RICH:
        print(
            "[visualize] 'rich' library is not installed. "
            "Install it with: pip install rich"
        )
        return

    console = Console()

    # ── Match header ──────────────────────────────────────────────────
    header = Text.assemble(
        ("⚽  ", "bold"),
        (home_team, "bold green"),
        ("  vs  ", "dim"),
        (away_team, "bold red"),
        ("  ⚽", "bold"),
    )
    console.print(
        Panel(
            header,
            title="[bold cyan]Match Prediction[/bold cyan]",
            border_style="cyan",
            padding=(1, 4),
        )
    )

    # ── Outcome probabilities ────────────────────────────────────────
    console.print()
    outcome_text = Text.assemble(
        ("  Home Win  ", "bold"),
        (f"{outcomes['home_win'] * 100:5.1f}%", "bold green"),
        ("   │   ", "dim"),
        ("Draw  ", "bold"),
        (f"{outcomes['draw'] * 100:5.1f}%", "bold yellow"),
        ("   │   ", "dim"),
        ("Away Win  ", "bold"),
        (f"{outcomes['away_win'] * 100:5.1f}%", "bold red"),
    )
    console.print(
        Panel(outcome_text, title="[bold]Outcome Probabilities[/bold]", border_style="blue")
    )

    # ── Top scorelines table ────────────────────────────────────────
    console.print()
    table = Table(
        title="Top Predicted Scorelines",
        title_style="bold magenta",
        border_style="bright_black",
        header_style="bold cyan",
        show_lines=True,
        padding=(0, 1),
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Score", justify="center", style="bold white", width=10)
    table.add_column("Prob %", justify="right", style="bold", width=8)
    table.add_column("Bar", width=30)

    max_prob = top_scores[0][2] if top_scores else 1.0
    for rank, (h, a, p) in enumerate(top_scores, start=1):
        pct = p * 100
        bar_length = int((p / max_prob) * 25)
        bar_char = "█" * bar_length + "░" * (25 - bar_length)

        # Color the bar based on outcome
        if h > a:
            bar_style = "green"
        elif h == a:
            bar_style = "yellow"
        else:
            bar_style = "red"

        table.add_row(
            str(rank),
            f"{h} - {a}",
            f"{pct:.1f}%",
            f"[{bar_style}]{bar_char}[/{bar_style}]",
        )

    console.print(table)

    # ── Model-specific predictions (optional) ────────────────────────
    if model_info:
        console.print()
        model_table = Table(
            title="Individual Model Predictions",
            title_style="bold magenta",
            border_style="bright_black",
            header_style="bold cyan",
            show_lines=True,
            padding=(0, 1),
        )
        model_table.add_column("Model", style="bold white", width=20)
        model_table.add_column("Home xG", justify="right", width=10)
        model_table.add_column("Away xG", justify="right", width=10)
        model_table.add_column("Predicted Score", justify="center", width=16)

        for name, info in model_info.items():
            home_xg = info.get("home_xg", "—")
            away_xg = info.get("away_xg", "—")
            pred = info.get("predicted_score", "—")
            model_table.add_row(
                name,
                f"{home_xg:.2f}" if isinstance(home_xg, (int, float)) else str(home_xg),
                f"{away_xg:.2f}" if isinstance(away_xg, (int, float)) else str(away_xg),
                str(pred),
            )

        console.print(model_table)

    console.print()
