"""Shared publication style for PCM EFA-AHP contextual-policy figures.

The style uses muted colors, compact serif typography, thin axes, pale panel
backgrounds, light grids, and vector-friendly exports.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


INK = "#2B3036"
MUTED_INK = "#56616F"
AXIS = "#AEB8C5"
GRID = "#E7EBF1"
AX_FACE = "#FCFCFD"

BLUE = "#6E95BF"
DEEP_BLUE = "#4F78A6"
TEAL = "#83BFC0"
GREEN = "#7DB59F"
MOSS = "#8BAA73"
ORANGE = "#D9A66A"
AMBER = "#E3C678"
PURPLE = "#B7A4CF"
MAUVE = "#C59AB1"
RED = "#C97B82"
GRAY = "#7E8794"

PALE_BLUE = "#EAF2FA"
PALE_GREEN = "#EDF6F1"
PALE_ORANGE = "#FBF1E4"
PALE_PURPLE = "#F4EFF8"
PALE_GRAY = "#F6F7F9"

MUTED_SEQUENCE = [DEEP_BLUE, TEAL, GREEN, ORANGE, PURPLE, MOSS, MAUVE, AMBER, GRAY]
FACTOR_COLORS = [DEEP_BLUE, TEAL, GREEN, ORANGE]
POLICY_COLORS = [DEEP_BLUE, TEAL, PURPLE, GREEN, ORANGE, MAUVE]


def configure_publication_style(base_font: float = 8.6) -> None:
    """Configure matplotlib/seaborn for calm high-density journal figures."""
    sns.set_theme(
        context="paper",
        style="white",
        palette=MUTED_SEQUENCE,
        rc={
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": AX_FACE,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": base_font,
            "axes.labelsize": base_font - 0.1,
            "axes.titlesize": base_font,
            "xtick.labelsize": base_font - 1.25,
            "ytick.labelsize": base_font - 1.25,
            "legend.fontsize": base_font - 1.35,
            "legend.title_fontsize": base_font - 1.1,
            "legend.frameon": False,
            "legend.handlelength": 1.55,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 1.05,
            "axes.linewidth": 0.58,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.50,
            "ytick.major.width": 0.50,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "grid.color": GRID,
            "grid.linewidth": 0.42,
            "grid.alpha": 0.78,
            "lines.linewidth": 1.20,
            "lines.markersize": 3.8,
            "patch.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        },
    )
    mpl.rcParams["axes.titlelocation"] = "left"
    mpl.rcParams["figure.constrained_layout.use"] = False


def wrap_label(value: object, width: int) -> str:
    return "\n".join(
        textwrap.wrap(str(value).replace("_", " "), width=width, break_long_words=False)
    )


def clean_axis(ax: plt.Axes, axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.set_facecolor(AX_FACE)
    ax.grid(True, axis=axis, color=GRID, linewidth=0.42, alpha=0.82, zorder=0)
    ax.grid(False, axis="x" if axis == "y" else "y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.58)
    ax.tick_params(width=0.50, length=2.4, colors=INK)


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.075, y: float = 1.035) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color=INK,
    )


def save_figure(fig: plt.Figure, output_base: Path, *, png: bool = True) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    if png:
        fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def style_colorbar(cbar, *, labelsize: float = 6.8) -> None:
    cbar.outline.set_linewidth(0.45)
    cbar.outline.set_edgecolor(AXIS)
    cbar.ax.tick_params(labelsize=labelsize, width=0.42, length=2.2, colors=INK)


def soft_diverging_cmap(name: str = "soft_blue_rose") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        name,
        ["#5E82AE", "#DDE8F1", "#FCFBF8", "#F1DDD3", "#C97B82"],
    )


def soft_blue_cmap(name: str = "soft_blue") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, ["#F7FAFD", "#D8E7F2", "#94B7D4", "#4F78A6"])


def soft_green_cmap(name: str = "soft_green") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, ["#FCFBEF", "#DDEEC9", "#9ACB93", "#4F9A78"])


def soft_rank_cmap(name: str = "soft_rank") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        name,
        ["#5E82AE", "#DDE9F2", "#FCFBF5", "#EACB8E", "#C97B82"],
    )


def add_heatmap_grid(ax: plt.Axes, ncols: int, nrows: int) -> None:
    ax.set_xticks([i - 0.5 for i in range(1, ncols)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, nrows)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.65, alpha=0.95)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
