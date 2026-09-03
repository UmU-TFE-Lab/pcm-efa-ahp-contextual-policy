"""Build an alternative Figure 1 in a layered journal-workflow style.

The manuscript is not modified. Outputs are vector PDF/SVG plus a 600 dpi PNG.
"""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Rectangle,
)
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("PCM_FIGURE_OUTPUT_DIR", ROOT / "figures"))
OUT_PDF = OUTPUT_DIR / "figure_1_pcm_efa_ahp_framework.pdf"
OUT_SVG = OUTPUT_DIR / "figure_1_pcm_efa_ahp_framework.svg"
OUT_PNG = OUTPUT_DIR / "figure_1_pcm_efa_ahp_framework_600dpi.png"

INK = "#263A43"
NAVY = "#3E5E70"
MUTED = "#66737C"
LIGHT_LINE = "#DDE5E9"
PANEL_BG = "#FFFFFF"
BLUE = "#7FA9D1"
BLUE_PALE = "#DCEAF5"
TEAL = "#79B9AE"
TEAL_PALE = "#DCEFEB"
ORANGE = "#DDA261"
ORANGE_PALE = "#F7E5D1"
PURPLE = "#A99AC8"
PURPLE_PALE = "#EAE4F2"
YELLOW = "#EACB71"
YELLOW_PALE = "#FAF2D7"
CORAL = "#D97C63"
GREEN_PALE = "#CBE8C2"
CYAN_PALE = "#BFE5F0"
GRAY_PALE = "#F1F4F6"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "text.color": INK,
            "axes.edgecolor": NAVY,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    *,
    face=PANEL_BG,
    edge=NAVY,
    lw=0.9,
    radius=0.06,
    zorder=1,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def dashed_panel(ax, y, h):
    patch = Rectangle(
        (2.12, y),
        11.70,
        h,
        facecolor=PANEL_BG,
        edgecolor=NAVY,
        linewidth=1.15,
        linestyle=(0, (6, 3)),
        zorder=0,
    )
    ax.add_patch(patch)


def arrow(ax, start, end, *, color=NAVY, lw=1.05, rad=0.0, scale=10.0, zorder=4):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=zorder,
        )
    )


def line(ax, xs, ys, *, color=NAVY, lw=0.9, zorder=3, ls="-"):
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, zorder=zorder)


def stage_tab(ax, y, h, label, number, face):
    rounded_box(ax, 0.86, y + 0.035, 1.10, h - 0.07, face=face, edge="none", radius=0.08)
    ax.text(
        1.41,
        y + h / 2,
        label,
        ha="center",
        va="center",
        rotation=90,
        fontsize=10.2,
        fontweight="bold",
        color=INK,
        linespacing=0.95,
        zorder=5,
    )
    ax.add_patch(Circle((1.88, y + h - 0.13), 0.12, facecolor=ORANGE_PALE, edgecolor=CORAL, lw=0.8, zorder=6))
    ax.text(1.88, y + h - 0.13, str(number), ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=7)
    arrow(ax, (1.97, y + h / 2), (2.10, y + h / 2), scale=8.0, lw=0.85)


def label(ax, x, y, text, *, size=8.4, weight="normal", color=INK, ha="center", va="center", style="normal", z=8):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        color=color,
        fontstyle=style,
        linespacing=1.02,
        zorder=z,
    )


def draw_wallboard(ax, x, y):
    for dx, dy in [(0.00, 0.00), (0.10, 0.08), (0.20, 0.16)]:
        ax.add_patch(
            Polygon(
                [(x + dx, y + dy), (x + 0.85 + dx, y + dy), (x + 0.95 + dx, y + 0.50 + dy), (x + 0.10 + dx, y + 0.50 + dy)],
                closed=True,
                facecolor="#F7FAFC",
                edgecolor=NAVY,
                linewidth=0.9,
                zorder=3,
            )
        )
    for px, py in [(0.42, 0.28), (0.68, 0.18), (0.56, 0.43)]:
        ax.add_patch(Circle((x + px, y + py), 0.055, facecolor=CORAL, edgecolor="white", lw=0.4, zorder=5))
    label(ax, x + 0.56, y - 0.13, "Wallboard experiment", size=8.4, weight="bold")
    label(ax, x + 0.56, y - 0.31, "DSC and surface response", size=7.4, color=MUTED)


def draw_multiscale(ax, x, y):
    pts = np.array(
        [
            [x + 0.08, y + 0.30],
            [x + 0.28, y + 0.52],
            [x + 0.56, y + 0.57],
            [x + 0.78, y + 0.35],
            [x + 0.61, y + 0.13],
            [x + 0.30, y + 0.16],
        ]
    )
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 5), (2, 4)]:
        line(ax, [pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color=NAVY, lw=1.0)
    for i, (px, py) in enumerate(pts):
        ax.add_patch(Circle((px, py), 0.06, facecolor=[PURPLE, TEAL, BLUE][i % 3], edgecolor=NAVY, lw=0.7, zorder=5))
    arrow(ax, (x + 0.86, y + 0.33), (x + 1.10, y + 0.33), scale=8)
    cube_x = x + 1.17
    cube_y = y + 0.08
    ax.add_patch(Rectangle((cube_x, cube_y), 0.48, 0.48, facecolor="#F8FAFB", edgecolor=NAVY, lw=0.9, zorder=3))
    line(ax, [cube_x, cube_x + 0.18, cube_x + 0.66, cube_x + 0.48], [cube_y + 0.48, cube_y + 0.66, cube_y + 0.66, cube_y + 0.48], lw=0.9)
    line(ax, [cube_x + 0.48, cube_x + 0.66], [cube_y, cube_y + 0.18], lw=0.9)
    line(ax, [cube_x + 0.66, cube_x + 0.66], [cube_y + 0.18, cube_y + 0.66], lw=0.9)
    for px, py in [(0.12, 0.16), (0.32, 0.32), (0.52, 0.13), (0.46, 0.49)]:
        ax.add_patch(Circle((cube_x + px, cube_y + py), 0.055, facecolor=PURPLE, edgecolor="none", alpha=0.9, zorder=4))
    label(ax, x + 0.84, y - 0.13, "ATE multiscale constraints", size=8.4, weight="bold")
    label(ax, x + 0.84, y - 0.31, "MD, RVE-FEM, Mori-Tanaka", size=7.4, color=MUTED)


def draw_ida(ax, x, y):
    ax.add_patch(Polygon([(x + 0.08, y + 0.20), (x + 0.55, y + 0.60), (x + 1.02, y + 0.20)], closed=False, fill=False, edgecolor=NAVY, lw=1.3, zorder=4))
    ax.add_patch(Rectangle((x + 0.19, y - 0.03), 0.72, 0.36, facecolor="#FAFCFD", edgecolor=NAVY, lw=1.0, zorder=3))
    ax.add_patch(Rectangle((x + 0.31, y + 0.14), 0.48, 0.10, facecolor=TEAL, edgecolor="none", alpha=0.9, zorder=4))
    label(ax, x + 0.55, y + 0.19, "PCM layer", size=7.1, color=INK)
    ax.add_patch(Circle((x + 0.03, y + 0.58), 0.13, facecolor=YELLOW_PALE, edgecolor=ORANGE, lw=1.0, zorder=4))
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        line(ax, [x + 0.03 + 0.18 * np.cos(ang), x + 0.03 + 0.25 * np.cos(ang)], [y + 0.58 + 0.18 * np.sin(ang), y + 0.58 + 0.25 * np.sin(ang)], color=ORANGE, lw=0.9)
    line(ax, [x + 1.10, x + 1.32, x + 1.54, x + 1.75], [y + 0.12, y + 0.30, y + 0.08, y + 0.38], color=BLUE, lw=1.4)
    label(ax, x + 0.82, y - 0.13, "IDA-ICE hourly simulation", size=8.4, weight="bold")
    label(ax, x + 0.82, y - 0.31, "Weather, load and comfort cases", size=7.4, color=MUTED)


def draw_checks(ax, x, y):
    colors = [BLUE, TEAL, YELLOW]
    captions = ["Unit", "Range", "Balance"]
    for i in range(3):
        cx = x + i * 0.43
        ax.add_patch(Circle((cx, y + 0.26), 0.16, facecolor=[BLUE_PALE, TEAL_PALE, YELLOW_PALE][i], edgecolor=NAVY, lw=1.0, zorder=3))
        line(ax, [cx - 0.07, cx - 0.01, cx + 0.09], [y + 0.26, y + 0.20, y + 0.34], color=colors[i], lw=1.7, zorder=5)
        label(ax, cx, y - 0.02, captions[i], size=6.8, color=MUTED)
    label(ax, x + 0.43, y - 0.23, "Physics and consistency checks", size=8.2, weight="bold")


def draw_dataset(ax, x, y):
    w, h = 1.05, 0.72
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#F5F8FA", edgecolor=NAVY, lw=1.0, zorder=3))
    ax.add_patch(Ellipse((x + w / 2, y + h), w, 0.22, facecolor="#F5F8FA", edgecolor=NAVY, lw=1.0, zorder=4))
    ax.add_patch(Ellipse((x + w / 2, y), w, 0.22, facecolor="#EFF4F7", edgecolor=NAVY, lw=1.0, zorder=4))
    label(ax, x + w / 2, y + 0.46, "Fused scenario", size=8.2, weight="bold")
    label(ax, x + w / 2, y + 0.28, "dataset", size=8.2, weight="bold")


def draw_criteria(ax, x, y):
    colors = [BLUE, TEAL, PURPLE, ORANGE]
    heights = [0.58, 0.38, 0.72, 0.49, 0.65, 0.30, 0.55, 0.44, 0.68, 0.35, 0.61]
    for i, height in enumerate(heights):
        ax.add_patch(Rectangle((x + i * 0.09, y), 0.058, height, facecolor=colors[i % 4], edgecolor="white", lw=0.3, zorder=3))
    line(ax, [x - 0.05, x + 1.02], [y, y], color=NAVY, lw=0.7)
    label(ax, x + 0.48, y - 0.15, "11 benefit-oriented criteria", size=8.1, weight="bold")


def draw_heatmap(ax, x, y):
    vals = np.array(
        [
            [0.9, 0.72, 0.12, -0.08, 0.18],
            [0.72, 0.9, 0.15, -0.12, 0.14],
            [0.12, 0.15, 0.9, 0.58, -0.05],
            [-0.08, -0.12, 0.58, 0.9, 0.22],
            [0.18, 0.14, -0.05, 0.22, 0.9],
        ]
    )
    for r in range(5):
        for c in range(5):
            v = vals[r, c]
            if v >= 0:
                color = mpl.colors.to_hex(np.array(mpl.colors.to_rgb("#FFFFFF")) * (1 - 0.72 * v) + np.array(mpl.colors.to_rgb(CORAL)) * (0.72 * v))
            else:
                color = mpl.colors.to_hex(np.array(mpl.colors.to_rgb("#FFFFFF")) * (1 - 0.55 * abs(v)) + np.array(mpl.colors.to_rgb(BLUE)) * (0.55 * abs(v)))
            ax.add_patch(Rectangle((x + c * 0.13, y + (4 - r) * 0.13), 0.125, 0.125, facecolor=color, edgecolor="white", lw=0.35, zorder=3))
    label(ax, x + 0.32, y - 0.15, "Correlation structure", size=8.1, weight="bold")


def draw_scree(ax, x, y):
    obs = np.array([5.40, 2.12, 1.57, 1.26, 0.35, 0.16])
    obs = obs / 5.8 * 0.70
    xs = x + np.arange(6) * 0.16
    line(ax, [x - 0.05, x - 0.05], [y, y + 0.78], color=NAVY, lw=0.7)
    line(ax, [x - 0.05, x + 0.88], [y, y], color=NAVY, lw=0.7)
    line(ax, [x - 0.03, x + 0.88], [y + 0.13, y + 0.13], color=ORANGE, lw=0.9, ls="--")
    line(ax, xs, y + obs, color=BLUE, lw=1.25)
    ax.scatter(xs, y + obs, s=17, facecolor=BLUE, edgecolor="white", linewidth=0.4, zorder=5)
    for i in range(4):
        ax.add_patch(Rectangle((xs[i] - 0.045, y), 0.09, obs[i], facecolor=BLUE_PALE, edgecolor="none", alpha=0.8, zorder=1))
    label(ax, x + 0.40, y - 0.15, "95% parallel analysis", size=8.1, weight="bold")
    label(ax, x + 0.76, y + 0.18, "retain 4", size=6.8, color=CORAL)


def draw_factor_cards(ax, x, y):
    cards = [
        ("Capacity and power", BLUE_PALE, BLUE),
        ("Fast thermal response", TEAL_PALE, TEAL),
        ("Durability and low loss", PURPLE_PALE, PURPLE),
        ("Efficiency and load offset", ORANGE_PALE, ORANGE),
    ]
    for i, (txt, face, edge) in enumerate(cards):
        cx = x + (i % 2) * 1.43
        cy = y + (1 - i // 2) * 0.48
        rounded_box(ax, cx, cy, 1.28, 0.36, face=face, edge=edge, lw=0.85, radius=0.04, zorder=2)
        label(ax, cx + 0.64, cy + 0.18, txt, size=7.5, weight="bold")


def draw_experts(ax, x, y):
    colors = [BLUE, TEAL, ORANGE, PURPLE, BLUE]
    for i in range(5):
        cx = x + i * 0.24
        ax.add_patch(Circle((cx, y + 0.45), 0.075, facecolor=colors[i], edgecolor=NAVY, lw=0.55, zorder=4))
        ax.add_patch(
            FancyBboxPatch(
                (cx - 0.095, y + 0.17),
                0.19,
                0.20,
                boxstyle="round,pad=0.003,rounding_size=0.05",
                facecolor=mpl.colors.to_rgba(colors[i], 0.45),
                edgecolor=NAVY,
                linewidth=0.55,
                zorder=3,
            )
        )
    label(ax, x + 0.48, y - 0.02, "Five independent experts", size=8.1, weight="bold")
    label(ax, x + 0.48, y - 0.19, "six pairwise judgments each", size=7.2, color=MUTED)


def draw_ahp_matrix(ax, x, y):
    values = [
        ["1", "2", "1/2", "1/3"],
        ["1/2", "1", "1/3", "1/5"],
        ["2", "3", "1", "1/2"],
        ["3", "5", "2", "1"],
    ]
    cell = 0.22
    for r in range(4):
        for c in range(4):
            ax.add_patch(Rectangle((x + c * cell, y + (3 - r) * cell), cell, cell, facecolor="#FBFCFD", edgecolor=NAVY, lw=0.55, zorder=2))
            label(ax, x + (c + 0.5) * cell, y + (3.5 - r) * cell, values[r][c], size=6.8)
    label(ax, x + 0.44, y - 0.13, "Geometric aggregation", size=7.7, weight="bold")
    rounded_box(ax, x + 0.93, y + 0.20, 0.78, 0.40, face=YELLOW_PALE, edge=ORANGE, lw=0.8, radius=0.04)
    label(ax, x + 1.32, y + 0.46, "Group CR", size=7.0, color=MUTED)
    label(ax, x + 1.32, y + 0.29, "0.001", size=10.0, weight="bold")


def draw_weight_bars(ax, x, y):
    names = ["Efficiency", "Durability", "Capacity", "Response"]
    values = [0.348, 0.249, 0.246, 0.158]
    colors = [ORANGE, TEAL, BLUE, PURPLE]
    for i, (name, value, color) in enumerate(zip(names, values, colors)):
        yy = y + 0.78 - i * 0.23
        label(ax, x, yy, name, size=7.2, ha="right")
        ax.add_patch(Rectangle((x + 0.12, yy - 0.055), value * 2.0, 0.11, facecolor=color, edgecolor="none", zorder=3))
        label(ax, x + 0.17 + value * 2.0, yy, f"{value:.3f}", size=7.0, ha="left")
    label(ax, x + 0.45, y - 0.08, "Expert group weights", size=8.1, weight="bold")


def draw_reward(ax, x, y):
    ax.add_patch(Ellipse((x + 0.57, y + 0.45), 1.14, 0.72, facecolor=ORANGE_PALE, edgecolor=ORANGE, lw=1.1, zorder=2))
    label(ax, x + 0.57, y + 0.56, r"$r=\mathbf{w}^{\mathsf{T}}\mathbf{f}$", size=12.5, weight="bold")
    label(ax, x + 0.57, y + 0.31, "EFA-AHP score", size=7.7, weight="bold")


def draw_ranking(ax, x, y):
    labels = ["OP-Batt.-SS", "OP-Batt.-Macro", "OP-Solar-SS"]
    values = [0.151, 0.135, 0.122]
    colors = [BLUE, TEAL, PURPLE]
    for i, (name, value, color) in enumerate(zip(labels, values, colors)):
        yy = y + 0.82 - i * 0.25
        label(ax, x, yy, name, size=7.2, ha="left")
        ax.add_patch(Rectangle((x + 1.10, yy - 0.06), value * 5.3, 0.12, facecolor=color, edgecolor="none", zorder=3))
        label(ax, x + 1.16 + value * 5.3, yy, f"{value:.3f}", size=6.9, ha="left")
    label(ax, x + 1.05, y + 0.02, "Engineering-default screening", size=8.0, weight="bold")


def draw_baseline_checks(ax, x, y):
    rounded_box(ax, x, y + 0.42, 1.33, 0.54, face=GRAY_PALE, edge=NAVY, lw=0.75, radius=0.03)
    label(ax, x + 0.24, y + 0.79, "TOPSIS", size=7.5, weight="bold", ha="left")
    label(ax, x + 1.13, y + 0.79, "same top", size=6.9, color=MUTED, ha="right")
    label(ax, x + 0.24, y + 0.58, "VIKOR", size=7.5, weight="bold", ha="left")
    label(ax, x + 1.13, y + 0.58, "same top", size=6.9, color=MUTED, ha="right")
    label(ax, x + 0.67, y + 0.17, "Entropy-weight baselines", size=7.8, weight="bold")


def draw_probabilities(ax, x, y):
    items = [("Top-1", 0.587, ORANGE), ("Top-3", 0.887, TEAL)]
    for i, (name, value, color) in enumerate(items):
        yy = y + 0.74 - i * 0.33
        label(ax, x, yy, name, size=7.4, ha="left")
        ax.add_patch(Rectangle((x + 0.47, yy - 0.06), 1.08, 0.12, facecolor=LIGHT_LINE, edgecolor="none", zorder=1))
        ax.add_patch(Rectangle((x + 0.47, yy - 0.06), 1.08 * value, 0.12, facecolor=color, edgecolor="none", zorder=3))
        label(ax, x + 1.64, yy, f"{value:.3f}", size=7.2, weight="bold", ha="right")
    label(ax, x + 0.82, y + 0.08, "Row-bootstrap stability", size=7.8, weight="bold")


def draw_projection_check(ax, x, y):
    names = ["Reduced", "Alternative"]
    values = [0.984, 0.9997]
    for i, (name, value) in enumerate(zip(names, values)):
        yy = y + 0.70 - i * 0.32
        label(ax, x, yy, name, size=7.1, ha="left")
        line(ax, [x + 0.62, x + 1.55], [yy, yy], color=LIGHT_LINE, lw=4.8)
        line(ax, [x + 0.62, x + 0.62 + value * 0.93], [yy, yy], color=BLUE, lw=4.8)
        ax.scatter([x + 0.62 + value * 0.93], [yy], s=18, facecolor=BLUE, edgecolor="white", lw=0.4, zorder=5)
        label(ax, x + 1.67, yy, f"{value:.4f}" if i else f"{value:.3f}", size=6.8, ha="right")
    label(ax, x + 0.80, y + 0.06, "Fixed-reference rank correlation", size=7.7, weight="bold")


def draw_split(ax, x, y):
    widths = [1.95, 0.65, 0.65]
    colors = [BLUE, ORANGE, PURPLE]
    texts = ["Train 60%", "Validation\n20%", "Test\n20%"]
    xx = x
    for w, color, txt in zip(widths, colors, texts):
        ax.add_patch(Rectangle((xx, y + 0.53), w, 0.36, facecolor=color, edgecolor=NAVY, lw=0.65, zorder=3))
        label(ax, xx + w / 2, y + 0.71, txt, size=7.2, weight="bold")
        xx += w
    label(ax, x + 1.63, y + 0.29, "chronological split", size=8.1, weight="bold")
    label(ax, x + 1.63, y + 0.08, "fit preprocessing and EFA on train only", size=7.2, color=MUTED)


def draw_state_policy(ax, x, y):
    state_items = [
        (x + 0.32, y + 0.73, "Weather", BLUE_PALE),
        (x + 1.14, y + 0.73, "Load", ORANGE_PALE),
        (x + 0.32, y + 0.30, "Cycle", BLUE_PALE),
        (x + 1.14, y + 0.30, "SOC", ORANGE_PALE),
    ]
    for cx, cy, txt, face in state_items:
        ax.add_patch(Ellipse((cx, cy), 0.67, 0.28, facecolor=face, edgecolor=NAVY, lw=0.8, zorder=3))
        label(ax, cx, cy, txt, size=7.1)
    arrow(ax, (x + 1.52, y + 0.52), (x + 1.88, y + 0.52), scale=8.0)
    hx, hy = x + 2.35, y + 0.52
    points = [(hx - 0.43, hy), (hx - 0.22, hy + 0.38), (hx + 0.28, hy + 0.38), (hx + 0.49, hy), (hx + 0.28, hy - 0.38), (hx - 0.22, hy - 0.38)]
    ax.add_patch(Polygon(points, closed=True, facecolor=TEAL_PALE, edgecolor=NAVY, lw=1.0, zorder=3))
    label(ax, hx + 0.03, hy + 0.08, "Support-aware", size=7.4, weight="bold")
    label(ax, hx + 0.03, hy - 0.11, "policy", size=7.4, weight="bold")


def draw_ope(ax, x, y):
    xmin, xmax = -0.05, 0.25
    x0, x1 = x + 0.18, x + 2.55
    def sx(v):
        return x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)

    line(ax, [sx(0), sx(0)], [y + 0.18, y + 1.20], color=MUTED, lw=0.75)
    for val in [0.0, 0.1, 0.2]:
        line(ax, [sx(val), sx(val)], [y + 0.16, y + 0.22], color=NAVY, lw=0.55)
        label(ax, sx(val), y + 0.04, f"{val:.1f}", size=6.5, color=MUTED)
    rows = [
        ("Static EFA-AHP", 0.119, 0.018, 0.220, BLUE, y + 0.93),
        ("Selected policy", 0.087, -0.035, 0.204, TEAL, y + 0.47),
    ]
    for name, value, lo, hi, color, yy in rows:
        label(ax, x0 - 0.10, yy, name, size=7.2, ha="right")
        line(ax, [sx(lo), sx(hi)], [yy, yy], color=color, lw=2.0)
        line(ax, [sx(lo), sx(lo)], [yy - 0.07, yy + 0.07], color=color, lw=1.0)
        line(ax, [sx(hi), sx(hi)], [yy - 0.07, yy + 0.07], color=color, lw=1.0)
        ax.scatter([sx(value)], [yy], s=34, facecolor=color, edgecolor=NAVY, lw=0.7, zorder=5)
        label(ax, sx(value) + 0.07, yy, f"{value:.3f}", size=6.9, ha="left")
    label(ax, x + 1.35, y + 1.32, "Locked-test doubly robust value", size=8.1, weight="bold")


def build_figure() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.0, 8.6))
    fig.subplots_adjust(left=0.012, right=0.994, bottom=0.018, top=0.992)
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 8.55)
    ax.axis("off")

    rows = [
        (6.55, 1.75),
        (5.00, 1.40),
        (3.55, 1.35),
        (2.10, 1.35),
        (0.25, 1.75),
    ]
    for y, h in rows:
        dashed_panel(ax, y, h)

    rounded_box(ax, 0.12, 6.55, 1.86, 1.75, face=GREEN_PALE, edge="none", radius=0.08)
    label(ax, 1.05, 7.43, "Evidence and\ndata fusion", size=11.6, weight="bold")
    arrow(ax, (0.42, 6.56), (0.42, 6.39), color=ORANGE, lw=2.2, scale=11)

    rounded_box(ax, 0.12, 0.25, 0.62, 6.15, face=GREEN_PALE, edge="none", radius=0.08)
    ax.text(0.43, 3.32, "PCM-TES configuration decision workflow", ha="center", va="center", rotation=90, fontsize=11.5, fontweight="bold", color=INK, zorder=8)

    stage_tab(ax, 5.00, 1.40, "Factor\nconstruction", 1, CYAN_PALE)
    stage_tab(ax, 3.55, 1.35, "Expert\nweighting", 2, CYAN_PALE)
    stage_tab(ax, 2.10, 1.35, "Static screening\nand robustness", 3, CYAN_PALE)
    stage_tab(ax, 0.25, 1.75, "Contextual\nevaluation", 4, CYAN_PALE)

    draw_wallboard(ax, 2.46, 7.34)
    arrow(ax, (3.64, 7.56), (4.14, 7.56), scale=8)
    draw_multiscale(ax, 4.25, 7.24)
    arrow(ax, (5.96, 7.56), (6.31, 7.56), scale=8)
    draw_ida(ax, 6.42, 7.27)
    draw_checks(ax, 8.86, 7.30)
    arrow(ax, (8.23, 7.49), (8.69, 7.49), scale=9)
    arrow(ax, (10.10, 7.49), (10.44, 7.49), scale=9)
    draw_dataset(ax, 10.51, 7.05)
    label(ax, 12.05, 7.77, "Hourly indexed scenario records", size=8.5, weight="bold", ha="left")
    label(ax, 12.05, 7.49, "Source and derived variables", size=8.0, ha="left")
    label(ax, 12.05, 7.23, "Material-system-encapsulation", size=8.0, ha="left")
    label(ax, 12.05, 7.02, "configuration space", size=8.0, ha="left")
    label(ax, 12.05, 6.77, "scenario calendar, not field observations", size=7.1, color=MUTED, style="italic", ha="left")

    draw_criteria(ax, 2.48, 5.38)
    arrow(ax, (3.58, 5.72), (3.94, 5.72), scale=8)
    draw_heatmap(ax, 4.08, 5.39)
    arrow(ax, (4.82, 5.72), (5.24, 5.72), scale=8)
    draw_scree(ax, 5.39, 5.35)
    arrow(ax, (6.36, 5.72), (6.78, 5.72), scale=8)
    draw_factor_cards(ax, 6.93, 5.23)
    rounded_box(ax, 10.03, 5.20, 3.47, 1.00, face=GRAY_PALE, edge=LIGHT_LINE, lw=0.7, radius=0.04)
    label(ax, 10.28, 5.91, "EFA diagnostics", size=8.4, weight="bold", ha="left")
    label(ax, 10.28, 5.64, "KMO", size=7.5, color=MUTED, ha="left")
    label(ax, 11.20, 5.64, "0.717", size=8.4, weight="bold", ha="left")
    label(ax, 11.86, 5.64, "Bartlett p", size=7.5, color=MUTED, ha="left")
    label(ax, 13.20, 5.64, "< 0.001", size=8.2, weight="bold", ha="right")
    label(ax, 10.28, 5.36, "Retained factors", size=7.5, color=MUTED, ha="left")
    label(ax, 11.45, 5.36, "4", size=8.4, weight="bold", ha="left")
    label(ax, 11.86, 5.36, "Variance", size=7.5, color=MUTED, ha="left")
    label(ax, 13.20, 5.36, "94.13%", size=8.2, weight="bold", ha="right")

    draw_experts(ax, 2.48, 3.86)
    arrow(ax, (3.62, 4.20), (3.92, 4.20), scale=8)
    draw_ahp_matrix(ax, 4.03, 3.77)
    arrow(ax, (5.78, 4.20), (6.08, 4.20), scale=8)
    draw_weight_bars(ax, 6.45, 3.68)
    arrow(ax, (8.70, 4.20), (9.03, 4.20), scale=8)
    draw_reward(ax, 9.14, 3.76)
    arrow(ax, (10.32, 4.20), (10.68, 4.20), scale=8)
    rounded_box(ax, 10.78, 3.73, 2.75, 0.95, face=ORANGE_PALE, edge=ORANGE, lw=0.85, radius=0.04)
    label(ax, 11.03, 4.43, "Shared objective", size=8.2, weight="bold", ha="left")
    label(ax, 11.03, 4.15, "Static configuration score", size=7.7, ha="left")
    label(ax, 11.03, 3.91, "Contextual decision reward", size=7.7, ha="left")
    label(ax, 13.29, 4.15, "screening", size=6.8, color=MUTED, ha="right")
    label(ax, 13.29, 3.91, "evaluation", size=6.8, color=MUTED, ha="right")

    draw_ranking(ax, 2.38, 2.28)
    arrow(ax, (5.15, 2.75), (5.43, 2.75), scale=8)
    draw_baseline_checks(ax, 5.52, 2.20)
    arrow(ax, (6.89, 2.75), (7.18, 2.75), scale=8)
    draw_probabilities(ax, 7.30, 2.25)
    arrow(ax, (9.03, 2.75), (9.29, 2.75), scale=8)
    draw_projection_check(ax, 9.38, 2.25)
    rounded_box(ax, 11.26, 2.28, 2.30, 0.97, face=YELLOW_PALE, edge=ORANGE, lw=0.85, radius=0.04)
    label(ax, 12.41, 3.02, "Interpretation boundary", size=8.1, weight="bold")
    label(ax, 12.41, 2.76, "cross-application screen", size=7.4)
    label(ax, 12.41, 2.54, "within-system shortlist", size=7.4)

    draw_split(ax, 2.38, 0.57)
    arrow(ax, (5.76, 1.09), (6.03, 1.09), scale=8)
    draw_state_policy(ax, 6.12, 0.57)
    arrow(ax, (9.02, 1.09), (9.27, 1.09), scale=8)
    draw_ope(ax, 9.30, 0.48)
    rounded_box(ax, 12.05, 0.55, 1.50, 1.06, face=YELLOW_PALE, edge=ORANGE, lw=0.85, radius=0.04)
    label(ax, 12.80, 1.34, "Locked-test result", size=8.1, weight="bold")
    label(ax, 12.80, 1.05, "No confirmed", size=8.5, weight="bold", color=CORAL)
    label(ax, 12.80, 0.82, "policy gain", size=8.5, weight="bold", color=CORAL)
    label(ax, 12.80, 0.61, "over static EFA-AHP", size=6.8, color=MUTED)

    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
