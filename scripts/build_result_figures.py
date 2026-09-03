"""Build manuscript Figures 2--7 from locally generated analysis outputs."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import textwrap

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
STYLE_DIR = PACKAGE_ROOT / "pcm_journal_extension"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from publication_style import (  # noqa: E402
    AXIS,
    DEEP_BLUE,
    FACTOR_COLORS,
    GRAY,
    GREEN,
    INK,
    MAUVE,
    MUTED_INK,
    ORANGE,
    PURPLE,
    RED,
    TEAL,
    clean_axis,
    configure_publication_style,
    panel_label,
    save_figure,
    soft_blue_cmap,
    soft_diverging_cmap,
    soft_rank_cmap,
    style_colorbar,
)


ANALYSIS_ROOT = Path(os.environ.get("PCM_ANALYSIS_OUTPUT_ROOT", PACKAGE_ROOT / "outputs"))
STATIC = ANALYSIS_ROOT / "static_efa_ahp"
SUPPLEMENT = ANALYSIS_ROOT / "supplementary"
ROBUSTNESS = SUPPLEMENT / "reviewer_required"
POLICY = ANALYSIS_ROOT / "contextual_policy" / "locked_evaluation"
OUTPUT = Path(os.environ.get("PCM_FIGURE_OUTPUT_DIR", PACKAGE_ROOT / "figures"))

FACTOR_ORDER = [
    "Storage capacity and power",
    "Fast thermal response",
    "Durability and low loss",
    "Efficiency and load offset",
]

OUTPUT_NAMES = {
    2: "figure_2_source_plausibility",
    3: "figure_3_efa_diagnostics",
    4: "figure_4_ahp_weights",
    5: "figure_5_static_ranking",
    6: "figure_6_uncertainty_baselines",
    7: "figure_7_robustness_policy",
}


configure_publication_style(base_font=8.6)


def wrap(value: object, width: int) -> str:
    return "\n".join(
        textwrap.wrap(str(value).replace("_", " "), width=width, break_long_words=False)
    )


def compact_alternative(value: str) -> str:
    parts = [part.strip() for part in str(value).split("|")]
    if len(parts) != 3:
        return wrap(value, 20)
    pcm, system, encapsulation = parts
    pcm_short = {
        "Organic_Paraffin": "OP",
        "Inorganic_SaltHydrate": "ISH",
        "Eutectic": "EU",
    }.get(pcm, pcm)
    system_short = {
        "BatteryCooling": "Batt.",
        "BuildingEnvelope": "Bldg.",
        "HVACStorage": "HVAC",
        "SolarTES": "Solar",
    }.get(system, system)
    encapsulation_short = {
        "ShapeStabilized": "SS",
        "Macro": "Macro",
        "Micro": "Micro",
    }.get(encapsulation, encapsulation)
    return f"{pcm_short}-{system_short}-{encapsulation_short}"


def compact_policy(value: str) -> str:
    return {
        "static_efa_ahp_best": "Static EFA-AHP",
        "linear_support_penalty_0.05": "Selected support penalty",
    }.get(value, value.replace("_", " "))


def heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    *,
    cmap,
    vmin: float,
    vmax: float,
    fmt: str,
):
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            color = "white" if abs(value) > 0.62 * max(abs(vmin), abs(vmax)) else INK
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=6.1, color=color)
    return image


def save(fig: plt.Figure, number: int) -> None:
    save_figure(fig, OUTPUT / OUTPUT_NAMES[number])


def source_plausibility_figure() -> None:
    # These compact source-level values are transcribed in the manuscript audit;
    # they are not record-level observations from the private scenario table.
    experiment = pd.DataFrame(
        {
            "board_type": ["Gypsum", "cPCM", "mPCM60", "mPCM82"],
            "decrement_factor_change_pct_vs_gypsum": [0.0, 5.3, -7.7, -13.0],
        }
    )
    idaice = pd.DataFrame(
        {
            "baseline_value": [9150.0, 17706.0, 8.12, 34349.0],
            "pcm_value": [8523.0, 17274.0, 7.81, 31163.0],
        }
    )
    idaice["computed_reduction_pct"] = (
        (idaice["baseline_value"] - idaice["pcm_value"])
        / idaice["baseline_value"]
        * 100.0
    )
    ranges = pd.read_csv(SUPPLEMENT / "physical_plausibility_audit.csv").head(9)

    fig = plt.figure(figsize=(7.2, 6.05))
    grid = GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.34)

    ax = fig.add_subplot(grid[0, 0])
    bars = ax.bar(
        experiment["board_type"],
        experiment["decrement_factor_change_pct_vs_gypsum"],
        color=[GRAY, ORANGE, TEAL, GREEN],
        width=0.62,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.axhline(0, color=GRAY, lw=0.72, alpha=0.82)
    ax.set_ylabel("DF change vs gypsum (%)")
    ax.set_ylim(-15.5, 7.5)
    clean_axis(ax, axis="y")
    panel_label(ax, "(a)")
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.8 if value >= 0 else -0.8),
            f"{value:.1f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=7.2,
        )

    ax = fig.add_subplot(grid[0, 1])
    idaice = idaice.copy()
    idaice["short_label"] = ["Stockholm\nheat", "Stockholm\ncool", "Stockholm\npeak", "Svalbard\nheat"]
    x = np.arange(len(idaice))
    width = 0.34
    ax.bar(x - width / 2, idaice["baseline_value"], width=width, color=DEEP_BLUE, alpha=0.92, edgecolor="white", linewidth=0.45, label="Baseline", zorder=3)
    ax.bar(x + width / 2, idaice["pcm_value"], width=width, color=ORANGE, alpha=0.92, edgecolor="white", linewidth=0.45, label="PCM", zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("Reported value")
    ax.set_xticks(x)
    ax.set_xticklabels(idaice["short_label"])
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    clean_axis(ax, axis="y")
    panel_label(ax, "(b)")

    ax = fig.add_subplot(grid[1, 0])
    ax.barh(x, idaice["computed_reduction_pct"], color=GREEN, alpha=0.92, edgecolor="white", linewidth=0.45, height=0.58, zorder=3)
    ax.set_yticks(x)
    ax.set_yticklabels(idaice["short_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Reduction (%)")
    ax.set_xlim(0, idaice["computed_reduction_pct"].max() * 1.22)
    clean_axis(ax, axis="x")
    panel_label(ax, "(c)")
    for idx, value in enumerate(idaice["computed_reduction_pct"]):
        ax.text(value + 0.22, idx, f"{value:.2f}", va="center", fontsize=7.1)

    ax = fig.add_subplot(grid[1, 1])
    y = np.arange(len(ranges))
    for idx, row in enumerate(ranges.itertuples()):
        expected_low = float(row.expected_min)
        expected_high = float(row.expected_max)
        span = expected_high - expected_low if expected_high > expected_low else 1.0
        observed_low = np.clip((float(row.observed_min) - expected_low) / span, 0, 1)
        observed_high = np.clip((float(row.observed_max) - expected_low) / span, 0, 1)
        ax.plot([0, 1], [idx, idx], color=AXIS, lw=3.6, alpha=0.44, solid_capstyle="round", zorder=1)
        ax.plot([observed_low, observed_high], [idx, idx], color=DEEP_BLUE, lw=3.6, alpha=0.86, solid_capstyle="round", zorder=2)
        ax.scatter([observed_low, observed_high], [idx, idx], s=11, color=DEEP_BLUE, edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(value, 21) for value in ranges["variable"]])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Normalized admissible interval")
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["min", "mid", "max"])
    clean_axis(ax, axis="x")
    panel_label(ax, "(d)")
    save(fig, 2)


def efa_diagnostics_figure() -> None:
    loadings = pd.read_csv(STATIC / "factor_loadings.csv", index_col=0)[FACTOR_ORDER]
    eigenvalues = pd.read_csv(STATIC / "efa_eigenvalues.csv")
    parallel = pd.read_csv(STATIC / "parallel_analysis_eigenvalues.csv")
    diagnostics = pd.read_csv(STATIC / "efa_criterion_diagnostics.csv")
    residuals = pd.read_csv(STATIC / "efa_residual_correlations.csv", index_col=0)

    fig = plt.figure(figsize=(7.2, 6.1))
    grid = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.40)
    diverging = soft_diverging_cmap()

    ax = fig.add_subplot(grid[0, 0])
    image = heatmap(ax, loadings.to_numpy(), [wrap(value, 14) for value in loadings.columns], [wrap(value, 22) for value in loadings.index], cmap=diverging, vmin=-1, vmax=1, fmt="{:.2f}")
    style_colorbar(fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02))
    panel_label(ax, "(a)")

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(eigenvalues["factor_number"], eigenvalues["eigenvalue"], marker="o", color=DEEP_BLUE, lw=1.35, ms=3.8, label="Observed")
    ax.plot(parallel["factor_number"], parallel["random_95pct_eigenvalue"], marker="s", color=ORANGE, lw=1.15, ms=3.4, label="Random 95%")
    ax.axhline(1.0, color=GRAY, lw=0.8, ls="--", label="Kaiser = 1")
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue")
    ax.set_xlim(0.7, 6.3)
    ax.set_ylim(0, eigenvalues["eigenvalue"].max() * 1.12)
    ax.legend(frameon=False, loc="upper right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    clean_axis(ax, axis="y")
    panel_label(ax, "(b)")

    ax = fig.add_subplot(grid[1, 0])
    ordered = diagnostics.sort_values("communality")
    y = np.arange(len(ordered))
    ax.barh(y, ordered["communality"], color=GREEN, alpha=0.92, edgecolor="white", linewidth=0.45, height=0.58, label="Communality", zorder=3)
    ax.scatter(ordered["uniqueness"], y, color=RED, s=16, alpha=0.9, zorder=4, label="Uniqueness")
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(value, 24) for value in ordered["criterion"]])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Diagnostic value")
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax, axis="x")
    panel_label(ax, "(c)")

    ax = fig.add_subplot(grid[1, 1])
    order = diagnostics.sort_values("primary_factor")["criterion"].tolist()
    image = ax.imshow(residuals.loc[order, order].to_numpy(), cmap=diverging, vmin=-0.20, vmax=0.20, aspect="auto")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([str(index + 1) for index in range(len(order))])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([str(index + 1) for index in range(len(order))])
    ax.set_xlabel("Criterion index")
    ax.set_ylabel("Criterion index")
    for spine in ax.spines.values():
        spine.set_visible(False)
    style_colorbar(fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02))
    panel_label(ax, "(d)")
    save(fig, 3)


def ahp_weights_figure() -> None:
    individual = pd.read_csv(STATIC / "expert_individual_weights.csv")
    expert = individual.pivot(index="expert_id", columns="factor", values="weight").loc[:, FACTOR_ORDER]
    expert_cr = individual.groupby("expert_id", sort=True)["cr"].first()
    group = pd.read_csv(STATIC / "expert_group_weights.csv").set_index("factor").loc[FACTOR_ORDER]
    scenario = pd.read_csv(STATIC / "scenario_weights.csv")
    scenario_order = ["balanced", "engineering_default", "storage_capacity_priority", "efficiency_load_priority", "fast_response_priority", "durability_low_loss_priority"]
    scenario_pivot = scenario.pivot(index="scenario", columns="factor", values="weight").loc[scenario_order, FACTOR_ORDER]

    fig = plt.figure(figsize=(7.2, 6.15))
    grid = GridSpec(2, 2, figure=fig, hspace=0.66, wspace=0.48)

    ax = fig.add_subplot(grid[0, 0])
    image = heatmap(ax, expert.to_numpy(), [wrap(value, 13) for value in FACTOR_ORDER], expert.index.tolist(), cmap=soft_blue_cmap(), vmin=0, vmax=0.56, fmt="{:.2f}")
    style_colorbar(fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02))
    panel_label(ax, "(a)")

    ax = fig.add_subplot(grid[0, 1])
    group_order = ["Efficiency and load offset", "Durability and low loss", "Storage capacity and power", "Fast thermal response"]
    group_show = group.loc[group_order]
    x = np.arange(len(group_show))
    bars = ax.bar(x, group_show["weight"], color=[ORANGE, GREEN, DEEP_BLUE, TEAL], alpha=0.94, edgecolor="white", linewidth=0.45, width=0.58, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([wrap(value, 14) for value in group_show.index], rotation=25, ha="right")
    ax.set_ylabel("Group AHP weight")
    ax.set_ylim(0, 0.39)
    clean_axis(ax, axis="y")
    panel_label(ax, "(b)")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{bar.get_height():.3f}", ha="center", fontsize=7.0)

    ax = fig.add_subplot(grid[1, 0])
    bottoms = np.zeros(len(scenario_pivot))
    for factor, color in zip(FACTOR_ORDER, FACTOR_COLORS):
        ax.bar(np.arange(len(scenario_pivot)), scenario_pivot[factor], bottom=bottoms, color=color, edgecolor="white", linewidth=0.4, label=wrap(factor, 13), zorder=3)
        bottoms += scenario_pivot[factor].to_numpy()
    ax.set_xticks(np.arange(len(scenario_pivot)))
    ax.set_xticklabels(["Balanced", "Default", "Storage", "Efficiency", "Response", "Durability"], rotation=25, ha="right")
    ax.set_ylabel("Scenario weight")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.28), borderaxespad=0)
    clean_axis(ax, axis="y")
    panel_label(ax, "(c)")

    ax = fig.add_subplot(grid[1, 1])
    ax.bar(expert_cr.index, expert_cr, color=MAUVE, alpha=0.92, edgecolor="white", linewidth=0.45, zorder=3)
    ax.axhline(0.10, color=RED, lw=0.9, ls="--", label="CR = 0.10")
    ax.set_ylabel("Consistency ratio")
    ax.set_ylim(0, 0.11)
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax, axis="y")
    panel_label(ax, "(d)")
    save(fig, 4)


def static_ranking_figure() -> None:
    rankings = pd.read_csv(STATIC / "scenario_rankings.csv")
    default = rankings[rankings["scenario"] == "engineering_default"].sort_values("rank").head(10).copy()
    stability = pd.read_csv(STATIC / "rank_stability.csv")
    stability = stability[stability["alternative"].isin(default["alternative"])]

    fig = plt.figure(figsize=(7.2, 6.0))
    grid = GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.42)

    ax = fig.add_subplot(grid[0, 0])
    ordered = default.sort_values("final_score")
    y = np.arange(len(ordered))
    ax.barh(y, ordered["final_score"], color=DEEP_BLUE, alpha=0.92, edgecolor="white", linewidth=0.45, height=0.58, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([compact_alternative(value) for value in ordered["alternative"]])
    ax.set_xlabel("EFA-AHP score")
    clean_axis(ax, axis="x")
    panel_label(ax, "(a)")

    ax = fig.add_subplot(grid[0, 1])
    factors = [f"{factor}_score" for factor in FACTOR_ORDER]
    image = heatmap(ax, default.set_index("alternative")[factors].to_numpy(), [wrap(value.replace("_score", ""), 12) for value in factors], [compact_alternative(value) for value in default["alternative"]], cmap=soft_diverging_cmap(), vmin=-0.8, vmax=0.8, fmt="{:.2f}")
    style_colorbar(fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02))
    panel_label(ax, "(b)")

    ax = fig.add_subplot(grid[1, 0])
    rank_columns = ["balanced", "engineering_default", "storage_capacity_priority", "efficiency_load_priority", "fast_response_priority", "durability_low_loss_priority"]
    rank_values = stability.set_index("alternative").loc[default["alternative"], rank_columns]
    image = heatmap(ax, rank_values.to_numpy(), ["Bal.", "Def.", "Stor.", "Eff.", "Resp.", "Dur."], [compact_alternative(value) for value in rank_values.index], cmap=soft_rank_cmap(), vmin=1, vmax=15, fmt="{:.0f}")
    style_colorbar(fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02))
    panel_label(ax, "(c)")

    ax = fig.add_subplot(grid[1, 1])
    counts = default.groupby(["pcm_type", "system_type"]).size().unstack(fill_value=0)
    image = heatmap(ax, counts.to_numpy(), [wrap(value, 12) for value in counts.columns], [wrap(value, 15) for value in counts.index], cmap=soft_blue_cmap(), vmin=0, vmax=max(1, counts.to_numpy().max()), fmt="{:.0f}")
    style_colorbar(fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02))
    panel_label(ax, "(d)")
    save(fig, 5)


def uncertainty_baselines_figure() -> None:
    row_bootstrap = pd.read_csv(SUPPLEMENT / "bootstrap_rank_probabilities.csv").head(6)
    block_bootstrap = pd.read_csv(ROBUSTNESS / "time_block_bootstrap_rank_uncertainty.csv").head(6)
    baselines = pd.read_csv(SUPPLEMENT / "mcdm_baseline_summary.csv")

    fig = plt.figure(figsize=(7.2, 5.9))
    grid = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.40)

    ax = fig.add_subplot(grid[0, 0])
    y = np.arange(len(row_bootstrap))
    ax.errorbar(row_bootstrap["mean_score"], y, xerr=[row_bootstrap["mean_score"] - row_bootstrap["ci95_lower"], row_bootstrap["ci95_upper"] - row_bootstrap["mean_score"]], fmt="o", color=DEEP_BLUE, ecolor="#D7E3EE", elinewidth=1.45, capsize=2.2, markersize=3.8, markeredgecolor="white", markeredgewidth=0.45, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([compact_alternative(value) for value in row_bootstrap["alternative"]])
    ax.invert_yaxis()
    ax.set_xlabel("Row-bootstrap score")
    clean_axis(ax, axis="x")
    panel_label(ax, "(a)")

    ax = fig.add_subplot(grid[0, 1])
    ax.barh(y - 0.18, row_bootstrap["top1_probability"], height=0.32, color=ORANGE, alpha=0.92, edgecolor="white", linewidth=0.45, label="Top-1", zorder=3)
    ax.barh(y + 0.18, row_bootstrap["top3_probability"], height=0.32, color=GREEN, alpha=0.92, edgecolor="white", linewidth=0.45, label="Top-3", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([compact_alternative(value) for value in row_bootstrap["alternative"]])
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    clean_axis(ax, axis="x")
    panel_label(ax, "(b)")

    ax = fig.add_subplot(grid[1, 0])
    y = np.arange(len(block_bootstrap))
    ax.errorbar(block_bootstrap["block_bootstrap_mean_score"], y, xerr=[block_bootstrap["block_bootstrap_mean_score"] - block_bootstrap["block_bootstrap_ci95_lower"], block_bootstrap["block_bootstrap_ci95_upper"] - block_bootstrap["block_bootstrap_mean_score"]], fmt="o", color=PURPLE, ecolor="#E3D9EA", elinewidth=1.45, capsize=2.2, markersize=3.8, markeredgecolor="white", markeredgewidth=0.45, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([compact_alternative(value) for value in block_bootstrap["alternative"]])
    ax.invert_yaxis()
    ax.set_xlabel("Month-block score")
    clean_axis(ax, axis="x")
    panel_label(ax, "(c)")

    ax = fig.add_subplot(grid[1, 1])
    labels = ["TOPSIS\nEq.", "TOPSIS\nEnt.", "VIKOR\nEnt."]
    colors = [TEAL if rank == 1 else GRAY for rank in baselines["efa_ahp_rank_of_baseline_top"]]
    bars = ax.bar(labels, baselines["spearman_with_efa_ahp"], color=colors, alpha=0.92, edgecolor="white", linewidth=0.45, zorder=3)
    ax.set_ylabel("Spearman vs EFA-AHP")
    ax.set_ylim(0, 1.05)
    clean_axis(ax, axis="y")
    panel_label(ax, "(d)")
    for bar, rank in zip(bars, baselines["efa_ahp_rank_of_baseline_top"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"rank {int(rank)}", ha="center", fontsize=7.0)
    save(fig, 6)


def robustness_policy_figure() -> None:
    efa = pd.read_csv(ROBUSTNESS / "efa_robustness_summary.csv")
    models = pd.read_csv(POLICY / "reward_model_comparison.csv")
    selection = pd.read_csv(POLICY / "policy_selection_protocol.csv").iloc[0]
    policy = pd.read_csv(POLICY / "policy_test_ope.csv")
    distribution = pd.read_csv(POLICY / "policy_action_distribution.csv")
    selected_policy = str(selection["selected_policy"])
    distribution = distribution[distribution["policy"] == selected_policy].sort_values("selected_pct", ascending=False).head(6)

    fig = plt.figure(figsize=(7.2, 5.75))
    grid = GridSpec(2, 2, figure=fig, hspace=0.47, wspace=0.38)
    width = 0.33

    ax = fig.add_subplot(grid[0, 0])
    labels = efa["variant"].str.replace("_", " ", regex=False).map(lambda value: wrap(value, 15))
    x = np.arange(len(efa))
    ax.bar(x - width / 2, efa["mean_abs_tucker_congruence"], width=width, color=DEEP_BLUE, alpha=0.92, edgecolor="white", linewidth=0.45, label="Tucker congruence", zorder=3)
    ax.bar(x + width / 2, efa["spearman_vs_full_rank_fixed_reference"], width=width, color=GREEN, alpha=0.92, edgecolor="white", linewidth=0.45, label="Fixed rank corr.", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Diagnostic value")
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    for idx, factors in enumerate(efa["parallel_retained_factors_95pct"]):
        ax.text(idx, 1.035, f"k={int(factors)}", ha="center", fontsize=7.1, color=MUTED_INK)
    clean_axis(ax, axis="y")
    panel_label(ax, "(a)")

    ax = fig.add_subplot(grid[0, 1])
    model_labels = models["model"].map({"linear_state_action_ridge": "Linear", "poly2_state_action_ridge": "Polynomial", "rff_state_action_ridge": "RFF"})
    x = np.arange(len(models))
    ax.bar(x - width / 2, models["validation_r2"], width=width, color=DEEP_BLUE, alpha=0.92, edgecolor="white", linewidth=0.45, label="Validation", zorder=3)
    ax.bar(x + width / 2, models["test_r2"], width=width, color=TEAL, alpha=0.92, edgecolor="white", linewidth=0.45, label="Test", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel("Reward-model $R^2$")
    ax.set_ylim(0.50, 0.62)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    clean_axis(ax, axis="y")
    panel_label(ax, "(b)")

    ax = fig.add_subplot(grid[1, 0])
    shown = policy[policy["policy"].isin(["static_efa_ahp_best", selected_policy])].copy()
    shown["order"] = shown["policy"].map({"static_efa_ahp_best": 0, selected_policy: 1})
    shown = shown.sort_values("order")
    y = np.arange(len(shown))
    ax.errorbar(shown["doubly_robust_value"], y, xerr=[shown["doubly_robust_value"] - shown["dr_ci95_lower"], shown["dr_ci95_upper"] - shown["doubly_robust_value"]], fmt="o", color=DEEP_BLUE, ecolor="#D7E3EE", elinewidth=1.45, capsize=2.2, markersize=3.8, markeredgecolor="white", markeredgewidth=0.45, zorder=3, label="DR, 95% block CI")
    ax.scatter(shown["direct_method_value"], y, marker="D", s=18, color=TEAL, zorder=4, label="Direct method")
    ax.axvline(0, color=GRAY, lw=0.72, alpha=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels([compact_policy(value) for value in shown["policy"]])
    ax.invert_yaxis()
    ax.set_xlabel("Locked-test reward estimate")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    clean_axis(ax, axis="x")
    panel_label(ax, "(c)")

    ax = fig.add_subplot(grid[1, 1])
    y = np.arange(len(distribution))
    ax.barh(y, distribution["selected_pct"], color=GREEN, alpha=0.92, edgecolor="white", linewidth=0.45, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([compact_alternative(value) for value in distribution["alternative"]])
    ax.invert_yaxis()
    ax.set_xlabel("Selected states (%)")
    ax.set_xlim(0, distribution["selected_pct"].max() * 1.16)
    clean_axis(ax, axis="x")
    panel_label(ax, "(d)")
    for idx, value in enumerate(distribution["selected_pct"]):
        ax.text(value + 0.35, idx, f"{value:.1f}", va="center", fontsize=7.0)
    save(fig, 7)


def main() -> None:
    required_static = [
        STATIC / "factor_loadings.csv",
        STATIC / "efa_eigenvalues.csv",
        STATIC / "scenario_rankings.csv",
        STATIC / "expert_group_weights.csv",
        SUPPLEMENT / "physical_plausibility_audit.csv",
        SUPPLEMENT / "bootstrap_rank_probabilities.csv",
        SUPPLEMENT / "mcdm_baseline_summary.csv",
        ROBUSTNESS / "efa_robustness_summary.csv",
        ROBUSTNESS / "time_block_bootstrap_rank_uncertainty.csv",
        POLICY / "reward_model_comparison.csv",
        POLICY / "policy_test_ope.csv",
    ]
    missing = [path for path in required_static if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run scripts/run_all.py before building the result figures. Missing: "
            f"{', '.join(str(path) for path in missing)}"
        )

    source_plausibility_figure()
    efa_diagnostics_figure()
    ahp_weights_figure()
    static_ranking_figure()
    uncertainty_baselines_figure()
    robustness_policy_figure()
    print(f"Figures 2--7 written to: {OUTPUT}")


if __name__ == "__main__":
    main()
