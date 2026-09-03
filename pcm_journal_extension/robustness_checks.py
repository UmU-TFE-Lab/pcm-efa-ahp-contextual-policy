"""Reviewer-required supplementary experiments for the PCM EFA-AHP study.

This script turns the main reviewer requests into auditable artifacts:

1. Data-generation pipeline figure.
2. Variable-level provenance table.
3. Source-based physical validation figures.
4. EFA robustness checks under reduced criteria, alternative aggregation, and
   time-block bootstrap.
5. Support-aware policy summary that foregrounds conservative offline policies.

The figures are generated as polished vector PDF files using matplotlib and
seaborn so they can be included directly in the LaTeX manuscript.
"""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path
import sys
import textwrap
from typing import Iterable

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import seaborn as sns
import numpy as np
import pandas as pd

from publication_style import (
    AMBER,
    AXIS,
    BLUE,
    DEEP_BLUE,
    GRAY,
    GREEN,
    GRID,
    INK,
    MUTED_INK,
    ORANGE,
    PALE_BLUE,
    PALE_GRAY,
    PALE_GREEN,
    PALE_ORANGE,
    PURPLE,
    RED,
    TEAL,
    clean_axis as publication_clean_axis,
    configure_publication_style,
)


ROOT = Path(__file__).resolve().parents[1]
STATIC_MODULE = ROOT / "pcm_efa_ahp"
RL_MODULE = ROOT / "pcm_efa_ahp_rl"
for module_path in (STATIC_MODULE, RL_MODULE):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from pcm_efa_ahp_study import (  # noqa: E402
    CRITERIA,
    DEFAULT_INPUT,
    FACTOR_LABELS,
    SCENARIO_TARGET_WEIGHTS,
    add_pcm_criteria,
    apply_preprocessing,
    bartlett_sphericity_statistic,
    build_alternatives,
    efa_from_standardized,
    efa_model_diagnostics,
    kmo_measure,
    parallel_analysis,
    score_with_fitted_efa,
    score_scenarios,
    standardize,
    winsorize_frame,
)
from mcdm_uncertainty import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    physical_plausibility_audit,
    spearman_from_ranks,
    load_static_study,
)


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "reviewer_required"
POLICY_SUMMARY = ROOT / "outputs" / "contextual_policy" / "locked_evaluation" / "policy_summary.csv"


LIGHT_BLUE = PALE_BLUE
LIGHT_GREEN = PALE_GREEN
LIGHT_ORANGE = PALE_ORANGE
LIGHT_GRAY = PALE_GRAY


def configure_plot_style() -> None:
    configure_publication_style(base_font=8.6)
    mpl.rcParams["axes.titlelocation"] = "left"
    mpl.rcParams["figure.constrained_layout.use"] = True


configure_plot_style()


def safe_name(value: str) -> str:
    return value.replace(" | ", "--").replace("_", " ")


def wrap_text(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False)


def wrapped(text: str, width: int) -> str:
    return "\n".join(wrap_text(text, width))


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def clean_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    publication_clean_axis(ax, axis=grid_axis)


def annotate_bars(ax: plt.Axes, fmt: str = "{:.2f}", fontsize: float = 8.2) -> None:
    xmin, xmax = ax.get_xlim()
    xspan = xmax - xmin
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    for patch in ax.patches:
        height = patch.get_height()
        width = patch.get_width()
        if abs(height) < 1e-12 and abs(width) < 1e-12:
            continue
        if abs(height) >= abs(width):
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_y() + patch.get_height()
            offset = 0.045 * yspan if abs(patch.get_height()) < 1e-12 else (0.025 * yspan if patch.get_height() >= 0 else -0.045 * yspan)
            ax.text(x, y + offset, fmt.format(height), ha="center", va="bottom" if height >= 0 else "top", fontsize=fontsize, color=INK)
        else:
            x = patch.get_x() + patch.get_width()
            y = patch.get_y() + patch.get_height() / 2
            offset = 0.012 * xspan if width >= 0 else -0.012 * xspan
            ax.text(x + offset, y, fmt.format(width), ha="left" if width >= 0 else "right", va="center", fontsize=fontsize, color=INK)


def save_pipeline_figure(output_dir: Path, raw: pd.DataFrame) -> None:
    path = output_dir / "data_generation_pipeline.pdf"
    fig, ax = plt.subplots(figsize=(11.6, 6.25))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x: float, y: float, w: float, h: float, heading: str, body: str, face: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.78,
            edgecolor=AXIS,
            facecolor=face,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(x + 0.018, y + h - 0.035, heading, ha="left", va="top", fontsize=9.4, fontweight="bold", color=INK, zorder=3)
        ax.text(x + 0.018, y + h - 0.077, wrapped(body, 28), ha="left", va="top", fontsize=7.6, color=MUTED_INK, linespacing=1.22, zorder=3)

    def arrow(start: tuple[float, float], end: tuple[float, float], rad: float = 0.0) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=0.82,
                color=GRAY,
                alpha=0.92,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=4,
                shrinkB=4,
                zorder=1,
            )
        )

    n_records = len(raw)
    n_fields = len(raw.columns)
    n_alternatives = raw[
        ["pcm_type", "system_type", "encapsulation_type"]
    ].drop_duplicates().shape[0]
    boxes = {
        "exp": (0.045, 0.75, 0.245, 0.145, "PCM wallboard experiment", "DSC, board geometry, surface temperatures, indoor air, weather, decrement factor.", LIGHT_BLUE),
        "ate": (0.045, 0.56, 0.245, 0.145, "ATE multiscale model", "MD, RVE-FEM, Mori-Tanaka, effective conductivity, phase-change constraints.", LIGHT_BLUE),
        "ida": (0.045, 0.37, 0.245, 0.145, "IDA-ICE building model", "Geometry, zones, constructions, weather, HVAC setpoints, baseline and PCM runs.", LIGHT_BLUE),
        "phys": (0.045, 0.18, 0.245, 0.145, "Physical constraints", "Bounds, unit checks, energy-balance logic, admissible operating ranges.", LIGHT_ORANGE),
        "sample": (0.385, 0.68, 0.245, 0.145, "Scenario sampling", "Hourly states mapped to material-system-encapsulation alternatives.", LIGHT_GREEN),
        "feature": (0.385, 0.445, 0.245, 0.145, "Feature construction", "Derived criteria, larger-is-better transforms, normalization by mass and area.", LIGHT_GREEN),
        "csv": (
            0.385,
            0.21,
            0.245,
            0.145,
            "Fused scenario table",
            f"{n_records:,} indexed records, {n_fields} source fields, {n_alternatives} observed alternatives.",
            LIGHT_GREEN,
        ),
        "efa": (0.72, 0.565, 0.235, 0.145, "EFA-AHP decision layer", "Latent factors, expert weights, static ranking, uncertainty checks.", LIGHT_GRAY),
        "policy": (0.72, 0.325, 0.235, 0.145, "Contextual policy layer", "Offline support-aware recommendation using operating-state features.", LIGHT_GRAY),
    }
    for values in boxes.values():
        box(*values)

    for key in ["exp", "ate", "ida"]:
        x, y, w, h, *_ = boxes[key]
        sx, sy = x + w, y + h / 2
        tx, ty = boxes["sample"][0], boxes["sample"][1] + boxes["sample"][3] / 2
        arrow((sx, sy), (tx, ty), rad=0.05 if key == "exp" else (-0.04 if key == "ida" else 0))
    x, y, w, h, *_ = boxes["phys"]
    arrow((x + w, y + h / 2), (boxes["feature"][0], boxes["feature"][1] + boxes["feature"][3] / 2), rad=-0.06)
    arrow((boxes["sample"][0] + boxes["sample"][2] / 2, boxes["sample"][1]), (boxes["feature"][0] + boxes["feature"][2] / 2, boxes["feature"][1] + boxes["feature"][3]), rad=0)
    arrow((boxes["feature"][0] + boxes["feature"][2] / 2, boxes["feature"][1]), (boxes["csv"][0] + boxes["csv"][2] / 2, boxes["csv"][1] + boxes["csv"][3]), rad=0)
    arrow((boxes["csv"][0] + boxes["csv"][2], boxes["csv"][1] + boxes["csv"][3] * 0.72), (boxes["efa"][0], boxes["efa"][1] + boxes["efa"][3] / 2), rad=0.12)
    arrow((boxes["csv"][0] + boxes["csv"][2], boxes["csv"][1] + boxes["csv"][3] * 0.28), (boxes["policy"][0], boxes["policy"][1] + boxes["policy"][3] / 2), rad=-0.08)
    arrow((boxes["efa"][0] + boxes["efa"][2] / 2, boxes["efa"][1]), (boxes["policy"][0] + boxes["policy"][2] / 2, boxes["policy"][1] + boxes["policy"][3]), rad=0)

    ax.text(
        0.045,
        0.065,
        "Dataset status: experiment-informed, physics-constrained scenario table; records are not independent laboratory measurements.",
        fontsize=8.1,
        color=GRAY,
        ha="left",
        va="center",
    )
    save_figure(fig, path)


def build_variable_provenance(raw_columns: Iterable[str]) -> pd.DataFrame:
    rows = [
        ("timestamp", "time index", "sampled/indexed", "IDA-ICE/weather time base", "Hourly scenario index; not an independent measurement label."),
        ("pcm_type", "configuration", "sampled/design", "Study design", "PCM material family."),
        ("system_type", "configuration", "sampled/design", "Study design", "Application scenario."),
        ("encapsulation_type", "configuration", "sampled/design", "Study design", "Encapsulation or stabilization option."),
        ("air_temperature_c", "weather/context", "simulated/measured boundary", "Weather station and IDA-ICE boundary", "Operating state feature."),
        ("relative_humidity_pct", "weather/context", "simulated/measured boundary", "Weather station and IDA-ICE boundary", "Operating state feature."),
        ("wind_speed_mps", "weather/context", "simulated/measured boundary", "Weather station and IDA-ICE boundary", "Operating state feature."),
        ("cloud_cover_pct", "weather/context", "simulated/measured boundary", "Weather/IDA-ICE boundary", "Operating state feature."),
        ("solar_irradiance_wm2", "weather/context", "simulated/measured boundary", "Weather/IDA-ICE boundary", "Operating state feature."),
        ("inlet_fluid_temp_c", "operation", "simulated/sampled", "System scenario and IDA-ICE/thermal model", "Operating state feature."),
        ("melting_point_c", "thermophysical", "calibrated/sampled", "DSC, MD, literature constraints", "Material property."),
        ("latent_heat_kjkg", "thermophysical", "calibrated/sampled", "DSC and literature constraints", "Material property."),
        ("thermal_conductivity_wmk", "thermophysical", "simulated/calibrated", "MD, RVE-FEM, Mori-Tanaka, literature", "Material property."),
        ("density_kgm3", "thermophysical", "sampled/constrained", "Material literature and PCM type", "Material property."),
        ("specific_heat_jkgk", "thermophysical", "calibrated/sampled", "DSC and literature constraints", "Material property."),
        ("pcm_mass_kg", "geometry/system", "sampled/design", "Wallboard/module design and system scenario", "Normalization variable."),
        ("surface_area_m2", "geometry/system", "sampled/design", "Wallboard/module design and system scenario", "Normalization variable."),
        ("pcm_thickness_mm", "geometry/system", "sampled/design", "Wallboard/module design", "Geometry constraint."),
        ("mass_flow_rate_kgs", "operation/system", "simulated/sampled", "System scenario", "Heat-transfer process variable."),
        ("cycle_number", "operation", "sampled/indexed", "Scenario design", "Durability and operating-history state."),
        ("degradation_factor", "durability", "calculated/constrained", "Cycle degradation model and physical bounds", "Decision criterion."),
        ("temp_difference_c", "operation", "calculated", "Operating temperature minus melting range reference", "Phase-state driver."),
        ("phase_fraction", "operation", "calculated/constrained", "Phase-change model", "Bounded state variable."),
        ("heat_transfer_coeff_wm2k", "heat transfer", "simulated/calculated", "Heat-transfer model and system scenario", "Process variable."),
        ("heat_flux_wm2", "heat transfer", "simulated/calculated", "IDA-ICE and heat-transfer calculation", "Signed process variable."),
        ("stored_energy_kj", "storage process", "simulated/calculated", "Thermal model and energy balance", "Base performance variable."),
        ("energy_input_kj", "storage process", "simulated/calculated", "Thermal model and energy balance", "Efficiency denominator."),
        ("charging_time_min", "process performance", "simulated/calculated", "Thermal response model", "Cost-type criterion before reversal."),
        ("discharging_time_min", "process performance", "simulated/calculated", "Thermal response model", "Cost-type criterion before reversal."),
        ("energy_loss_pct", "performance", "calculated/constrained", "Energy-balance calculation", "Cost-type criterion before reversal."),
        ("state_of_charge_pct", "operation/performance", "calculated/constrained", "Phase-state and storage model", "Decision criterion."),
        ("cooling_load_offset_pct", "building performance", "simulated/calculated", "IDA-ICE baseline vs PCM-enhanced comparison", "Decision criterion."),
        ("thermal_storage_efficiency_pct", "performance", "calculated/constrained", "Useful stored/released energy over input", "Decision criterion."),
    ]
    known = {row[0] for row in rows}
    for column in raw_columns:
        if column not in known:
            rows.append((column, "raw variable", "review", "CSV", "Not yet classified."))
    for criterion in CRITERIA:
        if criterion.name not in known:
            rows.append((criterion.name, "constructed criterion", "normalized/calculated", criterion.source, criterion.description))
    return pd.DataFrame(rows, columns=["variable", "category", "provenance_type", "primary_source_or_transformation", "manuscript_use_or_note"])


def save_bar_chart(
    output_path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    y_label: str,
    zero_line: bool = True,
    colors_list: list[str] | None = None,
    note: str | None = None,
) -> None:
    del title, note
    fig, ax = plt.subplots(figsize=(6.8, 4.25))
    x = np.arange(len(values))
    bar_colors = colors_list or [BLUE] * len(values)
    ax.bar(x, values, color=bar_colors, width=0.60, edgecolor="white", linewidth=0.55, alpha=0.92, zorder=3)
    if zero_line:
        ax.axhline(0, color=GRAY, linewidth=0.72, alpha=0.82)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels([wrapped(label, 12) for label in labels])
    ymin, ymax = min(values + [0]), max(values + [0])
    padding = max((ymax - ymin) * 0.18, 1.0)
    ax.set_ylim(ymin - padding, ymax + padding)
    yspan = (ymax + padding) - (ymin - padding)
    for idx, value in enumerate(values):
        if value > 0:
            y_text, va = value + 0.032 * yspan, "bottom"
        elif value < 0:
            y_text, va = value - 0.032 * yspan, "top"
        else:
            y_text, va = 0 + 0.075 * yspan, "bottom"
        ax.text(idx, y_text, f"{value:.1f}", ha="center", va=va, fontsize=8.3, color=INK)
    clean_axis(ax, grid_axis="y")
    save_figure(fig, output_path)


def save_grouped_bar_chart(output_path: Path, data: pd.DataFrame) -> None:
    df = data.copy()
    df["label"] = [wrapped(f"{r.case}\n{r.metric}", 18) for r in df.itertuples()]
    x = np.arange(len(df))
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(9.4, 4.5),
        width_ratios=[1.55, 1.0],
        gridspec_kw={"wspace": 0.12},
    )
    width = 0.34
    ax1.bar(x - width / 2, df["baseline_value"], width=width, color=DEEP_BLUE, alpha=0.92, label="Baseline", edgecolor="white", linewidth=0.5, zorder=3)
    ax1.bar(x + width / 2, df["pcm_value"], width=width, color=ORANGE, alpha=0.92, label="PCM", edgecolor="white", linewidth=0.5, zorder=3)
    ax1.set_yscale("log")
    ax1.set_ylabel("Reported value (log scale)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["label"], rotation=0, ha="center")
    ax1.legend(frameon=False, loc="upper left", ncols=2)
    clean_axis(ax1, grid_axis="y")

    def compact_value(value: float) -> str:
        return f"{value:.2f}" if abs(value) < 100 else f"{value:.0f}"

    for i, row in enumerate(df.itertuples()):
        ax1.text(i - width / 2, row.baseline_value * 1.08, compact_value(row.baseline_value), ha="center", va="bottom", fontsize=7.4, color=INK)
        ax1.text(i + width / 2, row.pcm_value * 1.08, compact_value(row.pcm_value), ha="center", va="bottom", fontsize=7.4, color=INK)

    reduction_colors = [GREEN if value >= 0 else RED for value in df["computed_reduction_pct"]]
    ax2.barh(x, df["computed_reduction_pct"], color=reduction_colors, alpha=0.92, edgecolor="white", linewidth=0.5, height=0.60, zorder=3)
    ax2.set_yticks(x)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Reduction (%)")
    ax2.invert_yaxis()
    clean_axis(ax2, grid_axis="x")
    ax2.set_xlim(0, max(df["computed_reduction_pct"]) * 1.28)
    for i, value in enumerate(df["computed_reduction_pct"]):
        ax2.text(value + 0.25, i, f"{value:.2f}", ha="left", va="center", fontsize=8.1, color=INK)
    save_figure(fig, output_path)


def save_range_chart(output_path: Path, audit: pd.DataFrame) -> None:
    selected = audit.head(10).copy()
    fig, ax = plt.subplots(figsize=(7.7, 4.9))
    y = np.arange(len(selected))
    for idx, row in enumerate(selected.itertuples()):
        lo, hi = float(row.expected_min), float(row.expected_max)
        obs_lo, obs_hi = float(row.observed_min), float(row.observed_max)
        span = hi - lo if hi > lo else 1.0
        nlo = np.clip((obs_lo - lo) / span, 0, 1)
        nhi = np.clip((obs_hi - lo) / span, 0, 1)
        ax.plot([0, 1], [idx, idx], color=AXIS, alpha=0.44, linewidth=4.1, solid_capstyle="round", zorder=1)
        ax.plot([nlo, nhi], [idx, idx], color=DEEP_BLUE, alpha=0.86, linewidth=4.1, solid_capstyle="round", zorder=2)
        ax.scatter([nlo, nhi], [idx, idx], s=15, color=DEEP_BLUE, edgecolor="white", linewidth=0.45, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([value.replace("_", " ") for value in selected["variable"]])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Observed range within admissible interval")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["min", "25%", "50%", "75%", "max"])
    expected_proxy = plt.Line2D([0], [0], color=AXIS, alpha=0.44, lw=4.1, solid_capstyle="round")
    observed_proxy = plt.Line2D([0], [0], color=DEEP_BLUE, alpha=0.86, lw=4.1, solid_capstyle="round")
    ax.legend(
        [expected_proxy, observed_proxy],
        ["Admissible interval", "Observed range"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncols=2,
        borderaxespad=0,
    )
    clean_axis(ax, grid_axis="x")
    save_figure(fig, output_path)


def save_policy_chart(output_path: Path, policy: pd.DataFrame) -> None:
    selected_names = policy.loc[policy["recommended_for_main_reporting"], "policy"].astype(str).tolist()
    names = list(dict.fromkeys([
        "static_efa_ahp_best",
        *selected_names,
        "sampled_knn_argmax",
        "hybrid_linear_knn_30_70",
        "hybrid_poly_knn_50_50",
    ]))
    df = policy[policy["policy"].isin(names)].copy()
    df["policy"] = pd.Categorical(df["policy"], names, ordered=True)
    df = df.sort_values("policy")
    labels = df["policy"].astype(str).str.replace("_", " ", regex=False).map(lambda s: wrapped(s, 24))
    fig, ax = plt.subplots(figsize=(7.9, 4.75))
    y = np.arange(len(df))
    lower = df["doubly_robust_value"] - df["dr_ci95_lower"]
    upper = df["dr_ci95_upper"] - df["doubly_robust_value"]
    ax.errorbar(
        df["doubly_robust_value"],
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color=DEEP_BLUE,
        ecolor=DEEP_BLUE,
        elinewidth=1.0,
        capsize=2.4,
        markersize=4.8,
        label="Doubly robust, 95% block CI",
        zorder=3,
    )
    ax.scatter(df["direct_method_value"], y, marker="D", s=20, color=TEAL, label="Direct model estimate", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color=GRAY, linewidth=0.7, alpha=0.7)
    ax.set_xlabel("Test-set EFA-AHP reward estimate")
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncols=2, borderaxespad=0)
    clean_axis(ax, grid_axis="x")
    save_figure(fig, output_path)


def source_based_physical_evidence(output_dir: Path, physical_audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp = pd.DataFrame(
        [
            {"board_type": "Gypsum", "decrement_factor_change_pct_vs_gypsum": 0.0, "source": "Reference board"},
            {"board_type": "cPCM", "decrement_factor_change_pct_vs_gypsum": 5.3, "source": "Pilot experiment"},
            {"board_type": "mPCM60", "decrement_factor_change_pct_vs_gypsum": -7.7, "source": "Pilot experiment"},
            {"board_type": "mPCM82", "decrement_factor_change_pct_vs_gypsum": -13.0, "source": "Pilot experiment"},
        ]
    )
    exp.to_csv(output_dir / "experimental_calibration_evidence.csv", index=False)
    save_bar_chart(
        output_dir / "experimental_decrement_factor_evidence.pdf",
        "Experimental wallboard calibration evidence",
        exp["board_type"].tolist(),
        exp["decrement_factor_change_pct_vs_gypsum"].tolist(),
        "Decrement factor change vs gypsum (%)",
        colors_list=[GRAY, ORANGE, GREEN, GREEN],
        note="Negative values indicate reduced temperature attenuation factor relative to gypsum.",
    )

    ida = pd.DataFrame(
        [
            {"case": "Stockholm", "metric": "Fuel heating purchased energy (kWh)", "baseline_value": 9150.0, "pcm_value": 8523.0, "reported_reduction_pct": 7.376},
            {"case": "Stockholm", "metric": "Electric cooling (kWh)", "baseline_value": 17706.0, "pcm_value": 17274.0, "reported_reduction_pct": np.nan},
            {"case": "Stockholm", "metric": "Heating peak demand (kW)", "baseline_value": 8.12, "pcm_value": 7.81, "reported_reduction_pct": np.nan},
            {"case": "Svalbard", "metric": "Fuel heating purchased energy (kWh)", "baseline_value": 34349.0, "pcm_value": 31163.0, "reported_reduction_pct": np.nan},
        ]
    )
    ida["computed_reduction_pct"] = (ida["baseline_value"] - ida["pcm_value"]) / ida["baseline_value"] * 100.0
    ida["reported_minus_computed_pct_points"] = ida["reported_reduction_pct"] - ida["computed_reduction_pct"]
    ida["audit_note"] = np.where(
        ida["reported_reduction_pct"].notna() & (ida["reported_minus_computed_pct_points"].abs() > 0.25),
        "review source values or reported percent",
        "internally consistent or no reported percent",
    )
    ida.to_csv(output_dir / "idaice_source_level_load_comparison.csv", index=False)
    save_grouped_bar_chart(output_dir / "idaice_load_comparison.pdf", ida)
    save_range_chart(output_dir / "physical_range_check.pdf", physical_audit)
    return exp, ida


def engineering_score(alternatives: pd.DataFrame) -> pd.DataFrame:
    score_columns = [f"{factor}_score" for factor in FACTOR_LABELS]
    weights = np.asarray(SCENARIO_TARGET_WEIGHTS["engineering_default"], dtype=float)
    weights = weights / weights.sum()
    frame = alternatives.copy()
    frame["final_score"] = np.einsum(
        "ij,j->i", frame[score_columns].to_numpy(dtype=float), weights, optimize=True
    )
    frame["rank"] = frame["final_score"].rank(ascending=False, method="min").astype(int)
    return frame.sort_values(["rank", "alternative"]).reset_index(drop=True)


def tucker_congruence(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    if denominator <= 0:
        return np.nan
    return float(np.sum(left * right) / denominator)


def match_refitted_factors(
    reference_loadings: pd.DataFrame,
    variant_loadings: pd.DataFrame,
    variant_name: str,
) -> pd.DataFrame:
    """Match refitted factors to reference factors by absolute Tucker congruence."""
    common = [criterion for criterion in variant_loadings.index if criterion in reference_loadings.index]
    reference = reference_loadings.loc[common].to_numpy(dtype=float)
    variant = variant_loadings.loc[common].to_numpy(dtype=float)
    congruence = np.zeros((reference.shape[1], variant.shape[1]), dtype=float)
    for reference_idx in range(reference.shape[1]):
        for variant_idx in range(variant.shape[1]):
            congruence[reference_idx, variant_idx] = tucker_congruence(
                reference[:, reference_idx], variant[:, variant_idx]
            )

    best_assignment: tuple[int, ...] | None = None
    best_score = -np.inf
    for assignment in permutations(range(reference.shape[1]), variant.shape[1]):
        score = float(sum(abs(congruence[reference_idx, variant_idx]) for variant_idx, reference_idx in enumerate(assignment)))
        if score > best_score:
            best_score = score
            best_assignment = assignment
    if best_assignment is None:
        raise RuntimeError("Could not match refitted factors to the reference model")

    rows = []
    for variant_idx, reference_idx in enumerate(best_assignment):
        signed = float(congruence[reference_idx, variant_idx])
        rows.append(
            {
                "variant": variant_name,
                "variant_factor": variant_loadings.columns[variant_idx],
                "matched_reference_factor": reference_loadings.columns[reference_idx],
                "signed_tucker_congruence": signed,
                "absolute_tucker_congruence": abs(signed),
                "sign_alignment": 1 if signed >= 0 else -1,
                "n_common_criteria": len(common),
            }
        )
    return pd.DataFrame(rows)


def fit_reference_measurement_model(raw: pd.DataFrame, criteria_names: list[str]) -> dict[str, object]:
    data = add_pcm_criteria(raw)
    clipped, bounds = winsorize_frame(data, criteria_names)
    z, standardization = standardize(clipped, criteria_names)
    efa = efa_from_standardized(z, n_factors=len(FACTOR_LABELS), factor_labels=FACTOR_LABELS)
    return {
        "data": data,
        "bounds": bounds,
        "standardization": standardization,
        "standardized": z,
        "efa": efa,
    }


def fixed_reference_projection(
    data: pd.DataFrame,
    criteria_names: list[str],
    reference: dict[str, object],
) -> pd.DataFrame:
    """Score a variant on the fixed four-factor reference axes.

    Omitted criteria are assigned zero after reference standardization, which is
    equivalent to setting them to their reference means. This preserves factor
    identity and AHP-weight meaning without pretending that a lower-dimensional
    refitted model still contains all four original factors.
    """
    reference_criteria = list(reference["efa"]["coefficients"].index)
    z_available = apply_preprocessing(
        data,
        criteria_names,
        reference["bounds"],
        reference["standardization"],
    )
    z_reference = pd.DataFrame(0.0, index=data.index, columns=reference_criteria)
    z_reference.loc[:, criteria_names] = z_available.loc[:, criteria_names]
    factor_scores = score_with_fitted_efa(z_reference, reference["efa"]["coefficients"])
    return engineering_score(build_alternatives(data, factor_scores))


def efa_variant(
    raw: pd.DataFrame,
    criteria_names: list[str],
    variant_name: str,
    reference: dict[str, object],
    full_rank: pd.DataFrame,
    aggregate_first: bool = False,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = add_pcm_criteria(raw)
    if aggregate_first:
        group_cols = ["pcm_type", "system_type", "encapsulation_type"]
        numeric = [col for col in data.columns if col not in group_cols + ["timestamp"] and pd.api.types.is_numeric_dtype(data[col])]
        data = data.groupby(group_cols, as_index=False)[numeric].mean()

    clipped, _ = winsorize_frame(data, criteria_names)
    z, _ = standardize(clipped, criteria_names)
    parallel = parallel_analysis(z, n_repeats=100)
    n_factors = max(1, int(parallel["retain_by_parallel_95pct"].sum()))
    factor_names = [f"Refit factor {idx + 1}" for idx in range(n_factors)]
    efa = efa_from_standardized(z, n_factors=n_factors, factor_labels=factor_names)
    corr = efa["correlation"].to_numpy(dtype=float)
    eigenvalues = efa["eigenvalues"]
    cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    diagnostics = efa_model_diagnostics(z, efa, n_factors=n_factors, parallel_repeats=100)
    chi2, dof = bartlett_sphericity_statistic(corr, len(z))
    alignment = match_refitted_factors(reference["efa"]["loadings"], efa["loadings"], variant_name)

    ranking = fixed_reference_projection(data, criteria_names, reference)
    comparison = ranking[["alternative", "rank"]].merge(full_rank, on="alternative", how="left")
    spearman = spearman_from_ranks(comparison["rank"], comparison["full_rank"])
    matched_reference = set(alignment["matched_reference_factor"])
    unmatched = [factor for factor in FACTOR_LABELS if factor not in matched_reference]
    summary = {
        "variant": variant_name,
        "n_records": len(z),
        "n_criteria": len(criteria_names),
        "parallel_retained_factors_95pct": n_factors,
        "kmo": kmo_measure(corr),
        "bartlett_chi_square": chi2,
        "bartlett_dof": dof,
        "retained_cumulative_variance": float(cumulative[n_factors - 1]),
        "mean_communality": diagnostics["summary"]["mean_communality"],
        "max_uniqueness": diagnostics["summary"]["max_uniqueness"],
        "rmsr_offdiag": diagnostics["summary"]["rmsr_offdiag"],
        "mean_abs_tucker_congruence": float(alignment["absolute_tucker_congruence"].mean()),
        "min_abs_tucker_congruence": float(alignment["absolute_tucker_congruence"].min()),
        "unmatched_reference_factors": "; ".join(unmatched) if unmatched else "none",
        "decision_projection_method": "fixed reference loadings; omitted criteria set to reference mean",
        "spearman_vs_full_rank_fixed_reference": spearman,
        "top_alternative_fixed_reference": ranking.iloc[0]["alternative"],
        "top_score_fixed_reference": float(ranking.iloc[0]["final_score"]),
        "refitted_factor_ranking_reported": False,
        "criteria": "; ".join(criteria_names),
    }
    parallel = parallel.assign(variant=variant_name)
    return summary, ranking, alignment, parallel


def time_block_bootstrap(study, output_dir: Path, n_bootstrap: int = 300, random_state: int = 2026) -> pd.DataFrame:
    weights = np.asarray(SCENARIO_TARGET_WEIGHTS["engineering_default"], dtype=float)
    weights = weights / weights.sum()
    reward = np.einsum(
        "ij,j->i",
        study.factor_scores[FACTOR_LABELS].to_numpy(dtype=float),
        weights,
        optimize=True,
    )
    data = study.raw[["timestamp", "pcm_type", "system_type", "encapsulation_type"]].copy()
    data["alternative"] = data["pcm_type"] + " | " + data["system_type"] + " | " + data["encapsulation_type"]
    data["period"] = data["timestamp"].dt.to_period("M").astype(str)
    data["reward"] = reward
    monthly = data.groupby(["alternative", "period"])["reward"].agg(["sum", "count"]).reset_index()
    alternatives = sorted(data["alternative"].unique())
    periods = sorted(data["period"].unique())
    sum_mat = monthly.pivot(index="alternative", columns="period", values="sum").reindex(index=alternatives, columns=periods).fillna(0.0).to_numpy()
    cnt_mat = monthly.pivot(index="alternative", columns="period", values="count").reindex(index=alternatives, columns=periods).fillna(0.0).to_numpy()
    rng = np.random.default_rng(random_state)
    boot_scores = np.zeros((n_bootstrap, len(alternatives)), dtype=float)
    for b in range(n_bootstrap):
        sampled = rng.integers(0, len(periods), size=len(periods))
        sums = sum_mat[:, sampled].sum(axis=1)
        counts = cnt_mat[:, sampled].sum(axis=1)
        boot_scores[b] = sums / np.maximum(counts, 1.0)
    rank_samples = np.argsort(np.argsort(-boot_scores, axis=1), axis=1) + 1
    result = pd.DataFrame(
        {
            "alternative": alternatives,
            "block_bootstrap_mean_score": boot_scores.mean(axis=0),
            "block_bootstrap_ci95_lower": np.quantile(boot_scores, 0.025, axis=0),
            "block_bootstrap_ci95_upper": np.quantile(boot_scores, 0.975, axis=0),
            "block_bootstrap_top1_probability": (rank_samples == 1).mean(axis=0),
            "block_bootstrap_top3_probability": (rank_samples <= 3).mean(axis=0),
        }
    ).sort_values(["block_bootstrap_mean_score", "alternative"], ascending=[False, True])
    result.to_csv(output_dir / "time_block_bootstrap_rank_uncertainty.csv", index=False)
    return result


def save_efa_robustness_chart(output_path: Path, robustness: pd.DataFrame) -> None:
    df = robustness.copy()
    df["variant_label"] = df["variant"].str.replace("_", " ", regex=False).map(lambda s: wrapped(s, 18))
    long = df.melt(
        id_vars="variant_label",
        value_vars=["mean_abs_tucker_congruence", "spearman_vs_full_rank_fixed_reference"],
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].map(
        {
            "mean_abs_tucker_congruence": "Matched-factor congruence",
            "spearman_vs_full_rank_fixed_reference": "Fixed-reference rank correlation",
        }
    )
    fig, ax = plt.subplots(figsize=(6.9, 4.35))
    sns.barplot(data=long, x="variant_label", y="value", hue="metric", palette=[DEEP_BLUE, GREEN], ax=ax, edgecolor="white", linewidth=0.55, alpha=0.92)
    ax.set_ylim(0, 1.14)
    ax.set_xlabel("")
    ax.set_ylabel("Diagnostic value")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncols=2, borderaxespad=0)
    for patch in ax.patches:
        value = patch.get_height()
        if np.isfinite(value) and value > 1e-9:
            ax.text(patch.get_x() + patch.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=7.9, color=INK)
    clean_axis(ax, grid_axis="y")
    save_figure(fig, output_path)


def efa_robustness(output_dir: Path, raw: pd.DataFrame, full_ranking: pd.DataFrame) -> pd.DataFrame:
    full_criteria = [criterion.name for criterion in CRITERIA]
    reduced = [
        "storage_density_kjkg",
        "charge_power_kjmin",
        "thermal_storage_efficiency_pct",
        "cooling_load_offset_pct",
        "state_of_charge_pct",
        "loss_score_pct",
        "degradation_factor",
        "charge_speed_score_min",
    ]
    variants = [
        ("full_record_11criteria", full_criteria, False),
        ("reduced_record_8criteria", reduced, False),
        ("alternative_mean_11criteria", full_criteria, True),
    ]
    reference = fit_reference_measurement_model(raw, full_criteria)
    rows = []
    rankings = []
    alignments = []
    parallel_frames = []
    full = full_ranking[["alternative", "rank"]].rename(columns={"rank": "full_rank"})
    for name, criteria, aggregate_first in variants:
        summary, ranking, alignment, parallel = efa_variant(
            raw,
            criteria,
            name,
            reference,
            full,
            aggregate_first,
        )
        rows.append(summary)
        ranking.insert(0, "variant", name)
        rankings.append(ranking)
        alignments.append(alignment)
        parallel_frames.append(parallel)
    robustness = pd.DataFrame(rows)
    ranking_all = pd.concat(rankings, ignore_index=True)
    robustness.to_csv(output_dir / "efa_robustness_summary.csv", index=False)
    ranking_all.to_csv(output_dir / "efa_robustness_rankings.csv", index=False)
    pd.concat(alignments, ignore_index=True).to_csv(output_dir / "efa_factor_alignment.csv", index=False)
    pd.concat(parallel_frames, ignore_index=True).to_csv(output_dir / "efa_robustness_parallel_analysis.csv", index=False)
    save_efa_robustness_chart(output_dir / "efa_robustness_summary.pdf", robustness)
    return robustness


def support_aware_policy(output_dir: Path, policy_summary: Path = POLICY_SUMMARY) -> pd.DataFrame:
    if not policy_summary.exists():
        raise FileNotFoundError(f"Missing locked policy summary: {policy_summary}")
    policy = pd.read_csv(policy_summary)
    if "recommended_for_main_reporting" not in policy.columns:
        raise ValueError("Policy summary does not contain validation-based policy selection")
    policy.to_csv(output_dir / "support_aware_policy_recommendation.csv", index=False)
    save_policy_chart(output_dir / "support_aware_policy_comparison.pdf", policy)
    return policy


def write_summary_report(
    output_dir: Path,
    provenance: pd.DataFrame,
    exp: pd.DataFrame,
    ida: pd.DataFrame,
    robustness: pd.DataFrame,
    block_bootstrap: pd.DataFrame,
    policy: pd.DataFrame,
) -> None:
    top_block = block_bootstrap.head(3)
    recommended = policy[policy["recommended_for_main_reporting"]].head(3)
    lines = [
        "# Reviewer-required experiment package",
        "",
        "## Added artifacts",
        "",
        "- Data-generation pipeline figure: `data_generation_pipeline.pdf`",
        "- Variable-level provenance table: `variable_level_provenance.csv`",
        "- Experimental calibration evidence: `experimental_decrement_factor_evidence.pdf`",
        "- IDA-ICE load comparison: `idaice_load_comparison.pdf`",
        "- Physical range check: `physical_range_check.pdf`",
        "- EFA robustness summary: `efa_robustness_summary.csv` and `efa_robustness_summary.pdf`",
        "- Time-block bootstrap ranking uncertainty: `time_block_bootstrap_rank_uncertainty.csv`",
        "- Support-aware policy recommendation: `support_aware_policy_recommendation.csv` and `support_aware_policy_comparison.pdf`",
        "",
        "## Key audit findings",
        "",
        f"- Variable-level provenance rows: {len(provenance)}.",
        "- Experimental evidence is source-based. Raw wallboard temperature logs are still needed for true curve-level validation.",
        "- IDA-ICE source-level values support energy/load reduction trends, but at least one reported percentage should be checked.",
    ]
    flags = ida[ida["audit_note"].str.contains("review", case=False, na=False)]
    for row in flags.itertuples():
        lines.append(
            f"- Review flag: {row.case} {row.metric}: computed reduction={row.computed_reduction_pct:.3f}%, "
            f"reported={row.reported_reduction_pct:.3f}%."
        )
    lines.extend(["", "## EFA robustness", ""])
    for row in robustness.itertuples():
        lines.append(
            f"- {row.variant}: parallel-analysis factors={row.parallel_retained_factors_95pct}, "
            f"mean matched-factor congruence={row.mean_abs_tucker_congruence:.3f}, "
            f"fixed-reference Spearman={row.spearman_vs_full_rank_fixed_reference:.3f}, "
            f"top={row.top_alternative_fixed_reference}."
        )
    lines.extend(["", "## Time-block bootstrap, top alternatives", ""])
    for row in top_block.itertuples():
        lines.append(
            f"- {row.alternative}: mean={row.block_bootstrap_mean_score:.3f}, "
            f"P(top1)={row.block_bootstrap_top1_probability:.3f}, P(top3)={row.block_bootstrap_top3_probability:.3f}."
        )
    lines.extend(["", "## Support-aware policy candidates", ""])
    for row in recommended.itertuples():
        lines.append(
            f"- {row.policy}: validation-selected; test DR={row.doubly_robust_value:.3f} "
            f"(95% block CI {row.dr_ci95_lower:.3f} to {row.dr_ci95_upper:.3f}), "
            f"ESS={row.importance_weight_ess:.1f}."
        )
    (output_dir / "reviewer_required_experiments_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    output_dir: Path = OUTPUT_DIR,
    input_csv: Path = DEFAULT_INPUT,
    policy_summary: Path = POLICY_SUMMARY,
    n_bootstrap: int = 300,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    study = load_static_study(input_csv)
    raw = study.raw.copy()
    save_pipeline_figure(output_dir, raw)

    provenance = build_variable_provenance(raw.columns)
    provenance.to_csv(output_dir / "variable_level_provenance.csv", index=False)

    physical_audit = physical_plausibility_audit(study)
    physical_audit.to_csv(output_dir / "physical_plausibility_audit_reviewer.csv", index=False)
    exp, ida = source_based_physical_evidence(output_dir, physical_audit)

    full_ranking = study.rankings[study.rankings["scenario"] == "engineering_default"].copy()
    robustness = efa_robustness(output_dir, raw, full_ranking)
    block_bootstrap = time_block_bootstrap(study, output_dir, n_bootstrap=n_bootstrap)
    policy = support_aware_policy(output_dir, policy_summary)

    write_summary_report(output_dir, provenance, exp, ida, robustness, block_bootstrap, policy)
    return {
        "provenance": provenance,
        "experimental_evidence": exp,
        "idaice_evidence": ida,
        "efa_robustness": robustness,
        "time_block_bootstrap": block_bootstrap,
        "support_aware_policy": policy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--policy-summary", type=Path, default=POLICY_SUMMARY)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    result = run(args.output, args.input, args.policy_summary, args.bootstrap)
    print("Reviewer-required experiments completed")
    print(f"Outputs written to: {args.output}")
    print()
    print("EFA robustness")
    print(
        result["efa_robustness"][[
            "variant",
            "parallel_retained_factors_95pct",
            "mean_abs_tucker_congruence",
            "spearman_vs_full_rank_fixed_reference",
            "top_alternative_fixed_reference",
        ]].round(4).to_string(index=False)
    )
    print()
    print("Support-aware recommended policies")
    cols = [
        "policy",
        "direct_method_value",
        "doubly_robust_value",
        "dr_ci95_lower",
        "dr_ci95_upper",
        "importance_weight_ess",
        "recommended_for_main_reporting",
    ]
    print(result["support_aware_policy"][cols].head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
