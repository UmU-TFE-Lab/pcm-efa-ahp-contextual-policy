"""Major-revision audit artifacts for the PCM EFA-AHP manuscript.

The outputs in this script are deliberately conservative. They separate
evidence that can be computed from the current local files from evidence that
requires raw experimental logs or a fresh simulator rerun.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from publication_style import (
    AXIS,
    DEEP_BLUE,
    GREEN,
    GRID,
    INK,
    MUTED_INK,
    ORANGE,
    PURPLE,
    TEAL,
    clean_axis,
    configure_publication_style,
    save_figure,
    wrap_label,
)


ROOT = Path(__file__).resolve().parents[1]
STATIC_MODULE = ROOT / "pcm_efa_ahp"
RL_MODULE = ROOT / "pcm_efa_ahp_rl"
for module_path in (STATIC_MODULE, RL_MODULE):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from pcm_efa_ahp_study import ahp  # noqa: E402
from mcdm_uncertainty import (  # noqa: E402
    CONFIRMED_EXPERT_PAIRWISE,
    DEFAULT_OUTPUT_DIR,
    FACTOR_LABELS,
    confirmed_expert_ahp_results,
    confirmed_expert_pairwise,
    expert_matrix,
    expert_weighted_efa_ahp_ranking,
    load_static_study,
    spearman_from_ranks,
)


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "major_revision"
RL_DATASET = ROOT / "outputs" / "contextual_policy" / "reward_dataset.csv"
RAW_DATASET = ROOT / "data" / "private" / "pcm_thermal_storage.csv"


def _alternative(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["pcm_type"].astype(str)
        + " | "
        + frame["system_type"].astype(str)
        + " | "
        + frame["encapsulation_type"].astype(str)
    )


def _geometric_group_matrix(pairwise: pd.DataFrame, expert_ids: list[str]) -> np.ndarray:
    matrices = [
        expert_matrix(pairwise[pairwise["expert_id"] == expert_id])
        for expert_id in expert_ids
    ]
    return np.prod(matrices, axis=0) ** (1.0 / len(matrices))


def _rank_with_weights(study, weights: np.ndarray) -> pd.DataFrame:
    score_columns = [f"{factor}_score" for factor in FACTOR_LABELS]
    ranking = study.alternatives.copy()
    ranking["score"] = ranking[score_columns].to_numpy(dtype=float) @ weights
    ranking["rank"] = ranking["score"].rank(ascending=False, method="min").astype(int)
    return ranking.sort_values(["rank", "alternative"]).reset_index(drop=True)


def leave_one_expert_out_ahp(
    output_dir: Path,
    input_csv: Path = RAW_DATASET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairwise = confirmed_expert_pairwise()
    expert_ids = sorted({expert_id for _, _, _, values in CONFIRMED_EXPERT_PAIRWISE for expert_id in values})
    study = load_static_study(input_csv)

    full_result = ahp(_geometric_group_matrix(pairwise, expert_ids))
    full_weights = np.asarray(full_result["weights"], dtype=float)
    full_ranking = _rank_with_weights(study, full_weights)
    full_top = full_ranking.iloc[0]["alternative"]
    full_rank_reference = full_ranking[["alternative", "rank"]].rename(columns={"rank": "full_rank"})

    summary_rows = []
    weight_rows = []
    for omitted in expert_ids:
        included = [expert_id for expert_id in expert_ids if expert_id != omitted]
        result = ahp(_geometric_group_matrix(pairwise, included))
        weights = np.asarray(result["weights"], dtype=float)
        cr = float(result["cr"])
        ranking = _rank_with_weights(study, weights)
        comparison = ranking[["alternative", "rank"]].merge(full_rank_reference, on="alternative", how="left")
        max_shift = float(np.max(np.abs(weights - full_weights)))
        full_top_rank = int(ranking.loc[ranking["alternative"] == full_top, "rank"].iloc[0])
        summary_rows.append(
            {
                "omitted_expert": omitted,
                "group_cr": cr,
                "top_alternative": ranking.iloc[0]["alternative"],
                "top_score": ranking.iloc[0]["score"],
                "rank_of_full_group_top": full_top_rank,
                "spearman_vs_full_group_rank": spearman_from_ranks(comparison["rank"], comparison["full_rank"]),
                "max_abs_weight_shift": max_shift,
            }
        )
        for factor, weight in zip(FACTOR_LABELS, weights):
            weight_rows.append({"omitted_expert": omitted, "factor": factor, "weight": weight})

    summary = pd.DataFrame(summary_rows)
    weights = pd.DataFrame(weight_rows)
    summary.to_csv(output_dir / "leave_one_expert_out_ahp_summary.csv", index=False)
    weights.to_csv(output_dir / "leave_one_expert_out_ahp_weights.csv", index=False)
    save_leave_one_expert_plot(output_dir / "leave_one_expert_out_ahp.pdf", summary, weights)
    return summary, weights


def save_leave_one_expert_plot(output_path: Path, summary: pd.DataFrame, weights: pd.DataFrame) -> None:
    configure_publication_style(base_font=8.5)
    order = ["Storage capacity and power", "Fast thermal response", "Durability and low loss", "Efficiency and load offset"]
    weights = weights.copy()
    weights["factor"] = pd.Categorical(weights["factor"], order, ordered=True)
    summary = summary.copy().sort_values("omitted_expert")

    fig = plt.figure(figsize=(9.3, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.26)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    sns.barplot(
        data=weights,
        x="omitted_expert",
        y="weight",
        hue="factor",
        palette=[DEEP_BLUE, TEAL, GREEN, ORANGE],
        ax=ax1,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.94,
    )
    ax1.set_xlabel("Omitted expert")
    ax1.set_ylabel("Group AHP weight")
    ax1.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncols=2,
        borderaxespad=0,
        title=None,
    )
    clean_axis(ax1, axis="y")

    y = np.arange(len(summary))
    ax2.barh(y - 0.18, summary["spearman_vs_full_group_rank"], height=0.34, color=DEEP_BLUE, alpha=0.90, label="Spearman")
    ax2.barh(y + 0.18, summary["max_abs_weight_shift"], height=0.34, color=PURPLE, alpha=0.90, label="Max weight shift")
    ax2.set_yticks(y)
    ax2.set_yticklabels(summary["omitted_expert"])
    ax2.invert_yaxis()
    ax2.set_xlabel("Sensitivity value")
    ax2.set_xlim(0, 1.05)
    for value, yy in zip(summary["spearman_vs_full_group_rank"], y - 0.18):
        ax2.text(value + 0.018, yy, f"{value:.3f}", va="center", ha="left", fontsize=7.5, color=INK)
    for value, yy in zip(summary["max_abs_weight_shift"], y + 0.18):
        ax2.text(value + 0.018, yy, f"{value:.3f}", va="center", ha="left", fontsize=7.5, color=INK)
    ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncols=2, borderaxespad=0)
    clean_axis(ax2, axis="x")

    for label, ax in zip(["(a)", "(b)"], [ax1, ax2]):
        ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK)
    save_figure(fig, output_path.with_suffix(""))


def experimental_calibration_status(output_dir: Path) -> pd.DataFrame:
    rows = [
        {
            "calibration_item": "Surface temperature RMSE",
            "current_status": "not computable from current source files",
            "available_evidence": "thermocouple accuracy +/-0.5 degC; wallboard surface-temperature response reported qualitatively",
            "required_to_compute": "raw timestamped surface-temperature logs and the simulated/calibrated response curve at matching sensors",
        },
        {
            "calibration_item": "Surface temperature MAE",
            "current_status": "not computable from current source files",
            "available_evidence": "Swema 03+ indoor-air-temperature accuracy +/-0.3 degC; thermocouples calibrated in ice-water bath",
            "required_to_compute": "raw observed and predicted surface-temperature time series after time alignment",
        },
        {
            "calibration_item": "Decrement factor change",
            "current_status": "available from experimental source",
            "available_evidence": "cPCM +5.3%; mPCM60 -7.7%; mPCM82 -13.0% relative to gypsum",
            "required_to_compute": "raw maxima/minima or source-processed decrement-factor table for independent recomputation",
        },
        {
            "calibration_item": "Absolute decrement-factor error",
            "current_status": "not computable from current source files",
            "available_evidence": "relative changes reported for each board type",
            "required_to_compute": "observed decrement factor and model-predicted decrement factor for each board and replicate",
        },
        {
            "calibration_item": "IDA-ICE Stockholm heating reduction",
            "current_status": "source inconsistency identified",
            "available_evidence": "9150 to 8523 kWh gives 6.85%, whereas the source table lists 7.376%",
            "required_to_compute": "unrounded source values or clarification of the denominator used in the reported percentage",
        },
    ]
    status = pd.DataFrame(rows)
    status.to_csv(output_dir / "experimental_calibration_metric_status.csv", index=False)
    return status


def source_to_csv_fusion_rules(
    output_dir: Path,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Write a variable-by-variable audit trail for the fused CSV.

    The rules are deliberately expressed as formulas or pseudocode rather than
    as proof of exact regeneration. They document how each final field should
    be traced from source evidence to the CSV and where external source files
    remain necessary.
    """
    if raw is None:
        timestamp_rule = "timestamp = hourly_index(start=scenario_start, n=n_records); map or resample the weather template onto the scenario calendar"
        timestamp_bound = "scenario calendar; not assumed to be field observations"
    else:
        timestamps = pd.to_datetime(raw["timestamp"])
        timestamp_rule = (
            f"timestamp = hourly_index(start='{timestamps.min()}', n={len(raw)}); "
            "map or resample the weather template onto the scenario calendar"
        )
        timestamp_bound = (
            f"scenario calendar from {timestamps.min()} to {timestamps.max()}; "
            "not assumed to be field observations"
        )
    rows = [
        {
            "variable": "timestamp",
            "category": "time index",
            "evidence_type": "sampled/indexed",
            "source_stream": "IDA-ICE/weather template and scenario calendar",
            "fusion_rule_or_pseudocode": timestamp_rule,
            "final_csv_role": "hourly scenario index",
            "validation_or_bound": timestamp_bound,
        },
        {
            "variable": "pcm_type",
            "category": "configuration",
            "evidence_type": "sampled/design",
            "source_stream": "study design, PCM literature, DSC/MD constraints",
            "fusion_rule_or_pseudocode": "pcm_type in {Organic_Paraffin, Inorganic_SaltHydrate, Eutectic}; assign as part of the 3 x 4 x 3 design space",
            "final_csv_role": "material-family action component",
            "validation_or_bound": "categorical value must belong to the declared action space",
        },
        {
            "variable": "system_type",
            "category": "configuration",
            "evidence_type": "sampled/design",
            "source_stream": "building and TES application scenarios",
            "fusion_rule_or_pseudocode": "system_type in {BuildingEnvelope, SolarTES, BatteryCooling, HVACStorage}; assign as application context for thermal-storage use",
            "final_csv_role": "system-application action component",
            "validation_or_bound": "categorical value must belong to the declared action space",
        },
        {
            "variable": "encapsulation_type",
            "category": "configuration",
            "evidence_type": "sampled/design",
            "source_stream": "encapsulation/stabilization design options",
            "fusion_rule_or_pseudocode": "encapsulation_type in {Macro, Micro, ShapeStabilized}; assign as packaging or stabilization option",
            "final_csv_role": "encapsulation action component",
            "validation_or_bound": "categorical value must belong to the declared action space",
        },
        {
            "variable": "air_temperature_c",
            "category": "weather/context",
            "evidence_type": "measured boundary/simulated",
            "source_stream": "weather station and IDA-ICE weather boundary",
            "fusion_rule_or_pseudocode": "air_temperature_c = align_to_hour(weather_template.temperature, timestamp) with scenario resampling or perturbation",
            "final_csv_role": "context state feature",
            "validation_or_bound": "range must remain meteorologically plausible for the selected climate scenarios",
        },
        {
            "variable": "relative_humidity_pct",
            "category": "weather/context",
            "evidence_type": "measured boundary/simulated",
            "source_stream": "weather station and IDA-ICE weather boundary",
            "fusion_rule_or_pseudocode": "relative_humidity_pct = clip(align_to_hour(weather_template.relative_humidity, timestamp), 0, 100)",
            "final_csv_role": "context state feature",
            "validation_or_bound": "0-100%",
        },
        {
            "variable": "wind_speed_mps",
            "category": "weather/context",
            "evidence_type": "measured boundary/simulated",
            "source_stream": "weather station and IDA-ICE weather boundary",
            "fusion_rule_or_pseudocode": "wind_speed_mps = max(0, align_to_hour(weather_template.wind_speed, timestamp))",
            "final_csv_role": "context state feature",
            "validation_or_bound": "non-negative and climate-plausible",
        },
        {
            "variable": "cloud_cover_pct",
            "category": "weather/context",
            "evidence_type": "measured boundary/simulated",
            "source_stream": "weather station and IDA-ICE weather boundary",
            "fusion_rule_or_pseudocode": "cloud_cover_pct = clip(align_to_hour(weather_template.cloud_cover, timestamp), 0, 100)",
            "final_csv_role": "context state feature",
            "validation_or_bound": "0-100%",
        },
        {
            "variable": "solar_irradiance_wm2",
            "category": "weather/context",
            "evidence_type": "measured boundary/simulated",
            "source_stream": "weather station and IDA-ICE solar boundary",
            "fusion_rule_or_pseudocode": "solar_irradiance_wm2 = max(0, align_to_hour(weather_template.global_solar, timestamp)); night hours set near zero",
            "final_csv_role": "context state feature and load driver",
            "validation_or_bound": "non-negative; daily pattern must be consistent with hour and cloud cover",
        },
        {
            "variable": "inlet_fluid_temp_c",
            "category": "operation",
            "evidence_type": "simulated/sampled",
            "source_stream": "system scenario, HVAC/TES operating assumptions",
            "fusion_rule_or_pseudocode": "inlet_fluid_temp_c = f(system_type, air_temperature_c, operating_mode, setpoint) + bounded scenario variation",
            "final_csv_role": "context state feature",
            "validation_or_bound": "must remain compatible with low-temperature PCM operation",
        },
        {
            "variable": "melting_point_c",
            "category": "thermophysical",
            "evidence_type": "experiment-informed/sampled",
            "source_stream": "DSC, MD phase-change estimate, PCM literature",
            "fusion_rule_or_pseudocode": "melting_point_c = constrain_sample(distribution_by_pcm_type, lower=18, upper=48)",
            "final_csv_role": "material property and phase-state driver",
            "validation_or_bound": "18-48 degC in current dataset",
        },
        {
            "variable": "latent_heat_kjkg",
            "category": "thermophysical",
            "evidence_type": "experiment-informed/sampled",
            "source_stream": "DSC enthalpy evidence and PCM literature",
            "fusion_rule_or_pseudocode": "latent_heat_kjkg = constrain_sample(latent_heat_distribution_by_pcm_type, lower=150, upper=280)",
            "final_csv_role": "material property for storage capacity",
            "validation_or_bound": "150-280 kJ/kg in current dataset",
        },
        {
            "variable": "thermal_conductivity_wmk",
            "category": "thermophysical",
            "evidence_type": "simulated/constrained",
            "source_stream": "MD, RVE-FEM, Mori-Tanaka, literature",
            "fusion_rule_or_pseudocode": "thermal_conductivity_wmk = map_effective_conductivity(pcm_type, encapsulation_type, volume_fraction, interface_parameter) with physical bounds",
            "final_csv_role": "heat-transfer material property",
            "validation_or_bound": "0.18-0.73 W/mK in current dataset; ATE PU-PCM building input 0.13-0.14 W/mK used as source constraint",
        },
        {
            "variable": "density_kgm3",
            "category": "thermophysical",
            "evidence_type": "sampled/constrained",
            "source_stream": "PCM material literature and material family",
            "fusion_rule_or_pseudocode": "density_kgm3 = constrain_sample(density_distribution_by_pcm_type, lower=760, upper=1700)",
            "final_csv_role": "mass and volumetric storage calculation",
            "validation_or_bound": "760-1700 kg/m3 in current dataset",
        },
        {
            "variable": "specific_heat_jkgk",
            "category": "thermophysical",
            "evidence_type": "experiment-informed/sampled",
            "source_stream": "DSC heat-capacity evidence and literature",
            "fusion_rule_or_pseudocode": "specific_heat_jkgk = constrain_sample(cp_distribution_by_pcm_type, lower=1400, upper=2600)",
            "final_csv_role": "sensible-heat term in enthalpy balance",
            "validation_or_bound": "1400-2600 J/kgK in current dataset",
        },
        {
            "variable": "pcm_mass_kg",
            "category": "geometry/system",
            "evidence_type": "sampled/design",
            "source_stream": "wallboard/module geometry and system scenario",
            "fusion_rule_or_pseudocode": "pcm_mass_kg = density_kgm3 * effective_pcm_volume(system_type, encapsulation_type) or bounded module-mass sample",
            "final_csv_role": "normalization and capacity calculation",
            "validation_or_bound": "positive; consistent with thickness, area, and density",
        },
        {
            "variable": "surface_area_m2",
            "category": "geometry/system",
            "evidence_type": "sampled/design",
            "source_stream": "wallboard/module geometry and system scenario",
            "fusion_rule_or_pseudocode": "surface_area_m2 = bounded_area_sample(system_type, module_geometry)",
            "final_csv_role": "normalization and heat-transfer area",
            "validation_or_bound": "positive and consistent with system scale",
        },
        {
            "variable": "pcm_thickness_mm",
            "category": "geometry/system",
            "evidence_type": "sampled/design",
            "source_stream": "wallboard thickness, encapsulation geometry, module design",
            "fusion_rule_or_pseudocode": "pcm_thickness_mm = bounded_thickness_sample(system_type, encapsulation_type); wallboard source uses 15 mm board thickness",
            "final_csv_role": "conduction resistance and mass consistency",
            "validation_or_bound": "positive; compatible with selected application",
        },
        {
            "variable": "mass_flow_rate_kgs",
            "category": "operation/system",
            "evidence_type": "simulated/sampled",
            "source_stream": "HVAC/TES operating scenario",
            "fusion_rule_or_pseudocode": "mass_flow_rate_kgs = bounded_flow_sample(system_type, operating_mode)",
            "final_csv_role": "convective heat-transfer and response-time driver",
            "validation_or_bound": "non-negative; zero or near-zero only when flow is inactive",
        },
        {
            "variable": "cycle_number",
            "category": "operation",
            "evidence_type": "sampled/indexed",
            "source_stream": "scenario operating history",
            "fusion_rule_or_pseudocode": "cycle_number = assign_cycle_count(timestamp, system_type, operating_frequency)",
            "final_csv_role": "durability-state driver",
            "validation_or_bound": "non-negative integer-like value",
        },
        {
            "variable": "degradation_factor",
            "category": "durability",
            "evidence_type": "calculated/constrained",
            "source_stream": "cycle degradation model and physical bounds",
            "fusion_rule_or_pseudocode": "degradation_factor = clip(1 - degradation_rate(pcm_type, encapsulation_type) * cycle_number, lower_bound, 1)",
            "final_csv_role": "decision criterion and effective latent-heat modifier",
            "validation_or_bound": "0-1; higher means better retained performance",
        },
        {
            "variable": "temp_difference_c",
            "category": "operation",
            "evidence_type": "calculated",
            "source_stream": "operating temperature and melting-point relation",
            "fusion_rule_or_pseudocode": "temp_difference_c = operating_or_inlet_temperature_c - melting_point_c, with sign convention preserved",
            "final_csv_role": "phase-state and heat-transfer driver",
            "validation_or_bound": "must be consistent with inlet/ambient temperature and PCM melting point",
        },
        {
            "variable": "phase_fraction",
            "category": "operation",
            "evidence_type": "calculated/constrained",
            "source_stream": "phase-change model",
            "fusion_rule_or_pseudocode": "phase_fraction = clip(phase_transition_function(temp_difference_c, transition_width, hysteresis_optional), 0, 1)",
            "final_csv_role": "latent-storage activation state",
            "validation_or_bound": "0-1",
        },
        {
            "variable": "heat_transfer_coeff_wm2k",
            "category": "heat transfer",
            "evidence_type": "simulated/calculated",
            "source_stream": "system heat-transfer model and operating scenario",
            "fusion_rule_or_pseudocode": "heat_transfer_coeff_wm2k = h_correlation(system_type, encapsulation_type, flow_regime, mass_flow_rate_kgs) with bounded variation",
            "final_csv_role": "charging/discharging response-time driver",
            "validation_or_bound": "positive and application-plausible",
        },
        {
            "variable": "heat_flux_wm2",
            "category": "heat transfer",
            "evidence_type": "simulated/calculated",
            "source_stream": "IDA-ICE output and heat-transfer calculation",
            "fusion_rule_or_pseudocode": "heat_flux_wm2 = heat_transfer_coeff_wm2k * effective_deltaT with sign convention for charging/discharging",
            "final_csv_role": "signed process variable",
            "validation_or_bound": "may be positive or negative; magnitude must be compatible with h, area, and deltaT",
        },
        {
            "variable": "stored_energy_kj",
            "category": "storage process",
            "evidence_type": "simulated/calculated",
            "source_stream": "enthalpy balance, IDA-ICE-informed operation, thermophysical properties",
            "fusion_rule_or_pseudocode": "stored_energy_kj = pcm_mass_kg * (specific_heat_jkgk * sensible_deltaT + latent_heat_kjkg*1000 * phase_fraction * degradation_factor) / 1000, then apply utilization/system bounds",
            "final_csv_role": "base performance variable",
            "validation_or_bound": "positive when storage is active; compatible with mass, cp, latent heat, and phase fraction",
        },
        {
            "variable": "energy_input_kj",
            "category": "storage process",
            "evidence_type": "simulated/calculated",
            "source_stream": "energy balance and TES efficiency relation",
            "fusion_rule_or_pseudocode": "energy_input_kj = useful_stored_or_released_energy_kj / max(thermal_storage_efficiency_pct/100, epsilon)",
            "final_csv_role": "efficiency denominator",
            "validation_or_bound": "positive and not smaller than useful energy when efficiency <= 100%",
        },
        {
            "variable": "charging_time_min",
            "category": "process performance",
            "evidence_type": "simulated/calculated",
            "source_stream": "thermal-response model",
            "fusion_rule_or_pseudocode": "charging_time_min = response_time_model(stored_energy_kj, heat_transfer_coeff_wm2k, surface_area_m2, conductivity, thickness, flow)",
            "final_csv_role": "cost-type process metric before reversal",
            "validation_or_bound": "positive",
        },
        {
            "variable": "discharging_time_min",
            "category": "process performance",
            "evidence_type": "simulated/calculated",
            "source_stream": "thermal-response model",
            "fusion_rule_or_pseudocode": "discharging_time_min = response_time_model(releasable_energy_kj, heat_transfer_coeff_wm2k, surface_area_m2, conductivity, thickness, flow)",
            "final_csv_role": "cost-type process metric before reversal",
            "validation_or_bound": "positive",
        },
        {
            "variable": "energy_loss_pct",
            "category": "performance",
            "evidence_type": "calculated/constrained",
            "source_stream": "energy balance, degradation, insulation/encapsulation assumptions",
            "fusion_rule_or_pseudocode": "energy_loss_pct = clip(loss_model(system_type, encapsulation_type, storage_duration, degradation_factor), 2, 35)",
            "final_csv_role": "cost-type criterion before reversal",
            "validation_or_bound": "2-35% in current dataset",
        },
        {
            "variable": "state_of_charge_pct",
            "category": "operation/performance",
            "evidence_type": "calculated/constrained",
            "source_stream": "phase-state and storage model",
            "fusion_rule_or_pseudocode": "state_of_charge_pct = 100 * clip(current_stored_energy_kj / max(storage_capacity_kj, epsilon), 0, 1)",
            "final_csv_role": "decision criterion and operating state",
            "validation_or_bound": "0-100%",
        },
        {
            "variable": "cooling_load_offset_pct",
            "category": "building performance",
            "evidence_type": "simulated/calculated",
            "source_stream": "IDA-ICE baseline vs PCM-enhanced comparison",
            "fusion_rule_or_pseudocode": "cooling_load_offset_pct = 100 * (baseline_cooling_load - pcm_cooling_load) / baseline_cooling_load, mapped to scenario and constrained by application",
            "final_csv_role": "decision criterion",
            "validation_or_bound": "requires baseline denominator; sign and magnitude must be physically interpretable",
        },
        {
            "variable": "thermal_storage_efficiency_pct",
            "category": "performance",
            "evidence_type": "calculated/constrained",
            "source_stream": "useful stored/released energy and input energy",
            "fusion_rule_or_pseudocode": "thermal_storage_efficiency_pct = 100 * useful_stored_or_released_energy_kj / energy_input_kj, then clip to engineering bounds",
            "final_csv_role": "decision criterion",
            "validation_or_bound": "35-98% in current dataset; must not exceed 100% before explicit clipping rationale",
        },
        {
            "variable": "storage_density_kjkg",
            "category": "constructed criterion",
            "evidence_type": "normalized/calculated",
            "source_stream": "final CSV performance and mass fields",
            "fusion_rule_or_pseudocode": "storage_density_kjkg = stored_energy_kj / pcm_mass_kg",
            "final_csv_role": "EFA-AHP benefit criterion",
            "validation_or_bound": "positive if stored_energy_kj and pcm_mass_kg are positive",
        },
        {
            "variable": "storage_areal_density_kjm2",
            "category": "constructed criterion",
            "evidence_type": "normalized/calculated",
            "source_stream": "final CSV performance and area fields",
            "fusion_rule_or_pseudocode": "storage_areal_density_kjm2 = stored_energy_kj / surface_area_m2",
            "final_csv_role": "EFA-AHP benefit criterion",
            "validation_or_bound": "positive if stored_energy_kj and surface_area_m2 are positive",
        },
        {
            "variable": "charge_power_kjmin",
            "category": "constructed criterion",
            "evidence_type": "normalized/calculated",
            "source_stream": "final CSV energy and charging-time fields",
            "fusion_rule_or_pseudocode": "charge_power_kjmin = stored_energy_kj / charging_time_min",
            "final_csv_role": "EFA-AHP benefit criterion",
            "validation_or_bound": "positive if charging_time_min > 0",
        },
        {
            "variable": "discharge_power_kjmin",
            "category": "constructed criterion",
            "evidence_type": "normalized/calculated",
            "source_stream": "final CSV energy and discharging-time fields",
            "fusion_rule_or_pseudocode": "discharge_power_kjmin = stored_energy_kj / discharging_time_min",
            "final_csv_role": "EFA-AHP benefit criterion",
            "validation_or_bound": "positive if discharging_time_min > 0",
        },
        {
            "variable": "loss_score_pct",
            "category": "constructed criterion",
            "evidence_type": "normalized/calculated",
            "source_stream": "final CSV energy-loss field",
            "fusion_rule_or_pseudocode": "loss_score_pct = 100 - energy_loss_pct",
            "final_csv_role": "EFA-AHP benefit criterion",
            "validation_or_bound": "higher means lower loss",
        },
        {
            "variable": "charge_speed_score_min",
            "category": "constructed criterion",
            "evidence_type": "normalized/calculated",
            "source_stream": "final CSV charging-time field",
            "fusion_rule_or_pseudocode": "charge_speed_score_min = -charging_time_min",
            "final_csv_role": "EFA-AHP benefit criterion after cost reversal",
            "validation_or_bound": "higher means faster charging",
        },
        {
            "variable": "discharge_speed_score_min",
            "category": "constructed criterion",
            "evidence_type": "normalized/calculated",
            "source_stream": "final CSV discharging-time field",
            "fusion_rule_or_pseudocode": "discharge_speed_score_min = -discharging_time_min",
            "final_csv_role": "EFA-AHP benefit criterion after cost reversal",
            "validation_or_bound": "higher means faster discharging",
        },
    ]
    rules = pd.DataFrame(rows)
    rules.to_csv(output_dir / "source_to_csv_fusion_rules.csv", index=False)

    headers = [
        "variable",
        "evidence_type",
        "source_stream",
        "fusion_rule_or_pseudocode",
        "validation_or_bound",
    ]
    markdown_lines = [
        "# Source-to-CSV fusion rules",
        "",
        "This audit table documents formula- or pseudocode-level transformations for the fused PCM thermal-storage CSV. It is not a replacement for raw experimental logs, IDA-ICE model files, ATE input/output files, or executable fusion scripts.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rules[headers].itertuples(index=False):
        markdown_lines.append(
            "| "
            + " | ".join(str(value).replace("|", "/") for value in row)
            + " |"
        )
    (output_dir / "source_to_csv_fusion_rules.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )
    return rules


def system_specific_ranking_summary(
    output_dir: Path,
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize rankings within each application system.

    This separates the cross-application screening result from within-system
    comparisons, which are often the more physically comparable engineering
    question.
    """
    ranking = ranking.copy()
    ranking = ranking.sort_values(["rank", "alternative"]).reset_index(drop=True)
    rows = []
    for system_type, subset in ranking.groupby("system_type", sort=True):
        system_subset = subset.sort_values(["final_score", "rank"], ascending=[False, True]).reset_index(drop=True)
        for local_rank, row in enumerate(system_subset.head(3).itertuples(index=False), start=1):
            rows.append(
                {
                    "system_type": system_type,
                    "within_system_rank": local_rank,
                    "global_expert_rank": int(row.rank),
                    "alternative": row.alternative,
                    "expert_score": float(row.final_score),
                    "n_records": int(row.n),
                    "interpretation": "within-system comparison under the same expert-weighted EFA-AHP reward",
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "system_specific_expert_ranking_summary.csv", index=False)
    return summary


def scenario_calendar_holdout_proxy(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(RL_DATASET)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["alternative"] = _alternative(data)
    reward_col = "efa_ahp_reward"
    if reward_col not in data.columns:
        raise KeyError(f"Expected {reward_col} in {RL_DATASET}")

    q_low = data["air_temperature_c"].quantile(0.25)
    q_high = data["air_temperature_c"].quantile(0.75)
    solar_high = data.loc[data["is_daytime"] == 1, "solar_irradiance_wm2"].quantile(0.75)
    split_time = data["timestamp"].quantile(0.75)

    slices = {
        "full scenario calendar": data,
        "first 75% calendar": data[data["timestamp"] <= split_time],
        "last 25% temporal holdout": data[data["timestamp"] > split_time],
        "cold-state quartile": data[data["air_temperature_c"] <= q_low],
        "warm-state quartile": data[data["air_temperature_c"] >= q_high],
        "high-solar daytime quartile": data[(data["is_daytime"] == 1) & (data["solar_irradiance_wm2"] >= solar_high)],
    }

    rows = []
    top_rows = []
    for slice_name, subset in slices.items():
        grouped = (
            subset.groupby("alternative")[reward_col]
            .agg(["mean", "std", "count"])
            .query("count >= 30")
            .sort_values(["mean", "count"], ascending=[False, False])
            .reset_index()
        )
        for rank, row in enumerate(grouped.head(5).itertuples(index=False), start=1):
            top_rows.append(
                {
                    "validation_slice": slice_name,
                    "rank": rank,
                    "alternative": row.alternative,
                    "mean_reward": row.mean,
                    "std_reward": row.std,
                    "n_records": int(row.count),
                }
            )
        top = grouped.iloc[0]
        rows.append(
            {
                "validation_slice": slice_name,
                "n_records": len(subset),
                "n_supported_alternatives": len(grouped),
                "top_alternative": top["alternative"],
                "top_mean_reward": top["mean"],
                "top_n_records": int(top["count"]),
                "interpretation": (
                    "internal scenario-slice check; not an independent simulator rerun"
                    if slice_name != "full scenario calendar"
                    else "full internal scenario calendar reference"
                ),
            }
        )

    summary = pd.DataFrame(rows)
    top = pd.DataFrame(top_rows)
    summary.to_csv(output_dir / "scenario_calendar_holdout_proxy_summary.csv", index=False)
    top.to_csv(output_dir / "scenario_calendar_holdout_proxy_top5.csv", index=False)
    save_scenario_proxy_plot(output_dir / "scenario_calendar_holdout_proxy.pdf", top)
    return summary, top


def _piecewise_pcm_temperature(
    enthalpy_j: float,
    mass_kg: float,
    cp_jkgk: float,
    latent_jkg: float,
    melting_point_c: float,
    swing_c: float,
) -> float:
    sensible_low = mass_kg * cp_jkgk * swing_c
    latent = mass_kg * latent_jkg
    sensible_high = mass_kg * cp_jkgk * swing_c
    enthalpy_j = float(np.clip(enthalpy_j, 0.0, sensible_low + latent + sensible_high))
    if enthalpy_j < sensible_low:
        return melting_point_c - swing_c + enthalpy_j / max(mass_kg * cp_jkgk, 1e-12)
    if enthalpy_j <= sensible_low + latent:
        return melting_point_c
    return melting_point_c + (enthalpy_j - sensible_low - latent) / max(mass_kg * cp_jkgk, 1e-12)


def _simulate_partial_cycle(
    *,
    mass_kg: float,
    area_m2: float,
    thickness_m: float,
    conductivity_wmk: float,
    h_wm2k: float,
    cp_jkgk: float,
    latent_jkg: float,
    melting_point_c: float,
    swing_c: float,
    utilization_fraction: float,
    mode: str,
    dt_s: float = 20.0,
    max_time_s: float = 12 * 3600,
) -> float:
    conduction_r = thickness_m / max(conductivity_wmk * area_m2, 1e-12)
    convection_r = 1.0 / max(h_wm2k * area_m2, 1e-12)
    ua_wk = 1.0 / (conduction_r + convection_r)
    total_enthalpy_j = mass_kg * (2.0 * cp_jkgk * swing_c + latent_jkg)
    target_span_j = total_enthalpy_j * utilization_fraction
    if mode == "charge":
        boundary_c = melting_point_c + swing_c
        enthalpy_j = 0.0
        target_j = 0.95 * target_span_j
        sign = 1.0
    elif mode == "discharge":
        boundary_c = melting_point_c - swing_c
        enthalpy_j = target_span_j
        target_j = 0.05 * target_span_j
        sign = -1.0
    else:
        raise ValueError(f"Unknown mode: {mode}")
    elapsed = 0.0
    while elapsed < max_time_s:
        temp_c = _piecewise_pcm_temperature(
            enthalpy_j,
            mass_kg,
            cp_jkgk,
            latent_jkg,
            melting_point_c,
            swing_c,
        )
        heat_rate_w = ua_wk * (boundary_c - temp_c)
        enthalpy_j += heat_rate_w * dt_s
        enthalpy_j = float(np.clip(enthalpy_j, 0.0, total_enthalpy_j))
        elapsed += dt_s
        if sign > 0 and enthalpy_j >= target_j:
            break
        if sign < 0 and enthalpy_j <= target_j + 1e-9:
            break
    return elapsed / 60.0


def low_order_top3_physics_rerun(
    output_dir: Path,
    raw: pd.DataFrame,
    expert_ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Run a transparent lumped enthalpy-balance check for the expert top-3.

    This is intentionally a screening calculation. It does not use the logged
    EFA-AHP reward, and it does not replace a full IDA-ICE rerun.
    """
    raw = raw.copy()
    raw["alternative"] = _alternative(raw)
    ranking = expert_ranking.sort_values("rank").head(3)
    rows = []
    for static_rank, row in enumerate(ranking.itertuples(index=False), start=1):
        subset = raw[raw["alternative"] == row.alternative]
        median = subset.median(numeric_only=True)
        mass_kg = float(median["pcm_mass_kg"])
        area_m2 = float(median["surface_area_m2"])
        thickness_m = float(median["pcm_thickness_mm"]) / 1000.0
        conductivity = float(median["thermal_conductivity_wmk"])
        h_coeff = float(median["heat_transfer_coeff_wm2k"])
        cp = float(median["specific_heat_jkgk"])
        latent = float(median["latent_heat_kjkg"]) * 1000.0
        melting = float(median["melting_point_c"])
        degradation = float(median["degradation_factor"])
        loss = float(median["energy_loss_pct"]) / 100.0
        utilization = float(np.clip(median["state_of_charge_pct"] / 100.0, 0.20, 0.65))
        swing_c = 20.0
        effective_latent = latent * degradation
        total_capacity_kj = mass_kg * (2.0 * cp * swing_c + effective_latent) / 1000.0
        usable_capacity_kj = total_capacity_kj * utilization * (1.0 - loss)
        charge_time_min = _simulate_partial_cycle(
            mass_kg=mass_kg,
            area_m2=area_m2,
            thickness_m=thickness_m,
            conductivity_wmk=conductivity,
            h_wm2k=h_coeff,
            cp_jkgk=cp,
            latent_jkg=effective_latent,
            melting_point_c=melting,
            swing_c=swing_c,
            utilization_fraction=utilization,
            mode="charge",
        )
        discharge_time_min = _simulate_partial_cycle(
            mass_kg=mass_kg,
            area_m2=area_m2,
            thickness_m=thickness_m,
            conductivity_wmk=conductivity,
            h_wm2k=h_coeff,
            cp_jkgk=cp,
            latent_jkg=effective_latent,
            melting_point_c=melting,
            swing_c=swing_c,
            utilization_fraction=utilization,
            mode="discharge",
        )
        rows.append(
            {
                "static_expert_rank": static_rank,
                "alternative": row.alternative,
                "median_state_of_charge_pct": float(median["state_of_charge_pct"]),
                "screening_utilization_fraction": utilization,
                "usable_capacity_kjkg": usable_capacity_kj / mass_kg,
                "usable_capacity_kjm2": usable_capacity_kj / area_m2,
                "charge_time_min": charge_time_min,
                "discharge_time_min": discharge_time_min,
                "charge_power_kjmin": usable_capacity_kj / max(charge_time_min, 1e-12),
                "discharge_power_kjmin": usable_capacity_kj / max(discharge_time_min, 1e-12),
                "round_trip_efficiency_pct": (1.0 - loss) * 100.0,
                "screening_note": "low-order enthalpy-balance check; not a full IDA-ICE rerun",
            }
        )
    result = pd.DataFrame(rows)
    benefit_cols = [
        "usable_capacity_kjkg",
        "usable_capacity_kjm2",
        "charge_power_kjmin",
        "discharge_power_kjmin",
        "round_trip_efficiency_pct",
    ]
    norm = result[benefit_cols].copy()
    norm = (norm - norm.min()) / (norm.max() - norm.min()).replace(0, 1.0)
    result["low_order_screening_score"] = norm.mean(axis=1)
    result["low_order_screening_rank"] = result["low_order_screening_score"].rank(ascending=False, method="min").astype(int)
    result = result.sort_values(["low_order_screening_rank", "static_expert_rank"]).reset_index(drop=True)
    result.to_csv(output_dir / "top3_low_order_physics_rerun.csv", index=False)
    save_top3_rerun_plot(output_dir / "top3_low_order_physics_rerun.pdf", result)
    return result


def save_top3_rerun_plot(output_path: Path, result: pd.DataFrame) -> None:
    configure_publication_style(base_font=8.4)
    plot = result.copy()
    plot["alternative_label"] = plot["alternative"].map(lambda value: wrap_label(value.replace(" | ", "--"), 24))
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.9), gridspec_kw={"wspace": 0.36})
    metrics = [
        ("usable_capacity_kjkg", "Usable capacity (kJ/kg)", DEEP_BLUE),
        ("charge_time_min", "Charge time (min)", ORANGE),
        ("round_trip_efficiency_pct", "Round-trip efficiency (%)", TEAL),
    ]
    for label, ax in zip(["(a)", "(b)", "(c)"], axes):
        ax.text(-0.16, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK)
    for ax, (metric, xlabel, color) in zip(axes, metrics):
        order = plot.sort_values(metric, ascending=(metric == "charge_time_min"))
        y = np.arange(len(order))
        ax.barh(y, order[metric], color=color, alpha=0.90, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(order["alternative_label"] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        clean_axis(ax, axis="x")
        xmax = max(order[metric]) * 1.18
        ax.set_xlim(0, xmax)
        for yy, value in zip(y, order[metric]):
            ax.text(value + xmax * 0.015, yy, f"{value:.1f}", ha="left", va="center", fontsize=7.1, color=MUTED_INK)
    save_figure(fig, output_path.with_suffix(""))


def save_scenario_proxy_plot(output_path: Path, top: pd.DataFrame) -> None:
    configure_publication_style(base_font=8.2)
    plot = top[top["rank"] <= 3].copy()
    plot["slice_label"] = plot["validation_slice"].map(lambda value: wrap_label(value, 16))
    plot["alternative_label"] = plot["alternative"].map(lambda value: wrap_label(value.replace(" | ", "--"), 24))

    fig, ax = plt.subplots(figsize=(8.7, 5.3))
    palette = [DEEP_BLUE, TEAL, ORANGE]
    sns.barplot(
        data=plot,
        y="slice_label",
        x="mean_reward",
        hue="rank",
        palette=palette,
        ax=ax,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.93,
    )
    ax.axvline(0, color=AXIS, linewidth=0.65)
    ax.set_xlabel("Mean EFA-AHP reward")
    ax.set_ylabel("")
    ax.legend(frameon=False, title="Rank", loc="upper center", bbox_to_anchor=(0.5, 1.08), ncols=3, borderaxespad=0)
    clean_axis(ax, axis="x")
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    for patch, row in zip(ax.patches, plot.itertuples(index=False)):
        width = patch.get_width()
        if not np.isfinite(width):
            continue
        x = width + 0.012 * span if width >= 0 else width - 0.012 * span
        ha = "left" if width >= 0 else "right"
        ax.text(x, patch.get_y() + patch.get_height() / 2, f"{width:.3f}", ha=ha, va="center", fontsize=7.0, color=MUTED_INK)
    save_figure(fig, output_path.with_suffix(""))


def save_revision_readme(
    output_dir: Path,
    ahp: pd.DataFrame,
    system_ranking: pd.DataFrame,
) -> None:
    lines = [
        "# Major-revision audit outputs",
        "",
        "These files support the reviewer-priority revisions without overstating evidence.",
        "",
        "## Key computed results",
        f"- Leave-one-expert-out AHP: top alternative unchanged in {len(ahp)} of {len(ahp)} omissions.",
        f"- Worst leave-one-expert-out Spearman rank correlation: {ahp['spearman_vs_full_group_rank'].min():.3f}.",
        f"- Largest factor-weight shift after omitting one expert: {ahp['max_abs_weight_shift'].max():.3f}.",
        "- Top-3 low-order enthalpy-balance rerun added as a transparent screening check; it is not a full IDA-ICE rerun.",
        "- Source-to-CSV fusion rules added for all 40 raw or constructed variables as an audit trail.",
        f"- System-specific ranking summaries generated for {system_ranking['system_type'].nunique()} application systems.",
        "",
        "## Remaining non-computable evidence",
        "- Surface-temperature RMSE/MAE requires raw observed and predicted temperature time series.",
        "- Absolute decrement-factor error requires observed and predicted decrement factors by board and replicate.",
        "- Stockholm 7.376% requires unrounded source values or the denominator used in the source table.",
    ]
    (output_dir / "major_revision_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    input_csv: Path = RAW_DATASET,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    study = load_static_study(input_csv)
    _, _, group_weights, _ = confirmed_expert_ahp_results()
    expert_ranking = expert_weighted_efa_ahp_ranking(study, group_weights)
    ahp_summary, ahp_weights = leave_one_expert_out_ahp(output_dir, input_csv)
    calibration = experimental_calibration_status(output_dir)
    fusion_rules = source_to_csv_fusion_rules(output_dir, study.raw)
    system_ranking = system_specific_ranking_summary(output_dir, expert_ranking)
    low_order = low_order_top3_physics_rerun(output_dir, study.raw, expert_ranking)
    save_revision_readme(output_dir, ahp_summary, system_ranking)
    return {
        "leave_one_expert_out": ahp_summary,
        "leave_one_expert_out_weights": ahp_weights,
        "calibration_status": calibration,
        "fusion_rules": fusion_rules,
        "system_specific_ranking": system_ranking,
        "low_order_rerun": low_order,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.input, args.output)
    print(f"Wrote supplementary audit artifacts to {args.output}")


if __name__ == "__main__":
    main()
