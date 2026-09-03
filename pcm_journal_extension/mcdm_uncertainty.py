"""MCDM baselines, bootstrap uncertainty, expert AHP, and range audits.

This module contains only analyses used by the manuscript. Earlier exploratory
policy routines are deliberately excluded; the canonical contextual analysis
is implemented in ``pcm_efa_ahp_rl/locked_policy_evaluation.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATIC_MODULE = ROOT / "pcm_efa_ahp"
if str(STATIC_MODULE) not in sys.path:
    sys.path.insert(0, str(STATIC_MODULE))

from pcm_efa_ahp_study import (  # noqa: E402
    CRITERIA,
    DEFAULT_INPUT,
    FACTOR_LABELS,
    SCENARIO_TARGET_WEIGHTS,
    add_pcm_criteria,
    ahp,
    build_alternatives,
    efa_from_standardized,
    score_scenarios,
    scenario_weights,
    standardize,
    winsorize_frame,
)
from study_config import (  # noqa: E402
    CONFIRMED_EXPERT_PAIRWISE,
    EXPERT_PROFILES,
    expert_pairwise_frame,
    parse_saaty_value,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "supplementary"


PHYSICAL_PLAUSIBILITY_RULES = [
    ("melting_point_c", 10.0, 70.0, "DSC/MD phase-change calibration and low-to-medium-temperature application range"),
    ("latent_heat_kjkg", 50.0, 350.0, "DSC latent-heat calibration and PCM literature range"),
    ("thermal_conductivity_wmk", 0.05, 1.50, "MD/RVE-FEM/Mori-Tanaka conductivity constraint"),
    ("density_kgm3", 500.0, 2200.0, "Organic-to-inorganic PCM density envelope"),
    ("specific_heat_jkgk", 800.0, 4000.0, "DSC heat-capacity behavior and engineering input range"),
    ("phase_fraction", 0.0, 1.0, "Bounded phase-change state variable"),
    ("state_of_charge_pct", 0.0, 100.0, "Bounded thermal state-of-charge variable"),
    ("energy_loss_pct", 0.0, 60.0, "Nonnegative energy-loss indicator with an engineering upper bound"),
    ("cooling_load_offset_pct", 0.0, 80.0, "Baseline-versus-PCM cooling-load reduction indicator"),
    ("thermal_storage_efficiency_pct", 0.0, 100.0, "Bounded useful-energy efficiency"),
    ("charging_time_min", 1.0, 300.0, "Positive charging-time process variable"),
    ("discharging_time_min", 1.0, 300.0, "Positive discharging-time process variable"),
    ("heat_flux_wm2", -1000.0, 3000.0, "Signed heat-flux range under the modeled boundary conditions"),
]


@dataclass(frozen=True)
class StaticStudyData:
    raw: pd.DataFrame
    criteria_data: pd.DataFrame
    standardized_criteria: pd.DataFrame
    factor_scores: pd.DataFrame
    alternatives: pd.DataFrame
    rankings: pd.DataFrame
    weights: pd.DataFrame


def load_static_study(input_csv: Path = DEFAULT_INPUT) -> StaticStudyData:
    raw = pd.read_csv(input_csv)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    criteria_names = [criterion.name for criterion in CRITERIA]
    with_criteria = add_pcm_criteria(raw)
    clipped, _ = winsorize_frame(with_criteria, criteria_names)
    standardized, _ = standardize(clipped, criteria_names)
    efa = efa_from_standardized(
        standardized,
        n_factors=len(FACTOR_LABELS),
        factor_labels=FACTOR_LABELS,
    )
    factor_scores = efa["factor_scores"]
    alternatives = build_alternatives(with_criteria, factor_scores)
    weights = scenario_weights()
    rankings, _ = score_scenarios(alternatives, weights)
    return StaticStudyData(raw, with_criteria, standardized, factor_scores, alternatives, rankings, weights)


def minmax_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    minimum = values.min(axis=0)
    span = values.max(axis=0) - minimum
    span[span == 0] = 1.0
    return (values - minimum) / span


def entropy_weights(normalized: np.ndarray) -> np.ndarray:
    x = np.asarray(normalized, dtype=float) + 1e-12
    p = x / x.sum(axis=0, keepdims=True)
    entropy = -(p * np.log(p)).sum(axis=0) / np.log(x.shape[0])
    divergence = 1.0 - entropy
    if np.allclose(divergence.sum(), 0):
        return np.ones(x.shape[1]) / x.shape[1]
    return divergence / divergence.sum()


def topsis_scores(normalized: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted = normalized * weights
    ideal = weighted.max(axis=0)
    nadir = weighted.min(axis=0)
    d_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - nadir) ** 2).sum(axis=1))
    return d_neg / (d_pos + d_neg + 1e-12)


def vikor_scores(
    normalized: np.ndarray,
    weights: np.ndarray,
    v: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best = normalized.max(axis=0)
    worst = normalized.min(axis=0)
    regret = weights * (best - normalized) / (best - worst + 1e-12)
    utility = regret.sum(axis=1)
    maximum_regret = regret.max(axis=1)
    q = (
        v * (utility - utility.min()) / (utility.max() - utility.min() + 1e-12)
        + (1 - v)
        * (maximum_regret - maximum_regret.min())
        / (maximum_regret.max() - maximum_regret.min() + 1e-12)
    )
    return q, utility, maximum_regret


def mcda_baselines(study: StaticStudyData) -> tuple[pd.DataFrame, pd.DataFrame]:
    criteria_names = [criterion.name for criterion in CRITERIA]
    data = study.criteria_data.copy()
    data["alternative"] = (
        data["pcm_type"] + " | " + data["system_type"] + " | " + data["encapsulation_type"]
    )
    alternative_criteria = (
        data.groupby(
            ["pcm_type", "system_type", "encapsulation_type", "alternative"],
            as_index=False,
        )[criteria_names]
        .mean()
        .sort_values("alternative")
        .reset_index(drop=True)
    )
    normalized = minmax_normalize(alternative_criteria[criteria_names].to_numpy())
    equal = np.ones(len(criteria_names)) / len(criteria_names)
    entropy = entropy_weights(normalized)

    baseline = alternative_criteria[
        ["alternative", "pcm_type", "system_type", "encapsulation_type"]
    ].copy()
    baseline["topsis_equal_score"] = topsis_scores(normalized, equal)
    baseline["topsis_entropy_score"] = topsis_scores(normalized, entropy)
    vikor_q, vikor_s, vikor_r = vikor_scores(normalized, entropy)
    baseline["vikor_entropy_q"] = vikor_q
    baseline["vikor_entropy_s"] = vikor_s
    baseline["vikor_entropy_r"] = vikor_r
    baseline["rank_topsis_equal"] = baseline["topsis_equal_score"].rank(ascending=False, method="min").astype(int)
    baseline["rank_topsis_entropy"] = baseline["topsis_entropy_score"].rank(ascending=False, method="min").astype(int)
    baseline["rank_vikor_entropy"] = baseline["vikor_entropy_q"].rank(ascending=True, method="min").astype(int)

    default = study.rankings.loc[
        study.rankings["scenario"] == "engineering_default",
        ["alternative", "final_score", "rank"],
    ].rename(columns={"final_score": "efa_ahp_score", "rank": "rank_efa_ahp"})
    baseline = baseline.merge(default, on="alternative", how="left")
    baseline = baseline.sort_values(["rank_efa_ahp", "alternative"])
    weights = pd.DataFrame(
        {"criterion": criteria_names, "equal_weight": equal, "entropy_weight": entropy}
    )
    return baseline, weights


def spearman_from_ranks(left: pd.Series, right: pd.Series) -> float:
    x = left.to_numpy(dtype=float)
    y = right.to_numpy(dtype=float)
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt(np.sum(x**2) * np.sum(y**2))
    return float(np.sum(x * y) / denominator) if denominator else 0.0


def baseline_rank_summary(baselines: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "TOPSIS_equal": ("rank_topsis_equal", "topsis_equal_score"),
        "TOPSIS_entropy": ("rank_topsis_entropy", "topsis_entropy_score"),
        "VIKOR_entropy": ("rank_vikor_entropy", "vikor_entropy_q"),
    }
    rows = []
    for method, (rank_col, score_col) in methods.items():
        top = baselines.sort_values([rank_col, "alternative"]).iloc[0]
        rows.append(
            {
                "baseline": method,
                "top_alternative": top["alternative"],
                "top_score_or_q": float(top[score_col]),
                "spearman_with_efa_ahp": spearman_from_ranks(
                    baselines["rank_efa_ahp"], baselines[rank_col]
                ),
                "efa_ahp_rank_of_baseline_top": int(top["rank_efa_ahp"]),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_efa_ahp_uncertainty(
    study: StaticStudyData,
    scenario: str = "engineering_default",
    n_bootstrap: int = 300,
    random_state: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = np.asarray(SCENARIO_TARGET_WEIGHTS[scenario], dtype=float)
    weights /= weights.sum()
    record_reward = np.einsum(
        "ij,j->i",
        study.factor_scores[FACTOR_LABELS].to_numpy(dtype=float),
        weights,
        optimize=True,
    )
    data = study.raw[["pcm_type", "system_type", "encapsulation_type"]].copy()
    data["alternative"] = (
        data["pcm_type"] + " | " + data["system_type"] + " | " + data["encapsulation_type"]
    )
    data["reward"] = record_reward
    groups = {
        name: frame["reward"].to_numpy(dtype=float)
        for name, frame in data.groupby("alternative")
    }
    alternatives = sorted(groups)
    rng = np.random.default_rng(random_state)
    boot = np.zeros((n_bootstrap, len(alternatives)))
    for alt_idx, alternative in enumerate(alternatives):
        values = groups[alternative]
        for bootstrap_idx in range(n_bootstrap):
            boot[bootstrap_idx, alt_idx] = rng.choice(
                values, size=len(values), replace=True
            ).mean()
    rank_samples = np.argsort(np.argsort(-boot, axis=1), axis=1) + 1
    score_ci = pd.DataFrame(
        {
            "alternative": alternatives,
            "mean_score": [groups[alternative].mean() for alternative in alternatives],
            "ci95_lower": np.quantile(boot, 0.025, axis=0),
            "ci95_upper": np.quantile(boot, 0.975, axis=0),
            "bootstrap_mean_rank": rank_samples.mean(axis=0),
            "top1_probability": (rank_samples == 1).mean(axis=0),
            "top3_probability": (rank_samples <= 3).mean(axis=0),
        }
    )
    score_ci.insert(
        4,
        "ci95_width",
        score_ci["ci95_upper"] - score_ci["ci95_lower"],
    )
    score_ci = score_ci.sort_values(
        ["mean_score", "alternative"], ascending=[False, True]
    )
    top_probs = score_ci.sort_values(
        ["top1_probability", "mean_score"], ascending=[False, False]
    ).reset_index(drop=True)
    return score_ci, top_probs


def confirmed_expert_panel() -> pd.DataFrame:
    return pd.DataFrame(EXPERT_PROFILES)


def confirmed_expert_pairwise() -> pd.DataFrame:
    return expert_pairwise_frame()


def expert_matrix(expert_df: pd.DataFrame) -> np.ndarray:
    matrix = np.ones((len(FACTOR_LABELS), len(FACTOR_LABELS)), dtype=float)
    index = {factor: idx for idx, factor in enumerate(FACTOR_LABELS)}
    for row in expert_df.itertuples():
        value = parse_saaty_value(row.saaty_value_left_over_right)
        left = index[row.left_factor]
        right = index[row.right_factor]
        matrix[left, right] = value
        matrix[right, left] = 1.0 / value
    return matrix


def confirmed_expert_ahp_results(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairwise = confirmed_expert_pairwise()
    matrices = []
    rows = []
    for expert_id, expert_df in pairwise.groupby("expert_id", sort=True):
        matrix = expert_matrix(expert_df)
        matrices.append(matrix)
        result = ahp(matrix)
        for factor, weight in zip(FACTOR_LABELS, result["weights"]):
            rows.append(
                {
                    "expert_id": expert_id,
                    "factor": factor,
                    "weight": float(weight),
                    "lambda_max": float(result["lambda_max"]),
                    "ci": float(result["ci"]),
                    "cr": float(result["cr"]),
                }
            )
    expert_weights = pd.DataFrame(rows)
    group_matrix = np.exp(np.mean(np.log(np.stack(matrices)), axis=0))
    group_result = ahp(group_matrix)
    group_weights = pd.DataFrame(
        {
            "factor": FACTOR_LABELS,
            "group_weight": group_result["weights"],
            "rank": pd.Series(group_result["weights"])
            .rank(ascending=False, method="min")
            .astype(int),
            "lambda_max": group_result["lambda_max"],
            "ci": group_result["ci"],
            "cr": group_result["cr"],
        }
    ).sort_values(["rank", "factor"])
    group_matrix_frame = pd.DataFrame(
        group_matrix, index=FACTOR_LABELS, columns=FACTOR_LABELS
    )
    return pairwise, expert_weights, group_weights, group_matrix_frame


def expert_weighted_efa_ahp_ranking(
    study: StaticStudyData,
    group_weights: pd.DataFrame,
) -> pd.DataFrame:
    score_columns = [f"{factor}_score" for factor in FACTOR_LABELS]
    weights = (
        group_weights.set_index("factor")
        .reindex(FACTOR_LABELS)["group_weight"]
        .to_numpy(dtype=float)
    )
    frame = study.alternatives.copy()
    frame["scenario"] = "expert_group_ahp"
    frame["final_score"] = frame[score_columns].to_numpy(dtype=float) @ weights
    frame["rank"] = frame["final_score"].rank(ascending=False, method="min").astype(int)
    return frame.sort_values(["rank", "alternative"]).reset_index(drop=True)


def physical_plausibility_audit(study: StaticStudyData) -> pd.DataFrame:
    rows = []
    for variable, expected_min, expected_max, source_basis in PHYSICAL_PLAUSIBILITY_RULES:
        values = study.raw[variable].dropna().astype(float)
        observed_min = float(values.min())
        observed_max = float(values.max())
        rows.append(
            {
                "variable": variable,
                "observed_min": observed_min,
                "observed_mean": float(values.mean()),
                "observed_max": observed_max,
                "expected_min": expected_min,
                "expected_max": expected_max,
                "status": (
                    "pass"
                    if observed_min >= expected_min - 1e-9
                    and observed_max <= expected_max + 1e-9
                    else "review"
                ),
                "source_basis": source_basis,
            }
        )
    return pd.DataFrame(rows)


def run(
    input_csv: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_bootstrap: int = 300,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    study = load_static_study(input_csv)
    baselines, criteria_weights = mcda_baselines(study)
    baseline_summary = baseline_rank_summary(baselines)
    score_ci, rank_probabilities = bootstrap_efa_ahp_uncertainty(
        study, n_bootstrap=n_bootstrap
    )
    physical_audit = physical_plausibility_audit(study)
    pairwise, expert_weights, group_weights, group_matrix = confirmed_expert_ahp_results()
    expert_ranking = expert_weighted_efa_ahp_ranking(study, group_weights)

    tables = {
        "mcdm_baseline_rankings": baselines,
        "mcdm_criteria_weights": criteria_weights,
        "mcdm_baseline_summary": baseline_summary,
        "bootstrap_score_uncertainty": score_ci,
        "bootstrap_rank_probabilities": rank_probabilities,
        "physical_plausibility_audit": physical_audit,
        "expert_pairwise_judgments": pairwise,
        "expert_individual_weights": expert_weights,
        "expert_group_weights": group_weights,
        "expert_group_pairwise_matrix": group_matrix,
        "expert_group_rankings": expert_ranking,
    }
    for name, frame in tables.items():
        frame.to_csv(
            output_dir / f"{name}.csv",
            index=name == "expert_group_pairwise_matrix",
        )
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    result = run(args.input, args.output, args.bootstrap)
    print(result["mcdm_baseline_summary"].round(4).to_string(index=False))
    print(f"Outputs written to: {args.output}")


if __name__ == "__main__":
    main()
