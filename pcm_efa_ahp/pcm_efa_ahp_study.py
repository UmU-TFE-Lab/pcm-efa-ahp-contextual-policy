"""EFA-AHP analysis of PCM thermal-storage configurations.

The script constructs the eleven decision criteria, fits the four-factor
reference model, and evaluates predefined engineering-weight scenarios.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .study_config import expert_pairwise_frame
except ImportError:  # Supports direct execution from this directory.
    from study_config import expert_pairwise_frame


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PACKAGE_ROOT / "data" / "private" / "pcm_thermal_storage.csv"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "static_efa_ahp"
DEFAULT_EXPERT_JUDGMENTS: Path | None = None

FACTOR_LABELS = [
    "Storage capacity and power",
    "Fast thermal response",
    "Durability and low loss",
    "Efficiency and load offset",
]


@dataclass(frozen=True)
class Criterion:
    name: str
    source: str
    direction: str
    description: str


CRITERIA: list[Criterion] = [
    Criterion(
        "storage_density_kjkg",
        "stored_energy_kj / pcm_mass_kg",
        "benefit",
        "Stored energy normalized by PCM mass.",
    ),
    Criterion(
        "storage_areal_density_kjm2",
        "stored_energy_kj / surface_area_m2",
        "benefit",
        "Stored energy normalized by heat-transfer surface area.",
    ),
    Criterion(
        "charge_power_kjmin",
        "stored_energy_kj / charging_time_min",
        "benefit",
        "Average charging power proxy.",
    ),
    Criterion(
        "discharge_power_kjmin",
        "stored_energy_kj / discharging_time_min",
        "benefit",
        "Average discharging power proxy.",
    ),
    Criterion(
        "thermal_storage_efficiency_pct",
        "thermal_storage_efficiency_pct",
        "benefit",
        "Thermal storage efficiency.",
    ),
    Criterion(
        "cooling_load_offset_pct",
        "cooling_load_offset_pct",
        "benefit",
        "Cooling load offset contribution.",
    ),
    Criterion(
        "state_of_charge_pct",
        "state_of_charge_pct",
        "benefit",
        "State of charge.",
    ),
    Criterion(
        "loss_score_pct",
        "100 - energy_loss_pct",
        "benefit_after_reversal",
        "Reversed energy loss score; higher means lower loss.",
    ),
    Criterion(
        "degradation_factor",
        "degradation_factor",
        "benefit",
        "Cycle degradation factor; higher means less degradation.",
    ),
    Criterion(
        "charge_speed_score_min",
        "-charging_time_min",
        "benefit_after_reversal",
        "Reversed charging time; higher means faster charging.",
    ),
    Criterion(
        "discharge_speed_score_min",
        "-discharging_time_min",
        "benefit_after_reversal",
        "Reversed discharging time; higher means faster discharging.",
    ),
]

SCENARIO_TARGET_WEIGHTS = {
    "balanced": [0.25, 0.25, 0.25, 0.25],
    "engineering_default": [0.35, 0.15, 0.20, 0.30],
    "storage_capacity_priority": [0.45, 0.15, 0.15, 0.25],
    "efficiency_load_priority": [0.20, 0.10, 0.20, 0.50],
    "fast_response_priority": [0.25, 0.45, 0.15, 0.15],
    "durability_low_loss_priority": [0.20, 0.15, 0.45, 0.20],
}


def symmetric_pseudoinverse(matrix: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Stable pseudoinverse for a real symmetric matrix."""
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    threshold = rcond * max(float(np.max(np.abs(values))), 1.0)
    inverse_values = np.zeros_like(values)
    keep = np.abs(values) > threshold
    inverse_values[keep] = 1.0 / values[keep]
    result = np.einsum("ik,k,jk->ij", vectors, inverse_values, vectors, optimize=False)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite symmetric pseudoinverse")
    return result


def add_pcm_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """Add decision criteria with all directions transformed to larger-is-better."""
    data = df.copy()
    eps = 1e-9
    data["storage_density_kjkg"] = data["stored_energy_kj"] / data["pcm_mass_kg"].clip(lower=eps)
    data["storage_areal_density_kjm2"] = data["stored_energy_kj"] / data["surface_area_m2"].clip(lower=eps)
    data["charge_power_kjmin"] = data["stored_energy_kj"] / data["charging_time_min"].clip(lower=eps)
    data["discharge_power_kjmin"] = data["stored_energy_kj"] / data["discharging_time_min"].clip(lower=eps)
    data["loss_score_pct"] = 100.0 - data["energy_loss_pct"]
    data["charge_speed_score_min"] = -data["charging_time_min"]
    data["discharge_speed_score_min"] = -data["discharging_time_min"]
    return data


def winsorize_frame(df: pd.DataFrame, columns: Iterable[str], lower_q: float = 0.01, upper_q: float = 0.99) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clip columns to quantile bounds and return the clipped frame plus bounds."""
    clipped = df.copy()
    rows = []
    for column in columns:
        lower = float(clipped[column].quantile(lower_q))
        upper = float(clipped[column].quantile(upper_q))
        clipped[column] = clipped[column].clip(lower, upper)
        rows.append({"criterion": column, "lower_bound": lower, "upper_bound": upper})
    return clipped, pd.DataFrame(rows)


def standardize(df: pd.DataFrame, columns: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Column-wise z-score standardization using sample standard deviation."""
    stats_rows = []
    z = pd.DataFrame(index=df.index)
    for column in columns:
        mean = float(df[column].mean())
        std = float(df[column].std(ddof=1))
        if std == 0:
            raise ValueError(f"Cannot standardize zero-variance criterion: {column}")
        z[column] = (df[column] - mean) / std
        stats_rows.append({"criterion": column, "mean": mean, "std": std})
    return z, pd.DataFrame(stats_rows)


def apply_preprocessing(
    df: pd.DataFrame,
    columns: Iterable[str],
    bounds: pd.DataFrame,
    standardization: pd.DataFrame,
) -> pd.DataFrame:
    """Apply previously fitted winsorization and z-score parameters.

    This transformation is used for held-out policy records and fixed-reference
    robustness projections. It prevents validation or test observations from
    contributing to preprocessing estimates.
    """
    column_list = list(columns)
    bounds_by_name = bounds.set_index("criterion")
    stats_by_name = standardization.set_index("criterion")
    missing = [
        column
        for column in column_list
        if column not in df.columns or column not in bounds_by_name.index or column not in stats_by_name.index
    ]
    if missing:
        raise KeyError(f"Missing preprocessing inputs for: {', '.join(missing)}")

    z = pd.DataFrame(index=df.index)
    for column in column_list:
        lower = float(bounds_by_name.loc[column, "lower_bound"])
        upper = float(bounds_by_name.loc[column, "upper_bound"])
        mean = float(stats_by_name.loc[column, "mean"])
        std = float(stats_by_name.loc[column, "std"])
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid fitted standard deviation for {column}: {std}")
        z[column] = (df[column].clip(lower, upper) - mean) / std
    return z


def kmo_measure(correlation: np.ndarray) -> float:
    corr = np.array(correlation, dtype=float)
    inv_corr = symmetric_pseudoinverse(corr, rcond=1e-10)
    scale = np.diag(1.0 / np.sqrt(np.diag(inv_corr)))
    partial = -np.einsum("ij,jk,kl->il", scale, inv_corr, scale, optimize=True)
    np.fill_diagonal(partial, 0.0)
    corr_no_diag = corr.copy()
    np.fill_diagonal(corr_no_diag, 0.0)
    r2 = np.sum(corr_no_diag**2)
    p2 = np.sum(partial**2)
    return float(r2 / (r2 + p2))


def bartlett_sphericity_statistic(correlation: np.ndarray, n_samples: int) -> tuple[float, int]:
    corr = np.array(correlation, dtype=float)
    n_variables = corr.shape[0]
    determinant = np.linalg.det(corr)
    chi_square = -(n_samples - 1 - (2 * n_variables + 5) / 6) * np.log(determinant)
    dof = n_variables * (n_variables - 1) // 2
    return float(chi_square), dof


def parallel_analysis(
    z: pd.DataFrame,
    n_repeats: int = 100,
    percentile: float = 0.95,
    random_state: int = 2026,
) -> pd.DataFrame:
    """Horn-style parallel analysis using independently permuted criteria."""
    values = z.to_numpy(dtype=float)
    n_rows, n_columns = values.shape
    rng = np.random.default_rng(random_state)
    random_eigenvalues = np.zeros((n_repeats, n_columns), dtype=float)

    for repeat_idx in range(n_repeats):
        permuted = np.empty_like(values)
        for column_idx in range(n_columns):
            permuted[:, column_idx] = values[rng.permutation(n_rows), column_idx]
        random_corr = np.corrcoef(permuted, rowvar=False)
        random_eigenvalues[repeat_idx] = np.linalg.eigvalsh(random_corr)[::-1]

    observed = np.linalg.eigvalsh(z.corr().to_numpy(dtype=float))[::-1]
    random_mean = random_eigenvalues.mean(axis=0)
    random_threshold = np.quantile(random_eigenvalues, percentile, axis=0)
    return pd.DataFrame(
        {
            "factor_number": np.arange(1, n_columns + 1),
            "observed_eigenvalue": observed,
            "random_mean_eigenvalue": random_mean,
            "random_95pct_eigenvalue": random_threshold,
            "retain_by_parallel_95pct": observed > random_threshold,
        }
    )


def efa_model_diagnostics(
    z: pd.DataFrame,
    efa_result: dict[str, object],
    n_factors: int,
    parallel_repeats: int = 100,
) -> dict[str, object]:
    """Return diagnostics needed for a stricter EFA methods section."""
    corr = efa_result["correlation"].to_numpy(dtype=float)
    loadings = efa_result["loadings"].copy()
    loading_values = loadings.to_numpy(dtype=float)
    communalities = np.sum(loading_values**2, axis=1)
    uniquenesses = 1.0 - communalities
    reproduced = loading_values @ loading_values.T + np.diag(uniquenesses)
    residual = corr - reproduced
    off_diagonal = ~np.eye(corr.shape[0], dtype=bool)
    offdiag_residual = residual[off_diagonal]

    abs_loadings = np.abs(loading_values)
    primary_ids = np.argmax(abs_loadings, axis=1)
    sorted_abs = np.sort(abs_loadings, axis=1)[:, ::-1]
    primary_loading = loading_values[np.arange(len(primary_ids)), primary_ids]
    cross_loading_margin = sorted_abs[:, 0] - sorted_abs[:, 1]
    complexity = np.sum(loading_values**2, axis=1) ** 2 / np.sum(loading_values**4, axis=1)

    criterion_diagnostics = loadings.copy()
    criterion_diagnostics.insert(0, "criterion", criterion_diagnostics.index)
    criterion_diagnostics["primary_factor"] = [FACTOR_LABELS[idx] for idx in primary_ids]
    criterion_diagnostics["primary_loading"] = primary_loading
    criterion_diagnostics["communality"] = communalities
    criterion_diagnostics["uniqueness"] = uniquenesses
    criterion_diagnostics["cross_loading_margin"] = cross_loading_margin
    criterion_diagnostics["complexity"] = complexity
    criterion_diagnostics = criterion_diagnostics.reset_index(drop=True)

    parallel = parallel_analysis(z, n_repeats=parallel_repeats)
    parallel_retained = int(parallel["retain_by_parallel_95pct"].sum())

    summary = {
        "determinant_correlation": float(np.linalg.det(corr)),
        "parallel_repeats": int(parallel_repeats),
        "parallel_retained_factors_95pct": parallel_retained,
        "mean_communality": float(np.mean(communalities)),
        "min_communality": float(np.min(communalities)),
        "max_communality": float(np.max(communalities)),
        "mean_uniqueness": float(np.mean(uniquenesses)),
        "max_uniqueness": float(np.max(uniquenesses)),
        "rmsr_offdiag": float(np.sqrt(np.mean(offdiag_residual**2))),
        "max_abs_residual_correlation": float(np.max(np.abs(offdiag_residual))),
        "mean_cross_loading_margin": float(np.mean(cross_loading_margin)),
        "min_cross_loading_margin": float(np.min(cross_loading_margin)),
        "mean_factor_complexity": float(np.mean(complexity)),
        "criteria_with_margin_below_0_20": int(np.sum(cross_loading_margin < 0.20)),
    }
    return {
        "summary": summary,
        "criterion_diagnostics": criterion_diagnostics,
        "parallel_analysis": parallel,
        "residual_correlations": pd.DataFrame(residual, index=z.columns, columns=z.columns),
    }


def varimax(loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 500, tol: float = 1e-7) -> np.ndarray:
    phi = np.array(loadings, dtype=float)
    n_rows, n_cols = phi.shape
    rotation = np.eye(n_cols)
    previous = 0.0
    for _ in range(max_iter):
        rotated = phi @ rotation
        u, singular_values, vh = np.linalg.svd(
            phi.T @ (rotated**3 - (gamma / n_rows) * rotated @ np.diag(np.diag(rotated.T @ rotated)))
        )
        rotation = u @ vh
        current = singular_values.sum()
        if previous and current < previous * (1 + tol):
            break
        previous = current
    return phi @ rotation


def efa_from_standardized(
    z: pd.DataFrame,
    n_factors: int = 4,
    factor_labels: list[str] | None = None,
) -> dict[str, object]:
    """Run PCA-style EFA with varimax rotation on standardized criteria."""
    if n_factors < 1 or n_factors > z.shape[1]:
        raise ValueError(f"n_factors must be between 1 and {z.shape[1]}; received {n_factors}")
    if factor_labels is None:
        factor_labels = FACTOR_LABELS if n_factors == len(FACTOR_LABELS) else [f"Factor {idx + 1}" for idx in range(n_factors)]
    if len(factor_labels) != n_factors:
        raise ValueError("factor_labels must contain one label per retained factor")

    corr_df = z.corr()
    corr = corr_df.to_numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    unrotated = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])
    loadings = varimax(unrotated)

    # All criteria have been transformed so higher is better. Flip factor signs so
    # the dominant loading direction is positive and factor scores are intuitive.
    for factor_idx in range(loadings.shape[1]):
        if loadings[:, factor_idx].sum() < 0:
            loadings[:, factor_idx] *= -1

    corr_inv = symmetric_pseudoinverse(corr, rcond=1e-10)
    score_information = np.einsum("pi,pq,qj->ij", loadings, corr_inv, loadings, optimize=True)
    score_information_inv = symmetric_pseudoinverse(score_information, rcond=1e-10)
    coefficients = np.einsum("pq,qj,jk->pk", corr_inv, loadings, score_information_inv, optimize=True)
    factor_scores = np.einsum("ij,jk->ik", z.to_numpy(dtype=float), coefficients, optimize=True)
    if not np.isfinite(coefficients).all() or not np.isfinite(factor_scores).all():
        raise FloatingPointError("Non-finite factor-score coefficients or scores")

    return {
        "correlation": corr_df,
        "eigenvalues": eigenvalues,
        "loadings": pd.DataFrame(loadings, index=z.columns, columns=factor_labels),
        "coefficients": pd.DataFrame(coefficients, index=z.columns, columns=factor_labels),
        "factor_scores": pd.DataFrame(factor_scores, index=z.index, columns=factor_labels),
    }


def score_with_fitted_efa(z: pd.DataFrame, coefficients: pd.DataFrame) -> pd.DataFrame:
    """Project standardized observations onto a previously fitted factor model."""
    missing = [column for column in coefficients.index if column not in z.columns]
    if missing:
        raise KeyError(f"Missing standardized criteria for factor scoring: {', '.join(missing)}")
    values = np.einsum(
        "ij,jk->ik",
        z.loc[:, coefficients.index].to_numpy(dtype=float),
        coefficients.to_numpy(dtype=float),
        optimize=True,
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite held-out factor scores")
    return pd.DataFrame(values, index=z.index, columns=coefficients.columns)


def ahp(pairwise_matrix: np.ndarray) -> dict[str, object]:
    matrix = np.array(pairwise_matrix, dtype=float)
    values, vectors = np.linalg.eig(matrix)
    idx = int(np.argmax(values.real))
    lambda_max = float(values[idx].real)
    weights = np.abs(vectors[:, idx].real)
    weights = weights / weights.sum()
    n = matrix.shape[0]
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    random_index = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}[n]
    cr = ci / random_index if random_index else 0.0
    if abs(ci) < 1e-12:
        ci = 0.0
    if abs(cr) < 1e-12:
        cr = 0.0
    if abs(lambda_max - n) < 1e-12:
        lambda_max = float(n)
    return {"weights": weights, "lambda_max": lambda_max, "ci": float(ci), "cr": float(cr)}


def expert_panel_analysis(
    judgments_path: Path | None = DEFAULT_EXPERT_JUDGMENTS,
) -> dict[str, object]:
    """Reconstruct individual and geometrically aggregated expert AHP matrices."""
    judgments = (
        expert_pairwise_frame()
        if judgments_path is None
        else pd.read_csv(judgments_path)
    )
    required = {"expert_id", "left_factor", "right_factor", "numeric_value"}
    missing = required - set(judgments.columns)
    if missing:
        raise KeyError(f"Missing expert-judgment fields: {', '.join(sorted(missing))}")

    factor_index = {factor: idx for idx, factor in enumerate(FACTOR_LABELS)}
    matrices: dict[str, np.ndarray] = {}
    individual_rows = []
    for expert_id, expert_rows in judgments.groupby("expert_id", sort=True):
        matrix = np.ones((len(FACTOR_LABELS), len(FACTOR_LABELS)), dtype=float)
        for row in expert_rows.itertuples():
            if row.left_factor not in factor_index or row.right_factor not in factor_index:
                raise ValueError(f"Unknown factor in judgments for {expert_id}")
            left = factor_index[row.left_factor]
            right = factor_index[row.right_factor]
            value = float(row.numeric_value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid Saaty judgment for {expert_id}: {value}")
            matrix[left, right] = value
            matrix[right, left] = 1.0 / value

        result = ahp(matrix)
        matrices[str(expert_id)] = matrix
        for factor, weight in zip(FACTOR_LABELS, result["weights"]):
            individual_rows.append(
                {
                    "expert_id": expert_id,
                    "factor": factor,
                    "weight": float(weight),
                    "lambda_max": float(result["lambda_max"]),
                    "ci": float(result["ci"]),
                    "cr": float(result["cr"]),
                }
            )

    matrix_stack = np.stack(list(matrices.values()))
    group_matrix = np.exp(np.mean(np.log(matrix_stack), axis=0))
    group_result = ahp(group_matrix)
    group_weights = pd.DataFrame(
        {
            "scenario": "expert_group",
            "factor": FACTOR_LABELS,
            "weight": group_result["weights"],
            "lambda_max": group_result["lambda_max"],
            "ci": group_result["ci"],
            "cr": group_result["cr"],
        }
    )
    return {
        "matrices": matrices,
        "individual_weights": pd.DataFrame(individual_rows),
        "group_matrix": group_matrix,
        "group_weights": group_weights,
    }


def leave_one_expert_out(
    matrices: dict[str, np.ndarray],
    alternatives: pd.DataFrame,
    full_group_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute group weights and alternative ranks after omitting each expert."""
    score_columns = [f"{factor}_score" for factor in FACTOR_LABELS]
    full_vector = full_group_weights.set_index("factor").loc[FACTOR_LABELS, "weight"].to_numpy()
    full_scores = alternatives[score_columns].to_numpy() @ full_vector
    full_ranks = pd.Series(
        pd.Series(full_scores).rank(ascending=False, method="min").to_numpy(),
        index=alternatives["alternative"],
    )
    full_top = str(alternatives.iloc[int(np.argmax(full_scores))]["alternative"])

    rows = []
    for omitted in sorted(matrices):
        retained = [matrix for expert, matrix in matrices.items() if expert != omitted]
        group_matrix = np.exp(np.mean(np.log(np.stack(retained)), axis=0))
        result = ahp(group_matrix)
        weights = np.asarray(result["weights"], dtype=float)
        scores = alternatives[score_columns].to_numpy() @ weights
        ranks = pd.Series(
            pd.Series(scores).rank(ascending=False, method="min").to_numpy(),
            index=alternatives["alternative"],
        )
        rows.append(
            {
                "omitted_expert": omitted,
                "group_cr": float(result["cr"]),
                "top_alternative": str(alternatives.iloc[int(np.argmax(scores))]["alternative"]),
                "top_score": float(np.max(scores)),
                "rank_of_full_group_top": int(ranks.loc[full_top]),
                "spearman_vs_full_group_rank": float(full_ranks.corr(ranks, method="pearson")),
                "max_abs_weight_shift": float(np.max(np.abs(weights - full_vector))),
            }
        )
    return pd.DataFrame(rows)


def pairwise_from_weights(weights: Iterable[float]) -> np.ndarray:
    vector = np.array(list(weights), dtype=float)
    if np.any(vector <= 0):
        raise ValueError("AHP target weights must be positive.")
    vector = vector / vector.sum()
    return vector[:, None] / vector[None, :]


def factor_clusters(loadings: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    rows = []
    for criterion, row in loadings.iterrows():
        for factor, value in row.items():
            if abs(value) >= threshold:
                rows.append({"criterion": criterion, "factor": factor, "loading": float(value)})
    return pd.DataFrame(rows).sort_values(["factor", "loading"], ascending=[True, False])


def scenario_weights() -> pd.DataFrame:
    rows = []
    for scenario, target in SCENARIO_TARGET_WEIGHTS.items():
        result = ahp(pairwise_from_weights(target))
        for factor, weight in zip(FACTOR_LABELS, result["weights"]):
            rows.append(
                {
                    "scenario": scenario,
                    "factor": factor,
                    "weight": float(weight),
                    "lambda_max": float(result["lambda_max"]),
                    "ci": float(result["ci"]),
                    "cr": float(result["cr"]),
                }
            )
    return pd.DataFrame(rows)


def build_alternatives(data: pd.DataFrame, factor_scores: pd.DataFrame) -> pd.DataFrame:
    enriched = pd.concat([data.reset_index(drop=True), factor_scores.reset_index(drop=True)], axis=1)
    enriched["alternative"] = (
        enriched["pcm_type"] + " | " + enriched["system_type"] + " | " + enriched["encapsulation_type"]
    )
    aggregations = {
        "n": ("stored_energy_kj", "size"),
        "stored_energy_kj_mean": ("stored_energy_kj", "mean"),
        "thermal_storage_efficiency_pct_mean": ("thermal_storage_efficiency_pct", "mean"),
        "energy_loss_pct_mean": ("energy_loss_pct", "mean"),
        "charging_time_min_mean": ("charging_time_min", "mean"),
        "discharging_time_min_mean": ("discharging_time_min", "mean"),
        "cooling_load_offset_pct_mean": ("cooling_load_offset_pct", "mean"),
        "state_of_charge_pct_mean": ("state_of_charge_pct", "mean"),
    }
    for factor in FACTOR_LABELS:
        aggregations[f"{factor}_score"] = (factor, "mean")
    return (
        enriched.groupby(["pcm_type", "system_type", "encapsulation_type", "alternative"], as_index=False)
        .agg(**aggregations)
        .sort_values("alternative")
        .reset_index(drop=True)
    )


def score_scenarios(alternatives: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_columns = [f"{factor}_score" for factor in FACTOR_LABELS]
    scenario_frames = []
    for scenario, scenario_df in weights.groupby("scenario", sort=False):
        vector = scenario_df.set_index("factor").loc[FACTOR_LABELS, "weight"].to_numpy()
        scores = alternatives[score_columns].to_numpy() @ vector
        frame = alternatives.copy()
        frame["scenario"] = scenario
        frame["final_score"] = scores
        frame["rank"] = frame["final_score"].rank(ascending=False, method="min").astype(int)
        scenario_frames.append(frame)
    rankings = pd.concat(scenario_frames, ignore_index=True).sort_values(["scenario", "rank", "alternative"])

    stability = rankings.pivot_table(index="alternative", columns="scenario", values="rank", aggfunc="first")
    stability["mean_rank"] = stability.mean(axis=1)
    stability["best_rank"] = stability.min(axis=1)
    stability["worst_rank"] = stability.max(axis=1)
    stability["rank_range"] = stability["worst_rank"] - stability["best_rank"]
    stability = stability.reset_index().sort_values(["mean_rank", "best_rank", "alternative"])
    return rankings, stability


def sensitivity_analysis(
    alternatives: pd.DataFrame,
    base_weights: pd.Series,
    scenario_name: str = "engineering_default",
    perturbation: float = 0.20,
) -> pd.DataFrame:
    score_columns = [f"{factor}_score" for factor in FACTOR_LABELS]
    base_vector = base_weights.loc[FACTOR_LABELS].to_numpy(dtype=float)
    base_scores = alternatives[score_columns].to_numpy() @ base_vector
    base_order = alternatives.assign(score=base_scores).sort_values("score", ascending=False)["alternative"].tolist()

    rows = []
    for factor_idx, factor in enumerate(FACTOR_LABELS):
        for label, multiplier in [("-20%", 1 - perturbation), ("+20%", 1 + perturbation)]:
            changed = base_vector.copy()
            changed[factor_idx] *= multiplier
            scores = alternatives[score_columns].to_numpy() @ changed
            order_df = alternatives[["alternative"]].copy()
            order_df["score"] = scores
            order_df = order_df.sort_values("score", ascending=False).reset_index(drop=True)
            rows.append(
                {
                    "base_scenario": scenario_name,
                    "changed_factor": factor,
                    "change": label,
                    "ranking_changed": order_df["alternative"].tolist() != base_order,
                    "top_1": order_df.loc[0, "alternative"],
                    "top_2": order_df.loc[1, "alternative"],
                    "top_3": order_df.loc[2, "alternative"],
                }
            )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    diagnostics: dict[str, float],
    clusters: pd.DataFrame,
    rankings: pd.DataFrame,
    stability: pd.DataFrame,
    expert_rankings: pd.DataFrame,
) -> None:
    top_default = rankings[rankings["scenario"] == "engineering_default"].sort_values("rank").head(10)
    stable_top = stability.head(10)
    lines = [
        "# PCM EFA-AHP multi-criteria selection study",
        "",
        "## Research design",
        "",
        "Alternatives are all combinations of `pcm_type x system_type x encapsulation_type`.",
        "The criteria are transformed so that larger values always indicate better performance.",
        "Engineering preference scenarios and the supplied five-expert AHP panel are evaluated separately.",
        "",
        "## EFA diagnostics",
        "",
        f"- Number of records: {int(diagnostics['n_records'])}",
        f"- Number of alternatives: {int(diagnostics['n_alternatives'])}",
        f"- Number of criteria: {int(diagnostics['n_criteria'])}",
        f"- KMO: {diagnostics['kmo']:.3f}",
        f"- Bartlett chi-square statistic: {diagnostics['bartlett_chi_square']:.3f}",
        f"- Bartlett degrees of freedom: {int(diagnostics['bartlett_dof'])}",
        f"- Retained factors: {int(diagnostics['n_factors'])}",
        f"- Cumulative variance of retained factors: {diagnostics['retained_cumulative_variance']:.3%}",
        f"- Parallel-analysis retained factors, 95% threshold: {int(diagnostics['parallel_retained_factors_95pct'])}",
        f"- Mean communality: {diagnostics['mean_communality']:.3f}",
        f"- Maximum uniqueness: {diagnostics['max_uniqueness']:.3f}",
        f"- Off-diagonal RMS residual correlation: {diagnostics['rmsr_offdiag']:.3f}",
        f"- Maximum absolute off-diagonal residual correlation: {diagnostics['max_abs_residual_correlation']:.3f}",
        f"- Criteria with cross-loading margin below 0.20: {int(diagnostics['criteria_with_margin_below_0_20'])}",
        "",
        "## Factor interpretation",
        "",
    ]
    for factor in FACTOR_LABELS:
        subset = clusters[clusters["factor"] == factor]
        joined = ", ".join(f"{row.criterion} ({row.loading:.2f})" for row in subset.itertuples())
        lines.append(f"- {factor}: {joined}")
    lines.extend(["", "## Top alternatives under engineering_default", ""])
    for row in top_default.itertuples():
        lines.append(f"{row.rank}. {row.alternative}: final_score={row.final_score:.3f}")
    lines.extend(["", "## Top alternatives under expert_group", ""])
    for row in expert_rankings.sort_values("rank").head(10).itertuples():
        lines.append(f"{row.rank}. {row.alternative}: final_score={row.final_score:.3f}")
    lines.extend(["", "## Stable alternatives across scenarios", ""])
    for row in stable_top.itertuples():
        lines.append(
            f"- {row.alternative}: mean_rank={row.mean_rank:.2f}, best={int(row.best_rank)}, worst={int(row.worst_rank)}"
        )
    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `criteria_metadata.csv`",
            "- `criteria_preprocessing_summary.csv`",
            "- `efa_eigenvalues.csv`",
            "- `parallel_analysis_eigenvalues.csv`",
            "- `efa_model_fit_summary.csv`",
            "- `efa_criterion_diagnostics.csv`",
            "- `efa_residual_correlations.csv`",
            "- `factor_loadings.csv`",
            "- `factor_score_coefficients.csv`",
            "- `alternative_factor_scores.csv`",
            "- `scenario_weights.csv`",
            "- `scenario_rankings.csv`",
            "- `rank_stability.csv`",
            "- `sensitivity_engineering_default.csv`",
            "- `expert_individual_weights.csv`",
            "- `expert_group_weights.csv`",
            "- `expert_group_rankings.csv`",
            "- `expert_leave_one_out.csv`",
        ]
    )
    (output_dir / "PCM_EFA_AHP_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(input_csv: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT_DIR, n_factors: int = 4) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_csv)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    data = add_pcm_criteria(raw)
    criteria_names = [criterion.name for criterion in CRITERIA]

    clipped, bounds = winsorize_frame(data, criteria_names)
    z, standardization = standardize(clipped, criteria_names)
    preprocessing = bounds.merge(standardization, on="criterion", how="left")

    efa = efa_from_standardized(z, n_factors=n_factors)
    stricter_efa = efa_model_diagnostics(z, efa, n_factors=n_factors)
    eigenvalues = efa["eigenvalues"]
    cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    corr = efa["correlation"].to_numpy()
    kmo = kmo_measure(corr)
    bartlett_chi_square, bartlett_dof = bartlett_sphericity_statistic(corr, len(raw))

    loadings = efa["loadings"]
    clusters = factor_clusters(loadings)
    alternatives = build_alternatives(data, efa["factor_scores"])
    weights = scenario_weights()
    rankings, stability = score_scenarios(alternatives, weights)
    expert = expert_panel_analysis()
    expert_rankings, _ = score_scenarios(alternatives, expert["group_weights"])
    expert_loo = leave_one_expert_out(expert["matrices"], alternatives, expert["group_weights"])
    base_weights = weights[weights["scenario"] == "engineering_default"].set_index("factor")["weight"]
    sensitivity = sensitivity_analysis(alternatives, base_weights)

    diagnostics = {
        "n_records": len(raw),
        "n_alternatives": len(alternatives),
        "n_criteria": len(criteria_names),
        "n_factors": n_factors,
        "kmo": kmo,
        "bartlett_chi_square": bartlett_chi_square,
        "bartlett_dof": bartlett_dof,
        "retained_cumulative_variance": float(cumulative[n_factors - 1]),
    }
    diagnostics.update(stricter_efa["summary"])

    pd.DataFrame([criterion.__dict__ for criterion in CRITERIA]).to_csv(output_dir / "criteria_metadata.csv", index=False)
    preprocessing.to_csv(output_dir / "criteria_preprocessing_summary.csv", index=False)
    pd.DataFrame(
        {
            "factor_number": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues,
            "cumulative_variance": cumulative,
        }
    ).to_csv(output_dir / "efa_eigenvalues.csv", index=False)
    stricter_efa["parallel_analysis"].to_csv(output_dir / "parallel_analysis_eigenvalues.csv", index=False)
    pd.DataFrame([diagnostics]).to_csv(output_dir / "efa_model_fit_summary.csv", index=False)
    stricter_efa["criterion_diagnostics"].to_csv(output_dir / "efa_criterion_diagnostics.csv", index=False)
    stricter_efa["residual_correlations"].to_csv(output_dir / "efa_residual_correlations.csv")
    loadings.to_csv(output_dir / "factor_loadings.csv")
    efa["coefficients"].to_csv(output_dir / "factor_score_coefficients.csv")
    clusters.to_csv(output_dir / "factor_clusters.csv", index=False)
    alternatives.to_csv(output_dir / "alternative_factor_scores.csv", index=False)
    weights.to_csv(output_dir / "scenario_weights.csv", index=False)
    rankings.to_csv(output_dir / "scenario_rankings.csv", index=False)
    stability.to_csv(output_dir / "rank_stability.csv", index=False)
    sensitivity.to_csv(output_dir / "sensitivity_engineering_default.csv", index=False)
    expert["individual_weights"].to_csv(output_dir / "expert_individual_weights.csv", index=False)
    expert["group_weights"].to_csv(output_dir / "expert_group_weights.csv", index=False)
    expert_rankings.to_csv(output_dir / "expert_group_rankings.csv", index=False)
    expert_loo.to_csv(output_dir / "expert_leave_one_out.csv", index=False)
    write_report(output_dir, diagnostics, clusters, rankings, stability, expert_rankings)

    return {
        "diagnostics": diagnostics,
        "clusters": clusters,
        "alternatives": alternatives,
        "weights": weights,
        "rankings": rankings,
        "stability": stability,
        "sensitivity": sensitivity,
        "expert": expert,
        "expert_rankings": expert_rankings,
        "expert_leave_one_out": expert_loo,
        "efa_diagnostics": stricter_efa,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Private fused scenario table.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for derived outputs.")
    parser.add_argument("--factors", type=int, default=4, help="Number of retained factors in the reference model.")
    args = parser.parse_args()
    result = run(input_csv=args.input, output_dir=args.output, n_factors=args.factors)
    diagnostics = result["diagnostics"]
    rankings = result["rankings"]
    stability = result["stability"]
    output_dir = result["output_dir"]

    print("PCM EFA-AHP study completed")
    print("=" * 72)
    print(f"Records: {diagnostics['n_records']}")
    print(f"Alternatives: {diagnostics['n_alternatives']}")
    print(f"Criteria: {diagnostics['n_criteria']}")
    print(f"KMO: {diagnostics['kmo']:.3f}")
    print(f"Bartlett chi-square: {diagnostics['bartlett_chi_square']:.3f}, dof={diagnostics['bartlett_dof']}")
    print(f"Retained cumulative variance: {diagnostics['retained_cumulative_variance']:.2%}")
    print(f"Parallel-analysis retained factors: {diagnostics['parallel_retained_factors_95pct']}")
    print(f"RMSR off-diagonal residual correlation: {diagnostics['rmsr_offdiag']:.4f}")
    print()

    print("Top 10 alternatives under engineering_default")
    cols = ["rank", "alternative", "final_score"] + [f"{factor}_score" for factor in FACTOR_LABELS]
    top = rankings[rankings["scenario"] == "engineering_default"].sort_values("rank").head(10)
    print(top[cols].round(3).to_string(index=False))
    print()

    print("Top 10 alternatives under expert_group")
    expert_top = result["expert_rankings"].sort_values("rank").head(10)
    print(expert_top[cols].round(3).to_string(index=False))
    print()

    print("Most stable alternatives across all scenarios")
    print(stability.head(10).round(3).to_string(index=False))
    print()
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
