"""Leakage-controlled offline contextual-policy evaluation for PCM EFA-AHP.

The analysis compares the following prespecified reward models and conservative
policy variants:

1. Polynomial state features with state-action ridge interactions.
2. Random Fourier features with state-action ridge interactions.
3. Bootstrap ensemble lower-confidence-bound policies.
4. Support-aware conservative policies that penalize or filter actions far
   from their logged training-state coverage.

All estimates remain offline and model-based. The script reports three support
diagnostics for each policy:

- model_mean_reward: reward predicted by the fitted reward model.
- local_knn_mean_reward: average reward of nearest logged training samples
  with the same selected action, used as a less model-dependent support check.
- mean_support_distance: distance from the selected action's logged state
  distribution; lower is safer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from contextual_policy_core import (  # noqa: E402
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    STATE_FEATURES,
    build_reward_dataset,
    encode_actions,
    prepare_action_table,
    regression_metrics,
    standardize_state,
)


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "locked_evaluation"


@dataclass(frozen=True)
class RidgeActionModel:
    name: str
    beta: np.ndarray
    n_state: int
    n_actions: int
    actions: list[str]
    feature_names: list[str]


def temporal_train_validation_test_indices(
    dataset: pd.DataFrame,
    train_size: float = 0.60,
    validation_size: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return chronological train, validation, and test indices."""
    if train_size <= 0 or validation_size <= 0 or train_size + validation_size >= 1:
        raise ValueError("train_size and validation_size must be positive and sum to less than one")
    ordered = dataset.assign(timestamp=pd.to_datetime(dataset["timestamp"])).sort_values("timestamp").index.to_numpy()
    train_end = int(round(len(ordered) * train_size))
    validation_end = int(round(len(ordered) * (train_size + validation_size)))
    return (
        np.sort(ordered[:train_end]),
        np.sort(ordered[train_end:validation_end]),
        np.sort(ordered[validation_end:]),
    )


def augment_linear(state_z: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    return state_z, names


def augment_poly2(state_z: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    pieces = [state_z, state_z**2]
    feature_names = names + [f"{name}^2" for name in names]
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    interactions = [
        ("air_temperature_c", "solar_irradiance_wm2"),
        ("air_temperature_c", "relative_humidity_pct"),
        ("air_temperature_c", "inlet_fluid_temp_c"),
        ("air_temperature_c", "cycle_number"),
        ("inlet_fluid_temp_c", "solar_irradiance_wm2"),
        ("solar_irradiance_wm2", "is_daytime"),
        ("solar_irradiance_wm2", "cloud_cover_pct"),
        ("month_sin", "air_temperature_c"),
        ("month_cos", "air_temperature_c"),
        ("dayofyear_sin", "solar_irradiance_wm2"),
        ("dayofyear_cos", "air_temperature_c"),
        ("wind_speed_mps", "air_temperature_c"),
    ]
    for left, right in interactions:
        if left in name_to_idx and right in name_to_idx:
            pieces.append((state_z[:, name_to_idx[left]] * state_z[:, name_to_idx[right]])[:, None])
            feature_names.append(f"{left}*{right}")
    return np.hstack(pieces), feature_names


def augment_rff(
    state_z: np.ndarray,
    names: list[str],
    train_idx: np.ndarray,
    n_features: int = 48,
    gamma: float = 0.20,
    random_state: int = 2026,
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(random_state)
    w = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(state_z.shape[1], n_features))
    b = rng.uniform(0.0, 2 * np.pi, size=n_features)
    projection = np.einsum("ij,jk->ik", state_z, w, optimize=True)
    rff = np.sqrt(2.0 / n_features) * np.cos(projection + b)
    # Re-standardize RFFs using training rows so ridge penalties are comparable.
    mean = rff[train_idx].mean(axis=0)
    std = rff[train_idx].std(axis=0, ddof=1)
    std[std == 0] = 1.0
    rff = (rff - mean) / std
    feature_names = names + [f"rff_{idx:02d}" for idx in range(n_features)]
    return np.hstack([state_z, rff]), feature_names


def design_matrix(state_features: np.ndarray, action_ids: np.ndarray, n_actions: int) -> np.ndarray:
    n, n_state = state_features.shape
    one_hot = np.zeros((n, n_actions), dtype=float)
    one_hot[np.arange(n), action_ids] = 1.0
    interactions = (one_hot[:, :, None] * state_features[:, None, :]).reshape(n, n_actions * n_state)
    return np.hstack([np.ones((n, 1), dtype=float), state_features, one_hot, interactions])


def fit_ridge_action_model(
    name: str,
    state_features: np.ndarray,
    feature_names: list[str],
    action_ids: np.ndarray,
    actions: list[str],
    y: np.ndarray,
    train_idx: np.ndarray,
    alpha: float,
) -> RidgeActionModel:
    x_train = design_matrix(state_features[train_idx], action_ids[train_idx], len(actions))
    y_train = y[train_idx]
    penalty = np.eye(x_train.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    gram = np.einsum("ni,nj->ij", x_train, x_train, optimize=True)
    rhs = np.einsum("ni,n->i", x_train, y_train, optimize=True)
    beta = np.linalg.solve(gram + penalty, rhs)
    return RidgeActionModel(
        name=name,
        beta=beta,
        n_state=state_features.shape[1],
        n_actions=len(actions),
        actions=actions,
        feature_names=feature_names,
    )


def predict_observed(model: RidgeActionModel, state_features: np.ndarray, action_ids: np.ndarray) -> np.ndarray:
    n_state = model.n_state
    n_actions = model.n_actions
    intercept = model.beta[0]
    state_coef = model.beta[1 : 1 + n_state]
    action_coef = model.beta[1 + n_state : 1 + n_state + n_actions]
    interaction_coef = model.beta[1 + n_state + n_actions :].reshape(n_actions, n_state)
    return (
        intercept
        + np.einsum("ij,j->i", state_features, state_coef, optimize=True)
        + action_coef[action_ids]
        + np.sum(interaction_coef[action_ids] * state_features, axis=1)
    )


def predict_all_actions(model: RidgeActionModel, state_features: np.ndarray) -> np.ndarray:
    n_state = model.n_state
    n_actions = model.n_actions
    intercept = model.beta[0]
    state_coef = model.beta[1 : 1 + n_state]
    action_coef = model.beta[1 + n_state : 1 + n_state + n_actions]
    interaction_coef = model.beta[1 + n_state + n_actions :].reshape(n_actions, n_state)
    base = intercept + np.einsum("ij,j->i", state_features, state_coef, optimize=True)
    interactions = np.einsum("ij,kj->ik", state_features, interaction_coef, optimize=True)
    return base[:, None] + action_coef[None, :] + interactions


def action_support_statistics(
    state_z: np.ndarray,
    action_ids: np.ndarray,
    train_idx: np.ndarray,
    n_actions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.zeros((n_actions, state_z.shape[1]), dtype=float)
    stds = np.ones_like(means)
    thresholds = np.zeros(n_actions, dtype=float)
    for action_id in range(n_actions):
        rows = train_idx[action_ids[train_idx] == action_id]
        if len(rows) == 0:
            thresholds[action_id] = np.inf
            continue
        values = state_z[rows]
        means[action_id] = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        std[std < 0.10] = 0.10
        stds[action_id] = std
        self_distance = np.sqrt(np.mean(((values - means[action_id]) / std) ** 2, axis=1))
        thresholds[action_id] = float(np.quantile(self_distance, 0.95))
    return means, stds, thresholds


def support_distances_for_all_actions(state_z: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(((state_z[:, None, :] - means[None, :, :]) / stds[None, :, :]) ** 2, axis=2))


def summarize_selected_policy(
    name: str,
    selected_ids: np.ndarray,
    score_matrix: np.ndarray,
    support_distances: np.ndarray,
    test_action_ids: np.ndarray,
    true_rewards: np.ndarray,
    local_rewards: np.ndarray,
    local_distances: np.ndarray,
    actions: list[str],
) -> dict[str, object]:
    row_index = np.arange(len(selected_ids))
    predicted = score_matrix[row_index, selected_ids]
    support = support_distances[row_index, selected_ids]
    matched = selected_ids == test_action_ids
    counts = np.bincount(selected_ids, minlength=len(actions))
    probs = counts[counts > 0] / counts.sum()
    entropy = float(-(probs * np.log(probs)).sum() / np.log(len(actions)))
    top_action_id = int(np.argmax(counts))
    return {
        "policy": name,
        "model_mean_reward": float(predicted.mean()),
        "model_median_reward": float(np.median(predicted)),
        "local_knn_mean_reward": float(local_rewards.mean()),
        "local_knn_median_reward": float(np.median(local_rewards)),
        "mean_support_distance": float(support.mean()),
        "p90_support_distance": float(np.quantile(support, 0.90)),
        "mean_knn_distance": float(local_distances.mean()),
        "logged_action_match_rate": float(matched.mean()),
        "matched_true_reward_mean": float(true_rewards[matched].mean()) if np.any(matched) else np.nan,
        "action_entropy": entropy,
        "top_action": actions[top_action_id],
        "top_action_pct": float(counts[top_action_id] / counts.sum() * 100.0),
    }


def local_knn_value_for_policy(
    train_state_z: np.ndarray,
    train_action_ids: np.ndarray,
    train_rewards: np.ndarray,
    test_state_z: np.ndarray,
    selected_action_ids: np.ndarray,
    k: int = 25,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(test_state_z), dtype=float)
    distances = np.zeros(len(test_state_z), dtype=float)
    for action_id in np.unique(selected_action_ids):
        test_positions = np.where(selected_action_ids == action_id)[0]
        train_positions = np.where(train_action_ids == action_id)[0]
        if len(train_positions) == 0:
            values[test_positions] = np.nan
            distances[test_positions] = np.inf
            continue
        train_x = train_state_z[train_positions]
        train_y = train_rewards[train_positions]
        kk = min(k, len(train_positions))
        for start in range(0, len(test_positions), chunk_size):
            pos = test_positions[start : start + chunk_size]
            diff = test_state_z[pos, None, :] - train_x[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            nearest = np.argpartition(dist2, kk - 1, axis=1)[:, :kk]
            nearest_dist = np.take_along_axis(dist2, nearest, axis=1)
            values[pos] = train_y[nearest].mean(axis=1)
            distances[pos] = np.sqrt(nearest_dist.mean(axis=1))
    return values, distances


def sampled_knn_scores_all_actions(
    train_state_z: np.ndarray,
    train_action_ids: np.ndarray,
    train_rewards: np.ndarray,
    test_state_z: np.ndarray,
    n_actions: int,
    k: int = 35,
    max_train_per_action: int = 1200,
    chunk_size: int = 256,
    random_state: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate action-conditional KNN rewards for every test state/action."""
    rng = np.random.default_rng(random_state)
    scores = np.zeros((len(test_state_z), n_actions), dtype=float)
    distances = np.zeros_like(scores)
    global_mean = float(train_rewards.mean())
    for action_id in range(n_actions):
        train_positions = np.where(train_action_ids == action_id)[0]
        if len(train_positions) == 0:
            scores[:, action_id] = global_mean
            distances[:, action_id] = np.inf
            continue
        if len(train_positions) > max_train_per_action:
            train_positions = rng.choice(train_positions, size=max_train_per_action, replace=False)
        train_x = train_state_z[train_positions]
        train_y = train_rewards[train_positions]
        kk = min(k, len(train_positions))
        for start in range(0, len(test_state_z), chunk_size):
            end = min(start + chunk_size, len(test_state_z))
            diff = test_state_z[start:end, None, :] - train_x[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            nearest = np.argpartition(dist2, kk - 1, axis=1)[:, :kk]
            nearest_dist = np.take_along_axis(dist2, nearest, axis=1)
            scores[start:end, action_id] = train_y[nearest].mean(axis=1)
            distances[start:end, action_id] = np.sqrt(nearest_dist.mean(axis=1))
    return scores, distances


def bootstrap_ensemble_predictions(
    state_features: np.ndarray,
    feature_names: list[str],
    action_ids: np.ndarray,
    actions: list[str],
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float = 10.0,
    n_models: int = 7,
    random_state: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    predictions = []
    for model_id in range(n_models):
        boot_train = rng.choice(train_idx, size=len(train_idx), replace=True)
        model = fit_ridge_action_model(
            name=f"bootstrap_linear_{model_id}",
            state_features=state_features,
            feature_names=feature_names,
            action_ids=action_ids,
            actions=actions,
            y=y,
            train_idx=boot_train,
            alpha=alpha,
        )
        predictions.append(predict_all_actions(model, state_features[test_idx]))
    stack = np.stack(predictions, axis=0)
    return stack.mean(axis=0), stack.std(axis=0, ddof=1)


def policy_action_distribution(selected_ids: np.ndarray, actions: list[str], policy_name: str) -> pd.DataFrame:
    counts = np.bincount(selected_ids, minlength=len(actions))
    rows = []
    total = counts.sum()
    for action_id, count in enumerate(counts):
        if count:
            rows.append(
                {
                    "policy": policy_name,
                    "alternative": actions[action_id],
                    "selected_count": int(count),
                    "selected_pct": float(count / total * 100.0),
                }
            )
    return pd.DataFrame(rows).sort_values(["policy", "selected_pct"], ascending=[True, False])


def estimate_behavior_probability_matrix(
    dataset: pd.DataFrame,
    action_ids: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    n_actions: int,
    shrinkage: float = 100.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Estimate coarse behavior propensities from training records only.

    The available data do not contain logged behavior-policy probabilities. We
    therefore estimate action frequencies within season-by-daytime strata and
    shrink them toward the global training action distribution. These are
    sensitivity-analysis propensities, not known logging probabilities.
    """
    train_counts = np.bincount(action_ids[train_idx], minlength=n_actions).astype(float)
    global_prob = (train_counts + 1.0) / (train_counts.sum() + n_actions)
    train_keys = (
        dataset.iloc[train_idx]["season"].astype(str)
        + "|"
        + dataset.iloc[train_idx]["is_daytime"].astype(str)
    ).to_numpy()
    eval_keys = (
        dataset.iloc[eval_idx]["season"].astype(str)
        + "|"
        + dataset.iloc[eval_idx]["is_daytime"].astype(str)
    ).to_numpy()
    probabilities = np.zeros((len(eval_idx), n_actions), dtype=float)
    for key in np.unique(eval_keys):
        eval_positions = np.where(eval_keys == key)[0]
        train_positions = np.where(train_keys == key)[0]
        local_counts = np.bincount(action_ids[train_idx][train_positions], minlength=n_actions).astype(float)
        probabilities[eval_positions] = (local_counts + shrinkage * global_prob) / (
            local_counts.sum() + shrinkage
        )
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    diagnostics = pd.DataFrame(
        {
            "action_id": np.arange(n_actions),
            "train_count": train_counts.astype(int),
            "global_smoothed_probability": global_prob,
            "minimum_stratified_probability": probabilities.min(axis=0),
            "maximum_stratified_probability": probabilities.max(axis=0),
        }
    )
    return probabilities, diagnostics


def evaluate_deterministic_policy_ope(
    name: str,
    selected_ids: np.ndarray,
    q_scores: np.ndarray,
    logged_action_ids: np.ndarray,
    true_rewards: np.ndarray,
    behavior_probabilities: np.ndarray,
    timestamps: pd.Series,
    actions: list[str],
    propensity_floor: float = 0.01,
    n_bootstrap: int = 300,
    random_state: int = 2026,
) -> dict[str, object]:
    """Estimate deterministic-policy value with DM, IPS, SNIPS, and DR.

    Confidence intervals use month-block bootstrap resampling. Propensities are
    estimated rather than logged, so the resulting OPE remains assumption-bound.
    """
    row = np.arange(len(selected_ids))
    q_policy = q_scores[row, selected_ids]
    q_logged = q_scores[row, logged_action_ids]
    matched = selected_ids == logged_action_ids
    logged_propensity_raw = behavior_probabilities[row, logged_action_ids]
    logged_propensity = np.maximum(logged_propensity_raw, propensity_floor)
    importance = matched.astype(float) / logged_propensity
    dr_contribution = q_policy + importance * (true_rewards - q_logged)
    ips_contribution = importance * true_rewards

    direct_value = float(q_policy.mean())
    ips_value = float(ips_contribution.mean())
    importance_sum = float(importance.sum())
    snips_value = float(ips_contribution.sum() / importance_sum) if importance_sum > 0 else np.nan
    dr_value = float(dr_contribution.mean())
    ess = float(importance_sum**2 / np.sum(importance**2)) if np.sum(importance**2) > 0 else 0.0

    periods = pd.to_datetime(timestamps).dt.to_period("M").astype(str).to_numpy()
    unique_periods = np.unique(periods)
    period_positions = {period: np.where(periods == period)[0] for period in unique_periods}
    rng = np.random.default_rng(random_state)
    bootstrap_dr = np.zeros(n_bootstrap, dtype=float)
    bootstrap_ips = np.zeros(n_bootstrap, dtype=float)
    bootstrap_snips = np.full(n_bootstrap, np.nan, dtype=float)
    for bootstrap_idx in range(n_bootstrap):
        sampled_periods = rng.choice(unique_periods, size=len(unique_periods), replace=True)
        sampled_rows = np.concatenate([period_positions[period] for period in sampled_periods])
        bootstrap_dr[bootstrap_idx] = dr_contribution[sampled_rows].mean()
        bootstrap_ips[bootstrap_idx] = ips_contribution[sampled_rows].mean()
        sampled_importance = importance[sampled_rows]
        if sampled_importance.sum() > 0:
            bootstrap_snips[bootstrap_idx] = (
                ips_contribution[sampled_rows].sum() / sampled_importance.sum()
            )

    counts = np.bincount(selected_ids, minlength=len(actions))
    top_action_id = int(np.argmax(counts))
    return {
        "policy": name,
        "direct_method_value": direct_value,
        "ips_value": ips_value,
        "snips_value": snips_value,
        "doubly_robust_value": dr_value,
        "dr_ci95_lower": float(np.quantile(bootstrap_dr, 0.025)),
        "dr_ci95_upper": float(np.quantile(bootstrap_dr, 0.975)),
        "ips_ci95_lower": float(np.quantile(bootstrap_ips, 0.025)),
        "ips_ci95_upper": float(np.quantile(bootstrap_ips, 0.975)),
        "snips_ci95_lower": float(np.nanquantile(bootstrap_snips, 0.025)),
        "snips_ci95_upper": float(np.nanquantile(bootstrap_snips, 0.975)),
        "logged_action_match_count": int(matched.sum()),
        "logged_action_match_rate": float(matched.mean()),
        "importance_weight_ess": ess,
        "minimum_raw_logged_propensity": float(logged_propensity_raw.min()),
        "n_propensities_floored": int(np.sum(logged_propensity_raw < propensity_floor)),
        "propensity_floor": propensity_floor,
        "top_action": actions[top_action_id],
        "top_action_pct": float(counts[top_action_id] / counts.sum() * 100.0),
    }


def build_policy_candidates(
    linear_scores: np.ndarray,
    poly_scores: np.ndarray,
    rff_scores: np.ndarray,
    ensemble_mean: np.ndarray,
    ensemble_std: np.ndarray,
    sampled_knn_scores: np.ndarray,
    sampled_knn_distances: np.ndarray,
    support_all: np.ndarray,
    support_thresholds: np.ndarray,
    static_id: int,
    random_state: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(random_state)
    n_rows, n_actions = linear_scores.shape
    candidates: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("static_efa_ahp_best", np.full(n_rows, static_id, dtype=int), linear_scores),
        ("random_policy", rng.integers(0, n_actions, size=n_rows), linear_scores),
        ("linear_argmax", np.argmax(linear_scores, axis=1), linear_scores),
        ("poly2_argmax", np.argmax(poly_scores, axis=1), poly_scores),
        ("rff_argmax", np.argmax(rff_scores, axis=1), rff_scores),
        ("ensemble_mean_argmax", np.argmax(ensemble_mean, axis=1), ensemble_mean),
        ("ensemble_lcb_beta_0_5", np.argmax(ensemble_mean - 0.5 * ensemble_std, axis=1), ensemble_mean),
        ("ensemble_lcb_beta_1_0", np.argmax(ensemble_mean - ensemble_std, axis=1), ensemble_mean),
        ("sampled_knn_argmax", np.argmax(sampled_knn_scores, axis=1), linear_scores),
        ("hybrid_linear_knn_50_50", np.argmax(0.50 * linear_scores + 0.50 * sampled_knn_scores, axis=1), linear_scores),
        ("hybrid_linear_knn_30_70", np.argmax(0.30 * linear_scores + 0.70 * sampled_knn_scores, axis=1), linear_scores),
        ("hybrid_poly_knn_50_50", np.argmax(0.50 * poly_scores + 0.50 * sampled_knn_scores, axis=1), poly_scores),
    ]
    for penalty in (0.02, 0.05, 0.10):
        conservative_scores = linear_scores - penalty * support_all
        candidates.append((f"linear_support_penalty_{penalty:.2f}", np.argmax(conservative_scores, axis=1), linear_scores))
    for quantile, multiplier in (("q95", 1.00), ("q90", 0.90)):
        filtered_scores = linear_scores.copy()
        filtered_scores[support_all > (support_thresholds * multiplier)[None, :]] = -np.inf
        no_valid = ~np.isfinite(filtered_scores).any(axis=1)
        filtered_scores[no_valid] = linear_scores[no_valid]
        candidates.append((f"linear_support_filter_{quantile}", np.argmax(filtered_scores, axis=1), linear_scores))
    for penalty in (0.01, 0.02):
        conservative_knn = sampled_knn_scores - penalty * sampled_knn_distances
        candidates.append((f"sampled_knn_distance_penalty_{penalty:.2f}", np.argmax(conservative_knn, axis=1), linear_scores))
    return candidates


def run(output_dir: Path = OUTPUT_DIR, input_csv: Path = DEFAULT_INPUT) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    index_frame = pd.read_csv(input_csv, usecols=["timestamp"])
    train_idx, validation_idx, test_idx = temporal_train_validation_test_indices(index_frame)
    dataset, metadata = build_reward_dataset(
        input_csv=input_csv,
        scenario="engineering_default",
        fit_idx=train_idx,
    )
    actions = prepare_action_table(dataset)["alternative"].tolist()
    action_ids = encode_actions(dataset["alternative"], actions)
    y = dataset["efa_ahp_reward"].to_numpy(dtype=float)

    state_z, _, _ = standardize_state(dataset.iloc[train_idx], dataset, STATE_FEATURES)
    train_state_z = state_z[train_idx]
    train_action_ids = action_ids[train_idx]
    validation_state_z = state_z[validation_idx]
    test_state_z = state_z[test_idx]
    validation_action_ids = action_ids[validation_idx]
    test_action_ids = action_ids[test_idx]
    validation_rewards = y[validation_idx]
    test_rewards = y[test_idx]

    linear_state, linear_names = augment_linear(state_z, STATE_FEATURES)
    poly_state, poly_names = augment_poly2(state_z, STATE_FEATURES)
    rff_state, rff_names = augment_rff(state_z, STATE_FEATURES, train_idx)
    model_specs = [
        ("linear_state_action_ridge", linear_state, linear_names, 10.0),
        ("poly2_state_action_ridge", poly_state, poly_names, 30.0),
        ("rff_state_action_ridge", rff_state, rff_names, 50.0),
    ]

    model_rows = []
    prediction_validation: dict[str, np.ndarray] = {}
    prediction_test: dict[str, np.ndarray] = {}
    for name, features, names, alpha in model_specs:
        model = fit_ridge_action_model(name, features, names, action_ids, actions, y, train_idx, alpha=alpha)
        validation_pred = predict_observed(model, features[validation_idx], validation_action_ids)
        test_pred = predict_observed(model, features[test_idx], test_action_ids)
        validation_metrics = regression_metrics(validation_rewards, validation_pred)
        test_metrics = regression_metrics(test_rewards, test_pred)
        model_rows.append(
            {
                "model": name,
                "n_features": model.n_state,
                "alpha": alpha,
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
                "mae": test_metrics["mae"],
                "rmse": test_metrics["rmse"],
                "r2": test_metrics["r2"],
            }
        )
        prediction_validation[name] = predict_all_actions(model, features[validation_idx])
        prediction_test[name] = predict_all_actions(model, features[test_idx])

    model_comparison = pd.DataFrame(model_rows)
    selected_reward_model = str(model_comparison.sort_values("validation_r2", ascending=False).iloc[0]["model"])
    model_comparison["selected_by_validation"] = model_comparison["model"] == selected_reward_model
    model_comparison = model_comparison.sort_values(["selected_by_validation", "validation_r2"], ascending=[False, False])

    support_means, support_stds, support_thresholds = action_support_statistics(
        state_z, action_ids, train_idx, len(actions)
    )
    validation_support = support_distances_for_all_actions(validation_state_z, support_means, support_stds)
    test_support = support_distances_for_all_actions(test_state_z, support_means, support_stds)

    combined_eval_idx = np.concatenate([validation_idx, test_idx])
    combined_ensemble_mean, combined_ensemble_std = bootstrap_ensemble_predictions(
        linear_state,
        linear_names,
        action_ids,
        actions,
        y,
        train_idx,
        combined_eval_idx,
        alpha=10.0,
    )
    validation_ensemble_mean = combined_ensemble_mean[: len(validation_idx)]
    validation_ensemble_std = combined_ensemble_std[: len(validation_idx)]
    test_ensemble_mean = combined_ensemble_mean[len(validation_idx) :]
    test_ensemble_std = combined_ensemble_std[len(validation_idx) :]

    combined_knn_scores, combined_knn_distances = sampled_knn_scores_all_actions(
        train_state_z,
        train_action_ids,
        y[train_idx],
        state_z[combined_eval_idx],
        len(actions),
    )
    validation_knn_scores = combined_knn_scores[: len(validation_idx)]
    validation_knn_distances = combined_knn_distances[: len(validation_idx)]
    test_knn_scores = combined_knn_scores[len(validation_idx) :]
    test_knn_distances = combined_knn_distances[len(validation_idx) :]

    static_action = str(metadata["default_static_top"])
    static_id = actions.index(static_action)
    validation_candidates = build_policy_candidates(
        prediction_validation["linear_state_action_ridge"],
        prediction_validation["poly2_state_action_ridge"],
        prediction_validation["rff_state_action_ridge"],
        validation_ensemble_mean,
        validation_ensemble_std,
        validation_knn_scores,
        validation_knn_distances,
        validation_support,
        support_thresholds,
        static_id,
        random_state=2026,
    )
    test_candidates = build_policy_candidates(
        prediction_test["linear_state_action_ridge"],
        prediction_test["poly2_state_action_ridge"],
        prediction_test["rff_state_action_ridge"],
        test_ensemble_mean,
        test_ensemble_std,
        test_knn_scores,
        test_knn_distances,
        test_support,
        support_thresholds,
        static_id,
        random_state=2027,
    )

    validation_behavior, behavior_diagnostics = estimate_behavior_probability_matrix(
        dataset, action_ids, train_idx, validation_idx, len(actions)
    )
    test_behavior, _ = estimate_behavior_probability_matrix(
        dataset, action_ids, train_idx, test_idx, len(actions)
    )
    behavior_diagnostics["alternative"] = actions

    q_validation = prediction_validation[selected_reward_model]
    validation_ope_rows = []
    for name, selected_ids, _ in validation_candidates:
        validation_ope_rows.append(
            evaluate_deterministic_policy_ope(
                name,
                selected_ids,
                q_validation,
                validation_action_ids,
                validation_rewards,
                validation_behavior,
                dataset.iloc[validation_idx]["timestamp"],
                actions,
                random_state=2026,
            )
        )
    validation_ope = pd.DataFrame(validation_ope_rows)
    support_aware_mask = validation_ope["policy"].str.contains("knn|support|lcb", case=False, regex=True)
    eligible = validation_ope[
        support_aware_mask
        & (validation_ope["logged_action_match_count"] >= 50)
        & (validation_ope["importance_weight_ess"] >= 30)
    ]
    if eligible.empty:
        selected_policy = "static_efa_ahp_best"
    else:
        selected_policy = str(eligible.sort_values("dr_ci95_lower", ascending=False).iloc[0]["policy"])
    validation_ope["eligible_for_policy_selection"] = support_aware_mask & (
        validation_ope["logged_action_match_count"] >= 50
    ) & (validation_ope["importance_weight_ess"] >= 30)
    validation_ope["selected_on_validation"] = validation_ope["policy"] == selected_policy

    q_test = prediction_test[selected_reward_model]
    test_ope_rows = []
    for name, selected_ids, _ in test_candidates:
        test_ope_rows.append(
            evaluate_deterministic_policy_ope(
                name,
                selected_ids,
                q_test,
                test_action_ids,
                test_rewards,
                test_behavior,
                dataset.iloc[test_idx]["timestamp"],
                actions,
                random_state=2027,
            )
        )
    test_ope = pd.DataFrame(test_ope_rows)
    test_ope["selected_on_validation"] = test_ope["policy"] == selected_policy
    test_ope["confirmatory_test_role"] = np.where(
        test_ope["policy"] == selected_policy,
        "validation-selected policy",
        np.where(test_ope["policy"] == "static_efa_ahp_best", "pre-specified static comparator", "exploratory comparator"),
    )

    summary_rows = []
    distribution_frames = []
    selected_frames = []
    for name, selected_ids, score_matrix in test_candidates:
        local_rewards, local_distances = local_knn_value_for_policy(
            train_state_z,
            train_action_ids,
            y[train_idx],
            test_state_z,
            selected_ids,
        )
        summary_rows.append(
            summarize_selected_policy(
                name,
                selected_ids,
                score_matrix,
                test_support,
                test_action_ids,
                test_rewards,
                local_rewards,
                local_distances,
                actions,
            )
        )
        distribution_frames.append(policy_action_distribution(selected_ids, actions, name))
        selected_frames.append(
            pd.DataFrame(
                {
                    "policy": name,
                    "row_index": test_idx,
                    "selected_alternative": [actions[idx] for idx in selected_ids],
                    "model_reward": score_matrix[np.arange(len(test_idx)), selected_ids],
                    "support_distance": test_support[np.arange(len(test_idx)), selected_ids],
                    "local_knn_reward": local_rewards,
                    "local_knn_distance": local_distances,
                    "logged_alternative": dataset.iloc[test_idx]["alternative"].to_numpy(),
                    "logged_true_reward": test_rewards,
                }
            )
        )

    validation_columns = validation_ope[
        ["policy", "doubly_robust_value", "dr_ci95_lower", "dr_ci95_upper", "importance_weight_ess"]
    ].rename(
        columns={
            "doubly_robust_value": "validation_dr_value",
            "dr_ci95_lower": "validation_dr_ci95_lower",
            "dr_ci95_upper": "validation_dr_ci95_upper",
            "importance_weight_ess": "validation_importance_weight_ess",
        }
    )
    policy_summary = (
        pd.DataFrame(summary_rows)
        .merge(test_ope, on="policy", how="left", suffixes=("_support", "_ope"))
        .merge(validation_columns, on="policy", how="left")
    )
    policy_summary["recommended_for_main_reporting"] = policy_summary["policy"] == selected_policy
    policy_summary["reporting_role"] = np.where(
        policy_summary["policy"] == selected_policy,
        "validation-selected support-aware policy",
        np.where(policy_summary["policy"] == "static_efa_ahp_best", "pre-specified static comparator", "exploratory comparator"),
    )
    policy_summary = policy_summary.sort_values(
        ["recommended_for_main_reporting", "doubly_robust_value"], ascending=[False, False]
    )
    action_distribution = pd.concat(distribution_frames, ignore_index=True)
    selected_actions = pd.concat(selected_frames, ignore_index=True)

    split_summary = pd.DataFrame(
        [
            {
                "split": name,
                "n_records": len(indices),
                "start_timestamp": dataset.iloc[indices]["timestamp"].min(),
                "end_timestamp": dataset.iloc[indices]["timestamp"].max(),
            }
            for name, indices in (("train", train_idx), ("validation", validation_idx), ("test", test_idx))
        ]
    )
    selection = pd.DataFrame(
        [
            {
                "selected_reward_model": selected_reward_model,
                "selected_policy": selected_policy,
                "selection_split": "validation",
                "final_evaluation_split": "test",
                "policy_selection_criterion": "largest validation month-block-bootstrap DR lower bound among support-aware candidates with match>=50 and ESS>=30",
                "propensity_model": "season-by-daytime action frequencies estimated on training data with shrinkage to the global action distribution",
            }
        ]
    )

    model_comparison.to_csv(output_dir / "reward_model_comparison.csv", index=False)
    policy_summary.to_csv(output_dir / "policy_summary.csv", index=False)
    validation_ope.to_csv(output_dir / "policy_validation_ope.csv", index=False)
    test_ope.to_csv(output_dir / "policy_test_ope.csv", index=False)
    selection.to_csv(output_dir / "policy_selection_protocol.csv", index=False)
    behavior_diagnostics.to_csv(output_dir / "estimated_behavior_propensity_diagnostics.csv", index=False)
    split_summary.to_csv(output_dir / "temporal_train_validation_test_split.csv", index=False)
    action_distribution.to_csv(output_dir / "policy_action_distribution.csv", index=False)
    selected_actions.to_csv(output_dir / "policy_selected_actions.csv", index=False)
    metadata["efa_loadings"].to_csv(output_dir / "train_only_efa_loadings.csv")
    metadata["efa_coefficients"].to_csv(output_dir / "train_only_efa_coefficients.csv")
    metadata["preprocessing_bounds"].to_csv(output_dir / "train_only_reward_bounds.csv", index=False)
    metadata["preprocessing_standardization"].to_csv(output_dir / "train_only_reward_standardization.csv", index=False)

    selected_test = test_ope[test_ope["policy"] == selected_policy].iloc[0]
    static_test = test_ope[test_ope["policy"] == "static_efa_ahp_best"].iloc[0]
    report = [
        "# Leakage-controlled offline contextual policy experiments",
        "",
        "The EFA-AHP reward transformation and all policy models are fitted on the first 60% of the scenario calendar.",
        "Policy/model selection uses the next 20%, and the final reported OPE uses the last 20% once.",
        "KNN values are support diagnostics. Policy value is reported with DM, IPS, SNIPS, and doubly robust estimates.",
        "Behavior propensities are estimated from training action frequencies and are not known logging probabilities.",
        "",
        "## Selection",
        "",
        f"- Reward model selected on validation: {selected_reward_model}",
        f"- Support-aware policy selected on validation: {selected_policy}",
        "",
        "## Final test OPE",
        "",
        f"- Selected policy: DR={selected_test.doubly_robust_value:.4f}, 95% block CI=[{selected_test.dr_ci95_lower:.4f}, {selected_test.dr_ci95_upper:.4f}], ESS={selected_test.importance_weight_ess:.1f}.",
        f"- Static comparator: DR={static_test.doubly_robust_value:.4f}, 95% block CI=[{static_test.dr_ci95_lower:.4f}, {static_test.dr_ci95_upper:.4f}], ESS={static_test.importance_weight_ess:.1f}.",
        "",
        "These estimates remain internal and assumption-bound; they are not simulator or field validation.",
    ]
    (output_dir / "contextual_policy_report.md").write_text("\n".join(report), encoding="utf-8")

    return {
        "model_comparison": model_comparison,
        "policy_summary": policy_summary,
        "validation_ope": validation_ope,
        "test_ope": test_ope,
        "selection": selection,
        "behavior_diagnostics": behavior_diagnostics,
        "split_summary": split_summary,
        "action_distribution": action_distribution,
        "selected_actions": selected_actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Private fused scenario table.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Directory for derived policy outputs.")
    args = parser.parse_args()
    result = run(output_dir=args.output, input_csv=args.input)
    print("Locked offline contextual policy evaluation completed")
    print("=" * 76)
    print("Reward-model comparison")
    print(result["model_comparison"].round(4).to_string(index=False))
    print()
    print("Policy summary")
    cols = [
        "policy",
        "selected_on_validation",
        "direct_method_value",
        "doubly_robust_value",
        "dr_ci95_lower",
        "dr_ci95_upper",
        "importance_weight_ess",
        "logged_action_match_count",
        "top_action_ope",
    ]
    print(result["policy_summary"][cols].round(4).to_string(index=False))
    print()
    print(f"Outputs written to: {args.output}")


if __name__ == "__main__":
    main()
