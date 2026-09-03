"""EFA-AHP-guided contextual policy learning for PCM thermal storage.

This module extends the static EFA-AHP study into an offline contextual
recommendation problem:

    state + action -> EFA-AHP reward

The reward model is a ridge-regularized contextual linear model with action
intercepts and state-action interactions. The numerical core deliberately
avoids external machine-learning dependencies; optional manuscript figures are
rendered with matplotlib and seaborn for publication-quality output.

Scientific boundary:
    The output is an offline, model-based contextual policy evaluation. It is
    not evidence of online closed-loop RL control because the CSV does not
    provide explicit episodes, behaviour-policy propensities, or reliable
    state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
from pathlib import Path
import sys
import textwrap

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = CODE_ROOT
STATIC_MODULE = CODE_ROOT / "pcm_efa_ahp"
STYLE_MODULE = CODE_ROOT / "pcm_journal_extension"
for module_path in (STATIC_MODULE, STYLE_MODULE):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from pcm_efa_ahp_study import (  # noqa: E402
    CRITERIA,
    DEFAULT_INPUT,
    FACTOR_LABELS,
    SCENARIO_TARGET_WEIGHTS,
    add_pcm_criteria,
    apply_preprocessing,
    build_alternatives,
    efa_from_standardized,
    score_with_fitted_efa,
    score_scenarios,
    scenario_weights,
    standardize,
    winsorize_frame,
)
from publication_style import (  # noqa: E402
    DEEP_BLUE,
    GRAY,
    GREEN,
    INK,
    ORANGE,
    clean_axis,
    configure_publication_style,
)


DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "contextual_policy"
STATE_FEATURES = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "is_daytime",
    "air_temperature_c",
    "relative_humidity_pct",
    "wind_speed_mps",
    "cloud_cover_pct",
    "solar_irradiance_wm2",
    "inlet_fluid_temp_c",
    "cycle_number",
]


BLUE = DEEP_BLUE


def configure_plot_style() -> None:
    configure_publication_style(base_font=8.6)


configure_plot_style()


@dataclass(frozen=True)
class ContextualRidgeModel:
    intercept: float
    state_coef: np.ndarray
    action_coef: np.ndarray
    interaction_coef: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray
    actions: list[str]
    action_parts: pd.DataFrame
    state_features: list[str]


def add_state_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    hour = data["timestamp"].dt.hour.astype(float)
    month = data["timestamp"].dt.month.astype(float)
    dayofyear = data["timestamp"].dt.dayofyear.astype(float)

    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    data["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    data["dayofyear_sin"] = np.sin(2 * np.pi * (dayofyear - 1) / 365)
    data["dayofyear_cos"] = np.cos(2 * np.pi * (dayofyear - 1) / 365)
    data["is_daytime"] = ((hour >= 7) & (hour <= 18)).astype(int)
    data["season"] = month.astype(int).map(season_from_month)
    return data


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def build_reward_dataset(
    input_csv: Path = DEFAULT_INPUT,
    scenario: str = "engineering_default",
    fit_idx: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the contextual dataset using a fit-only EFA-AHP reward model.

    When ``fit_idx`` is supplied, winsorization limits, standardization
    statistics, loadings, and factor-score coefficients are estimated only from
    those records. The fitted transformation is then applied unchanged to the
    remaining records. This is required for leakage-free temporal evaluation.
    """
    raw = pd.read_csv(input_csv)
    data = add_state_features(add_pcm_criteria(raw))
    criteria_names = [criterion.name for criterion in CRITERIA]
    if fit_idx is None:
        fit_idx = np.arange(len(data), dtype=int)
    fit_idx = np.asarray(fit_idx, dtype=int)
    if fit_idx.ndim != 1 or len(fit_idx) == 0:
        raise ValueError("fit_idx must be a non-empty one-dimensional index array")
    if fit_idx.min() < 0 or fit_idx.max() >= len(data):
        raise IndexError("fit_idx contains rows outside the input dataset")

    fit_data = data.iloc[fit_idx]
    clipped_fit, bounds = winsorize_frame(fit_data, criteria_names)
    z_fit, standardization = standardize(clipped_fit, criteria_names)
    efa = efa_from_standardized(z_fit, n_factors=4)
    z_all = apply_preprocessing(data, criteria_names, bounds, standardization)
    factor_scores = score_with_fitted_efa(z_all, efa["coefficients"])
    weights = np.array(SCENARIO_TARGET_WEIGHTS[scenario], dtype=float)
    weights = weights / weights.sum()

    reward = np.einsum(
        "ij,j->i",
        factor_scores[FACTOR_LABELS].to_numpy(dtype=float),
        weights,
        optimize=True,
    )
    data["alternative"] = data["pcm_type"] + " | " + data["system_type"] + " | " + data["encapsulation_type"]
    data["efa_ahp_reward"] = reward
    for factor in FACTOR_LABELS:
        data[f"{factor}_score"] = factor_scores[factor].to_numpy()

    alternatives = build_alternatives(data.iloc[fit_idx], factor_scores.iloc[fit_idx])
    rankings, stability = score_scenarios(alternatives, scenario_weights())
    default_top = (
        rankings[rankings["scenario"] == scenario]
        .sort_values("rank")
        .iloc[0]["alternative"]
    )

    rl_columns = (
        STATE_FEATURES
        + [
            "timestamp",
            "season",
            "pcm_type",
            "system_type",
            "encapsulation_type",
            "alternative",
            "efa_ahp_reward",
        ]
        + [f"{factor}_score" for factor in FACTOR_LABELS]
    )
    metadata = {
        "scenario": scenario,
        "scenario_weights": dict(zip(FACTOR_LABELS, weights)),
        "default_static_top": default_top,
        "efa_loadings": efa["loadings"],
        "efa_coefficients": efa["coefficients"],
        "preprocessing_bounds": bounds,
        "preprocessing_standardization": standardization,
        "reward_fit_n": int(len(fit_idx)),
        "reward_fit_min_timestamp": data.iloc[fit_idx]["timestamp"].min(),
        "reward_fit_max_timestamp": data.iloc[fit_idx]["timestamp"].max(),
        "alternative_rankings": rankings,
        "rank_stability": stability,
    }
    return data[rl_columns].copy(), metadata


def prepare_action_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["pcm_type", "system_type", "encapsulation_type", "alternative"]]
        .drop_duplicates()
        .sort_values("alternative")
        .reset_index(drop=True)
    )


def encode_actions(alternatives: pd.Series, actions: list[str]) -> np.ndarray:
    action_index = {action: idx for idx, action in enumerate(actions)}
    return alternatives.map(action_index).to_numpy(dtype=int)


def standardize_state(train: pd.DataFrame, full: pd.DataFrame, state_features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train[state_features].mean().to_numpy(dtype=float)
    std = train[state_features].std(ddof=1).to_numpy(dtype=float)
    std[std == 0] = 1.0
    state_z = (full[state_features].to_numpy(dtype=float) - mean) / std
    return state_z, mean, std


def design_matrix(state_z: np.ndarray, action_ids: np.ndarray, n_actions: int) -> np.ndarray:
    n, n_state = state_z.shape
    one_hot = np.zeros((n, n_actions), dtype=float)
    one_hot[np.arange(n), action_ids] = 1.0
    interactions = (one_hot[:, :, None] * state_z[:, None, :]).reshape(n, n_actions * n_state)
    intercept = np.ones((n, 1), dtype=float)
    return np.hstack([intercept, state_z, one_hot, interactions])


def fit_contextual_ridge(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    state_features: list[str] = STATE_FEATURES,
    alpha: float = 10.0,
) -> tuple[ContextualRidgeModel, np.ndarray]:
    actions_df = prepare_action_table(df)
    actions = actions_df["alternative"].tolist()
    action_ids = encode_actions(df["alternative"], actions)
    state_z, state_mean, state_std = standardize_state(df.iloc[train_idx], df, state_features)

    x_train = design_matrix(state_z[train_idx], action_ids[train_idx], len(actions))
    y_train = df.iloc[train_idx]["efa_ahp_reward"].to_numpy(dtype=float)

    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    gram = np.einsum("ni,nj->ij", x_train, x_train, optimize=True)
    rhs = np.einsum("ni,n->i", x_train, y_train, optimize=True)
    beta = np.linalg.solve(gram + penalty, rhs)

    n_state = len(state_features)
    n_actions = len(actions)
    intercept = float(beta[0])
    state_coef = beta[1 : 1 + n_state]
    action_coef = beta[1 + n_state : 1 + n_state + n_actions]
    interaction_coef = beta[1 + n_state + n_actions :].reshape(n_actions, n_state)

    model = ContextualRidgeModel(
        intercept=intercept,
        state_coef=state_coef,
        action_coef=action_coef,
        interaction_coef=interaction_coef,
        state_mean=state_mean,
        state_std=state_std,
        actions=actions,
        action_parts=actions_df,
        state_features=state_features,
    )
    return model, state_z


def predict_observed(model: ContextualRidgeModel, state_z: np.ndarray, alternatives: pd.Series) -> np.ndarray:
    action_ids = encode_actions(alternatives, model.actions)
    return (
        model.intercept
        + np.einsum("ij,j->i", state_z, model.state_coef, optimize=True)
        + model.action_coef[action_ids]
        + np.sum(model.interaction_coef[action_ids] * state_z, axis=1)
    )


def predict_all_actions(model: ContextualRidgeModel, state_z: np.ndarray) -> np.ndarray:
    base = model.intercept + np.einsum("ij,j->i", state_z, model.state_coef, optimize=True)
    interactions = np.einsum("ij,kj->ik", state_z, model.interaction_coef, optimize=True)
    return base[:, None] + model.action_coef[None, :] + interactions


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_policies(
    df: pd.DataFrame,
    model: ContextualRidgeModel,
    state_z: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    static_action: str,
    random_state: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_action_pred = predict_all_actions(model, state_z[test_idx])
    learned_action_ids = np.argmax(all_action_pred, axis=1)
    learned_rewards = all_action_pred[np.arange(len(test_idx)), learned_action_ids]

    static_id = model.actions.index(static_action)
    static_rewards = all_action_pred[:, static_id]

    rng = np.random.default_rng(random_state)
    random_action_ids = rng.integers(0, len(model.actions), size=len(test_idx))
    random_rewards = all_action_pred[np.arange(len(test_idx)), random_action_ids]

    observed_rewards_pred = predict_observed(model, state_z[test_idx], df.iloc[test_idx]["alternative"])
    observed_rewards_true = df.iloc[test_idx]["efa_ahp_reward"].to_numpy(dtype=float)

    train_best_action = (
        df.iloc[train_idx]
        .groupby("alternative")["efa_ahp_reward"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    train_best_id = model.actions.index(train_best_action)
    train_static_rewards = all_action_pred[:, train_best_id]

    policy_eval = pd.DataFrame(
        [
            summarize_policy("learned_contextual_policy", learned_rewards),
            summarize_policy("static_efa_ahp_best_full", static_rewards),
            summarize_policy("static_train_mean_best", train_static_rewards),
            summarize_policy("random_policy", random_rewards),
            summarize_policy("observed_logged_policy_predicted", observed_rewards_pred),
            summarize_policy("observed_logged_policy_true", observed_rewards_true),
        ]
    )

    learned_policy = pd.DataFrame(
        {
            "row_index": test_idx,
            "selected_alternative": [model.actions[i] for i in learned_action_ids],
            "predicted_reward": learned_rewards,
            "second_best_reward": np.partition(all_action_pred, -2, axis=1)[:, -2],
        }
    )
    learned_policy["reward_margin"] = learned_policy["predicted_reward"] - learned_policy["second_best_reward"]
    learned_policy = learned_policy.merge(model.action_parts, left_on="selected_alternative", right_on="alternative", how="left")

    action_distribution = (
        learned_policy["selected_alternative"]
        .value_counts()
        .rename_axis("alternative")
        .reset_index(name="selected_count")
    )
    action_distribution["selected_pct"] = action_distribution["selected_count"] / len(learned_policy) * 100

    context = df.iloc[test_idx][["season", "is_daytime", "solar_irradiance_wm2", "cycle_number"]].reset_index(drop=True)
    contextual_policy = pd.concat([context, learned_policy.reset_index(drop=True)], axis=1)
    seasonal = (
        contextual_policy.groupby(["season", "selected_alternative"])
        .size()
        .rename("count")
        .reset_index()
    )
    seasonal["season_total"] = seasonal.groupby("season")["count"].transform("sum")
    seasonal["pct"] = seasonal["count"] / seasonal["season_total"] * 100
    seasonal = seasonal.sort_values(["season", "pct"], ascending=[True, False])

    return policy_eval, learned_policy, action_distribution, seasonal


def summarize_policy(name: str, rewards: np.ndarray) -> dict[str, float | str]:
    return {
        "policy": name,
        "mean_reward": float(np.mean(rewards)),
        "median_reward": float(np.median(rewards)),
        "std_reward": float(np.std(rewards)),
        "p05_reward": float(np.quantile(rewards, 0.05)),
        "p95_reward": float(np.quantile(rewards, 0.95)),
    }


def write_report(
    output_dir: Path,
    metadata: dict[str, object],
    reward_metrics: dict[str, float],
    policy_eval: pd.DataFrame,
    action_distribution: pd.DataFrame,
) -> None:
    top_actions = action_distribution.head(10)
    lines = [
        "# EFA-AHP-guided contextual policy learning report",
        "",
        "## Scope",
        "",
        "This report evaluates an offline contextual policy layer on top of the EFA-AHP reward model.",
        "The results are model-based counterfactual estimates, not online deployment evidence.",
        "",
        "## Reward model metrics",
        "",
        f"- MAE: {reward_metrics['mae']:.4f}",
        f"- RMSE: {reward_metrics['rmse']:.4f}",
        f"- R2: {reward_metrics['r2']:.4f}",
        "",
        "## EFA-AHP scenario",
        "",
        f"- Scenario: {metadata['scenario']}",
        f"- Static EFA-AHP top alternative: {metadata['default_static_top']}",
        "",
        "## Policy comparison",
        "",
    ]
    for row in policy_eval.itertuples():
        lines.append(f"- {row.policy}: mean_reward={row.mean_reward:.4f}, median_reward={row.median_reward:.4f}")
    lines.extend(["", "## Most frequently selected actions", ""])
    for row in top_actions.itertuples():
        lines.append(f"- {row.alternative}: {row.selected_pct:.2f}%")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- policy_mean_reward.svg",
            "- learned_action_distribution.svg",
            "- seasonal_top_action_share.svg",
        ]
    )
    (output_dir / "rl_policy_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_horizontal_bar_svg(
    path: Path,
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    color: str = BLUE,
) -> None:
    del title
    label_width = 40 if len(labels) <= 6 else 46
    wrapped_labels = ["\n".join(textwrap.wrap(label.replace("_", " "), width=label_width, break_long_words=False)) for label in labels]
    values_array = np.asarray(values, dtype=float)
    fig_height = max(3.2, 0.42 * len(labels) + 1.25)
    fig, ax = plt.subplots(figsize=(8.8, fig_height))
    y = np.arange(len(labels))
    ax.barh(y, values_array, color=color, alpha=0.92, edgecolor="white", linewidth=0.45, height=0.58, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(wrapped_labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    xmin = min(0.0, float(np.nanmin(values_array)))
    xmax = max(0.0, float(np.nanmax(values_array)))
    span = max(xmax - xmin, 1e-9)
    ax.set_xlim(xmin - 0.03 * span, xmax + 0.30 * span)
    if xmin < 0 < xmax:
        ax.axvline(0, color=GRAY, linewidth=0.72, alpha=0.82)
    for idx, value in enumerate(values_array):
        if not np.isfinite(value):
            continue
        if value > xmin + 0.18 * span:
            ax.text(
                value - 0.018 * span,
                idx,
                f"{value:.3f}",
                ha="right",
                va="center",
                fontsize=8.0,
                color="white",
            )
        else:
            offset = 0.012 * span if value >= 0 else -0.012 * span
            ax.text(value + offset, idx, f"{value:.3f}", ha="left" if value >= 0 else "right", va="center", fontsize=8.0, color=INK)
    clean_axis(ax, axis="x")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def write_policy_figures(
    output_dir: Path,
    policy_eval: pd.DataFrame,
    action_distribution: pd.DataFrame,
    seasonal: pd.DataFrame,
) -> None:
    write_horizontal_bar_svg(
        output_dir / "policy_mean_reward.svg",
        policy_eval["policy"].tolist(),
        policy_eval["mean_reward"].astype(float).tolist(),
        "Policy comparison on held-out states",
        "Predicted or observed EFA-AHP reward",
        color=DEEP_BLUE,
    )

    top_actions = action_distribution.head(10)
    write_horizontal_bar_svg(
        output_dir / "learned_action_distribution.svg",
        top_actions["alternative"].tolist(),
        top_actions["selected_pct"].astype(float).tolist(),
        "Most frequent actions selected by the learned policy",
        "Share of held-out states selected (%)",
        color=GREEN,
    )

    season_order = ["winter", "spring", "summer", "autumn"]
    top_season = seasonal.groupby("season", as_index=False).head(1).copy()
    top_season["season_order"] = top_season["season"].map({s: i for i, s in enumerate(season_order)})
    top_season = top_season.sort_values("season_order")
    labels = [
        f"{row.season}: {row.selected_alternative}"
        for row in top_season.itertuples()
    ]
    write_horizontal_bar_svg(
        output_dir / "seasonal_top_action_share.svg",
        labels,
        top_season["pct"].astype(float).tolist(),
        "Dominant learned action by season",
        "Share within season (%)",
        color=ORANGE,
    )


def main() -> None:
    """Delegate direct execution to the manuscript's locked evaluation."""
    from locked_policy_evaluation import main as locked_evaluation_main

    locked_evaluation_main()


if __name__ == "__main__":
    main()
