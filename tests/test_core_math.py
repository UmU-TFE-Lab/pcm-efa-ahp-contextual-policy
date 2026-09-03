from __future__ import annotations

import numpy as np
import pandas as pd

from pcm_efa_ahp_study import FACTOR_LABELS, ahp, expert_panel_analysis
from locked_policy_evaluation import temporal_train_validation_test_indices
from mcdm_uncertainty import entropy_weights, topsis_scores, vikor_scores
from robustness_checks import match_refitted_factors, tucker_congruence


def test_confirmed_expert_group_weights_and_consistency() -> None:
    result = expert_panel_analysis()
    individual = result["individual_weights"]
    group = result["group_weights"].set_index("factor").loc[FACTOR_LABELS]
    assert (individual.groupby("expert_id")["cr"].first() < 0.10).all()
    assert float(group["cr"].iloc[0]) < 0.01
    np.testing.assert_allclose(
        group["weight"].to_numpy(),
        [0.245520, 0.157648, 0.249103, 0.347729],
        atol=1e-5,
    )


def test_ahp_recovers_consistent_weight_ratios() -> None:
    expected = np.array([0.4, 0.3, 0.2, 0.1])
    matrix = expected[:, None] / expected[None, :]
    result = ahp(matrix)
    np.testing.assert_allclose(result["weights"], expected, atol=1e-12)
    assert result["cr"] == 0.0


def test_mcda_primitives_have_expected_direction() -> None:
    normalized = np.array([[1.0, 1.0], [0.5, 0.4], [0.0, 0.0]])
    weights = entropy_weights(normalized)
    scores = topsis_scores(normalized, weights)
    q, _, _ = vikor_scores(normalized, weights)
    assert np.argmax(scores) == 0
    assert np.argmin(q) == 0
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_tucker_matching_recovers_permutation_and_sign() -> None:
    index = ["c1", "c2", "c3", "c4"]
    reference = pd.DataFrame(np.eye(4), index=index, columns=FACTOR_LABELS)
    variant = pd.DataFrame(
        np.column_stack([-reference.iloc[:, 2], reference.iloc[:, 0], reference.iloc[:, 3]]),
        index=index,
        columns=["v1", "v2", "v3"],
    )
    alignment = match_refitted_factors(reference, variant, "test")
    assert set(alignment["matched_reference_factor"]) == {
        FACTOR_LABELS[0],
        FACTOR_LABELS[2],
        FACTOR_LABELS[3],
    }
    np.testing.assert_allclose(alignment["absolute_tucker_congruence"], 1.0)
    assert tucker_congruence(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == -1.0


def test_chronological_split_uses_sorted_time() -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2024-01-05", "2024-01-01", "2024-01-04", "2024-01-02", "2024-01-03"])}
    )
    train, validation, test = temporal_train_validation_test_indices(
        frame, train_size=0.4, validation_size=0.2
    )
    assert set(train) == {1, 3}
    assert set(validation) == {4}
    assert set(test) == {0, 2}

