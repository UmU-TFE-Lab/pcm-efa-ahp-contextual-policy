# Manuscript method-to-code map

| Manuscript component | Canonical implementation | Principal outputs |
|---|---|---|
| Eleven benefit-oriented criteria | `add_pcm_criteria()` in `pcm_efa_ahp_study.py` | criteria metadata and transformed criterion matrix |
| Winsorization and standardization | `winsorize_frame()`, `standardize()`, `apply_preprocessing()` | fitted bounds and scaling parameters |
| KMO, Bartlett test, and parallel analysis | `kmo_measure()`, `bartlett_sphericity_statistic()`, `parallel_analysis()` | EFA diagnostic tables |
| Four-factor reference EFA | `efa_from_standardized()`, `varimax()`, `score_with_fitted_efa()` | loadings, score coefficients, factor scores |
| Individual and group AHP | `expert_panel_analysis()` plus `study_config.py` | individual CR values, geometric group matrix, group weights |
| Static scores and rankings | `build_alternatives()`, `score_scenarios()` | alternative scores, rankings, weight scenarios |
| Leave-one-expert-out AHP | `leave_one_expert_out()` and `supplementary_audits.py` | omitted-expert weight and rank sensitivity |
| TOPSIS and VIKOR baselines | `mcda_baselines()` in `mcdm_uncertainty.py` | baseline rankings and rank correlations |
| Row bootstrap | `bootstrap_efa_ahp_uncertainty()` | score intervals and top-rank probabilities |
| EFA structural sensitivity | `efa_robustness()` in `robustness_checks.py` | parallel-analysis factor counts, Tucker congruence, fixed-reference ranks |
| Month-block bootstrap | `time_block_bootstrap()` | temporally blocked uncertainty summaries |
| Physical bounds | `physical_plausibility_audit()` | observed-versus-admissible range table |
| Low-order enthalpy screen | `low_order_top3_physics_rerun()` | candidate plausibility metrics; not rank validation |
| Chronological 60/20/20 design | `temporal_train_validation_test_indices()` in `locked_policy_evaluation.py` | split summary |
| Train-only EFA-AHP reward | `build_reward_dataset(..., fit_idx=train_idx)` | train-fitted loadings, coefficients, bounds, reward |
| Reward models and support-aware policies | `fit_ridge_action_model()`, `build_policy_candidates()` | validation model comparison and policy candidates |
| Estimated behavior propensities | `estimate_behavior_probability_matrix()` | action-level propensity diagnostics |
| Locked-test off-policy evaluation | `evaluate_deterministic_policy_ope()` | DM, IPS, SNIPS, DR, ESS, action matches, block CI |
| Publication figures | `scripts/build_figure1.py`, `scripts/build_result_figures.py` | vector PDF/SVG and high-resolution PNG files |

The source-to-fusion data-generation algorithm is not present in the available archive and is not reconstructed by assumption.

