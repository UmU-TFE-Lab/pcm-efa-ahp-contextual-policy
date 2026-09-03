# Reproducibility guide

## Canonical order

`scripts/run_all.py` executes the analyses in the following order:

1. fit the record-level four-factor EFA reference model and static AHP rankings;
2. compute MCDM baselines, row-bootstrap uncertainty, expert summaries, and physical range checks;
3. fit all reward transformations and contextual models on the chronological training segment, select on validation, and evaluate once on the locked test segment;
4. refit structural EFA variants for diagnostic comparison, match factors with Tucker congruence, and calculate decision sensitivity only by fixed-reference projection;
5. compute expert leave-one-out results, within-system summaries, provenance-rule documentation, and the low-order enthalpy screen;
6. optionally generate manuscript figures from the new local outputs.

## Leakage control

The locked contextual analysis first reads only the timestamp column to define the chronological split. Winsorization bounds, standardization statistics, EFA loadings, factor-score coefficients, state scaling, reward models, support models, and estimated behavior propensities are fitted from the training segment. Reward-model and policy selection use the validation segment. The test segment is used once for final reporting.

## Determinism

Random seeds are explicit in the analysis functions. Small floating-point differences can occur across BLAS implementations and package versions. Compare rounded manuscript values and rank order, not binary output files.

## Outputs

All generated outputs are written under `outputs/` and `figures/`, both of which are ignored. This prevents derived tables, row-level recommended actions, and model artifacts from entering the public repository accidentally.

## Figure sources

The Python builders for manuscript Figures 1-7 produce the canonical publication outputs. Editable PowerPoint sources for Figure 1 and the AHP hierarchy are maintained outside the public code repository and are excluded by the repository ignore rules.
