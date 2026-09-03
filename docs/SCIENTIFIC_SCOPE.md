# Scientific scope and evidence boundary

## Supported claims

The code supports reproducible analysis of an authorized fused PCM-TES scenario table. It implements correlated-criterion reduction, expert-informed factor weighting, static screening, uncertainty checks, and a leakage-controlled offline contextual-policy comparison.

## Unsupported claims

The repository does not support independent regeneration of the fused table from original experiment, ATE, weather, and IDA-ICE files. It also does not establish:

- a universal ranking across all PCM applications;
- independent experimental or simulator validation of the reported rank order;
- causal performance improvement from changing a configuration;
- a sequential Markov decision process or online reinforcement-learning controller;
- known behavior-policy probabilities for logged actions.

The estimated behavior propensities, sparse action matches, and internal EFA-AHP reward must remain visible when interpreting off-policy estimates. The low-order enthalpy calculation supports physical plausibility of shortlisted candidates only.

## Analysis unit

The primary EFA is fitted at the scenario-record level. The resulting alternative score therefore integrates over the sampled operating-state distribution. Alternative-aggregated and reduced-criterion variants are structural sensitivity checks; their decision scores are projected onto the fixed reference factor axes so that the AHP weight meanings are not reassigned to unmatched rotated factors.

