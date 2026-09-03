# PCM-TES EFA-AHP and contextual-policy code

This repository contains the analysis and figure-generation code supporting the manuscript *Phase change material thermal energy storage configuration screening using EFA-AHP and offline contextual policy evaluation*.

The reproducible workflow covers:

- construction and transformation of the decision criteria;
- EFA diagnostics, parallel analysis, varimax rotation, and factor scoring;
- individual and geometrically aggregated expert AHP;
- static cross-application screening and within-system interpretation;
- TOPSIS, entropy-weight TOPSIS, VIKOR, row bootstrap, and month-block bootstrap;
- EFA sensitivity with Tucker congruence and fixed-reference score projection;
- leave-one-expert-out sensitivity and a low-order enthalpy-balance screen;
- leakage-controlled chronological contextual-policy selection and locked-test off-policy evaluation;
- publication figure generation.

## Data boundary

No raw, fused, processed, or derived record-level dataset is included. The repository-wide `.gitignore` blocks common tabular, array, model, and database formats. The analysis accepts an authorized local copy of the fused scenario table through `--input`; the file may remain anywhere outside the repository.

The code analyzes the fused scenario table but does not regenerate it from the experimental wallboard records, ATE calculations, weather inputs, or IDA-ICE model files. See [Data schema](docs/DATA_SCHEMA.md) and [Scientific scope](docs/SCIENTIFIC_SCOPE.md).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the repository safety check before and after any change:

```bash
python scripts/check_repository.py
```

Run the complete analysis against a private local file:

```bash
python scripts/run_all.py \
  --input /absolute/path/to/pcm_thermal_storage.csv \
  --bootstrap 300 \
  --figures
```

For a faster structural run without contextual-policy evaluation:

```bash
python scripts/run_all.py \
  --input /absolute/path/to/pcm_thermal_storage.csv \
  --bootstrap 50 \
  --skip-policy
```

Generated tables, reports, and figures are written to ignored local directories. They are not part of the GitHub upload candidate.

## Code map

| Location | Purpose |
|---|---|
| `pcm_efa_ahp/pcm_efa_ahp_study.py` | Criteria, EFA, AHP, static ranking, scenario sensitivity |
| `pcm_efa_ahp/study_config.py` | Auditable expert profiles and pairwise judgments |
| `pcm_efa_ahp_rl/contextual_policy_core.py` | Train-only reward construction and contextual feature utilities |
| `pcm_efa_ahp_rl/locked_policy_evaluation.py` | Chronological model selection, support diagnostics, propensity estimation, DM/IPS/SNIPS/DR evaluation |
| `pcm_journal_extension/mcdm_uncertainty.py` | MCDM baselines, row bootstrap, expert summaries, physical range checks |
| `pcm_journal_extension/robustness_checks.py` | Factor retention, Tucker matching, fixed-reference projections, month-block bootstrap |
| `pcm_journal_extension/supplementary_audits.py` | Expert leave-one-out, provenance rules, within-system summaries, low-order enthalpy screen |
| `scripts/build_figure1.py` | Main evidence-to-decision workflow figure |
| `scripts/build_result_figures.py` | Result Figures 2-7 from locally generated outputs |

The detailed manuscript-to-code crosswalk is in [Method-to-code map](docs/METHOD_TO_CODE_MAP.md).

## Verification

The public CI checks code syntax, mathematical unit tests, and repository hygiene without requiring the private dataset:

```bash
python -m compileall -q pcm_efa_ahp pcm_efa_ahp_rl pcm_journal_extension scripts
pytest
python scripts/check_repository.py
```

Full numerical reproduction requires the authorized local input table. Random seeds used by parallel analysis, bootstrap calculations, random Fourier features, and locked-test intervals are fixed in source.

## Interpretation

The static result is a state-distribution-weighted screening result, not proof of a universal optimum across physically different applications. The contextual component is an offline contextual-bandit evaluation of an internally constructed EFA-AHP reward, not online reinforcement learning or validated closed-loop control. The low-order rerun checks candidate-set plausibility and does not validate the final rank order.

## Figure-source boundary

The Python builders for manuscript Figures 1-7 are included. Editable PowerPoint sources for Figure 1 and the AHP hierarchy are maintained separately and are intentionally excluded from this code release.

## Release status

The source repository is public. An explicit software license and archived release DOI should be selected before publication. Follow [Release checklist](docs/RELEASE_CHECKLIST.md).
