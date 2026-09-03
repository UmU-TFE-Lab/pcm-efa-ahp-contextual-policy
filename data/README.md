# Private data location

No study dataset is distributed in this repository.

The safest approach is to keep the authorized fused scenario table outside the repository and pass its absolute path to `scripts/run_all.py --input`. If a local in-repository copy is necessary, place it under `data/private/`; that directory and common data formats are ignored by Git.

The required fields, units, and analysis roles are listed in [`docs/DATA_SCHEMA.md`](../docs/DATA_SCHEMA.md). The repository does not contain the original wallboard time series, IDA-ICE model files, ATE input/output files, or an executable source-to-fusion data generator.

