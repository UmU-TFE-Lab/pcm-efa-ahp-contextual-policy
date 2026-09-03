"""Run the manuscript analysis pipeline against a private local data file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
for module_dir in (
    ROOT / "pcm_efa_ahp",
    ROOT / "pcm_efa_ahp_rl",
    ROOT / "pcm_journal_extension",
):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from pcm_efa_ahp_study import run as run_static  # noqa: E402
from locked_policy_evaluation import run as run_policy  # noqa: E402
from mcdm_uncertainty import run as run_mcdm  # noqa: E402
from robustness_checks import run as run_robustness  # noqa: E402
from supplementary_audits import run as run_audits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the private fused PCM scenario table. The file is never copied.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs",
        help="Root directory for generated tables and reports.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=300,
        help="Number of row and month-block bootstrap replicates.",
    )
    parser.add_argument(
        "--skip-policy",
        action="store_true",
        help="Skip the computationally heavier locked contextual-policy evaluation.",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Build manuscript figures after all required analyses finish.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input.expanduser().resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"Private input file not found: {input_csv}")
    output_root = args.output_root.expanduser().resolve()

    static_dir = output_root / "static_efa_ahp"
    supplementary_dir = output_root / "supplementary"
    policy_dir = output_root / "contextual_policy" / "locked_evaluation"
    robustness_dir = supplementary_dir / "reviewer_required"
    audit_dir = supplementary_dir / "major_revision"

    print("[1/5] Static EFA-AHP")
    run_static(input_csv=input_csv, output_dir=static_dir, n_factors=4)

    print("[2/5] MCDM baselines, row bootstrap, expert AHP, and range audit")
    run_mcdm(
        input_csv=input_csv,
        output_dir=supplementary_dir,
        n_bootstrap=args.bootstrap,
    )

    if args.skip_policy:
        print("[3/5] Locked contextual-policy evaluation skipped")
    else:
        print("[3/5] Locked contextual-policy evaluation")
        run_policy(output_dir=policy_dir, input_csv=input_csv)

    print("[4/5] EFA alignment and month-block robustness")
    if args.skip_policy:
        print("      Policy summary copy skipped; running structural checks only")
        # The structural calculations do not require policy results. Run them
        # directly so the optional policy artifact is not treated as mandatory.
        from mcdm_uncertainty import load_static_study
        from robustness_checks import (
            build_variable_provenance,
            efa_robustness,
            time_block_bootstrap,
        )

        study = load_static_study(input_csv)
        robustness_dir.mkdir(parents=True, exist_ok=True)
        build_variable_provenance(study.raw.columns).to_csv(
            robustness_dir / "variable_level_provenance.csv", index=False
        )
        full = study.rankings[study.rankings["scenario"] == "engineering_default"]
        efa_robustness(robustness_dir, study.raw.copy(), full)
        time_block_bootstrap(
            study,
            robustness_dir,
            n_bootstrap=args.bootstrap,
        )
    else:
        run_robustness(
            output_dir=robustness_dir,
            input_csv=input_csv,
            policy_summary=policy_dir / "policy_summary.csv",
            n_bootstrap=args.bootstrap,
        )

    print("[5/5] Expert sensitivity, provenance rules, and low-order screen")
    run_audits(input_csv=input_csv, output_dir=audit_dir)

    if args.figures:
        if args.skip_policy:
            raise RuntimeError("Figures 2-7 require policy outputs; omit --skip-policy.")
        environment = os.environ.copy()
        environment["PCM_ANALYSIS_OUTPUT_ROOT"] = str(output_root)
        environment["PCM_FIGURE_OUTPUT_DIR"] = str(ROOT / "figures")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_figure1.py")],
            check=True,
            env=environment,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_result_figures.py")],
            check=True,
            env=environment,
        )

    print(f"Completed. Derived outputs are in: {output_root}")


if __name__ == "__main__":
    main()
