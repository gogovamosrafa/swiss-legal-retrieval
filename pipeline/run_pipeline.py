"""
Orchestrate the full retrieval pipeline: steps 01 → 05.

Usage:
  python pipeline/run_pipeline.py                    # full pipeline on test split
  python pipeline/run_pipeline.py --split train      # evaluate on train split
  python pipeline/run_pipeline.py --skip 1 2         # skip index building
  python pipeline/run_pipeline.py --steps 2 3 4 5    # explicit step selection
  python pipeline/run_pipeline.py --skip-dense       # BM25-only mode
  python pipeline/run_pipeline.py --dry-run          # mock Claude API calls
"""

import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent


STEPS = {
    1: ("01_build_index.py",       "Build BM25 + dense indices"),
    2: ("02_retrieve.py",          "Hybrid retrieval (BM25 + dense, RRF)"),
    3: ("03_rerank.py",            "Claude API reranking"),
    4: ("04_evaluate.py",          "Compute macro F1 (train split only)"),
    5: ("05_prepare_submission.py","Generate submission.csv"),
}


def run_step(step_num: int, extra_args: list[str], config: str) -> bool:
    script = Path(__file__).parent / STEPS[step_num][0]
    label = STEPS[step_num][1]
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {label}")
    print(f"{'='*60}")

    cmd = [sys.executable, str(script), "--config", config] + extra_args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Step {step_num} failed (exit code {result.returncode})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the full legal retrieval pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--steps", nargs="+", type=int, metavar="N",
                        help="Only run these step numbers (e.g. --steps 2 3)")
    parser.add_argument("--skip", nargs="+", type=int, metavar="N",
                        help="Skip these step numbers (e.g. --skip 1)")
    parser.add_argument("--skip-dense", action="store_true",
                        help="BM25-only mode (no dense index / FAISS)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mock Claude API calls in step 3")
    parser.add_argument("--validate-corpus", action="store_true",
                        help="Filter citations not in corpus (step 5)")
    parser.add_argument("--per-query", action="store_true",
                        help="Print per-query F1 in step 4")
    args = parser.parse_args()

    selected = set(args.steps) if args.steps else set(STEPS.keys())
    skipped = set(args.skip) if args.skip else set()
    steps_to_run = sorted(selected - skipped)

    if not steps_to_run:
        print("[ERROR] No steps to run after applying --steps and --skip.")
        sys.exit(1)

    # Step 4 (evaluate) only makes sense on train split
    if 4 in steps_to_run and args.split == "test":
        print("[INFO] Step 4 (evaluate) skipped for test split (no gold labels).")
        steps_to_run = [s for s in steps_to_run if s != 4]

    print(f"Pipeline config:  {args.config}")
    print(f"Split:            {args.split}")
    print(f"Steps to run:     {steps_to_run}")
    print(f"Dense index:      {'DISABLED' if args.skip_dense else 'ENABLED'}")
    print(f"Claude reranking: {'DRY RUN' if args.dry_run else 'ENABLED'}")

    common = ["--config", args.config]

    step_args: dict[int, list[str]] = {
        1: common + (["--skip-dense"] if args.skip_dense else []),
        2: common + ["--split", args.split] + (["--skip-dense"] if args.skip_dense else []),
        3: common + (["--dry-run"] if args.dry_run else []),
        4: common + (["--per-query"] if args.per_query else []),
        5: common + (["--validate-corpus"] if args.validate_corpus else []),
    }

    for step_num in steps_to_run:
        ok = run_step(step_num, step_args[step_num], args.config)
        if not ok:
            print(f"\nPipeline aborted at step {step_num}.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}")
    if 5 in steps_to_run:
        output_dir = ROOT / "output"
        submission = output_dir / "submission.csv"
        if submission.exists():
            print(f"  Submission: {submission}")


if __name__ == "__main__":
    main()
