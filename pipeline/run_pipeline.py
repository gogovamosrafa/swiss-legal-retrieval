"""
Orchestrate the full retrieval pipeline: steps 01 → 05.

Usage:
  python pipeline/run_pipeline.py                       # full pipeline on test split
  python pipeline/run_pipeline.py --split train         # evaluate on train split
  python pipeline/run_pipeline.py --skip 1 2            # skip index building
  python pipeline/run_pipeline.py --steps 2 3 4 5       # explicit step selection
  python pipeline/run_pipeline.py --skip-dense          # BM25-only mode
  python pipeline/run_pipeline.py --dry-run             # print plan, execute nothing

Path overrides (env vars take precedence over config.yaml):
  DATA_DIR=data/ INDEX_DIR=indices/ OUTPUT_DIR=output/ python pipeline/run_pipeline.py
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


# ── Step registry ─────────────────────────────────────────────────────────────

STEPS = {
    1: ("01_build_index.py",        "Build BM25 + dense indices"),
    2: ("02_retrieve.py",           "Hybrid retrieval (BM25 + dense, RRF)"),
    3: ("03_rerank.py",             "Claude API reranking"),
    4: ("04_evaluate.py",           "Compute macro F1 (train split only)"),
    5: ("05_prepare_submission.py", "Generate submission.csv"),
}


# ── Path resolution ───────────────────────────────────────────────────────────

def resolve_paths(config: dict) -> dict:
    """
    Return effective path dict: config.yaml values, overridden by env vars.
    Supported overrides: DATA_DIR, INDEX_DIR, OUTPUT_DIR.
    """
    raw = config.get("paths", {})

    def _resolve(key: str, env_var: str, default: str) -> Path:
        env_val = os.environ.get(env_var)
        if env_val:
            return Path(env_val)
        return ROOT / raw.get(key, default)

    data_dir   = _resolve("data_dir",    "DATA_DIR",   "data/")
    index_dir  = _resolve("indices_dir", "INDEX_DIR",  "indices/")
    output_dir = _resolve("output_dir",  "OUTPUT_DIR", "output/")

    split_csv = {
        "train": data_dir / Path(raw.get("train_csv", "data/train.csv")).name,
        "val":   data_dir / "val.csv",
        "test":  data_dir / Path(raw.get("test_csv",  "data/test.csv")).name,
    }

    return {
        "data_dir":   data_dir,
        "index_dir":  index_dir,
        "output_dir": output_dir,
        "laws":       data_dir / "laws_de.csv",
        "courts":     data_dir / "court_considerations.csv",
        "split_csv":  split_csv,
        "candidates": output_dir / "candidates.jsonl",
        "reranked":   output_dir / "reranked.jsonl",
        "submission": output_dir / "submission.csv",
        "bm25_laws":       index_dir / "bm25_laws.pkl",
        "bm25_courts":     index_dir / "bm25_courts.pkl",
        "faiss_laws":      index_dir / "dense_laws.faiss",
        "faiss_courts":    index_dir / "dense_courts.faiss",
    }


# ── Per-step IO description ───────────────────────────────────────────────────

def step_io(step_num: int, paths: dict, split: str, skip_dense: bool) -> tuple[list[str], list[str]]:
    """Return (inputs, outputs) as lists of path strings for display."""
    p = paths
    if step_num == 1:
        inputs  = [str(p["laws"]), str(p["courts"])]
        outputs = [str(p["bm25_laws"]), str(p["bm25_courts"])]
        if not skip_dense:
            outputs += [str(p["faiss_laws"]), str(p["faiss_courts"])]
    elif step_num == 2:
        inputs  = [str(p["split_csv"][split]), str(p["bm25_laws"]), str(p["bm25_courts"])]
        if not skip_dense:
            inputs += [str(p["faiss_laws"]), str(p["faiss_courts"])]
        outputs = [str(p["candidates"])]
    elif step_num == 3:
        inputs  = [str(p["candidates"])]
        outputs = [str(p["reranked"])]
    elif step_num == 4:
        inputs  = [str(p["reranked"]), str(p["split_csv"]["train"])]
        outputs = ["(stdout — macro F1 scores)"]
    elif step_num == 5:
        inputs  = [str(p["reranked"])]
        outputs = [str(p["submission"])]
    else:
        inputs = outputs = []
    return inputs, outputs


# ── Execution ─────────────────────────────────────────────────────────────────

def print_step_header(step_num: int, paths: dict, split: str, skip_dense: bool, dry_run: bool) -> None:
    label = STEPS[step_num][1]
    inputs, outputs = step_io(step_num, paths, split, skip_dense)

    prefix = "[DRY-RUN] " if dry_run else ""
    print(f"\n{'='*62}")
    print(f"  {prefix}Step {step_num}: {label}")
    print(f"{'='*62}")
    print("  Input:")
    for i in inputs:
        exists = Path(i).exists()
        flag = "✓" if exists else "✗ MISSING"
        print(f"    [{flag}] {i}")
    print("  Output:")
    for o in outputs:
        print(f"    →  {o}")


def run_step(step_num: int, extra_args: list[str], config_path: str, paths: dict,
             split: str, skip_dense: bool, dry_run: bool) -> bool:
    print_step_header(step_num, paths, split, skip_dense, dry_run)

    if dry_run:
        print("  (skipped — dry-run mode)")
        return True

    script = Path(__file__).parent / STEPS[step_num][0]
    cmd = [sys.executable, str(script), "--config", config_path] + extra_args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Step {step_num} failed (exit code {result.returncode})")
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the Swiss legal citation retrieval pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to config YAML (default: config/config.yaml)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Query split to retrieve/evaluate (default: test)")
    parser.add_argument("--steps", nargs="+", type=int, metavar="N",
                        help="Only run these step numbers, e.g. --steps 2 3")
    parser.add_argument("--skip", nargs="+", type=int, metavar="N",
                        help="Skip these step numbers, e.g. --skip 1")
    parser.add_argument("--skip-dense", action="store_true",
                        help="BM25-only mode — skip FAISS dense index & retrieval")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the full execution plan (inputs/outputs) without running anything")
    parser.add_argument("--validate-corpus", action="store_true",
                        help="Step 5: filter citations not found in corpus")
    parser.add_argument("--per-query", action="store_true",
                        help="Step 4: print per-query F1 breakdown")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    paths = resolve_paths(config)

    # ── Determine steps ───────────────────────────────────────────────────────
    selected = set(args.steps) if args.steps else set(STEPS.keys())
    skipped  = set(args.skip)  if args.skip  else set()
    steps_to_run = sorted(selected - skipped)

    if not steps_to_run:
        print("[ERROR] No steps to run after applying --steps and --skip.")
        sys.exit(1)

    if 4 in steps_to_run and args.split == "test":
        print("[INFO] Step 4 (evaluate) requires gold labels — skipped for test split.")
        steps_to_run = [s for s in steps_to_run if s != 4]

    # ── Print plan header ─────────────────────────────────────────────────────
    env_overrides = {k: os.environ[k] for k in ("DATA_DIR", "INDEX_DIR", "OUTPUT_DIR") if k in os.environ}

    print("╔══════════════════════════════════════════════════════════╗")
    print("║         Swiss Legal Retrieval Pipeline                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Config:       {args.config}")
    print(f"  Split:        {args.split}")
    print(f"  Steps:        {steps_to_run}")
    print(f"  Dense index:  {'DISABLED (BM25 only)' if args.skip_dense else 'ENABLED'}")
    print(f"  Claude rerank:{'ENABLED' if 3 in steps_to_run else 'N/A (step 3 not running)'}")
    if env_overrides:
        print(f"  Env overrides: {env_overrides}")
    if args.dry_run:
        print("\n  *** DRY-RUN MODE — nothing will be executed ***")

    # ── Build per-step arg lists ──────────────────────────────────────────────
    common = ["--config", str(ROOT / args.config)]

    step_args: dict[int, list[str]] = {
        1: common + (["--skip-dense"] if args.skip_dense else []),
        2: common + ["--split", args.split] + (["--skip-dense"] if args.skip_dense else []),
        3: common,
        4: common + (["--per-query"] if args.per_query else []),
        5: common + (["--validate-corpus"] if args.validate_corpus else []),
    }

    # ── Execute ───────────────────────────────────────────────────────────────
    for step_num in steps_to_run:
        ok = run_step(
            step_num,
            step_args[step_num],
            str(ROOT / args.config),
            paths,
            args.split,
            args.skip_dense,
            args.dry_run,
        )
        if not ok:
            print(f"\nPipeline aborted at step {step_num}.")
            sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    if args.dry_run:
        print("  Dry-run complete — no steps were executed.")
    else:
        print("  Pipeline complete!")
        if 5 in steps_to_run and paths["submission"].exists():
            size_kb = paths["submission"].stat().st_size / 1024
            print(f"  Submission: {paths['submission']}  ({size_kb:.1f} KB)")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
