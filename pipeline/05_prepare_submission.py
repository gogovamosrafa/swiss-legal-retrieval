"""
Generate final submission.csv from reranked results.

Reads:  output/reranked.jsonl
        data/federal_laws.jsonl  (corpus — for citation validation)
        data/court_decisions.jsonl
Writes: output/submission.csv
        query_id,predicted_citations
"""

import sys
import json
import argparse
from pathlib import Path

import yaml
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from omnilex.citations.normalizer import CitationNormalizer
from omnilex.evaluation.scorer import validate_submission_format
from omnilex.retrieval.bm25_index import load_jsonl_corpus


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_corpus_citations(laws_path: Path, courts_path: Path) -> set[str]:
    """Return the set of canonical citation IDs present in the corpus."""
    normalizer = CitationNormalizer()
    valid = set()

    for path in [laws_path, courts_path]:
        if not path.exists():
            continue
        docs = load_jsonl_corpus(str(path))
        for doc in docs:
            raw = doc.get("citation") or doc.get("id") or ""
            canonical = normalizer.canonicalize(raw)
            if canonical:
                valid.add(canonical)

    return valid


def main():
    parser = argparse.ArgumentParser(description="Prepare Kaggle submission CSV")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input", default=None, help="Override reranked.jsonl path")
    parser.add_argument("--output", default=None, help="Override submission.csv path")
    parser.add_argument("--validate-corpus", action="store_true",
                        help="Filter out citations not found in the corpus")
    parser.add_argument("--test-csv", default=None,
                        help="test.csv path; ensures all query_ids are present in submission")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = cfg["paths"]
    eval_cfg = cfg["evaluation"]
    sep = eval_cfg["citation_separator"]

    output_dir = ROOT / paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input) if args.input else output_dir / "reranked.jsonl"
    output_path = Path(args.output) if args.output else output_dir / "submission.csv"

    if not input_path.exists():
        print(f"[ERROR] Reranked file not found: {input_path}")
        print("  Run: python pipeline/03_rerank.py")
        sys.exit(1)

    corpus_cids: set[str] | None = None
    if args.validate_corpus:
        print("Loading corpus for citation validation...")
        corpus_cids = load_corpus_citations(
            ROOT / paths["federal_laws_corpus"],
            ROOT / paths["court_decisions_corpus"],
        )
        print(f"  Corpus contains {len(corpus_cids)} canonical citations")

    normalizer = CitationNormalizer()

    rows = []
    stats = {"total_queries": 0, "total_citations": 0, "filtered": 0, "empty": 0}

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            query_id = record["query_id"]
            raw_citations = record.get("final_citations", [])
            stats["total_queries"] += 1

            canonical_citations = []
            for raw in raw_citations:
                canonical = normalizer.canonicalize(raw)
                if not canonical:
                    continue
                if corpus_cids is not None and canonical not in corpus_cids:
                    stats["filtered"] += 1
                    continue
                canonical_citations.append(canonical)

            # deduplicate while preserving order
            seen = set()
            deduped = []
            for c in canonical_citations:
                if c not in seen:
                    seen.add(c)
                    deduped.append(c)

            stats["total_citations"] += len(deduped)
            if not deduped:
                stats["empty"] += 1

            rows.append({
                "query_id": query_id,
                "predicted_citations": sep.join(deduped),
            })

    submission_df = pd.DataFrame(rows)

    # Ensure all test query_ids are present
    test_csv_path = Path(args.test_csv) if args.test_csv else ROOT / paths["test_csv"]
    if test_csv_path.exists():
        test_df = pd.read_csv(test_csv_path)
        missing_ids = set(test_df["query_id"]) - set(submission_df["query_id"])
        if missing_ids:
            print(f"[WARN] {len(missing_ids)} query_ids from test.csv missing in output — adding empty rows")
            empty_rows = pd.DataFrame({
                "query_id": list(missing_ids),
                "predicted_citations": [""] * len(missing_ids),
            })
            submission_df = pd.concat([submission_df, empty_rows], ignore_index=True)
        submission_df = submission_df.merge(
            test_df[["query_id"]], on="query_id", how="right"
        ).fillna("")

    submission_df.to_csv(output_path, index=False)

    print(f"\n=== Submission stats ===")
    print(f"  Queries:                {stats['total_queries']}")
    print(f"  Total citations:        {stats['total_citations']}")
    print(f"  Queries with 0 cits:   {stats['empty']}")
    if args.validate_corpus:
        print(f"  Citations filtered:     {stats['filtered']}")
    print(f"\nSaved → {output_path}")

    # Run format validation
    errors = validate_submission_format(str(output_path))
    if errors:
        print("\n[WARN] Submission format issues:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("  Format validation: PASSED")


if __name__ == "__main__":
    main()
