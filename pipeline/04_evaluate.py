"""
Evaluate predictions against gold labels, reporting macro F1.

Reads:  output/reranked.jsonl  OR  a submission CSV
        data/train.csv          (gold labels, train split)
Prints: macro F1, precision, recall + per-query breakdown
"""

import sys
import json
import argparse
from pathlib import Path

import yaml
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from omnilex.evaluation.scorer import Scorer, evaluate_submission
from omnilex.citations.normalizer import CitationNormalizer


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def reranked_to_submission(reranked_path: Path, sep: str = ";") -> pd.DataFrame:
    rows = []
    with open(reranked_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            citations = record.get("final_citations", [])
            rows.append({
                "query_id": record["query_id"],
                "predicted_citations": sep.join(citations) if citations else "",
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate macro F1 on train split")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--predictions", default=None,
                        help="Path to reranked.jsonl or submission CSV. "
                             "Default: output/reranked.jsonl")
    parser.add_argument("--gold", default=None,
                        help="Path to gold CSV. Default: data/train.csv")
    parser.add_argument("--per-query", action="store_true",
                        help="Print per-query F1 scores")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = cfg["paths"]
    eval_cfg = cfg["evaluation"]
    sep = eval_cfg["citation_separator"]

    output_dir = ROOT / paths["output_dir"]
    predictions_path = Path(args.predictions) if args.predictions else output_dir / "reranked.jsonl"
    gold_path = Path(args.gold) if args.gold else ROOT / paths["train_csv"]

    if not predictions_path.exists():
        print(f"[ERROR] Predictions not found: {predictions_path}")
        sys.exit(1)
    if not gold_path.exists():
        print(f"[ERROR] Gold file not found: {gold_path}")
        sys.exit(1)

    # Load predictions
    if predictions_path.suffix == ".jsonl":
        pred_df = reranked_to_submission(predictions_path, sep)
    else:
        pred_df = pd.read_csv(predictions_path)

    gold_df = pd.read_csv(gold_path)

    # Align on shared query_ids
    shared_ids = set(pred_df["query_id"]) & set(gold_df["query_id"])
    if not shared_ids:
        print("[ERROR] No overlapping query_ids between predictions and gold.")
        sys.exit(1)

    pred_df = pred_df[pred_df["query_id"].isin(shared_ids)].copy()
    gold_df = gold_df[gold_df["query_id"].isin(shared_ids)].copy()

    print(f"Evaluating {len(shared_ids)} queries...")

    metrics = evaluate_submission(pred_df, gold_df, metrics=eval_cfg["metrics"])

    print("\n=== Results ===")
    print(f"  Macro F1:        {metrics.get('macro_f1', 0):.4f}")
    print(f"  Macro Precision: {metrics.get('macro_precision', 0):.4f}")
    print(f"  Macro Recall:    {metrics.get('macro_recall', 0):.4f}")
    if "micro_f1" in metrics:
        print(f"  Micro F1:        {metrics.get('micro_f1', 0):.4f}")

    if args.per_query:
        normalizer = CitationNormalizer()
        scorer = Scorer(normalizer, citation_separator=sep)

        pred_map = dict(zip(pred_df["query_id"], pred_df["predicted_citations"].fillna("")))
        gold_map = dict(zip(gold_df["query_id"], gold_df["gold_citations"].fillna("")))

        from omnilex.evaluation.metrics import citation_f1

        print("\n=== Per-query F1 ===")
        rows = []
        for qid in sorted(shared_ids):
            pred_cits = set(scorer.parse_citations(pred_map.get(qid, "")))
            gold_cits = set(scorer.parse_citations(gold_map.get(qid, "")))
            q_metrics = citation_f1(pred_cits, gold_cits)
            rows.append((qid, q_metrics["f1"], q_metrics["precision"], q_metrics["recall"]))
            print(f"  {qid:20s}  F1={q_metrics['f1']:.3f}  P={q_metrics['precision']:.3f}  R={q_metrics['recall']:.3f}")

    return metrics


if __name__ == "__main__":
    main()
