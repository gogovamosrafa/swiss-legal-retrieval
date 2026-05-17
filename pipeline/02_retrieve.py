"""
Hybrid retrieval: BM25 + dense (FAISS), fused with Reciprocal Rank Fusion (RRF).

Reads:  data/test.csv (or train.csv)
        indices/bm25_*.pkl
        indices/dense_*.faiss + dense_*_meta.jsonl
Writes: output/candidates.jsonl
        {query_id, query, candidates: [{citation, text, score, source}, ...]}
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import yaml
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from omnilex.retrieval.bm25_index import BM25Index, load_jsonl_corpus


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_bm25(index_path: Path) -> BM25Index:
    idx = BM25Index()
    idx.load(str(index_path))
    return idx


def bm25_search(index: BM25Index, query: str, top_k: int) -> list[tuple[dict, float]]:
    results = index.search(query, top_k=top_k, return_scores=True)
    if results and isinstance(results[0], tuple):
        return results  # already (doc, score)
    # fallback: BM25Index.search may return just docs
    scores = index.bm25.get_scores(index.tokenize(query))
    docs = index.documents
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def dense_search(model, faiss_index, meta: list[dict], query: str, top_k: int, prefix: str, normalize: bool):
    import numpy as np
    q_emb = model.encode([prefix + query], normalize_embeddings=normalize)
    q_emb = q_emb.astype("float32")
    scores, indices = faiss_index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append((meta[idx], float(score)))
    return results


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    weights: list[float],
    k: int = 60,
) -> dict[str, float]:
    """
    ranked_lists: list of [(citation_id, score), ...] already sorted best-first
    Returns: {citation_id: rrf_score}
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    for ranked, weight in zip(ranked_lists, weights):
        for rank, (cid, _) in enumerate(ranked, start=1):
            rrf_scores[cid] += weight / (rank + k)
    return rrf_scores


def main():
    parser = argparse.ArgumentParser(description="Hybrid BM25 + dense retrieval")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                        help="Which split to retrieve for")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--output", default=None, help="Override output path")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = cfg["paths"]
    ret_cfg = cfg["retrieval"]
    dense_cfg = cfg["dense"]

    indices_dir = ROOT / paths["indices_dir"]
    output_dir = ROOT / paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_key = "train_csv" if args.split == "train" else "test_csv"
    queries_df = pd.read_csv(ROOT / paths[csv_key])
    print(f"Loaded {len(queries_df)} queries from {paths[csv_key]}")

    print("Loading BM25 indices...")
    bm25_laws = load_bm25(indices_dir / "bm25_laws.pkl")
    bm25_courts = load_bm25(indices_dir / "bm25_courts.pkl")

    use_dense = not args.skip_dense
    dense_model = None
    dense_laws_index = dense_courts_index = None
    dense_laws_meta = dense_courts_meta = []

    if use_dense:
        laws_faiss = indices_dir / "dense_laws.faiss"
        courts_faiss = indices_dir / "dense_courts.faiss"
        if not laws_faiss.exists():
            print("[WARN] Dense index not found, falling back to BM25-only")
            use_dense = False
        else:
            try:
                import faiss
                from sentence_transformers import SentenceTransformer
                print(f"Loading dense model: {dense_cfg['model_name']}")
                dense_model = SentenceTransformer(dense_cfg["model_name"])
                dense_laws_index = faiss.read_index(str(laws_faiss))
                dense_courts_index = faiss.read_index(str(indices_dir / "dense_courts.faiss"))
                dense_laws_meta = load_jsonl_corpus(str(indices_dir / "dense_laws_meta.jsonl"))
                dense_courts_meta = load_jsonl_corpus(str(indices_dir / "dense_courts_meta.jsonl"))
                print(f"  Laws FAISS: {dense_laws_index.ntotal} vectors")
                print(f"  Courts FAISS: {dense_courts_index.ntotal} vectors")
            except ImportError:
                print("[WARN] faiss/sentence-transformers not available, BM25-only mode")
                use_dense = False

    output_path = Path(args.output) if args.output else output_dir / "candidates.jsonl"
    bm25_top_k = ret_cfg["bm25_top_k"]
    dense_top_k = ret_cfg["dense_top_k"]
    rrf_k = ret_cfg["rrf_k"]
    rrf_weights = [ret_cfg["rrf_bm25_weight"], ret_cfg["rrf_dense_weight"]]

    print(f"\nRetrieving candidates for {len(queries_df)} queries...")
    with open(output_path, "w", encoding="utf-8") as fout:
        for _, row in queries_df.iterrows():
            query_id = row["query_id"]
            query_text = row["query"]

            # BM25 retrieval (laws + courts combined)
            bm25_law_results = bm25_search(bm25_laws, query_text, bm25_top_k)
            bm25_court_results = bm25_search(bm25_courts, query_text, bm25_top_k)
            bm25_all = bm25_law_results + bm25_court_results
            bm25_ranked = [(doc.get("citation", doc.get("id", "")), score)
                           for doc, score in sorted(bm25_all, key=lambda x: x[1], reverse=True)]

            all_docs: dict[str, dict] = {}
            for doc, _ in bm25_all:
                cid = doc.get("citation", doc.get("id", ""))
                all_docs[cid] = doc

            ranked_lists = [bm25_ranked]

            if use_dense:
                ql = dense_search(
                    dense_model, dense_laws_index, dense_laws_meta,
                    query_text, dense_top_k,
                    dense_cfg["prefix_query"], dense_cfg["normalize_embeddings"]
                )
                qc = dense_search(
                    dense_model, dense_courts_index, dense_courts_meta,
                    query_text, dense_top_k,
                    dense_cfg["prefix_query"], dense_cfg["normalize_embeddings"]
                )
                dense_all = ql + qc
                dense_ranked = [(doc.get("citation", doc.get("id", "")), score)
                                for doc, score in sorted(dense_all, key=lambda x: x[1], reverse=True)]
                for doc, _ in dense_all:
                    cid = doc.get("citation", doc.get("id", ""))
                    if cid not in all_docs:
                        all_docs[cid] = doc
                ranked_lists.append(dense_ranked)

            rrf_scores = reciprocal_rank_fusion(ranked_lists, rrf_weights[:len(ranked_lists)], k=rrf_k)
            sorted_cids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

            candidates = []
            for cid in sorted_cids:
                doc = all_docs.get(cid, {})
                candidates.append({
                    "citation": cid,
                    "text": doc.get("text") or doc.get("regeste") or "",
                    "title": doc.get("title") or "",
                    "rrf_score": rrf_scores[cid],
                    "source": "hybrid" if use_dense else "bm25",
                })

            record = {
                "query_id": query_id,
                "query": query_text,
                "candidates": candidates,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved candidates → {output_path}")
    print(f"Mode: {'BM25 + Dense (RRF)' if use_dense else 'BM25-only'}")


if __name__ == "__main__":
    main()
