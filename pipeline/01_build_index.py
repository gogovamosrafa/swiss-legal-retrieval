"""
Build BM25 and dense (FAISS) indices from the Swiss legal corpus.

Source files:
  data/laws_de.csv              columns: citation, text, title  (175K rows)
  data/court_considerations.csv columns: citation, text         (2.47M rows)

Outputs to indices/:
  indices/bm25_laws.pkl
  indices/bm25_courts.pkl
  indices/dense_laws.faiss + indices/dense_laws_meta.jsonl   (laws only)

Note: dense index is built for laws only. court_considerations has 2.47M rows
which would produce a ~10 GB FAISS index — impractical locally.
Set dense.laws_only: false in config.yaml to override (requires large RAM/VRAM).
"""

import sys
import csv
import json
import argparse
from pathlib import Path

import yaml
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from omnilex.retrieval.bm25_index import BM25Index


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv_corpus(path: Path, max_rows: int = None) -> list[dict]:
    """Load citation corpus from CSV into a list of dicts."""
    docs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            docs.append(dict(row))
    return docs


def build_bm25(documents: list[dict], output_path: Path) -> None:
    print(f"  Building BM25 index for {len(documents):,} documents...")
    index = BM25Index(text_field="text", citation_field="citation")
    index.build(documents)
    index.save(str(output_path))
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Saved → {output_path}  ({size_mb:.0f} MB)")


def build_dense(
    documents: list[dict],
    faiss_path: Path,
    meta_path: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    normalize: bool,
    prefix: str,
) -> None:
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [SKIP] faiss or sentence-transformers not installed.")
        return

    print(f"  Loading dense model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [prefix + (doc.get("text") or "") for doc in documents]
    print(f"  Encoding {len(texts):,} passages (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) if normalize else faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(faiss_path))
    print(f"  Saved FAISS ({index.ntotal:,} vectors, dim={dim}) → {faiss_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  Saved metadata → {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Build BM25 + dense indices")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-dense", action="store_true",
                        help="Skip dense index building (BM25 only)")
    parser.add_argument("--max-court-rows", type=int, default=None,
                        help="Limit court_considerations rows (for testing)")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths     = cfg["paths"]
    dense_cfg = cfg["dense"]
    idx_cfg   = cfg.get("indexing", {})

    indices_dir = ROOT / paths["indices_dir"]
    indices_dir.mkdir(parents=True, exist_ok=True)

    laws_path   = ROOT / paths["laws_corpus"]
    courts_path = ROOT / paths["courts_corpus"]

    # max_court_rows: from --max-court-rows flag, then config, then None (all rows)
    max_court_rows = args.max_court_rows or idx_cfg.get("max_court_rows") or None

    for p in [laws_path, courts_path]:
        if not p.exists():
            print(f"[ERROR] Corpus not found: {p}")
            print("  Run: .venv/Scripts/kaggle competitions download "
                  "-c llm-agentic-legal-information-retrieval -p data/")
            sys.exit(1)

    # ── BM25 ─────────────────────────────────────────────────────────────────
    print("=== Loading laws corpus ===")
    laws = load_csv_corpus(laws_path)
    print(f"  {len(laws):,} articles loaded")

    print("\n=== Building BM25: laws ===")
    build_bm25(laws, indices_dir / "bm25_laws.pkl")

    skip_court_bm25 = idx_cfg.get("skip_court_bm25", True)

    if skip_court_bm25:
        print("\n[SKIP] Court BM25 skipped (indexing.skip_court_bm25=true)")
        print("       Reason: 2.47M rows causes MemoryError in rank-bm25 pickle")
        print("       Impact: minimal — only 1.2% of gold citations are BGE")
        courts = []
    else:
        print("\n=== Loading court corpus ===")
        print(f"  Streaming {courts_path.name} ({courts_path.stat().st_size/1e9:.1f} GB)...")
        if max_court_rows:
            print(f"  Limiting to {max_court_rows:,} rows (indexing.max_court_rows in config)")
        courts = load_csv_corpus(courts_path, max_rows=max_court_rows)
        print(f"  {len(courts):,} considerations loaded")

        print("\n=== Building BM25: courts ===")
        build_bm25(courts, indices_dir / "bm25_courts.pkl")
        del courts

    # ── Dense ────────────────────────────────────────────────────────────────
    laws_only = dense_cfg.get("laws_only", True)
    skip_dense = args.skip_dense

    if skip_dense:
        print("\n[SKIP] Dense index skipped (--skip-dense)")
    else:
        print("\n=== Building dense index: laws ===")
        build_dense(
            laws,
            faiss_path=indices_dir / "dense_laws.faiss",
            meta_path=indices_dir  / "dense_laws_meta.jsonl",
            model_name=dense_cfg["model_name"],
            batch_size=dense_cfg["batch_size"],
            max_length=dense_cfg["max_length"],
            normalize=dense_cfg["normalize_embeddings"],
            prefix=dense_cfg["prefix_passage"],
        )

        if not laws_only:
            print("\n=== Building dense index: courts ===")
            print("  [WARN] This will use ~10 GB disk and large RAM/VRAM")
            courts = load_csv_corpus(courts_path, max_rows=args.max_court_rows)
            build_dense(
                courts,
                faiss_path=indices_dir / "dense_courts.faiss",
                meta_path=indices_dir  / "dense_courts_meta.jsonl",
                model_name=dense_cfg["model_name"],
                batch_size=dense_cfg["batch_size"],
                max_length=dense_cfg["max_length"],
                normalize=dense_cfg["normalize_embeddings"],
                prefix=dense_cfg["prefix_passage"],
            )
        else:
            print("\n[SKIP] Dense index for courts skipped (dense.laws_only=true in config)")

    print("\n=== Done ===")
    for p in sorted(indices_dir.iterdir()):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:45s}  {size_mb:7.0f} MB")


if __name__ == "__main__":
    main()
