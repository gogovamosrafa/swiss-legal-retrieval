"""
Build BM25 and dense (FAISS) indices from the Swiss legal corpus.

Outputs to indices/:
  - indices/bm25_laws.pkl
  - indices/bm25_courts.pkl
  - indices/dense_laws.faiss + indices/dense_laws_meta.jsonl
  - indices/dense_courts.faiss + indices/dense_courts_meta.jsonl
"""

import sys
import json
import pickle
import argparse
from pathlib import Path

import yaml
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from omnilex.retrieval.bm25_index import BM25Index, load_jsonl_corpus


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_bm25(documents: list[dict], output_path: Path, text_field: str = "text") -> None:
    print(f"  Building BM25 index for {len(documents)} documents...")
    index = BM25Index(text_field=text_field, citation_field="citation")
    index.build(documents)
    index.save(str(output_path))
    print(f"  Saved BM25 index → {output_path}")


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
        print("  [SKIP] faiss or sentence-transformers not installed. Dense index skipped.")
        return

    print(f"  Loading dense model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [prefix + (doc.get("text") or doc.get("regeste") or "") for doc in documents]
    print(f"  Encoding {len(texts)} passages (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    if normalize:
        index = faiss.IndexFlatIP(dim)   # inner product == cosine when normalized
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(faiss_path))
    print(f"  Saved FAISS index ({index.ntotal} vectors, dim={dim}) → {faiss_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  Saved metadata → {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Build BM25 + dense indices")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense index building")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = cfg["paths"]
    dense_cfg = cfg["dense"]

    indices_dir = ROOT / paths["indices_dir"]
    indices_dir.mkdir(parents=True, exist_ok=True)

    laws_path = ROOT / paths["federal_laws_corpus"]
    courts_path = ROOT / paths["court_decisions_corpus"]

    if not laws_path.exists():
        print(f"[ERROR] Corpus not found: {laws_path}")
        print("  Run: python scripts/download_data.py")
        sys.exit(1)

    print("=== Loading corpora ===")
    laws = load_jsonl_corpus(str(laws_path))
    courts = load_jsonl_corpus(str(courts_path))
    print(f"  Federal laws: {len(laws)} documents")
    print(f"  Court decisions: {len(courts)} documents")

    print("\n=== Building BM25 indices ===")
    build_bm25(laws, indices_dir / "bm25_laws.pkl")
    build_bm25(courts, indices_dir / "bm25_courts.pkl")

    if not args.skip_dense:
        print("\n=== Building dense (FAISS) indices ===")
        build_dense(
            laws,
            faiss_path=indices_dir / "dense_laws.faiss",
            meta_path=indices_dir / "dense_laws_meta.jsonl",
            model_name=dense_cfg["model_name"],
            batch_size=dense_cfg["batch_size"],
            max_length=dense_cfg["max_length"],
            normalize=dense_cfg["normalize_embeddings"],
            prefix=dense_cfg["prefix_passage"],
        )
        build_dense(
            courts,
            faiss_path=indices_dir / "dense_courts.faiss",
            meta_path=indices_dir / "dense_courts_meta.jsonl",
            model_name=dense_cfg["model_name"],
            batch_size=dense_cfg["batch_size"],
            max_length=dense_cfg["max_length"],
            normalize=dense_cfg["normalize_embeddings"],
            prefix=dense_cfg["prefix_passage"],
        )
    else:
        print("\n[SKIP] Dense index building skipped (--skip-dense)")

    print("\n=== Done ===")
    for p in sorted(indices_dir.iterdir()):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:40s}  {size_mb:7.1f} MB")


if __name__ == "__main__":
    main()
