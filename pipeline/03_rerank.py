"""
Claude API reranking of hybrid retrieval candidates.

Reads:  output/candidates.jsonl
Writes: output/reranked.jsonl
        {query_id, query, reranked: [{citation, text, score}, ...]}

Each query's top-K candidates are sent to Claude, which returns
a relevance score (0.0–1.0) for each candidate.
Only candidates above final_threshold are kept.
"""

import sys
import json
import time
import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

RERANK_PROMPT_TEMPLATE = """\
You are an expert in Swiss law. Given a legal query and a list of candidate citations, \
score each citation's relevance to the query on a scale from 0.0 (not relevant) to 1.0 (highly relevant).

Query: {query}

Candidates:
{candidates_block}

Return a JSON array with one object per candidate in the same order:
[{{"citation": "<citation>", "score": <0.0-1.0>}}, ...]

Only return the JSON array, no other text."""


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_candidates_block(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates, start=1):
        text_snippet = (c.get("text") or "")[:300].replace("\n", " ")
        lines.append(f"{i}. [{c['citation']}] {text_snippet}")
    return "\n".join(lines)


def rerank_with_claude(
    client,
    query: str,
    candidates: list[dict],
    model: str,
    max_tokens: int,
) -> list[dict]:
    """Call Claude to score candidates; returns list with added 'rerank_score'."""
    prompt = RERANK_PROMPT_TEMPLATE.format(
        query=query,
        candidates_block=format_candidates_block(candidates),
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    scored = json.loads(raw)

    cid_to_score = {item["citation"]: float(item["score"]) for item in scored}

    result = []
    for c in candidates:
        cid = c["citation"]
        result.append({**c, "rerank_score": cid_to_score.get(cid, 0.0)})
    return result


def main():
    parser = argparse.ArgumentParser(description="Claude API reranking")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input", default=None, help="Override candidates.jsonl path")
    parser.add_argument("--output", default=None, help="Override reranked.jsonl path")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls, use random scores")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = cfg["paths"]
    rerank_cfg = cfg["reranking"]

    output_dir = ROOT / paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input) if args.input else output_dir / "candidates.jsonl"
    output_path = Path(args.output) if args.output else output_dir / "reranked.jsonl"

    if not input_path.exists():
        print(f"[ERROR] Candidates file not found: {input_path}")
        print("  Run: python pipeline/02_retrieve.py")
        sys.exit(1)

    model = rerank_cfg["rerank_model"]
    rerank_top_k = rerank_cfg["rerank_top_k"]
    threshold = rerank_cfg["final_threshold"]
    max_tokens = rerank_cfg["max_tokens"]

    client = None
    if not args.dry_run:
        try:
            import anthropic
            client = anthropic.Anthropic()
        except ImportError:
            print("[ERROR] anthropic package not installed. Run: pip install anthropic")
            sys.exit(1)

    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Reranking {len(records)} queries with {model}...")
    print(f"  top_k={rerank_top_k}, threshold={threshold}")

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, record in enumerate(records):
            query_id = record["query_id"]
            query_text = record["query"]
            candidates = record["candidates"][:rerank_top_k]

            if args.dry_run:
                import random
                reranked = [{**c, "rerank_score": random.random()} for c in candidates]
            else:
                try:
                    reranked = rerank_with_claude(client, query_text, candidates, model, max_tokens)
                    time.sleep(0.2)  # gentle rate limiting
                except Exception as e:
                    print(f"  [WARN] Query {query_id} reranking failed: {e}. Using RRF scores.")
                    reranked = [{**c, "rerank_score": c.get("rrf_score", 0.0)} for c in candidates]

            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            kept = [c for c in reranked if c["rerank_score"] >= threshold]

            out = {
                "query_id": query_id,
                "query": query_text,
                "reranked": reranked,
                "final_citations": [c["citation"] for c in kept],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(records)} done")

    print(f"\nSaved reranked results → {output_path}")


if __name__ == "__main__":
    main()
