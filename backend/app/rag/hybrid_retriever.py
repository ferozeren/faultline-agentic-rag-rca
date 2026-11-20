"""Collect semantic and BM25 results without combining their rankings."""

from __future__ import annotations

from typing import Any

from app.rag.bm25_retriever import bm25_search
from app.rag.semantic_retriever import semantic_search

DEFAULT_TOP_K = 5


def hybrid_search(query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Return independently ranked semantic and keyword results."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    semantic_results = [
        {**result, "retrieval_method": "semantic"}
        for result in semantic_search(query, top_k=top_k)
    ]
    bm25_results = [
        {**result, "retrieval_method": "bm25"}
        for result in bm25_search(query, top_k=top_k)
    ]

    return {
        "query": query,
        "semantic_results": semantic_results,
        "bm25_results": bm25_results,
        "candidates": semantic_results + bm25_results,
    }


def main() -> int:
    """Run an example hybrid search against the saved incident data."""
    query = "Payment API database connection errors and high latency"
    results = hybrid_search(query)
    print(f"Query: {results['query']}")

    for result_set in ("semantic_results", "bm25_results"):
        print(f"\n{result_set.replace('_', ' ').title()}:")
        for result in results[result_set]:
            score_key = "distance" if result["retrieval_method"] == "semantic" else "score"
            print(
                f"- {result['chunk_id']} | {score_key}: {result[score_key]} | "
                f"retrieval_method: {result['retrieval_method']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())