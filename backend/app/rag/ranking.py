"""Fuse semantic and BM25 rankings using Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Any

from app.rag.hybrid_retriever import hybrid_search

RRF_K = 60
DEFAULT_TOP_K = 5


def reciprocal_rank_fusion(
    semantic_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Merge retrieval lists by chunk ID and rank them using 1-based RRF."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer.")

    fused_by_id: dict[str, dict[str, Any]] = {}
    for source, results in (
        ("semantic", semantic_results),
        ("bm25", bm25_results),
    ):
        rank_field = f"{source}_rank"
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]
            fused = fused_by_id.get(chunk_id)
            if fused is None:
                fused = {
                    key: value
                    for key, value in result.items()
                    if key != "retrieval_method"
                }
                fused.update(
                    {
                        "rrf_score": 0.0,
                        "retrieval_sources": [],
                        "semantic_rank": None,
                        "bm25_rank": None,
                    }
                )
                fused_by_id[chunk_id] = fused
            else:
                for key, value in result.items():
                    if key not in fused and key != "retrieval_method":
                        fused[key] = value

            fused["rrf_score"] += 1 / (k + rank)
            if fused[rank_field] is None:
                fused[rank_field] = rank
            if source not in fused["retrieval_sources"]:
                fused["retrieval_sources"].append(source)

    ranked_results = sorted(
        fused_by_id.values(),
        key=lambda result: result["rrf_score"],
        reverse=True,
    )
    return ranked_results[:top_k]


def ranked_hybrid_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Run hybrid retrieval and return its top RRF-ranked results."""
    results = hybrid_search(query, top_k=top_k)
    return reciprocal_rank_fusion(
        results["semantic_results"],
        results["bm25_results"],
        top_k=top_k,
    )


def main() -> int:
    """Run and print an example RRF ranking against saved incident data."""
    query = "Payment API database connection errors and high latency"
    results = ranked_hybrid_search(query)
    print(f"Query: {query}")
    for rank, result in enumerate(results, start=1):
        print(
            f"\n{rank}. {result['chunk_id']} | "
            f"RRF: {result['rrf_score']:.6f} | "
            f"semantic rank: {result['semantic_rank']} | "
            f"BM25 rank: {result['bm25_rank']} | "
            f"sources: {', '.join(result['retrieval_sources'])}"
        )
        print(result.get("text", "")[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())