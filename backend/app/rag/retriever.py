"""Common document retrieval interface for downstream components."""

from __future__ import annotations

from typing import Any

from app.rag.ranking import DEFAULT_TOP_K, ranked_hybrid_search


def retrieve_documents(
	query: str,
	top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
	"""Return RRF-ranked documents in a consistent downstream format."""
	if not isinstance(query, str) or not query.strip():
		raise ValueError("query must be a non-empty string.")
	if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
		raise ValueError("top_k must be a positive integer.")

	ranked_results = ranked_hybrid_search(query, top_k=top_k)
	documents = []
	for result in ranked_results[:top_k]:
		document = {
			**result,
			"score": result["rrf_score"],
			"retrieval_method": "hybrid_rrf",
		}
		if "score" in result:
			document["bm25_score"] = result["score"]
		documents.append(document)
	return documents


def main() -> int:
	"""Run the common retriever against the saved incident data."""
	query = "Payment API database connection errors and high latency"
	documents = retrieve_documents(query)
	print(f"Query: {query}")
	for rank, document in enumerate(documents, start=1):
		print(
			f"\n{rank}. {document['chunk_id']} | "
			f"RRF: {document['score']:.6f} | "
			f"retrieval method: {document['retrieval_method']} | "
			f"semantic rank: {document['semantic_rank']} | "
			f"BM25 rank: {document['bm25_rank']}"
		)
		print(document.get("text", "")[:500])
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
