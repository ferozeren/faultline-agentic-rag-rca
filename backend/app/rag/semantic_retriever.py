"""Semantic search over the persistent historical incident collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import RAG_EMBEDDING_MODEL
from app.rag.vector_store import (
    COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
)

DEFAULT_TOP_K = 5


def semantic_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    persist_directory: Path = DEFAULT_PERSIST_DIRECTORY,
    embedding_function: Any | None = None,
) -> list[dict[str, Any]]:
    """Return the closest stored chunks, ordered by ascending Chroma distance."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if embedding_function is None:
        embedding_function = HuggingFaceEmbeddings(
            model_name=RAG_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_directory),
        embedding_function=embedding_function,
    )
    matches = vector_store.similarity_search_with_score(query.strip(), k=top_k)
    return [
        {
            "chunk_id": document.metadata["chunk_id"],
            "text": document.page_content,
            "metadata": document.metadata,
            "distance": float(distance),
        }
        for document, distance in matches
    ]


def main() -> int:
    """Run a small example query against the saved incident collection."""
    query = "Payment API is experiencing high latency and database connection errors"
    results = semantic_search(query)
    print(f"Query: {query}")
    for rank, result in enumerate(results, start=1):
        print(
            f"\n{rank}. {result['chunk_id']} "
            f"(Chroma distance: {result['distance']:.4f})"
        )
        print(result["text"][:500])
        print(f"Metadata: {result['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
