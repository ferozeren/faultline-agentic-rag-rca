"""Keyword retrieval over the saved historical incident chunks."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = (
    BACKEND_ROOT / "data" / "processed" / "cleaned_documents_chunks.jsonl"
)
DEFAULT_TOP_K = 5
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    """Keep technical identifiers intact and also index their components."""
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(text.lower()):
        tokens.append(token)
        if "-" in token or "_" in token:
            tokens.extend(re.split(r"[-_]", token))
    return tokens


def load_chunk_documents(
    input_path: Path = DEFAULT_CHUNKS_PATH,
) -> list[Document]:
    """Load saved JSONL chunks as LangChain Documents."""
    documents: list[Document] = []
    with input_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Invalid chunk record on line {line_number}.")

            chunk_id = record.get("chunk_id")
            text = record.get("text")
            metadata = record.get("metadata")
            if (
                not isinstance(chunk_id, str)
                or not chunk_id
                or not isinstance(text, str)
                or not text.strip()
                or not isinstance(metadata, Mapping)
            ):
                raise ValueError(f"Invalid chunk record on line {line_number}.")

            document_metadata = dict(metadata)
            document_metadata["chunk_id"] = chunk_id
            documents.append(
                Document(
                    id=chunk_id,
                    page_content=text,
                    metadata=document_metadata,
                )
            )

    if not documents:
        raise ValueError(f"No chunks found in {input_path}.")
    return documents


def create_bm25_retriever(documents: list[Document]) -> BM25Retriever:
    """Build a LangChain BM25 retriever from already loaded chunk Documents."""
    if not documents:
        raise ValueError("At least one document is required to build BM25.")
    return BM25Retriever.from_documents(
        documents,
        preprocess_func=_tokenize,
        k=DEFAULT_TOP_K,
    )


def bm25_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    input_path: Path = DEFAULT_CHUNKS_PATH,
) -> list[dict[str, Any]]:
    """Return matching chunks ranked by descending BM25 relevance score."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    query_tokens = _tokenize(query)
    if not query_tokens:
        raise ValueError("query must contain searchable terms.")

    retriever = create_bm25_retriever(load_chunk_documents(input_path))
    scores = retriever.vectorizer.get_scores(query_tokens)
    ranked_documents = sorted(
        zip(retriever.docs, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [
        {
            "chunk_id": document.metadata["chunk_id"],
            "text": document.page_content,
            "metadata": document.metadata,
            "score": float(score),
        }
        for document, score in ranked_documents[:top_k]
    ]


def main() -> int:
    """Run a sample keyword search against the saved chunk JSONL."""
    query = "Payment API database connection errors and high latency"
    results = bm25_search(query)
    print(f"Query: {query}")
    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. {result['chunk_id']} (BM25 score: {result['score']:.4f})")
        print(result["text"][:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
