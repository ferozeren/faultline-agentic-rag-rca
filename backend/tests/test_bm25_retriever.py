"""Tests for BM25 keyword retrieval using chunk JSONL data."""

import json

import pytest
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.rag.bm25_retriever import (
    bm25_search,
    create_bm25_retriever,
    load_chunk_documents,
)


def write_chunks(path, chunks):
    path.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def chunks_file(tmp_path):
    chunks = [
        {
            "chunk_id": "payment_chunk",
            "text": (
                "Payment API experienced ConnectionTimeout and HTTP 503 errors "
                "after database connection pool exhaustion."
            ),
            "metadata": {"incident_id": "payment-1", "severity": "HIGH"},
        },
        {
            "chunk_id": "oom_chunk",
            "text": "Worker restart caused OOMKilled errors in the cache service.",
            "metadata": {"incident_id": "cache-1", "severity": "MEDIUM"},
        },
        {
            "chunk_id": "latency_chunk",
            "text": "Database connection failures caused high latency for the API.",
            "metadata": {"incident_id": "database-1", "severity": "HIGH"},
        },
        {
            "chunk_id": "unrelated_chunk",
            "text": "Static website certificate renewal completed successfully.",
            "metadata": {"incident_id": "web-1", "severity": "LOW"},
        },
    ]
    return write_chunks(tmp_path / "chunks.jsonl", chunks)


def test_loads_chunks_as_documents_with_preserved_ids_and_metadata(chunks_file):
    documents = load_chunk_documents(chunks_file)

    assert len(documents) == 4
    assert isinstance(documents[0], Document)
    assert documents[0].id == "payment_chunk"
    assert documents[0].page_content.startswith("Payment API")
    assert documents[0].metadata == {
        "incident_id": "payment-1",
        "severity": "HIGH",
        "chunk_id": "payment_chunk",
    }


def test_initializes_langchain_bm25_retriever(chunks_file):
    retriever = create_bm25_retriever(load_chunk_documents(chunks_file))

    assert isinstance(retriever, BM25Retriever)
    assert len(retriever.docs) == 4


def test_search_ranks_exact_and_technical_terms_by_score(chunks_file):
    results = bm25_search("payment-api ConnectionTimeout HTTP 503", input_path=chunks_file)

    assert results[0]["chunk_id"] == "payment_chunk"
    assert results[0]["score"] > results[1]["score"]
    assert results == sorted(results, key=lambda result: result["score"], reverse=True)


def test_search_returns_requested_result_structure(chunks_file):
    results = bm25_search("database connection failures", top_k=2, input_path=chunks_file)

    assert len(results) == 2
    assert set(results[0]) == {"chunk_id", "text", "metadata", "score"}
    assert all(isinstance(result["score"], float) for result in results)


def test_search_matches_error_codes_and_infrastructure_tokens(chunks_file):
    assert (
        bm25_search("HTTP 503", top_k=1, input_path=chunks_file)[0]["chunk_id"]
        == "payment_chunk"
    )
    assert (
        bm25_search("OOMKilled", top_k=1, input_path=chunks_file)[0]["chunk_id"]
        == "oom_chunk"
    )


def test_loads_existing_chunk_jsonl():
    documents = load_chunk_documents()

    assert documents
    assert all(isinstance(document, Document) for document in documents)


@pytest.mark.parametrize("query", ["", "  ", None])
def test_rejects_empty_or_invalid_query(query, chunks_file):
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        bm25_search(query, input_path=chunks_file)


def test_rejects_query_without_searchable_tokens(chunks_file):
    with pytest.raises(ValueError, match="query must contain searchable terms"):
        bm25_search("---___", input_path=chunks_file)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_rejects_invalid_top_k(top_k, chunks_file):
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        bm25_search("payment api", top_k=top_k, input_path=chunks_file)
