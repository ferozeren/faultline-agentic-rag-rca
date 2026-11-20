"""Tests for the common document retriever interface."""

import pytest

from app.rag import retriever


def test_retrieve_documents_standardizes_ranked_results(monkeypatch):
    query = "Payment API database connection errors"
    metadata = {"incident_id": "incident-1", "severity": "HIGH"}
    ranked_result = {
        "chunk_id": "incident-1_chunk_0",
        "text": "Database connection pool exhaustion caused API latency.",
        "metadata": metadata,
        "rrf_score": 0.03252,
        "retrieval_sources": ["semantic", "bm25"],
        "semantic_rank": 1,
        "bm25_rank": 2,
        "distance": 0.82,
        "score": 8.42,
    }
    calls = []

    def fake_ranked_hybrid_search(received_query, top_k):
        calls.append((received_query, top_k))
        return [ranked_result]

    monkeypatch.setattr(retriever, "ranked_hybrid_search", fake_ranked_hybrid_search)

    documents = retriever.retrieve_documents(query, top_k=3)

    assert calls == [(query, 3)]
    assert len(documents) == 1
    document = documents[0]
    assert document["chunk_id"] == "incident-1_chunk_0"
    assert document["text"] == ranked_result["text"]
    assert document["metadata"] == metadata
    assert document["score"] == 0.03252
    assert document["retrieval_method"] == "hybrid_rrf"
    assert document["semantic_rank"] == 1
    assert document["bm25_rank"] == 2
    assert document["retrieval_sources"] == ["semantic", "bm25"]
    assert document["distance"] == 0.82
    assert document["bm25_score"] == 8.42
    assert document["rrf_score"] == 0.03252


def test_retrieve_documents_respects_top_k(monkeypatch):
    results = [
        {"chunk_id": str(index), "rrf_score": 1 / (60 + index)}
        for index in range(1, 4)
    ]
    monkeypatch.setattr(retriever, "ranked_hybrid_search", lambda query, top_k: results)

    documents = retriever.retrieve_documents("database latency", top_k=2)

    assert [document["chunk_id"] for document in documents] == ["1", "2"]


def test_retrieve_documents_returns_empty_list(monkeypatch):
    monkeypatch.setattr(retriever, "ranked_hybrid_search", lambda query, top_k: [])

    assert retriever.retrieve_documents("database latency") == []


@pytest.mark.parametrize("query", ["", "  ", None])
def test_retrieve_documents_rejects_empty_or_invalid_query(query):
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        retriever.retrieve_documents(query)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_retrieve_documents_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve_documents("database latency", top_k=top_k)