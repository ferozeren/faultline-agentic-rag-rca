"""Tests for the hybrid retrieval result collector."""

import pytest

from app.rag import hybrid_retriever


def test_hybrid_search_calls_both_retrievers_and_preserves_results(monkeypatch):
    query = "Payment API database connection errors and high latency"
    semantic_result = {
        "chunk_id": "shared_chunk",
        "text": "Semantic result text",
        "metadata": {"incident_id": "semantic-1"},
        "distance": 0.82,
    }
    bm25_result = {
        "chunk_id": "shared_chunk",
        "text": "BM25 result text",
        "metadata": {"incident_id": "bm25-1"},
        "score": 8.42,
    }
    calls = []

    def fake_semantic_search(received_query, top_k):
        calls.append(("semantic", received_query, top_k))
        return [semantic_result]

    def fake_bm25_search(received_query, top_k):
        calls.append(("bm25", received_query, top_k))
        return [bm25_result]

    monkeypatch.setattr(hybrid_retriever, "semantic_search", fake_semantic_search)
    monkeypatch.setattr(hybrid_retriever, "bm25_search", fake_bm25_search)

    results = hybrid_retriever.hybrid_search(query, top_k=3)

    assert calls == [("semantic", query, 3), ("bm25", query, 3)]
    assert results["query"] == query
    assert results["semantic_results"] == [
        {**semantic_result, "retrieval_method": "semantic"}
    ]
    assert results["bm25_results"] == [{**bm25_result, "retrieval_method": "bm25"}]
    assert results["semantic_results"][0]["distance"] == 0.82
    assert results["bm25_results"][0]["score"] == 8.42
    assert results["semantic_results"][0]["metadata"] == semantic_result["metadata"]
    assert results["bm25_results"][0]["metadata"] == bm25_result["metadata"]
    assert [candidate["chunk_id"] for candidate in results["candidates"]] == [
        "shared_chunk",
        "shared_chunk",
    ]


@pytest.mark.parametrize("query", ["", "  ", None])
def test_hybrid_search_rejects_empty_or_invalid_query(query):
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        hybrid_retriever.hybrid_search(query)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_hybrid_search_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        hybrid_retriever.hybrid_search("database latency", top_k=top_k)


def test_hybrid_search_propagates_retriever_failure(monkeypatch):
    def failing_semantic_search(query, top_k):
        raise RuntimeError("semantic retrieval failed")

    def unexpected_bm25_search(query, top_k):
        pytest.fail("BM25 should not run after semantic retrieval fails")

    monkeypatch.setattr(hybrid_retriever, "semantic_search", failing_semantic_search)
    monkeypatch.setattr(hybrid_retriever, "bm25_search", unexpected_bm25_search)

    with pytest.raises(RuntimeError, match="semantic retrieval failed"):
        hybrid_retriever.hybrid_search("database latency")