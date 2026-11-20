"""Tests for Reciprocal Rank Fusion over retrieval result lists."""

import pytest

from app.rag import ranking


def result(chunk_id, **fields):
    return {"chunk_id": chunk_id, **fields}


def test_rrf_uses_one_based_rank_positions():
    fused = ranking.reciprocal_rank_fusion(
        [result("first"), result("second")],
        [],
        top_k=2,
    )

    assert fused[0]["chunk_id"] == "first"
    assert fused[0]["rrf_score"] == pytest.approx(1 / 61)
    assert fused[0]["semantic_rank"] == 1
    assert fused[0]["bm25_rank"] is None
    assert fused[1]["rrf_score"] == pytest.approx(1 / 62)
    assert fused[1]["semantic_rank"] == 2


def test_rrf_merges_duplicates_and_adds_both_contributions():
    fused = ranking.reciprocal_rank_fusion(
        [result("chunk_A"), result("chunk_B")],
        [result("chunk_B"), result("chunk_A")],
        top_k=2,
    )

    by_id = {item["chunk_id"]: item for item in fused}
    assert len(fused) == 2
    assert by_id["chunk_A"]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["chunk_B"]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["chunk_A"]["semantic_rank"] == 1
    assert by_id["chunk_A"]["bm25_rank"] == 2
    assert by_id["chunk_A"]["retrieval_sources"] == ["semantic", "bm25"]


def test_rrf_keeps_single_source_documents_and_sorts_by_score():
    fused = ranking.reciprocal_rank_fusion(
        [result("shared"), result("semantic_only")],
        [result("shared"), result("bm25_only")],
        top_k=3,
    )

    assert [item["chunk_id"] for item in fused] == [
        "shared",
        "semantic_only",
        "bm25_only",
    ]
    semantic_only, bm25_only = fused[1:]
    assert semantic_only["retrieval_sources"] == ["semantic"]
    assert semantic_only["semantic_rank"] == 2
    assert semantic_only["bm25_rank"] is None
    assert bm25_only["retrieval_sources"] == ["bm25"]
    assert bm25_only["semantic_rank"] is None
    assert bm25_only["bm25_rank"] == 2


def test_rrf_preserves_document_fields_and_original_scores():
    metadata = {"incident_id": "incident-1", "severity": "HIGH"}
    semantic = result(
        "shared",
        text="Original incident text",
        metadata=metadata,
        distance=0.82,
        retrieval_method="semantic",
    )
    bm25 = result(
        "shared",
        text="Original incident text",
        metadata=metadata,
        score=8.42,
        retrieval_method="bm25",
    )

    fused = ranking.reciprocal_rank_fusion([semantic], [bm25])[0]

    assert fused["text"] == "Original incident text"
    assert fused["metadata"] == metadata
    assert fused["distance"] == 0.82
    assert fused["score"] == 8.42
    assert "retrieval_method" not in fused


def test_rrf_limits_results_to_top_k():
    fused = ranking.reciprocal_rank_fusion(
        [result("a"), result("b"), result("c")],
        [],
        top_k=2,
    )

    assert [item["chunk_id"] for item in fused] == ["a", "b"]


@pytest.mark.parametrize(
    ("semantic_results", "bm25_results"),
    [([], []), ([result("semantic")], []), ([], [result("bm25")])],
)
def test_rrf_handles_empty_retrieval_lists(semantic_results, bm25_results):
    fused = ranking.reciprocal_rank_fusion(semantic_results, bm25_results)

    assert len(fused) == len(semantic_results) + len(bm25_results)


def test_ranked_hybrid_search_uses_existing_hybrid_results(monkeypatch):
    query = "database latency"
    semantic_results = [result("semantic", distance=0.4)]
    bm25_results = [result("keyword", score=2.5)]
    calls = []

    def fake_hybrid_search(received_query, top_k):
        calls.append((received_query, top_k))
        return {
            "query": received_query,
            "semantic_results": semantic_results,
            "bm25_results": bm25_results,
            "candidates": semantic_results + bm25_results,
        }

    monkeypatch.setattr(ranking, "hybrid_search", fake_hybrid_search)

    fused = ranking.ranked_hybrid_search(query, top_k=1)

    assert calls == [(query, 1)]
    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "semantic"
    assert fused[0]["distance"] == 0.4


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_rrf_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        ranking.reciprocal_rank_fusion([], [], top_k=top_k)


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_rrf_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        ranking.reciprocal_rank_fusion([], [], k=k)