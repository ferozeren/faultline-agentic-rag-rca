"""Mocked tests for the end-to-end RAG pipeline validation helpers."""

from types import SimpleNamespace

import pytest

from app.rag import rag_chain, validate_rag_pipeline as validation


def document(chunk_id, **fields):
    return {
        "chunk_id": chunk_id,
        "text": f"Database connection pool evidence for {chunk_id}.",
        "metadata": {"incident_id": chunk_id.split("_chunk")[0]},
        **fields,
    }


class FakeChatModel:
    def __init__(self, content="Evidence indicates connection pool exhaustion."):
        self.content = content
        self.prompt = None

    def invoke(self, prompt):
        self.prompt = prompt
        return SimpleNamespace(content=self.content)


def test_collection_count_uses_existing_collection(monkeypatch, tmp_path):
    calls = []

    class FakeCollection:
        def count(self):
            return 17

    class FakeClient:
        def __init__(self, path):
            calls.append(("path", path))

        def get_collection(self, name):
            calls.append(("collection", name))
            return FakeCollection()

    monkeypatch.setattr(validation.chromadb, "PersistentClient", FakeClient)

    assert validation.collection_document_count() == 17
    assert calls == [
        ("path", str(validation.DEFAULT_PERSIST_DIRECTORY)),
        ("collection", "historical_incidents"),
    ]


@pytest.mark.parametrize(
    ("results", "fields", "expected"),
    [
        ([{"chunk_id": "s", "text": "x", "metadata": {}, "distance": 0.2}],
         {"chunk_id", "text", "metadata", "distance"}, True),
        ([{"chunk_id": "b", "text": "x", "metadata": {}, "score": 1.0}],
         {"chunk_id", "text", "metadata", "score"}, True),
        ([], {"chunk_id"}, False),
    ],
)
def test_semantic_and_bm25_result_validation(results, fields, expected):
    assert validation.valid_results(results, fields) is expected


def test_hybrid_results_and_rrf_merge_duplicates(monkeypatch):
    semantic = [document("a_chunk_0", distance=0.2), document("b_chunk_0", distance=0.4)]
    bm25 = [document("a_chunk_0", score=2.1), document("c_chunk_0", score=1.2)]
    monkeypatch.setattr(validation, "collection_document_count", lambda: 10)
    monkeypatch.setattr(validation, "semantic_search", lambda query, top_k: semantic)
    monkeypatch.setattr(validation, "bm25_search", lambda query, top_k: bm25)
    monkeypatch.setattr(
        validation,
        "hybrid_search",
        lambda query, top_k: {
            "semantic_results": semantic,
            "bm25_results": bm25,
        },
    )
    standard = [
        document(
            result["chunk_id"],
            score=0.03,
            rrf_score=0.03,
            retrieval_method="hybrid_rrf",
            retrieval_sources=["semantic", "bm25"],
        )
        for result in (semantic[0], semantic[1], bm25[1])
    ]
    monkeypatch.setattr(validation, "retrieve_documents", lambda query, top_k: standard)
    monkeypatch.setattr(
        validation,
        "answer_question",
        lambda query, top_k, llm: {
            "answer": "Possible cause is connection pool exhaustion; see a_chunk_0.",
            "sources": [{"chunk_id": "a_chunk_0", "incident_id": "a"}],
            "retrieved_documents": standard,
        },
    )
    llm = validation.CapturingChatModel(FakeChatModel())

    report = validation.run_pipeline_validation(llm)

    assert report["candidate_duplicate_count"] == 1
    assert len(report["fused_results"]) == 3
    assert len({item["chunk_id"] for item in report["fused_results"]}) == 3
    assert report["checks"]["Hybrid search"]
    assert report["checks"]["RRF ranking"]
    assert report["checks"]["Retriever integration"]


def test_rag_chain_receives_context_and_returns_source_references(monkeypatch):
    docs = [document("incident-a_chunk_0", score=0.03, rrf_score=0.03)]
    monkeypatch.setattr(rag_chain, "retrieve_documents", lambda query, top_k: docs)
    chat_model = FakeChatModel("The evidence indicates pool exhaustion; see incident-a_chunk_0.")
    captured = validation.CapturingChatModel(chat_model)

    result = rag_chain.answer_question("Investigate API latency", llm=captured)

    assert "incident-a_chunk_0" in str(chat_model.prompt.to_messages()[1].content)
    assert docs[0]["text"] in str(chat_model.prompt.to_messages()[1].content)
    assert result["answer"] == chat_model.content
    assert result["sources"] == [
        {"chunk_id": "incident-a_chunk_0", "incident_id": "incident-a"}
    ]
    assert result["retrieved_documents"] == docs


def test_empty_retrieval_does_not_call_llm(monkeypatch):
    monkeypatch.setattr(rag_chain, "retrieve_documents", lambda query, top_k: [])

    class UnexpectedModel:
        def invoke(self, prompt):
            pytest.fail("The LLM must not be called for empty retrieval results")

    result = rag_chain.answer_question("Investigate API latency", llm=UnexpectedModel())

    assert "Insufficient evidence" in result["answer"]
    assert result["sources"] == []


def test_llm_error_does_not_produce_a_fabricated_answer(monkeypatch):
    monkeypatch.setattr(validation, "collection_document_count", lambda: 1)
    retrieved = [document("incident-a_chunk_0", distance=0.1, score=1.0)]
    monkeypatch.setattr(validation, "semantic_search", lambda query, top_k: retrieved)
    monkeypatch.setattr(validation, "bm25_search", lambda query, top_k: retrieved)
    monkeypatch.setattr(
        validation,
        "hybrid_search",
        lambda query, top_k: {
            "semantic_results": retrieved,
            "bm25_results": retrieved,
        },
    )
    monkeypatch.setattr(validation, "retrieve_documents", lambda query, top_k: retrieved)
    monkeypatch.setattr(
        validation,
        "answer_question",
        lambda query, top_k, llm: (_ for _ in ()).throw(RuntimeError("API unavailable")),
    )

    report = validation.run_pipeline_validation(
        validation.CapturingChatModel(FakeChatModel())
    )

    assert report["answer_result"] is None
    assert report["answer"] == ""
    assert report["answer_error"] == "API unavailable"
    assert not report["checks"]["NVIDIA live API call"]
    assert not report["checks"]["Final answer generation"]