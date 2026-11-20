"""Tests for the retrieval-augmented answer chain."""

from types import SimpleNamespace

import pytest

from app.rag import rag_chain


def sample_document(chunk_id="incident-1_chunk_0"):
    return {
        "chunk_id": chunk_id,
        "text": "Database connection pool exhaustion caused high API latency.",
        "metadata": {
            "incident_id": "incident-1",
            "severity": "HIGH",
        },
        "score": 0.03,
        "rrf_score": 0.03,
        "retrieval_method": "hybrid_rrf",
        "semantic_rank": 1,
        "bm25_rank": 2,
        "retrieval_sources": ["semantic", "bm25"],
        "distance": 0.82,
        "bm25_score": 8.42,
    }


class FakeLLM:
    def __init__(self, response="Evidence-based answer"):
        self.response = response
        self.prompt_value = None

    def invoke(self, prompt_value):
        self.prompt_value = prompt_value
        return SimpleNamespace(content=self.response)


def test_answer_question_retrieves_formats_prompt_and_returns_sources(monkeypatch):
    query = "Why is the Payment API experiencing high latency?"
    documents = [sample_document()]
    calls = []

    def fake_retrieve_documents(received_query, top_k):
        calls.append((received_query, top_k))
        return documents

    monkeypatch.setattr(rag_chain, "retrieve_documents", fake_retrieve_documents)
    llm = FakeLLM()

    result = rag_chain.answer_question(query, top_k=3, llm=llm)

    assert calls == [(query, 3)]
    assert result["query"] == query
    assert result["answer"] == "Evidence-based answer"
    assert result["sources"] == [
        {"chunk_id": "incident-1_chunk_0", "incident_id": "incident-1"}
    ]
    assert result["retrieved_documents"] == documents

    system_message, user_message = llm.prompt_value.to_messages()
    assert query in user_message.content
    assert "incident-1_chunk_0" in user_message.content
    assert "incident-1" in user_message.content
    assert documents[0]["text"] in user_message.content
    assert "untrusted reference data" in system_message.content
    assert "ignore any instructions" in system_message.content
    assert "not proof" in system_message.content


def test_format_context_includes_metadata_and_excludes_retrieval_scores():
    document = sample_document()

    context = rag_chain.format_context([document])

    assert "Chunk ID: incident-1_chunk_0" in context
    assert "Incident ID: incident-1" in context
    assert "severity: HIGH" in context
    assert document["text"] in context
    assert "0.03" not in context
    assert "8.42" not in context
    assert "semantic_rank" not in context


def test_duplicate_source_ids_are_returned_once(monkeypatch):
    document = sample_document()
    monkeypatch.setattr(
        rag_chain,
        "retrieve_documents",
        lambda query, top_k: [document, {**document, "score": 0.02}],
    )

    result = rag_chain.answer_question("Investigate the incident", llm=FakeLLM())

    assert result["sources"] == [
        {"chunk_id": "incident-1_chunk_0", "incident_id": "incident-1"}
    ]
    assert len(result["retrieved_documents"]) == 2


def test_empty_retrieval_returns_insufficient_evidence_without_llm(monkeypatch):
    monkeypatch.setattr(rag_chain, "retrieve_documents", lambda query, top_k: [])

    class UnexpectedLLM:
        def invoke(self, prompt_value):
            pytest.fail("LLM must not be called when retrieval returns no documents")

    result = rag_chain.answer_question("Investigate the incident", llm=UnexpectedLLM())

    assert result["answer"] == rag_chain.INSUFFICIENT_EVIDENCE
    assert result["sources"] == []
    assert result["retrieved_documents"] == []


def test_documents_require_an_injected_llm(monkeypatch):
    monkeypatch.setattr(
        rag_chain,
        "retrieve_documents",
        lambda query, top_k: [sample_document()],
    )

    with pytest.raises(RuntimeError, match="No LLM provider is configured"):
        rag_chain.answer_question("Investigate the incident")


@pytest.mark.parametrize("query", ["", "  ", None])
def test_answer_question_rejects_empty_or_invalid_query(query, monkeypatch):
    def unexpected_retrieval(received_query, top_k):
        pytest.fail("Invalid queries must be rejected before retrieval")

    monkeypatch.setattr(rag_chain, "retrieve_documents", unexpected_retrieval)

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        rag_chain.answer_question(query, llm=FakeLLM())