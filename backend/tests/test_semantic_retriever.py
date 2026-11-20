"""Tests for semantic search using precomputed temporary Chroma vectors."""

import pytest
from langchain_chroma import Chroma

from app.rag.semantic_retriever import semantic_search
from app.rag.vector_store import COLLECTION_NAME


class FakeEmbeddings:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.0, 0.0, 0.0]


def create_test_collection(persist_directory):
    collection = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_directory),
        embedding_function=None,
    )
    collection._collection.upsert(
        ids=["far", "closest", "middle"],
        embeddings=[[2.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0]],
        documents=["far text", "closest text", "middle text"],
        metadatas=[
            {"chunk_id": "far", "incident_id": "incident_far"},
            {"chunk_id": "closest", "incident_id": "incident_closest"},
            {"chunk_id": "middle", "incident_id": "incident_middle"},
        ],
    )
    return collection


def test_semantic_search_embeds_query_and_orders_by_distance(tmp_path):
    persist_directory = tmp_path / "chroma"
    collection = create_test_collection(persist_directory)
    original_count = collection._collection.count()
    embeddings = FakeEmbeddings()

    results = semantic_search(
        "database latency",
        top_k=2,
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    assert embeddings.queries == ["database latency"]
    assert [result["chunk_id"] for result in results] == ["closest", "middle"]
    assert results[0]["distance"] < results[1]["distance"]
    assert results[0]["text"] == "closest text"
    assert results[0]["metadata"]["incident_id"] == "incident_closest"
    assert collection._collection.count() == original_count


def test_semantic_search_returns_requested_result_fields(tmp_path):
    persist_directory = tmp_path / "chroma"
    create_test_collection(persist_directory)

    results = semantic_search(
        "database latency",
        persist_directory=persist_directory,
        embedding_function=FakeEmbeddings(),
    )

    assert len(results) == 3
    assert set(results[0]) == {"chunk_id", "text", "metadata", "distance"}


@pytest.mark.parametrize("query", ["", "  ", None])
def test_semantic_search_rejects_empty_or_invalid_query(query, tmp_path):
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        semantic_search(
            query,
            persist_directory=tmp_path / "chroma",
            embedding_function=FakeEmbeddings(),
        )


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_semantic_search_rejects_invalid_top_k(top_k, tmp_path):
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        semantic_search(
            "database latency",
            top_k=top_k,
            persist_directory=tmp_path / "chroma",
            embedding_function=FakeEmbeddings(),
        )
