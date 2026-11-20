"""Tests for storing precomputed vectors in persistent ChromaDB."""

import json

from langchain_chroma import Chroma

from app.rag.vector_store import store_embeddings


def test_stores_and_persists_precomputed_vectors(tmp_path):
    input_path = tmp_path / "embeddings.jsonl"
    persist_directory = tmp_path / "chroma"
    records = [
        {
            "chunk_id": f"incident_chunk_{index}",
            "text": f"incident text {index}",
            "metadata": {"incident_id": f"incident_{index}", "chunk_index": index},
            "embedding": [float(index), 0.5, 0.25],
        }
        for index in range(3)
    ]
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    assert store_embeddings(input_path, persist_directory, batch_size=2) == 3
    assert store_embeddings(input_path, persist_directory, batch_size=2) == 3

    reopened_store = Chroma(
        collection_name="historical_incidents",
        persist_directory=str(persist_directory),
        embedding_function=None,
    )
    stored_records = reopened_store.get(
        include=["documents", "metadatas", "embeddings"]
    )

    assert len(stored_records["ids"]) == 3
    assert set(stored_records["ids"]) == {record["chunk_id"] for record in records}
    stored_by_id = dict(zip(stored_records["ids"], stored_records["documents"]))
    assert stored_by_id == {
        record["chunk_id"]: record["text"] for record in records
    }
