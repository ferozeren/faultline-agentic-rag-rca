"""Tests for batched embedding output using a fake LangChain embedder."""

import json

from app.rag.embedding import EMBEDDING_DIMENSIONS, generate_embeddings


class FakeEmbeddings:
    def __init__(self):
        self.batches = []

    def embed_documents(self, texts):
        self.batches.append(list(texts))
        return [
            [float(index) / EMBEDDING_DIMENSIONS] * EMBEDDING_DIMENSIONS
            for index, _ in enumerate(texts)
        ]


def test_generates_embedding_jsonl_in_batches(tmp_path):
    input_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "processed" / "embeddings.jsonl"
    chunks = [
        {
            "chunk_id": f"INC-001_chunk_{index:03d}",
            "text": f"chunk text {index}",
            "metadata": {"incident_id": "INC-001", "chunk_index": index},
        }
        for index in range(3)
    ]
    input_path.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    embedder = FakeEmbeddings()

    processed = generate_embeddings(
        input_path,
        output_path,
        batch_size=2,
        embedder=embedder,
    )

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert processed == len(chunks)
    assert len(embedder.batches) == 2
    assert len(records) == len(chunks)
    assert records[0]["chunk_id"] == chunks[0]["chunk_id"]
    assert records[0]["text"] == chunks[0]["text"]
    assert records[0]["metadata"] == chunks[0]["metadata"]
    assert len(records[0]["embedding"]) == EMBEDDING_DIMENSIONS
    assert all(isinstance(value, float) for value in records[0]["embedding"])
