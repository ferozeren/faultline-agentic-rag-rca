"""Store precomputed incident chunk embeddings in persistent ChromaDB."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = BACKEND_ROOT / "data" / "processed" / "embeddings.jsonl"
DEFAULT_PERSIST_DIRECTORY = BACKEND_ROOT / "data" / "vector_db" / "chroma"
COLLECTION_NAME = "historical_incidents"
BATCH_SIZE = 100


def _read_batches(
    input_path: Path,
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    """Read embedding records from JSONL in manageable batches."""
    batch: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def store_embeddings(
    input_path: Path = DEFAULT_INPUT_PATH,
    persist_directory: Path = DEFAULT_PERSIST_DIRECTORY,
    *,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Upsert saved vectors and return the number of input records stored."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Embedding input file not found: {input_path}")

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=None,
    )
    stored_count = 0
    for batch in _read_batches(input_path, batch_size):
        # LangChain Chroma has no public method for adding precomputed vectors.
        vector_store._collection.upsert(
            ids=[record["chunk_id"] for record in batch],
            embeddings=[record["embedding"] for record in batch],
            documents=[record["text"] for record in batch],
            metadatas=[record["metadata"] for record in batch],
        )
        stored_count += len(batch)
    return stored_count


def main() -> int:
    """Store the canonical embeddings file in the persistent collection."""
    stored_count = store_embeddings()
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vectors inserted: {stored_count}")
    print(f"Database directory: {DEFAULT_PERSIST_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
