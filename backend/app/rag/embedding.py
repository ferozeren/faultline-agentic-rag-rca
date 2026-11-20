"""Generate normalized LangChain embeddings for saved incident chunks."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import RAG_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    BACKEND_ROOT / "data" / "processed" / "cleaned_documents_chunks.jsonl"
)
DEFAULT_OUTPUT_PATH = BACKEND_ROOT / "data" / "processed" / "embeddings.jsonl"
EMBEDDING_DIMENSIONS = 384
BATCH_SIZE = 32


def _read_batches(
    input_path: Path,
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    """Read chunk JSONL records in batches."""
    batch: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"Invalid chunk record on line {line_number}.")
            batch.append(record)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def generate_embeddings(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    batch_size: int = BATCH_SIZE,
    embedder: Any | None = None,
) -> int:
    """Embed the input chunk JSONL and write chunk records with vectors."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Chunk input file not found: {input_path}")

    if embedder is None:
        embedder = HuggingFaceEmbeddings(
            model_name=RAG_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_chunks = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for batch in _read_batches(input_path, batch_size):
            vectors = embedder.embed_documents(
                [record["text"] for record in batch]
            )
            if len(vectors) != len(batch):
                raise ValueError("Embedding count does not match the input batch size.")

            for record, vector in zip(batch, vectors, strict=True):
                if len(vector) != EMBEDDING_DIMENSIONS:
                    raise ValueError(
                        f"Expected {EMBEDDING_DIMENSIONS} embedding values, "
                        f"received {len(vector)}."
                    )
                output_record = {
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "metadata": record["metadata"],
                    "embedding": [float(value) for value in vector],
                }
                output_file.write(
                    json.dumps(output_record, ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
                processed_chunks += 1

    return processed_chunks


def main() -> int:
    """Generate embeddings from the canonical chunk file."""
    input_path = Path(
        os.getenv("RAG_CHUNKS_PATH", str(DEFAULT_INPUT_PATH))
    ).expanduser()
    output_path = Path(
        os.getenv("RAG_EMBEDDINGS_PATH", str(DEFAULT_OUTPUT_PATH))
    ).expanduser()
    if not input_path.is_absolute():
        input_path = BACKEND_ROOT / input_path
    if not output_path.is_absolute():
        output_path = BACKEND_ROOT / output_path

    start_time = time.perf_counter()
    try:
        processed_chunks = generate_embeddings(input_path, output_path)
    except (
        FileNotFoundError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ):
        logger.exception("Embedding generation failed.")
        return 1

    elapsed_seconds = time.perf_counter() - start_time
    print(f"Processed chunks: {processed_chunks}")
    print(f"Embedding model: {RAG_EMBEDDING_MODEL}")
    print(f"Embedding dimensions: {EMBEDDING_DIMENSIONS}")
    print(f"Output file: {output_path}")
    print(f"Execution time: {elapsed_seconds:.2f} seconds")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
