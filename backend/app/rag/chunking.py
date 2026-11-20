"""Token-aware chunking for historical incident LangChain Documents."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from app.config import (
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_EMBEDDING_MAX_TOKENS,
    RAG_EMBEDDING_MODEL,
    RAG_LOCAL_FILES_ONLY,
)

logger = logging.getLogger(__name__)

CHUNKING_STRATEGY = "recursive_character_token_aware"
SPLITTER_SEPARATORS = ("\n\n", "\n", ". ", " ", "")
BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = (
    BACKEND_ROOT / "data" / "processed" / "cleaned_documents_chunks.jsonl"
)
GENERATED_METADATA_FIELDS = {
    "body_token_count",
    "chunk_id",
    "chunk_index",
    "chunk_overlap",
    "chunk_size",
    "chunking_strategy",
    "embedding_input_token_count",
    "embedding_model",
    "parent_document_id",
    "total_chunks",
}


class ChunkingError(Exception):
    """Base exception for historical incident chunking failures."""


class TokenizerInitializationError(ChunkingError):
    """Raised when the configured Hugging Face tokenizer cannot be loaded."""


class ChunkValidationError(ChunkingError):
    """Raised when generated chunks do not satisfy the pipeline invariants."""


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str, local_files_only: bool) -> PreTrainedTokenizerBase:
    """Load each model tokenizer once for reuse across chunker instances."""
    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=local_files_only,
        )
    except Exception as error:
        logger.error(
            "Unable to initialize tokenizer for embedding model %s",
            model_name,
        )
        raise TokenizerInitializationError(
            f"Unable to load tokenizer for '{model_name}'. "
            "Check model availability and local-only settings."
        ) from error


class HistoricalIncidentChunker:
    """Split incident Documents with verified body and embedding token limits."""

    def __init__(
        self,
        chunk_size: int = RAG_CHUNK_SIZE,
        chunk_overlap: int = RAG_CHUNK_OVERLAP,
        embedding_max_tokens: int = RAG_EMBEDDING_MAX_TOKENS,
        embedding_model: str = RAG_EMBEDDING_MODEL,
        *,
        local_files_only: bool = RAG_LOCAL_FILES_ONLY,
    ) -> None:
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
            raise ValueError("chunk_overlap must be a non-negative integer.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and less than chunk_size."
            )
        if (
            not isinstance(embedding_max_tokens, int)
            or isinstance(embedding_max_tokens, bool)
            or embedding_max_tokens <= 0
        ):
            raise ValueError("embedding_max_tokens must be a positive integer.")
        if not isinstance(embedding_model, str) or not embedding_model.strip():
            raise ValueError("embedding_model must be a non-empty string.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_max_tokens = embedding_max_tokens
        self.embedding_model = embedding_model.strip()
        self.local_files_only = bool(local_files_only)
        self.tokenizer = _load_tokenizer(
            self.embedding_model,
            self.local_files_only,
        )

        if self._embedding_input_token_count("") > self.embedding_max_tokens:
            raise ValueError(
                "embedding_max_tokens is smaller than the tokenizer's special-token "
                "overhead."
            )

        self._reset_state()

    def chunk_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Create validated, deterministic chunks without mutating source Documents."""
        self._reset_state()
        source_documents = list(documents)
        self._source_documents = source_documents
        self._input_documents = len(source_documents)
        identities: Counter[str] = Counter()
        output_chunks: list[Document] = []

        for document_index, document in enumerate(source_documents):
            if not isinstance(document, Document):
                raise TypeError(
                    f"Input document {document_index} is not a LangChain Document."
                )

            base_identity = self._get_document_identity(document)
            identities[base_identity] += 1
            parent_id = base_identity
            if identities[base_identity] > 1:
                parent_id = f"{base_identity}_source_{document_index:04d}"

            metadata = self._normalize_metadata(document.metadata)
            metadata_snapshot = self._copy_metadata(document.metadata)
            self._source_metadata_by_parent[parent_id] = metadata
            self._source_metadata_snapshots[document_index] = metadata_snapshot
            self._source_content_snapshots[document_index] = document.page_content
            self._source_id_snapshots[document_index] = document.id

            if not document.page_content.strip():
                self._record_failure(
                    document_index,
                    parent_id,
                    "Document content is empty or whitespace-only.",
                )
                continue

            try:
                pieces = self._split_text_safely(document.page_content)
                self._validate_source_coverage(document.page_content, pieces)
            except ChunkingError as error:
                self._record_failure(document_index, parent_id, str(error))
                continue

            total_chunks = len(pieces)
            parent_chunks: list[Document] = []
            for chunk_index, piece in enumerate(pieces):
                body_tokens = self._body_token_count(piece)
                embedding_tokens = self._embedding_input_token_count(piece)
                chunk_id = f"{parent_id}_chunk_{chunk_index:03d}"
                chunk_metadata = dict(metadata)
                chunk_metadata.update(
                    {
                        "parent_document_id": parent_id,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "chunking_strategy": CHUNKING_STRATEGY,
                        "embedding_model": self.embedding_model,
                        "body_token_count": body_tokens,
                        "embedding_input_token_count": embedding_tokens,
                    }
                )
                parent_chunks.append(
                    Document(
                        id=chunk_id,
                        page_content=piece,
                        metadata=chunk_metadata,
                    )
                )

            self._processed_chunk_counts[parent_id] = total_chunks
            output_chunks.extend(parent_chunks)

        self._chunks = output_chunks
        self.validate_chunks(output_chunks)
        return list(output_chunks)

    def validate_chunks(self, chunks: Sequence[Document]) -> bool:
        """Validate chunk identity, metadata, ordering, and actual tokenizer limits."""
        errors: list[str] = []
        valid_documents = [
            chunk for chunk in chunks if isinstance(chunk, Document)
        ]
        id_counts = Counter(
            str(chunk.metadata.get("chunk_id", chunk.id or ""))
            for chunk in valid_documents
        )
        duplicate_ids = sum(count - 1 for count in id_counts.values() if count > 1)
        if duplicate_ids:
            errors.append(f"{duplicate_ids} duplicate chunk ID(s)")

        grouped: dict[str, list[Document]] = defaultdict(list)

        for chunk in chunks:
            if not isinstance(chunk, Document):
                errors.append("Output contains a non-LangChain Document.")
                continue
            if not chunk.page_content.strip():
                errors.append("Output contains an empty or whitespace-only chunk.")
                continue
            if self._is_heading_only(chunk.page_content):
                errors.append("Output contains a section-heading-only chunk.")
                continue

            metadata = chunk.metadata
            if any(
                not isinstance(value, (str, int, float, bool))
                or isinstance(value, float) and not math.isfinite(value)
                for value in metadata.values()
            ):
                errors.append("Chunk metadata contains a non-scalar Chroma value.")
            parent_id = metadata.get("parent_document_id")
            if not isinstance(parent_id, str) or not parent_id:
                errors.append("A chunk is missing its parent document identity.")
                continue
            if parent_id not in self._source_metadata_by_parent:
                errors.append(f"Chunk refers to unknown parent document '{parent_id}'.")
                continue

            chunk_id = metadata.get("chunk_id")
            if not isinstance(chunk_id, str) or chunk.id != chunk_id:
                errors.append(
                    "Chunk ID metadata does not match the LangChain Document ID."
                )
            grouped[parent_id].append(chunk)

            original_metadata = self._source_metadata_by_parent[parent_id]
            for key, value in original_metadata.items():
                if key not in GENERATED_METADATA_FIELDS and metadata.get(key) != value:
                    errors.append(
                        f"Parent metadata field '{key}' was not preserved for {parent_id}."
                    )

            incident_id = original_metadata.get("incident_id")
            if incident_id is not None and metadata.get("incident_id") != incident_id:
                errors.append(f"Incident ID was not preserved for {parent_id}.")

            actual_body_tokens = self._body_token_count(chunk.page_content)
            actual_embedding_tokens = self._embedding_input_token_count(
                chunk.page_content
            )
            if actual_body_tokens != metadata.get("body_token_count"):
                errors.append(f"Body token count is incorrect for {chunk_id}.")
            if actual_embedding_tokens != metadata.get("embedding_input_token_count"):
                errors.append(f"Embedding token count is incorrect for {chunk_id}.")
            if actual_body_tokens > self.chunk_size:
                errors.append(
                    f"Chunk {chunk_id} has {actual_body_tokens} body tokens; "
                    f"the limit is {self.chunk_size}."
                )
            if actual_embedding_tokens > self.embedding_max_tokens:
                errors.append(
                    f"Chunk {chunk_id} has {actual_embedding_tokens} embedding-input "
                    f"tokens; the limit is {self.embedding_max_tokens}."
                )

        for parent_id, parent_chunks in grouped.items():
            parent_chunks.sort(
                key=lambda item: int(item.metadata.get("chunk_index", -1))
            )
            expected_total = len(parent_chunks)
            indexes = [chunk.metadata.get("chunk_index") for chunk in parent_chunks]
            if indexes != list(range(expected_total)):
                errors.append(f"Chunk indexes are not sequential for {parent_id}.")
            for chunk in parent_chunks:
                if chunk.metadata.get("total_chunks") != expected_total:
                    errors.append(f"Total chunk count is incorrect for {parent_id}.")
                expected_id = (
                    f"{parent_id}_chunk_{chunk.metadata.get('chunk_index', -1):03d}"
                )
                if chunk.id != expected_id:
                    errors.append(f"Chunk ID is unstable or malformed for {parent_id}.")

        for parent_id, expected_count in self._processed_chunk_counts.items():
            if len(grouped.get(parent_id, ())) != expected_count:
                errors.append(f"Output chunks were lost for parent {parent_id}.")

        if len(self._processed_chunk_counts) + len(self._failed_documents) != (
            self._input_documents
        ):
            errors.append("One or more input documents were not accounted for.")

        for document_index, source in enumerate(self._source_documents):
            if document_index not in self._source_metadata_snapshots:
                errors.append(f"Input document {document_index} has no metadata snapshot.")
                continue
            try:
                current = self._copy_metadata(source.metadata)
            except ChunkingError as error:
                errors.append(f"Could not verify source metadata: {error}")
                continue
            if source.page_content != self._source_content_snapshots[document_index]:
                errors.append(f"Input document {document_index} content was modified.")
            if source.id != self._source_id_snapshots[document_index]:
                errors.append(f"Input document {document_index} ID was modified.")
            if current != self._source_metadata_snapshots[document_index]:
                errors.append(f"Input document {document_index} metadata was modified.")

        if errors:
            raise ChunkValidationError("Chunk validation failed: " + "; ".join(errors))
        return True

    def get_chunking_summary(self) -> dict[str, Any]:
        """Return calculated chunk counts, token distributions, and duplicate audits."""
        body_counts = [
            self._body_token_count(chunk.page_content) for chunk in self._chunks
        ]
        embedding_counts = [
            self._embedding_input_token_count(chunk.page_content)
            for chunk in self._chunks
        ]
        chunks_by_parent: dict[str, list[Document]] = defaultdict(list)
        for chunk in self._chunks:
            chunks_by_parent[str(chunk.metadata["parent_document_id"])].append(chunk)

        chunk_id_counts = Counter(
            str(chunk.metadata.get("chunk_id", chunk.id or ""))
            for chunk in self._chunks
        )
        fingerprints: dict[str, list[Document]] = defaultdict(list)
        for chunk in self._chunks:
            fingerprints[self._content_fingerprint(chunk.page_content)].append(chunk)

        duplicate_fingerprints = {
            fingerprint: matching
            for fingerprint, matching in fingerprints.items()
            if len(matching) > 1
        }
        same_document_duplicate_pairs = 0
        cross_document_duplicate_pairs = 0
        for matching in duplicate_fingerprints.values():
            per_parent = Counter(
                str(chunk.metadata["parent_document_id"]) for chunk in matching
            )
            same_document_duplicate_pairs += sum(
                count * (count - 1) // 2 for count in per_parent.values()
            )
            total_pairs = len(matching) * (len(matching) - 1) // 2
            cross_document_duplicate_pairs += total_pairs - sum(
                count * (count - 1) // 2 for count in per_parent.values()
            )

        document_chunk_counts = list(self._processed_chunk_counts.values())
        unusual_threshold, unusual_documents = self._unusually_large_documents(
            document_chunk_counts,
            chunks_by_parent,
        )

        return {
            "input_documents": self._input_documents,
            "successfully_processed_documents": len(self._processed_chunk_counts),
            "failed_documents": len(self._failed_documents),
            "failed_document_details": list(self._failed_documents),
            "total_output_chunks": len(self._chunks),
            "average_chunks_per_document": (
                mean(document_chunk_counts) if document_chunk_counts else 0.0
            ),
            "minimum_chunks_per_document": (
                min(document_chunk_counts) if document_chunk_counts else 0
            ),
            "maximum_chunks_per_document": (
                max(document_chunk_counts) if document_chunk_counts else 0
            ),
            "average_body_tokens": mean(body_counts) if body_counts else 0.0,
            "median_body_tokens": median(body_counts) if body_counts else 0.0,
            "p95_body_tokens": self._percentile(body_counts, 0.95),
            "p99_body_tokens": self._percentile(body_counts, 0.99),
            "minimum_body_tokens": min(body_counts) if body_counts else 0,
            "maximum_body_tokens": max(body_counts) if body_counts else 0,
            "average_embedding_input_tokens": (
                mean(embedding_counts) if embedding_counts else 0.0
            ),
            "minimum_embedding_input_tokens": (
                min(embedding_counts) if embedding_counts else 0
            ),
            "maximum_embedding_input_tokens": (
                max(embedding_counts) if embedding_counts else 0
            ),
            "chunks_below_50_tokens": sum(count < 50 for count in body_counts),
            "chunks_below_100_tokens": sum(count < 100 for count in body_counts),
            "chunks_above_body_limit": sum(
                count > self.chunk_size for count in body_counts
            ),
            "chunks_above_embedding_input_limit": sum(
                count > self.embedding_max_tokens for count in embedding_counts
            ),
            "empty_chunks": sum(
                not chunk.page_content.strip() for chunk in self._chunks
            ),
            "duplicate_chunk_ids": sum(
                count - 1 for count in chunk_id_counts.values() if count > 1
            ),
            "duplicate_content_fingerprints": len(duplicate_fingerprints),
            "same_document_duplicate_chunks": same_document_duplicate_pairs,
            "cross_document_duplicate_chunks": cross_document_duplicate_pairs,
            "unusually_high_chunk_count_threshold": unusual_threshold,
            "documents_with_unusually_high_chunk_counts": unusual_documents,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_max_tokens": self.embedding_max_tokens,
            "embedding_model": self.embedding_model,
        }

    def format_chunking_summary(self) -> str:
        """Format the actual chunking audit for command-line display."""
        summary = self.get_chunking_summary()
        labels = (
            ("Input documents", "input_documents"),
            ("Successfully processed documents", "successfully_processed_documents"),
            ("Failed documents", "failed_documents"),
            ("Total output chunks", "total_output_chunks"),
            ("Average chunks per document", "average_chunks_per_document"),
            ("Minimum chunks per document", "minimum_chunks_per_document"),
            ("Maximum chunks per document", "maximum_chunks_per_document"),
            ("Average body tokens", "average_body_tokens"),
            ("Median body tokens", "median_body_tokens"),
            ("P95 body tokens", "p95_body_tokens"),
            ("P99 body tokens", "p99_body_tokens"),
            ("Minimum body tokens", "minimum_body_tokens"),
            ("Maximum body tokens", "maximum_body_tokens"),
            ("Average embedding-input tokens", "average_embedding_input_tokens"),
            ("Minimum embedding-input tokens", "minimum_embedding_input_tokens"),
            ("Maximum embedding-input tokens", "maximum_embedding_input_tokens"),
            ("Chunks below 50 tokens", "chunks_below_50_tokens"),
            ("Chunks below 100 tokens", "chunks_below_100_tokens"),
            ("Chunks above body limit", "chunks_above_body_limit"),
            (
                "Chunks above embedding-input limit",
                "chunks_above_embedding_input_limit",
            ),
            ("Empty chunks", "empty_chunks"),
            ("Duplicate chunk IDs", "duplicate_chunk_ids"),
            ("Duplicate content fingerprints", "duplicate_content_fingerprints"),
            ("Same-document duplicate chunks", "same_document_duplicate_chunks"),
            ("Cross-document duplicate chunks", "cross_document_duplicate_chunks"),
            (
                "Unusually high chunk-count threshold",
                "unusually_high_chunk_count_threshold",
            ),
        )
        lines = ["DOCUMENT CHUNKING AUDIT", ""]
        lines.extend(f"- {label}: {summary[key]}" for label, key in labels)
        lines.append(
            "- Documents with unusually high chunk counts: "
            + json.dumps(
                summary["documents_with_unusually_high_chunk_counts"],
                ensure_ascii=False,
            )
        )
        if summary["failed_document_details"]:
            lines.append(
                "- Failed document details: "
                + json.dumps(summary["failed_document_details"], ensure_ascii=False)
            )
        return "\n".join(lines)

    def _split_text_safely(self, text: str) -> list[str]:
        effective_size = self.chunk_size
        while effective_size > 0:
            splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                self.tokenizer,
                chunk_size=effective_size,
                chunk_overlap=min(self.chunk_overlap, effective_size - 1),
                separators=list(SPLITTER_SEPARATORS),
                keep_separator="start",
                strip_whitespace=False,
            )
            pieces = self._merge_heading_only_chunks(splitter.split_text(text))
            if not pieces:
                raise ChunkingError("Splitter returned no chunks for non-empty content.")

            over_limit: list[tuple[int, int]] = []
            for piece in pieces:
                body_count = self._body_token_count(piece)
                embedding_count = self._embedding_input_token_count(piece)
                if (
                    body_count > self.chunk_size
                    or embedding_count > self.embedding_max_tokens
                ):
                    over_limit.append((body_count, embedding_count))
            if not over_limit:
                return pieces

            reductions = []
            for body_count, embedding_count in over_limit:
                reduction = max(
                    body_count - self.chunk_size,
                    embedding_count - self.embedding_max_tokens,
                    1,
                )
                reductions.append(body_count - reduction)
            next_size = max(1, min(effective_size - 1, min(reductions)))
            if next_size >= effective_size:
                next_size = effective_size - 1
            effective_size = next_size

        raise ChunkingError(
            "Unable to split content within the configured body and embedding limits."
        )

    def _merge_heading_only_chunks(self, pieces: Sequence[str]) -> list[str]:
        """Keep section labels attached to neighboring incident evidence."""
        merged: list[str] = []
        index = 0
        while index < len(pieces):
            piece = pieces[index]
            if not self._is_heading_only(piece):
                merged.append(piece)
                index += 1
                continue

            if index + 1 < len(pieces):
                merged.append(piece + pieces[index + 1])
                index += 2
                continue
            if merged:
                merged[-1] += piece
                index += 1
                continue
            raise ChunkingError(
                "A section heading has no neighboring content to attach to."
            )
        return merged

    @staticmethod
    def _is_heading_only(text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return bool(lines) and all(
            re.fullmatch(r"[\w /&()'-]+:", line, flags=re.UNICODE) is not None
            for line in lines
        )

    def _validate_source_coverage(self, source: str, pieces: Sequence[str]) -> None:
        if not pieces:
            raise ChunkingError("No chunks were generated for a non-empty document.")
        expected_content = source.strip()
        joined_chunks = "".join(pieces)
        cursor = 0
        for character in expected_content:
            position = joined_chunks.find(character, cursor)
            if position < 0:
                raise ChunkingError(
                    "Chunking changed, reordered, or omitted source content."
                )
            cursor = position + 1

    def _body_token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _embedding_input_token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def _get_document_identity(self, document: Document) -> str:
        for key in ("incident_id", "document_id"):
            value = document.metadata.get(key)
            if value is not None and str(value).strip():
                return self._safe_identity(value)
        if document.id and str(document.id).strip():
            return self._safe_identity(document.id)

        identity_data = {
            "page_content": document.page_content,
            "metadata": self._normalize_metadata(document.metadata),
        }
        fingerprint = hashlib.sha256(
            self._stable_json(identity_data).encode("utf-8")
        ).hexdigest()[:16]
        return f"doc_{fingerprint}"

    @staticmethod
    def _safe_identity(value: Any) -> str:
        identity = re.sub(r"[^\w.-]+", "_", str(value).strip(), flags=re.UNICODE)
        return identity.strip("_.") or "document"

    def _normalize_metadata(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, bool):
                normalized[str(key)] = value
            elif isinstance(value, int):
                normalized[str(key)] = value
            elif isinstance(value, float):
                normalized[str(key)] = value if math.isfinite(value) else str(value)
            elif isinstance(value, str):
                normalized[str(key)] = value
            else:
                normalized[str(key)] = self._stable_json(value)
        return normalized

    def _copy_metadata(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return json.loads(self._stable_json(metadata))
        except (TypeError, ValueError) as error:
            raise ChunkingError("Could not snapshot source Document metadata.") from error

    @staticmethod
    def _stable_json(value: Any) -> str:
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ChunkingError(
                "Metadata contains a value that cannot be serialized safely."
            ) from error

    @staticmethod
    def _content_fingerprint(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _percentile(values: Sequence[int], percentile: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return ordered[index]

    def _unusually_large_documents(
        self,
        counts: Sequence[int],
        chunks_by_parent: Mapping[str, Sequence[Document]],
    ) -> tuple[int, list[dict[str, Any]]]:
        if not counts:
            return 10, []
        lower_quartile = self._percentile(counts, 0.25)
        upper_quartile = self._percentile(counts, 0.75)
        threshold = max(
            10,
            math.ceil(upper_quartile + 1.5 * (upper_quartile - lower_quartile)),
        )
        unusual = [
            {
                "parent_document_id": parent_id,
                "chunk_count": len(parent_chunks),
            }
            for parent_id, parent_chunks in chunks_by_parent.items()
            if len(parent_chunks) > threshold
        ]
        return threshold, unusual

    def _record_failure(
        self,
        document_index: int,
        parent_id: str,
        reason: str,
    ) -> None:
        self._failed_documents.append(
            {
                "input_document_index": document_index,
                "parent_document_id": parent_id,
                "reason": reason,
            }
        )
        logger.warning(
            "Could not safely chunk input document %d (%s): %s",
            document_index,
            parent_id,
            reason,
        )

    def _reset_state(self) -> None:
        self._input_documents = 0
        self._chunks: list[Document] = []
        self._failed_documents: list[dict[str, Any]] = []
        self._processed_chunk_counts: dict[str, int] = {}
        self._source_documents: list[Document] = []
        self._source_content_snapshots: dict[int, str] = {}
        self._source_id_snapshots: dict[int, str | None] = {}
        self._source_metadata_snapshots: dict[int, dict[str, Any]] = {}
        self._source_metadata_by_parent: dict[str, dict[str, Any]] = {}


def main() -> int:
    """Run the default loader-to-chunker pipeline and persist validated JSONL."""
    from app.rag.document_loader import DocumentLoadError, HistoricalIncidentLoader

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        loader = HistoricalIncidentLoader()
        documents = loader.convert_to_documents(loader.load())
        chunker = HistoricalIncidentChunker()
        chunks = chunker.chunk_documents(documents)
        chunker.validate_chunks(chunks)
    except (ChunkingError, DocumentLoadError, FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1

    configured_output = os.getenv("RAG_CHUNKS_PATH")
    output_path = (
        Path(configured_output).expanduser()
        if configured_output
        else DEFAULT_OUTPUT_PATH
    )
    if not output_path.is_absolute():
        output_path = BACKEND_ROOT / output_path

    try:
        write_chunks_jsonl(chunks, output_path)
    except (OSError, TypeError, ValueError) as error:
        logger.error("Could not write chunk JSONL output to %s: %s", output_path, error)
        return 1

    print(chunker.format_chunking_summary())
    print(f"\nChunk JSONL: {output_path}")
    for index, chunk in enumerate(chunks[:2], start=1):
        print(f"\n--- Sample chunk {index} ---")
        print(chunk.page_content)
        print(json.dumps(chunk.metadata, ensure_ascii=False, indent=2))

    return 1 if chunker.get_chunking_summary()["failed_documents"] else 0


def write_chunks_jsonl(chunks: Sequence[Document], output_path: Path) -> None:
    """Write chunk text, IDs, and metadata as one JSON record per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for chunk in chunks:
            output_file.write(
                json.dumps(
                    {
                        "chunk_id": chunk.id,
                        "text": chunk.page_content,
                        "metadata": chunk.metadata,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    sys.exit(main())