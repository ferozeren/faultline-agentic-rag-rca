"""Tests for deterministic token-aware historical incident chunking."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest
from langchain_core.documents import Document
from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from app.rag.chunking import (
    HistoricalIncidentChunker,
    TokenizerInitializationError,
    _load_tokenizer,
    write_chunks_jsonl,
)

TEST_MODEL = "test/character-tokenizer"


@pytest.fixture
def character_tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2}
    backend_tokenizer = Tokenizer(
        models.WordLevel(vocabulary, unk_token="[UNK]")
    )
    backend_tokenizer.pre_tokenizer = pre_tokenizers.Split(
        Regex(""),
        behavior="isolated",
    )
    backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


@pytest.fixture
def make_chunker(
    monkeypatch: pytest.MonkeyPatch,
    character_tokenizer: PreTrainedTokenizerFast,
) -> Callable[..., HistoricalIncidentChunker]:
    _load_tokenizer.cache_clear()
    monkeypatch.setattr(
        "app.rag.chunking.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: character_tokenizer,
    )

    def factory(**overrides: object) -> HistoricalIncidentChunker:
        settings: dict[str, object] = {
            "chunk_size": 64,
            "chunk_overlap": 12,
            "embedding_max_tokens": 72,
            "embedding_model": TEST_MODEL,
        }
        settings.update(overrides)
        return HistoricalIncidentChunker(**settings)

    yield factory
    _load_tokenizer.cache_clear()


def test_short_document_produces_one_token_measured_chunk(make_chunker):
    source = Document(
        page_content="Incident Title:\nShort outage.",
        metadata={"incident_id": "INC-001"},
    )
    chunker = make_chunker()

    chunks = chunker.chunk_documents([source])

    assert len(chunks) == 1
    assert chunks[0].page_content == source.page_content
    assert chunks[0].metadata["body_token_count"] == len(
        chunker.tokenizer.encode(source.page_content, add_special_tokens=False)
    )


def test_long_document_splits_with_token_overlap(make_chunker):
    source_text = " ".join(f"evidence-{index}" for index in range(90))
    chunker = make_chunker()

    chunks = chunker.chunk_documents(
        [Document(page_content=source_text, metadata={"incident_id": "INC-002"})]
    )

    assert len(chunks) > 1
    assert all(
        chunk.metadata["body_token_count"] <= chunker.chunk_size for chunk in chunks
    )
    assert any(
        left.page_content[-10:] in right.page_content
        for left, right in zip(chunks, chunks[1:])
    )


def test_complete_embedding_input_respects_special_token_limit(make_chunker):
    chunker = make_chunker(chunk_size=200, chunk_overlap=20, embedding_max_tokens=50)
    text = "incident evidence " * 20

    chunks = chunker.chunk_documents([Document(page_content=text)])

    assert len(chunks) > 1
    assert max(chunk.metadata["body_token_count"] for chunk in chunks) <= 200
    assert max(
        chunk.metadata["embedding_input_token_count"] for chunk in chunks
    ) <= 50
    assert all(
        len(chunker.tokenizer.encode(chunk.page_content, add_special_tokens=True))
        <= 50
        for chunk in chunks
    )


def test_section_headings_and_metadata_are_preserved(make_chunker):
    source = Document(
        page_content=(
            "Incident ID:\nINC-003\n\nIncident Title:\nPayment API failure\n\n"
            "Symptoms and Logs:\nTimeouts affected checkout.\n\n"
            "Remediation:\nRestarted the failed workers."
        ),
        metadata={
            "incident_id": "INC-003",
            "document_id": "INC-003",
            "company": "Example",
            "date": "2026-01-02",
            "severity": "HIGH",
            "source_file": "incidents.csv",
            "source_url": "https://example.test/incident/3",
            "service_stack": '["API"]',
            "categories": '["TIMEOUT"]',
        },
    )
    chunker = make_chunker(chunk_size=45, chunk_overlap=8)

    chunks = chunker.chunk_documents([source])

    joined = "\n".join(chunk.page_content for chunk in chunks)
    assert "Incident Title:" in joined
    assert "Symptoms and Logs:" in joined
    assert "Remediation:" in joined
    for chunk in chunks:
        for key, value in source.metadata.items():
            assert chunk.metadata[key] == value
        assert chunk.metadata["chunking_strategy"] == (
            "recursive_character_token_aware"
        )
        assert chunk.metadata["embedding_model"] == TEST_MODEL


def test_heading_only_chunks_are_attached_to_evidence(make_chunker):
    source = Document(
        page_content="Incident Title:\nPayment service outage details.",
        metadata={"incident_id": "INC-HEADING"},
    )
    chunks = make_chunker(
        chunk_size=20,
        chunk_overlap=4,
        embedding_max_tokens=24,
    ).chunk_documents([source])

    assert chunks
    assert all(
        not HistoricalIncidentChunker._is_heading_only(chunk.page_content)
        for chunk in chunks
    )
    assert "".join(chunk.page_content for chunk in chunks).find(
        source.page_content.strip()
    ) >= 0


def test_chunk_ids_are_unique_stable_and_sequential(make_chunker):
    source = Document(
        page_content="error details " * 20,
        metadata={"incident_id": "INC-004"},
    )
    chunker = make_chunker()

    first_run = chunker.chunk_documents([source])
    first_ids = [chunk.id for chunk in first_run]
    second_run = chunker.chunk_documents([source])

    assert len(first_ids) == len(set(first_ids))
    assert [chunk.id for chunk in second_run] == first_ids
    assert [chunk.metadata["chunk_index"] for chunk in first_run] == list(
        range(len(first_run))
    )
    assert all(
        chunk.metadata["total_chunks"] == len(first_run) for chunk in first_run
    )
    assert first_ids[0] == "INC-004_chunk_000"


def test_empty_and_whitespace_documents_are_reported(make_chunker):
    chunker = make_chunker()
    chunks = chunker.chunk_documents(
        [
            Document(page_content="", metadata={"incident_id": "INC-EMPTY"}),
            Document(page_content=" \n\t ", metadata={"incident_id": "INC-SPACE"}),
            Document(page_content="valid", metadata={"incident_id": "INC-OK"}),
        ]
    )

    summary = chunker.get_chunking_summary()
    assert len(chunks) == 1
    assert summary["input_documents"] == 3
    assert summary["successfully_processed_documents"] == 1
    assert summary["failed_documents"] == 2
    assert len(summary["failed_document_details"]) == 2
    assert chunker.validate_chunks(chunks)


def test_unicode_and_very_long_unbroken_text_are_split_safely(make_chunker):
    text = "服务异常🚨é " + ("x" * 250)
    chunker = make_chunker(chunk_size=32, chunk_overlap=5, embedding_max_tokens=36)
    source = Document(page_content=text, metadata={"incident_id": "INC-UNICODE"})

    chunks = chunker.chunk_documents([source])

    assert len(chunks) > 1
    assert "".join(chunk.page_content for chunk in chunks).find("服务异常🚨é") >= 0
    assert max(chunk.metadata["body_token_count"] for chunk in chunks) <= 32
    assert max(
        chunk.metadata["embedding_input_token_count"] for chunk in chunks
    ) <= 36


def test_missing_optional_metadata_uses_deterministic_parent_fallback(make_chunker):
    source = Document(page_content="Evidence with no incident metadata.")
    chunker = make_chunker()

    chunks = chunker.chunk_documents([source])

    assert chunks[0].metadata["parent_document_id"].startswith("doc_")
    assert "incident_id" not in chunks[0].metadata
    assert chunks[0].id.startswith("doc_")


def test_multiple_incidents_duplicate_fingerprints_are_audited(make_chunker):
    text = "same short evidence"
    chunker = make_chunker()
    chunks = chunker.chunk_documents(
        [
            Document(page_content=text, metadata={"incident_id": "INC-A"}),
            Document(page_content=text, metadata={"incident_id": "INC-B"}),
        ]
    )

    summary = chunker.get_chunking_summary()
    assert len(chunks) == 2
    assert summary["duplicate_content_fingerprints"] == 1
    assert summary["cross_document_duplicate_chunks"] == 1
    assert summary["same_document_duplicate_chunks"] == 0
    assert summary["duplicate_chunk_ids"] == 0


def test_same_document_duplicate_chunks_are_reported_without_removal(make_chunker):
    source = Document(
        page_content="x" * 50,
        metadata={"incident_id": "INC-REPEATED"},
    )
    chunker = make_chunker(chunk_size=12, chunk_overlap=4, embedding_max_tokens=16)

    chunks = chunker.chunk_documents([source])

    summary = chunker.get_chunking_summary()
    assert len(chunks) > 1
    assert summary["duplicate_content_fingerprints"] > 0
    assert summary["same_document_duplicate_chunks"] > 0


def test_oversized_chunks_are_resplit_not_truncated(make_chunker):
    source_text = "critical evidence " * 100
    source = Document(page_content=source_text, metadata={"incident_id": "INC-LONG"})
    chunker = make_chunker(
        chunk_size=224,
        chunk_overlap=40,
        embedding_max_tokens=32,
    )

    chunks = chunker.chunk_documents([source])

    assert len(chunks) > 1
    assert all(chunk.metadata["body_token_count"] <= 224 for chunk in chunks)
    assert all(
        chunk.metadata["embedding_input_token_count"] <= 32 for chunk in chunks
    )
    chunker._validate_source_coverage(
        source_text,
        [chunk.page_content for chunk in chunks],
    )


def test_original_documents_are_not_modified(make_chunker):
    source = Document(
        page_content="original evidence " * 8,
        metadata={"incident_id": "INC-ORIGINAL", "categories": ["API"]},
    )
    original_content = source.page_content
    original_metadata = dict(source.metadata)

    make_chunker().chunk_documents([source])

    assert source.page_content == original_content
    assert source.metadata == original_metadata
    assert source.id is None


def test_write_chunks_jsonl_preserves_chunk_records(tmp_path, make_chunker):
    source = Document(
        page_content="incident text",
        metadata={"incident_id": "INC-JSONL"},
    )
    chunks = make_chunker().chunk_documents([source])
    output_path = tmp_path / "processed" / "chunks.jsonl"

    write_chunks_jsonl(chunks, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(chunks)
    record = json.loads(lines[0])
    assert record["chunk_id"] == chunks[0].id
    assert record["text"] == chunks[0].page_content
    assert record["metadata"]["incident_id"] == "INC-JSONL"
    assert record["metadata"]["chunk_index"] == 0


def test_duplicate_incident_ids_still_generate_unique_chunk_ids(make_chunker):
    chunker = make_chunker()
    chunks = chunker.chunk_documents(
        [
            Document(page_content="first evidence", metadata={"incident_id": "INC-X"}),
            Document(page_content="second evidence", metadata={"incident_id": "INC-X"}),
        ]
    )

    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert {chunk.metadata["parent_document_id"] for chunk in chunks} == {
        "INC-X",
        "INC-X_source_0001",
    }


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"chunk_size": 0}, "chunk_size"),
        ({"chunk_size": -1}, "chunk_size"),
        ({"chunk_size": 10, "chunk_overlap": -1}, "chunk_overlap"),
        ({"chunk_size": 10, "chunk_overlap": 10}, "chunk_overlap"),
        ({"embedding_max_tokens": 0}, "embedding_max_tokens"),
    ],
)
def test_rejects_invalid_chunk_configuration(make_chunker, settings, message):
    with pytest.raises(ValueError, match=message):
        make_chunker(**settings)


def test_tokenizer_initialization_error_is_explicit(monkeypatch):
    _load_tokenizer.cache_clear()

    def fail_loading(*args, **kwargs):
        raise OSError("model cache unavailable")

    monkeypatch.setattr("app.rag.chunking.AutoTokenizer.from_pretrained", fail_loading)
    with pytest.raises(TokenizerInitializationError, match="Unable to load tokenizer"):
        HistoricalIncidentChunker(
            embedding_model="test/unavailable-tokenizer",
            chunk_size=64,
            chunk_overlap=8,
            embedding_max_tokens=72,
        )
    _load_tokenizer.cache_clear()


def test_very_small_embedding_limit_rejects_special_token_overhead(make_chunker):
    with pytest.raises(ValueError, match="special-token overhead"):
        make_chunker(embedding_max_tokens=1)
