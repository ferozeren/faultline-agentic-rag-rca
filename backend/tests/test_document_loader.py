"""Tests for historical incident dataset loading and document conversion."""

import json
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.documents import Document as LangChainDocument

from app.rag.document_loader import (
    DatasetReadError,
    EmptyDatasetError,
    HistoricalIncidentLoader,
    MissingRequiredColumnsError,
    UnsupportedFormatError,
)
from app.rag.document_schema import Document


def incident(**overrides):
    record = {
        "incident_id": "INC-001",
        "company": "Example",
        "date": "2026-01-02",
        "title": "Elevated latency",
        "severity": "MEDIUM",
        "service_stack": ["API", "Workers"],
        "categories": ["LATENCY"],
        "symptom_logs": "Requests timed out.",
        "root_cause_analysis": "An upstream dependency was throttling requests.",
        "breaking_config_code": None,
        "remediation_patch": "Scaled healthy workers.",
        "prevention_checklist": ["Add capacity alerts."],
        "source_url": "https://example.test/incident/1",
        "metadata": {"status": "resolved"},
    }
    record.update(overrides)
    return record


def test_loads_csv_and_converts_to_document(tmp_path: Path):
    source = tmp_path / "incidents.csv"
    pd.DataFrame([incident()]).to_csv(source, index=False)
    loader = HistoricalIncidentLoader(source)

    records = loader.load()
    documents = loader.convert_to_documents(records)

    assert loader.get_loading_summary()["loading_status"] == "SUCCESS"
    assert len(documents) == 1
    assert isinstance(documents[0], LangChainDocument)
    assert Document is LangChainDocument
    assert "Elevated latency" in documents[0].page_content
    assert "Incident Title:\nElevated latency" in documents[0].page_content
    assert "API\nWorkers" in documents[0].page_content
    assert documents[0].metadata["incident_id"] == "INC-001"
    assert documents[0].id == "INC-001"
    assert documents[0].metadata["document_id"] == "INC-001"


def test_document_id_is_stable_across_conversions(tmp_path: Path):
    source = tmp_path / "incidents.jsonl"
    source.write_text(json.dumps(incident()) + "\n", encoding="utf-8")
    loader = HistoricalIncidentLoader(source)
    records = loader.load()

    first = loader.convert_to_documents(records)[0]
    second = loader.convert_to_documents(records)[0]

    assert first.id == second.id == "INC-001"


def test_loads_jsonl(tmp_path: Path):
    source = tmp_path / "incidents.jsonl"
    source.write_text(json.dumps(incident()) + "\n", encoding="utf-8")

    loader = HistoricalIncidentLoader(source)

    assert loader.load()[0]["incident_id"] == "INC-001"


def test_loads_parquet(tmp_path: Path):
    source = tmp_path / "incidents.parquet"
    pd.DataFrame([incident()]).to_parquet(source, index=False)

    loader = HistoricalIncidentLoader(source)

    assert loader.load()[0]["incident_id"] == "INC-001"


def test_preserves_metadata_with_chroma_compatible_values(tmp_path: Path):
    source = tmp_path / "incidents.jsonl"
    source.write_text(json.dumps(incident()) + "\n", encoding="utf-8")
    loader = HistoricalIncidentLoader(source)

    document = loader.convert_to_documents(loader.load())[0]

    assert document.metadata["company"] == "Example"
    assert document.metadata["service_stack"] == '["API", "Workers"]'
    assert document.metadata["metadata"] == '{"status": "resolved"}'
    assert document.metadata["source_file"] == "incidents.jsonl"
    assert "symptom_logs" not in document.metadata
    assert all(
        isinstance(value, (str, int, float, bool))
        for value in document.metadata.values()
    )


def test_normalizes_serialized_lists_and_dictionaries(tmp_path: Path):
    record = incident(
        service_stack='["API", "Workers"]',
        categories="['LATENCY', 'TIMEOUT']",
        metadata='{"status": "resolved", "details": {"region": "west"}}',
    )
    source = tmp_path / "incidents.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    loader = HistoricalIncidentLoader(source)

    document = loader.convert_to_documents(loader.load())[0]

    assert "Service Stack:\nAPI\nWorkers" in document.page_content
    assert "Categories:\nLATENCY\nTIMEOUT" in document.page_content
    assert document.metadata["service_stack"] == '["API", "Workers"]'
    assert document.metadata["categories"] == '["LATENCY", "TIMEOUT"]'
    assert document.metadata["metadata"] == (
        '{"status": "resolved", "details": {"region": "west"}}'
    )


def test_missing_root_cause_is_explicit_and_not_counted(tmp_path: Path):
    source = tmp_path / "incidents.jsonl"
    source.write_text(
        json.dumps(incident(root_cause_analysis=None)) + "\n", encoding="utf-8"
    )
    loader = HistoricalIncidentLoader(source)

    documents = loader.convert_to_documents(loader.load())

    assert "A confirmed root cause was not provided." in documents[0].page_content
    assert documents[0].metadata["has_root_cause"] is False
    assert loader.get_loading_summary()["records_without_root_cause"] == 1


def test_unconfirmed_timeline_is_not_labeled_as_root_cause(tmp_path: Path):
    source = tmp_path / "incidents.jsonl"
    source.write_text(
        json.dumps(
            incident(
                root_cause_analysis=(
                    "A detailed root cause analysis will be shared when available."
                )
            )
        )
        + "\n",
        encoding="utf-8",
    )
    loader = HistoricalIncidentLoader(source)

    document = loader.convert_to_documents(loader.load())[0]

    assert document.metadata["has_root_cause"] is False
    assert "A confirmed root cause was not provided." in document.page_content
    assert "Source notes:\nA detailed root cause analysis" in document.page_content


def test_unconfirmed_root_cause_note_overrides_unrelated_cause_text(
    tmp_path: Path,
):
    source = tmp_path / "incidents.jsonl"
    source.write_text(
        json.dumps(
            incident(
                root_cause_analysis=(
                    "A detailed root cause analysis will be shared when available. "
                    "The service degraded due to upstream throttling."
                )
            )
        )
        + "\n",
        encoding="utf-8",
    )
    loader = HistoricalIncidentLoader(source)

    document = loader.convert_to_documents(loader.load())[0]

    assert document.metadata["has_root_cause"] is False
    assert "A confirmed root cause was not provided." in document.page_content


def test_missing_remediation_is_explicit_and_not_counted(tmp_path: Path):
    source = tmp_path / "incidents.jsonl"
    source.write_text(
        json.dumps(incident(remediation_patch=None)) + "\n", encoding="utf-8"
    )
    loader = HistoricalIncidentLoader(source)

    document = loader.convert_to_documents(loader.load())[0]

    assert "Remediation:\nNot provided." in document.page_content
    assert document.metadata["has_remediation"] is False
    assert loader.get_loading_summary()["records_without_remediation"] == 1


def test_skips_invalid_records_and_counts_invalid_jsonl(tmp_path: Path):
    source = tmp_path / "incidents.jsonl"
    source.write_text(
        "\n".join(
            (
                json.dumps(incident()),
                json.dumps(incident(incident_id="")),
                "{invalid json",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    loader = HistoricalIncidentLoader(source)

    assert len(loader.load()) == 1
    summary = loader.get_loading_summary()
    assert summary["total_records_detected"] == 3
    assert summary["invalid_records"] == 2
    assert summary["loading_status"] == "PARTIAL"


def test_deduplicates_exact_incident_ids(tmp_path: Path):
    source = tmp_path / "incidents.jsonl"
    source.write_text(
        "\n".join(json.dumps(incident(title=title)) for title in ("First", "Second"))
        + "\n",
        encoding="utf-8",
    )
    loader = HistoricalIncidentLoader(source)

    records = loader.load()

    assert len(records) == 1
    assert records[0]["title"] == "First"
    assert loader.get_loading_summary()["duplicate_records"] == 1


def test_empty_dataset_raises_and_reports_failure(tmp_path: Path):
    source = tmp_path / "empty.csv"
    pd.DataFrame(
        columns=["incident_id", "title", "symptom_logs", "root_cause_analysis"]
    ).to_csv(source, index=False)
    loader = HistoricalIncidentLoader(source)

    with pytest.raises(EmptyDatasetError):
        loader.load()

    assert loader.get_loading_summary()["loading_status"] == "FAILED"


def test_unsupported_extension_raises(tmp_path: Path):
    source = tmp_path / "incidents.txt"
    source.write_text("not a supported dataset", encoding="utf-8")

    with pytest.raises(UnsupportedFormatError):
        HistoricalIncidentLoader(source).load()


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        HistoricalIncidentLoader(tmp_path / "missing.csv").load()


def test_missing_required_columns_raises(tmp_path: Path):
    source = tmp_path / "incidents.csv"
    pd.DataFrame([{"title": "Missing incident ID"}]).to_csv(source, index=False)

    with pytest.raises(MissingRequiredColumnsError):
        HistoricalIncidentLoader(source).load()


def test_unreadable_parquet_raises_a_dataset_error(tmp_path: Path):
    source = tmp_path / "invalid.parquet"
    source.write_text("not a parquet file", encoding="utf-8")

    with pytest.raises(DatasetReadError, match="Could not read PARQUET dataset"):
        HistoricalIncidentLoader(source).load()
