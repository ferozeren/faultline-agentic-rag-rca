"""Load historical incident records and convert them into typed documents."""

from __future__ import annotations

import ast
import json
import logging
import os
import sys
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
DEFAULT_SOURCE_PATH = (
    BACKEND_ROOT / "data" / "historical_incidents" / "raw" / "incidentdb.csv"
)

DESCRIPTION_FIELDS = ("title", "symptom_logs", "root_cause_analysis")
METADATA_FIELDS = (
    "company",
    "date",
    "title",
    "severity",
    "service_stack",
    "categories",
    "source_url",
    "metadata",
)
LIST_FIELDS = {"service_stack", "categories", "prevention_checklist"}
MISSING_VALUES = {"", "nan", "nat", "none", "null", "n/a", "na"}
NOT_CONFIRMED_PHRASES = (
    "no confirmed root cause",
    "root cause is unknown",
    "root cause has not been identified",
    "root cause has not yet been identified",
    "root cause has not been determined",
    "root cause is still under investigation",
    "root cause remains under investigation",
    "root cause analysis will be shared",
    "detailed root cause analysis will be shared",
)


class DocumentLoadError(Exception):
    """Base exception for historical incident loading failures."""


class UnsupportedFormatError(DocumentLoadError):
    """Raised when the source file extension is not supported."""


class MissingRequiredColumnsError(DocumentLoadError):
    """Raised when a tabular source lacks required incident fields."""


class EmptyDatasetError(DocumentLoadError):
    """Raised when a valid source contains no incident records."""


class DatasetReadError(DocumentLoadError):
    """Raised when a supported dataset cannot be read."""


class HistoricalIncidentLoader:
    """Load one selected historical incident export without combining sources."""

    def __init__(self, source_path: str | Path | None = None) -> None:
        configured_path = source_path or os.getenv("HISTORICAL_INCIDENTS_PATH")
        if configured_path is None:
            self.source_path = DEFAULT_SOURCE_PATH
        else:
            path = Path(configured_path).expanduser()
            self.source_path = path if path.is_absolute() else PROJECT_ROOT / path

        self.source_path = self.source_path.resolve()
        self.source_format = self.source_path.suffix.lower().lstrip(".")
        self.total_records_detected = 0
        self.successfully_loaded = 0
        self.invalid_records = 0
        self.duplicate_records = 0
        self.empty_documents = 0
        self.records_with_root_cause = 0
        self.records_with_remediation = 0
        self._records: list[dict[str, Any]] = []

    def load(self) -> list[dict[str, Any]]:
        """Read, validate, and deduplicate incident records from the selected file."""
        self._reset_summary()
        if not self.source_path.is_file():
            raise FileNotFoundError(
                f"Historical incident source not found: {self.source_path}"
            )

        readers = {
            "csv": self._read_tabular,
            "parquet": self._read_tabular,
            "jsonl": self._read_jsonl,
        }
        reader = readers.get(self.source_format)
        if reader is None:
            raise UnsupportedFormatError(
                f"Unsupported historical incident format '.{self.source_format}'. "
                "Supported formats are .csv, .jsonl, and .parquet."
            )

        raw_records = reader()
        if not raw_records:
            raise EmptyDatasetError(f"No records found in {self.source_path.name}.")

        seen_ids: set[str] = set()
        for record_number, record in enumerate(raw_records, start=1):
            if not self.validate_record(record):
                self.invalid_records += 1
                logger.warning(
                    "Skipping invalid incident record %d from %s",
                    record_number,
                    self.source_path.name,
                )
                continue

            incident_id = str(record["incident_id"]).strip()
            record["incident_id"] = incident_id
            if incident_id in seen_ids:
                self.duplicate_records += 1
                logger.warning(
                    "Skipping duplicate incident record %d from %s",
                    record_number,
                    self.source_path.name,
                )
                continue

            seen_ids.add(incident_id)
            self._records.append(record)

        self.successfully_loaded = len(self._records)
        self.records_with_root_cause = sum(
            self._has_confirmed_root_cause(record.get("root_cause_analysis"))
            for record in self._records
        )
        self.records_with_remediation = sum(
            self._has_content(record.get("remediation_patch"))
            for record in self._records
        )

        if not self._records:
            raise EmptyDatasetError(
                f"No valid incident records found in {self.source_path.name}."
            )

        logger.info(
            "Loaded %d of %d incident records from %s",
            self.successfully_loaded,
            self.total_records_detected,
            self.source_path.name,
        )
        return list(self._records)

    def validate_record(self, record: Mapping[str, Any]) -> bool:
        """Return whether a record has an ID and at least one description field."""
        if not isinstance(record, Mapping) or not self._has_content(
            record.get("incident_id")
        ):
            return False
        return any(self._has_content(record.get(field)) for field in DESCRIPTION_FIELDS)

    def convert_to_documents(
        self, records: Iterable[Mapping[str, Any]] | None = None
    ) -> list[Document]:
        """Convert loaded records into standardized text documents."""
        source_records = self._records if records is None else records
        documents = []
        for record in source_records:
            if not self.validate_record(record):
                raise ValueError(
                    "Cannot convert an invalid incident record; expected an incident "
                    "ID and a title, symptom log, or root cause analysis."
                )
            document = self._to_document(record)
            if not document.page_content.strip():
                self.empty_documents += 1
                logger.error(
                    "Generated an empty document from %s",
                    self.source_path.name,
                )
                raise ValueError("Incident conversion produced empty page_content.")
            documents.append(document)
        return documents

    def get_loading_summary(self) -> dict[str, int | str]:
        """Return calculated counts and a success, partial, or failed status."""
        if self.successfully_loaded == 0:
            status = "FAILED"
        elif self.invalid_records or self.duplicate_records:
            status = "PARTIAL"
        else:
            status = "SUCCESS"

        return {
            "source_filename": self.source_path.name,
            "source_format": self.source_format.upper(),
            "total_records_detected": self.total_records_detected,
            "successfully_loaded": self.successfully_loaded,
            "invalid_records": self.invalid_records,
            "duplicate_records": self.duplicate_records,
            "empty_documents": self.empty_documents,
            "records_with_root_cause": self.records_with_root_cause,
            "records_without_root_cause": (
                self.successfully_loaded - self.records_with_root_cause
            ),
            "records_with_remediation": self.records_with_remediation,
            "records_without_remediation": (
                self.successfully_loaded - self.records_with_remediation
            ),
            "loading_status": status,
        }

    def format_loading_summary(self) -> str:
        """Format loading statistics for the command-line entry point."""
        summary = self.get_loading_summary()
        return "\n".join(
            (
                "HISTORICAL INCIDENT DOCUMENT LOADING",
                "",
                f"Source: {summary['source_filename']}",
                f"Format: {summary['source_format']}",
                f"Total records: {summary['total_records_detected']}",
                f"Successfully loaded: {summary['successfully_loaded']}",
                f"Invalid records: {summary['invalid_records']}",
                f"Duplicate records: {summary['duplicate_records']}",
                f"Empty documents: {summary['empty_documents']}",
                f"Records with root cause: {summary['records_with_root_cause']}",
                f"Records without root cause: {summary['records_without_root_cause']}",
                f"Records with remediation: {summary['records_with_remediation']}",
                "Records without remediation: "
                f"{summary['records_without_remediation']}",
                f"Loading status: {summary['loading_status']}",
            )
        )

    def _read_tabular(self) -> list[dict[str, Any]]:
        try:
            if self.source_format == "csv":
                dataframe = pd.read_csv(self.source_path, dtype=object)
            else:
                dataframe = pd.read_parquet(self.source_path)
        except pd.errors.EmptyDataError as error:
            raise EmptyDatasetError(
                f"No records found in {self.source_path.name}."
            ) from error
        except (
            ImportError,
            ValueError,
            OSError,
            UnicodeDecodeError,
            pd.errors.ParserError,
        ) as error:
            raise DatasetReadError(
                f"Could not read {self.source_format.upper()} dataset "
                f"{self.source_path}: {error}"
            ) from error

        columns = set(dataframe.columns)
        missing_columns = []
        if "incident_id" not in columns:
            missing_columns.append("incident_id")
        if not columns.intersection(DESCRIPTION_FIELDS):
            missing_columns.append("one of " + ", ".join(DESCRIPTION_FIELDS))
        if missing_columns:
            raise MissingRequiredColumnsError(
                f"{self.source_path.name} is missing required columns: "
                f"{'; '.join(missing_columns)}"
            )

        self.total_records_detected = len(dataframe.index)
        return [
            {
                str(key): self._normalize_record_value(str(key), value)
                for key, value in row.items()
            }
            for row in dataframe.to_dict(orient="records")
        ]

    def _read_jsonl(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        try:
            with self.source_path.open(encoding="utf-8") as source_file:
                for line_number, line in enumerate(source_file, start=1):
                    if not line.strip():
                        continue
                    self.total_records_detected += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        self.invalid_records += 1
                        logger.warning(
                            "Malformed JSONL record at %s:%d",
                            self.source_path.name,
                            line_number,
                        )
                        continue
                    if not isinstance(record, dict):
                        self.invalid_records += 1
                        logger.warning(
                            "JSONL record at %s:%d is not an object",
                            self.source_path.name,
                            line_number,
                        )
                        continue
                    records.append(
                        {
                            str(key): self._normalize_record_value(str(key), value)
                            for key, value in record.items()
                        }
                    )
        except (OSError, UnicodeDecodeError) as error:
            raise DatasetReadError(
                f"Could not read JSONL dataset {self.source_path}: {error}"
            ) from error

        if records:
            columns = {key for record in records for key in record}
            missing_columns = []
            if "incident_id" not in columns:
                missing_columns.append("incident_id")
            if not columns.intersection(DESCRIPTION_FIELDS):
                missing_columns.append("one of " + ", ".join(DESCRIPTION_FIELDS))
            if missing_columns:
                raise MissingRequiredColumnsError(
                    f"{self.source_path.name} is missing required columns: "
                    f"{'; '.join(missing_columns)}"
                )

        if not records and self.total_records_detected == 0:
            raise EmptyDatasetError(f"No records found in {self.source_path.name}.")
        return records

    def _to_document(self, record: Mapping[str, Any]) -> Document:
        incident_id = self._render(record.get("incident_id"))
        root_cause = record.get("root_cause_analysis")
        remediation = record.get("remediation_patch")
        rendered_root_cause = self._render(root_cause)
        if not self._has_confirmed_root_cause(root_cause):
            root_cause_content = "A confirmed root cause was not provided."
            if rendered_root_cause:
                root_cause_content += f"\n\nSource notes:\n{rendered_root_cause}"
        else:
            root_cause_content = rendered_root_cause
        fields = (
            ("Incident ID", incident_id),
            ("Company", self._display_or_unavailable(record.get("company"))),
            ("Date", self._format_date(record.get("date"))),
            ("Incident Title", self._display_or_unavailable(record.get("title"))),
            ("Severity", self._display_or_unavailable(record.get("severity"))),
            (
                "Service Stack",
                self._display_or_unavailable(record.get("service_stack")),
            ),
            ("Categories", self._display_or_unavailable(record.get("categories"))),
            (
                "Symptoms and Logs",
                self._display_or_unavailable(record.get("symptom_logs")),
            ),
            ("Root Cause Analysis", root_cause_content),
            (
                "Breaking Configuration",
                self._display_or_unavailable(record.get("breaking_config_code")),
            ),
            (
                "Remediation",
                self._display_or_unavailable(remediation),
            ),
            (
                "Prevention Checklist",
                self._display_or_unavailable(record.get("prevention_checklist")),
            ),
            ("Source", self._display_or_unavailable(record.get("source_url"))),
        )

        metadata: dict[str, str | int | float | bool] = {
            "incident_id": incident_id,
            "document_id": incident_id,
            "source_file": self.source_path.name,
            "has_root_cause": self._has_confirmed_root_cause(root_cause),
            "has_remediation": self._has_content(remediation),
        }
        for field in METADATA_FIELDS:
            value = record.get(field)
            if not self._has_content(value):
                continue
            if field == "date":
                metadata[field] = self._format_date(value)
            else:
                metadata[field] = self._metadata_value(value)

        content = "\n\n".join(f"{label}:\n{value}" for label, value in fields)
        return Document(id=incident_id, page_content=content, metadata=metadata)

    def _reset_summary(self) -> None:
        self.total_records_detected = 0
        self.successfully_loaded = 0
        self.invalid_records = 0
        self.duplicate_records = 0
        self.empty_documents = 0
        self.records_with_root_cause = 0
        self.records_with_remediation = 0
        self._records = []

    @classmethod
    def _has_content(cls, value: Any) -> bool:
        value = cls._normalize_value(value)
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().casefold() not in MISSING_VALUES
        if isinstance(value, (list, tuple, set, dict)):
            return bool(value) and any(cls._has_content(item) for item in value)
        return True

    @classmethod
    def _has_confirmed_root_cause(cls, value: Any) -> bool:
        if not cls._has_content(value):
            return False
        normalized = cls._render(value).casefold()
        return not any(phrase in normalized for phrase in NOT_CONFIRMED_PHRASES)

    @classmethod
    def _normalize_record_value(cls, field: str, value: Any) -> Any:
        value = cls._normalize_value(value)
        if not isinstance(value, str):
            return value

        expected_type = (
            list if field in LIST_FIELDS else dict if field == "metadata" else None
        )
        if expected_type is not None and value.lstrip().startswith(("[", "{")):
            try:
                parsed_value = json.loads(value)
            except json.JSONDecodeError:
                try:
                    parsed_value = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    return value
            return parsed_value if isinstance(parsed_value, expected_type) else value

        return value

    @classmethod
    def _normalize_value(cls, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "item") and callable(value.item):
            try:
                value = value.item()
            except (ValueError, TypeError):
                pass
        if not isinstance(value, (list, tuple, set, dict)):
            try:
                if pd.isna(value):
                    return None
            except (TypeError, ValueError):
                pass
        if isinstance(value, str) and value.strip().casefold() in MISSING_VALUES:
            return None
        return value

    @classmethod
    def _render(cls, value: Any) -> str:
        value = cls._normalize_value(value)
        if value is None:
            return ""
        if isinstance(value, Mapping):
            return json.dumps(value, ensure_ascii=False, default=str, indent=2)
        if isinstance(value, (list, tuple, set)):
            return "\n".join(
                cls._render(item) for item in value if cls._has_content(item)
            )
        return str(value).strip()

    @classmethod
    def _display_or_unavailable(cls, value: Any) -> str:
        rendered = cls._render(value)
        return rendered if rendered else "Not provided."

    @classmethod
    def _metadata_value(cls, value: Any) -> str | int | float | bool:
        value = cls._normalize_value(value)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, (Mapping, list, tuple, set)):
            return json.dumps(value, ensure_ascii=False, default=str)
        return cls._render(value)

    @classmethod
    def _format_date(cls, value: Any) -> str:
        value = cls._normalize_value(value)
        if value is None:
            return "Not provided."
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, (int, float)):
            timestamp = value / 1000 if abs(value) >= 100_000_000_000 else value
            try:
                return datetime.fromtimestamp(
                    timestamp, timezone.utc
                ).date().isoformat()
            except (OverflowError, OSError, ValueError):
                return str(value)
        text = str(value).strip()
        try:
            return datetime.fromisoformat(
                text.replace("Z", "+00:00")
            ).date().isoformat()
        except ValueError:
            return text or "Not provided."


def main() -> int:
    """Load the default source and print its summary and two document previews."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    loader = HistoricalIncidentLoader()
    try:
        records = loader.load()
        documents = loader.convert_to_documents(records)
    except (DocumentLoadError, FileNotFoundError) as error:
        logger.error("%s", error)
        print(loader.format_loading_summary())
        return 1

    print(loader.format_loading_summary())
    for index, document in enumerate(documents[:2], start=1):
        print(f"\n--- Sample document {index} ---")
        print(document.page_content)
        print(f"Metadata: {json.dumps(document.metadata, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())