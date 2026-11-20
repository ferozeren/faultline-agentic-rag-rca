"""Validate the existing incident RAG pipeline end to end."""

from __future__ import annotations

import re
import sys
import time
import warnings
from importlib.metadata import version
from typing import Any

import chromadb
import langchain_core
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.rag.bm25_retriever import bm25_search
from app.rag.hybrid_retriever import hybrid_search
from app.rag.rag_chain import answer_question
from app.rag.ranking import reciprocal_rank_fusion
from app.rag.retriever import retrieve_documents
from app.rag.semantic_retriever import semantic_search
from app.rag.test_nvidia_api import (
    NVIDIA_ENDPOINT,
    _read_setting,
    _safe_error_message,
)
from app.rag.vector_store import COLLECTION_NAME, DEFAULT_PERSIST_DIRECTORY

QUERY = (
    "Payment API is experiencing high latency, database connection errors, and "
    "increased request failures. Investigate similar historical incidents and "
    "suggest the probable root cause and corrective actions."
)
TOP_K = 5
RETRIEVER_FIELDS = {
    "chunk_id",
    "text",
    "metadata",
    "score",
    "rrf_score",
    "retrieval_method",
    "retrieval_sources",
}


class CapturingChatModel:
    """Record the prompt sent to the configured LangChain chat model."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.last_prompt: Any | None = None
        self.response_seconds: float | None = None

    def invoke(self, prompt: Any) -> Any:
        self.last_prompt = prompt
        start_time = time.perf_counter()
        response = self.model.invoke(prompt)
        self.response_seconds = time.perf_counter() - start_time
        return response


def collection_document_count() -> int:
    """Count records in the existing collection without creating a collection."""
    client = chromadb.PersistentClient(path=str(DEFAULT_PERSIST_DIRECTORY))
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection.count()


def valid_results(results: list[dict[str, Any]], fields: set[str]) -> bool:
    """Check that retrieval returned at least one record with the required fields."""
    return bool(results) and all(fields.issubset(result) for result in results)


def _prompt_text(prompt: Any | None) -> str:
    if prompt is None:
        return ""
    if hasattr(prompt, "to_messages"):
        return "\n".join(str(message.content) for message in prompt.to_messages())
    return str(prompt)


def run_pipeline_validation(llm: CapturingChatModel) -> dict[str, Any]:
    """Run each existing pipeline stage and return its results and checks."""
    count = collection_document_count()
    semantic_results = semantic_search(QUERY, top_k=TOP_K)
    bm25_results = bm25_search(QUERY, top_k=TOP_K)
    hybrid_results = hybrid_search(QUERY, top_k=TOP_K)

    hybrid_semantic = hybrid_results["semantic_results"]
    hybrid_bm25 = hybrid_results["bm25_results"]
    all_candidate_ids = [
        result["chunk_id"] for result in hybrid_semantic + hybrid_bm25
    ]
    unique_candidate_count = len(set(all_candidate_ids))
    fused_results = reciprocal_rank_fusion(
        hybrid_semantic,
        hybrid_bm25,
        top_k=max(1, len(all_candidate_ids)),
    )

    retrieved_documents = retrieve_documents(QUERY, top_k=TOP_K)
    answer_error = None
    answer_result = None
    try:
        answer_result = answer_question(QUERY, top_k=TOP_K, llm=llm)
    except Exception as error:
        answer_error = _safe_error_message(
            error,
            _read_setting("NVIDIA_API_KEY"),
        )

    answer = "" if answer_result is None else str(answer_result.get("answer", ""))
    prompt = _prompt_text(llm.last_prompt)
    answer_documents = (
        [] if answer_result is None else answer_result["retrieved_documents"]
    )
    context_received = bool(answer_documents) and all(
        document["chunk_id"] in prompt and document["text"] in prompt
        for document in answer_documents
    )
    sources = [] if answer_result is None else answer_result["sources"]
    source_references = bool(answer) and any(
        source["chunk_id"] in answer
        or source.get("incident_id", "") in answer
        for source in sources
    )

    missing_fields = sorted(
        field
        for document in retrieved_documents
        for field in RETRIEVER_FIELDS
        if field not in document
    )
    checks = {
        "ChromaDB connection": count >= 0,
        "Collection validation": count > 0,
        "Semantic retrieval": valid_results(
            semantic_results, {"chunk_id", "text", "metadata", "distance"}
        ),
        "BM25 retrieval": valid_results(
            bm25_results, {"chunk_id", "text", "metadata", "score"}
        ),
        "Hybrid search": bool(hybrid_semantic and hybrid_bm25),
        "RRF ranking": (
            len(fused_results) == unique_candidate_count
            and len({result["chunk_id"] for result in fused_results})
            == len(fused_results)
            and [result["rrf_score"] for result in fused_results]
            == sorted(
                (result["rrf_score"] for result in fused_results),
                reverse=True,
            )
        ),
        "Retriever integration": bool(retrieved_documents) and not missing_fields,
        "NVIDIA live API call": answer_error is None and answer_result is not None,
        "Final answer generation": bool(answer.strip()),
        "Source references": source_references,
    }
    return {
        "collection_count": count,
        "semantic_results": semantic_results,
        "bm25_results": bm25_results,
        "hybrid_semantic_results": hybrid_semantic,
        "hybrid_bm25_results": hybrid_bm25,
        "candidate_duplicate_count": len(all_candidate_ids) - unique_candidate_count,
        "fused_results": fused_results,
        "retrieved_documents": retrieved_documents,
        "missing_retriever_fields": missing_fields,
        "llm_context_received": context_received,
        "answer_result": answer_result,
        "answer_error": answer_error,
        "answer": answer,
        "nvidia_response_seconds": llm.response_seconds,
        "checks": checks,
    }


def _build_live_model() -> tuple[CapturingChatModel, str]:
    model_name = _read_setting("NVIDIA_MODEL")
    api_key = _read_setting("NVIDIA_API_KEY")
    if not model_name or not api_key:
        raise RuntimeError("NVIDIA_MODEL or NVIDIA_API_KEY is not configured.")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                rf"Found {re.escape(model_name)} in available_models, but type is "
                r"unknown and inference may fail\."
            ),
            category=UserWarning,
            module=r"langchain_nvidia_ai_endpoints\._common",
        )
        model = ChatNVIDIA(
            model=model_name,
            api_key=api_key,
            base_url=NVIDIA_ENDPOINT,
            temperature=0.1,
            max_completion_tokens=1024,
        )
    return CapturingChatModel(model), model_name


def _evaluate_answer(answer: str, sources: list[dict[str, Any]]) -> dict[str, bool]:
    text = answer.casefold()
    return {
        "Symptoms identified": any(
            term in text
            for term in ("latency", "connection", "request failure", "errors")
        ),
        "Probable cause discussed": any(
            term in text
            for term in ("root cause", "caused by", "likely", "probably", "due to")
        ),
        "Corrective actions discussed": any(
            term in text
            for term in (
                "recommend",
                "should",
                "action",
                "mitigate",
                "reduce",
                "increase",
                "validate",
                "monitor",
            )
        ),
        "Evidence distinguished from hypotheses": any(
            term in text
            for term in ("evidence", "historical", "possible", "may", "hypothesis")
        ),
        "Source IDs referenced": bool(answer) and any(
            source["chunk_id"] in answer
            or source.get("incident_id", "") in answer
            for source in sources
        ),
    }


def print_report(report: dict[str, Any], model_name: str) -> None:
    answer_result = report["answer_result"]
    sources = [] if answer_result is None else answer_result["sources"]
    evaluation = _evaluate_answer(report["answer"], sources)

    print("Environment")
    print(f"Python: {sys.version.split()[0]}")
    print(f"ChromaDB: {chromadb.__version__}")
    print(f"LangChain Core: {langchain_core.__version__}")
    print(
        "NVIDIA integration: langchain-nvidia-ai-endpoints "
        f"{version('langchain-nvidia-ai-endpoints')}"
    )
    print(f"NVIDIA model: {model_name}")

    print("\nPipeline Results")
    for component, passed in report["checks"].items():
        print(f"| {component} | {'PASS' if passed else 'FAIL'} |")
    print(f"| LLM received retrieved context | {'PASS' if report['llm_context_received'] else 'FAIL'} |")
    response_seconds = report["nvidia_response_seconds"]
    if response_seconds is not None:
        print(f"NVIDIA response time: {response_seconds:.2f} seconds")

    print("\nCounts")
    print(f"ChromaDB collection: {COLLECTION_NAME}")
    print(f"Stored documents: {report['collection_count']}")
    print(f"Semantic results: {len(report['semantic_results'])}")
    print(f"BM25 results: {len(report['bm25_results'])}")
    print(f"Unique fused results: {len(report['fused_results'])}")
    print(f"Overlapping candidate IDs: {report['candidate_duplicate_count']}")

    print("\nTop Semantic Results")
    for result in report["semantic_results"][:3]:
        print(f"{result['chunk_id']} | distance: {result['distance']:.6f}")
    print("\nTop BM25 Results")
    for result in report["bm25_results"][:3]:
        print(f"{result['chunk_id']} | score: {result['score']:.6f}")
    print("\nTop Fused Results")
    for result in report["fused_results"][:TOP_K]:
        print(
            f"{result['chunk_id']} | RRF: {result['rrf_score']:.6f} | "
            f"sources: {', '.join(result['retrieval_sources'])}"
        )

    print("\nFinal Answer")
    if report["answer_error"]:
        print(f"Answer generation error: {report['answer_error']}")
    else:
        print(report["answer"])
    print("\nSource Incidents")
    for source in sources:
        print(
            f"{source['chunk_id']} | "
            f"incident: {source.get('incident_id', 'not available')}"
        )

    if report["missing_retriever_fields"]:
        print(
            "\nMissing retriever fields: "
            + ", ".join(report["missing_retriever_fields"])
        )
    if answer_result is not None:
        print("\nFinal Answer Evaluation")
        for check, passed in evaluation.items():
            print(f"{check}: {'PASS' if passed else 'FAIL'}")
        print("Claim-to-source support: review the cited passages and answer above.")

    complete = all(report["checks"].values()) and report["llm_context_received"]
    print(
        "\nFinal Verdict: "
        + ("COMPLETE — Entire end-to-end pipeline validated." if complete else "PARTIAL — One or more validation checks failed.")
    )


def main() -> int:
    try:
        llm, model_name = _build_live_model()
        report = run_pipeline_validation(llm)
    except Exception as error:
        safe_error = _safe_error_message(error, _read_setting("NVIDIA_API_KEY"))
        print(f"Pipeline validation failed: {safe_error}")
        print("Final Verdict: FAILED — Pipeline execution stopped.")
        return 1

    print_report(report, model_name)
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())