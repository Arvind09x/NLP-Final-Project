from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from part2_rag.answer_generation import (
    AnswerGenerationResult,
    RetrievedSnippet,
    build_prompt,
    generate_grounded_answer,
)
from part2_rag.config import get_default_runs_dir, get_paths
from part2_rag.llm_providers import get_provider_configuration_status
from part2_rag.query_classification import QueryRoutingResult, route_and_retrieve


LATEST_GROQ_EVAL_RUN_ID = "20260425T210127Z"
LATEST_GROQ_MANUAL_REVIEW_RUN_ID = "20260425T122319Z"
LATEST_GEMINI_EVAL_RUN_ID = "20260426T054042Z"
LATEST_EVAL_RUN_ID = LATEST_GEMINI_EVAL_RUN_ID


@dataclass(frozen=True)
class RetrievalOnlyResult:
    query: str
    normalized_query: str
    query_type: str
    retrieval_mode: str
    provider: str
    model: str
    insufficient_evidence: bool
    retrieved_snippets: tuple[RetrievedSnippet, ...]
    raw_response_path: str | None = None


def _snippets_from_routing(routing_result: QueryRoutingResult) -> tuple[RetrievedSnippet, ...]:
    return tuple(
        RetrievedSnippet(
            source_label=f"S{index}",
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            title=result.title,
            source_type=result.source_type,
            created_utc=result.created_utc,
            snippet=result.snippet,
            retrieval_source=result.retrieval_source,
            score=result.score,
        )
        for index, result in enumerate(routing_result.retrieval_results, start=1)
    )


def run_rag_query(
    query: str,
    *,
    provider_name: str = "groq",
    retrieval_only: bool = False,
    save_raw_response: bool = False,
) -> tuple[AnswerGenerationResult | RetrievalOnlyResult, QueryRoutingResult]:
    """Run the same RAG route/retrieve/generate path used by the CLI."""
    routed = route_and_retrieve(query)
    if retrieval_only:
        return (
            RetrievalOnlyResult(
                query=routed.classification.query,
                normalized_query=routed.classification.normalized_query,
                query_type=routed.classification.query_type,
                retrieval_mode=routed.retrieval_mode_used,
                provider="retrieval-only",
                model="none",
                insufficient_evidence=False,
                retrieved_snippets=_snippets_from_routing(routed),
            ),
            routed,
        )

    result = generate_grounded_answer(
        query,
        provider_name=provider_name,
        save_raw_response=save_raw_response,
        runs_dir=get_default_runs_dir(),
        routing_result=routed,
    )
    return result, routed


def build_prompt_debug_text(query: str, routing_result: QueryRoutingResult) -> str:
    return build_prompt(
        query=query,
        normalized_query=routing_result.classification.normalized_query,
        query_type=routing_result.classification.query_type,
        retrieval_results=routing_result.retrieval_results,
    )


def get_provider_status(provider_name: str) -> tuple[bool, str]:
    return get_provider_configuration_status(provider_name)


def get_report_artifacts() -> dict[str, Path]:
    paths = get_paths()
    groq_eval_run_dir = paths.eval_runs_dir / LATEST_GROQ_EVAL_RUN_ID
    groq_manual_run_dir = paths.eval_runs_dir / LATEST_GROQ_MANUAL_REVIEW_RUN_ID
    gemini_eval_run_dir = paths.eval_runs_dir / LATEST_GEMINI_EVAL_RUN_ID
    return {
        "report": paths.reports_dir / "rag_eval_report_v1.md",
        "groq_eval_summary": groq_eval_run_dir / "eval_summary.md",
        "groq_eval_summary_json": groq_eval_run_dir / "eval_summary.json",
        "groq_eval_results_csv": groq_eval_run_dir / "results.csv",
        "groq_manual_review_summary": groq_manual_run_dir / "manual_review_summary.md",
        "groq_manual_review_summary_json": groq_manual_run_dir / "manual_review_summary.json",
        "gemini_eval_summary": gemini_eval_run_dir / "eval_summary.md",
        "gemini_eval_summary_json": gemini_eval_run_dir / "eval_summary.json",
        "gemini_eval_results_csv": gemini_eval_run_dir / "results.csv",
        "gemini_manual_review_summary": gemini_eval_run_dir / "manual_review_summary.md",
        "gemini_manual_review_summary_json": gemini_eval_run_dir / "manual_review_summary.json",
    }


__all__ = [
    "LATEST_EVAL_RUN_ID",
    "LATEST_GEMINI_EVAL_RUN_ID",
    "LATEST_GROQ_EVAL_RUN_ID",
    "LATEST_GROQ_MANUAL_REVIEW_RUN_ID",
    "RetrievalOnlyResult",
    "build_prompt_debug_text",
    "get_provider_status",
    "get_report_artifacts",
    "run_rag_query",
]
