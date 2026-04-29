from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from part2_rag.config import (
    ADVERSARIAL_NO_ANSWER_QUERY_TYPE,
    FACTUAL_QUERY_TYPE,
    OPINION_SUMMARY_QUERY_TYPE,
    get_default_runs_dir,
)
from part2_rag.llm_providers import (
    BaseLLMProvider,
    ProviderConfigurationError,
    ProviderResponse,
    get_default_generation_settings,
    get_provider,
)
from part2_rag.query_classification import QueryRoutingResult, route_and_retrieve
from part2_rag.retrieval import RetrievalResult


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
DETERMINISTIC_ABSTENTION_MODEL = "deterministic-adversarial-abstention-v1"
PROMPT_INJECTION_RULE_PREFIXES = (
    "adversarial:prompt_injection",
    "adversarial:impossible_system_request",
)


class AnswerGenerationError(RuntimeError):
    """Raised when prompt construction or response normalization fails."""


@dataclass(frozen=True)
class Citation:
    source_label: str
    chunk_id: str
    document_id: str
    title: str | None
    source_type: str
    created_utc: int
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedSnippet:
    source_label: str
    chunk_id: str
    document_id: str
    title: str | None
    source_type: str
    created_utc: int
    snippet: str
    retrieval_source: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerGenerationResult:
    query: str
    normalized_query: str
    query_type: str
    answer_text: str
    citations: tuple[Citation, ...]
    retrieved_snippets: tuple[RetrievedSnippet, ...]
    insufficient_evidence: bool
    provider: str
    model: str
    raw_response_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "query_type": self.query_type,
            "answer_text": self.answer_text,
            "citations": [citation.to_dict() for citation in self.citations],
            "retrieved_snippets": [
                snippet.to_dict() for snippet in self.retrieved_snippets
            ],
            "insufficient_evidence": self.insufficient_evidence,
            "provider": self.provider,
            "model": self.model,
            "raw_response_path": self.raw_response_path,
        }


def _utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slugify_query(query: str, *, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    return slug[:limit] or "query"


def _query_type_instruction(query_type: str) -> str:
    if query_type == FACTUAL_QUERY_TYPE:
        return (
            "This is a factual question. Give a concise direct answer grounded in the strongest "
            "retrieved evidence."
        )
    if query_type == OPINION_SUMMARY_QUERY_TYPE:
        return (
            "This is an opinion-summary question. Summarize community perspectives from the "
            "retrieved evidence, note disagreement or variation, and avoid presenting a single "
            "opinion as universal."
        )
    if query_type == ADVERSARIAL_NO_ANSWER_QUERY_TYPE:
        return (
            "This is an adversarial or likely no-answer question. Prefer abstention unless the "
            "retrieved evidence clearly supports the answer. If support is weak or off-topic, say "
            "the evidence is insufficient."
        )
    raise AnswerGenerationError(f"Unsupported query_type={query_type!r}")


def build_prompt(
    *,
    query: str,
    normalized_query: str,
    query_type: str,
    retrieval_results: Sequence[RetrievalResult],
) -> str:
    if not retrieval_results:
        raise AnswerGenerationError("Cannot build a grounded prompt without retrieved context.")

    allowed_source_labels = [f"S{index}" for index, _ in enumerate(retrieval_results, start=1)]
    context_blocks: list[str] = []
    for source_label, result in zip(allowed_source_labels, retrieval_results, strict=True):
        context_blocks.append(
            "\n".join(
                [
                    f"source_label: {source_label}",
                    f"source_type: {result.source_type}",
                    f"title: {result.title!r}",
                    f"created_utc: {result.created_utc}",
                    f"retrieval_source: {result.retrieval_source}",
                    f"score: {result.score:.6f}",
                    f"snippet: {result.snippet}",
                ]
            )
        )

    prompt_sections = [
        "Answer using only the retrieved r/fitness context below.",
        _query_type_instruction(query_type),
        "Use only the retrieved context below. Do not use outside knowledge.",
        "If the evidence is missing, conflicting, or insufficient, say so clearly.",
        "Do not invent facts, sources, or citations.",
        "Return exactly one JSON object with this schema:",
        (
            '{'
            '"answer_text": string, '
            '"insufficient_evidence": boolean, '
            '"citations": [string, ...]'
            "}"
        ),
        "Rules:",
        f"- Use citations only from this exact list: {', '.join(allowed_source_labels)}.",
        "- Do not cite chunk IDs or document IDs directly.",
        "- `citations` must contain only source labels from the retrieved context.",
        "- If you answer substantively, cite the source labels that support the answer.",
        "- If evidence is insufficient, set `insufficient_evidence` to true and still cite the most relevant source labels if any were reviewed.",
        f"raw_query: {query}",
        f"normalized_query: {normalized_query}",
        f"query_type: {query_type}",
        "retrieved_context:",
        "\n\n".join(context_blocks),
    ]
    return "\n\n".join(prompt_sections)


def _extract_json_payload(response_text: str) -> dict[str, Any]:
    stripped = response_text.strip()
    if not stripped:
        raise AnswerGenerationError("Provider returned an empty response.")

    candidates = [stripped]
    fenced_match = JSON_BLOCK_PATTERN.search(stripped)
    if fenced_match is not None:
        candidates.insert(0, fenced_match.group(1).strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise AnswerGenerationError("Provider response did not contain a valid JSON object.")


def _normalize_citation_values(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise AnswerGenerationError(
            f"Provider response field `{field_name}` must be a list."
        )
    normalized: list[str] = []
    for item in value:
        if isinstance(item, dict):
            raw_value = (
                item.get("source_label")
                or item.get("citation")
                or item.get("chunk_id")
                or item.get("document_id")
            )
        else:
            raw_value = item
        citation_value = str(raw_value).strip()
        if citation_value:
            normalized.append(citation_value)
    return tuple(normalized)


def _build_source_label_map(
    retrieval_results: Sequence[RetrievalResult],
) -> dict[str, RetrievalResult]:
    return {
        f"S{index}": result for index, result in enumerate(retrieval_results, start=1)
    }


def _resolve_citation_target(
    citation_value: str,
    *,
    source_label_map: dict[str, RetrievalResult],
    document_id_map: dict[str, tuple[RetrievalResult, ...]],
) -> tuple[str, RetrievalResult]:
    normalized_value = citation_value.strip()
    if not normalized_value:
        raise AnswerGenerationError("Provider returned an empty citation entry.")

    canonical_label = normalized_value.upper()
    if canonical_label in source_label_map:
        return canonical_label, source_label_map[canonical_label]

    document_matches = document_id_map.get(normalized_value)
    if document_matches:
        if len(document_matches) == 1:
            matched_result = document_matches[0]
            for source_label, candidate in source_label_map.items():
                if candidate.chunk_id == matched_result.chunk_id:
                    return source_label, matched_result
            raise AnswerGenerationError(
                f"Provider cited document_id={normalized_value!r}, but the mapped source label could not be resolved."
            )
        raise AnswerGenerationError(
            f"Provider cited document_id={normalized_value!r}, but it matched multiple retrieved snippets. Use only allowed source labels."
        )

    raise AnswerGenerationError(
        f"Provider cited source_label={normalized_value!r}, which was not present in retrieved context."
    )


def _extract_citations(payload: dict[str, Any]) -> tuple[str, ...]:
    if "citations" in payload:
        return _normalize_citation_values(payload.get("citations"), field_name="citations")
    if "cited_chunk_ids" in payload:
        return _normalize_citation_values(
            payload.get("cited_chunk_ids"),
            field_name="cited_chunk_ids",
        )
    return ()


def normalize_provider_answer(
    *,
    query: str,
    normalized_query: str,
    query_type: str,
    provider_response: ProviderResponse,
    retrieval_results: Sequence[RetrievalResult],
    raw_response_path: str | None = None,
) -> AnswerGenerationResult:
    payload = _extract_json_payload(provider_response.text)
    answer_text = str(payload.get("answer_text", "")).strip()
    insufficient_evidence = bool(payload.get("insufficient_evidence", False))
    cited_values = _extract_citations(payload)

    if not answer_text:
        raise AnswerGenerationError("Provider response is missing a non-empty `answer_text`.")

    source_label_map = _build_source_label_map(retrieval_results)
    document_id_map: dict[str, tuple[RetrievalResult, ...]] = {}
    for result in retrieval_results:
        document_id_map.setdefault(result.document_id, ())
        document_id_map[result.document_id] = (
            *document_id_map[result.document_id],
            result,
        )
    citations: list[Citation] = []
    seen_source_labels: set[str] = set()
    for citation_value in cited_values:
        source_label, result = _resolve_citation_target(
            citation_value,
            source_label_map=source_label_map,
            document_id_map=document_id_map,
        )
        if source_label in seen_source_labels:
            continue
        seen_source_labels.add(source_label)
        citations.append(
            Citation(
                source_label=source_label,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                title=result.title,
                source_type=result.source_type,
                created_utc=result.created_utc,
                snippet=result.snippet,
            )
        )

    if not insufficient_evidence and not citations:
        raise AnswerGenerationError(
            "Provider returned an answer without citations. Grounded answers must cite retrieved source labels."
        )

    retrieved_snippets = tuple(
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
        for index, result in enumerate(retrieval_results, start=1)
    )
    return AnswerGenerationResult(
        query=query,
        normalized_query=normalized_query,
        query_type=query_type,
        answer_text=answer_text,
        citations=tuple(citations),
        retrieved_snippets=retrieved_snippets,
        insufficient_evidence=insufficient_evidence,
        provider=provider_response.provider,
        model=provider_response.model,
        raw_response_path=raw_response_path,
    )


def persist_raw_provider_response(
    *,
    provider_response: ProviderResponse,
    prompt: str,
    query: str,
    normalized_query: str,
    query_type: str,
    runs_dir: Path | None = None,
) -> Path:
    target_dir = (runs_dir or get_default_runs_dir()).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{_utc_timestamp_slug()}_"
        f"{provider_response.provider}_"
        f"{_slugify_query(query)}.json"
    )
    output_path = target_dir / filename
    payload = {
        "query": query,
        "normalized_query": normalized_query,
        "query_type": query_type,
        "provider": provider_response.provider,
        "model": provider_response.model,
        "prompt": prompt,
        "provider_text": provider_response.text,
        "provider_raw_response": provider_response.raw_response,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def should_bypass_provider_for_query_type(
    query_type: str,
    *,
    allow_adversarial_provider: bool = False,
) -> bool:
    return (
        query_type == ADVERSARIAL_NO_ANSWER_QUERY_TYPE
        and not allow_adversarial_provider
    )


def _safe_adversarial_answer_text(routing_result: QueryRoutingResult) -> str:
    matched_rules = routing_result.classification.matched_rules
    if any(
        rule.startswith(PROMPT_INJECTION_RULE_PREFIXES)
        for rule in matched_rules
    ):
        return (
            "I can't help with requests to ignore instructions or reveal hidden guidance. "
            "There is not enough corpus evidence to answer that safely."
        )
    return (
        "I can't answer that from the frozen r/fitness corpus. "
        "The available evidence is insufficient or out of domain."
    )


def build_safe_abstention_result(
    *,
    routing_result: QueryRoutingResult,
    provider_name: str,
) -> AnswerGenerationResult:
    retrieved_snippets = tuple(
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
    return AnswerGenerationResult(
        query=routing_result.classification.query,
        normalized_query=routing_result.classification.normalized_query,
        query_type=routing_result.classification.query_type,
        answer_text=_safe_adversarial_answer_text(routing_result),
        citations=(),
        retrieved_snippets=retrieved_snippets,
        insufficient_evidence=True,
        provider=provider_name,
        model=DETERMINISTIC_ABSTENTION_MODEL,
        raw_response_path=None,
    )


def generate_grounded_answer(
    query: str,
    *,
    provider_name: str,
    provider: BaseLLMProvider | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    save_raw_response: bool = False,
    runs_dir: Path | None = None,
    routing_result: QueryRoutingResult | None = None,
    allow_adversarial_provider: bool = False,
) -> AnswerGenerationResult:
    routed = routing_result or route_and_retrieve(query)
    if should_bypass_provider_for_query_type(
        routed.classification.query_type,
        allow_adversarial_provider=allow_adversarial_provider,
    ):
        return build_safe_abstention_result(
            routing_result=routed,
            provider_name=provider_name,
        )
    if not routed.retrieval_results:
        raise AnswerGenerationError("No retrieved context is available for answer generation.")

    chosen_provider = provider or get_provider(provider_name)
    generation_settings = get_default_generation_settings()
    chosen_model = model or chosen_provider.default_model()
    chosen_temperature = (
        generation_settings["temperature"] if temperature is None else temperature
    )
    chosen_max_tokens = generation_settings["max_tokens"] if max_tokens is None else max_tokens

    prompt = build_prompt(
        query=routed.classification.query,
        normalized_query=routed.classification.normalized_query,
        query_type=routed.classification.query_type,
        retrieval_results=routed.retrieval_results,
    )

    def _call_provider(prompt_text: str) -> ProviderResponse:
        return chosen_provider.generate(
            prompt_text,
            model=chosen_model,
            temperature=float(chosen_temperature),
            max_tokens=int(chosen_max_tokens),
            generation_mode="json",
        )

    provider_response = _call_provider(prompt)
    retry_prompt: str | None = None
    try:
        result = normalize_provider_answer(
            query=routed.classification.query,
            normalized_query=routed.classification.normalized_query,
            query_type=routed.classification.query_type,
            provider_response=provider_response,
            retrieval_results=routed.retrieval_results,
        )
    except AnswerGenerationError as exc:
        retry_prompt = (
            f"{prompt}\n\n"
            "Your previous response was invalid.\n"
            f"Validation error: {exc}\n"
            "Return exactly one JSON object and use only the allowed source labels in `citations`.\n"
            "Do not cite chunk IDs or document IDs directly."
        )
        provider_response = _call_provider(retry_prompt)
        result = normalize_provider_answer(
            query=routed.classification.query,
            normalized_query=routed.classification.normalized_query,
            query_type=routed.classification.query_type,
            provider_response=provider_response,
            retrieval_results=routed.retrieval_results,
        )

    raw_response_path: str | None = None
    if save_raw_response:
        raw_path = persist_raw_provider_response(
            provider_response=provider_response,
            prompt=retry_prompt or prompt,
            query=routed.classification.query,
            normalized_query=routed.classification.normalized_query,
            query_type=routed.classification.query_type,
            runs_dir=runs_dir,
        )
        raw_response_path = str(raw_path)

    return AnswerGenerationResult(
        query=result.query,
        normalized_query=result.normalized_query,
        query_type=result.query_type,
        answer_text=result.answer_text,
        citations=result.citations,
        retrieved_snippets=result.retrieved_snippets,
        insufficient_evidence=result.insufficient_evidence,
        provider=result.provider,
        model=result.model,
        raw_response_path=raw_response_path,
    )


def format_answer_generation_result(result: AnswerGenerationResult) -> str:
    lines = [
        "Grounded QA result",
        "",
        "request:",
        f"  query: {result.query}",
        f"  normalized_query: {result.normalized_query}",
        f"  query_type: {result.query_type}",
        f"  provider_model: {result.provider}/{result.model}",
        f"  insufficient_evidence: {str(result.insufficient_evidence).lower()}",
        "",
        "final_answer:",
        f"  {result.answer_text}",
        "",
        "citations:",
    ]
    if result.citations:
        for citation in result.citations:
            lines.append(
                f"  - source_label={citation.source_label} | chunk_id={citation.chunk_id} "
                f"| document_id={citation.document_id} | source_type={citation.source_type} "
                f"| created_utc={citation.created_utc} "
                f"| title={citation.title!r}"
            )
            lines.append(f"    snippet: {citation.snippet}")
    else:
        lines.append("  (none)")

    lines.extend(["", "retrieved_source_previews:"])
    for snippet in result.retrieved_snippets:
        lines.append(
            f"  [{snippet.source_label}] chunk_id={snippet.chunk_id} | document_id={snippet.document_id} "
            f"| source_type={snippet.source_type} | retrieval_source={snippet.retrieval_source} "
            f"| score={snippet.score:.6f} | title={snippet.title!r}"
        )
        lines.append(f"    snippet: {snippet.snippet}")

    if result.raw_response_path is not None:
        lines.extend(["", f"raw_response_path: {result.raw_response_path}"])
    return "\n".join(lines)


__all__ = [
    "AnswerGenerationError",
    "AnswerGenerationResult",
    "Citation",
    "DETERMINISTIC_ABSTENTION_MODEL",
    "RetrievedSnippet",
    "build_safe_abstention_result",
    "build_prompt",
    "format_answer_generation_result",
    "generate_grounded_answer",
    "normalize_provider_answer",
    "persist_raw_provider_response",
    "should_bypass_provider_for_query_type",
]
