from __future__ import annotations

import csv
import importlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from part2_rag.answer_generation import (
    AnswerGenerationError,
    AnswerGenerationResult,
    build_safe_abstention_result,
    build_prompt,
    normalize_provider_answer,
    persist_raw_provider_response,
    should_bypass_provider_for_query_type,
)
from part2_rag.config import get_default_chunk_artifact_path, get_default_eval_path, get_default_eval_runs_dir
from part2_rag.eval_validation import EvalValidationError, validate_eval_set
from part2_rag.llm_providers import (
    BaseLLMProvider,
    ProviderConfigurationError,
    ProviderInvocationError,
    ProviderResponse,
    get_default_generation_settings,
    get_provider,
    get_provider_configuration_status,
)
from part2_rag.query_classification import (
    QueryClassificationError,
    QueryClassificationResult,
    QueryRoutingResult,
    route_and_retrieve,
)
from part2_rag.retrieval import RetrievalError


SUPPORTED_PROVIDER_SELECTIONS = frozenset({"groq", "gemini", "both"})
JSON_INDENT = 2
DEFAULT_BERT_SCORE_MODEL = "distilbert-base-uncased"


class EvalRunnerError(RuntimeError):
    """Raised when eval-runner inputs or artifacts are invalid."""


class BertScoreUnavailableError(RuntimeError):
    """Raised when BERTScore cannot be computed in the current environment."""


@dataclass(frozen=True)
class BertScoreValues:
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class EvalExample:
    question_id: str
    question: str
    question_type: str
    gold_answer: str
    expected_has_answer: bool
    supporting_document_ids: tuple[str, ...]
    supporting_chunk_ids: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalProviderRunRow:
    run_id: str
    question_id: str
    question_type: str
    provider: str
    model: str | None
    expected_has_answer: bool
    insufficient_evidence: bool | None
    answer_text: str
    citation_count: int
    retrieved_chunk_ids: list[str]
    supporting_chunk_ids: list[str]
    retrieved_document_ids: list[str]
    supporting_document_ids: list[str]
    retrieval_hit_at_k: float
    document_hit_at_k: float | None
    rouge_l_f1: float | None
    bert_score_precision: float | None
    bert_score_recall: float | None
    bert_score_f1: float | None
    bert_score_error: str | None
    latency_seconds: float
    classification_latency_seconds: float
    retrieval_latency_seconds: float
    generation_latency_seconds: float | None
    status: str
    error: str | None
    artifact_path: str
    raw_response_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalRunResult:
    run_id: str
    run_dir: str
    results_jsonl_path: str
    results_csv_path: str
    row_count: int
    rows: tuple[EvalProviderRunRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "results_jsonl_path": self.results_jsonl_path,
            "results_csv_path": self.results_csv_path,
            "row_count": self.row_count,
            "rows": [row.to_dict() for row in self.rows],
        }


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload_text = line.strip()
            if not payload_text:
                continue
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                raise EvalRunnerError(
                    f"Eval file line {line_number} is not valid JSON."
                ) from exc
            if not isinstance(payload, dict):
                raise EvalRunnerError(
                    f"Eval file line {line_number} must contain a JSON object."
                )
            rows.append(payload)
    return rows


def load_eval_examples(eval_path: Path) -> tuple[EvalExample, ...]:
    rows = _read_jsonl_rows(eval_path)
    examples: list[EvalExample] = []
    for payload in rows:
        examples.append(
            EvalExample(
                question_id=str(payload["question_id"]).strip(),
                question=str(payload["question"]).strip(),
                question_type=str(payload["question_type"]).strip(),
                gold_answer=str(payload["gold_answer"]).strip(),
                expected_has_answer=bool(payload["expected_has_answer"]),
                supporting_document_ids=tuple(str(item).strip() for item in payload["supporting_document_ids"]),
                supporting_chunk_ids=tuple(str(item).strip() for item in payload["supporting_chunk_ids"]),
                notes=str(payload["notes"]).strip(),
            )
        )
    return tuple(examples)


def filter_eval_examples(
    examples: Sequence[EvalExample],
    *,
    max_examples: int | None = None,
    question_ids: Sequence[str] | None = None,
) -> tuple[EvalExample, ...]:
    normalized_filter = {question_id.strip() for question_id in (question_ids or ()) if question_id.strip()}
    filtered = [
        example
        for example in examples
        if not normalized_filter or example.question_id in normalized_filter
    ]
    if max_examples is not None:
        if max_examples <= 0:
            raise EvalRunnerError("max_examples must be a positive integer.")
        filtered = filtered[:max_examples]
    return tuple(filtered)


def get_provider_names(provider_selection: str) -> tuple[str, ...]:
    normalized = provider_selection.strip().lower()
    if normalized not in SUPPORTED_PROVIDER_SELECTIONS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDER_SELECTIONS))
        raise EvalRunnerError(
            f"Unsupported provider selection {provider_selection!r}. Supported values: {supported}"
        )
    if normalized == "both":
        return ("groq", "gemini")
    return (normalized,)


def compute_retrieval_hit_at_k(
    retrieved_chunk_ids: Sequence[str],
    supporting_chunk_ids: Sequence[str],
) -> float:
    if not supporting_chunk_ids:
        return 1.0 if not retrieved_chunk_ids else 0.0
    supporting = set(supporting_chunk_ids)
    return 1.0 if any(chunk_id in supporting for chunk_id in retrieved_chunk_ids) else 0.0


def compute_document_hit_at_k(
    retrieved_document_ids: Sequence[str],
    supporting_document_ids: Sequence[str],
) -> float | None:
    if not supporting_document_ids:
        return 1.0 if not retrieved_document_ids else 0.0
    supporting = set(supporting_document_ids)
    return 1.0 if any(document_id in supporting for document_id in retrieved_document_ids) else 0.0


def _tokenize_for_rouge(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def compute_rouge_l_f1(reference_text: str, candidate_text: str) -> float:
    reference_tokens = _tokenize_for_rouge(reference_text)
    candidate_tokens = _tokenize_for_rouge(candidate_text)
    if not reference_tokens or not candidate_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, candidate_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _tensor_scalar_to_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def build_bert_score_scorer(
    *,
    model_type: str = DEFAULT_BERT_SCORE_MODEL,
    num_layers: int | None = 6,
    lang: str = "en",
    rescale_with_baseline: bool = False,
) -> Callable[[str, str], BertScoreValues]:
    """Build a lazy BERTScore scorer if the optional dependency is installed."""
    try:
        bert_score_module = importlib.import_module("bert_score")
    except ImportError as exc:
        raise BertScoreUnavailableError(
            "BERTScore unavailable: optional package `bert-score` is not installed. "
            "Install it in Part2/.venv312 to populate bert_score_* metrics."
        ) from exc

    score_fn = getattr(bert_score_module, "score", None)
    if score_fn is None:
        raise BertScoreUnavailableError(
            "BERTScore unavailable: installed `bert_score` package does not expose score()."
        )

    def score(reference_text: str, candidate_text: str) -> BertScoreValues:
        if not reference_text.strip() or not candidate_text.strip():
            return BertScoreValues(precision=0.0, recall=0.0, f1=0.0)
        try:
            score_kwargs: dict[str, Any] = {
                "lang": lang,
                "model_type": model_type,
                "rescale_with_baseline": rescale_with_baseline,
                "verbose": False,
            }
            if num_layers is not None:
                score_kwargs["num_layers"] = num_layers
            precision, recall, f1 = score_fn(
                [candidate_text],
                [reference_text],
                **score_kwargs,
            )
        except Exception as exc:  # pragma: no cover - depends on local model/cache/network state.
            raise BertScoreUnavailableError(
                f"BERTScore unavailable while scoring with {model_type!r}: {exc}"
            ) from exc
        return BertScoreValues(
            precision=_tensor_scalar_to_float(precision[0]),
            recall=_tensor_scalar_to_float(recall[0]),
            f1=_tensor_scalar_to_float(f1[0]),
        )

    return score


def build_results_row(
    *,
    run_id: str,
    example: EvalExample,
    provider_name: str,
    model: str | None,
    routing_result: QueryRoutingResult,
    answer_result: AnswerGenerationResult | None,
    status: str,
    error: str | None,
    artifact_path: Path,
    classification_latency_seconds: float,
    retrieval_latency_seconds: float,
    generation_latency_seconds: float | None,
    total_latency_seconds: float,
    bert_score_scorer: Callable[[str, str], BertScoreValues] | None = None,
) -> EvalProviderRunRow:
    retrieved_chunk_ids = [result.chunk_id for result in routing_result.retrieval_results]
    retrieved_document_ids = [result.document_id for result in routing_result.retrieval_results]
    answer_text = answer_result.answer_text if answer_result is not None else ""
    rouge_l_f1 = (
        compute_rouge_l_f1(example.gold_answer, answer_text)
        if answer_text
        else None
    )
    bert_score_values: BertScoreValues | None = None
    bert_score_error: str | None = None
    if answer_text and bert_score_scorer is not None:
        try:
            bert_score_values = bert_score_scorer(example.gold_answer, answer_text)
        except BertScoreUnavailableError as exc:
            bert_score_error = str(exc)
    return EvalProviderRunRow(
        run_id=run_id,
        question_id=example.question_id,
        question_type=example.question_type,
        provider=provider_name,
        model=model,
        expected_has_answer=example.expected_has_answer,
        insufficient_evidence=(
            answer_result.insufficient_evidence if answer_result is not None else None
        ),
        answer_text=answer_text,
        citation_count=len(answer_result.citations) if answer_result is not None else 0,
        retrieved_chunk_ids=retrieved_chunk_ids,
        supporting_chunk_ids=list(example.supporting_chunk_ids),
        retrieved_document_ids=retrieved_document_ids,
        supporting_document_ids=list(example.supporting_document_ids),
        retrieval_hit_at_k=compute_retrieval_hit_at_k(
            retrieved_chunk_ids,
            example.supporting_chunk_ids,
        ),
        document_hit_at_k=compute_document_hit_at_k(
            retrieved_document_ids,
            example.supporting_document_ids,
        ),
        rouge_l_f1=rouge_l_f1,
        bert_score_precision=(
            bert_score_values.precision if bert_score_values is not None else None
        ),
        bert_score_recall=(
            bert_score_values.recall if bert_score_values is not None else None
        ),
        bert_score_f1=bert_score_values.f1 if bert_score_values is not None else None,
        bert_score_error=bert_score_error,
        latency_seconds=total_latency_seconds,
        classification_latency_seconds=classification_latency_seconds,
        retrieval_latency_seconds=retrieval_latency_seconds,
        generation_latency_seconds=generation_latency_seconds,
        status=status,
        error=error,
        artifact_path=str(artifact_path),
        raw_response_path=answer_result.raw_response_path if answer_result is not None else None,
    )


def _build_provider_response(
    *,
    provider: BaseLLMProvider,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> ProviderResponse:
    return provider.generate(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=JSON_INDENT, sort_keys=True) + "\n", encoding="utf-8")


def _write_results_jsonl(path: Path, rows: Sequence[EvalProviderRunRow]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def _write_results_csv(path: Path, rows: Sequence[EvalProviderRunRow]) -> None:
    fieldnames = list(EvalProviderRunRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.to_dict().items()
                }
            )


def run_eval(
    *,
    eval_path: Path | None = None,
    provider_selection: str = "groq",
    output_root_dir: Path | None = None,
    retrieval_only: bool = False,
    max_examples: int | None = None,
    question_ids: Sequence[str] | None = None,
    save_raw_response: bool = False,
    chunk_artifact_path: Path | None = None,
    route_and_retrieve_fn: Callable[[str], QueryRoutingResult] = route_and_retrieve,
    provider_factory: Callable[[str], BaseLLMProvider] = get_provider,
    provider_status_fn: Callable[[str], tuple[bool, str]] = get_provider_configuration_status,
    bert_score_scorer: Callable[[str, str], BertScoreValues] | None | bool = True,
) -> EvalRunResult:
    chosen_eval_path = (eval_path or get_default_eval_path()).resolve()
    chosen_chunk_artifact_path = (chunk_artifact_path or get_default_chunk_artifact_path()).resolve()
    try:
        validate_eval_set(
            eval_path=chosen_eval_path,
            chunk_artifact_path=chosen_chunk_artifact_path,
        )
    except EvalValidationError as exc:
        raise EvalRunnerError(f"Eval set validation failed: {exc}") from exc

    examples = filter_eval_examples(
        load_eval_examples(chosen_eval_path),
        max_examples=max_examples,
        question_ids=question_ids,
    )
    if not examples:
        raise EvalRunnerError("No eval examples matched the requested filters.")

    provider_names = get_provider_names(provider_selection)
    output_root = (output_root_dir or get_default_eval_runs_dir()).resolve()
    run_id = _utc_run_id()
    run_dir = output_root / run_id
    artifacts_dir = run_dir / "artifacts"
    raw_dir = run_dir / "raw_responses"
    artifacts_dir.mkdir(parents=True, exist_ok=False)
    if save_raw_response:
        raw_dir.mkdir(parents=True, exist_ok=True)

    requested_generation_settings = get_default_generation_settings()
    rows: list[EvalProviderRunRow] = []
    provider_status: dict[str, tuple[bool, str]] = {
        provider_name: provider_status_fn(provider_name) for provider_name in provider_names
    }
    active_bert_score_scorer: Callable[[str, str], BertScoreValues] | None
    bert_score_status: dict[str, Any]
    if bert_score_scorer is True:
        try:
            active_bert_score_scorer = build_bert_score_scorer()
            bert_score_status = {
                "enabled": True,
                "model": DEFAULT_BERT_SCORE_MODEL,
                "message": "BERTScore scorer initialized.",
            }
        except BertScoreUnavailableError as exc:
            active_bert_score_scorer = None
            bert_score_status = {
                "enabled": False,
                "model": DEFAULT_BERT_SCORE_MODEL,
                "message": str(exc),
            }
    elif bert_score_scorer:
        active_bert_score_scorer = bert_score_scorer
        bert_score_status = {
            "enabled": True,
            "model": "custom",
            "message": "BERTScore scorer supplied by caller.",
        }
    else:
        active_bert_score_scorer = None
        bert_score_status = {
            "enabled": False,
            "model": None,
            "message": "BERTScore disabled by caller.",
        }

    for example in examples:
        example_start = time.perf_counter()
        routing_result: QueryRoutingResult | None = None
        route_error: str | None = None
        classification_latency_seconds = 0.0
        retrieval_latency_seconds = 0.0

        try:
            route_started = time.perf_counter()
            routing_result = route_and_retrieve_fn(example.question)
            route_elapsed = time.perf_counter() - route_started
            classification_latency_seconds = 0.0
            retrieval_latency_seconds = route_elapsed
        except (QueryClassificationError, RetrievalError, ValueError, KeyError, TypeError) as exc:
            route_error = str(exc)

        for provider_name in provider_names:
            artifact_path = artifacts_dir / f"{example.question_id}__{provider_name}.json"
            status = "success"
            error: str | None = None
            model_name: str | None = None
            prompt_text: str | None = None
            raw_response_path: str | None = None
            answer_result: AnswerGenerationResult | None = None
            generation_latency_seconds: float | None = None
            provider_response: ProviderResponse | None = None

            if routing_result is None:
                status = "retrieval_error"
                error = route_error or "Retrieval routing failed."
            elif not retrieval_only:
                if should_bypass_provider_for_query_type(
                    routing_result.classification.query_type
                ):
                    answer_result = build_safe_abstention_result(
                        routing_result=routing_result,
                        provider_name=provider_name,
                    )
                    model_name = answer_result.model
                    generation_latency_seconds = 0.0
                else:
                    try:
                        prompt_text = build_prompt(
                            query=routing_result.classification.query,
                            normalized_query=routing_result.classification.normalized_query,
                            query_type=routing_result.classification.query_type,
                            retrieval_results=routing_result.retrieval_results,
                        )
                    except AnswerGenerationError as exc:
                        status = "prompt_error"
                        error = str(exc)

            if status == "success" and retrieval_only:
                status = "retrieval_only"
            elif status == "success" and answer_result is None:
                is_configured, status_message = provider_status[provider_name]
                if not is_configured:
                    status = "provider_unavailable"
                    error = status_message
                else:
                    try:
                        provider = provider_factory(provider_name)
                        model_name = provider.default_model()
                        generation_started = time.perf_counter()
                        provider_response = _build_provider_response(
                            provider=provider,
                            prompt=str(prompt_text),
                            model=model_name,
                            temperature=float(requested_generation_settings["temperature"]),
                            max_tokens=int(requested_generation_settings["max_tokens"]),
                        )
                        generation_latency_seconds = time.perf_counter() - generation_started
                        if save_raw_response:
                            raw_path = persist_raw_provider_response(
                                provider_response=provider_response,
                                prompt=str(prompt_text),
                                query=routing_result.classification.query,
                                normalized_query=routing_result.classification.normalized_query,
                                query_type=routing_result.classification.query_type,
                                runs_dir=raw_dir,
                            )
                            raw_response_path = str(raw_path)
                        answer_result = normalize_provider_answer(
                            query=routing_result.classification.query,
                            normalized_query=routing_result.classification.normalized_query,
                            query_type=routing_result.classification.query_type,
                            provider_response=provider_response,
                            retrieval_results=routing_result.retrieval_results,
                            raw_response_path=raw_response_path,
                        )
                    except ProviderConfigurationError as exc:
                        status = "provider_unavailable"
                        error = str(exc)
                    except ProviderInvocationError as exc:
                        status = "provider_error"
                        error = str(exc)
                    except AnswerGenerationError as exc:
                        status = "normalization_error"
                        error = str(exc)

            total_latency_seconds = time.perf_counter() - example_start
            if model_name is None and provider_response is not None:
                model_name = provider_response.model

            artifact_payload = {
                "run_id": run_id,
                "question_id": example.question_id,
                "provider": provider_name,
                "model": model_name,
                "status": status,
                "error": error,
                "latency_seconds": total_latency_seconds,
                "classification_latency_seconds": classification_latency_seconds,
                "retrieval_latency_seconds": retrieval_latency_seconds,
                "generation_latency_seconds": generation_latency_seconds,
                "eval_example": example.to_dict(),
                "classification_result": (
                    routing_result.classification.to_dict() if routing_result is not None else None
                ),
                "retrieval_mode_used": (
                    routing_result.retrieval_mode_used if routing_result is not None else None
                ),
                "effective_retrieval_config": (
                    routing_result.effective_retrieval_config if routing_result is not None else None
                ),
                "retrieval_results": (
                    [result.to_dict() for result in routing_result.retrieval_results]
                    if routing_result is not None
                    else []
                ),
                "final_prompt": prompt_text,
                "normalized_answer_result": (
                    answer_result.to_dict() if answer_result is not None else None
                ),
                "raw_response_path": raw_response_path,
                "provider_response": (
                    provider_response.to_dict() if provider_response is not None and not save_raw_response else None
                ),
            }
            _write_json(artifact_path, artifact_payload)

            if routing_result is None:
                fallback_routing_result = QueryRoutingResult(
                    classification=QueryClassificationResult(
                        query=example.question,
                        normalized_query=example.question,
                        query_type=example.question_type,
                        confidence=0.0,
                        matched_rules=(),
                    ),
                    retrieval_mode_used="unavailable",
                    effective_retrieval_config={},
                    retrieval_results=(),
                )
                row = build_results_row(
                    run_id=run_id,
                    example=example,
                    provider_name=provider_name,
                    model=model_name,
                    routing_result=fallback_routing_result,
                    answer_result=answer_result,
                    status=status,
                    error=error,
                    artifact_path=artifact_path,
                    classification_latency_seconds=classification_latency_seconds,
                    retrieval_latency_seconds=retrieval_latency_seconds,
                    generation_latency_seconds=generation_latency_seconds,
                    total_latency_seconds=total_latency_seconds,
                    bert_score_scorer=active_bert_score_scorer,
                )
            else:
                row = build_results_row(
                    run_id=run_id,
                    example=example,
                    provider_name=provider_name,
                    model=model_name,
                    routing_result=routing_result,
                    answer_result=answer_result,
                    status=status,
                    error=error,
                    artifact_path=artifact_path,
                    classification_latency_seconds=classification_latency_seconds,
                    retrieval_latency_seconds=retrieval_latency_seconds,
                    generation_latency_seconds=generation_latency_seconds,
                    total_latency_seconds=total_latency_seconds,
                    bert_score_scorer=active_bert_score_scorer,
                )
            rows.append(row)

    results_jsonl_path = run_dir / "results.jsonl"
    results_csv_path = run_dir / "results.csv"
    _write_results_jsonl(results_jsonl_path, rows)
    _write_results_csv(results_csv_path, rows)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "eval_path": str(chosen_eval_path),
            "chunk_artifact_path": str(chosen_chunk_artifact_path),
            "provider_selection": provider_selection,
            "providers": list(provider_names),
            "retrieval_only": retrieval_only,
            "max_examples": max_examples,
            "question_ids": list(question_ids or ()),
            "save_raw_response": save_raw_response,
            "row_count": len(rows),
            "example_count": len(examples),
            "bert_score": bert_score_status,
        },
    )
    return EvalRunResult(
        run_id=run_id,
        run_dir=str(run_dir),
        results_jsonl_path=str(results_jsonl_path),
        results_csv_path=str(results_csv_path),
        row_count=len(rows),
        rows=tuple(rows),
    )
