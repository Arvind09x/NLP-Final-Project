from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from part2_rag.eval_runner import EvalExample, EvalProviderRunRow, load_eval_examples


JSON_INDENT = 2
REVIEW_CSV_FIELDNAMES = (
    "run_id",
    "question_id",
    "question_type",
    "question",
    "gold_answer",
    "expected_has_answer",
    "provider",
    "model",
    "status",
    "answer_text",
    "insufficient_evidence",
    "citation_count",
    "retrieval_hit_at_k",
    "document_hit_at_k",
    "rouge_l_f1",
    "bert_score_precision",
    "bert_score_recall",
    "bert_score_f1",
    "bert_score_error",
    "supporting_chunk_ids",
    "retrieved_chunk_ids",
    "answer_faithful",
    "citation_valid",
    "correct_abstention",
    "retrieval_error_type",
    "answer_error_type",
    "review_notes",
)
ADVERSARIAL_LEAKAGE_PATTERN = re.compile(
    r"\b(?:system prompt|hidden instructions?|developer prompt|internal prompt)\b",
    re.IGNORECASE,
)


class EvalReportingError(RuntimeError):
    """Raised when eval reporting inputs or artifacts are invalid."""


@dataclass(frozen=True)
class LoadedEvalRun:
    run_id: str
    run_dir: str
    results_jsonl_path: str
    results_csv_path: str | None
    artifacts_dir: str
    row_count: int
    rows: tuple[EvalProviderRunRow, ...]
    artifact_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManualReviewRow:
    run_id: str
    question_id: str
    question_type: str
    question: str
    gold_answer: str
    expected_has_answer: bool
    provider: str
    model: str | None
    status: str
    answer_text: str
    insufficient_evidence: bool | None
    citation_count: int
    retrieval_hit_at_k: float
    document_hit_at_k: float | None
    rouge_l_f1: float | None
    bert_score_precision: float | None
    bert_score_recall: float | None
    bert_score_f1: float | None
    bert_score_error: str | None
    supporting_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    answer_faithful: str
    citation_valid: str
    correct_abstention: str
    retrieval_error_type: str
    answer_error_type: str
    review_notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalSummary:
    run_id: str
    run_dir: str
    total_rows: int
    status_counts: dict[str, int]
    question_type_counts: dict[str, int]
    answer_bearing_row_count: int
    answer_bearing_success_count: int
    retrieval_hit_at_k_answer_bearing: float | None
    document_hit_at_k_answer_bearing: float | None
    average_rouge_l_f1_answer_bearing_success: float | None
    average_bert_score_f1_answer_bearing_success: float | None
    adversarial_row_count: int
    adversarial_correct_abstention_count: int
    retrieval_miss_count: int
    failed_abstention_count: int
    top_failure_examples: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManualReviewSummary:
    run_id: str
    total_rows: int
    answer_bearing_row_count: int
    adversarial_row_count: int
    faithfulness_yes_count: int
    faithfulness_partial_count: int
    faithfulness_no_count: int
    faithfulness_percentage: float | None
    citation_valid_yes_count: int
    citation_valid_partial_count: int
    citation_valid_no_count: int
    citation_validity_percentage: float | None
    correct_abstention_yes_count: int
    correct_abstention_no_count: int
    correct_abstention_percentage: float | None
    faithfulness_by_question_type: dict[str, dict[str, Any]]
    retrieval_miss_count: int
    answer_failure_count: int
    partial_answer_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
                raise EvalReportingError(
                    f"Run results line {line_number} is not valid JSON."
                ) from exc
            if not isinstance(payload, dict):
                raise EvalReportingError(
                    f"Run results line {line_number} must contain a JSON object."
                )
            rows.append(payload)
    return rows


def _parse_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    raise EvalReportingError(f"Field `{field_name}` must be a list.")


def _build_run_row(payload: dict[str, Any]) -> EvalProviderRunRow:
    return EvalProviderRunRow(
        run_id=str(payload["run_id"]),
        question_id=str(payload["question_id"]),
        question_type=str(payload["question_type"]),
        provider=str(payload["provider"]),
        model=str(payload["model"]) if payload.get("model") is not None else None,
        expected_has_answer=bool(payload["expected_has_answer"]),
        insufficient_evidence=(
            bool(payload["insufficient_evidence"])
            if payload.get("insufficient_evidence") is not None
            else None
        ),
        answer_text=str(payload.get("answer_text", "")),
        citation_count=int(payload["citation_count"]),
        retrieved_chunk_ids=_parse_string_list(
            payload.get("retrieved_chunk_ids"),
            field_name="retrieved_chunk_ids",
        ),
        supporting_chunk_ids=_parse_string_list(
            payload.get("supporting_chunk_ids"),
            field_name="supporting_chunk_ids",
        ),
        retrieved_document_ids=_parse_string_list(
            payload.get("retrieved_document_ids"),
            field_name="retrieved_document_ids",
        ),
        supporting_document_ids=_parse_string_list(
            payload.get("supporting_document_ids"),
            field_name="supporting_document_ids",
        ),
        retrieval_hit_at_k=float(payload["retrieval_hit_at_k"]),
        document_hit_at_k=(
            float(payload["document_hit_at_k"])
            if payload.get("document_hit_at_k") is not None
            else None
        ),
        rouge_l_f1=float(payload["rouge_l_f1"]) if payload.get("rouge_l_f1") is not None else None,
        bert_score_precision=(
            float(payload["bert_score_precision"])
            if payload.get("bert_score_precision") is not None
            else None
        ),
        bert_score_recall=(
            float(payload["bert_score_recall"])
            if payload.get("bert_score_recall") is not None
            else None
        ),
        bert_score_f1=(
            float(payload["bert_score_f1"])
            if payload.get("bert_score_f1") is not None
            else None
        ),
        bert_score_error=(
            str(payload["bert_score_error"])
            if payload.get("bert_score_error") is not None
            else None
        ),
        latency_seconds=float(payload["latency_seconds"]),
        classification_latency_seconds=float(payload["classification_latency_seconds"]),
        retrieval_latency_seconds=float(payload["retrieval_latency_seconds"]),
        generation_latency_seconds=(
            float(payload["generation_latency_seconds"])
            if payload.get("generation_latency_seconds") is not None
            else None
        ),
        status=str(payload["status"]),
        error=str(payload["error"]) if payload.get("error") is not None else None,
        artifact_path=str(payload["artifact_path"]),
        raw_response_path=(
            str(payload["raw_response_path"])
            if payload.get("raw_response_path") is not None
            else None
        ),
    )


def load_eval_run(run_dir: Path) -> LoadedEvalRun:
    resolved_run_dir = run_dir.resolve()
    results_jsonl_path = resolved_run_dir / "results.jsonl"
    if not results_jsonl_path.exists():
        raise EvalReportingError(
            f"Run directory is missing results.jsonl: {results_jsonl_path}"
        )

    raw_rows = _read_jsonl_rows(results_jsonl_path)
    rows = tuple(_build_run_row(payload) for payload in raw_rows)
    artifacts_dir = resolved_run_dir / "artifacts"
    if not artifacts_dir.exists():
        raise EvalReportingError(
            f"Run directory is missing artifacts directory: {artifacts_dir}"
        )

    artifact_paths = tuple(sorted(str(path.resolve()) for path in artifacts_dir.glob("*.json")))
    if len(artifact_paths) < len(rows):
        raise EvalReportingError(
            "Run artifacts are incomplete: expected at least one artifact per result row."
        )

    missing_artifacts = [row.artifact_path for row in rows if not Path(row.artifact_path).exists()]
    if missing_artifacts:
        raise EvalReportingError(
            f"Run results reference missing artifacts: {missing_artifacts[0]}"
        )

    results_csv_path = resolved_run_dir / "results.csv"
    return LoadedEvalRun(
        run_id=resolved_run_dir.name,
        run_dir=str(resolved_run_dir),
        results_jsonl_path=str(results_jsonl_path),
        results_csv_path=str(results_csv_path) if results_csv_path.exists() else None,
        artifacts_dir=str(artifacts_dir),
        row_count=len(rows),
        rows=rows,
        artifact_paths=artifact_paths,
    )


def _build_example_map(examples: Sequence[EvalExample]) -> dict[str, EvalExample]:
    return {example.question_id: example for example in examples}


def _prefill_retrieval_error_type(row: EvalProviderRunRow) -> str:
    if row.expected_has_answer and row.retrieval_hit_at_k == 0.0:
        return "retrieval_miss"
    return ""


def _prefill_answer_error_type(row: EvalProviderRunRow) -> str:
    if (
        not row.expected_has_answer
        and row.answer_text
        and ADVERSARIAL_LEAKAGE_PATTERN.search(row.answer_text)
    ):
        return "failed_adversarial_abstention"
    if not row.expected_has_answer and row.insufficient_evidence is False:
        return "failed_abstention"
    return ""


def build_manual_review_rows(
    *,
    run: LoadedEvalRun,
    eval_examples: Sequence[EvalExample],
) -> tuple[ManualReviewRow, ...]:
    example_map = _build_example_map(eval_examples)
    review_rows: list[ManualReviewRow] = []
    for row in run.rows:
        example = example_map.get(row.question_id)
        if example is None:
            raise EvalReportingError(
                f"Run results contain unknown question_id={row.question_id!r}."
            )
        review_rows.append(
            ManualReviewRow(
                run_id=row.run_id,
                question_id=row.question_id,
                question_type=row.question_type,
                question=example.question,
                gold_answer=example.gold_answer,
                expected_has_answer=row.expected_has_answer,
                provider=row.provider,
                model=row.model,
                status=row.status,
                answer_text=row.answer_text,
                insufficient_evidence=row.insufficient_evidence,
                citation_count=row.citation_count,
                retrieval_hit_at_k=row.retrieval_hit_at_k,
                document_hit_at_k=row.document_hit_at_k,
                rouge_l_f1=row.rouge_l_f1,
                bert_score_precision=row.bert_score_precision,
                bert_score_recall=row.bert_score_recall,
                bert_score_f1=row.bert_score_f1,
                bert_score_error=row.bert_score_error,
                supporting_chunk_ids=list(row.supporting_chunk_ids),
                retrieved_chunk_ids=list(row.retrieved_chunk_ids),
                answer_faithful="",
                citation_valid="",
                correct_abstention="",
                retrieval_error_type=_prefill_retrieval_error_type(row),
                answer_error_type=_prefill_answer_error_type(row),
                review_notes="",
            )
        )
    return tuple(review_rows)


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def summarize_eval_run(
    *,
    run: LoadedEvalRun,
    review_rows: Sequence[ManualReviewRow],
) -> EvalSummary:
    status_counts = Counter(row.status for row in run.rows)
    question_type_counts = Counter(row.question_type for row in run.rows)
    answer_bearing_rows = [
        row for row in run.rows if row.expected_has_answer
    ]
    answer_bearing_success_rows = [
        row for row in answer_bearing_rows if row.status == "success"
    ]
    adversarial_rows = [
        row for row in run.rows if not row.expected_has_answer
    ]
    retrieval_miss_rows = [
        row for row in review_rows if row.retrieval_error_type == "retrieval_miss"
    ]
    failed_abstention_rows = [
        row
        for row in review_rows
        if row.answer_error_type in {"failed_abstention", "failed_adversarial_abstention"}
    ]
    adversarial_correct_abstention_count = sum(
        1
        for row in review_rows
        if row.status == "success"
        and not row.expected_has_answer
        and row.insufficient_evidence is True
        and not row.answer_error_type
    )

    top_failure_examples: list[dict[str, str]] = []
    seen_question_ids: set[str] = set()
    for review_row in review_rows:
        reasons = [value for value in (review_row.retrieval_error_type, review_row.answer_error_type) if value]
        if not reasons:
            continue
        if review_row.question_id in seen_question_ids:
            continue
        seen_question_ids.add(review_row.question_id)
        top_failure_examples.append(
            {
                "question_id": review_row.question_id,
                "question_type": review_row.question_type,
                "reason": "; ".join(reasons),
            }
        )

    return EvalSummary(
        run_id=run.run_id,
        run_dir=run.run_dir,
        total_rows=run.row_count,
        status_counts=dict(status_counts),
        question_type_counts=dict(question_type_counts),
        answer_bearing_row_count=len(answer_bearing_rows),
        answer_bearing_success_count=len(answer_bearing_success_rows),
        retrieval_hit_at_k_answer_bearing=_mean(
            row.retrieval_hit_at_k for row in answer_bearing_rows
        ),
        document_hit_at_k_answer_bearing=_mean(
            row.document_hit_at_k
            for row in answer_bearing_rows
            if row.document_hit_at_k is not None
        ),
        average_rouge_l_f1_answer_bearing_success=_mean(
            row.rouge_l_f1 for row in answer_bearing_success_rows if row.rouge_l_f1 is not None
        ),
        average_bert_score_f1_answer_bearing_success=_mean(
            row.bert_score_f1
            for row in answer_bearing_success_rows
            if row.bert_score_f1 is not None
        ),
        adversarial_row_count=len(adversarial_rows),
        adversarial_correct_abstention_count=adversarial_correct_abstention_count,
        retrieval_miss_count=len(retrieval_miss_rows),
        failed_abstention_count=len(failed_abstention_rows),
        top_failure_examples=top_failure_examples,
    )


def write_manual_review_csv(path: Path, rows: Sequence[ManualReviewRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_CSV_FIELDNAMES))
        writer.writeheader()
        for row in rows:
            payload = row.to_dict()
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, list)
                    else value
                    for key, value in payload.items()
                }
            )


def _parse_bool_string(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise EvalReportingError(f"Expected boolean string, got {value!r}.")


def _parse_optional_bool_string(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized == "":
        return None
    return _parse_bool_string(value)


def _parse_optional_float_string(value: str) -> float | None:
    normalized = value.strip()
    if normalized == "":
        return None
    return float(normalized)


def _parse_json_list_string(value: str, *, field_name: str) -> list[str]:
    normalized = value.strip()
    if normalized == "":
        return []
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise EvalReportingError(
            f"Manual review field `{field_name}` must contain a JSON list string."
        ) from exc
    if not isinstance(payload, list):
        raise EvalReportingError(
            f"Manual review field `{field_name}` must contain a JSON list."
        )
    return [str(item) for item in payload]


def load_manual_review_csv(path: Path) -> tuple[ManualReviewRow, ...]:
    if not path.exists():
        raise EvalReportingError(f"Manual review CSV does not exist: {path}")

    rows: list[ManualReviewRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, payload in enumerate(reader, start=2):
            try:
                rows.append(
                    ManualReviewRow(
                        run_id=str(payload["run_id"]),
                        question_id=str(payload["question_id"]),
                        question_type=str(payload["question_type"]),
                        question=str(payload["question"]),
                        gold_answer=str(payload["gold_answer"]),
                        expected_has_answer=_parse_bool_string(
                            str(payload["expected_has_answer"])
                        ),
                        provider=str(payload["provider"]),
                        model=(
                            str(payload["model"])
                            if payload.get("model") not in (None, "")
                            else None
                        ),
                        status=str(payload["status"]),
                        answer_text=str(payload["answer_text"]),
                        insufficient_evidence=_parse_optional_bool_string(
                            str(payload.get("insufficient_evidence", ""))
                        ),
                        citation_count=int(str(payload["citation_count"])),
                        retrieval_hit_at_k=float(str(payload["retrieval_hit_at_k"])),
                        document_hit_at_k=_parse_optional_float_string(
                            str(payload.get("document_hit_at_k", ""))
                        ),
                        rouge_l_f1=_parse_optional_float_string(
                            str(payload.get("rouge_l_f1", ""))
                        ),
                        bert_score_precision=_parse_optional_float_string(
                            str(payload.get("bert_score_precision", ""))
                        ),
                        bert_score_recall=_parse_optional_float_string(
                            str(payload.get("bert_score_recall", ""))
                        ),
                        bert_score_f1=_parse_optional_float_string(
                            str(payload.get("bert_score_f1", ""))
                        ),
                        bert_score_error=(
                            str(payload.get("bert_score_error", ""))
                            if payload.get("bert_score_error", "") != ""
                            else None
                        ),
                        supporting_chunk_ids=_parse_json_list_string(
                            str(payload.get("supporting_chunk_ids", "")),
                            field_name="supporting_chunk_ids",
                        ),
                        retrieved_chunk_ids=_parse_json_list_string(
                            str(payload.get("retrieved_chunk_ids", "")),
                            field_name="retrieved_chunk_ids",
                        ),
                        answer_faithful=str(payload.get("answer_faithful", "")),
                        citation_valid=str(payload.get("citation_valid", "")),
                        correct_abstention=str(payload.get("correct_abstention", "")),
                        retrieval_error_type=str(payload.get("retrieval_error_type", "")),
                        answer_error_type=str(payload.get("answer_error_type", "")),
                        review_notes=str(payload.get("review_notes", "")),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise EvalReportingError(
                    f"Failed to parse manual review CSV row at line {line_number}."
                ) from exc
    return tuple(rows)


def _normalized_review_value(value: str) -> str:
    return value.strip().lower()


def _count_review_values(
    rows: Sequence[ManualReviewRow],
    *,
    field_name: str,
) -> dict[str, int]:
    counts = {"yes": 0, "partial": 0, "no": 0}
    for row in rows:
        value = _normalized_review_value(str(getattr(row, field_name)))
        if value in counts:
            counts[value] += 1
    return counts


def _safe_percentage(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def summarize_manual_review(rows: Sequence[ManualReviewRow]) -> ManualReviewSummary:
    if not rows:
        raise EvalReportingError("Manual review summary requires at least one row.")

    run_id = rows[0].run_id
    answer_bearing_rows = [row for row in rows if row.expected_has_answer]
    adversarial_rows = [row for row in rows if not row.expected_has_answer]

    faithfulness_counts = _count_review_values(
        answer_bearing_rows,
        field_name="answer_faithful",
    )
    citation_counts = _count_review_values(
        answer_bearing_rows,
        field_name="citation_valid",
    )
    abstention_counts = _count_review_values(
        adversarial_rows,
        field_name="correct_abstention",
    )

    faithfulness_by_question_type: dict[str, dict[str, Any]] = {}
    for question_type in sorted({row.question_type for row in answer_bearing_rows}):
        subset = [row for row in answer_bearing_rows if row.question_type == question_type]
        counts = _count_review_values(subset, field_name="answer_faithful")
        faithfulness_by_question_type[question_type] = {
            "row_count": len(subset),
            "yes_count": counts["yes"],
            "partial_count": counts["partial"],
            "no_count": counts["no"],
            "faithfulness_percentage": _safe_percentage(counts["yes"], len(subset)),
        }

    retrieval_miss_count = sum(
        1
        for row in rows
        if _normalized_review_value(row.retrieval_error_type) == "retrieval_miss"
    )
    answer_failure_count = sum(
        1 for row in rows if _normalized_review_value(row.answer_error_type) not in {"", "n/a"}
    )
    partial_answer_count = faithfulness_counts["partial"]

    return ManualReviewSummary(
        run_id=run_id,
        total_rows=len(rows),
        answer_bearing_row_count=len(answer_bearing_rows),
        adversarial_row_count=len(adversarial_rows),
        faithfulness_yes_count=faithfulness_counts["yes"],
        faithfulness_partial_count=faithfulness_counts["partial"],
        faithfulness_no_count=faithfulness_counts["no"],
        faithfulness_percentage=_safe_percentage(
            faithfulness_counts["yes"],
            len(answer_bearing_rows),
        ),
        citation_valid_yes_count=citation_counts["yes"],
        citation_valid_partial_count=citation_counts["partial"],
        citation_valid_no_count=citation_counts["no"],
        citation_validity_percentage=_safe_percentage(
            citation_counts["yes"],
            len(answer_bearing_rows),
        ),
        correct_abstention_yes_count=abstention_counts["yes"],
        correct_abstention_no_count=abstention_counts["no"],
        correct_abstention_percentage=_safe_percentage(
            abstention_counts["yes"],
            len(adversarial_rows),
        ),
        faithfulness_by_question_type=faithfulness_by_question_type,
        retrieval_miss_count=retrieval_miss_count,
        answer_failure_count=answer_failure_count,
        partial_answer_count=partial_answer_count,
    )


def write_eval_summary_json(path: Path, summary: EvalSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.to_dict(), indent=JSON_INDENT, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_eval_summary_markdown(summary: EvalSummary) -> str:
    retrieval_value = (
        f"{summary.retrieval_hit_at_k_answer_bearing:.4f}"
        if summary.retrieval_hit_at_k_answer_bearing is not None
        else "n/a"
    )
    document_value = (
        f"{summary.document_hit_at_k_answer_bearing:.4f}"
        if summary.document_hit_at_k_answer_bearing is not None
        else "n/a"
    )
    rouge_value = (
        f"{summary.average_rouge_l_f1_answer_bearing_success:.4f}"
        if summary.average_rouge_l_f1_answer_bearing_success is not None
        else "n/a"
    )
    bert_score_value = (
        f"{summary.average_bert_score_f1_answer_bearing_success:.4f}"
        if summary.average_bert_score_f1_answer_bearing_success is not None
        else "n/a"
    )

    lines = [
        f"# Eval Summary: {summary.run_id}",
        "",
        "## Core Metrics",
        f"- Total rows: {summary.total_rows}",
        f"- Answer-bearing rows: {summary.answer_bearing_row_count}",
        f"- Answer-bearing successful rows: {summary.answer_bearing_success_count}",
        f"- Retrieval hit@k (answer-bearing only): {retrieval_value}",
        f"- Document hit@k (answer-bearing only): {document_value}",
        f"- Average ROUGE-L F1 (answer-bearing successful rows only): {rouge_value}",
        f"- Average BERTScore F1 (answer-bearing successful rows only): {bert_score_value}",
        f"- Adversarial rows: {summary.adversarial_row_count}",
        f"- Correct adversarial abstentions: {summary.adversarial_correct_abstention_count}",
        f"- Retrieval misses: {summary.retrieval_miss_count}",
        f"- Failed abstentions: {summary.failed_abstention_count}",
        "",
        "## Status Counts",
    ]
    for status, count in sorted(summary.status_counts.items()):
        lines.append(f"- {status}: {count}")

    lines.extend(["", "## Question Type Counts"])
    for question_type, count in sorted(summary.question_type_counts.items()):
        lines.append(f"- {question_type}: {count}")

    lines.extend(["", "## Top Failure Examples"])
    if summary.top_failure_examples:
        for item in summary.top_failure_examples:
            lines.append(
                f"- {item['question_id']} ({item['question_type']}): {item['reason']}"
            )
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def write_eval_summary_markdown(path: Path, summary: EvalSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_eval_summary_markdown(summary), encoding="utf-8")


def write_manual_review_summary_json(path: Path, summary: ManualReviewSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.to_dict(), indent=JSON_INDENT, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_manual_review_summary_markdown(summary: ManualReviewSummary) -> str:
    faithfulness_value = (
        f"{summary.faithfulness_percentage:.4f}"
        if summary.faithfulness_percentage is not None
        else "n/a"
    )
    citation_value = (
        f"{summary.citation_validity_percentage:.4f}"
        if summary.citation_validity_percentage is not None
        else "n/a"
    )
    abstention_value = (
        f"{summary.correct_abstention_percentage:.4f}"
        if summary.correct_abstention_percentage is not None
        else "n/a"
    )

    lines = [
        f"# Manual Review Summary: {summary.run_id}",
        "",
        "## Core Metrics",
        f"- Total rows: {summary.total_rows}",
        f"- Answer-bearing rows: {summary.answer_bearing_row_count}",
        f"- Adversarial rows: {summary.adversarial_row_count}",
        f"- Faithfulness percentage: {faithfulness_value}",
        f"- Citation validity percentage: {citation_value}",
        f"- Correct abstention percentage: {abstention_value}",
        f"- Retrieval misses: {summary.retrieval_miss_count}",
        f"- Answer failures: {summary.answer_failure_count}",
        f"- Partial answers: {summary.partial_answer_count}",
        "",
        "## Count Breakdown",
        f"- Faithful yes / partial / no: {summary.faithfulness_yes_count} / {summary.faithfulness_partial_count} / {summary.faithfulness_no_count}",
        f"- Citation valid yes / partial / no: {summary.citation_valid_yes_count} / {summary.citation_valid_partial_count} / {summary.citation_valid_no_count}",
        f"- Correct abstention yes / no: {summary.correct_abstention_yes_count} / {summary.correct_abstention_no_count}",
        "",
        "## Faithfulness By Question Type",
    ]
    for question_type, payload in sorted(summary.faithfulness_by_question_type.items()):
        percentage = payload["faithfulness_percentage"]
        percentage_text = f"{percentage:.4f}" if percentage is not None else "n/a"
        lines.append(
            f"- {question_type}: {percentage_text} "
            f"(yes={payload['yes_count']}, partial={payload['partial_count']}, no={payload['no_count']}, total={payload['row_count']})"
        )
    return "\n".join(lines) + "\n"


def write_manual_review_summary_markdown(path: Path, summary: ManualReviewSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_manual_review_summary_markdown(summary), encoding="utf-8")


def load_eval_examples_for_reporting(eval_path: Path) -> tuple[EvalExample, ...]:
    return load_eval_examples(eval_path.resolve())
