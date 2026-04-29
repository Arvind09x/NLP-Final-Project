from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from part2_rag.config import (
    ADVERSARIAL_NO_ANSWER_QUERY_TYPE,
    FACTUAL_QUERY_TYPE,
    OPINION_SUMMARY_QUERY_TYPE,
    get_default_chunk_artifact_path,
    get_default_eval_path,
)
from part2_rag.retrieval import RetrievalError, load_frozen_chunk_catalog


REQUIRED_FIELDS: tuple[str, ...] = (
    "question_id",
    "question",
    "question_type",
    "gold_answer",
    "expected_has_answer",
    "supporting_document_ids",
    "supporting_chunk_ids",
    "notes",
)
VALID_QUERY_TYPES: frozenset[str] = frozenset(
    {
        FACTUAL_QUERY_TYPE,
        OPINION_SUMMARY_QUERY_TYPE,
        ADVERSARIAL_NO_ANSWER_QUERY_TYPE,
    }
)
MINIMUM_EXAMPLE_COUNT = 15
MINIMUM_ADVERSARIAL_COUNT = 2


class EvalValidationError(RuntimeError):
    """Raised when the frozen eval set is malformed or unsupported."""


@dataclass(frozen=True)
class EvalValidationReport:
    eval_path: str
    chunk_artifact_path: str
    example_count: int
    counts_by_question_type: dict[str, int]
    adversarial_no_answer_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_non_empty_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise EvalValidationError(f"Field `{field_name}` must be a non-empty string.")
    return value.strip()


def _require_string_list(payload: dict[str, Any], field_name: str) -> list[str]:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise EvalValidationError(f"Field `{field_name}` must be a list of strings.")

    normalized: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str) or not item.strip():
            raise EvalValidationError(
                f"Field `{field_name}` item {index} must be a non-empty string."
            )
        normalized.append(item.strip())
    return normalized


def _load_eval_rows(eval_path: Path) -> list[dict[str, Any]]:
    if not eval_path.exists():
        raise EvalValidationError(f"Eval file does not exist: {eval_path}")

    rows: list[dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload_text = line.strip()
            if not payload_text:
                continue
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                raise EvalValidationError(
                    f"Eval file line {line_number} is not valid JSON."
                ) from exc
            if not isinstance(payload, dict):
                raise EvalValidationError(
                    f"Eval file line {line_number} must contain a JSON object."
                )
            rows.append(payload)
    return rows


def validate_eval_set(
    *,
    eval_path: Path,
    chunk_artifact_path: Path,
) -> EvalValidationReport:
    rows = _load_eval_rows(eval_path)
    if len(rows) < MINIMUM_EXAMPLE_COUNT:
        raise EvalValidationError(
            f"Eval set must contain at least {MINIMUM_EXAMPLE_COUNT} examples."
        )

    try:
        catalog = load_frozen_chunk_catalog(str(chunk_artifact_path.resolve()))
    except RetrievalError as exc:
        raise EvalValidationError(f"Unable to load frozen chunk artifact: {exc}") from exc

    seen_question_ids: set[str] = set()
    counts_by_question_type = {query_type: 0 for query_type in sorted(VALID_QUERY_TYPES)}
    adversarial_no_answer_count = 0

    for index, payload in enumerate(rows, start=1):
        missing_fields = [field for field in REQUIRED_FIELDS if field not in payload]
        if missing_fields:
            joined = ", ".join(missing_fields)
            raise EvalValidationError(
                f"Eval example {index} is missing required field(s): {joined}."
            )

        question_id = _require_non_empty_string(payload, "question_id")
        if question_id in seen_question_ids:
            raise EvalValidationError(f"Duplicate question_id detected: {question_id}")
        seen_question_ids.add(question_id)

        _require_non_empty_string(payload, "question")
        question_type = _require_non_empty_string(payload, "question_type")
        if question_type not in VALID_QUERY_TYPES:
            supported = ", ".join(sorted(VALID_QUERY_TYPES))
            raise EvalValidationError(
                f"question_id={question_id} uses unsupported question_type={question_type!r}. "
                f"Supported values: {supported}"
            )
        counts_by_question_type[question_type] += 1

        _require_non_empty_string(payload, "gold_answer")

        expected_has_answer = payload.get("expected_has_answer")
        if not isinstance(expected_has_answer, bool):
            raise EvalValidationError(
                f"question_id={question_id} field `expected_has_answer` must be a boolean."
            )

        supporting_document_ids = _require_string_list(payload, "supporting_document_ids")
        supporting_chunk_ids = _require_string_list(payload, "supporting_chunk_ids")
        _require_non_empty_string(payload, "notes")

        if question_type == ADVERSARIAL_NO_ANSWER_QUERY_TYPE:
            adversarial_no_answer_count += 1
            if expected_has_answer:
                raise EvalValidationError(
                    f"question_id={question_id} must set expected_has_answer=false "
                    "for adversarial/no-answer examples."
                )

        if expected_has_answer:
            if not supporting_document_ids:
                raise EvalValidationError(
                    f"question_id={question_id} must include at least one supporting document."
                )
            if not supporting_chunk_ids:
                raise EvalValidationError(
                    f"question_id={question_id} must include at least one supporting chunk."
                )

        for document_id in supporting_document_ids:
            if document_id not in catalog.document_ids:
                raise EvalValidationError(
                    f"question_id={question_id} references unknown supporting document_id="
                    f"{document_id!r}."
                )

        referenced_document_ids: set[str] = set()
        for chunk_id in supporting_chunk_ids:
            chunk = catalog.chunk_by_id.get(chunk_id)
            if chunk is None:
                raise EvalValidationError(
                    f"question_id={question_id} references unknown supporting chunk_id="
                    f"{chunk_id!r}."
                )
            referenced_document_ids.add(chunk.document_id)

        if supporting_document_ids and not referenced_document_ids.issubset(
            set(supporting_document_ids)
        ):
            raise EvalValidationError(
                f"question_id={question_id} has supporting chunks whose document_ids are not "
                "listed in supporting_document_ids."
            )

    if adversarial_no_answer_count < MINIMUM_ADVERSARIAL_COUNT:
        raise EvalValidationError(
            "Eval set must contain at least "
            f"{MINIMUM_ADVERSARIAL_COUNT} adversarial/no-answer examples."
        )

    return EvalValidationReport(
        eval_path=str(eval_path),
        chunk_artifact_path=str(chunk_artifact_path),
        example_count=len(rows),
        counts_by_question_type=counts_by_question_type,
        adversarial_no_answer_count=adversarial_no_answer_count,
    )


def print_summary(report: EvalValidationReport) -> None:
    print("Part 2 eval-set validation complete")
    print(f"Eval path: {report.eval_path}")
    print(f"Frozen chunk artifact: {report.chunk_artifact_path}")
    print(f"Examples: {report.example_count}")
    print("Counts by question type:")
    for question_type, count in report.counts_by_question_type.items():
        print(f"  - {question_type}: {count}")
    print(f"Adversarial/no-answer examples: {report.adversarial_no_answer_count}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the frozen Part 2 RAG evaluation set."
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=get_default_eval_path(),
        help="Path to the frozen eval JSONL file.",
    )
    parser.add_argument(
        "--chunk-artifact-path",
        type=Path,
        default=get_default_chunk_artifact_path(),
        help="Path to the frozen chunk JSONL artifact used for support checks.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        report = validate_eval_set(
            eval_path=args.eval_path.resolve(),
            chunk_artifact_path=args.chunk_artifact_path.resolve(),
        )
    except EvalValidationError as exc:
        parser.exit(status=1, message=f"Eval validation failed: {exc}\n")
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
