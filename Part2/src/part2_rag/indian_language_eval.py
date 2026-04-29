from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from part2_rag.config import (
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_MULTILINGUAL_BERT_SCORE_MODEL,
    get_default_hindi_eval_path,
    get_default_hindi_manual_review_template_path,
    get_default_indian_language_runs_dir,
)
from part2_rag.eval_runner import (
    BertScoreUnavailableError,
    BertScoreValues,
    build_bert_score_scorer,
)
from part2_rag.llm_providers import (
    BaseLLMProvider,
    ProviderGenerationMode,
    ProviderConfigurationError,
    ProviderInvocationError,
    get_provider,
    get_provider_configuration_status,
)


SUPPORTED_PROVIDER_SELECTIONS = frozenset({"groq", "gemini", "both"})
JSON_INDENT = 2
DEFAULT_MANUAL_REVIEW_MAX_ROWS = 8
HINDI_TRANSLATION_GENERATION_MODE: ProviderGenerationMode = "text"


class IndianLanguageEvalError(RuntimeError):
    """Raised when the Hindi evaluation inputs, metrics, or artifacts are invalid."""


@dataclass(frozen=True)
class HindiEvalExample:
    example_id: str
    source_text: str
    reference_text: str
    task_type: str
    source_kind: str
    difficulty_tags: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HindiEvalRunRow:
    run_id: str
    example_id: str
    provider: str
    model: str | None
    task_type: str
    source_kind: str
    difficulty_tags: list[str]
    source_text: str
    reference_text: str
    prompt_text: str
    output_text: str
    chrf: float | None
    bert_score_precision: float | None
    bert_score_recall: float | None
    bert_score_f1: float | None
    bert_score_error: str | None
    latency_seconds: float
    status: str
    error: str | None
    artifact_path: str
    raw_response_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HindiEvalRunResult:
    run_id: str
    run_dir: str
    results_jsonl_path: str
    results_csv_path: str
    row_count: int
    rows: tuple[HindiEvalRunRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "results_jsonl_path": self.results_jsonl_path,
            "results_csv_path": self.results_csv_path,
            "row_count": self.row_count,
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class HindiManualReviewRow:
    run_id: str
    example_id: str
    provider: str
    model: str | None
    source_text: str
    reference_text: str
    output_text: str
    difficulty_tags: list[str]
    fluency: str
    adequacy: str
    review_notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HindiEvalSummary:
    run_id: str
    run_dir: str
    total_rows: int
    provider_metrics: dict[str, dict[str, Any]]
    edge_case_metrics: dict[str, dict[str, dict[str, Any]]]
    bert_score_status: dict[str, Any]
    top_low_chrf_examples: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
                raise IndianLanguageEvalError(
                    f"Hindi eval file line {line_number} is not valid JSON."
                ) from exc
            if not isinstance(payload, dict):
                raise IndianLanguageEvalError(
                    f"Hindi eval file line {line_number} must contain a JSON object."
                )
            rows.append(payload)
    return rows


def load_hindi_eval_examples(path: Path | None = None) -> tuple[HindiEvalExample, ...]:
    eval_path = (path or get_default_hindi_eval_path()).resolve()
    rows = _read_jsonl_rows(eval_path)
    examples: list[HindiEvalExample] = []
    seen_ids: set[str] = set()
    for payload in rows:
        example_id = str(payload["example_id"]).strip()
        if not example_id:
            raise IndianLanguageEvalError("Hindi eval examples must have a non-empty example_id.")
        if example_id in seen_ids:
            raise IndianLanguageEvalError(f"Duplicate Hindi eval example_id: {example_id}")
        seen_ids.add(example_id)
        difficulty_tags = tuple(
            sorted(
                {
                    str(tag).strip()
                    for tag in payload.get("difficulty_tags", [])
                    if str(tag).strip()
                }
            )
        )
        examples.append(
            HindiEvalExample(
                example_id=example_id,
                source_text=str(payload["source_text"]).strip(),
                reference_text=str(payload["reference_text"]).strip(),
                task_type=str(payload["task_type"]).strip(),
                source_kind=str(payload["source_kind"]).strip(),
                difficulty_tags=difficulty_tags,
                notes=str(payload.get("notes", "")).strip(),
            )
        )
    if len(examples) < 20:
        raise IndianLanguageEvalError(
            f"Hindi eval set must contain at least 20 examples; found {len(examples)}."
        )
    return tuple(examples)


def filter_hindi_eval_examples(
    examples: Sequence[HindiEvalExample],
    *,
    max_examples: int | None = None,
    example_ids: Sequence[str] | None = None,
) -> tuple[HindiEvalExample, ...]:
    normalized_filter = {
        example_id.strip() for example_id in (example_ids or ()) if example_id.strip()
    }
    filtered = [
        example
        for example in examples
        if not normalized_filter or example.example_id in normalized_filter
    ]
    if max_examples is not None:
        if max_examples <= 0:
            raise IndianLanguageEvalError("max_examples must be a positive integer.")
        filtered = filtered[:max_examples]
    return tuple(filtered)


def get_provider_names(provider_selection: str) -> tuple[str, ...]:
    normalized = provider_selection.strip().lower()
    if normalized not in SUPPORTED_PROVIDER_SELECTIONS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDER_SELECTIONS))
        raise IndianLanguageEvalError(
            f"Unsupported provider selection {provider_selection!r}. Supported values: {supported}"
        )
    if normalized == "both":
        return ("groq", "gemini")
    return (normalized,)


def normalize_translation_text(text: str) -> str:
    return " ".join(text.strip().split())


def _extract_char_ngrams(text: str, order: int) -> Counter[str]:
    normalized = normalize_translation_text(text)
    if len(normalized) < order or order <= 0:
        return Counter()
    return Counter(normalized[index : index + order] for index in range(len(normalized) - order + 1))


def compute_chrf(
    reference_text: str,
    candidate_text: str,
    *,
    char_order: int = 6,
    beta: float = 2.0,
) -> float:
    reference = normalize_translation_text(reference_text)
    candidate = normalize_translation_text(candidate_text)
    if not reference or not candidate:
        return 0.0

    beta_squared = beta * beta
    precisions: list[float] = []
    recalls: list[float] = []
    for order in range(1, char_order + 1):
        reference_ngrams = _extract_char_ngrams(reference, order)
        candidate_ngrams = _extract_char_ngrams(candidate, order)
        if not reference_ngrams or not candidate_ngrams:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        overlap = sum((reference_ngrams & candidate_ngrams).values())
        precision = overlap / sum(candidate_ngrams.values())
        recall = overlap / sum(reference_ngrams.values())
        precisions.append(precision)
        recalls.append(recall)

    average_precision = sum(precisions) / len(precisions)
    average_recall = sum(recalls) / len(recalls)
    if average_precision == 0.0 or average_recall == 0.0:
        return 0.0
    return (
        (1 + beta_squared)
        * average_precision
        * average_recall
        / ((beta_squared * average_precision) + average_recall)
    )


def build_multilingual_bert_score_scorer(
    *,
    model_type: str = DEFAULT_MULTILINGUAL_BERT_SCORE_MODEL,
) -> Callable[[str, str], BertScoreValues]:
    return build_bert_score_scorer(
        model_type=model_type,
        num_layers=None,
        lang="hi",
        rescale_with_baseline=False,
    )


def build_hindi_translation_prompt(source_text: str) -> str:
    return (
        "Translate the following English or code-mixed r/fitness-style text into natural Hindi.\n"
        "Preserve meaning, tone, uncertainty, and advice strength.\n"
        "Keep common fitness abbreviations and named programs in Latin script when that is the natural choice, "
        "including terms like PPL, TDEE, PR, 5x5, recomp, deload, macros, and StrongLifts.\n"
        "Do not add explanations, bullet points, or extra advice.\n"
        "Output only the Hindi translation.\n\n"
        f"Source text:\n{source_text}"
    )


def _build_provider_response(
    *,
    provider: BaseLLMProvider,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    generation_mode: ProviderGenerationMode = HINDI_TRANSLATION_GENERATION_MODE,
) -> tuple[str, dict[str, Any]]:
    response = provider.generate(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        generation_mode=generation_mode,
    )
    return response.text.strip(), response.raw_response


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=JSON_INDENT, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_results_jsonl(path: Path, rows: Sequence[HindiEvalRunRow]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")


def _write_results_csv(path: Path, rows: Sequence[HindiEvalRunRow]) -> None:
    fieldnames = list(HindiEvalRunRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.to_dict().items()
                }
            )


def select_manual_review_rows(
    *,
    rows: Sequence[HindiEvalRunRow],
    max_rows: int = DEFAULT_MANUAL_REVIEW_MAX_ROWS,
) -> tuple[HindiManualReviewRow, ...]:
    if max_rows <= 0:
        raise IndianLanguageEvalError("manual review selection requires max_rows > 0.")

    successful_rows = [row for row in rows if row.status == "success" and row.output_text.strip()]
    if not successful_rows:
        return ()

    priority_tags = (
        "code_mixed",
        "reddit_slang",
        "fitness_slang",
        "abbreviation",
        "named_entity",
    )
    scored_rows: list[tuple[tuple[int, float, str], HindiEvalRunRow]] = []
    for row in successful_rows:
        tag_bonus = sum(1 for tag in priority_tags if tag in row.difficulty_tags)
        chrf_value = row.chrf if row.chrf is not None else math.inf
        scored_rows.append(((-tag_bonus, chrf_value, row.example_id), row))
    scored_rows.sort(key=lambda item: item[0])

    selected: list[HindiManualReviewRow] = []
    used_pairs: set[tuple[str, str]] = set()
    for _, row in scored_rows:
        pair = (row.provider, row.example_id)
        if pair in used_pairs:
            continue
        selected.append(
            HindiManualReviewRow(
                run_id=row.run_id,
                example_id=row.example_id,
                provider=row.provider,
                model=row.model,
                source_text=row.source_text,
                reference_text=row.reference_text,
                output_text=row.output_text,
                difficulty_tags=list(row.difficulty_tags),
                fluency="",
                adequacy="",
                review_notes="",
            )
        )
        used_pairs.add(pair)
        if len(selected) >= max_rows:
            break
    return tuple(selected)


def write_manual_review_csv(path: Path, rows: Sequence[HindiManualReviewRow]) -> None:
    fieldnames = list(HindiManualReviewRow.__dataclass_fields__)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = row.to_dict()
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, ensure_ascii=False)
                    if isinstance(value, list)
                    else value
                    for key, value in payload.items()
                }
            )


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def summarize_hindi_eval_run(
    *,
    run_id: str,
    run_dir: Path,
    rows: Sequence[HindiEvalRunRow],
    bert_score_status: dict[str, Any],
) -> HindiEvalSummary:
    provider_buckets: dict[str, list[HindiEvalRunRow]] = defaultdict(list)
    edge_case_buckets: dict[str, dict[str, list[HindiEvalRunRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        provider_buckets[row.provider].append(row)
        for tag in row.difficulty_tags:
            edge_case_buckets[row.provider][tag].append(row)

    provider_metrics: dict[str, dict[str, Any]] = {}
    for provider, provider_rows in sorted(provider_buckets.items()):
        success_rows = [row for row in provider_rows if row.status == "success"]
        provider_metrics[provider] = {
            "row_count": len(provider_rows),
            "success_count": len(success_rows),
            "average_chrf": _mean(row.chrf for row in success_rows if row.chrf is not None),
            "average_bert_score_f1": _mean(
                row.bert_score_f1 for row in success_rows if row.bert_score_f1 is not None
            ),
            "bert_score_populated_rows": sum(
                1 for row in success_rows if row.bert_score_f1 is not None
            ),
            "bert_score_error_rows": sum(
                1 for row in success_rows if row.bert_score_error is not None
            ),
        }

    edge_case_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for provider, tag_map in sorted(edge_case_buckets.items()):
        edge_case_metrics[provider] = {}
        for tag, tag_rows in sorted(tag_map.items()):
            success_rows = [row for row in tag_rows if row.status == "success"]
            edge_case_metrics[provider][tag] = {
                "row_count": len(tag_rows),
                "average_chrf": _mean(row.chrf for row in success_rows if row.chrf is not None),
                "average_bert_score_f1": _mean(
                    row.bert_score_f1 for row in success_rows if row.bert_score_f1 is not None
                ),
            }

    low_rows = sorted(
        [row for row in rows if row.status == "success" and row.chrf is not None],
        key=lambda row: (row.chrf if row.chrf is not None else math.inf, row.provider, row.example_id),
    )[:6]
    top_low_chrf_examples = [
        {
            "provider": row.provider,
            "example_id": row.example_id,
            "chrf": row.chrf,
            "difficulty_tags": list(row.difficulty_tags),
        }
        for row in low_rows
    ]

    return HindiEvalSummary(
        run_id=run_id,
        run_dir=str(run_dir),
        total_rows=len(rows),
        provider_metrics=provider_metrics,
        edge_case_metrics=edge_case_metrics,
        bert_score_status=bert_score_status,
        top_low_chrf_examples=top_low_chrf_examples,
    )


def render_hindi_eval_summary_markdown(summary: HindiEvalSummary) -> str:
    lines = [
        f"# Hindi Eval Summary: {summary.run_id}",
        "",
        "## Provider Metrics",
        "| Provider | Rows | Success | Average chrF | Average BERTScore F1 | BERTScore rows | BERTScore errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for provider, payload in sorted(summary.provider_metrics.items()):
        average_chrf = (
            f"{payload['average_chrf']:.4f}" if payload["average_chrf"] is not None else "n/a"
        )
        average_bert = (
            f"{payload['average_bert_score_f1']:.4f}"
            if payload["average_bert_score_f1"] is not None
            else "n/a"
        )
        lines.append(
            f"| {provider} | {payload['row_count']} | {payload['success_count']} | "
            f"{average_chrf} | {average_bert} | "
            f"{payload['bert_score_populated_rows']} | {payload['bert_score_error_rows']} |"
        )

    lines.extend(
        [
            "",
            "## BERTScore Status",
            f"- Enabled: {summary.bert_score_status.get('enabled')}",
            f"- Model: {summary.bert_score_status.get('model')}",
            f"- Message: {summary.bert_score_status.get('message')}",
            "",
            "## Edge-Case Metrics",
        ]
    )

    if summary.edge_case_metrics:
        for provider, tag_map in sorted(summary.edge_case_metrics.items()):
            lines.append(f"### {provider}")
            for tag, payload in sorted(tag_map.items()):
                average_chrf = (
                    f"{payload['average_chrf']:.4f}" if payload["average_chrf"] is not None else "n/a"
                )
                average_bert = (
                    f"{payload['average_bert_score_f1']:.4f}"
                    if payload["average_bert_score_f1"] is not None
                    else "n/a"
                )
                lines.append(
                    f"- {tag}: rows={payload['row_count']}, avg chrF={average_chrf}, "
                    f"avg BERTScore F1={average_bert}"
                )
    else:
        lines.append("- No edge-case metrics available.")

    lines.extend(["", "## Lowest chrF Examples"])
    if summary.top_low_chrf_examples:
        for item in summary.top_low_chrf_examples:
            chrf_text = f"{item['chrf']:.4f}" if item["chrf"] is not None else "n/a"
            tags_text = ", ".join(item["difficulty_tags"]) if item["difficulty_tags"] else "none"
            lines.append(
                f"- {item['provider']} / {item['example_id']}: chrF={chrf_text}; tags={tags_text}"
            )
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def write_hindi_eval_summary_json(path: Path, summary: HindiEvalSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.to_dict(), indent=JSON_INDENT, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_hindi_eval_summary_markdown(path: Path, summary: HindiEvalSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_hindi_eval_summary_markdown(summary), encoding="utf-8")


def write_manual_review_template(
    *,
    examples: Sequence[HindiEvalExample],
    output_path: Path | None = None,
    max_rows: int = DEFAULT_MANUAL_REVIEW_MAX_ROWS,
) -> Path:
    target_path = (output_path or get_default_hindi_manual_review_template_path()).resolve()
    ranked_examples = sorted(
        examples,
        key=lambda example: (
            -sum(
                1
                for tag in ("code_mixed", "reddit_slang", "fitness_slang", "abbreviation", "named_entity")
                if tag in example.difficulty_tags
            ),
            example.example_id,
        ),
    )
    rows = [
        {
            "example_id": example.example_id,
            "source_text": example.source_text,
            "reference_text": example.reference_text,
            "difficulty_tags": json.dumps(list(example.difficulty_tags), ensure_ascii=False),
            "provider": "",
            "model": "",
            "output_text": "",
            "fluency": "",
            "adequacy": "",
            "review_notes": "",
        }
        for example in ranked_examples[:max_rows]
    ]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "example_id",
        "source_text",
        "reference_text",
        "difficulty_tags",
        "provider",
        "model",
        "output_text",
        "fluency",
        "adequacy",
        "review_notes",
    ]
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return target_path


def run_hindi_eval(
    *,
    eval_path: Path | None = None,
    provider_selection: str = "groq",
    output_root_dir: Path | None = None,
    max_examples: int | None = None,
    example_ids: Sequence[str] | None = None,
    save_raw_response: bool = False,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    provider_factory: Callable[[str], BaseLLMProvider] = get_provider,
    provider_status_fn: Callable[[str], tuple[bool, str]] = get_provider_configuration_status,
    bert_score_scorer: Callable[[str, str], BertScoreValues] | None | bool = True,
) -> HindiEvalRunResult:
    examples = filter_hindi_eval_examples(
        load_hindi_eval_examples(eval_path),
        max_examples=max_examples,
        example_ids=example_ids,
    )
    if not examples:
        raise IndianLanguageEvalError("No Hindi eval examples matched the requested filters.")

    provider_names = get_provider_names(provider_selection)
    output_root = (output_root_dir or get_default_indian_language_runs_dir()).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = _utc_run_id()
    run_dir = output_root / run_id
    artifacts_dir = run_dir / "artifacts"
    raw_dir = run_dir / "raw_responses"
    artifacts_dir.mkdir(parents=True, exist_ok=False)
    if save_raw_response:
        raw_dir.mkdir(parents=True, exist_ok=True)

    provider_status: dict[str, tuple[bool, str]] = {
        provider_name: provider_status_fn(provider_name) for provider_name in provider_names
    }
    if bert_score_scorer is True:
        try:
            active_bert_score_scorer = build_multilingual_bert_score_scorer()
            bert_score_status = {
                "enabled": True,
                "model": DEFAULT_MULTILINGUAL_BERT_SCORE_MODEL,
                "message": "Multilingual BERTScore scorer initialized.",
            }
        except BertScoreUnavailableError as exc:
            active_bert_score_scorer = None
            bert_score_status = {
                "enabled": False,
                "model": DEFAULT_MULTILINGUAL_BERT_SCORE_MODEL,
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

    rows: list[HindiEvalRunRow] = []
    for example in examples:
        prompt_text = build_hindi_translation_prompt(example.source_text)
        for provider_name in provider_names:
            artifact_path = artifacts_dir / f"{example.example_id}__{provider_name}.json"
            raw_response_path: Path | None = None
            provider_ok, provider_message = provider_status.get(provider_name, (False, "unknown"))

            if not provider_ok:
                row = HindiEvalRunRow(
                    run_id=run_id,
                    example_id=example.example_id,
                    provider=provider_name,
                    model=None,
                    task_type=example.task_type,
                    source_kind=example.source_kind,
                    difficulty_tags=list(example.difficulty_tags),
                    source_text=example.source_text,
                    reference_text=example.reference_text,
                    prompt_text=prompt_text,
                    output_text="",
                    chrf=None,
                    bert_score_precision=None,
                    bert_score_recall=None,
                    bert_score_f1=None,
                    bert_score_error=None,
                    latency_seconds=0.0,
                    status="provider_not_configured",
                    error=provider_message,
                    artifact_path=str(artifact_path),
                    raw_response_path=None,
                )
                _write_json(
                    artifact_path,
                    {
                        "run_id": run_id,
                        "example": example.to_dict(),
                        "provider": provider_name,
                        "status": row.status,
                        "error": row.error,
                    },
                )
                rows.append(row)
                continue

            example_start = time.perf_counter()
            chosen_model: str | None = None
            output_text = ""
            raw_payload: dict[str, Any] | None = None
            chrf_value: float | None = None
            bert_values: BertScoreValues | None = None
            bert_error: str | None = None
            status = "success"
            error: str | None = None

            try:
                provider = provider_factory(provider_name)
                chosen_model = provider.default_model()
                output_text, raw_payload = _build_provider_response(
                    provider=provider,
                    prompt=prompt_text,
                    model=chosen_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    generation_mode=HINDI_TRANSLATION_GENERATION_MODE,
                )
                output_text = normalize_translation_text(output_text)
                chrf_value = compute_chrf(example.reference_text, output_text)
                if active_bert_score_scorer is not None and output_text:
                    try:
                        bert_values = active_bert_score_scorer(example.reference_text, output_text)
                    except BertScoreUnavailableError as exc:
                        bert_error = str(exc)
            except ProviderConfigurationError as exc:
                status = "provider_not_configured"
                error = str(exc)
            except ProviderInvocationError as exc:
                status = "provider_error"
                error = str(exc)

            latency_seconds = time.perf_counter() - example_start
            if save_raw_response and raw_payload is not None:
                raw_response_path = raw_dir / f"{example.example_id}__{provider_name}.json"
                _write_json(
                    raw_response_path,
                    {
                        "provider": provider_name,
                        "model": chosen_model,
                        "raw_response": raw_payload,
                    },
                )

            row = HindiEvalRunRow(
                run_id=run_id,
                example_id=example.example_id,
                provider=provider_name,
                model=chosen_model,
                task_type=example.task_type,
                source_kind=example.source_kind,
                difficulty_tags=list(example.difficulty_tags),
                source_text=example.source_text,
                reference_text=example.reference_text,
                prompt_text=prompt_text,
                output_text=output_text,
                chrf=chrf_value,
                bert_score_precision=bert_values.precision if bert_values is not None else None,
                bert_score_recall=bert_values.recall if bert_values is not None else None,
                bert_score_f1=bert_values.f1 if bert_values is not None else None,
                bert_score_error=bert_error,
                latency_seconds=latency_seconds,
                status=status,
                error=error,
                artifact_path=str(artifact_path),
                raw_response_path=str(raw_response_path) if raw_response_path is not None else None,
            )
            _write_json(
                artifact_path,
                {
                    "run_id": run_id,
                    "example": example.to_dict(),
                    "provider": provider_name,
                    "model": chosen_model,
                    "status": status,
                    "error": error,
                    "output_text": output_text,
                    "chrf": chrf_value,
                    "bert_score": bert_values.to_dict() if bert_values is not None else None,
                    "bert_score_error": bert_error,
                    "latency_seconds": latency_seconds,
                    "raw_response_path": str(raw_response_path) if raw_response_path is not None else None,
                },
            )
            rows.append(row)

    results_jsonl_path = run_dir / "results.jsonl"
    results_csv_path = run_dir / "results.csv"
    _write_results_jsonl(results_jsonl_path, rows)
    _write_results_csv(results_csv_path, rows)

    summary = summarize_hindi_eval_run(
        run_id=run_id,
        run_dir=run_dir,
        rows=rows,
        bert_score_status=bert_score_status,
    )
    write_hindi_eval_summary_json(run_dir / "summary.json", summary)
    write_hindi_eval_summary_markdown(run_dir / "summary.md", summary)

    manual_review_rows = select_manual_review_rows(rows=rows)
    write_manual_review_csv(run_dir / "manual_review.csv", manual_review_rows)

    run_manifest = {
        "run_id": run_id,
        "eval_path": str((eval_path or get_default_hindi_eval_path()).resolve()),
        "provider_selection": provider_selection,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "generation_mode": HINDI_TRANSLATION_GENERATION_MODE,
        "row_count": len(rows),
        "provider_status": {
            provider_name: {"configured": configured, "message": message}
            for provider_name, (configured, message) in provider_status.items()
        },
        "bert_score": bert_score_status,
        "results_jsonl_path": str(results_jsonl_path),
        "results_csv_path": str(results_csv_path),
        "summary_json_path": str((run_dir / "summary.json").resolve()),
        "summary_markdown_path": str((run_dir / "summary.md").resolve()),
        "manual_review_path": str((run_dir / "manual_review.csv").resolve()),
    }
    _write_json(run_dir / "run_manifest.json", run_manifest)

    return HindiEvalRunResult(
        run_id=run_id,
        run_dir=str(run_dir),
        results_jsonl_path=str(results_jsonl_path),
        results_csv_path=str(results_csv_path),
        row_count=len(rows),
        rows=tuple(rows),
    )


__all__ = [
    "DEFAULT_MANUAL_REVIEW_MAX_ROWS",
    "HindiEvalExample",
    "HindiEvalRunResult",
    "HindiEvalRunRow",
    "HindiEvalSummary",
    "HindiManualReviewRow",
    "HINDI_TRANSLATION_GENERATION_MODE",
    "IndianLanguageEvalError",
    "build_hindi_translation_prompt",
    "build_multilingual_bert_score_scorer",
    "compute_chrf",
    "filter_hindi_eval_examples",
    "get_provider_names",
    "load_hindi_eval_examples",
    "normalize_translation_text",
    "render_hindi_eval_summary_markdown",
    "run_hindi_eval",
    "select_manual_review_rows",
    "summarize_hindi_eval_run",
    "write_hindi_eval_summary_json",
    "write_hindi_eval_summary_markdown",
    "write_manual_review_csv",
    "write_manual_review_template",
]
