from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import unquote, urlparse

from part2_rag.config import (
    DEFAULT_CHUNK_ARTIFACT_FILENAME,
    DEFAULT_CHUNK_MANIFEST_FILENAME,
    get_default_chunk_artifact_path,
    get_default_chunk_manifest_path,
    get_default_manifest_path,
    get_paths,
)


DEFAULT_MAX_CHUNK_TOKENS = 220
DEFAULT_POST_OVERLAP_SENTENCES = 1
DEFAULT_SENTENCE_OVERLAP_TOKEN_TARGET = "20-30"
TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", re.IGNORECASE)
PLAIN_URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
COMMON_DOMAIN_SUFFIXES = {"com", "org", "net", "edu", "gov", "wiki", "io", "co", "au", "uk"}
COMMON_PATH_STOPWORDS = {
    "http",
    "https",
    "www",
    "com",
    "org",
    "net",
    "wiki",
    "html",
    "htm",
    "php",
    "asp",
    "aspx",
    "reddit",
    "comments",
    "comment",
    "user",
    "users",
    "r",
    "u",
    "utm",
    "source",
    "share",
    "medium",
    "context",
}
DOMAIN_OVERRIDES = {
    "thefitness.wiki": "fitness wiki",
    "www.thefitness.wiki": "fitness wiki",
    "strongerbyscience.com": "stronger by science",
    "www.strongerbyscience.com": "stronger by science",
}


class ChunkGenerationError(RuntimeError):
    """Raised when chunk generation cannot produce a usable artifact."""


@dataclass(frozen=True)
class SelectedWindow:
    subreddit: str
    window_start_utc: int
    window_end_utc: int | None


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    source_type: str
    source_id: str
    post_id: str | None
    parent_id: str | None
    link_id: str | None
    created_utc: int
    author_id: str | None
    title: str | None
    chunk_text: str
    chunk_index: int
    token_estimate: int
    chunk_origin: str


@dataclass(frozen=True)
class ChunkingParameters:
    max_chunk_tokens: int
    post_overlap_sentences: int
    sentence_overlap_token_target: str
    post_split_order: list[str]
    comment_split_behavior: str


@dataclass(frozen=True)
class ChunkValidationReport:
    total_chunk_count: int
    empty_chunk_count: int
    missing_metadata_count: int
    max_chunk_char_length: int
    max_chunk_token_estimate: int
    source_documents_represented: int


@dataclass(frozen=True)
class MissingDocumentDiagnostics:
    counts: dict[str, int]
    sample_ids: dict[str, list[str]]


@dataclass(frozen=True)
class ChunkBuildResult:
    chunk_artifact_path: str
    chunk_manifest_path: str
    source_corpus_manifest_path: str
    db_path: str
    selected_window: dict[str, Any]
    source_document_selection_rule: str
    generation_timestamp: str
    chunking_parameters: dict[str, Any]
    chunk_counts: dict[str, int]
    source_document_counts: dict[str, int]
    fallback_chunk_count: int
    fallback_chunk_percentage: float
    coverage_rate: float
    coverage_explanation: dict[str, int]
    missing_document_diagnostics: dict[str, Any]
    validation: dict[str, Any]
    resumed_from_existing_artifact: bool


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_path_for_comparison(path_value: str | Path) -> str:
    return str(Path(path_value).resolve())


def estimate_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def get_raw_text_candidate(document: dict[str, Any]) -> str:
    return str(document.get("raw_text") or "")


def is_deleted_or_removed_document(document: dict[str, Any]) -> bool:
    deletion_markers = {"[deleted]", "[removed]"}
    normalized_raw = normalize_text(get_raw_text_candidate(document)).lower()
    normalized_text = normalize_text(document.get("text")).lower()
    return bool(document.get("is_deleted")) or bool(document.get("is_removed")) or (
        normalized_raw in deletion_markers or normalized_text in deletion_markers
    )


def classify_missing_document(document: dict[str, Any]) -> str:
    raw_text_candidate = get_raw_text_candidate(document)
    normalized_raw_text = normalize_text(raw_text_candidate)
    normalized_text = normalize_text(document.get("text"))

    if is_deleted_or_removed_document(document):
        return "deleted_or_removed"
    if not raw_text_candidate.strip():
        return "empty_raw_text"
    if not normalized_raw_text or not normalized_text:
        return "normalized_to_empty"
    if estimate_tokens(normalized_text) <= 0:
        return "tokenization_failed"
    return "other"


def initialize_missing_document_diagnostics() -> MissingDocumentDiagnostics:
    categories = (
        "empty_raw_text",
        "deleted_or_removed",
        "normalized_to_empty",
        "tokenization_failed",
        "other",
    )
    return MissingDocumentDiagnostics(
        counts={category: 0 for category in categories},
        sample_ids={category: [] for category in categories},
    )


def record_missing_document(
    diagnostics: MissingDocumentDiagnostics,
    *,
    category: str,
    document_id: str,
    sample_limit: int = 10,
) -> MissingDocumentDiagnostics:
    counts = dict(diagnostics.counts)
    sample_ids = {key: list(value) for key, value in diagnostics.sample_ids.items()}
    counts[category] = counts.get(category, 0) + 1
    if len(sample_ids.setdefault(category, [])) < sample_limit:
        sample_ids[category].append(document_id)
    return MissingDocumentDiagnostics(counts=counts, sample_ids=sample_ids)


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(normalized) if part.strip()]
    return sentences or [normalized]


def split_paragraphs(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    paragraphs = [part.strip() for part in PARAGRAPH_SPLIT_RE.split(normalized) if part.strip()]
    return paragraphs or [normalized]


def join_units(units: Iterable[str]) -> str:
    return " ".join(unit.strip() for unit in units if unit and unit.strip()).strip()


def tokenize_slug_text(text: str) -> list[str]:
    normalized = unquote(text).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    tokens = [token for token in normalized.split() if token]
    return [token for token in tokens if not token.isdigit()]


def normalize_domain(hostname: str | None) -> str:
    if not hostname:
        return "external source"
    hostname = hostname.lower().strip(".")
    if hostname in DOMAIN_OVERRIDES:
        return DOMAIN_OVERRIDES[hostname]
    if hostname.startswith("www."):
        hostname = hostname[4:]
    labels = [label for label in hostname.split(".") if label]
    if labels and labels[0] == "the":
        labels = labels[1:]
    elif labels and labels[0].startswith("the") and len(labels[0]) > 3:
        labels[0] = labels[0][3:]
    semantic_labels = [
        label for label in labels if label and label not in COMMON_DOMAIN_SUFFIXES
    ]
    if not semantic_labels:
        semantic_labels = labels[:1] or ["external", "source"]

    words: list[str] = []
    for label in semantic_labels:
        if not label:
            continue
        split_label = label.replace("by", " by ")
        label_words = [word for word in tokenize_slug_text(split_label) if word]
        words.extend(label_words or [label])
    return " ".join(words).strip() or "external source"


def extract_semantic_text_from_url(url: str) -> str:
    parsed = urlparse(url)
    path_segments = [segment for segment in parsed.path.split("/") if segment]
    meaningful_tokens: list[str] = []
    for segment in path_segments:
        for token in tokenize_slug_text(segment):
            if token not in COMMON_PATH_STOPWORDS:
                meaningful_tokens.append(token)
    if meaningful_tokens:
        return " ".join(meaningful_tokens)
    return normalize_domain(parsed.netloc)


def extract_url_references(raw_text: str) -> list[tuple[str | None, str]]:
    references: list[tuple[str | None, str]] = []
    consumed_spans: list[tuple[int, int]] = []

    for match in MARKDOWN_LINK_RE.finditer(raw_text):
        references.append((match.group(1).strip(), match.group(2).strip()))
        consumed_spans.append(match.span())
        if len(references) >= 2:
            return references

    for match in PLAIN_URL_RE.finditer(raw_text):
        start, end = match.span()
        if any(start >= span_start and end <= span_end for span_start, span_end in consumed_spans):
            continue
        references.append((None, match.group(0).strip()))
        if len(references) >= 2:
            break
    return references


def build_url_fallback_chunk_text(raw_text: str) -> str | None:
    references = extract_url_references(raw_text)
    if not references:
        return None

    parts: list[str] = []
    for anchor_text, url in references[:2]:
        parsed = urlparse(url)
        domain_text = normalize_domain(parsed.netloc)
        semantic_text = normalize_text(anchor_text) if anchor_text else extract_semantic_text_from_url(url)
        semantic_text = semantic_text or domain_text
        parts.append(f"[source: {domain_text}] {semantic_text}")
    return " ".join(parts).strip() or None


def make_chunk_id(document_id: str, chunk_index: int, chunk_text: str) -> str:
    digest = hashlib.sha1(
        f"{document_id}:{chunk_index}:{chunk_text}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{document_id}_chunk_{chunk_index:04d}_{digest}"


def split_unit_by_words(unit: str, max_chunk_tokens: int) -> list[str]:
    words = unit.split()
    if not words:
        return []

    chunks: list[str] = []
    current_words: list[str] = []
    for word in words:
        candidate = current_words + [word]
        if current_words and estimate_tokens(" ".join(candidate)) > max_chunk_tokens:
            chunks.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words = candidate
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def enforce_text_chunks(
    units: list[str],
    *,
    max_chunk_tokens: int,
    overlap_sentences: int,
) -> list[str]:
    if not units:
        return []

    chunks: list[str] = []
    current_units: list[str] = []
    for unit in units:
        if estimate_tokens(unit) > max_chunk_tokens:
            chunk_text = join_units(current_units)
            if chunk_text:
                chunks.append(chunk_text)
            chunks.extend(split_unit_by_words(unit, max_chunk_tokens))
            current_units = []
            continue
        candidate_units = current_units + [unit]
        candidate_text = join_units(candidate_units)
        if current_units and estimate_tokens(candidate_text) > max_chunk_tokens:
            chunk_text = join_units(current_units)
            if chunk_text:
                chunks.append(chunk_text)
            overlap = current_units[-overlap_sentences:] if overlap_sentences else []
            current_units = [*overlap, unit]
            if estimate_tokens(join_units(current_units)) > max_chunk_tokens:
                current_units = [unit]
        else:
            current_units = candidate_units

    final_text = join_units(current_units)
    if final_text:
        chunks.append(final_text)

    return chunks


def split_long_text_to_units(text: str) -> list[str]:
    paragraphs = split_paragraphs(text)
    units: list[str] = []
    for paragraph in paragraphs:
        sentences = split_sentences(paragraph)
        if len(sentences) == 1 and sentences[0] == paragraph:
            units.append(paragraph)
        else:
            units.extend(sentences)
    return units or split_sentences(text)


def split_comment_text(text: str, max_chunk_tokens: int) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if estimate_tokens(normalized) <= max_chunk_tokens:
        return [normalized]
    return enforce_text_chunks(
        split_long_text_to_units(normalized),
        max_chunk_tokens=max_chunk_tokens,
        overlap_sentences=0,
    )


def split_post_text(
    text: str,
    *,
    max_chunk_tokens: int,
    overlap_sentences: int,
) -> list[str]:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_paragraphs: list[str] = []

    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)
        candidate_text = join_units(current_paragraphs + [paragraph])
        if paragraph_tokens > max_chunk_tokens:
            if current_paragraphs:
                chunks.extend(
                    enforce_text_chunks(
                        current_paragraphs,
                        max_chunk_tokens=max_chunk_tokens,
                        overlap_sentences=overlap_sentences,
                    )
                )
                current_paragraphs = []
            chunks.extend(
                enforce_text_chunks(
                    split_sentences(paragraph),
                    max_chunk_tokens=max_chunk_tokens,
                    overlap_sentences=overlap_sentences,
                )
            )
            continue
        if current_paragraphs and estimate_tokens(candidate_text) > max_chunk_tokens:
            chunks.extend(
                enforce_text_chunks(
                    current_paragraphs,
                    max_chunk_tokens=max_chunk_tokens,
                    overlap_sentences=overlap_sentences,
                )
            )
            current_paragraphs = [paragraph]
        else:
            current_paragraphs.append(paragraph)

    if current_paragraphs:
        chunks.extend(
            enforce_text_chunks(
                current_paragraphs,
                max_chunk_tokens=max_chunk_tokens,
                overlap_sentences=overlap_sentences,
            )
        )
    return chunks


def chunk_document(
    document: dict[str, Any],
    *,
    max_chunk_tokens: int,
    overlap_sentences: int,
) -> list[ChunkRecord]:
    text = normalize_text(document.get("text"))
    chunk_origin = "text"
    if not text:
        fallback_chunk_text = build_url_fallback_chunk_text(get_raw_text_candidate(document))
        if fallback_chunk_text:
            chunk_texts = [fallback_chunk_text]
            chunk_origin = "url_fallback"
        else:
            chunk_texts = []
    elif document["source_type"] == "comment":
        chunk_texts = split_comment_text(text, max_chunk_tokens=max_chunk_tokens)
    else:
        chunk_texts = split_post_text(
            text,
            max_chunk_tokens=max_chunk_tokens,
            overlap_sentences=overlap_sentences,
        )

    records: list[ChunkRecord] = []
    for chunk_index, chunk_text in enumerate(chunk_texts):
        token_estimate = estimate_tokens(chunk_text)
        records.append(
            ChunkRecord(
                chunk_id=make_chunk_id(document["document_id"], chunk_index, chunk_text),
                document_id=document["document_id"],
                source_type=document["source_type"],
                source_id=document["source_id"],
                post_id=document.get("post_id"),
                parent_id=document.get("parent_id"),
                link_id=document.get("link_id"),
                created_utc=int(document["created_utc"]),
                author_id=document.get("author_id"),
                title=document.get("title"),
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                token_estimate=token_estimate,
                chunk_origin=chunk_origin,
            )
        )
    return records


def load_selected_window(manifest_path: Path) -> SelectedWindow:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ChunkGenerationError(
            f"Corpus manifest does not exist: {manifest_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ChunkGenerationError(
            f"Corpus manifest is not valid JSON: {manifest_path}"
        ) from exc

    selected_window = manifest.get("selected_window")
    if not isinstance(selected_window, dict):
        raise ChunkGenerationError(
            f"Corpus manifest is missing selected_window: {manifest_path}"
        )
    if not selected_window.get("eligible_as_default_rag_corpus"):
        raise ChunkGenerationError(
            "Selected manifest window is not eligible as the default RAG corpus."
        )
    return SelectedWindow(
        subreddit=str(selected_window["subreddit"]),
        window_start_utc=int(selected_window["window_start_utc"]),
        window_end_utc=int(selected_window["window_end_utc"])
        if selected_window["window_end_utc"] is not None
        else None,
    )


def iter_source_documents(
    connection: sqlite3.Connection,
    selected_window: SelectedWindow,
) -> Iterator[dict[str, Any]]:
    query = """
        SELECT d.document_id,
               d.source_type,
               d.source_id,
               CASE
                   WHEN d.source_type = 'post' THEN p.post_id
                   ELSE c.post_id
               END AS post_id,
               d.parent_id,
               d.link_id,
               d.created_utc,
               d.author_id,
               CASE
                   WHEN d.source_type = 'post' THEN p.title
                   ELSE cp.title
               END AS title,
               COALESCE(
                   d.raw_text,
                   CASE
                       WHEN d.source_type = 'post' THEN p.raw_text
                       ELSE c.raw_text
                   END,
                   CASE
                       WHEN d.source_type = 'post' THEN TRIM(COALESCE(p.title, '') || '\n\n' || COALESCE(p.selftext, ''))
                       ELSE c.body
                   END,
                   ''
               ) AS raw_text,
               COALESCE(
                   d.clean_text,
                   CASE
                       WHEN d.source_type = 'post' THEN p.clean_text
                       ELSE c.clean_text
                   END,
                   d.raw_text,
                   CASE
                       WHEN d.source_type = 'post' THEN p.raw_text
                       ELSE c.raw_text
                   END,
                   CASE
                       WHEN d.source_type = 'post' THEN TRIM(COALESCE(p.title, '') || '\n\n' || COALESCE(p.selftext, ''))
                       ELSE c.body
                   END,
                   ''
               ) AS text,
               CASE
                   WHEN d.source_type = 'post' THEN COALESCE(p.is_deleted, 0)
                   ELSE COALESCE(c.is_deleted, 0)
               END AS is_deleted,
               CASE
                   WHEN d.source_type = 'post' THEN COALESCE(p.is_removed, 0)
                   ELSE COALESCE(c.is_removed, 0)
               END AS is_removed
        FROM documents AS d
        LEFT JOIN posts AS p
            ON d.source_type = 'post' AND p.post_id = d.source_id
        LEFT JOIN comments AS c
            ON d.source_type = 'comment' AND c.comment_id = d.source_id
        LEFT JOIN posts AS cp
            ON c.post_id = cp.post_id
        WHERE LOWER(d.subreddit) = LOWER(?)
          AND d.created_utc >= ?
          AND (? IS NULL OR d.created_utc < ?)
        ORDER BY d.created_utc ASC, d.document_id ASC
    """
    cursor = connection.execute(
        query,
        (
            selected_window.subreddit,
            selected_window.window_start_utc,
            selected_window.window_end_utc,
            selected_window.window_end_utc,
        ),
    )
    columns = [column[0] for column in cursor.description]
    for row in cursor:
        yield dict(zip(columns, row, strict=True))


def scan_existing_artifact(chunk_artifact_path: Path) -> tuple[set[str], int]:
    represented_document_ids: set[str] = set()
    chunk_count = 0
    if not chunk_artifact_path.exists():
        return represented_document_ids, chunk_count

    with chunk_artifact_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            document_id = payload.get("document_id")
            if document_id:
                represented_document_ids.add(str(document_id))
            chunk_count += 1
    return represented_document_ids, chunk_count


def append_chunks(
    chunk_artifact_path: Path,
    chunks: Iterable[ChunkRecord],
) -> int:
    chunk_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with chunk_artifact_path.open("a", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), sort_keys=True) + "\n")
            written += 1
    return written


def validate_chunk_artifact(chunk_artifact_path: Path) -> ChunkValidationReport:
    total_chunk_count = 0
    empty_chunk_count = 0
    missing_metadata_count = 0
    max_chunk_char_length = 0
    max_chunk_token_estimate = 0
    represented_document_ids: set[str] = set()

    required_keys = {
        "chunk_id",
        "document_id",
        "source_type",
        "source_id",
        "post_id",
        "created_utc",
        "chunk_text",
        "chunk_index",
        "token_estimate",
        "chunk_origin",
    }

    with chunk_artifact_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            total_chunk_count += 1
            chunk_text = normalize_text(payload.get("chunk_text"))
            token_estimate = int(payload.get("token_estimate") or 0)
            if not chunk_text:
                empty_chunk_count += 1
            if any(key not in payload for key in required_keys):
                missing_metadata_count += 1
            elif (
                not payload.get("chunk_id")
                or not payload.get("document_id")
                or not payload.get("source_type")
                or not payload.get("source_id")
                or payload.get("post_id") in (None, "")
            ):
                missing_metadata_count += 1
            represented_document_ids.add(str(payload.get("document_id")))
            max_chunk_char_length = max(max_chunk_char_length, len(chunk_text))
            max_chunk_token_estimate = max(max_chunk_token_estimate, token_estimate)

    return ChunkValidationReport(
        total_chunk_count=total_chunk_count,
        empty_chunk_count=empty_chunk_count,
        missing_metadata_count=missing_metadata_count,
        max_chunk_char_length=max_chunk_char_length,
        max_chunk_token_estimate=max_chunk_token_estimate,
        source_documents_represented=len(represented_document_ids),
    )


def diagnose_missing_documents(
    *,
    db_path: Path,
    source_corpus_manifest_path: Path,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    overlap_sentences: int = DEFAULT_POST_OVERLAP_SENTENCES,
) -> MissingDocumentDiagnostics:
    if not db_path.exists():
        raise ChunkGenerationError(f"Part 1 database does not exist: {db_path}")

    selected_window = load_selected_window(source_corpus_manifest_path)
    diagnostics = initialize_missing_document_diagnostics()

    with sqlite3.connect(db_path) as connection:
        for document in iter_source_documents(connection, selected_window):
            chunks = chunk_document(
                document,
                max_chunk_tokens=max_chunk_tokens,
                overlap_sentences=overlap_sentences,
            )
            if chunks:
                continue
            diagnostics = record_missing_document(
                diagnostics,
                category=classify_missing_document(document),
                document_id=document["document_id"],
            )
    return diagnostics


def build_chunk_manifest(
    *,
    chunk_artifact_path: Path,
    chunk_manifest_path: Path,
    source_corpus_manifest_path: Path,
    db_path: Path,
    selected_window: SelectedWindow,
    parameters: ChunkingParameters,
    validation_report: ChunkValidationReport,
    source_documents_seen: int,
    source_documents_chunked: int,
    fallback_chunk_count: int,
    missing_document_diagnostics: MissingDocumentDiagnostics,
    resumed_from_existing_artifact: bool,
) -> ChunkBuildResult:
    coverage_rate = (
        validation_report.source_documents_represented / source_documents_seen
        if source_documents_seen
        else 0.0
    )
    return ChunkBuildResult(
        chunk_artifact_path=str(chunk_artifact_path),
        chunk_manifest_path=str(chunk_manifest_path),
        source_corpus_manifest_path=str(source_corpus_manifest_path),
        db_path=str(db_path),
        selected_window=asdict(selected_window),
        source_document_selection_rule=(
            "Include every document in the selected default RAG corpus window using "
            "end-exclusive bounds on documents.created_utc. Do not reuse Part 1 "
            "include_in_modeling filtering for chunk generation."
        ),
        generation_timestamp=utc_now_iso(),
        chunking_parameters=asdict(parameters),
        chunk_counts={
            "total_chunks": validation_report.total_chunk_count,
            "empty_chunks": validation_report.empty_chunk_count,
        },
        source_document_counts={
            "documents_seen_in_selected_window": source_documents_seen,
            "documents_represented_in_chunk_artifact": validation_report.source_documents_represented,
            "documents_chunked_in_this_run": source_documents_chunked,
        },
        fallback_chunk_count=fallback_chunk_count,
        fallback_chunk_percentage=(
            fallback_chunk_count / validation_report.total_chunk_count
            if validation_report.total_chunk_count
            else 0.0
        ),
        coverage_rate=coverage_rate,
        coverage_explanation={
            "empty_or_deleted": (
                missing_document_diagnostics.counts.get("empty_raw_text", 0)
                + missing_document_diagnostics.counts.get("deleted_or_removed", 0)
            ),
            "normalization_loss": missing_document_diagnostics.counts.get(
                "normalized_to_empty", 0
            ),
            "other": (
                missing_document_diagnostics.counts.get("tokenization_failed", 0)
                + missing_document_diagnostics.counts.get("other", 0)
            ),
        },
        missing_document_diagnostics=asdict(missing_document_diagnostics),
        validation=asdict(validation_report),
        resumed_from_existing_artifact=resumed_from_existing_artifact,
    )


def write_chunk_manifest(chunk_manifest_path: Path, result: ChunkBuildResult) -> None:
    chunk_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_manifest_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_existing_chunk_manifest(chunk_manifest_path: Path) -> dict[str, Any] | None:
    if not chunk_manifest_path.exists():
        return None
    try:
        return json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ChunkGenerationError(
            f"Existing chunk manifest is not valid JSON: {chunk_manifest_path}"
        ) from exc


def assert_resume_compatible(
    *,
    chunk_artifact_path: Path,
    chunk_manifest_path: Path,
    source_corpus_manifest_path: Path,
    db_path: Path,
    selected_window: SelectedWindow,
    parameters: ChunkingParameters,
) -> None:
    existing_manifest = load_existing_chunk_manifest(chunk_manifest_path)
    if existing_manifest is None:
        raise ChunkGenerationError(
            "Resume requested with an existing chunk artifact but no prior chunk manifest. "
            "Re-run with --no-resume or restore the matching chunk manifest."
        )

    comparisons: list[tuple[str, Any, Any]] = [
        (
            "chunk_artifact_path",
            resolve_path_for_comparison(existing_manifest.get("chunk_artifact_path", "")),
            resolve_path_for_comparison(chunk_artifact_path),
        ),
        (
            "source_corpus_manifest_path",
            resolve_path_for_comparison(
                existing_manifest.get("source_corpus_manifest_path", "")
            ),
            resolve_path_for_comparison(source_corpus_manifest_path),
        ),
        (
            "db_path",
            resolve_path_for_comparison(existing_manifest.get("db_path", "")),
            resolve_path_for_comparison(db_path),
        ),
        (
            "selected_window",
            existing_manifest.get("selected_window"),
            asdict(selected_window),
        ),
        (
            "chunking_parameters",
            existing_manifest.get("chunking_parameters"),
            asdict(parameters),
        ),
    ]

    mismatches = [
        field_name for field_name, existing_value, current_value in comparisons
        if existing_value != current_value
    ]
    if mismatches:
        joined = ", ".join(mismatches)
        raise ChunkGenerationError(
            "Resume requested with inputs that do not match the existing chunk artifact. "
            f"Mismatched fields: {joined}. Re-run with --no-resume to rebuild a clean artifact."
        )


def build_chunks(
    *,
    db_path: Path,
    source_corpus_manifest_path: Path,
    chunk_artifact_path: Path,
    chunk_manifest_path: Path,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    overlap_sentences: int = DEFAULT_POST_OVERLAP_SENTENCES,
    resume: bool = True,
) -> ChunkBuildResult:
    if not db_path.exists():
        raise ChunkGenerationError(f"Part 1 database does not exist: {db_path}")

    selected_window = load_selected_window(source_corpus_manifest_path)
    parameters = ChunkingParameters(
        max_chunk_tokens=max_chunk_tokens,
        post_overlap_sentences=overlap_sentences,
        sentence_overlap_token_target=DEFAULT_SENTENCE_OVERLAP_TOKEN_TARGET,
        post_split_order=["paragraph", "sentence"],
        comment_split_behavior="single_chunk_unless_too_long",
    )

    resumed_from_existing_artifact = resume and chunk_artifact_path.exists()
    existing_document_ids: set[str] = set()
    if resumed_from_existing_artifact:
        assert_resume_compatible(
            chunk_artifact_path=chunk_artifact_path,
            chunk_manifest_path=chunk_manifest_path,
            source_corpus_manifest_path=source_corpus_manifest_path,
            db_path=db_path,
            selected_window=selected_window,
            parameters=parameters,
        )
        existing_document_ids, _ = scan_existing_artifact(chunk_artifact_path)
    elif chunk_artifact_path.exists():
        chunk_artifact_path.unlink()

    source_documents_seen = 0
    source_documents_chunked = 0
    missing_document_diagnostics = initialize_missing_document_diagnostics()
    chunk_origin_counts: Counter[str] = Counter()

    with sqlite3.connect(db_path) as connection:
        for document in iter_source_documents(connection, selected_window):
            source_documents_seen += 1
            if document["document_id"] in existing_document_ids:
                continue
            chunks = chunk_document(
                document,
                max_chunk_tokens=max_chunk_tokens,
                overlap_sentences=overlap_sentences,
            )
            if not chunks:
                missing_document_diagnostics = record_missing_document(
                    missing_document_diagnostics,
                    category=classify_missing_document(document),
                    document_id=document["document_id"],
                )
                continue
            append_chunks(chunk_artifact_path, chunks)
            source_documents_chunked += 1
            chunk_origin_counts.update(chunk.chunk_origin for chunk in chunks)

    validation_report = validate_chunk_artifact(chunk_artifact_path)
    if validation_report.total_chunk_count == 0:
        raise ChunkGenerationError("Chunk generation produced zero chunks.")

    result = build_chunk_manifest(
        chunk_artifact_path=chunk_artifact_path,
        chunk_manifest_path=chunk_manifest_path,
        source_corpus_manifest_path=source_corpus_manifest_path,
        db_path=db_path,
        selected_window=selected_window,
        parameters=parameters,
        validation_report=validation_report,
        source_documents_seen=source_documents_seen,
        source_documents_chunked=source_documents_chunked,
        fallback_chunk_count=chunk_origin_counts.get("url_fallback", 0),
        missing_document_diagnostics=missing_document_diagnostics,
        resumed_from_existing_artifact=resumed_from_existing_artifact,
    )
    write_chunk_manifest(chunk_manifest_path, result)
    return result


def print_summary(result: ChunkBuildResult) -> None:
    print("Part 2 chunk generation complete")
    print(f"Source corpus manifest: {result.source_corpus_manifest_path}")
    print(f"DB path: {result.db_path}")
    print(
        "Selected default window: "
        f"subreddit={result.selected_window['subreddit']}, "
        f"start={result.selected_window['window_start_utc']}, "
        f"end={result.selected_window['window_end_utc']}"
    )
    print("")
    print("Chunking parameters:")
    print(f"  - max_chunk_tokens: {result.chunking_parameters['max_chunk_tokens']}")
    print(
        "  - post_overlap_sentences: "
        f"{result.chunking_parameters['post_overlap_sentences']}"
    )
    print(
        "  - sentence_overlap_token_target: "
        f"{result.chunking_parameters['sentence_overlap_token_target']}"
    )
    print("")
    print(f"Coverage rate: {result.coverage_rate:.6f}")
    print("Coverage explanation:")
    print(
        "  - empty_or_deleted: "
        f"{result.coverage_explanation['empty_or_deleted']}"
    )
    print(
        "  - normalization_loss: "
        f"{result.coverage_explanation['normalization_loss']}"
    )
    print(f"  - other: {result.coverage_explanation['other']}")
    print("")
    print(
        "Fallback chunks: "
        f"count={result.fallback_chunk_count}, "
        f"percentage={result.fallback_chunk_percentage:.6f}"
    )
    print("")
    print("Validation:")
    print(f"  - total_chunk_count: {result.validation['total_chunk_count']}")
    print(f"  - empty_chunk_count: {result.validation['empty_chunk_count']}")
    print(
        "  - missing_metadata_count: "
        f"{result.validation['missing_metadata_count']}"
    )
    print(
        "  - max_chunk_char_length: "
        f"{result.validation['max_chunk_char_length']}"
    )
    print(
        "  - max_chunk_token_estimate: "
        f"{result.validation['max_chunk_token_estimate']}"
    )
    print(
        "  - source_documents_represented: "
        f"{result.validation['source_documents_represented']}"
    )
    print("")
    print(
        "Source documents: "
        f"seen={result.source_document_counts['documents_seen_in_selected_window']}, "
        f"newly_chunked={result.source_document_counts['documents_chunked_in_this_run']}"
    )
    print(f"Resumed from existing artifact: {result.resumed_from_existing_artifact}")
    print(f"Chunk artifact written to: {result.chunk_artifact_path}")
    print(f"Chunk manifest written to: {result.chunk_manifest_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    paths = get_paths()
    parser = argparse.ArgumentParser(
        description="Generate Part 2 RAG chunks from the frozen corpus manifest."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=paths.part1_db_path,
        help="Path to the Part 1 SQLite database.",
    )
    parser.add_argument(
        "--source-corpus-manifest",
        type=Path,
        default=get_default_manifest_path(),
        help="Path to the frozen corpus manifest JSON.",
    )
    parser.add_argument(
        "--chunk-artifact-path",
        type=Path,
        default=get_default_chunk_artifact_path(),
        help=f"Path to write the chunk JSONL artifact. Default: {DEFAULT_CHUNK_ARTIFACT_FILENAME}",
    )
    parser.add_argument(
        "--chunk-manifest-path",
        type=Path,
        default=get_default_chunk_manifest_path(),
        help=f"Path to write the chunk manifest JSON. Default: {DEFAULT_CHUNK_MANIFEST_FILENAME}",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=DEFAULT_MAX_CHUNK_TOKENS,
        help="Approximate token budget per chunk.",
    )
    parser.add_argument(
        "--post-overlap-sentences",
        type=int,
        default=DEFAULT_POST_OVERLAP_SENTENCES,
        help="Number of overlapping sentences between adjacent post chunks.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a fresh chunk artifact instead of resuming from an existing JSONL file.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        result = build_chunks(
            db_path=args.db_path.resolve(),
            source_corpus_manifest_path=args.source_corpus_manifest.resolve(),
            chunk_artifact_path=args.chunk_artifact_path.resolve(),
            chunk_manifest_path=args.chunk_manifest_path.resolve(),
            max_chunk_tokens=args.max_chunk_tokens,
            overlap_sentences=args.post_overlap_sentences,
            resume=not args.no_resume,
        )
    except ChunkGenerationError as exc:
        parser.exit(status=1, message=f"Chunk generation failed: {exc}\n")
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
