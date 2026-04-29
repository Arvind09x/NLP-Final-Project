from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from part2_rag.config import (
    DEFAULT_DENSE_TOP_K,
    DEFAULT_HYBRID_FINAL_TOP_K,
    DEFAULT_LEXICAL_TOP_K,
    DEFAULT_RRF_CONSTANT,
    get_default_chunk_artifact_path,
    get_default_chunk_manifest_path,
    get_default_embedding_artifact_path,
    get_default_embedding_manifest_path,
    get_default_faiss_index_path,
    get_default_manifest_path,
    get_default_part1_db_path,
)
from part2_rag.embedding_index import (
    EmbeddingIndexBuildError,
    load_embedding_manifest,
    load_json_file,
    require_numpy,
    load_sentence_transformer,
    require_faiss,
)


MAX_SNIPPET_CHARS = 280
FTS_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
QUESTION_START_PATTERN = re.compile(
    r"^(?:what|how|why|when|where|which|who|can|could|should|would|do|does|did|is|are|am|will|need)\b"
)
ANSWER_CUE_PATTERN = re.compile(
    r"\b(?:you should|you can|i would|start with|need to|continue|keep|recommend|optional|unnecessary|no need)\b"
)
LEXICAL_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "good",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "people",
        "say",
        "should",
        "subreddit",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "this",
        "think",
        "to",
        "what",
        "while",
        "without",
    }
)
MAX_PROMOTED_REPLIES_PER_PARENT = 2


class RetrievalError(RuntimeError):
    """Raised when retrieval artifacts or queries are invalid."""


@dataclass(frozen=True)
class RetrievalConfig:
    dense_top_k: int = DEFAULT_DENSE_TOP_K
    lexical_top_k: int = DEFAULT_LEXICAL_TOP_K
    hybrid_final_top_k: int = DEFAULT_HYBRID_FINAL_TOP_K
    rrf_constant: int = DEFAULT_RRF_CONSTANT
    faiss_index_path: Path = get_default_faiss_index_path()
    embedding_store_path: Path = get_default_embedding_artifact_path()
    chunk_artifact_path: Path = get_default_chunk_artifact_path()
    corpus_manifest_path: Path = get_default_manifest_path()
    chunk_manifest_path: Path = get_default_chunk_manifest_path()
    embedding_manifest_path: Path = get_default_embedding_manifest_path()
    part1_db_path: Path = get_default_part1_db_path()


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
    chunk_index: int
    token_estimate: int
    chunk_origin: str
    chunk_text: str


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    chunk_id: str
    document_id: str
    source_type: str
    chunk_index: int
    chunk_origin: str
    title: str | None
    created_utc: int
    score: float
    retrieval_source: str
    snippet: str
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    rrf_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenChunkCatalog:
    chunk_by_id: dict[str, ChunkRecord]
    chunks_by_document_id: dict[str, tuple[ChunkRecord, ...]]
    children_by_parent_id: dict[str, tuple[ChunkRecord, ...]]

    @property
    def document_ids(self) -> set[str]:
        return set(self.chunks_by_document_id)


def get_default_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig()


def _normalize_snippet(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= MAX_SNIPPET_CHARS:
        return normalized
    return normalized[: MAX_SNIPPET_CHARS - 3].rstrip() + "..."


def _chunk_sort_key(record: ChunkRecord) -> tuple[int, str]:
    return (record.chunk_index, record.chunk_id)


@lru_cache(maxsize=4)
def load_frozen_chunk_catalog(chunk_artifact_path: str) -> FrozenChunkCatalog:
    path = Path(chunk_artifact_path)
    if not path.exists():
        raise RetrievalError(f"Frozen chunk artifact does not exist: {path}")

    chunk_by_id: dict[str, ChunkRecord] = {}
    chunks_by_document_id: dict[str, list[ChunkRecord]] = {}
    children_by_parent_id: dict[str, list[ChunkRecord]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload_text = line.strip()
            if not payload_text:
                continue
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                raise RetrievalError(
                    f"Chunk artifact line {line_number} is not valid JSON."
                ) from exc

            record = ChunkRecord(
                chunk_id=str(payload["chunk_id"]),
                document_id=str(payload["document_id"]),
                source_type=str(payload["source_type"]),
                source_id=str(payload["source_id"]),
                post_id=(
                    str(payload["post_id"]) if payload.get("post_id") is not None else None
                ),
                parent_id=(
                    str(payload["parent_id"]) if payload.get("parent_id") is not None else None
                ),
                link_id=(
                    str(payload["link_id"]) if payload.get("link_id") is not None else None
                ),
                created_utc=int(payload["created_utc"]),
                author_id=(
                    str(payload["author_id"]) if payload.get("author_id") is not None else None
                ),
                title=str(payload["title"]) if payload.get("title") is not None else None,
                chunk_index=int(payload["chunk_index"]),
                token_estimate=int(payload["token_estimate"]),
                chunk_origin=str(payload["chunk_origin"]),
                chunk_text=str(payload["chunk_text"]),
            )
            if record.chunk_id in chunk_by_id:
                raise RetrievalError(
                    f"Duplicate chunk_id detected in frozen chunk artifact: {record.chunk_id}"
                )
            chunk_by_id[record.chunk_id] = record
            chunks_by_document_id.setdefault(record.document_id, []).append(record)
            if record.parent_id:
                children_by_parent_id.setdefault(record.parent_id, []).append(record)

    frozen_mapping = {
        document_id: tuple(sorted(records, key=_chunk_sort_key))
        for document_id, records in chunks_by_document_id.items()
    }
    child_mapping = {
        parent_id: tuple(sorted(records, key=_chunk_sort_key))
        for parent_id, records in children_by_parent_id.items()
    }
    return FrozenChunkCatalog(
        chunk_by_id=chunk_by_id,
        chunks_by_document_id=frozen_mapping,
        children_by_parent_id=child_mapping,
    )


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise RetrievalError(f"{name} must be positive.")


def extract_query_terms(query_text: str) -> tuple[str, ...]:
    raw_tokens = FTS_TOKEN_PATTERN.findall(query_text.lower())
    if not raw_tokens:
        raise RetrievalError("Retrieval query must contain at least one alphanumeric token.")

    filtered_tokens: list[str] = []
    seen_tokens: set[str] = set()
    for token in raw_tokens:
        if len(token) <= 2:
            continue
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        if token in LEXICAL_STOPWORDS:
            continue
        filtered_tokens.append(token)

    if filtered_tokens:
        return tuple(filtered_tokens)

    return tuple(dict.fromkeys(raw_tokens))


def _normalize_lexical_query(query_text: str) -> str:
    tokens = extract_query_terms(query_text)
    if not tokens:
        raise RetrievalError("Lexical retrieval query must contain at least one alphanumeric token.")
    return " OR ".join(tokens)


def _embedding_connection(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RetrievalError(f"Embedding store does not exist: {path}")
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _part1_connection(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RetrievalError(f"Part 1 SQLite database does not exist: {path}")
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _build_result(
    *,
    rank: int,
    chunk: ChunkRecord,
    score: float,
    retrieval_source: str,
    dense_rank: int | None = None,
    dense_score: float | None = None,
    lexical_rank: int | None = None,
    lexical_score: float | None = None,
    rrf_score: float | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source_type=chunk.source_type,
        chunk_index=chunk.chunk_index,
        chunk_origin=chunk.chunk_origin,
        title=chunk.title,
        created_utc=chunk.created_utc,
        score=float(score),
        retrieval_source=retrieval_source,
        snippet=_normalize_snippet(chunk.chunk_text),
        dense_rank=dense_rank,
        dense_score=dense_score,
        lexical_rank=lexical_rank,
        lexical_score=lexical_score,
        rrf_score=rrf_score,
    )


def _get_chunk_for_document(
    catalog: FrozenChunkCatalog,
    document_id: str,
) -> ChunkRecord | None:
    chunks = catalog.chunks_by_document_id.get(document_id)
    if not chunks:
        return None
    return chunks[0]


def _record_fullname(record: ChunkRecord) -> str | None:
    if record.source_type == "comment":
        return f"t1_{record.source_id}"
    if record.source_type == "post":
        return f"t3_{record.source_id}"
    return None


def _is_question_like_chunk(record: ChunkRecord) -> bool:
    if record.source_type != "comment":
        return False
    text = record.chunk_text.strip().lower()
    return "?" in text or QUESTION_START_PATTERN.match(text) is not None


def _is_reply_promotion_eligible_parent(record: ChunkRecord) -> bool:
    return _is_question_like_chunk(record) and record.token_estimate <= 24


def _score_reply_candidate(
    reply_record: ChunkRecord,
    *,
    query_terms: Sequence[str],
) -> float:
    reply_terms = set(FTS_TOKEN_PATTERN.findall(reply_record.chunk_text.lower()))
    overlap_count = sum(1 for term in query_terms if term in reply_terms)
    if overlap_count == 0:
        return 0.0

    score = float(overlap_count)
    if ANSWER_CUE_PATTERN.search(reply_record.chunk_text.lower()) is not None:
        score += 0.5
    if _is_question_like_chunk(reply_record):
        score -= 0.75
    return score


def fetch_chunk_ids_for_row_positions(
    connection: sqlite3.Connection,
    row_positions: Sequence[int],
) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for row_position in row_positions:
        row = connection.execute(
            "SELECT chunk_id FROM embeddings WHERE row_index = ?",
            (int(row_position),),
        ).fetchone()
        if row is None:
            raise RetrievalError(
                f"Embedding store is missing row_index={row_position} returned by FAISS."
            )
        mapping[int(row_position)] = str(row["chunk_id"])
    return mapping


def _prepare_faiss_query_vector(
    query_vector: Any,
    *,
    expected_dimension: int,
) -> Any:
    np = require_numpy()
    vector = np.asarray(query_vector, dtype=np.float32, order="C")
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    if vector.ndim != 2 or vector.shape[0] != 1:
        raise RetrievalError(
            "Dense retrieval query embedding must have shape (1, embedding_dimension)."
        )
    if int(vector.shape[1]) != expected_dimension:
        raise RetrievalError(
            "Dense retrieval query embedding dimension does not match the saved FAISS index. "
            f"query_dimension={int(vector.shape[1])}, expected={expected_dimension}"
        )
    return np.ascontiguousarray(vector, dtype=np.float32)


def dense_retrieve(
    query_text: str,
    *,
    config: RetrievalConfig | None = None,
    top_k: int | None = None,
    model: Any | None = None,
    faiss_module: Any | None = None,
) -> list[RetrievalResult]:
    settings = config or get_default_retrieval_config()
    top_k = settings.dense_top_k if top_k is None else int(top_k)
    _require_positive("dense_top_k", top_k)

    catalog = load_frozen_chunk_catalog(str(settings.chunk_artifact_path.resolve()))
    manifest = load_embedding_manifest(settings.embedding_manifest_path.resolve())
    embedder = model or load_sentence_transformer(manifest.model_name)

    query_vector = embedder.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_vector = _prepare_faiss_query_vector(
        query_vector,
        expected_dimension=manifest.embedding_dimension,
    )

    faiss = faiss_module or require_faiss()
    index = faiss.read_index(str(settings.faiss_index_path.resolve()))
    if int(index.ntotal) != manifest.embedding_count:
        raise RetrievalError(
            "Saved FAISS index vector count does not match embedding_manifest_v1.json."
        )
    if int(index.d) != manifest.embedding_dimension:
        raise RetrievalError(
            "Saved FAISS index dimension does not match embedding_manifest_v1.json."
        )
    scores, indices = index.search(query_vector, top_k)
    row_positions = [int(row_position) for row_position in indices[0].tolist() if int(row_position) >= 0]

    connection = _embedding_connection(settings.embedding_store_path.resolve())
    try:
        row_to_chunk_id = fetch_chunk_ids_for_row_positions(connection, row_positions)
    finally:
        connection.close()

    results: list[RetrievalResult] = []
    for rank, (score, row_position) in enumerate(
        zip(scores[0].tolist(), indices[0].tolist(), strict=True),
        start=1,
    ):
        row_position = int(row_position)
        if row_position < 0:
            continue
        chunk_id = row_to_chunk_id[row_position]
        chunk = catalog.chunk_by_id.get(chunk_id)
        if chunk is None:
            raise RetrievalError(
                f"Chunk id {chunk_id} from embedding store is missing from the frozen chunk artifact."
            )
        results.append(
            _build_result(
                rank=rank,
                chunk=chunk,
                score=float(score),
                retrieval_source="dense",
                dense_rank=rank,
                dense_score=float(score),
            )
        )
    return results


def lexical_retrieve(
    query_text: str,
    *,
    config: RetrievalConfig | None = None,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    settings = config or get_default_retrieval_config()
    top_k = settings.lexical_top_k if top_k is None else int(top_k)
    _require_positive("lexical_top_k", top_k)

    catalog = load_frozen_chunk_catalog(str(settings.chunk_artifact_path.resolve()))
    fts_query = _normalize_lexical_query(query_text)

    connection = _part1_connection(settings.part1_db_path.resolve())
    try:
        rows = connection.execute(
            """
            SELECT document_id, bm25(documents_fts) AS bm25_score
            FROM documents_fts
            WHERE documents_fts MATCH ?
            ORDER BY bm25(documents_fts) ASC, document_id ASC
            LIMIT ?
            """,
            (fts_query, top_k * 8),
        ).fetchall()
    finally:
        connection.close()

    results: list[RetrievalResult] = []
    seen_document_ids: set[str] = set()
    for row in rows:
        document_id = str(row["document_id"])
        if document_id in seen_document_ids:
            continue
        chunk = _get_chunk_for_document(catalog, document_id)
        if chunk is None:
            continue
        seen_document_ids.add(document_id)
        bm25_score = float(row["bm25_score"])
        lexical_score = -bm25_score
        results.append(
            _build_result(
                rank=len(results) + 1,
                chunk=chunk,
                score=lexical_score,
                retrieval_source="lexical",
                lexical_rank=len(results) + 1,
                lexical_score=lexical_score,
            )
        )
        if len(results) >= top_k:
            break
    return results


def merge_results_with_rrf(
    dense_results: Sequence[RetrievalResult],
    lexical_results: Sequence[RetrievalResult],
    *,
    rrf_constant: int,
    final_top_k: int,
) -> list[RetrievalResult]:
    _require_positive("rrf_constant", rrf_constant)
    _require_positive("hybrid_final_top_k", final_top_k)

    aggregated: dict[str, dict[str, Any]] = {}

    def upsert(result: RetrievalResult, source_name: str) -> None:
        entry = aggregated.setdefault(
            result.document_id,
            {
                "document_id": result.document_id,
                "canonical_result": result,
                "dense_rank": None,
                "dense_score": None,
                "lexical_rank": None,
                "lexical_score": None,
                "rrf_score": 0.0,
            },
        )
        entry["rrf_score"] += 1.0 / float(rrf_constant + result.rank)

        if source_name == "dense":
            if entry["dense_rank"] is None or result.rank < entry["dense_rank"]:
                entry["dense_rank"] = result.rank
                entry["dense_score"] = result.score
                entry["canonical_result"] = result
        else:
            if entry["lexical_rank"] is None or result.rank < entry["lexical_rank"]:
                entry["lexical_rank"] = result.rank
                entry["lexical_score"] = result.score
                if entry["dense_rank"] is None:
                    entry["canonical_result"] = result

    for result in dense_results:
        upsert(result, "dense")
    for result in lexical_results:
        upsert(result, "lexical")

    ranked_entries = sorted(
        aggregated.values(),
        key=lambda entry: (
            -float(entry["rrf_score"]),
            entry["dense_rank"] if entry["dense_rank"] is not None else 10**9,
            entry["lexical_rank"] if entry["lexical_rank"] is not None else 10**9,
            str(entry["document_id"]),
        ),
    )

    merged_results: list[RetrievalResult] = []
    for rank, entry in enumerate(ranked_entries[:final_top_k], start=1):
        canonical = entry["canonical_result"]
        merged_results.append(
            RetrievalResult(
                rank=rank,
                chunk_id=canonical.chunk_id,
                document_id=canonical.document_id,
                source_type=canonical.source_type,
                chunk_index=canonical.chunk_index,
                chunk_origin=canonical.chunk_origin,
                title=canonical.title,
                created_utc=canonical.created_utc,
                score=float(entry["rrf_score"]),
                retrieval_source="hybrid",
                snippet=canonical.snippet,
                dense_rank=entry["dense_rank"],
                dense_score=entry["dense_score"],
                lexical_rank=entry["lexical_rank"],
                lexical_score=entry["lexical_score"],
                rrf_score=float(entry["rrf_score"]),
            )
        )
    return merged_results


def hybrid_retrieve(
    query_text: str,
    *,
    config: RetrievalConfig | None = None,
) -> list[RetrievalResult]:
    settings = config or get_default_retrieval_config()
    dense_results = dense_retrieve(query_text, config=settings)
    lexical_results = lexical_retrieve(query_text, config=settings)
    candidate_top_k = max(
        settings.hybrid_final_top_k * 2,
        settings.hybrid_final_top_k + 3,
    )
    merged_results = merge_results_with_rrf(
        dense_results,
        lexical_results,
        rrf_constant=settings.rrf_constant,
        final_top_k=candidate_top_k,
    )
    return _promote_direct_replies(
        merged_results,
        query_text=query_text,
        config=settings,
        final_top_k=settings.hybrid_final_top_k,
    )


def _promote_direct_replies(
    results: Sequence[RetrievalResult],
    *,
    query_text: str,
    config: RetrievalConfig,
    final_top_k: int,
) -> list[RetrievalResult]:
    if not results:
        return []

    catalog = load_frozen_chunk_catalog(str(config.chunk_artifact_path.resolve()))
    query_terms = extract_query_terms(query_text)
    expanded_results: list[RetrievalResult] = []
    seen_document_ids: set[str] = set()

    for result_index, parent_result in enumerate(results, start=1):
        parent_record = catalog.chunk_by_id.get(parent_result.chunk_id)
        if parent_record is None:
            continue

        added_reply = False
        if result_index <= final_top_k and _is_reply_promotion_eligible_parent(parent_record):
            parent_fullname = _record_fullname(parent_record)
            child_records = catalog.children_by_parent_id.get(parent_fullname or "", ())
            ranked_children = sorted(
                (
                    (
                        _score_reply_candidate(child_record, query_terms=query_terms),
                        child_record,
                    )
                    for child_record in child_records
                    if child_record.document_id not in seen_document_ids
                    and child_record.document_id != parent_result.document_id
                ),
                key=lambda item: (-item[0], item[1].created_utc, item[1].chunk_id),
            )

            promoted_count = 0
            for reply_score, child_record in ranked_children:
                if reply_score <= 0.0:
                    continue
                added_reply = True
                promoted_count += 1
                seen_document_ids.add(child_record.document_id)
                expanded_results.append(
                    RetrievalResult(
                        rank=0,
                        chunk_id=child_record.chunk_id,
                        document_id=child_record.document_id,
                        source_type=child_record.source_type,
                        chunk_index=child_record.chunk_index,
                        chunk_origin=child_record.chunk_origin,
                        title=child_record.title,
                        created_utc=child_record.created_utc,
                        score=float(parent_result.score + (reply_score / 1000.0)),
                        retrieval_source=f"{parent_result.retrieval_source}+reply",
                        snippet=_normalize_snippet(child_record.chunk_text),
                        rrf_score=parent_result.rrf_score,
                    )
                )
                if promoted_count >= MAX_PROMOTED_REPLIES_PER_PARENT:
                    break

        if added_reply:
            continue
        if parent_result.document_id in seen_document_ids:
            continue
        seen_document_ids.add(parent_result.document_id)
        expanded_results.append(parent_result)

    final_results = expanded_results[:final_top_k]
    return [
        RetrievalResult(
            rank=index,
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            source_type=result.source_type,
            chunk_index=result.chunk_index,
            chunk_origin=result.chunk_origin,
            title=result.title,
            created_utc=result.created_utc,
            score=result.score,
            retrieval_source=result.retrieval_source,
            snippet=result.snippet,
            dense_rank=result.dense_rank,
            dense_score=result.dense_score,
            lexical_rank=result.lexical_rank,
            lexical_score=result.lexical_score,
            rrf_score=result.rrf_score,
        )
        for index, result in enumerate(final_results, start=1)
    ]


def retrieve(
    query_text: str,
    *,
    mode: str,
    config: RetrievalConfig | None = None,
) -> list[RetrievalResult]:
    selected_mode = mode.strip().lower()
    if selected_mode == "dense":
        return dense_retrieve(query_text, config=config)
    if selected_mode == "lexical":
        return lexical_retrieve(query_text, config=config)
    if selected_mode == "hybrid":
        return hybrid_retrieve(query_text, config=config)
    raise RetrievalError(f"Unsupported retrieval mode: {mode}")


def format_retrieval_results(
    mode: str,
    results: Sequence[RetrievalResult],
) -> str:
    lines = [f"Retrieval debug results ({mode})"]
    for result in results:
        score_parts = [f"score={result.score:.6f}"]
        if result.dense_rank is not None:
            score_parts.append(f"dense_rank={result.dense_rank}")
        if result.dense_score is not None:
            score_parts.append(f"dense_score={result.dense_score:.6f}")
        if result.lexical_rank is not None:
            score_parts.append(f"lexical_rank={result.lexical_rank}")
        if result.lexical_score is not None:
            score_parts.append(f"lexical_score={result.lexical_score:.6f}")
        if result.rrf_score is not None:
            score_parts.append(f"rrf_score={result.rrf_score:.6f}")

        lines.append(
            " | ".join(
                [
                    f"[{result.rank}] {result.retrieval_source}",
                    f"chunk_id={result.chunk_id}",
                    f"document_id={result.document_id}",
                    f"source_type={result.source_type}",
                    f"chunk_index={result.chunk_index}",
                    f"chunk_origin={result.chunk_origin}",
                    f"created_utc={result.created_utc}",
                    f"title={result.title!r}",
                    *score_parts,
                ]
            )
        )
        lines.append(f"  snippet: {result.snippet}")
    return "\n".join(lines)


def build_retrieval_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run dense, lexical, or hybrid retrieval against the frozen Part 2 artifacts."
    )
    parser.add_argument("--query", required=True, help="Query text to retrieve against.")
    parser.add_argument(
        "--mode",
        choices=("dense", "lexical", "hybrid"),
        default="hybrid",
        help="Retrieval mode to execute.",
    )
    parser.add_argument("--dense-top-k", type=int, default=DEFAULT_DENSE_TOP_K)
    parser.add_argument("--lexical-top-k", type=int, default=DEFAULT_LEXICAL_TOP_K)
    parser.add_argument("--hybrid-final-top-k", type=int, default=DEFAULT_HYBRID_FINAL_TOP_K)
    parser.add_argument("--rrf-constant", type=int, default=DEFAULT_RRF_CONSTANT)
    parser.add_argument(
        "--faiss-index-path",
        type=Path,
        default=get_default_faiss_index_path(),
    )
    parser.add_argument(
        "--embedding-store-path",
        type=Path,
        default=get_default_embedding_artifact_path(),
    )
    parser.add_argument(
        "--chunk-artifact-path",
        type=Path,
        default=get_default_chunk_artifact_path(),
    )
    parser.add_argument(
        "--embedding-manifest-path",
        type=Path,
        default=get_default_embedding_manifest_path(),
    )
    parser.add_argument(
        "--chunk-manifest-path",
        type=Path,
        default=get_default_chunk_manifest_path(),
    )
    parser.add_argument(
        "--corpus-manifest-path",
        type=Path,
        default=get_default_manifest_path(),
    )
    parser.add_argument(
        "--part1-db-path",
        type=Path,
        default=get_default_part1_db_path(),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RetrievalConfig:
    return RetrievalConfig(
        dense_top_k=args.dense_top_k,
        lexical_top_k=args.lexical_top_k,
        hybrid_final_top_k=args.hybrid_final_top_k,
        rrf_constant=args.rrf_constant,
        faiss_index_path=args.faiss_index_path.resolve(),
        embedding_store_path=args.embedding_store_path.resolve(),
        chunk_artifact_path=args.chunk_artifact_path.resolve(),
        corpus_manifest_path=args.corpus_manifest_path.resolve(),
        chunk_manifest_path=args.chunk_manifest_path.resolve(),
        embedding_manifest_path=args.embedding_manifest_path.resolve(),
        part1_db_path=args.part1_db_path.resolve(),
    )


def retrieval_debug_main() -> int:
    parser = build_retrieval_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    try:
        chunk_manifest = load_json_file(config.chunk_manifest_path)
        total_chunks = int(chunk_manifest["chunk_counts"]["total_chunks"])
        embedding_manifest = load_embedding_manifest(config.embedding_manifest_path)
        if embedding_manifest.embedding_count != total_chunks:
            raise RetrievalError(
                "Embedding manifest count does not match frozen chunk manifest count."
            )
        results = retrieve(args.query, mode=args.mode, config=config)
    except (EmbeddingIndexBuildError, KeyError, TypeError, ValueError, RetrievalError) as exc:
        parser.exit(status=1, message=f"Retrieval debug failed: {exc}\n")
    print(format_retrieval_results(args.mode, results))
    return 0


__all__ = [
    "ChunkRecord",
    "FrozenChunkCatalog",
    "RetrievalConfig",
    "RetrievalError",
    "RetrievalResult",
    "dense_retrieve",
    "fetch_chunk_ids_for_row_positions",
    "format_retrieval_results",
    "get_default_retrieval_config",
    "hybrid_retrieve",
    "lexical_retrieve",
    "load_frozen_chunk_catalog",
    "merge_results_with_rrf",
    "retrieve",
    "retrieval_debug_main",
]
