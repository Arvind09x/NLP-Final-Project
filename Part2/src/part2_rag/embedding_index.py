from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from array import array
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from part2_rag.config import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_EXPECTED_FROZEN_CHUNK_COUNT,
    get_default_chunk_artifact_path,
    get_default_chunk_manifest_path,
    get_default_embedding_artifact_path,
    get_default_embedding_manifest_path,
    get_default_faiss_index_path,
)


class EmbeddingIndexBuildError(RuntimeError):
    """Raised when embedding or index build inputs are invalid or inconsistent."""


@dataclass(frozen=True)
class ChunkArtifactRecord:
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
class EmbeddingBuildParameters:
    model_name: str
    embedding_dimension: int
    batch_size: int
    normalize_embeddings: bool


@dataclass(frozen=True)
class EmbeddingBuildInputs:
    chunk_artifact_path: str
    chunk_artifact_hash: str
    chunk_manifest_path: str
    chunk_manifest_hash: str
    expected_chunk_count: int


@dataclass(frozen=True)
class EmbeddingBuildResult:
    model_name: str
    embedding_dimension: int
    batch_size: int
    chunk_artifact_path: str
    chunk_artifact_hash: str
    chunk_manifest_path: str
    chunk_manifest_hash: str
    expected_chunk_count: int
    embedding_count: int
    embedding_artifact_path: str
    faiss_index_path: str
    build_timestamp: str
    resume_status: str


REQUIRED_CHUNK_KEYS = {
    "chunk_id",
    "document_id",
    "source_type",
    "source_id",
    "created_utc",
    "chunk_text",
    "chunk_index",
    "token_estimate",
    "chunk_origin",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_path_for_comparison(path_value: str | Path) -> str:
    return str(Path(path_value).resolve())


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EmbeddingIndexBuildError(f"Required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EmbeddingIndexBuildError(f"JSON file is not valid: {path}") from exc


def get_manifest_chunk_count(chunk_manifest: dict[str, Any]) -> int:
    try:
        return int(chunk_manifest["chunk_counts"]["total_chunks"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbeddingIndexBuildError(
            "Chunk manifest is missing chunk_counts.total_chunks."
        ) from exc


def iter_chunk_records(chunk_artifact_path: Path) -> Iterator[ChunkArtifactRecord]:
    if not chunk_artifact_path.exists():
        raise EmbeddingIndexBuildError(
            f"Frozen chunk artifact does not exist: {chunk_artifact_path}"
        )

    with chunk_artifact_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload_text = line.strip()
            if not payload_text:
                continue
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                raise EmbeddingIndexBuildError(
                    f"Chunk artifact line {line_number} is not valid JSON."
                ) from exc

            missing_keys = sorted(REQUIRED_CHUNK_KEYS - set(payload))
            if missing_keys:
                joined = ", ".join(missing_keys)
                raise EmbeddingIndexBuildError(
                    f"Chunk artifact line {line_number} is missing required keys: {joined}"
                )

            chunk_id = str(payload["chunk_id"]).strip()
            chunk_text = str(payload["chunk_text"]).strip()
            if not chunk_id:
                raise EmbeddingIndexBuildError(
                    f"Chunk artifact line {line_number} has an empty chunk_id."
                )
            if not chunk_text:
                raise EmbeddingIndexBuildError(
                    f"Chunk artifact line {line_number} has an empty chunk_text."
                )

            yield ChunkArtifactRecord(
                chunk_id=chunk_id,
                document_id=str(payload["document_id"]),
                source_type=str(payload["source_type"]),
                source_id=str(payload["source_id"]),
                post_id=str(payload["post_id"]) if payload.get("post_id") is not None else None,
                parent_id=(
                    str(payload["parent_id"]) if payload.get("parent_id") is not None else None
                ),
                link_id=str(payload["link_id"]) if payload.get("link_id") is not None else None,
                created_utc=int(payload["created_utc"]),
                author_id=(
                    str(payload["author_id"]) if payload.get("author_id") is not None else None
                ),
                title=str(payload["title"]) if payload.get("title") is not None else None,
                chunk_text=chunk_text,
                chunk_index=int(payload["chunk_index"]),
                token_estimate=int(payload["token_estimate"]),
                chunk_origin=str(payload["chunk_origin"]),
            )


def validate_frozen_chunk_inputs(
    *,
    chunk_artifact_path: Path,
    chunk_manifest_path: Path,
    authoritative_expected_chunk_count: int = DEFAULT_EXPECTED_FROZEN_CHUNK_COUNT,
) -> EmbeddingBuildInputs:
    chunk_manifest = load_json_file(chunk_manifest_path)
    manifest_chunk_count = get_manifest_chunk_count(chunk_manifest)

    observed_chunk_count = 0
    seen_chunk_ids: set[str] = set()
    for record in iter_chunk_records(chunk_artifact_path):
        if record.chunk_id in seen_chunk_ids:
            raise EmbeddingIndexBuildError(
                f"Duplicate chunk_id detected in frozen chunk artifact: {record.chunk_id}"
            )
        seen_chunk_ids.add(record.chunk_id)
        observed_chunk_count += 1

    if manifest_chunk_count != observed_chunk_count:
        raise EmbeddingIndexBuildError(
            "Frozen chunk manifest count does not match the chunk artifact line count. "
            f"manifest={manifest_chunk_count}, artifact={observed_chunk_count}"
        )

    if authoritative_expected_chunk_count != observed_chunk_count:
        raise EmbeddingIndexBuildError(
            "Frozen chunk alignment check failed. "
            f"Authoritative expected chunk count={authoritative_expected_chunk_count}, "
            f"but the local chunk artifact reports {observed_chunk_count}. "
            "Do not regenerate or reinterpret chunks; resolve the authoritative mismatch first."
        )

    return EmbeddingBuildInputs(
        chunk_artifact_path=str(chunk_artifact_path),
        chunk_artifact_hash=compute_file_sha256(chunk_artifact_path),
        chunk_manifest_path=str(chunk_manifest_path),
        chunk_manifest_hash=compute_file_sha256(chunk_manifest_path),
        expected_chunk_count=observed_chunk_count,
    )


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def connect_embedding_store(path: Path) -> sqlite3.Connection:
    ensure_parent_dir(path)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS build_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            row_index INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            document_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            post_id TEXT,
            parent_id TEXT,
            link_id TEXT,
            created_utc INTEGER NOT NULL,
            author_id TEXT,
            title TEXT,
            chunk_index INTEGER NOT NULL,
            token_estimate INTEGER NOT NULL,
            chunk_origin TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id)"
    )
    return connection


def read_store_metadata(connection: sqlite3.Connection) -> dict[str, Any]:
    rows = connection.execute("SELECT key, value FROM build_metadata").fetchall()
    metadata: dict[str, Any] = {}
    for key, value in rows:
        try:
            metadata[str(key)] = json.loads(value)
        except json.JSONDecodeError:
            metadata[str(key)] = value
    return metadata


def write_store_metadata(
    connection: sqlite3.Connection,
    *,
    build_inputs: EmbeddingBuildInputs,
    parameters: EmbeddingBuildParameters,
) -> None:
    payload = {
        "chunk_artifact_path": build_inputs.chunk_artifact_path,
        "chunk_artifact_hash": build_inputs.chunk_artifact_hash,
        "chunk_manifest_path": build_inputs.chunk_manifest_path,
        "chunk_manifest_hash": build_inputs.chunk_manifest_hash,
        "expected_chunk_count": build_inputs.expected_chunk_count,
        "model_name": parameters.model_name,
        "embedding_dimension": parameters.embedding_dimension,
        "batch_size": parameters.batch_size,
        "normalize_embeddings": parameters.normalize_embeddings,
    }
    connection.executemany(
        "INSERT OR REPLACE INTO build_metadata(key, value) VALUES (?, ?)",
        [(key, json.dumps(value, sort_keys=True)) for key, value in payload.items()],
    )
    connection.commit()


def assert_resume_compatible(
    *,
    connection: sqlite3.Connection,
    build_inputs: EmbeddingBuildInputs,
    parameters: EmbeddingBuildParameters,
) -> None:
    existing_metadata = read_store_metadata(connection)
    required_comparisons = {
        "chunk_artifact_path": resolve_path_for_comparison(build_inputs.chunk_artifact_path),
        "chunk_artifact_hash": build_inputs.chunk_artifact_hash,
        "chunk_manifest_path": resolve_path_for_comparison(build_inputs.chunk_manifest_path),
        "chunk_manifest_hash": build_inputs.chunk_manifest_hash,
        "expected_chunk_count": build_inputs.expected_chunk_count,
        "model_name": parameters.model_name,
        "embedding_dimension": parameters.embedding_dimension,
        "batch_size": parameters.batch_size,
        "normalize_embeddings": parameters.normalize_embeddings,
    }

    missing_keys = [key for key in required_comparisons if key not in existing_metadata]
    if missing_keys:
        joined = ", ".join(sorted(missing_keys))
        raise EmbeddingIndexBuildError(
            "Resume requested with an existing embedding artifact that is missing "
            f"required metadata: {joined}"
        )

    mismatches: list[str] = []
    for key, current_value in required_comparisons.items():
        existing_value = existing_metadata[key]
        if key.endswith("_path"):
            existing_value = resolve_path_for_comparison(str(existing_value))
        if existing_value != current_value:
            mismatches.append(key)
    if mismatches:
        joined = ", ".join(mismatches)
        raise EmbeddingIndexBuildError(
            "Resume requested with inputs that do not match the existing embedding store. "
            f"Mismatched fields: {joined}"
        )


def get_existing_embedding_state(
    connection: sqlite3.Connection,
) -> tuple[set[str], int]:
    rows = connection.execute(
        "SELECT chunk_id, row_index FROM embeddings ORDER BY row_index ASC"
    ).fetchall()
    chunk_ids = [str(row[0]) for row in rows]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise EmbeddingIndexBuildError("Duplicate chunk_id detected in embedding artifact.")
    row_indexes = [int(row[1]) for row in rows]
    if row_indexes and row_indexes != list(range(len(row_indexes))):
        raise EmbeddingIndexBuildError(
            "Embedding artifact row_index values are not contiguous from zero."
        )
    return set(chunk_ids), len(chunk_ids)


def store_embedding_batch(
    connection: sqlite3.Connection,
    *,
    records: Sequence[ChunkArtifactRecord],
    vectors: Sequence[Sequence[float]],
    starting_row_index: int,
) -> int:
    if len(records) != len(vectors):
        raise EmbeddingIndexBuildError(
            "Embedding batch size mismatch between chunk records and vectors."
        )

    rows = []
    for offset, (record, vector) in enumerate(zip(records, vectors, strict=True)):
        vector_blob = array("f", [float(value) for value in vector]).tobytes()
        rows.append(
            (
                starting_row_index + offset,
                record.chunk_id,
                record.document_id,
                record.source_type,
                record.source_id,
                record.post_id,
                record.parent_id,
                record.link_id,
                record.created_utc,
                record.author_id,
                record.title,
                record.chunk_index,
                record.token_estimate,
                record.chunk_origin,
                record.chunk_text,
                sqlite3.Binary(vector_blob),
            )
        )

    connection.executemany(
        """
        INSERT INTO embeddings (
            row_index, chunk_id, document_id, source_type, source_id, post_id,
            parent_id, link_id, created_utc, author_id, title, chunk_index,
            token_estimate, chunk_origin, chunk_text, embedding
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.commit()
    return len(rows)


def validate_embedding_store(
    *,
    connection: sqlite3.Connection,
    expected_chunk_ids: set[str],
    expected_chunk_count: int,
) -> int:
    rows = connection.execute("SELECT chunk_id FROM embeddings ORDER BY row_index ASC").fetchall()
    embedding_chunk_ids = [str(row[0]) for row in rows]
    if len(embedding_chunk_ids) != len(set(embedding_chunk_ids)):
        raise EmbeddingIndexBuildError("Duplicate chunk_id detected in embedding artifact.")

    embedding_count = len(embedding_chunk_ids)
    if embedding_count != expected_chunk_count:
        raise EmbeddingIndexBuildError(
            "Embedding count does not equal the frozen chunk count. "
            f"expected={expected_chunk_count}, observed={embedding_count}"
        )

    missing_chunk_ids = sorted(expected_chunk_ids - set(embedding_chunk_ids))
    if missing_chunk_ids:
        sample = ", ".join(missing_chunk_ids[:5])
        raise EmbeddingIndexBuildError(
            "Missing embeddings for frozen chunk_ids. "
            f"missing_count={len(missing_chunk_ids)} sample=[{sample}]"
        )

    extra_chunk_ids = sorted(set(embedding_chunk_ids) - expected_chunk_ids)
    if extra_chunk_ids:
        sample = ", ".join(extra_chunk_ids[:5])
        raise EmbeddingIndexBuildError(
            "Embedding artifact contains chunk_ids that are not present in the frozen chunk artifact. "
            f"extra_count={len(extra_chunk_ids)} sample=[{sample}]"
        )

    return embedding_count


def require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise EmbeddingIndexBuildError(
            "numpy is required for FAISS index build and search. Install project dependencies first."
        ) from exc
    return np


def prepare_faiss_query_vector(
    query_vector: Any,
    *,
    expected_dimension: int,
) -> Any:
    np = require_numpy()
    vector = np.asarray(query_vector, dtype=np.float32, order="C")
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    if vector.ndim != 2 or vector.shape[0] != 1:
        raise EmbeddingIndexBuildError(
            "FAISS query embedding must have shape (1, embedding_dimension)."
        )
    if int(vector.shape[1]) != expected_dimension:
        raise EmbeddingIndexBuildError(
            "FAISS query embedding dimension does not match the embedding manifest. "
            f"query_dimension={int(vector.shape[1])}, expected={expected_dimension}"
        )
    return np.ascontiguousarray(vector, dtype=np.float32)


def require_faiss():
    try:
        import faiss
    except ModuleNotFoundError:
        try:
            import faiss_cpu as faiss
        except ModuleNotFoundError as exc:
            raise EmbeddingIndexBuildError(
                "FAISS is required for dense index build and search. Install faiss-cpu first."
            ) from exc
    return faiss


def load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise EmbeddingIndexBuildError(
            "sentence-transformers is required for embedding generation."
        ) from exc
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except TypeError:
        return SentenceTransformer(model_name)
    except Exception:
        return SentenceTransformer(model_name)


def infer_embedding_dimension(model: Any) -> int:
    dimension = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
    if dimension is None:
        raise EmbeddingIndexBuildError(
            "Could not infer embedding dimension from the embedding model."
        )
    return int(dimension)


def embed_text_batch(
    model: Any,
    texts: Sequence[str],
    *,
    normalize_embeddings: bool,
) -> list[list[float]]:
    vectors = model.encode(
        list(texts),
        batch_size=len(texts),
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    return vectors.tolist()


def build_faiss_index_from_store(
    *,
    embedding_store_path: Path,
    faiss_index_path: Path,
    embedding_dimension: int,
    expected_embedding_count: int,
    batch_size: int,
    faiss_module: Any | None = None,
    numpy_module: Any | None = None,
) -> int:
    faiss = faiss_module or require_faiss()
    np = numpy_module or require_numpy()

    ensure_parent_dir(faiss_index_path)
    index = faiss.IndexFlatIP(embedding_dimension)

    connection = sqlite3.connect(embedding_store_path)
    try:
        cursor = connection.execute(
            """
            SELECT embedding
            FROM embeddings
            ORDER BY row_index ASC
            """
        )
        pending_vectors: list[Any] = []
        for row in cursor:
            vector = np.frombuffer(row[0], dtype=np.float32, count=embedding_dimension)
            if vector.shape[0] != embedding_dimension:
                raise EmbeddingIndexBuildError(
                    "Stored embedding dimension does not match the configured dimension."
                )
            pending_vectors.append(vector.copy())
            if len(pending_vectors) >= batch_size:
                index.add(np.vstack(pending_vectors))
                pending_vectors = []
        if pending_vectors:
            index.add(np.vstack(pending_vectors))
    finally:
        connection.close()

    if int(index.ntotal) != expected_embedding_count:
        raise EmbeddingIndexBuildError(
            "FAISS vector count does not equal the embedding count. "
            f"faiss={int(index.ntotal)}, embeddings={expected_embedding_count}"
        )

    faiss.write_index(index, str(faiss_index_path))
    return int(index.ntotal)


def write_embedding_manifest(
    manifest_path: Path,
    result: EmbeddingBuildResult,
) -> None:
    ensure_parent_dir(manifest_path)
    manifest_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_embeddings_and_index(
    *,
    chunk_artifact_path: Path,
    chunk_manifest_path: Path,
    embedding_artifact_path: Path,
    embedding_manifest_path: Path,
    faiss_index_path: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    authoritative_expected_chunk_count: int = DEFAULT_EXPECTED_FROZEN_CHUNK_COUNT,
    resume: bool = True,
) -> EmbeddingBuildResult:
    if batch_size <= 0:
        raise EmbeddingIndexBuildError("Embedding batch size must be positive.")

    build_inputs = validate_frozen_chunk_inputs(
        chunk_artifact_path=chunk_artifact_path,
        chunk_manifest_path=chunk_manifest_path,
        authoritative_expected_chunk_count=authoritative_expected_chunk_count,
    )

    model = load_sentence_transformer(model_name)
    embedding_dimension = infer_embedding_dimension(model)
    parameters = EmbeddingBuildParameters(
        model_name=model_name,
        embedding_dimension=embedding_dimension,
        batch_size=batch_size,
        normalize_embeddings=True,
    )

    connection = connect_embedding_store(embedding_artifact_path)
    try:
        existing_chunk_ids, next_row_index = get_existing_embedding_state(connection)
        resume_status = "fresh"
        if existing_chunk_ids:
            if not resume:
                connection.close()
                embedding_artifact_path.unlink()
                connection = connect_embedding_store(embedding_artifact_path)
                existing_chunk_ids, next_row_index = set(), 0
            else:
                assert_resume_compatible(
                    connection=connection,
                    build_inputs=build_inputs,
                    parameters=parameters,
                )
                resume_status = "resumed"
        if not existing_chunk_ids:
            write_store_metadata(connection, build_inputs=build_inputs, parameters=parameters)

        batch_records: list[ChunkArtifactRecord] = []
        batch_texts: list[str] = []
        expected_chunk_ids: set[str] = set()

        for record in iter_chunk_records(chunk_artifact_path):
            if record.chunk_id in expected_chunk_ids:
                raise EmbeddingIndexBuildError(
                    f"Duplicate chunk_id detected in frozen chunk artifact: {record.chunk_id}"
                )
            expected_chunk_ids.add(record.chunk_id)
            if record.chunk_id in existing_chunk_ids:
                continue
            batch_records.append(record)
            batch_texts.append(record.chunk_text)
            if len(batch_records) >= batch_size:
                vectors = embed_text_batch(
                    model,
                    batch_texts,
                    normalize_embeddings=parameters.normalize_embeddings,
                )
                next_row_index += store_embedding_batch(
                    connection,
                    records=batch_records,
                    vectors=vectors,
                    starting_row_index=next_row_index,
                )
                batch_records = []
                batch_texts = []

        if batch_records:
            vectors = embed_text_batch(
                model,
                batch_texts,
                normalize_embeddings=parameters.normalize_embeddings,
            )
            next_row_index += store_embedding_batch(
                connection,
                records=batch_records,
                vectors=vectors,
                starting_row_index=next_row_index,
            )

        embedding_count = validate_embedding_store(
            connection=connection,
            expected_chunk_ids=expected_chunk_ids,
            expected_chunk_count=build_inputs.expected_chunk_count,
        )
    finally:
        connection.close()

    faiss_count = build_faiss_index_from_store(
        embedding_store_path=embedding_artifact_path,
        faiss_index_path=faiss_index_path,
        embedding_dimension=embedding_dimension,
        expected_embedding_count=embedding_count,
        batch_size=batch_size,
    )
    if faiss_count != embedding_count:
        raise EmbeddingIndexBuildError(
            "FAISS vector count integrity check failed after index write."
        )

    result = EmbeddingBuildResult(
        model_name=model_name,
        embedding_dimension=embedding_dimension,
        batch_size=batch_size,
        chunk_artifact_path=build_inputs.chunk_artifact_path,
        chunk_artifact_hash=build_inputs.chunk_artifact_hash,
        chunk_manifest_path=build_inputs.chunk_manifest_path,
        chunk_manifest_hash=build_inputs.chunk_manifest_hash,
        expected_chunk_count=build_inputs.expected_chunk_count,
        embedding_count=embedding_count,
        embedding_artifact_path=str(embedding_artifact_path),
        faiss_index_path=str(faiss_index_path),
        build_timestamp=utc_now_iso(),
        resume_status=resume_status,
    )
    write_embedding_manifest(embedding_manifest_path, result)
    return result


def load_embedding_manifest(embedding_manifest_path: Path) -> EmbeddingBuildResult:
    payload = load_json_file(embedding_manifest_path)
    return EmbeddingBuildResult(
        model_name=str(payload["model_name"]),
        embedding_dimension=int(payload["embedding_dimension"]),
        batch_size=int(payload["batch_size"]),
        chunk_artifact_path=str(payload["chunk_artifact_path"]),
        chunk_artifact_hash=str(payload["chunk_artifact_hash"]),
        chunk_manifest_path=str(payload["chunk_manifest_path"]),
        chunk_manifest_hash=str(payload["chunk_manifest_hash"]),
        expected_chunk_count=int(payload["expected_chunk_count"]),
        embedding_count=int(payload["embedding_count"]),
        embedding_artifact_path=str(payload["embedding_artifact_path"]),
        faiss_index_path=str(payload["faiss_index_path"]),
        build_timestamp=str(payload["build_timestamp"]),
        resume_status=str(payload["resume_status"]),
    )


def search_faiss_index(
    *,
    embedding_manifest_path: Path,
    query_text: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise EmbeddingIndexBuildError("top_k must be positive.")

    manifest = load_embedding_manifest(embedding_manifest_path)
    model = load_sentence_transformer(manifest.model_name)
    query_vector = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_vector = prepare_faiss_query_vector(
        query_vector,
        expected_dimension=manifest.embedding_dimension,
    )

    faiss = require_faiss()
    index = faiss.read_index(manifest.faiss_index_path)
    if int(index.ntotal) != manifest.embedding_count:
        raise EmbeddingIndexBuildError(
            "Saved FAISS index vector count does not match the embedding manifest."
        )
    if int(index.d) != manifest.embedding_dimension:
        raise EmbeddingIndexBuildError(
            "Saved FAISS index dimension does not match the embedding manifest."
        )

    scores, indices = index.search(query_vector, top_k)
    connection = sqlite3.connect(manifest.embedding_artifact_path)
    try:
        results: list[dict[str, Any]] = []
        for rank, (score, row_index) in enumerate(
            zip(scores[0].tolist(), indices[0].tolist(), strict=True),
            start=1,
        ):
            if row_index < 0:
                continue
            row = connection.execute(
                """
                SELECT chunk_id, document_id, source_type, source_id, post_id,
                       created_utc, title, chunk_index, chunk_origin, token_estimate
                FROM embeddings
                WHERE row_index = ?
                """,
                (int(row_index),),
            ).fetchone()
            if row is None:
                raise EmbeddingIndexBuildError(
                    f"FAISS returned row_index={row_index}, but no embedding metadata exists for it."
                )
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "chunk_id": str(row[0]),
                    "document_id": str(row[1]),
                    "source_type": str(row[2]),
                    "source_id": str(row[3]),
                    "post_id": str(row[4]) if row[4] is not None else None,
                    "created_utc": int(row[5]),
                    "title": str(row[6]) if row[6] is not None else None,
                    "chunk_index": int(row[7]),
                    "chunk_origin": str(row[8]),
                    "token_estimate": int(row[9]),
                }
            )
    finally:
        connection.close()
    return results


def print_build_summary(result: EmbeddingBuildResult) -> None:
    print("Part 2 embedding + FAISS build complete")
    print(f"Model: {result.model_name}")
    print(f"Embedding dimension: {result.embedding_dimension}")
    print(f"Batch size: {result.batch_size}")
    print(f"Expected frozen chunk count: {result.expected_chunk_count}")
    print(f"Embedding count: {result.embedding_count}")
    print(f"Resume status: {result.resume_status}")
    print(f"Embedding artifact: {result.embedding_artifact_path}")
    print(f"FAISS index: {result.faiss_index_path}")


def print_search_results(results: Sequence[dict[str, Any]]) -> None:
    print("FAISS smoke search results")
    for result in results:
        print(
            f"[{result['rank']}] score={result['score']:.6f} "
            f"chunk_id={result['chunk_id']} "
            f"document_id={result['document_id']} "
            f"source_type={result['source_type']} "
            f"chunk_index={result['chunk_index']} "
            f"chunk_origin={result['chunk_origin']} "
            f"created_utc={result['created_utc']} "
            f"title={result['title']!r}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Part 2 embeddings and a FAISS dense index for the frozen chunk artifact."
    )
    parser.add_argument(
        "--chunk-artifact-path",
        type=Path,
        default=get_default_chunk_artifact_path(),
        help="Path to the frozen chunk JSONL artifact.",
    )
    parser.add_argument(
        "--chunk-manifest-path",
        type=Path,
        default=get_default_chunk_manifest_path(),
        help="Path to the frozen chunk manifest JSON.",
    )
    parser.add_argument(
        "--embedding-artifact-path",
        type=Path,
        default=get_default_embedding_artifact_path(),
        help="Path to the resumable embedding SQLite artifact.",
    )
    parser.add_argument(
        "--embedding-manifest-path",
        type=Path,
        default=get_default_embedding_manifest_path(),
        help="Path to write the embedding manifest JSON.",
    )
    parser.add_argument(
        "--faiss-index-path",
        type=Path,
        default=get_default_faiss_index_path(),
        help="Path to write the FAISS index.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_EMBEDDING_MODEL_NAME,
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--expected-chunk-count",
        type=int,
        default=DEFAULT_EXPECTED_FROZEN_CHUNK_COUNT,
        help="Authoritative expected frozen chunk count. Build fails if the local artifact differs.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a fresh embedding artifact instead of resuming.",
    )
    return parser


def build_search_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a one-query smoke search against the saved FAISS dense index."
    )
    parser.add_argument(
        "--embedding-manifest-path",
        type=Path,
        default=get_default_embedding_manifest_path(),
        help="Path to the embedding manifest JSON.",
    )
    parser.add_argument(
        "--query",
        default="best beginner strength routine for fat loss",
        help="Query text to embed and search.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to return.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        result = build_embeddings_and_index(
            chunk_artifact_path=args.chunk_artifact_path.resolve(),
            chunk_manifest_path=args.chunk_manifest_path.resolve(),
            embedding_artifact_path=args.embedding_artifact_path.resolve(),
            embedding_manifest_path=args.embedding_manifest_path.resolve(),
            faiss_index_path=args.faiss_index_path.resolve(),
            model_name=args.model_name,
            batch_size=args.batch_size,
            authoritative_expected_chunk_count=args.expected_chunk_count,
            resume=not args.no_resume,
        )
    except EmbeddingIndexBuildError as exc:
        parser.exit(status=1, message=f"Embedding/index build failed: {exc}\n")
    print_build_summary(result)
    return 0


def smoke_main() -> int:
    parser = build_search_arg_parser()
    args = parser.parse_args()
    try:
        results = search_faiss_index(
            embedding_manifest_path=args.embedding_manifest_path.resolve(),
            query_text=args.query,
            top_k=args.top_k,
        )
    except EmbeddingIndexBuildError as exc:
        parser.exit(status=1, message=f"FAISS smoke search failed: {exc}\n")
    print_search_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
