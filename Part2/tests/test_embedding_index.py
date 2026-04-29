from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from array import array
from pathlib import Path


PART2_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PART2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part2_rag.embedding_index import (
    ChunkArtifactRecord,
    EmbeddingBuildInputs,
    EmbeddingBuildParameters,
    EmbeddingIndexBuildError,
    assert_resume_compatible,
    build_faiss_index_from_store,
    compute_file_sha256,
    connect_embedding_store,
    validate_embedding_store,
    validate_frozen_chunk_inputs,
    write_store_metadata,
)


def write_chunk_files(
    temp_path: Path,
    *,
    chunk_ids: list[str],
    manifest_total_chunks: int | None = None,
) -> tuple[Path, Path]:
    chunk_artifact_path = temp_path / "chunks.jsonl"
    with chunk_artifact_path.open("w", encoding="utf-8") as handle:
        for index, chunk_id in enumerate(chunk_ids):
            payload = {
                "chunk_id": chunk_id,
                "document_id": f"doc-{index}",
                "source_type": "comment",
                "source_id": f"source-{index}",
                "post_id": f"post-{index}",
                "parent_id": f"parent-{index}",
                "link_id": f"link-{index}",
                "created_utc": 1681000000 + index,
                "author_id": f"author-{index}",
                "title": None,
                "chunk_text": f"chunk text {index}",
                "chunk_index": 0,
                "token_estimate": 3,
                "chunk_origin": "text",
            }
            handle.write(json.dumps(payload) + "\n")

    chunk_manifest_path = temp_path / "chunk_manifest.json"
    chunk_manifest_path.write_text(
        json.dumps(
            {
                "chunk_counts": {
                    "total_chunks": (
                        len(chunk_ids)
                        if manifest_total_chunks is None
                        else manifest_total_chunks
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    return chunk_artifact_path, chunk_manifest_path


def insert_embedding_row(
    connection: sqlite3.Connection,
    *,
    row_index: int,
    chunk_id: str,
    vector: list[float],
) -> None:
    connection.execute(
        """
        INSERT INTO embeddings (
            row_index, chunk_id, document_id, source_type, source_id, post_id,
            parent_id, link_id, created_utc, author_id, title, chunk_index,
            token_estimate, chunk_origin, chunk_text, embedding
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row_index,
            chunk_id,
            f"doc-{chunk_id}",
            "comment",
            f"source-{chunk_id}",
            f"post-{chunk_id}",
            None,
            None,
            1681000000,
            "author",
            None,
            0,
            3,
            "text",
            f"text for {chunk_id}",
            sqlite3.Binary(array("f", vector).tobytes()),
        ),
    )
    connection.commit()


class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values
        self.shape = (len(values),)

    def copy(self) -> list[float]:
        return list(self._values)


class FakeNumpy:
    float32 = "float32"

    @staticmethod
    def frombuffer(blob: bytes, dtype: str, count: int) -> FakeVector:
        values = array("f")
        values.frombytes(blob)
        return FakeVector(list(values)[:count])

    @staticmethod
    def vstack(vectors: list[list[float]]) -> list[list[float]]:
        return vectors


class FakeFaissIndex:
    def __init__(self, dimension: int, *, count_vectors: bool) -> None:
        self.dimension = dimension
        self.count_vectors = count_vectors
        self.ntotal = 0

    def add(self, vectors: list[list[float]]) -> None:
        if self.count_vectors:
            self.ntotal += len(vectors)


class FakeFaissModule:
    def __init__(self, *, count_vectors: bool) -> None:
        self.count_vectors = count_vectors
        self.written_paths: list[str] = []

    def IndexFlatIP(self, dimension: int) -> FakeFaissIndex:
        return FakeFaissIndex(dimension, count_vectors=self.count_vectors)

    def write_index(self, index: FakeFaissIndex, path: str) -> None:
        self.written_paths.append(path)


class EmbeddingIndexTests(unittest.TestCase):
    def test_validate_frozen_chunk_inputs_detects_duplicate_chunk_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunk_artifact_path, chunk_manifest_path = write_chunk_files(
                temp_path,
                chunk_ids=["chunk-1", "chunk-1"],
            )

            with self.assertRaises(EmbeddingIndexBuildError):
                validate_frozen_chunk_inputs(
                    chunk_artifact_path=chunk_artifact_path,
                    chunk_manifest_path=chunk_manifest_path,
                    authoritative_expected_chunk_count=2,
                )

    def test_validate_frozen_chunk_inputs_detects_manifest_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunk_artifact_path, chunk_manifest_path = write_chunk_files(
                temp_path,
                chunk_ids=["chunk-1", "chunk-2"],
                manifest_total_chunks=3,
            )

            with self.assertRaises(EmbeddingIndexBuildError):
                validate_frozen_chunk_inputs(
                    chunk_artifact_path=chunk_artifact_path,
                    chunk_manifest_path=chunk_manifest_path,
                    authoritative_expected_chunk_count=2,
                )

    def test_resume_mismatch_failure_on_batch_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunk_artifact_path, chunk_manifest_path = write_chunk_files(
                temp_path,
                chunk_ids=["chunk-1", "chunk-2"],
            )
            build_inputs = EmbeddingBuildInputs(
                chunk_artifact_path=str(chunk_artifact_path),
                chunk_artifact_hash=compute_file_sha256(chunk_artifact_path),
                chunk_manifest_path=str(chunk_manifest_path),
                chunk_manifest_hash=compute_file_sha256(chunk_manifest_path),
                expected_chunk_count=2,
            )
            original_parameters = EmbeddingBuildParameters(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                embedding_dimension=384,
                batch_size=64,
                normalize_embeddings=True,
            )
            resume_parameters = EmbeddingBuildParameters(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                embedding_dimension=384,
                batch_size=128,
                normalize_embeddings=True,
            )

            connection = connect_embedding_store(temp_path / "embeddings.sqlite")
            try:
                write_store_metadata(
                    connection,
                    build_inputs=build_inputs,
                    parameters=original_parameters,
                )
                with self.assertRaises(EmbeddingIndexBuildError):
                    assert_resume_compatible(
                        connection=connection,
                        build_inputs=build_inputs,
                        parameters=resume_parameters,
                    )
            finally:
                connection.close()

    def test_validate_embedding_store_detects_duplicate_embedding_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "embeddings.sqlite"
            connection = sqlite3.connect(db_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE embeddings (
                        row_index INTEGER PRIMARY KEY,
                        chunk_id TEXT NOT NULL,
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
                insert_embedding_row(connection, row_index=0, chunk_id="chunk-1", vector=[1.0, 0.0])
                insert_embedding_row(connection, row_index=1, chunk_id="chunk-1", vector=[0.0, 1.0])

                with self.assertRaises(EmbeddingIndexBuildError):
                    validate_embedding_store(
                        connection=connection,
                        expected_chunk_ids={"chunk-1"},
                        expected_chunk_count=1,
                    )
            finally:
                connection.close()

    def test_validate_embedding_store_detects_missing_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "embeddings.sqlite"
            connection = sqlite3.connect(db_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE embeddings (
                        row_index INTEGER PRIMARY KEY,
                        chunk_id TEXT NOT NULL,
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
                insert_embedding_row(connection, row_index=0, chunk_id="chunk-1", vector=[1.0, 0.0])

                with self.assertRaises(EmbeddingIndexBuildError):
                    validate_embedding_store(
                        connection=connection,
                        expected_chunk_ids={"chunk-1", "chunk-2"},
                        expected_chunk_count=2,
                    )
            finally:
                connection.close()

    def test_build_faiss_index_detects_vector_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            connection = connect_embedding_store(temp_path / "embeddings.sqlite")
            try:
                insert_embedding_row(connection, row_index=0, chunk_id="chunk-1", vector=[1.0, 0.0])
                insert_embedding_row(connection, row_index=1, chunk_id="chunk-2", vector=[0.0, 1.0])
            finally:
                connection.close()

            fake_faiss = FakeFaissModule(count_vectors=False)
            with self.assertRaises(EmbeddingIndexBuildError):
                build_faiss_index_from_store(
                    embedding_store_path=temp_path / "embeddings.sqlite",
                    faiss_index_path=temp_path / "index.faiss",
                    embedding_dimension=2,
                    expected_embedding_count=2,
                    batch_size=2,
                    faiss_module=fake_faiss,
                    numpy_module=FakeNumpy(),
                )


if __name__ == "__main__":
    unittest.main()
