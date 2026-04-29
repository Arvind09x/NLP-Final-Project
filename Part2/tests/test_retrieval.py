from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest.mock import patch


PART2_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PART2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part2_rag.retrieval import (
    RetrievalConfig,
    RetrievalError,
    RetrievalResult,
    dense_retrieve,
    extract_query_terms,
    fetch_chunk_ids_for_row_positions,
    format_retrieval_results,
    hybrid_retrieve,
    lexical_retrieve,
    load_frozen_chunk_catalog,
    merge_results_with_rrf,
)


class FakeVectorRow(list):
    def tolist(self) -> list[float]:
        return list(self)


class FakeMatrix(list):
    pass


class FakeFaissIndex:
    def __init__(self, ntotal: int, scores: list[float], indices: list[int]) -> None:
        self.d = 384
        self.ntotal = ntotal
        self._scores = scores
        self._indices = indices

    def search(self, query_vector: object, top_k: int) -> tuple[FakeMatrix, FakeMatrix]:
        del query_vector
        return (
            FakeMatrix([FakeVectorRow(self._scores[:top_k])]),
            FakeMatrix([FakeVectorRow(self._indices[:top_k])]),
        )


class FakeFaissModule:
    def __init__(self, index: FakeFaissIndex) -> None:
        self._index = index

    def read_index(self, path: str) -> FakeFaissIndex:
        del path
        return self._index


class FakeModel:
    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]


def write_chunk_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_embedding_manifest(path: Path, *, embedding_count: int, store_path: Path, faiss_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "batch_size": 8,
                "chunk_artifact_path": "unused",
                "chunk_artifact_hash": "unused",
                "chunk_manifest_path": "unused",
                "chunk_manifest_hash": "unused",
                "expected_chunk_count": embedding_count,
                "embedding_count": embedding_count,
                "embedding_artifact_path": str(store_path),
                "faiss_index_path": str(faiss_path),
                "build_timestamp": "2026-04-25T00:00:00+00:00",
                "resume_status": "fresh",
            }
        ),
        encoding="utf-8",
    )


def build_embedding_store(path: Path, rows: list[tuple[int, str, str]]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE embeddings (
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
        for row_index, chunk_id, document_id in rows:
            connection.execute(
                """
                INSERT INTO embeddings (
                    row_index, chunk_id, document_id, source_type, source_id, post_id,
                    parent_id, link_id, created_utc, author_id, title, chunk_index,
                    token_estimate, chunk_origin, chunk_text, embedding
                )
                VALUES (?, ?, ?, 'post', ?, NULL, NULL, NULL, 1680307200, NULL, ?, 0, 10, 'text', ?, ?)
                """,
                (
                    row_index,
                    chunk_id,
                    document_id,
                    document_id,
                    document_id,
                    f"text for {chunk_id}",
                    sqlite3.Binary(array("f", [0.1, 0.2, 0.3]).tobytes()),
                ),
            )
        connection.commit()
    finally:
        connection.close()


def build_part1_fts_db(path: Path, documents: list[tuple[str, str]]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                clean_text TEXT
            );
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                document_id UNINDEXED,
                clean_text
            );
            """
        )
        connection.executemany(
            "INSERT INTO documents(document_id, clean_text) VALUES (?, ?)",
            documents,
        )
        connection.executemany(
            "INSERT INTO documents_fts(document_id, clean_text) VALUES (?, ?)",
            documents,
        )
        connection.commit()
    finally:
        connection.close()


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        load_frozen_chunk_catalog.cache_clear()

    def _make_config(
        self,
        temp_path: Path,
        *,
        chunk_rows: list[dict[str, object]],
        embedding_rows: list[tuple[int, str, str]] | None = None,
        embedding_count: int = 0,
        documents: list[tuple[str, str]] | None = None,
    ) -> RetrievalConfig:
        chunk_artifact_path = temp_path / "chunks.jsonl"
        write_chunk_artifact(chunk_artifact_path, chunk_rows)

        embedding_store_path = temp_path / "embeddings.sqlite"
        faiss_index_path = temp_path / "index.faiss"
        faiss_index_path.write_text("placeholder", encoding="utf-8")
        embedding_manifest_path = temp_path / "embedding_manifest.json"
        if embedding_rows is not None:
            build_embedding_store(embedding_store_path, embedding_rows)
        write_embedding_manifest(
            embedding_manifest_path,
            embedding_count=embedding_count,
            store_path=embedding_store_path,
            faiss_path=faiss_index_path,
        )

        chunk_manifest_path = temp_path / "chunk_manifest.json"
        chunk_manifest_path.write_text(
            json.dumps({"chunk_counts": {"total_chunks": len(chunk_rows)}}),
            encoding="utf-8",
        )
        corpus_manifest_path = temp_path / "corpus_manifest.json"
        corpus_manifest_path.write_text(json.dumps({"selected_window": {}}), encoding="utf-8")

        part1_db_path = temp_path / "part1.sqlite"
        if documents is not None:
            build_part1_fts_db(part1_db_path, documents)
        else:
            build_part1_fts_db(part1_db_path, [])

        return RetrievalConfig(
            dense_top_k=4,
            lexical_top_k=4,
            hybrid_final_top_k=3,
            rrf_constant=60,
            faiss_index_path=faiss_index_path,
            embedding_store_path=embedding_store_path,
            chunk_artifact_path=chunk_artifact_path,
            corpus_manifest_path=corpus_manifest_path,
            chunk_manifest_path=chunk_manifest_path,
            embedding_manifest_path=embedding_manifest_path,
            part1_db_path=part1_db_path,
        )

    def test_dense_retrieve_returns_expected_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = self._make_config(
                temp_path,
                chunk_rows=[
                    {
                        "chunk_id": "chunk-a",
                        "document_id": "doc-a",
                        "source_type": "post",
                        "source_id": "source-a",
                        "post_id": "post-a",
                        "parent_id": None,
                        "link_id": "t3_post-a",
                        "created_utc": 1680307200,
                        "author_id": "author-a",
                        "title": "Alpha title",
                        "chunk_text": "Alpha text for dense retrieval.",
                        "chunk_index": 0,
                        "token_estimate": 5,
                        "chunk_origin": "text",
                    },
                    {
                        "chunk_id": "chunk-b",
                        "document_id": "doc-b",
                        "source_type": "comment",
                        "source_id": "source-b",
                        "post_id": "post-b",
                        "parent_id": "t3_post-b",
                        "link_id": "t3_post-b",
                        "created_utc": 1680307300,
                        "author_id": "author-b",
                        "title": "Beta title",
                        "chunk_text": "Beta text for dense retrieval.",
                        "chunk_index": 0,
                        "token_estimate": 5,
                        "chunk_origin": "text",
                    },
                ],
                embedding_rows=[
                    (0, "chunk-a", "doc-a"),
                    (1, "chunk-b", "doc-b"),
                ],
                embedding_count=2,
            )

            results = dense_retrieve(
                "strength routine",
                config=config,
                top_k=2,
                model=FakeModel(),
                faiss_module=FakeFaissModule(FakeFaissIndex(2, [0.95, 0.75], [1, 0])),
            )

            self.assertEqual([result.chunk_id for result in results], ["chunk-b", "chunk-a"])
            self.assertEqual(results[0].document_id, "doc-b")
            self.assertEqual(results[0].retrieval_source, "dense")
            self.assertEqual(results[0].rank, 1)
            self.assertEqual(results[0].dense_rank, 1)
            self.assertIsNone(results[0].lexical_rank)
            self.assertIn("Beta text", results[0].snippet)

    def test_dense_retrieve_raises_on_manifest_index_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = self._make_config(
                temp_path,
                chunk_rows=[
                    {
                        "chunk_id": "chunk-a",
                        "document_id": "doc-a",
                        "source_type": "post",
                        "source_id": "source-a",
                        "post_id": "post-a",
                        "parent_id": None,
                        "link_id": "t3_post-a",
                        "created_utc": 1680307200,
                        "author_id": "author-a",
                        "title": "Alpha title",
                        "chunk_text": "Alpha text",
                        "chunk_index": 0,
                        "token_estimate": 5,
                        "chunk_origin": "text",
                    }
                ],
                embedding_rows=[(0, "chunk-a", "doc-a")],
                embedding_count=1,
            )

            with self.assertRaises(RetrievalError):
                dense_retrieve(
                    "strength routine",
                    config=config,
                    top_k=1,
                    model=FakeModel(),
                    faiss_module=FakeFaissModule(FakeFaissIndex(2, [0.95], [0])),
                )

    def test_dense_retrieve_loads_and_encodes_before_importing_faiss(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = self._make_config(
                temp_path,
                chunk_rows=[
                    {
                        "chunk_id": "chunk-a",
                        "document_id": "doc-a",
                        "source_type": "post",
                        "source_id": "source-a",
                        "post_id": "post-a",
                        "parent_id": None,
                        "link_id": "t3_post-a",
                        "created_utc": 1680307200,
                        "author_id": "author-a",
                        "title": "Alpha title",
                        "chunk_text": "Alpha text",
                        "chunk_index": 0,
                        "token_estimate": 5,
                        "chunk_origin": "text",
                    }
                ],
                embedding_rows=[(0, "chunk-a", "doc-a")],
                embedding_count=1,
            )
            call_order: list[str] = []

            class OrderedModel(FakeModel):
                def encode(self, texts: list[str], **_: object) -> list[list[float]]:
                    call_order.append("encode")
                    return super().encode(texts, **_)

            def fake_loader(model_name: str) -> OrderedModel:
                del model_name
                call_order.append("load_model")
                return OrderedModel()

            def fake_require_faiss() -> FakeFaissModule:
                call_order.append("require_faiss")
                return FakeFaissModule(FakeFaissIndex(1, [0.95], [0]))

            with patch("part2_rag.retrieval.load_sentence_transformer", side_effect=fake_loader):
                with patch("part2_rag.retrieval.require_faiss", side_effect=fake_require_faiss):
                    results = dense_retrieve("strength routine", config=config, top_k=1)

            self.assertEqual(len(results), 1)
            self.assertEqual(call_order, ["load_model", "encode", "require_faiss"])

    def test_fetch_chunk_ids_for_row_positions_uses_embedding_store_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "embeddings.sqlite"
            build_embedding_store(
                db_path,
                [
                    (0, "chunk-a", "doc-a"),
                    (1, "chunk-b", "doc-b"),
                    (2, "chunk-c", "doc-c"),
                ],
            )
            connection = sqlite3.connect(db_path)
            connection.row_factory = sqlite3.Row
            try:
                mapping = fetch_chunk_ids_for_row_positions(connection, [2, 0])
            finally:
                connection.close()

            self.assertEqual(mapping, {2: "chunk-c", 0: "chunk-a"})

    def test_lexical_retrieve_excludes_documents_not_present_in_frozen_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = self._make_config(
                temp_path,
                chunk_rows=[
                    {
                        "chunk_id": "chunk-in-frozen",
                        "document_id": "doc-in-frozen",
                        "source_type": "post",
                        "source_id": "source-a",
                        "post_id": "post-a",
                        "parent_id": None,
                        "link_id": "t3_post-a",
                        "created_utc": 1680307200,
                        "author_id": "author-a",
                        "title": "Frozen title",
                        "chunk_text": "Protein routine for strength gains.",
                        "chunk_index": 0,
                        "token_estimate": 6,
                        "chunk_origin": "text",
                    }
                ],
                embedding_count=0,
                documents=[
                    ("doc-not-frozen", "protein routine for strength gains"),
                    ("doc-in-frozen", "protein routine for strength gains"),
                ],
            )

            results = lexical_retrieve("protein routine", config=config, top_k=2)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].document_id, "doc-in-frozen")

    def test_lexical_retrieve_uses_first_chunk_for_multi_chunk_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = self._make_config(
                temp_path,
                chunk_rows=[
                    {
                        "chunk_id": "chunk-second",
                        "document_id": "doc-a",
                        "source_type": "post",
                        "source_id": "source-a",
                        "post_id": "post-a",
                        "parent_id": None,
                        "link_id": "t3_post-a",
                        "created_utc": 1680307201,
                        "author_id": "author-a",
                        "title": "Frozen title",
                        "chunk_text": "Later chunk text.",
                        "chunk_index": 1,
                        "token_estimate": 4,
                        "chunk_origin": "text",
                    },
                    {
                        "chunk_id": "chunk-first",
                        "document_id": "doc-a",
                        "source_type": "post",
                        "source_id": "source-a",
                        "post_id": "post-a",
                        "parent_id": None,
                        "link_id": "t3_post-a",
                        "created_utc": 1680307200,
                        "author_id": "author-a",
                        "title": "Frozen title",
                        "chunk_text": "First chunk text.",
                        "chunk_index": 0,
                        "token_estimate": 4,
                        "chunk_origin": "text",
                    },
                ],
                embedding_count=0,
                documents=[("doc-a", "first chunk text later chunk text")],
            )

            results = lexical_retrieve("chunk", config=config, top_k=1)
            self.assertEqual(results[0].chunk_id, "chunk-first")
            self.assertEqual(results[0].chunk_index, 0)

    def test_extract_query_terms_removes_stopwords_and_preserves_domain_terms(self) -> None:
        self.assertEqual(
            extract_query_terms("What do people think about body recomposition?"),
            ("body", "recomposition"),
        )

    def test_hybrid_retrieve_promotes_direct_replies_over_question_stub(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = self._make_config(
                temp_path,
                chunk_rows=[
                    {
                        "chunk_id": "chunk-question",
                        "document_id": "doc-question",
                        "source_type": "comment",
                        "source_id": "question",
                        "post_id": "post-a",
                        "parent_id": "t3_post-a",
                        "link_id": "t3_post-a",
                        "created_utc": 1680307200,
                        "author_id": "author-a",
                        "title": "Thread",
                        "chunk_text": "How should I warm up for the beginner routine?",
                        "chunk_index": 0,
                        "token_estimate": 10,
                        "chunk_origin": "text",
                    },
                    {
                        "chunk_id": "chunk-answer",
                        "document_id": "doc-answer",
                        "source_type": "comment",
                        "source_id": "answer",
                        "post_id": "post-a",
                        "parent_id": "t1_question",
                        "link_id": "t3_post-a",
                        "created_utc": 1680307201,
                        "author_id": "author-b",
                        "title": "Thread",
                        "chunk_text": "Start with lighter sets and ramp up to your first working set.",
                        "chunk_index": 0,
                        "token_estimate": 12,
                        "chunk_origin": "text",
                    },
                ],
                embedding_count=0,
                documents=[],
            )
            parent_result = RetrievalResult(
                rank=1,
                chunk_id="chunk-question",
                document_id="doc-question",
                source_type="comment",
                chunk_index=0,
                chunk_origin="text",
                title="Thread",
                created_utc=1680307200,
                score=0.03,
                retrieval_source="hybrid",
                snippet="How should I warm up for the beginner routine?",
                dense_rank=1,
                dense_score=0.9,
                lexical_rank=1,
                lexical_score=12.0,
                rrf_score=0.03,
            )

            with patch("part2_rag.retrieval.dense_retrieve", return_value=[parent_result]):
                with patch("part2_rag.retrieval.lexical_retrieve", return_value=[]):
                    results = hybrid_retrieve(
                        "warm up beginner routine lighter sets working set",
                        config=config,
                    )

            self.assertEqual(results[0].document_id, "doc-answer")
            self.assertEqual(results[0].retrieval_source, "hybrid+reply")

    def test_merge_results_with_rrf_combines_component_ranks(self) -> None:
        dense_results = [
            RetrievalResult(
                rank=1,
                chunk_id="chunk-a",
                document_id="doc-a",
                source_type="post",
                chunk_index=0,
                chunk_origin="text",
                title="Doc A",
                created_utc=1,
                score=0.9,
                retrieval_source="dense",
                snippet="Dense A",
                dense_rank=1,
                dense_score=0.9,
            )
        ]
        lexical_results = [
            RetrievalResult(
                rank=1,
                chunk_id="chunk-b",
                document_id="doc-b",
                source_type="post",
                chunk_index=0,
                chunk_origin="text",
                title="Doc B",
                created_utc=2,
                score=12.0,
                retrieval_source="lexical",
                snippet="Lexical B",
                lexical_rank=1,
                lexical_score=12.0,
            ),
            RetrievalResult(
                rank=2,
                chunk_id="chunk-a",
                document_id="doc-a",
                source_type="post",
                chunk_index=0,
                chunk_origin="text",
                title="Doc A",
                created_utc=1,
                score=11.0,
                retrieval_source="lexical",
                snippet="Lexical A",
                lexical_rank=2,
                lexical_score=11.0,
            ),
        ]

        results = merge_results_with_rrf(
            dense_results,
            lexical_results,
            rrf_constant=60,
            final_top_k=2,
        )

        self.assertEqual(results[0].document_id, "doc-a")
        self.assertEqual(results[0].dense_rank, 1)
        self.assertEqual(results[0].lexical_rank, 2)
        self.assertGreater(results[0].rrf_score, results[1].rrf_score)

    def test_merge_results_with_rrf_deduplicates_at_document_level(self) -> None:
        dense_results = [
            RetrievalResult(
                rank=1,
                chunk_id="chunk-a0",
                document_id="doc-a",
                source_type="post",
                chunk_index=0,
                chunk_origin="text",
                title="Doc A",
                created_utc=1,
                score=0.95,
                retrieval_source="dense",
                snippet="Dense A0",
                dense_rank=1,
                dense_score=0.95,
            ),
            RetrievalResult(
                rank=2,
                chunk_id="chunk-a1",
                document_id="doc-a",
                source_type="post",
                chunk_index=1,
                chunk_origin="text",
                title="Doc A",
                created_utc=1,
                score=0.90,
                retrieval_source="dense",
                snippet="Dense A1",
                dense_rank=2,
                dense_score=0.90,
            ),
        ]
        lexical_results: list[RetrievalResult] = []

        results = merge_results_with_rrf(
            dense_results,
            lexical_results,
            rrf_constant=60,
            final_top_k=3,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "chunk-a0")
        self.assertEqual(results[0].document_id, "doc-a")

    def test_format_retrieval_results_includes_snippet_and_scores(self) -> None:
        output = format_retrieval_results(
            "hybrid",
            [
                RetrievalResult(
                    rank=1,
                    chunk_id="chunk-a",
                    document_id="doc-a",
                    source_type="post",
                    chunk_index=0,
                    chunk_origin="text",
                    title="Doc A",
                    created_utc=1,
                    score=0.5,
                    retrieval_source="hybrid",
                    snippet="A useful snippet.",
                    dense_rank=1,
                    dense_score=0.9,
                    lexical_rank=2,
                    lexical_score=11.0,
                    rrf_score=0.5,
                )
            ],
        )

        self.assertIn("Retrieval debug results (hybrid)", output)
        self.assertIn("chunk_id=chunk-a", output)
        self.assertIn("snippet: A useful snippet.", output)


if __name__ == "__main__":
    unittest.main()
