from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PART2_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PART2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part2_rag.chunk_generation import (
    ChunkGenerationError,
    build_chunks,
    chunk_document,
    diagnose_missing_documents,
)


def build_chunk_test_db(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL UNIQUE,
                subreddit TEXT NOT NULL,
                author_id TEXT,
                parent_id TEXT,
                link_id TEXT,
                created_utc INTEGER NOT NULL,
                raw_text TEXT,
                clean_text TEXT,
                include_in_modeling INTEGER DEFAULT 1,
                created_at_utc INTEGER
            );
            CREATE TABLE posts (
                post_id TEXT PRIMARY KEY,
                subreddit TEXT NOT NULL,
                author_id TEXT,
                title TEXT,
                selftext TEXT,
                raw_text TEXT,
                clean_text TEXT,
                created_utc INTEGER NOT NULL,
                score INTEGER,
                num_comments INTEGER,
                permalink TEXT,
                url TEXT,
                is_deleted INTEGER DEFAULT 0,
                is_removed INTEGER DEFAULT 0,
                raw_json TEXT NOT NULL,
                ingested_at_utc INTEGER NOT NULL
            );
            CREATE TABLE comments (
                comment_id TEXT PRIMARY KEY,
                post_id TEXT NOT NULL,
                parent_id TEXT,
                subreddit TEXT NOT NULL,
                author_id TEXT,
                body TEXT,
                raw_text TEXT,
                clean_text TEXT,
                created_utc INTEGER NOT NULL,
                score INTEGER,
                depth INTEGER,
                is_deleted INTEGER DEFAULT 0,
                is_removed INTEGER DEFAULT 0,
                raw_json TEXT NOT NULL,
                ingested_at_utc INTEGER NOT NULL
            );
            """
        )

        long_post_text = (
            "Warm up carefully before lifting weights and track your sets for steady progress. "
            "Focus on technique every session so your baseline stays consistent.\n\n"
            "Progressive overload matters when the plan stays realistic and repeatable. "
            "Add small amounts of weight only after the current load feels controlled.\n\n"
            "Recovery habits matter too because sleep and nutrition support adaptation. "
            "Deload when fatigue starts hiding your actual performance."
        )

        connection.executemany(
            """
            INSERT INTO posts (
                post_id, subreddit, author_id, title, selftext, raw_text, clean_text,
                created_utc, raw_json, ingested_at_utc
            )
            VALUES (?, 'fitness', ?, ?, ?, ?, ?, ?, '{}', ?)
            """,
            [
                (
                    "post_main",
                    "author_post",
                    "Training question",
                    "How should I structure this block?",
                    "Training question\n\n" + long_post_text,
                    "Training question\n\n" + long_post_text,
                    1680400000,
                    1680400100,
                ),
                (
                    "post_old",
                    "author_old",
                    "Old window question",
                    "This should never be chunked by the selected manifest.",
                    "Old window question\n\nThis should never be chunked by the selected manifest.",
                    "Old window question\n\nThis should never be chunked by the selected manifest.",
                    1528000000,
                    1528000100,
                ),
            ],
        )

        connection.executemany(
            """
            INSERT INTO comments (
                comment_id, post_id, parent_id, subreddit, author_id, body, raw_text,
                clean_text, created_utc, raw_json, ingested_at_utc
            )
            VALUES (?, ?, ?, 'fitness', ?, ?, ?, ?, ?, '{}', ?)
            """,
            [
                (
                    "comment_main",
                    "post_main",
                    "t3_post_main",
                    "author_comment",
                    "Short answer with one clear recommendation.",
                    "Short answer with one clear recommendation.",
                    "Short answer with one clear recommendation.",
                    1680400200,
                    1680400300,
                ),
                (
                    "comment_boundary",
                    "post_main",
                    "t3_post_main",
                    "boundary_author",
                    "At the exclusive boundary and should not be included.",
                    "At the exclusive boundary and should not be included.",
                    "At the exclusive boundary and should not be included.",
                    1714521600,
                    1714521700,
                ),
                (
                    "comment_filtered",
                    "post_main",
                    "t3_post_main",
                    "filtered_author",
                    "This comment is excluded from Part 1 modeling but should remain in RAG chunking.",
                    "This comment is excluded from Part 1 modeling but should remain in RAG chunking.",
                    "This comment is excluded from Part 1 modeling but should remain in RAG chunking.",
                    1680400250,
                    1680400350,
                ),
            ],
        )

        connection.executemany(
            """
            INSERT INTO documents (
                document_id, source_type, source_id, subreddit, author_id, parent_id,
                link_id, created_utc, raw_text, clean_text, include_in_modeling, created_at_utc
            )
            VALUES (?, ?, ?, 'fitness', ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            [
                (
                    "doc_post_main",
                    "post",
                    "post_main",
                    "author_post",
                    None,
                    "t3_post_main",
                    1680400000,
                    "Training question\n\n" + long_post_text,
                    "Training question\n\n" + long_post_text,
                    1680400100,
                ),
                (
                    "doc_comment_main",
                    "comment",
                    "comment_main",
                    "author_comment",
                    "t3_post_main",
                    "t3_post_main",
                    1680400200,
                    "Short answer with one clear recommendation.",
                    "Short answer with one clear recommendation.",
                    1680400300,
                ),
                (
                    "doc_comment_filtered_in_part1",
                    "comment",
                    "comment_filtered",
                    "filtered_author",
                    "t3_post_main",
                    "t3_post_main",
                    1680400250,
                    "This comment is excluded from Part 1 modeling but should remain in RAG chunking.",
                    "This comment is excluded from Part 1 modeling but should remain in RAG chunking.",
                    1680400350,
                ),
                (
                    "doc_post_old",
                    "post",
                    "post_old",
                    "author_old",
                    None,
                    "t3_post_old",
                    1528000000,
                    "Old window question\n\nThis should never be chunked by the selected manifest.",
                    "Old window question\n\nThis should never be chunked by the selected manifest.",
                    1528000100,
                ),
                (
                    "doc_comment_boundary",
                    "comment",
                    "comment_boundary",
                    "boundary_author",
                    "t3_post_main",
                    "t3_post_main",
                    1714521600,
                    "At the exclusive boundary and should not be included.",
                    "At the exclusive boundary and should not be included.",
                    1714521700,
                ),
            ],
        )
        connection.execute(
            """
            UPDATE documents
            SET include_in_modeling = 0
            WHERE document_id = 'doc_comment_filtered_in_part1'
            """
        )
        connection.commit()
    finally:
        connection.close()


def write_corpus_manifest(manifest_path: Path) -> None:
    manifest = {
        "db_path": "ignored-in-tests",
        "selected_window": {
            "subreddit": "fitness",
            "window_start_utc": 1680307200,
            "window_end_utc": 1714521600,
            "eligible_as_default_rag_corpus": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


class ChunkGenerationTests(unittest.TestCase):
    def test_comment_stays_single_chunk_when_short(self) -> None:
        document = {
            "document_id": "doc_comment",
            "source_type": "comment",
            "source_id": "c1",
            "post_id": "p1",
            "parent_id": "t3_p1",
            "link_id": "t3_p1",
            "created_utc": 1680400200,
            "author_id": "author_comment",
            "title": "Parent post title",
            "text": "Short answer with one clear recommendation.",
        }

        chunks = chunk_document(document, max_chunk_tokens=24, overlap_sentences=1)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].chunk_text, document["text"])
        self.assertEqual(chunks[0].chunk_origin, "text")

    def test_long_post_splits_into_multiple_chunks(self) -> None:
        document = {
            "document_id": "doc_post",
            "source_type": "post",
            "source_id": "p1",
            "post_id": "p1",
            "parent_id": None,
            "link_id": "t3_p1",
            "created_utc": 1680400000,
            "author_id": "author_post",
            "title": "Training question",
            "text": (
                "Sentence one has a solid amount of detail for the test case. "
                "Sentence two adds enough extra words to push the chunk over the limit. "
                "Sentence three keeps the paragraph moving with more context."
            ),
        }

        chunks = chunk_document(document, max_chunk_tokens=16, overlap_sentences=1)

        self.assertGreater(len(chunks), 1)

    def test_boundary_overlap_is_applied(self) -> None:
        document = {
            "document_id": "doc_post",
            "source_type": "post",
            "source_id": "p1",
            "post_id": "p1",
            "parent_id": None,
            "link_id": "t3_p1",
            "created_utc": 1680400000,
            "author_id": "author_post",
            "title": "Training question",
            "text": (
                "Sentence one has plenty of words to make the chunk fairly large. "
                "Sentence two also carries several words for overlap validation. "
                "Sentence three closes the example with extra detail."
            ),
        }

        chunks = chunk_document(document, max_chunk_tokens=24, overlap_sentences=1)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(
            chunks[0].chunk_text.endswith(
                "Sentence two also carries several words for overlap validation."
            )
        )
        self.assertTrue(
            chunks[1].chunk_text.startswith(
                "Sentence two also carries several words for overlap validation."
            )
        )

    def test_metadata_is_preserved_in_output_chunks(self) -> None:
        document = {
            "document_id": "doc_comment",
            "source_type": "comment",
            "source_id": "comment_main",
            "post_id": "post_main",
            "parent_id": "t3_post_main",
            "link_id": "t3_post_main",
            "created_utc": 1680400200,
            "author_id": "author_comment",
            "title": "Training question",
            "text": "Short answer with one clear recommendation.",
        }

        chunk = chunk_document(document, max_chunk_tokens=20, overlap_sentences=1)[0]

        self.assertEqual(chunk.document_id, "doc_comment")
        self.assertEqual(chunk.source_type, "comment")
        self.assertEqual(chunk.source_id, "comment_main")
        self.assertEqual(chunk.post_id, "post_main")
        self.assertEqual(chunk.parent_id, "t3_post_main")
        self.assertEqual(chunk.link_id, "t3_post_main")
        self.assertEqual(chunk.author_id, "author_comment")
        self.assertEqual(chunk.title, "Training question")
        self.assertEqual(chunk.chunk_origin, "text")

    def test_link_only_comment_uses_url_fallback_chunk(self) -> None:
        document = {
            "document_id": "doc_link_only",
            "source_type": "comment",
            "source_id": "comment_link_only",
            "post_id": "post_main",
            "parent_id": "t3_post_main",
            "link_id": "t3_post_main",
            "created_utc": 1680400200,
            "author_id": "author_comment",
            "title": "Training question",
            "text": "",
            "raw_text": "https://thefitness.wiki/muscle-building-101/",
        }

        chunks = chunk_document(document, max_chunk_tokens=220, overlap_sentences=1)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].chunk_text,
            "[source: fitness wiki] muscle building",
        )
        self.assertEqual(chunks[0].chunk_origin, "url_fallback")

    def test_markdown_and_plain_urls_are_combined_up_to_two(self) -> None:
        document = {
            "document_id": "doc_link_mix",
            "source_type": "comment",
            "source_id": "comment_link_mix",
            "post_id": "post_main",
            "parent_id": "t3_post_main",
            "link_id": "t3_post_main",
            "created_utc": 1680400200,
            "author_id": "author_comment",
            "title": "Training question",
            "text": "",
            "raw_text": (
                "[muscle building guide](https://thefitness.wiki/muscle-building-101/) "
                "https://www.strongerbyscience.com/rowing/ "
                "https://example.com/ignored-third-link"
            ),
        }

        chunks = chunk_document(document, max_chunk_tokens=220, overlap_sentences=1)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].chunk_text,
            "[source: fitness wiki] muscle building guide "
            "[source: stronger by science] rowing",
        )
        self.assertEqual(chunks[0].chunk_origin, "url_fallback")

    def test_only_selected_default_window_documents_are_included(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = temp_path / "test.sqlite"
            source_manifest_path = temp_path / "corpus_manifest.json"
            chunk_artifact_path = temp_path / "chunks.jsonl"
            chunk_manifest_path = temp_path / "chunk_manifest.json"
            build_chunk_test_db(db_path)
            write_corpus_manifest(source_manifest_path)

            result = build_chunks(
                db_path=db_path,
                source_corpus_manifest_path=source_manifest_path,
                chunk_artifact_path=chunk_artifact_path,
                chunk_manifest_path=chunk_manifest_path,
                max_chunk_tokens=24,
                overlap_sentences=1,
                resume=False,
            )

            lines = [
                json.loads(line)
                for line in chunk_artifact_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            document_ids = {line["document_id"] for line in lines}

            self.assertIn("doc_post_main", document_ids)
            self.assertIn("doc_comment_main", document_ids)
            self.assertIn("doc_comment_filtered_in_part1", document_ids)
            self.assertNotIn("doc_post_old", document_ids)
            self.assertNotIn("doc_comment_boundary", document_ids)
            self.assertEqual(
                result.source_document_counts["documents_seen_in_selected_window"], 3
            )
            self.assertEqual(result.validation["source_documents_represented"], 3)
            self.assertEqual(result.coverage_rate, 1.0)
            self.assertEqual(result.fallback_chunk_count, 0)
            self.assertEqual(result.fallback_chunk_percentage, 0.0)
            self.assertEqual(
                result.coverage_explanation,
                {"empty_or_deleted": 0, "normalization_loss": 0, "other": 0},
            )

    def test_resume_rejects_incompatible_chunking_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = temp_path / "test.sqlite"
            source_manifest_path = temp_path / "corpus_manifest.json"
            chunk_artifact_path = temp_path / "chunks.jsonl"
            chunk_manifest_path = temp_path / "chunk_manifest.json"
            build_chunk_test_db(db_path)
            write_corpus_manifest(source_manifest_path)

            build_chunks(
                db_path=db_path,
                source_corpus_manifest_path=source_manifest_path,
                chunk_artifact_path=chunk_artifact_path,
                chunk_manifest_path=chunk_manifest_path,
                max_chunk_tokens=24,
                overlap_sentences=1,
                resume=False,
            )

            with self.assertRaises(ChunkGenerationError):
                build_chunks(
                    db_path=db_path,
                    source_corpus_manifest_path=source_manifest_path,
                    chunk_artifact_path=chunk_artifact_path,
                    chunk_manifest_path=chunk_manifest_path,
                    max_chunk_tokens=32,
                    overlap_sentences=1,
                    resume=True,
                )

    def test_missing_document_diagnostics_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = temp_path / "test.sqlite"
            source_manifest_path = temp_path / "corpus_manifest.json"
            build_chunk_test_db(db_path)
            write_corpus_manifest(source_manifest_path)

            connection = sqlite3.connect(db_path)
            try:
                connection.execute(
                    """
                    INSERT INTO comments (
                        comment_id, post_id, parent_id, subreddit, author_id, body, raw_text,
                        clean_text, created_utc, raw_json, ingested_at_utc, is_deleted
                    )
                    VALUES (?, ?, ?, 'fitness', ?, ?, ?, ?, ?, '{}', ?, ?)
                    """,
                    (
                        "comment_deleted",
                        "post_main",
                        "t3_post_main",
                        "deleted_author",
                        "",
                        "",
                        "",
                        1680400400,
                        1680400500,
                        1,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO comments (
                        comment_id, post_id, parent_id, subreddit, author_id, body, raw_text,
                        clean_text, created_utc, raw_json, ingested_at_utc
                    )
                    VALUES (?, ?, ?, 'fitness', ?, ?, ?, ?, ?, '{}', ?)
                    """,
                    (
                        "comment_empty",
                        "post_main",
                        "t3_post_main",
                        "empty_author",
                        "",
                        "",
                        "",
                        1680400410,
                        1680400510,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO documents (
                        document_id, source_type, source_id, subreddit, author_id, parent_id,
                        link_id, created_utc, raw_text, clean_text, include_in_modeling, created_at_utc
                    )
                    VALUES (?, 'comment', ?, 'fitness', ?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    [
                        (
                            "doc_deleted",
                            "comment_deleted",
                            "deleted_author",
                            "t3_post_main",
                            "t3_post_main",
                            1680400400,
                            "",
                            "",
                            1680400500,
                        ),
                        (
                            "doc_empty_raw",
                            "comment_empty",
                            "empty_author",
                            "t3_post_main",
                            "t3_post_main",
                            1680400410,
                            "",
                            "",
                            1680400510,
                        ),
                    ],
                )
                connection.commit()
            finally:
                connection.close()

            diagnostics = diagnose_missing_documents(
                db_path=db_path,
                source_corpus_manifest_path=source_manifest_path,
                max_chunk_tokens=24,
                overlap_sentences=1,
            )

            self.assertEqual(diagnostics.counts["deleted_or_removed"], 1)
            self.assertEqual(diagnostics.counts["empty_raw_text"], 1)
            self.assertEqual(diagnostics.counts["normalized_to_empty"], 0)
            self.assertEqual(diagnostics.counts["tokenization_failed"], 0)
            self.assertEqual(diagnostics.counts["other"], 0)
            self.assertEqual(diagnostics.sample_ids["deleted_or_removed"], ["doc_deleted"])
            self.assertEqual(diagnostics.sample_ids["empty_raw_text"], ["doc_empty_raw"])


if __name__ == "__main__":
    unittest.main()
