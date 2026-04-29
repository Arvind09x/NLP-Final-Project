from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


PART2_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PART2_ROOT / "src"

import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part2_rag.validate_corpus import CorpusValidationError, validate_corpus


def build_test_db(
    db_path: Path,
    main_window_documents: int = 3,
    main_window_comment_documents: int = 1,
    old_window_post_documents: int = 1,
    old_window_comment_documents: int = 1,
    old_window_meta_comment_count: int = 5,
    old_window_running_checkpoint: bool = True,
    include_boundary_document: bool = False,
) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE posts (id TEXT);
            CREATE TABLE comments (id TEXT);
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
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
            CREATE TABLE documents_fts (document_id TEXT);
            CREATE TABLE document_embeddings (document_id TEXT);
            CREATE TABLE subreddit_meta (
                subreddit TEXT NOT NULL,
                window_start_utc INTEGER NOT NULL,
                window_end_utc INTEGER,
                selected_at_utc INTEGER,
                post_count INTEGER DEFAULT 0,
                comment_count INTEGER DEFAULT 0,
                notes TEXT,
                PRIMARY KEY (subreddit, window_start_utc)
            );
            CREATE TABLE pipeline_checkpoints (
                stage_name TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                payload_json TEXT,
                updated_at_utc INTEGER NOT NULL
            );
            """
        )

        connection.executemany(
            "INSERT INTO posts (id) VALUES (?)",
            [("p1",), ("p2",), ("p3",)],
        )
        connection.executemany(
            "INSERT INTO comments (id) VALUES (?)",
            [("c1",), ("c2",)],
        )
        connection.executemany(
            "INSERT INTO documents_fts (document_id) VALUES (?)",
            [("doc1",), ("doc2",), ("doc3",)],
        )
        connection.executemany(
            """
            INSERT INTO subreddit_meta (
                subreddit, window_start_utc, window_end_utc, selected_at_utc,
                post_count, comment_count, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("fitness", 1680307200, 1714521600, 1775639067, 10, 20, "main"),
                (
                    "fitness",
                    1527811200,
                    1534464000,
                    1776789074,
                    12,
                    old_window_meta_comment_count,
                    "partial",
                ),
            ],
        )
        if old_window_running_checkpoint:
            connection.execute(
                """
                INSERT INTO pipeline_checkpoints (stage_name, status, payload_json, updated_at_utc)
                VALUES (?, ?, ?, ?)
                """,
                (
                    "ingest_comments",
                    "running",
                    json.dumps(
                        {
                            "window_start_utc": 1527811200,
                            "window_end_utc": 1534464000,
                            "status": "running",
                        }
                    ),
                    1776841571,
                ),
            )

        for index in range(main_window_documents):
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, source_type, source_id, subreddit, created_utc
                )
                VALUES (?, 'post', ?, 'fitness', 1681000000)
                """,
                (f"doc-main-{index}", f"source-main-{index}"),
            )

        for index in range(main_window_comment_documents):
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, source_type, source_id, subreddit, created_utc
                )
                VALUES (?, 'comment', ?, 'fitness', 1681000001)
                """,
                (f"doc-main-comment-{index}", f"source-main-comment-{index}"),
            )

        for index in range(old_window_post_documents):
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, source_type, source_id, subreddit, created_utc
                )
                VALUES (?, 'post', ?, 'fitness', 1528000000)
                """,
                (f"doc-old-post-{index}", f"source-old-post-{index}"),
            )

        for index in range(old_window_comment_documents):
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, source_type, source_id, subreddit, created_utc
                )
                VALUES (?, 'comment', ?, 'fitness', 1528000000)
                """,
                (f"doc-old-{index}", f"source-old-{index}"),
            )

        if include_boundary_document:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, source_type, source_id, subreddit, created_utc
                )
                VALUES (
                    'doc-boundary-end',
                    'comment',
                    'source-boundary-end',
                    'fitness',
                    1714521600
                )
                """
            )
        connection.commit()
    finally:
        connection.close()


class ValidateCorpusTests(unittest.TestCase):
    def test_validator_selects_main_completed_window_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = temp_path / "test.sqlite"
            manifest_path = temp_path / "manifest.json"
            build_test_db(db_path)

            result = validate_corpus(db_path=db_path, manifest_path=manifest_path)

            self.assertEqual(result.selected_window["window_start_utc"], 1680307200)
            self.assertEqual(result.selected_window["window_end_utc"], 1714521600)
            self.assertEqual(result.counts["selected_corpus_documents"], 4)
            self.assertTrue(result.selected_window["eligible_as_default_rag_corpus"])
            self.assertTrue(manifest_path.exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["db_path"], str(db_path))
            self.assertEqual(
                manifest["selected_window"]["window_start_utc"], 1680307200
            )
            self.assertEqual(manifest["counts"]["documents"], 6)

    def test_validator_fails_when_selected_corpus_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = temp_path / "test.sqlite"
            manifest_path = temp_path / "manifest.json"
            build_test_db(
                db_path,
                main_window_documents=0,
                main_window_comment_documents=0,
                old_window_post_documents=0,
                old_window_comment_documents=0,
                old_window_meta_comment_count=0,
                old_window_running_checkpoint=False,
            )

            with self.assertRaises(CorpusValidationError):
                validate_corpus(db_path=db_path, manifest_path=manifest_path)

    def test_window_with_running_checkpoint_is_not_default_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = temp_path / "test.sqlite"
            manifest_path = temp_path / "manifest.json"
            build_test_db(
                db_path=db_path,
                main_window_documents=3,
                old_window_comment_documents=2,
                old_window_meta_comment_count=0,
                old_window_running_checkpoint=True,
            )

            result = validate_corpus(db_path=db_path, manifest_path=manifest_path)

            older_window = next(
                window
                for window in result.available_windows
                if window["window_start_utc"] == 1527811200
            )
            self.assertTrue(older_window["has_observed_data"])
            self.assertTrue(older_window["usable_for_exploration"])
            self.assertFalse(older_window["eligible_as_default_rag_corpus"])
            self.assertTrue(older_window["metadata_comment_count_is_stale"])
            self.assertTrue(older_window["running_checkpoint_is_stale"])
            self.assertEqual(older_window["observed_comment_document_count"], 2)
            self.assertIn(
                "manual_or_incomplete_comment_ingest", older_window["advisory_flags"]
            )
            self.assertEqual(result.selected_window["window_start_utc"], 1680307200)

    def test_window_counting_uses_end_exclusive_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = temp_path / "test.sqlite"
            manifest_path = temp_path / "manifest.json"
            build_test_db(db_path=db_path, include_boundary_document=True)

            result = validate_corpus(db_path=db_path, manifest_path=manifest_path)

            self.assertEqual(result.counts["selected_corpus_documents"], 4)
            main_window = next(
                window
                for window in result.available_windows
                if window["window_start_utc"] == 1680307200
            )
            self.assertEqual(main_window["observed_comment_document_count"], 1)


if __name__ == "__main__":
    unittest.main()
