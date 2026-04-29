from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from fitness_reddit_analyzer.config import DB_PATH, ensure_directories


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS subreddit_meta (
        subreddit TEXT NOT NULL,
        window_start_utc INTEGER NOT NULL,
        window_end_utc INTEGER,
        selected_at_utc INTEGER,
        post_count INTEGER DEFAULT 0,
        comment_count INTEGER DEFAULT 0,
        notes TEXT,
        PRIMARY KEY (subreddit, window_start_utc)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS authors (
        author_id TEXT PRIMARY KEY,
        username TEXT,
        pseudonym TEXT,
        is_deleted INTEGER DEFAULT 0,
        is_probable_bot INTEGER DEFAULT 0,
        created_at_utc INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS posts (
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
        ingested_at_utc INTEGER NOT NULL,
        FOREIGN KEY(author_id) REFERENCES authors(author_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS comments (
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
        ingested_at_utc INTEGER NOT NULL,
        FOREIGN KEY(post_id) REFERENCES posts(post_id),
        FOREIGN KEY(author_id) REFERENCES authors(author_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS documents (
        document_id TEXT PRIMARY KEY,
        source_type TEXT NOT NULL CHECK(source_type IN ('post', 'comment')),
        source_id TEXT NOT NULL UNIQUE,
        subreddit TEXT NOT NULL,
        author_id TEXT,
        parent_id TEXT,
        link_id TEXT,
        created_utc INTEGER NOT NULL,
        raw_text TEXT,
        clean_text TEXT,
        include_in_modeling INTEGER DEFAULT 1,
        created_at_utc INTEGER,
        FOREIGN KEY(author_id) REFERENCES authors(author_id)
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
        document_id UNINDEXED,
        clean_text
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS document_embeddings (
        document_id TEXT PRIMARY KEY,
        embedding_model TEXT,
        embedding_dim INTEGER,
        embedding_blob BLOB,
        created_at_utc INTEGER,
        FOREIGN KEY(document_id) REFERENCES documents(document_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS topic_definitions (
        topic_id INTEGER PRIMARY KEY,
        topic_label TEXT NOT NULL,
        topic_label_auto TEXT,
        top_keywords_json TEXT NOT NULL,
        share_of_posts REAL,
        topic_type TEXT,
        month_coverage_ratio REAL,
        peak_to_median_ratio REAL,
        representative_posts_json TEXT,
        notes TEXT,
        created_at_utc INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS document_topics (
        document_id TEXT NOT NULL,
        topic_id INTEGER NOT NULL,
        assignment_source TEXT NOT NULL,
        assignment_confidence REAL,
        is_major_topic INTEGER DEFAULT 0,
        PRIMARY KEY (document_id, topic_id),
        FOREIGN KEY(document_id) REFERENCES documents(document_id),
        FOREIGN KEY(topic_id) REFERENCES topic_definitions(topic_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS topic_time_series (
        topic_id INTEGER NOT NULL,
        month_start TEXT NOT NULL,
        post_count INTEGER NOT NULL,
        post_share REAL NOT NULL,
        PRIMARY KEY (topic_id, month_start),
        FOREIGN KEY(topic_id) REFERENCES topic_definitions(topic_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS comment_stances (
        comment_id TEXT PRIMARY KEY,
        topic_id INTEGER NOT NULL,
        stance_label TEXT NOT NULL,
        stance_confidence REAL,
        cluster_id INTEGER,
        prototype_distance REAL,
        is_substantive INTEGER DEFAULT 1,
        created_at_utc INTEGER NOT NULL,
        FOREIGN KEY(comment_id) REFERENCES comments(comment_id),
        FOREIGN KEY(topic_id) REFERENCES topic_definitions(topic_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS topic_argument_summaries (
        topic_id INTEGER NOT NULL,
        stance_label TEXT NOT NULL,
        summary_text TEXT NOT NULL,
        representative_comment_ids_json TEXT,
        generated_at_utc INTEGER NOT NULL,
        PRIMARY KEY (topic_id, stance_label),
        FOREIGN KEY(topic_id) REFERENCES topic_definitions(topic_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
        stage_name TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        payload_json TEXT,
        updated_at_utc INTEGER NOT NULL
    )
    """,
)


@contextmanager
def connect_db() -> Iterator[sqlite3.Connection]:
    ensure_directories()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database() -> None:
    with connect_db() as connection:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
