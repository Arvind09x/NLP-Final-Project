from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import pandas as pd

from fitness_reddit_analyzer.config import (
    MIN_COMMENT_TEXT_CHARS,
    MIN_COMMENT_TOKEN_COUNT,
    MIN_POST_TEXT_CHARS,
    PROCESSED_DIR,
    SUBREDDIT,
    checkpoint_path,
)
from fitness_reddit_analyzer.db import connect_db


TOKEN_RE = re.compile(r"\b\w+\b")


@dataclass
class FeatureSummary:
    stage: str
    generated_at_utc: int
    subreddit: str
    posts_total: int
    posts_model_ready: int
    comments_total: int
    comments_model_ready: int
    comments_substantive: int
    bot_authors: int
    deleted_or_removed_posts: int
    deleted_or_removed_comments: int


def run() -> None:
    checkpoint_file = checkpoint_path("prepare_features")
    ensure_ingestion_complete()
    posts_frame = load_posts_frame()
    comments_frame = load_comments_frame()
    apply_modeling_flags(posts_frame, comments_frame)
    feature_summary = build_summary(posts_frame, comments_frame)
    write_feature_outputs(posts_frame, comments_frame, feature_summary)
    save_checkpoint(checkpoint_file, feature_summary)
    print("prepare_features complete:", json.dumps(asdict(feature_summary), ensure_ascii=True))


def ensure_ingestion_complete() -> None:
    with connect_db() as connection:
        post_checkpoint = connection.execute(
            "SELECT status FROM pipeline_checkpoints WHERE stage_name = 'ingest_posts'"
        ).fetchone()
        comment_checkpoint = connection.execute(
            "SELECT status FROM pipeline_checkpoints WHERE stage_name = 'ingest_comments'"
        ).fetchone()
    if post_checkpoint is None or post_checkpoint["status"] != "completed":
        raise RuntimeError("Run ingest_posts successfully before prepare_features.")
    if comment_checkpoint is None or comment_checkpoint["status"] != "completed":
        raise RuntimeError("Run ingest_comments successfully before prepare_features.")


def load_posts_frame() -> pd.DataFrame:
    with connect_db() as connection:
        frame = pd.read_sql_query(
            """
            SELECT
                p.post_id,
                p.author_id,
                a.username,
                a.pseudonym,
                COALESCE(a.is_probable_bot, 0) AS is_probable_bot,
                p.title,
                p.selftext,
                p.raw_text,
                p.clean_text,
                p.created_utc,
                p.score,
                p.num_comments,
                p.permalink,
                p.url,
                p.is_deleted,
                p.is_removed
            FROM posts p
            LEFT JOIN authors a ON a.author_id = p.author_id
            WHERE LOWER(p.subreddit) = LOWER(?)
            ORDER BY p.created_utc ASC, p.post_id ASC
            """,
            connection,
            params=(SUBREDDIT,),
        )
    return frame.fillna(
        {
            "username": "[unknown]",
            "pseudonym": "user_unknown",
            "title": "",
            "selftext": "",
            "raw_text": "",
            "clean_text": "",
            "score": 0,
            "num_comments": 0,
            "permalink": "",
            "url": "",
            "is_probable_bot": 0,
            "is_deleted": 0,
            "is_removed": 0,
        }
    )


def load_comments_frame() -> pd.DataFrame:
    with connect_db() as connection:
        frame = pd.read_sql_query(
            """
            SELECT
                c.comment_id,
                c.post_id,
                c.parent_id,
                c.author_id,
                a.username,
                a.pseudonym,
                COALESCE(a.is_probable_bot, 0) AS is_probable_bot,
                c.body,
                c.raw_text,
                c.clean_text,
                c.created_utc,
                c.score,
                c.depth,
                c.is_deleted,
                c.is_removed
            FROM comments c
            LEFT JOIN authors a ON a.author_id = c.author_id
            WHERE LOWER(c.subreddit) = LOWER(?)
            ORDER BY c.created_utc ASC, c.comment_id ASC
            """,
            connection,
            params=(SUBREDDIT,),
        )
    return frame.fillna(
        {
            "username": "[unknown]",
            "pseudonym": "user_unknown",
            "body": "",
            "raw_text": "",
            "clean_text": "",
            "score": 0,
            "depth": 0,
            "is_probable_bot": 0,
            "is_deleted": 0,
            "is_removed": 0,
        }
    )


def apply_modeling_flags(posts_frame: pd.DataFrame, comments_frame: pd.DataFrame) -> None:
    posts_frame["clean_char_count"] = posts_frame["clean_text"].astype(str).str.len()
    posts_frame["token_count"] = posts_frame["clean_text"].astype(str).map(token_count)
    posts_frame["include_in_modeling"] = (
        (posts_frame["is_deleted"] == 0)
        & (posts_frame["is_removed"] == 0)
        & (posts_frame["clean_char_count"] >= MIN_POST_TEXT_CHARS)
    ).astype(int)
    posts_frame["post_month"] = pd.to_datetime(posts_frame["created_utc"], unit="s", utc=True).dt.strftime("%Y-%m-01")

    comments_frame["clean_char_count"] = comments_frame["clean_text"].astype(str).str.len()
    comments_frame["token_count"] = comments_frame["clean_text"].astype(str).map(token_count)
    comments_frame["is_substantive"] = (
        (comments_frame["clean_char_count"] >= MIN_COMMENT_TEXT_CHARS)
        & (comments_frame["token_count"] >= MIN_COMMENT_TOKEN_COUNT)
    ).astype(int)
    comments_frame["include_in_modeling"] = (
        (comments_frame["is_deleted"] == 0)
        & (comments_frame["is_removed"] == 0)
        & (comments_frame["is_probable_bot"] == 0)
        & (comments_frame["is_substantive"] == 1)
    ).astype(int)

    update_documents_table(posts_frame, comments_frame)


def update_documents_table(posts_frame: pd.DataFrame, comments_frame: pd.DataFrame) -> None:
    with connect_db() as connection:
        connection.executemany(
            "UPDATE documents SET include_in_modeling = ? WHERE document_id = ?",
            [
                (int(row.include_in_modeling), f"post_{row.post_id}")
                for row in posts_frame.itertuples(index=False)
            ],
        )
        connection.executemany(
            "UPDATE documents SET include_in_modeling = ? WHERE document_id = ?",
            [
                (int(row.include_in_modeling), f"comment_{row.comment_id}")
                for row in comments_frame.itertuples(index=False)
            ],
        )


def build_summary(posts_frame: pd.DataFrame, comments_frame: pd.DataFrame) -> FeatureSummary:
    now_utc = int(datetime.now(tz=UTC).timestamp())
    return FeatureSummary(
        stage="prepare_features",
        generated_at_utc=now_utc,
        subreddit=SUBREDDIT,
        posts_total=int(len(posts_frame)),
        posts_model_ready=int(posts_frame["include_in_modeling"].sum()),
        comments_total=int(len(comments_frame)),
        comments_model_ready=int(comments_frame["include_in_modeling"].sum()),
        comments_substantive=int(comments_frame["is_substantive"].sum()),
        bot_authors=count_bot_authors(),
        deleted_or_removed_posts=int(((posts_frame["is_deleted"] == 1) | (posts_frame["is_removed"] == 1)).sum()),
        deleted_or_removed_comments=int(
            ((comments_frame["is_deleted"] == 1) | (comments_frame["is_removed"] == 1)).sum()
        ),
    )


def count_bot_authors() -> int:
    with connect_db() as connection:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM authors WHERE is_probable_bot = 1"
            ).fetchone()[0]
        )


def write_feature_outputs(
    posts_frame: pd.DataFrame,
    comments_frame: pd.DataFrame,
    feature_summary: FeatureSummary,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    posts_output = posts_frame[
        [
            "post_id",
            "author_id",
            "username",
            "pseudonym",
            "created_utc",
            "post_month",
            "title",
            "selftext",
            "clean_text",
            "score",
            "num_comments",
            "token_count",
            "clean_char_count",
            "is_probable_bot",
            "is_deleted",
            "is_removed",
            "include_in_modeling",
            "permalink",
            "url",
        ]
    ]
    comments_output = comments_frame[
        [
            "comment_id",
            "post_id",
            "parent_id",
            "author_id",
            "username",
            "pseudonym",
            "created_utc",
            "body",
            "clean_text",
            "score",
            "depth",
            "token_count",
            "clean_char_count",
            "is_substantive",
            "is_probable_bot",
            "is_deleted",
            "is_removed",
            "include_in_modeling",
        ]
    ]
    posts_output.to_parquet(PROCESSED_DIR / "modeling_posts.parquet", index=False)
    comments_output.to_parquet(PROCESSED_DIR / "modeling_comments.parquet", index=False)
    summary_path = PROCESSED_DIR / "feature_summary.json"
    summary_path.write_text(json.dumps(asdict(feature_summary), indent=2), encoding="utf-8")


def save_checkpoint(path, feature_summary: FeatureSummary) -> None:
    payload = asdict(feature_summary)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with connect_db() as connection:
        connection.execute(
            """
            INSERT INTO pipeline_checkpoints (stage_name, status, payload_json, updated_at_utc)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(stage_name) DO UPDATE SET
                status = excluded.status,
                payload_json = excluded.payload_json,
                updated_at_utc = excluded.updated_at_utc
            """,
            (
                "prepare_features",
                "completed",
                json.dumps(payload, ensure_ascii=True),
                feature_summary.generated_at_utc,
            ),
        )


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text.lower()))
