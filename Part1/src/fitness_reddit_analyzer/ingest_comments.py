from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from fitness_reddit_analyzer.arctic import ArcticShiftClient
from fitness_reddit_analyzer.config import RAW_DIR, SUBREDDIT, WINDOW_END_UTC, WINDOW_START_UTC, checkpoint_path
from fitness_reddit_analyzer.db import connect_db
from fitness_reddit_analyzer.ingest_posts import clean_reddit_text, normalize_author, upsert_author
from tqdm import tqdm


@dataclass
class CommentCheckpoint:
    window_start_utc: int
    window_end_utc: int
    mode: str
    last_after_utc: int | None
    page_count: int
    current_post_id: str | None
    current_post_after_utc: int | None
    processed_posts: int
    inserted_comments: int
    status: str


def run() -> None:
    checkpoint_file = checkpoint_path("ingest_comments")
    post_window = load_post_window()
    checkpoint = load_checkpoint(checkpoint_file)
    client = ArcticShiftClient()
    inserted_comments = ingest_comments(client, post_window, checkpoint, checkpoint_file)
    print(
        f"ingest_comments complete: {inserted_comments} rows processed "
        f"for {SUBREDDIT} between {post_window['window_start_utc']} and {post_window['window_end_utc']}"
    )


def load_post_window() -> dict:
    # Prefer explicit env vars so the caller controls which era to ingest.
    if WINDOW_START_UTC and WINDOW_END_UTC:
        return {
            "window_start_utc": int(WINDOW_START_UTC),
            "window_end_utc": int(WINDOW_END_UTC),
        }
    # Backward compat: single-era mode, pick the most recent window.
    with connect_db() as connection:
        row = connection.execute(
            """
            SELECT window_start_utc, window_end_utc
            FROM subreddit_meta
            WHERE LOWER(subreddit) = LOWER(?)
            ORDER BY window_start_utc DESC
            LIMIT 1
            """,
            (SUBREDDIT,),
        ).fetchone()
    if row is None:
        raise RuntimeError("Run ingest_posts before ingest_comments so the subreddit window is defined.")
    return dict(row)


def fetch_target_posts(post_window: dict) -> list[sqlite3.Row]:
    with connect_db() as connection:
        rows = connection.execute(
            """
            SELECT post_id, created_utc
            FROM posts
            WHERE LOWER(subreddit) = LOWER(?)
              AND created_utc >= ?
              AND created_utc < ?
            ORDER BY created_utc ASC, post_id ASC
            """,
            (SUBREDDIT, post_window["window_start_utc"], post_window["window_end_utc"]),
        ).fetchall()
    return rows


def ingest_comments(
    client: ArcticShiftClient,
    post_window: dict,
    checkpoint: dict | None,
    checkpoint_file: Path,
) -> int:
    posts = fetch_target_posts(post_window)
    post_ids = {row["post_id"] for row in posts}
    raw_dump_path = RAW_DIR / f"{SUBREDDIT}_comments_{post_window['window_start_utc']}_{post_window['window_end_utc']}.jsonl"
    window_end_utc = int(post_window["window_end_utc"])
    if checkpoint and checkpoint.get("mode") == "subreddit_stream":
        cursor = checkpoint.get("last_after_utc")
        page_count = int(checkpoint.get("page_count", 0))
        total_inserted = int(checkpoint.get("inserted_comments", 0))
    else:
        cursor = post_window["window_start_utc"]
        page_count = 0
        total_inserted = 0
    progress = tqdm(
        desc="Ingesting comments",
        unit="comment",
        initial=total_inserted,
    )

    try:
        while True:
            page = client.search_comments(
                subreddit=SUBREDDIT,
                after=cursor,
                before=window_end_utc,
            )
            comments = page.items
            if cursor is not None:
                comments = [item for item in comments if int(item.get("created_utc", 0)) > cursor]
            comments = [item for item in comments if int(item.get("created_utc", 0)) < window_end_utc]
            if not comments:
                break
            grouped_comments = group_comments_by_post_id(
                comments,
                allowed_post_ids=post_ids,
            )
            inserted_this_page = 0
            for post_id, post_comments in grouped_comments.items():
                upsert_comments(post_id, post_comments)
                append_raw_comments(raw_dump_path, post_id, post_comments)
                inserted_this_page += len(post_comments)
            total_inserted += inserted_this_page
            page_count += 1
            progress.update(inserted_this_page)
            next_cursor = max(int(comment["created_utc"]) for comment in comments if comment.get("created_utc"))
            checkpoint_payload = CommentCheckpoint(
                window_start_utc=post_window["window_start_utc"],
                window_end_utc=post_window["window_end_utc"],
                mode="subreddit_stream",
                last_after_utc=next_cursor,
                page_count=page_count,
                current_post_id=None,
                current_post_after_utc=None,
                processed_posts=0,
                inserted_comments=total_inserted,
                status="running",
            )
            save_checkpoint(checkpoint_file, checkpoint_payload)
            if next_cursor <= (cursor or 0):
                break
            cursor = next_cursor
    finally:
        progress.close()

    final_payload = CommentCheckpoint(
        window_start_utc=post_window["window_start_utc"],
        window_end_utc=post_window["window_end_utc"],
        mode="subreddit_stream",
        last_after_utc=window_end_utc,
        page_count=page_count,
        current_post_id=None,
        current_post_after_utc=None,
        processed_posts=len(posts),
        inserted_comments=total_inserted,
        status="completed",
    )
    save_checkpoint(checkpoint_file, final_payload)
    update_comment_count(total_inserted, post_window)
    return total_inserted


def group_comments_by_post_id(comments: list[dict], *, allowed_post_ids: set[str]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for comment in comments:
        link_id = str(comment.get("link_id") or "")
        if not link_id.startswith("t3_"):
            continue
        post_id = link_id[3:]
        if post_id not in allowed_post_ids:
            continue
        grouped.setdefault(post_id, []).append(comment)
    return grouped


def ingest_comments_for_post(
    client: ArcticShiftClient,
    post_id: str,
    raw_dump_path: Path,
    *,
    checkpoint_file: Path,
    post_window: dict,
    processed_posts: int,
    inserted_comments_before_post: int,
    after_cursor: int | None = None,
) -> int:
    # Retained for reference while subreddit-stream ingestion is preferred.
    link_id = f"t3_{post_id}"
    cursor = after_cursor
    inserted = 0
    seen_ids: set[str] = set()

    while True:
        page = client.search_comments(link_id=link_id, subreddit=SUBREDDIT, after=cursor)
        comments = page.items
        if cursor is not None:
            comments = [item for item in comments if int(item.get("created_utc", 0)) > cursor]
        comments = [item for item in comments if item.get("id") not in seen_ids]
        if not comments:
            break
        upsert_comments(post_id, comments)
        append_raw_comments(raw_dump_path, post_id, comments)
        inserted += len(comments)
        seen_ids.update(comment["id"] for comment in comments if comment.get("id"))
        next_cursor = max(int(comment["created_utc"]) for comment in comments if comment.get("created_utc"))
        save_checkpoint(
            checkpoint_file,
            CommentCheckpoint(
                window_start_utc=post_window["window_start_utc"],
                window_end_utc=post_window["window_end_utc"],
                mode="per_post",
                last_after_utc=None,
                page_count=processed_posts,
                current_post_id=post_id,
                current_post_after_utc=next_cursor,
                processed_posts=processed_posts,
                inserted_comments=inserted_comments_before_post + inserted,
                status="running",
            ),
        )
        if cursor is not None and next_cursor <= cursor:
            break
        cursor = next_cursor
    return inserted


def upsert_comments(post_id: str, comments: list[dict]) -> None:
    now_utc = int(datetime.now(tz=UTC).timestamp())
    with connect_db() as connection:
        for comment in comments:
            author_id, username, is_deleted = normalize_author(comment.get("author"))
            upsert_author(connection, author_id, username, is_deleted, now_utc)
            raw_text = comment.get("body") or ""
            clean_text = clean_reddit_text(raw_text)
            parent_id = comment.get("parent_id")
            link_id = comment.get("link_id") or f"t3_{post_id}"
            body_marker = (comment.get("body") or "").strip().lower()
            comment_is_deleted = int(body_marker == "[deleted]" or is_deleted == 1)
            comment_is_removed = int(
                body_marker == "[removed]"
                or bool(comment.get("collapsed_reason_code") == "deleted")
                or bool(comment.get("collapsed_reason"))
            )
            include_in_modeling = int(
                comment_is_deleted == 0
                and comment_is_removed == 0
                and clean_text.strip() != ""
            )
            connection.execute(
                """
                INSERT INTO comments (
                    comment_id, post_id, parent_id, subreddit, author_id, body, raw_text, clean_text,
                    created_utc, score, depth, is_deleted, is_removed, raw_json, ingested_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(comment_id) DO UPDATE SET
                    post_id = excluded.post_id,
                    parent_id = excluded.parent_id,
                    author_id = excluded.author_id,
                    body = excluded.body,
                    raw_text = excluded.raw_text,
                    clean_text = excluded.clean_text,
                    score = excluded.score,
                    depth = excluded.depth,
                    is_deleted = excluded.is_deleted,
                    is_removed = excluded.is_removed,
                    raw_json = excluded.raw_json,
                    ingested_at_utc = excluded.ingested_at_utc
                """,
                (
                    comment["id"],
                    post_id,
                    parent_id,
                    comment.get("subreddit", SUBREDDIT),
                    author_id,
                    comment.get("body", ""),
                    raw_text,
                    clean_text,
                    int(comment["created_utc"]),
                    comment.get("score"),
                    comment.get("depth"),
                    comment_is_deleted,
                    comment_is_removed,
                    json.dumps(comment, ensure_ascii=True),
                    now_utc,
                ),
            )
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, source_type, source_id, subreddit, author_id, parent_id, link_id,
                    created_utc, raw_text, clean_text, include_in_modeling, created_at_utc
                ) VALUES (?, 'comment', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    author_id = excluded.author_id,
                    parent_id = excluded.parent_id,
                    link_id = excluded.link_id,
                    raw_text = excluded.raw_text,
                    clean_text = excluded.clean_text,
                    include_in_modeling = excluded.include_in_modeling
                """,
                (
                    f"comment_{comment['id']}",
                    comment["id"],
                    comment.get("subreddit", SUBREDDIT),
                    author_id,
                    parent_id,
                    link_id,
                    int(comment["created_utc"]),
                    raw_text,
                    clean_text,
                    include_in_modeling,
                    now_utc,
                ),
            )
            connection.execute(
                "DELETE FROM documents_fts WHERE document_id = ?",
                (f"comment_{comment['id']}",),
            )
            connection.execute(
                "INSERT INTO documents_fts(document_id, clean_text) VALUES (?, ?)",
                (f"comment_{comment['id']}", clean_text),
            )


def append_raw_comments(path: Path, post_id: str, comments: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for comment in comments:
            payload = {"post_id": post_id, "comment": comment}
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def save_checkpoint(path: Path, checkpoint: CommentCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(checkpoint)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    now_utc = int(datetime.now(tz=UTC).timestamp())
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
            ("ingest_comments", checkpoint.status, json.dumps(payload, ensure_ascii=True), now_utc),
        )


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_comment_count(total_comments: int, post_window: dict) -> None:
    with connect_db() as connection:
        connection.execute(
            """
            UPDATE subreddit_meta
            SET comment_count = ?
            WHERE subreddit = ?
              AND window_start_utc = ?
            """,
            (total_comments, SUBREDDIT, post_window["window_start_utc"]),
        )


def count_comments_in_window(post_window: dict) -> int:
    with connect_db() as connection:
        return int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM comments
                WHERE LOWER(subreddit) = LOWER(?)
                  AND created_utc >= ?
                  AND created_utc < ?
                """,
                (SUBREDDIT, post_window["window_start_utc"], post_window["window_end_utc"]),
            ).fetchone()[0]
        )
