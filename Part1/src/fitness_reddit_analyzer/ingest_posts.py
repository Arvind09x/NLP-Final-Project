from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from dateutil.relativedelta import relativedelta

from fitness_reddit_analyzer.arctic import ArcticShiftClient
from fitness_reddit_analyzer.config import (
    MAX_WINDOW_SCAN_MONTHS,
    MIN_MONTH_SPAN_REQUIRED,
    MIN_POSTS_REQUIRED,
    RAW_DIR,
    SUBREDDIT,
    WINDOW_END_UTC,
    WINDOW_START_UTC,
    checkpoint_path,
)
from fitness_reddit_analyzer.db import connect_db
from tqdm import tqdm


URL_RE = re.compile(r"https?://\S+")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


@dataclass
class WindowSelection:
    start_utc: int
    end_utc: int
    post_count: int
    span_months: int
    discovery_mode: str


@dataclass
class IngestionCheckpoint:
    window_start_utc: int
    window_end_utc: int
    last_after_utc: int
    inserted_posts: int
    page_count: int
    status: str


def run() -> None:
    checkpoint_file = checkpoint_path("ingest_posts")
    client = ArcticShiftClient()
    selection = select_or_resume_window(client, checkpoint_file)
    print(
        "Selected post window:",
        {
            "start_utc": selection.start_utc,
            "end_utc": selection.end_utc,
            "estimated_posts": selection.post_count,
            "span_months": selection.span_months,
            "discovery_mode": selection.discovery_mode,
        },
    )
    inserted_posts = ingest_posts_for_window(client, selection, checkpoint_file)
    print(
        f"ingest_posts complete: {inserted_posts} rows processed "
        f"for {SUBREDDIT} between {selection.start_utc} and {selection.end_utc}"
    )


def select_or_resume_window(client: ArcticShiftClient, checkpoint_file: Path) -> WindowSelection:
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint and checkpoint["status"] in {"running", "completed"}:
        return WindowSelection(
            start_utc=checkpoint["window_start_utc"],
            end_utc=checkpoint["window_end_utc"],
            post_count=checkpoint["inserted_posts"],
            span_months=month_span_from_bounds(checkpoint["window_start_utc"], checkpoint["window_end_utc"]),
            discovery_mode="checkpoint_resume",
        )
    if WINDOW_START_UTC and WINDOW_END_UTC:
        return build_override_window(client, int(WINDOW_START_UTC), int(WINDOW_END_UTC))
    return discover_window(client)


def build_override_window(client: ArcticShiftClient, start_utc: int, end_utc: int) -> WindowSelection:
    post_count = count_posts_by_pagination(client, after=start_utc, before=end_utc)
    span_months = month_span_from_bounds(start_utc, end_utc)
    if span_months < MIN_MONTH_SPAN_REQUIRED:
        raise RuntimeError(
            f"Override window spans {span_months} months, below required minimum of {MIN_MONTH_SPAN_REQUIRED}."
        )
    if post_count < MIN_POSTS_REQUIRED:
        raise RuntimeError(
            f"Override window contains {post_count} posts, below required minimum of {MIN_POSTS_REQUIRED}."
        )
    return WindowSelection(
        start_utc=start_utc,
        end_utc=end_utc,
        post_count=post_count,
        span_months=span_months,
        discovery_mode="env_override",
    )


def discover_window(client: ArcticShiftClient) -> WindowSelection:
    now = datetime.now(tz=UTC)
    month_end = datetime(year=now.year, month=now.month, day=1, tzinfo=UTC)
    month_starts = [month_end - relativedelta(months=offset + 1) for offset in range(MAX_WINDOW_SCAN_MONTHS)]
    month_counts: list[tuple[int, int]] = []

    for month_start in month_starts:
        next_month = month_start + relativedelta(months=1)
        after = int(month_start.timestamp())
        before = int(next_month.timestamp())
        count = client.metadata_only_post_count(SUBREDDIT, after=after, before=before)
        if count is None:
            count = count_posts_by_pagination(client, after=after, before=before)
            discovery_mode = "paginated_count"
        else:
            discovery_mode = "metadata_total"
        month_counts.append((after, count))
        print(
            "Discovery month:",
            {
                "month": month_start.strftime("%Y-%m"),
                "posts": count,
                "mode": discovery_mode,
            },
        )
        if len(month_counts) >= MIN_MONTH_SPAN_REQUIRED:
            window = select_best_window(month_counts, discovery_mode)
            if window is not None:
                start_label = datetime.fromtimestamp(window.start_utc, tz=UTC).strftime("%Y-%m")
                end_label = datetime.fromtimestamp(window.end_utc, tz=UTC).strftime("%Y-%m")
                print(
                    "Discovery selected rolling window:",
                    {
                        "start_month": start_label,
                        "end_month_exclusive": end_label,
                        "span_months": window.span_months,
                        "estimated_posts": window.post_count,
                    },
                )
                return window

    raise RuntimeError(
        f"Could not identify a {MIN_MONTH_SPAN_REQUIRED}-month window with at least {MIN_POSTS_REQUIRED} posts "
        f"for r/{SUBREDDIT} using the latest {MAX_WINDOW_SCAN_MONTHS} months."
    )


def select_best_window(month_counts: list[tuple[int, int]], discovery_mode: str) -> WindowSelection | None:
    ordered = sorted(month_counts, key=lambda item: item[0])
    for start_index in range(0, len(ordered) - MIN_MONTH_SPAN_REQUIRED + 1):
        running_total = 0
        for end_index in range(start_index, len(ordered)):
            running_total += ordered[end_index][1]
            span_months = end_index - start_index + 1
            if span_months >= MIN_MONTH_SPAN_REQUIRED and running_total >= MIN_POSTS_REQUIRED:
                start_utc = ordered[start_index][0]
                end_dt = datetime.fromtimestamp(ordered[end_index][0], tz=UTC) + relativedelta(months=1)
                return WindowSelection(
                    start_utc=start_utc,
                    end_utc=int(end_dt.timestamp()),
                    post_count=running_total,
                    span_months=span_months,
                    discovery_mode=discovery_mode,
                )
    return None


def count_posts_by_pagination(client: ArcticShiftClient, after: int, before: int) -> int:
    total = 0
    cursor = after
    seen_ids: set[str] = set()
    while True:
        page = client.search_posts(SUBREDDIT, after=cursor, before=before)
        items = [item for item in page.items if item.get("id") not in seen_ids]
        if not items:
            break
        total += len(items)
        seen_ids.update(item["id"] for item in items if item.get("id"))
        next_cursor = max(int(item["created_utc"]) for item in items if item.get("created_utc"))
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return total


def ingest_posts_for_window(client: ArcticShiftClient, selection: WindowSelection, checkpoint_file: Path) -> int:
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint and checkpoint["status"] in {"running", "completed"}:
        cursor = checkpoint["last_after_utc"]
        page_count = checkpoint["page_count"]
    else:
        cursor = selection.start_utc
        page_count = 0

    total_processed = checkpoint["inserted_posts"] if checkpoint else 0
    raw_dump_path = RAW_DIR / f"{SUBREDDIT}_posts_{selection.start_utc}_{selection.end_utc}.jsonl"
    progress = tqdm(
        total=selection.post_count or None,
        initial=total_processed,
        desc="Ingesting posts",
        unit="post",
    )

    try:
        while True:
            page = client.search_posts(SUBREDDIT, after=cursor, before=selection.end_utc)
            posts = page.items
            if not posts:
                break
            upsert_posts(posts)
            append_raw_posts(raw_dump_path, posts)
            total_processed += len(posts)
            page_count += 1
            progress.update(len(posts))
            next_cursor = max(int(post["created_utc"]) for post in posts if post.get("created_utc"))
            checkpoint = IngestionCheckpoint(
                window_start_utc=selection.start_utc,
                window_end_utc=selection.end_utc,
                last_after_utc=next_cursor,
                inserted_posts=total_processed,
                page_count=page_count,
                status="running",
            )
            save_checkpoint(checkpoint_file, checkpoint)
            if next_cursor <= cursor:
                break
            cursor = next_cursor
    finally:
        progress.close()

    final_checkpoint = IngestionCheckpoint(
        window_start_utc=selection.start_utc,
        window_end_utc=selection.end_utc,
        last_after_utc=selection.end_utc,
        inserted_posts=total_processed,
        page_count=page_count,
        status="completed",
    )
    save_checkpoint(checkpoint_file, final_checkpoint)
    update_subreddit_meta(selection, total_processed)
    return total_processed


def upsert_posts(posts: list[dict]) -> None:
    now_utc = int(datetime.now(tz=UTC).timestamp())
    with connect_db() as connection:
        for post in posts:
            author_id, username, is_deleted = normalize_author(post.get("author"))
            upsert_author(connection, author_id, username, is_deleted, now_utc)
            raw_text = build_post_raw_text(post)
            clean_text = clean_reddit_text(raw_text)
            connection.execute(
                """
                INSERT INTO posts (
                    post_id, subreddit, author_id, title, selftext, raw_text, clean_text,
                    created_utc, score, num_comments, permalink, url, is_deleted, is_removed,
                    raw_json, ingested_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(post_id) DO UPDATE SET
                    author_id = excluded.author_id,
                    title = excluded.title,
                    selftext = excluded.selftext,
                    raw_text = excluded.raw_text,
                    clean_text = excluded.clean_text,
                    score = excluded.score,
                    num_comments = excluded.num_comments,
                    permalink = excluded.permalink,
                    url = excluded.url,
                    is_deleted = excluded.is_deleted,
                    is_removed = excluded.is_removed,
                    raw_json = excluded.raw_json,
                    ingested_at_utc = excluded.ingested_at_utc
                """,
                (
                    post["id"],
                    post.get("subreddit", SUBREDDIT),
                    author_id,
                    post.get("title", ""),
                    post.get("selftext", ""),
                    raw_text,
                    clean_text,
                    int(post["created_utc"]),
                    post.get("score"),
                    post.get("num_comments"),
                    post.get("permalink"),
                    post.get("url"),
                    is_deleted,
                    int(bool(post.get("removed_by_category") or post.get("removed_by"))),
                    json.dumps(post, ensure_ascii=True),
                    now_utc,
                ),
            )
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, source_type, source_id, subreddit, author_id, parent_id, link_id,
                    created_utc, raw_text, clean_text, include_in_modeling, created_at_utc
                ) VALUES (?, 'post', ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    author_id = excluded.author_id,
                    raw_text = excluded.raw_text,
                    clean_text = excluded.clean_text,
                    include_in_modeling = excluded.include_in_modeling
                """,
                (
                    f"post_{post['id']}",
                    post["id"],
                    post.get("subreddit", SUBREDDIT),
                    author_id,
                    f"t3_{post['id']}",
                    int(post["created_utc"]),
                    raw_text,
                    clean_text,
                    int(not is_deleted and clean_text.strip() != ""),
                    now_utc,
                ),
            )
            connection.execute(
                "DELETE FROM documents_fts WHERE document_id = ?",
                (f"post_{post['id']}",),
            )
            connection.execute(
                "INSERT INTO documents_fts(document_id, clean_text) VALUES (?, ?)",
                (f"post_{post['id']}", clean_text),
            )


def upsert_author(
    connection: sqlite3.Connection,
    author_id: str,
    username: str,
    is_deleted: int,
    now_utc: int,
) -> None:
    stable_source = (username or author_id).encode("utf-8")
    pseudonym = f"user_{hashlib.sha256(stable_source).hexdigest()[:10]}"
    is_probable_bot = int((username or "").lower() == "automoderator" or (username or "").lower().endswith("bot"))
    connection.execute(
        """
        INSERT INTO authors (author_id, username, pseudonym, is_deleted, is_probable_bot, created_at_utc)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(author_id) DO UPDATE SET
            username = excluded.username,
            pseudonym = excluded.pseudonym,
            is_deleted = excluded.is_deleted,
            is_probable_bot = excluded.is_probable_bot
        """,
        (author_id, username, pseudonym, is_deleted, is_probable_bot, now_utc),
    )


def normalize_author(author_value: str | None) -> tuple[str, str, int]:
    if not author_value or author_value in {"[deleted]", "[removed]"}:
        return "author_deleted", "[deleted]", 1
    return f"author_{author_value}", author_value, 0


def build_post_raw_text(post: dict) -> str:
    title = post.get("title") or ""
    selftext = post.get("selftext") or ""
    return f"{title}\n\n{selftext}".strip()


def clean_reddit_text(text: str) -> str:
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub(" ", text)
    text = text.replace("&amp;", "&")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def append_raw_posts(path: Path, posts: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for post in posts:
            handle.write(json.dumps(post, ensure_ascii=True) + "\n")


def save_checkpoint(path: Path, checkpoint: IngestionCheckpoint) -> None:
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
            ("ingest_posts", checkpoint.status, json.dumps(payload, ensure_ascii=True), now_utc),
        )


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_subreddit_meta(selection: WindowSelection, total_processed: int) -> None:
    now_utc = int(datetime.now(tz=UTC).timestamp())
    post_count = count_posts_in_window(selection.start_utc, selection.end_utc)
    with connect_db() as connection:
        connection.execute(
            """
            INSERT INTO subreddit_meta (
                subreddit, window_start_utc, window_end_utc, selected_at_utc, post_count, notes
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(subreddit, window_start_utc) DO UPDATE SET
                window_end_utc = excluded.window_end_utc,
                selected_at_utc = excluded.selected_at_utc,
                post_count = excluded.post_count,
                notes = excluded.notes
            """,
            (
                SUBREDDIT,
                selection.start_utc,
                selection.end_utc,
                now_utc,
                post_count,
                (
                    f"window_discovery={selection.discovery_mode}; "
                    f"span_months={selection.span_months}; "
                    f"estimated_posts={selection.post_count}; "
                    f"processed_posts={total_processed}"
                ),
            ),
        )


def count_posts_in_window(start_utc: int, end_utc: int) -> int:
    with connect_db() as connection:
        return int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM posts
                WHERE LOWER(subreddit) = LOWER(?)
                  AND created_utc >= ?
                  AND created_utc < ?
                """,
                (SUBREDDIT, start_utc, end_utc),
            ).fetchone()[0]
        )


def month_span_from_bounds(start_utc: int, end_utc: int) -> int:
    start_dt = datetime.fromtimestamp(start_utc, tz=UTC)
    end_dt = datetime.fromtimestamp(end_utc, tz=UTC)
    return max(1, (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month))
