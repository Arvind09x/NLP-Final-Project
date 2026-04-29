from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from fitness_reddit_analyzer.arctic import ArcticShiftClient
from fitness_reddit_analyzer.config import MIN_POST_TEXT_CHARS, SUBREDDIT
from fitness_reddit_analyzer.db import connect_db
from fitness_reddit_analyzer.ingest_posts import clean_reddit_text


BOT_NAME_RE = re.compile(r"bot$", re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^a-z0-9\s]+")
DIGIT_RE = re.compile(r"\b\d+\b")
WHITESPACE_RE = re.compile(r"\s+")
DATE_TOKEN_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RecurringTitle:
    normalized_title: str
    count: int
    share_of_posts: float


@dataclass(frozen=True)
class CorpusAudit:
    source: str
    subreddit: str
    posts_examined: int
    is_sampled: bool
    sample_post_limit: int | None
    window_start_utc: int
    window_end_utc: int
    window_start_iso: str
    window_end_iso: str
    month_span: int
    total_posts: int
    total_comments: int
    non_removed_posts: int
    non_deleted_posts: int
    model_ready_posts: int
    removed_post_fraction: float
    title_only_or_low_text_posts: int
    title_only_or_low_text_fraction: float
    recurring_title_posts: int
    recurring_title_fraction: float
    top_recurring_titles: list[RecurringTitle]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.source == "sqlite":
        audit = audit_sqlite_corpus()
    else:
        if not args.start or not args.end:
            raise SystemExit("--start and --end are required for Arctic Shift audits.")
        start_utc = parse_date_to_utc(args.start)
        end_utc = parse_date_to_utc(args.end)
        audit = audit_arctic_window(start_utc=start_utc, end_utc=end_utc, max_posts=args.max_posts)
    print(json.dumps(asdict(audit), indent=2, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit r/fitness corpus quality for topic modeling")
    parser.add_argument("--source", choices=("sqlite", "arctic"), default="sqlite")
    parser.add_argument("--start", help="Inclusive UTC start date in YYYY-MM-DD format")
    parser.add_argument("--end", help="Exclusive UTC end date in YYYY-MM-DD format")
    parser.add_argument("--max-posts", type=int, help="Stop after this many posts when auditing Arctic Shift")
    return parser


def audit_sqlite_corpus() -> CorpusAudit:
    with connect_db() as connection:
        posts = pd.read_sql_query(
            """
            SELECT
                p.post_id,
                p.title,
                p.selftext,
                p.clean_text,
                p.created_utc,
                p.num_comments,
                p.is_deleted,
                p.is_removed,
                COALESCE(a.is_probable_bot, 0) AS is_probable_bot
            FROM posts p
            LEFT JOIN authors a ON a.author_id = p.author_id
            WHERE LOWER(p.subreddit) = LOWER(?)
            ORDER BY p.created_utc ASC, p.post_id ASC
            """,
            connection,
            params=(SUBREDDIT,),
        ).fillna(
            {
                "title": "",
                "selftext": "",
                "clean_text": "",
                "num_comments": 0,
                "is_deleted": 0,
                "is_removed": 0,
                "is_probable_bot": 0,
            }
        )
        total_comments = int(
            connection.execute(
                "SELECT COUNT(*) FROM comments WHERE LOWER(subreddit) = LOWER(?)",
                (SUBREDDIT,),
            ).fetchone()[0]
        )
        window = connection.execute(
            """
            SELECT window_start_utc, window_end_utc
            FROM subreddit_meta
            WHERE subreddit = ?
            """,
            (SUBREDDIT,),
        ).fetchone()
    if window is None:
        raise RuntimeError("No subreddit window found in SQLite. Run ingestion first.")
    return build_corpus_audit(
        posts_frame=posts,
        total_comments=total_comments,
        source="sqlite",
        posts_examined=int(len(posts)),
        sample_post_limit=None,
        window_start_utc=int(window["window_start_utc"]),
        window_end_utc=int(window["window_end_utc"]),
    )


def audit_arctic_window(*, start_utc: int, end_utc: int, max_posts: int | None = None) -> CorpusAudit:
    client = ArcticShiftClient()
    posts = fetch_posts_for_window(client=client, start_utc=start_utc, end_utc=end_utc, max_posts=max_posts)
    frame = pd.DataFrame(posts)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "post_id",
                "title",
                "selftext",
                "clean_text",
                "created_utc",
                "num_comments",
                "is_deleted",
                "is_removed",
                "is_probable_bot",
            ]
        )
    total_comments = int(frame["num_comments"].fillna(0).sum()) if "num_comments" in frame else 0
    return build_corpus_audit(
        posts_frame=frame,
        total_comments=total_comments,
        source="arctic",
        posts_examined=int(len(frame)),
        sample_post_limit=max_posts,
        window_start_utc=start_utc,
        window_end_utc=end_utc,
    )


def fetch_posts_for_window(
    *,
    client: ArcticShiftClient,
    start_utc: int,
    end_utc: int,
    max_posts: int | None = None,
) -> list[dict[str, Any]]:
    cursor = start_utc
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    while True:
        page = client.search_posts(SUBREDDIT, after=cursor, before=end_utc)
        page_items = [item for item in page.items if item.get("id") not in seen_ids]
        if not page_items:
            break
        for item in page_items:
            items.append(
                {
                    "post_id": item.get("id", ""),
                    "title": item.get("title", "") or "",
                    "selftext": item.get("selftext", "") or "",
                    "clean_text": clean_reddit_text(build_post_text(item)),
                    "created_utc": int(item.get("created_utc", 0) or 0),
                    "num_comments": int(item.get("num_comments", 0) or 0),
                    "is_deleted": infer_deleted_post(item),
                    "is_removed": infer_removed_post(item),
                    "is_probable_bot": infer_probable_bot(item.get("author")),
                }
            )
            if max_posts is not None and len(items) >= max_posts:
                return items[:max_posts]
        seen_ids.update(str(item.get("id")) for item in page_items if item.get("id"))
        next_cursor = max(int(item.get("created_utc", 0) or 0) for item in page_items)
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return items


def build_corpus_audit(
    *,
    posts_frame: pd.DataFrame,
    total_comments: int,
    source: str,
    posts_examined: int,
    sample_post_limit: int | None,
    window_start_utc: int,
    window_end_utc: int,
) -> CorpusAudit:
    frame = posts_frame.copy()
    for column, default in (
        ("title", ""),
        ("selftext", ""),
        ("clean_text", ""),
        ("num_comments", 0),
        ("is_deleted", 0),
        ("is_removed", 0),
        ("is_probable_bot", 0),
    ):
        if column not in frame.columns:
            frame[column] = default
    if "created_utc" not in frame.columns:
        frame["created_utc"] = 0

    frame["clean_text"] = frame["clean_text"].astype(str)
    frame["selftext"] = frame["selftext"].astype(str)
    frame["clean_char_count"] = frame["clean_text"].str.len()
    frame["title_only_or_low_text"] = (
        frame["selftext"].map(is_title_only_selftext)
        | (frame["clean_char_count"] < MIN_POST_TEXT_CHARS)
    ).astype(int)
    frame["model_ready"] = (
        (frame["is_deleted"].astype(int) == 0)
        & (frame["is_removed"].astype(int) == 0)
        & (frame["is_probable_bot"].astype(int) == 0)
        & (frame["clean_char_count"] >= MIN_POST_TEXT_CHARS)
    ).astype(int)
    frame["normalized_title"] = frame["title"].astype(str).map(normalize_title)

    total_posts = int(len(frame))
    recurring_counter = Counter(
        title
        for title in frame["normalized_title"]
        if title and title != "untitled"
    )
    recurring_counter = Counter({title: count for title, count in recurring_counter.items() if count >= 3})
    recurring_title_posts = int(sum(recurring_counter.values()))
    top_recurring_titles = [
        RecurringTitle(
            normalized_title=title,
            count=count,
            share_of_posts=round(count / total_posts, 4) if total_posts else 0.0,
        )
        for title, count in recurring_counter.most_common(10)
    ]

    return CorpusAudit(
        source=source,
        subreddit=SUBREDDIT,
        posts_examined=posts_examined,
        is_sampled=sample_post_limit is not None and posts_examined >= sample_post_limit,
        sample_post_limit=sample_post_limit,
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
        window_start_iso=to_iso_date(window_start_utc),
        window_end_iso=to_iso_date(window_end_utc),
        month_span=month_span(window_start_utc, window_end_utc),
        total_posts=total_posts,
        total_comments=int(total_comments),
        non_removed_posts=int((frame["is_removed"].astype(int) == 0).sum()),
        non_deleted_posts=int((frame["is_deleted"].astype(int) == 0).sum()),
        model_ready_posts=int(frame["model_ready"].sum()),
        removed_post_fraction=round(float((frame["is_removed"].astype(int) == 1).mean()) if total_posts else 0.0, 4),
        title_only_or_low_text_posts=int(frame["title_only_or_low_text"].sum()),
        title_only_or_low_text_fraction=round(
            float(frame["title_only_or_low_text"].mean()) if total_posts else 0.0,
            4,
        ),
        recurring_title_posts=recurring_title_posts,
        recurring_title_fraction=round(recurring_title_posts / total_posts, 4) if total_posts else 0.0,
        top_recurring_titles=top_recurring_titles,
    )


def build_post_text(post: dict[str, Any]) -> str:
    title = post.get("title") or ""
    selftext = post.get("selftext") or ""
    return f"{title}\n\n{selftext}".strip()


def infer_deleted_post(post: dict[str, Any]) -> int:
    author = (post.get("author") or "").strip().lower()
    selftext = (post.get("selftext") or "").strip().lower()
    return int(author in {"[deleted]", "[removed]"} or selftext == "[deleted]")


def infer_removed_post(post: dict[str, Any]) -> int:
    selftext = (post.get("selftext") or "").strip().lower()
    return int(bool(post.get("removed_by_category") or post.get("removed_by")) or selftext == "[removed]")


def infer_probable_bot(author: Any) -> int:
    username = str(author or "")
    lowered = username.lower()
    return int(lowered == "automoderator" or bool(BOT_NAME_RE.search(lowered)))


def is_title_only_selftext(selftext: str) -> bool:
    normalized = (selftext or "").strip().lower()
    return normalized in {"", "[removed]", "[deleted]", "[deleted by user]"}


def normalize_title(title: str) -> str:
    text = clean_reddit_text(title).lower()
    text = DATE_TOKEN_RE.sub(" ", text)
    text = DIGIT_RE.sub(" ", text)
    text = NON_WORD_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text or "untitled"


def parse_date_to_utc(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())


def to_iso_date(value: int) -> str:
    return datetime.fromtimestamp(value, tz=UTC).strftime("%Y-%m-%d")


def month_span(start_utc: int, end_utc: int) -> int:
    start_dt = datetime.fromtimestamp(start_utc, tz=UTC)
    end_dt = datetime.fromtimestamp(end_utc, tz=UTC)
    return max(1, (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month))


if __name__ == "__main__":
    main()
