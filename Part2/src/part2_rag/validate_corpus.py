from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from part2_rag.config import (
    DEFAULT_CORPUS_RULE,
    DEFAULT_MANIFEST_FILENAME,
    get_paths,
)


TABLES_TO_COUNT = (
    "posts",
    "comments",
    "documents",
    "documents_fts",
    "document_embeddings",
)


@dataclass(frozen=True)
class WindowInfo:
    subreddit: str
    window_start_utc: int
    window_end_utc: int | None
    selected_at_utc: int | None
    post_count: int
    comment_count: int
    notes: str | None
    has_running_comment_ingest: bool
    observed_document_count: int
    observed_post_document_count: int
    observed_comment_document_count: int
    has_observed_data: bool
    usable_for_exploration: bool
    eligible_as_default_rag_corpus: bool
    advisory_flags: list[str]
    trust_level: str
    metadata_comment_count_is_stale: bool
    running_checkpoint_is_stale: bool


@dataclass(frozen=True)
class ValidationResult:
    db_path: str
    selected_corpus_rule: str
    selected_window: dict[str, Any]
    available_windows: list[dict[str, Any]]
    counts: dict[str, int]
    generated_at: str
    manifest_path: str


class CorpusValidationError(RuntimeError):
    """Raised when the Part 1 corpus is not safe to use for Part 2."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_readable_db(db_path: Path) -> None:
    if not db_path.exists():
        raise CorpusValidationError(f"Part 1 database does not exist: {db_path}")
    if not db_path.is_file():
        raise CorpusValidationError(f"Part 1 database is not a file: {db_path}")
    if not db_path.stat().st_size:
        raise CorpusValidationError(f"Part 1 database is empty: {db_path}")
    with db_path.open("rb") as handle:
        header = handle.read(16)
    if not header:
        raise CorpusValidationError(f"Part 1 database is not readable: {db_path}")


def fetch_table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table_name in TABLES_TO_COUNT:
        query = f"SELECT COUNT(*) FROM {table_name}"
        counts[table_name] = int(connection.execute(query).fetchone()[0])
    return counts


def fetch_running_comment_windows(connection: sqlite3.Connection) -> set[tuple[int | None, int | None]]:
    rows = connection.execute(
        """
        SELECT payload_json
        FROM pipeline_checkpoints
        WHERE stage_name = 'ingest_comments' AND status = 'running'
        """
    ).fetchall()

    running_windows: set[tuple[int | None, int | None]] = set()
    for row in rows:
        payload_json = row[0]
        if not payload_json:
            continue
        payload = json.loads(payload_json)
        running_windows.add(
            (payload.get("window_start_utc"), payload.get("window_end_utc"))
        )
    return running_windows


def fetch_window_document_counts(
    connection: sqlite3.Connection,
    subreddit: str,
    window_start_utc: int,
    window_end_utc: int | None,
) -> tuple[int, int, int]:
    row = connection.execute(
        """
        SELECT COUNT(*) AS document_count,
               SUM(CASE WHEN source_type = 'post' THEN 1 ELSE 0 END) AS post_count,
               SUM(CASE WHEN source_type = 'comment' THEN 1 ELSE 0 END) AS comment_count
        FROM documents
        WHERE LOWER(subreddit) = LOWER(?)
          AND created_utc >= ?
          AND (? IS NULL OR created_utc < ?)
        """,
        (subreddit, window_start_utc, window_end_utc, window_end_utc),
    ).fetchone()
    return int(row[0] or 0), int(row[1] or 0), int(row[2] or 0)


def derive_window_provenance(
    metadata_comment_count: int,
    has_running_comment_ingest: bool,
    observed_document_count: int,
    observed_post_document_count: int,
    observed_comment_document_count: int,
) -> tuple[bool, bool, bool, list[str], str, bool, bool]:
    has_observed_data = observed_document_count > 0
    usable_for_exploration = (
        observed_post_document_count > 0 and observed_comment_document_count > 0
    )
    metadata_comment_count_is_stale = (
        metadata_comment_count == 0 and observed_comment_document_count > 0
    )
    running_checkpoint_is_stale = (
        has_running_comment_ingest and observed_comment_document_count > 0
    )

    advisory_flags: list[str] = []
    if metadata_comment_count_is_stale:
        advisory_flags.append("metadata_comment_count_stale")
    if running_checkpoint_is_stale:
        advisory_flags.append("running_checkpoint_stale")
    if has_running_comment_ingest:
        advisory_flags.append("manual_or_incomplete_comment_ingest")
    if has_observed_data and not usable_for_exploration:
        advisory_flags.append("observed_data_incomplete_for_rag")

    eligible_as_default_rag_corpus = (
        usable_for_exploration
        and not has_running_comment_ingest
        and not metadata_comment_count_is_stale
    )

    trust_level = "default_safe" if eligible_as_default_rag_corpus else "advisory_only"
    return (
        has_observed_data,
        usable_for_exploration,
        eligible_as_default_rag_corpus,
        advisory_flags,
        trust_level,
        metadata_comment_count_is_stale,
        running_checkpoint_is_stale,
    )


def fetch_available_windows(connection: sqlite3.Connection) -> list[WindowInfo]:
    running_windows = fetch_running_comment_windows(connection)
    rows = connection.execute(
        """
        SELECT subreddit, window_start_utc, window_end_utc, selected_at_utc,
               post_count, comment_count, notes
        FROM subreddit_meta
        ORDER BY window_end_utc DESC, window_start_utc DESC
        """
    ).fetchall()

    windows: list[WindowInfo] = []
    for row in rows:
        observed_document_count, observed_post_document_count, observed_comment_document_count = (
            fetch_window_document_counts(
                connection=connection,
                subreddit=row[0],
                window_start_utc=int(row[1]),
                window_end_utc=int(row[2]) if row[2] is not None else None,
            )
        )
        has_running_comment_ingest = (row[1], row[2]) in running_windows
        (
            has_observed_data,
            usable_for_exploration,
            eligible_as_default_rag_corpus,
            advisory_flags,
            trust_level,
            metadata_comment_count_is_stale,
            running_checkpoint_is_stale,
        ) = derive_window_provenance(
            metadata_comment_count=int(row[5] or 0),
            has_running_comment_ingest=has_running_comment_ingest,
            observed_document_count=observed_document_count,
            observed_post_document_count=observed_post_document_count,
            observed_comment_document_count=observed_comment_document_count,
        )
        metadata_comment_count = int(row[5] or 0)
        window = WindowInfo(
            subreddit=row[0],
            window_start_utc=int(row[1]),
            window_end_utc=int(row[2]) if row[2] is not None else None,
            selected_at_utc=int(row[3]) if row[3] is not None else None,
            post_count=int(row[4] or 0),
            comment_count=metadata_comment_count,
            notes=row[6],
            has_running_comment_ingest=has_running_comment_ingest,
            observed_document_count=observed_document_count,
            observed_post_document_count=observed_post_document_count,
            observed_comment_document_count=observed_comment_document_count,
            has_observed_data=has_observed_data,
            usable_for_exploration=usable_for_exploration,
            eligible_as_default_rag_corpus=eligible_as_default_rag_corpus,
            advisory_flags=advisory_flags,
            trust_level=trust_level,
            metadata_comment_count_is_stale=metadata_comment_count_is_stale,
            running_checkpoint_is_stale=running_checkpoint_is_stale,
        )
        windows.append(window)
    return windows


def choose_default_window(windows: list[WindowInfo]) -> WindowInfo:
    eligible = [window for window in windows if window.eligible_as_default_rag_corpus]
    if not eligible:
        raise CorpusValidationError(
            "No subreddit window is trusted enough to use as the default frozen RAG corpus."
        )
    return max(
        eligible,
        key=lambda window: (
            window.window_end_utc if window.window_end_utc is not None else -1,
            window.window_start_utc,
        ),
    )


def fetch_selected_corpus_document_count(
    connection: sqlite3.Connection, window: WindowInfo
) -> int:
    return window.observed_document_count


def build_manifest_payload(
    db_path: Path,
    selected_window: WindowInfo,
    available_windows: list[WindowInfo],
    counts: dict[str, int],
    manifest_path: Path,
) -> ValidationResult:
    generated_at = utc_now_iso()
    return ValidationResult(
        db_path=str(db_path),
        selected_corpus_rule=DEFAULT_CORPUS_RULE,
        selected_window={
            **asdict(selected_window),
            "document_count": counts["selected_corpus_documents"],
        },
        available_windows=[asdict(window) for window in available_windows],
        counts=counts,
        generated_at=generated_at,
        manifest_path=str(manifest_path),
    )


def write_manifest(manifest_path: Path, result: ValidationResult) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def format_window_summary(window: WindowInfo, selected: bool = False) -> str:
    label = "selected" if selected else "window"
    return (
        f"{label}: subreddit={window.subreddit}, "
        f"start={window.window_start_utc}, end={window.window_end_utc}, "
        f"posts={window.post_count}, comments={window.comment_count}, "
        f"running_comment_ingest={window.has_running_comment_ingest}"
    )


def validate_corpus(db_path: Path, manifest_path: Path) -> ValidationResult:
    ensure_readable_db(db_path)
    with sqlite3.connect(db_path) as connection:
        counts = fetch_table_counts(connection)
        available_windows = fetch_available_windows(connection)
        selected_window = choose_default_window(available_windows)
        selected_document_count = fetch_selected_corpus_document_count(
            connection, selected_window
        )
        if selected_document_count <= 0:
            raise CorpusValidationError(
                "Selected default RAG corpus is empty in the documents table."
            )
        counts["selected_corpus_documents"] = selected_document_count

    result = build_manifest_payload(
        db_path=db_path,
        selected_window=selected_window,
        available_windows=available_windows,
        counts=counts,
        manifest_path=manifest_path,
    )
    write_manifest(manifest_path, result)
    return result


def print_summary(result: ValidationResult) -> None:
    print("Part 2 corpus validation complete")
    print(f"DB path: {result.db_path}")
    print(f"Corpus rule: {result.selected_corpus_rule}")
    print("")
    print("Table counts:")
    for table_name in TABLES_TO_COUNT:
        print(f"  - {table_name}: {result.counts[table_name]}")
    print(f"  - selected_corpus_documents: {result.counts['selected_corpus_documents']}")
    print("")
    print("Available subreddit windows:")
    selected_start = result.selected_window["window_start_utc"]
    selected_end = result.selected_window["window_end_utc"]
    for window in result.available_windows:
        is_selected = (
            window["window_start_utc"] == selected_start
            and window["window_end_utc"] == selected_end
        )
        prefix = "*" if is_selected else "-"
        print(
            f"  {prefix} subreddit={window['subreddit']}, "
            f"start={window['window_start_utc']}, end={window['window_end_utc']}, "
            f"meta_posts={window['post_count']}, meta_comments={window['comment_count']}, "
            f"observed_posts={window['observed_post_document_count']}, "
            f"observed_comments={window['observed_comment_document_count']}, "
            f"has_observed_data={window['has_observed_data']}, "
            f"usable_for_exploration={window['usable_for_exploration']}, "
            f"eligible_as_default_rag_corpus={window['eligible_as_default_rag_corpus']}, "
            f"trust_level={window['trust_level']}, "
            f"running_comment_ingest={window['has_running_comment_ingest']}"
        )
        if window["advisory_flags"]:
            print(f"    advisory_flags={','.join(window['advisory_flags'])}")
    print("")
    print("Selected default window:")
    print(
        f"  - subreddit={result.selected_window['subreddit']}, "
        f"start={result.selected_window['window_start_utc']}, "
        f"end={result.selected_window['window_end_utc']}, "
        f"documents={result.selected_window['document_count']}, "
        f"observed_comments={result.selected_window['observed_comment_document_count']}"
    )
    print(f"Manifest written to: {result.manifest_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    paths = get_paths()
    parser = argparse.ArgumentParser(
        description="Validate the Part 1 SQLite corpus for Part 2 RAG."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=paths.part1_db_path,
        help="Path to the Part 1 SQLite database.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=paths.manifests_dir / DEFAULT_MANIFEST_FILENAME,
        help="Path where the corpus manifest JSON should be written.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        result = validate_corpus(
            db_path=args.db_path.resolve(),
            manifest_path=args.manifest_path.resolve(),
        )
    except CorpusValidationError as exc:
        parser.exit(status=1, message=f"Corpus validation failed: {exc}\n")
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
