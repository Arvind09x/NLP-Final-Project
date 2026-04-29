from __future__ import annotations

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "fitness_part1.sqlite"


def subreddit_meta_has_composite_pk(connection: sqlite3.Connection) -> bool:
    rows = connection.execute("PRAGMA table_info(subreddit_meta)").fetchall()
    pk_columns = {row[1]: row[5] for row in rows if row[5]}
    return pk_columns == {"subreddit": 1, "window_start_utc": 2}


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    connection = sqlite3.connect(DB_PATH)
    try:
        table_exists = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'subreddit_meta'
            """
        ).fetchone()
        if table_exists is None:
            print("subreddit_meta does not exist yet; nothing to migrate.")
            return

        if subreddit_meta_has_composite_pk(connection):
            print("subreddit_meta already uses PRIMARY KEY (subreddit, window_start_utc).")
            return

        with connection:
            connection.execute(
                """
                CREATE TABLE subreddit_meta_new (
                    subreddit TEXT NOT NULL,
                    window_start_utc INTEGER NOT NULL,
                    window_end_utc INTEGER,
                    selected_at_utc INTEGER,
                    post_count INTEGER DEFAULT 0,
                    comment_count INTEGER DEFAULT 0,
                    notes TEXT,
                    PRIMARY KEY (subreddit, window_start_utc)
                )
                """
            )
            connection.execute(
                """
                INSERT INTO subreddit_meta_new (
                    subreddit,
                    window_start_utc,
                    window_end_utc,
                    selected_at_utc,
                    post_count,
                    comment_count,
                    notes
                )
                SELECT
                    subreddit,
                    window_start_utc,
                    window_end_utc,
                    selected_at_utc,
                    post_count,
                    comment_count,
                    notes
                FROM subreddit_meta
                """
            )
            connection.execute("DROP TABLE subreddit_meta")
            connection.execute("ALTER TABLE subreddit_meta_new RENAME TO subreddit_meta")
        print("Migrated subreddit_meta to composite primary key.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
