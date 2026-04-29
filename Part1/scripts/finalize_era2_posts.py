"""
Finalize the partial Era 2 post ingest.

Treats the 21,200 posts already in the DB (Jun 1 2018 – Aug 16 2018) as the
complete Era 2 post dataset. Updates:
  1. pipeline_checkpoints: ingest_posts → completed
  2. subreddit_meta: inserts Era 2 row
  3. checkpoint file on disk
"""
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitness_reddit_analyzer.config import CHECKPOINTS_DIR, SUBREDDIT, ensure_directories
from fitness_reddit_analyzer.db import connect_db

# Era 2 actual data boundaries (half-open: >= start, < end)
ERA2_START_UTC = 1527811200   # 2018-06-01 00:00:00 UTC
ERA2_END_UTC   = 1534464000   # 2018-08-17 00:00:00 UTC  (day after last ingested post)
ERA2_POST_COUNT = 21200
ERA2_LAST_CURSOR = 1534415590  # timestamp of last ingested post

now_utc = int(datetime.now(tz=UTC).timestamp())


def main() -> None:
    ensure_directories()

    # 1. Verify the posts actually exist
    with connect_db() as conn:
        actual = conn.execute(
            "SELECT COUNT(*) FROM posts WHERE created_utc >= ? AND created_utc < ?",
            (ERA2_START_UTC, ERA2_END_UTC),
        ).fetchone()[0]
    print(f"Era 2 posts in DB: {actual} (expected {ERA2_POST_COUNT})")
    if actual < ERA2_POST_COUNT * 0.95:
        print("WARNING: Post count is significantly lower than expected. Proceeding anyway.")

    # 2. Update pipeline_checkpoints
    checkpoint_payload = {
        "window_start_utc": ERA2_START_UTC,
        "window_end_utc": ERA2_END_UTC,
        "last_after_utc": ERA2_END_UTC,
        "inserted_posts": actual,
        "page_count": -1,  # unknown, partial ingest
        "status": "completed",
    }
    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO pipeline_checkpoints (stage_name, status, payload_json, updated_at_utc)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(stage_name) DO UPDATE SET
                status = excluded.status,
                payload_json = excluded.payload_json,
                updated_at_utc = excluded.updated_at_utc
            """,
            ("ingest_posts", "completed", json.dumps(checkpoint_payload), now_utc),
        )
    print("pipeline_checkpoints: ingest_posts → completed")

    # 3. Insert Era 2 subreddit_meta row
    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO subreddit_meta (subreddit, window_start_utc, window_end_utc, selected_at_utc, post_count, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(subreddit, window_start_utc) DO UPDATE SET
                window_end_utc = excluded.window_end_utc,
                selected_at_utc = excluded.selected_at_utc,
                post_count = excluded.post_count,
                notes = excluded.notes
            """,
            (
                SUBREDDIT,
                ERA2_START_UTC,
                ERA2_END_UTC,
                now_utc,
                actual,
                f"window_discovery=partial_ingest_finalized; span_months=2; last_cursor={ERA2_LAST_CURSOR}",
            ),
        )
    print("subreddit_meta: Era 2 row inserted")

    # 4. Write checkpoint file to disk
    checkpoint_file = CHECKPOINTS_DIR / "ingest_posts.json"
    checkpoint_file.write_text(json.dumps(checkpoint_payload, indent=2), encoding="utf-8")
    print(f"Checkpoint file written: {checkpoint_file}")

    # 5. Verify final state
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT subreddit, window_start_utc, window_end_utc, post_count, comment_count FROM subreddit_meta ORDER BY window_start_utc"
        ).fetchall()
        print("\nsubreddit_meta rows:")
        for r in rows:
            start_dt = datetime.fromtimestamp(r["window_start_utc"], tz=UTC).strftime("%Y-%m-%d")
            end_dt = datetime.fromtimestamp(r["window_end_utc"], tz=UTC).strftime("%Y-%m-%d")
            print(f"  {r['subreddit']} | {start_dt} → {end_dt} | posts={r['post_count']} | comments={r['comment_count']}")

        cp = conn.execute(
            "SELECT stage_name, status FROM pipeline_checkpoints WHERE stage_name = 'ingest_posts'"
        ).fetchone()
        print(f"\ningest_posts checkpoint: {cp['status']}")

    print("\n✓ Era 2 post ingest finalized. Ready for Era 2 comment ingestion.")


if __name__ == "__main__":
    main()
