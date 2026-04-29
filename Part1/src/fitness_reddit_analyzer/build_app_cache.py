from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from fitness_reddit_analyzer.config import (
    APP_CACHE_PATH,
    APP_TITLE,
    METHODS_NOTES,
    SUBREDDIT,
    checkpoint_path,
)
from fitness_reddit_analyzer.db import connect_db


def run() -> None:
    payload = build_and_write_app_cache()
    print("build_app_cache complete:", json.dumps(payload, ensure_ascii=True))


def build_and_write_app_cache() -> dict[str, Any]:
    snapshot = build_snapshot()
    APP_CACHE_PATH.write_text(json.dumps(snapshot, ensure_ascii=True, indent=2), encoding="utf-8")

    now_utc = int(datetime.now(tz=UTC).timestamp())
    payload = {
        "stage": "build_app_cache",
        "generated_at_utc": now_utc,
        "subreddit": SUBREDDIT,
        "cache_path": str(APP_CACHE_PATH),
        "topic_count": snapshot["stats"]["total_topics"],
        "major_topic_count": snapshot["stats"]["major_topic_count"],
        "stance_topic_count": snapshot["stats"]["stance_topic_count"],
        "cache_source": snapshot["meta"]["cache_source"],
        "status": "completed",
    }
    checkpoint_file = checkpoint_path("build_app_cache")
    checkpoint_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

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
            ("build_app_cache", "completed", json.dumps(payload, ensure_ascii=True), now_utc),
        )

    return payload


def build_snapshot() -> dict[str, Any]:
    now_utc = int(datetime.now(tz=UTC).timestamp())
    with connect_db() as connection:
        fit_topics_checkpoint = load_checkpoint_payload(connection, "fit_topics", required=True)
        fit_stance_checkpoint = load_checkpoint_payload(connection, "fit_stance", required=False)
        meta_rows = load_subreddit_meta(connection)
        monthly_activity = load_monthly_activity(connection)
        topic_rows = load_topic_rows(connection)
        topic_trends = load_topic_trends(connection)
        stance_rows = load_stance_summary_rows(connection)
        representative_docs = hydrate_representative_documents(connection, topic_rows)
        stance_representatives = hydrate_representative_comments(connection, stance_rows)
        counts = load_corpus_counts(connection)

    window_meta = build_window_meta(meta_rows)
    stance_by_topic = build_stance_lookup(fit_stance_checkpoint, stance_rows, stance_representatives)
    topic_entries = assemble_topics(
        topic_rows,
        topic_trends,
        representative_docs,
        stance_by_topic,
        total_posts=counts["total_posts"],
        modeled_posts=int(fit_topics_checkpoint["posts_included"]),
    )

    selected_topic_ids = fit_stance_checkpoint.get("selected_topic_ids", []) if fit_stance_checkpoint else []
    topic_share_leader = max(topic_entries, key=lambda topic: topic["document_share"], default=None)
    top_topics = sorted(topic_entries, key=lambda topic: topic["document_share"], reverse=True)[:3]

    summary_line = (
        f"{counts['modeled_documents']:,} modeled documents power the topic layer "
        f"({fit_topics_checkpoint['posts_included']:,} posts + {fit_topics_checkpoint['comments_included']:,} top-level comments)."
    )
    if topic_share_leader:
        summary_line += (
            f" The largest topic is Topic {topic_share_leader['topic_id']} "
            f"({topic_share_leader['topic_label']}) at {topic_share_leader['document_share_pct']:.1f}% of modeled documents "
            f"and {topic_share_leader['corpus_post_share_pct']:.2f}% of all scraped posts."
        )

    snapshot = {
        "meta": {
            "title": APP_TITLE,
            "subreddit": SUBREDDIT,
            "generated_at_utc": now_utc,
            "cache_source": "sqlite_pipeline_snapshot",
            "selected_window": window_meta,
        },
        "stats": {
            "total_posts": counts["total_posts"],
            "total_comments": counts["total_comments"],
            "total_authors": counts["total_authors"],
            "modeled_authors": counts["modeled_authors"],
            "eligible_model_documents": counts["eligible_model_documents"],
            "modeled_documents": counts["modeled_documents"],
            "modeled_posts": fit_topics_checkpoint["posts_included"],
            "modeled_comments": fit_topics_checkpoint["comments_included"],
            "total_topics": len(topic_entries),
            "major_topic_count": len(selected_topic_ids),
            "stance_topic_count": len(selected_topic_ids),
            "window_label": window_meta["label"],
            "corpus_summary": summary_line,
        },
        "overview": {
            "summary_line": summary_line,
            "monthly_activity": monthly_activity,
            "activity_highlights": build_activity_highlights(monthly_activity),
            "top_topics": [
                {
                    "topic_id": topic["topic_id"],
                    "topic_label": topic["topic_label"],
                    "document_share_pct": topic["document_share_pct"],
                    "corpus_post_share_pct": topic["corpus_post_share_pct"],
                    "modeled_post_share_pct": topic["modeled_post_share_pct"],
                    "document_count": topic["document_count"],
                    "post_documents": topic["post_documents"],
                    "trend_label": topic["trend_label"],
                    "major_topic": topic["major_topic"],
                }
                for topic in top_topics
            ],
        },
        "topics": topic_entries,
        "topic_table": [
            {
                "topic_id": topic["topic_id"],
                "topic_label": topic["topic_label"],
                "top_keywords": ", ".join(topic["keyword_terms"][:5]),
                "document_share_pct": round(topic["document_share_pct"], 1),
                "modeled_post_share_pct": round(topic["modeled_post_share_pct"], 1),
                "corpus_post_share_pct": round(topic["corpus_post_share_pct"], 2),
                "document_count": topic["document_count"],
                "post_documents": topic["post_documents"],
                "comment_documents": topic["comment_documents"],
                "trend_label": topic["trend_label"],
                "major_topic": topic["major_topic"],
                "stance_status": topic["stance"]["status_label"],
            }
            for topic in topic_entries
        ],
        "stance": {
            "selected_topic_ids": selected_topic_ids,
            "skipped_topics": fit_stance_checkpoint.get("skipped_topics", []) if fit_stance_checkpoint else [],
            "analyzed_topics": fit_stance_checkpoint.get("analyzed_topics", []) if fit_stance_checkpoint else [],
        },
        "methods": build_methods_payload(fit_topics_checkpoint, fit_stance_checkpoint),
    }
    return snapshot


def load_checkpoint_payload(connection, stage_name: str, *, required: bool) -> dict[str, Any]:
    row = connection.execute(
        "SELECT payload_json FROM pipeline_checkpoints WHERE stage_name = ? AND status = 'completed'",
        (stage_name,),
    ).fetchone()
    if row is None:
        if required:
            raise RuntimeError(f"Run {stage_name} successfully before build_app_cache.")
        return {}
    payload = json.loads(row["payload_json"])
    return payload


def load_subreddit_meta(connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT subreddit, window_start_utc, window_end_utc, selected_at_utc, post_count, comment_count, notes
        FROM subreddit_meta
        WHERE LOWER(subreddit) = LOWER(?)
        ORDER BY window_start_utc ASC
        """,
        (SUBREDDIT,),
    ).fetchall()
    if not rows:
        raise RuntimeError("No subreddit metadata found. Run ingestion stages before build_app_cache.")
    return [dict(row) for row in rows]


def load_corpus_counts(connection) -> dict[str, int]:
    total_posts = int(connection.execute("SELECT COUNT(*) FROM posts").fetchone()[0])
    total_comments = int(connection.execute("SELECT COUNT(*) FROM comments").fetchone()[0])
    total_authors = int(
        connection.execute(
            "SELECT COUNT(*) FROM authors WHERE COALESCE(is_deleted, 0) = 0"
        ).fetchone()[0]
    )
    modeled_authors = int(
        connection.execute(
            """
            SELECT COUNT(DISTINCT author_id)
            FROM documents
            WHERE include_in_modeling = 1
              AND author_id IS NOT NULL
            """
        ).fetchone()[0]
    )
    eligible_model_documents = int(
        connection.execute(
            "SELECT COUNT(*) FROM documents WHERE include_in_modeling = 1"
        ).fetchone()[0]
    )
    modeled_documents = int(
        connection.execute(
            "SELECT COUNT(*) FROM document_topics"
        ).fetchone()[0]
    )
    return {
        "total_posts": total_posts,
        "total_comments": total_comments,
        "total_authors": total_authors,
        "modeled_authors": modeled_authors,
        "eligible_model_documents": eligible_model_documents,
        "modeled_documents": modeled_documents,
    }


def load_monthly_activity(connection) -> list[dict[str, Any]]:
    posts = pd.read_sql_query(
        """
        SELECT strftime('%Y-%m-01', datetime(created_utc, 'unixepoch')) AS month_start, COUNT(*) AS posts
        FROM posts
        GROUP BY month_start
        ORDER BY month_start
        """,
        connection,
    )
    comments = pd.read_sql_query(
        """
        SELECT strftime('%Y-%m-01', datetime(created_utc, 'unixepoch')) AS month_start, COUNT(*) AS comments
        FROM comments
        GROUP BY month_start
        ORDER BY month_start
        """,
        connection,
    )
    eligible_docs = pd.read_sql_query(
        """
        SELECT strftime('%Y-%m-01', datetime(created_utc, 'unixepoch')) AS month_start, COUNT(*) AS eligible_model_documents
        FROM documents
        WHERE include_in_modeling = 1
        GROUP BY month_start
        ORDER BY month_start
        """,
        connection,
    )
    modeled_docs = pd.read_sql_query(
        """
        SELECT t.month_start, SUM(t.post_count) AS modeled_documents
        FROM topic_time_series t
        GROUP BY t.month_start
        ORDER BY t.month_start
        """,
        connection,
    )

    frame = posts.merge(comments, how="outer", on="month_start")
    frame = frame.merge(eligible_docs, how="outer", on="month_start")
    frame = frame.merge(modeled_docs, how="outer", on="month_start")
    if frame.empty:
        return []
    frame["month_start"] = pd.to_datetime(frame["month_start"])
    full_range = pd.date_range(start=frame["month_start"].min(), end=frame["month_start"].max(), freq="MS")
    frame = frame.set_index("month_start").reindex(full_range, fill_value=0).reset_index()
    frame = frame.rename(columns={"index": "month_start"}).fillna(0).sort_values("month_start", kind="stable")
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        records.append(
            {
                "month_start": pd.Timestamp(row.month_start).strftime("%Y-%m-01"),
                "posts": int(row.posts),
                "comments": int(row.comments),
                "eligible_model_documents": int(row.eligible_model_documents),
                "modeled_documents": int(row.modeled_documents),
            }
        )
    return records


def load_topic_rows(connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            td.topic_id,
            td.topic_label,
            td.topic_label_auto,
            td.top_keywords_json,
            td.share_of_posts,
            td.topic_type,
            td.month_coverage_ratio,
            td.peak_to_median_ratio,
            td.representative_posts_json,
            td.notes,
            COUNT(dt.document_id) AS document_count,
            SUM(CASE WHEN d.source_type = 'post' THEN 1 ELSE 0 END) AS post_documents,
            SUM(CASE WHEN d.source_type = 'comment' THEN 1 ELSE 0 END) AS comment_documents,
            MAX(COALESCE(dt.is_major_topic, 0)) AS is_major_topic,
            AVG(COALESCE(dt.assignment_confidence, 0.0)) AS average_assignment_confidence
        FROM topic_definitions td
        LEFT JOIN document_topics dt ON td.topic_id = dt.topic_id
        LEFT JOIN documents d ON dt.document_id = d.document_id
        GROUP BY
            td.topic_id,
            td.topic_label,
            td.topic_label_auto,
            td.top_keywords_json,
            td.share_of_posts,
            td.topic_type,
            td.month_coverage_ratio,
            td.peak_to_median_ratio,
            td.representative_posts_json,
            td.notes
        ORDER BY td.topic_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def load_topic_trends(connection) -> dict[int, list[dict[str, Any]]]:
    rows = connection.execute(
        """
        SELECT topic_id, month_start, post_count, post_share
        FROM topic_time_series
        ORDER BY topic_id, month_start
        """
    ).fetchall()
    trends: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        trends.setdefault(int(row["topic_id"]), []).append(
            {
                "month_start": str(row["month_start"]),
                "document_count": int(row["post_count"]),
                "document_share": round(float(row["post_share"]) * 100.0, 2),
            }
        )
    return trends


def load_stance_summary_rows(connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            tas.topic_id,
            tas.stance_label,
            tas.summary_text,
            tas.representative_comment_ids_json,
            COUNT(cs.comment_id) AS comment_count,
            COUNT(DISTINCT c.author_id) AS user_count
        FROM topic_argument_summaries tas
        LEFT JOIN comment_stances cs
            ON cs.topic_id = tas.topic_id
           AND cs.stance_label = tas.stance_label
        LEFT JOIN comments c
            ON c.comment_id = cs.comment_id
        GROUP BY
            tas.topic_id,
            tas.stance_label,
            tas.summary_text,
            tas.representative_comment_ids_json
        ORDER BY tas.topic_id, tas.stance_label
        """
    ).fetchall()
    return [dict(row) for row in rows]


def hydrate_representative_documents(
    connection,
    topic_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    doc_ids: list[str] = []
    fallback_excerpts: dict[str, str] = {}
    for row in topic_rows:
        for item in json.loads(row["representative_posts_json"] or "[]"):
            document_id = str(item["document_id"])
            doc_ids.append(document_id)
            fallback_excerpts[document_id] = str(item.get("excerpt", ""))

    if not doc_ids:
        return {}

    placeholders = ", ".join("?" for _ in doc_ids)
    rows = connection.execute(
        f"""
        SELECT
            d.document_id,
            d.source_type,
            d.source_id,
            d.created_utc,
            COALESCE(a.username, a.pseudonym, '[deleted]') AS author_name,
            COALESCE(p_post.title, p_comment.title, '') AS title,
            COALESCE(p_post.clean_text, c.clean_text, d.clean_text, '') AS clean_text,
            COALESCE(p_post.score, c.score, 0) AS score,
            COALESCE(p_post.permalink, p_comment.permalink, '') AS permalink
        FROM documents d
        LEFT JOIN authors a ON d.author_id = a.author_id
        LEFT JOIN posts p_post
            ON d.source_type = 'post'
           AND p_post.post_id = d.source_id
        LEFT JOIN comments c
            ON d.source_type = 'comment'
           AND c.comment_id = d.source_id
        LEFT JOIN posts p_comment
            ON c.post_id = p_comment.post_id
        WHERE d.document_id IN ({placeholders})
        """,
        tuple(doc_ids),
    ).fetchall()
    hydrated: dict[str, dict[str, Any]] = {}
    for row in rows:
        document_id = str(row["document_id"])
        excerpt = str(row["clean_text"] or fallback_excerpts.get(document_id, "")).strip()
        hydrated[document_id] = {
            "document_id": document_id,
            "source_type": str(row["source_type"]),
            "source_id": str(row["source_id"]),
            "title": str(row["title"]),
            "author_name": str(row["author_name"]),
            "created_at": format_timestamp(int(row["created_utc"])),
            "score": int(row["score"] or 0),
            "excerpt": excerpt[:320],
            "reddit_url": build_reddit_url(str(row["permalink"] or "")),
        }
    return hydrated


def hydrate_representative_comments(
    connection,
    stance_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    comment_ids: list[str] = []
    for row in stance_rows:
        comment_ids.extend(json.loads(row["representative_comment_ids_json"] or "[]"))

    if not comment_ids:
        return {}

    placeholders = ", ".join("?" for _ in comment_ids)
    rows = connection.execute(
        f"""
        SELECT
            c.comment_id,
            c.created_utc,
            COALESCE(c.score, 0) AS score,
            COALESCE(c.clean_text, c.raw_text, c.body, '') AS comment_text,
            COALESCE(a.username, a.pseudonym, '[deleted]') AS author_name,
            p.title,
            p.permalink
        FROM comments c
        JOIN posts p ON c.post_id = p.post_id
        LEFT JOIN authors a ON c.author_id = a.author_id
        WHERE c.comment_id IN ({placeholders})
        """,
        tuple(comment_ids),
    ).fetchall()
    hydrated: dict[str, dict[str, Any]] = {}
    for row in rows:
        comment_id = str(row["comment_id"])
        hydrated[comment_id] = {
            "comment_id": comment_id,
            "title": str(row["title"]),
            "author_name": str(row["author_name"]),
            "created_at": format_timestamp(int(row["created_utc"])),
            "score": int(row["score"] or 0),
            "excerpt": str(row["comment_text"]).strip()[:320],
            "reddit_url": build_reddit_url(str(row["permalink"] or "")),
        }
    return hydrated


def build_stance_lookup(
    fit_stance_checkpoint: dict[str, Any],
    stance_rows: list[dict[str, Any]],
    stance_representatives: dict[str, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    summaries_by_topic: dict[int, list[dict[str, Any]]] = {}
    for row in stance_rows:
        topic_id = int(row["topic_id"])
        representative_comments = [
            stance_representatives[comment_id]
            for comment_id in json.loads(row["representative_comment_ids_json"] or "[]")
            if comment_id in stance_representatives
        ]
        summaries_by_topic.setdefault(topic_id, []).append(
            {
                "stance_label": str(row["stance_label"]),
                "summary_text": str(row["summary_text"]),
                "comment_count": int(row.get("comment_count") or 0),
                "user_count": int(row.get("user_count") or 0),
                "representative_comments": representative_comments,
            }
        )

    outcomes = {
        int(item["topic_id"]): item
        for item in fit_stance_checkpoint.get("analyzed_topics", [])
    } if fit_stance_checkpoint else {}
    skipped = {
        int(item["topic_id"]): item
        for item in fit_stance_checkpoint.get("skipped_topics", [])
    } if fit_stance_checkpoint else {}

    stance_lookup: dict[int, dict[str, Any]] = {}
    for topic_id, outcome in outcomes.items():
        mode = "cautious_stance_split" if outcome["outcome"] == "weak_split" else "stance_split"
        status_label = (
            "Cautious support/opposition"
            if mode == "cautious_stance_split"
            else "Support/opposition split"
        )
        stance_lookup[topic_id] = {
            "available": True,
            "mode": mode,
            "status_label": status_label,
            "detail_note": (
                "This topic was analyzed into a dominant position and an opposing/caveat position, but the cluster boundary is weak or overlapping, so read the second side as caveats rather than a clean opposition bloc."
                if mode == "cautious_stance_split"
                else "This topic passed the validated stance gate and is shown as a dominant position versus an opposing/caveat position."
            ),
            "metrics": outcome,
            "summaries": summaries_by_topic.get(topic_id, []),
        }

    for topic_id, item in skipped.items():
        stance_lookup.setdefault(
            topic_id,
            {
                "available": False,
                "mode": "not_analyzed",
                "status_label": "Not analyzed",
                "detail_note": item["reason"],
                "metrics": {},
                "summaries": [],
            },
        )
    return stance_lookup


def assemble_topics(
    topic_rows: list[dict[str, Any]],
    topic_trends: dict[int, list[dict[str, Any]]],
    representative_docs: dict[str, dict[str, Any]],
    stance_by_topic: dict[int, dict[str, Any]],
    *,
    total_posts: int,
    modeled_posts: int,
) -> list[dict[str, Any]]:
    topics: list[dict[str, Any]] = []
    for row in topic_rows:
        topic_id = int(row["topic_id"])
        keywords = json.loads(row["top_keywords_json"] or "[]")
        rep_items = []
        for item in json.loads(row["representative_posts_json"] or "[]"):
            document_id = str(item["document_id"])
            rep_items.append(
                representative_docs.get(
                    document_id,
                    {
                        "document_id": document_id,
                        "source_type": str(item.get("source_type", "")),
                        "source_id": str(item.get("source_id", "")),
                        "title": str(item.get("title", "")),
                        "author_name": "[unknown]",
                        "created_at": "",
                        "score": 0,
                        "excerpt": str(item.get("excerpt", "")),
                        "reddit_url": "",
                    },
                )
            )

        stance = stance_by_topic.get(
            topic_id,
            {
                "available": False,
                "mode": "not_analyzed",
                "status_label": "Not analyzed",
                "detail_note": "No stance analysis was run for this topic.",
                "metrics": {},
                "summaries": [],
            },
        )
        document_count = int(row["document_count"] or 0)
        post_documents = int(row["post_documents"] or 0)
        comment_documents = int(row["comment_documents"] or 0)
        modeled_post_share = (post_documents / modeled_posts) if modeled_posts else 0.0
        corpus_post_share = (post_documents / total_posts) if total_posts else 0.0
        topics.append(
            {
                "topic_id": topic_id,
                "topic_label": str(row["topic_label"]),
                "topic_label_auto": str(row["topic_label_auto"]),
                "keyword_terms": [str(item["term"]) for item in keywords],
                "keywords": keywords,
                "document_share": round(float(row["share_of_posts"] or 0.0), 4),
                "document_share_pct": round(float(row["share_of_posts"] or 0.0) * 100.0, 1),
                "document_count": document_count,
                "post_documents": post_documents,
                "comment_documents": comment_documents,
                "modeled_post_share": round(modeled_post_share, 4),
                "modeled_post_share_pct": round(modeled_post_share * 100.0, 1),
                "corpus_post_share": round(corpus_post_share, 6),
                "corpus_post_share_pct": round(corpus_post_share * 100.0, 2),
                "post_share_within_topic": round((post_documents / document_count) * 100.0, 1) if document_count else 0.0,
                "comment_share_within_topic": round((comment_documents / document_count) * 100.0, 1) if document_count else 0.0,
                "major_topic": bool(int(row["is_major_topic"] or 0)),
                "trend_label": str(row["topic_type"]),
                "month_coverage_ratio": round(float(row["month_coverage_ratio"] or 0.0) * 100.0, 1),
                "peak_to_median_ratio": round(float(row["peak_to_median_ratio"] or 0.0), 2),
                "average_assignment_confidence": round(float(row["average_assignment_confidence"] or 0.0), 3),
                "representative_documents": rep_items,
                "monthly_trend": topic_trends.get(topic_id, []),
                "stance": stance,
                "notes": str(row["notes"] or ""),
            }
        )
    return topics


def build_window_meta(meta_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eras: list[dict[str, Any]] = []
    for meta in meta_rows:
        start = datetime.fromtimestamp(int(meta["window_start_utc"]), tz=UTC)
        end_exclusive = datetime.fromtimestamp(int(meta["window_end_utc"]), tz=UTC)
        end_inclusive = end_exclusive - timedelta(seconds=1)
        eras.append(
            {
                "start_utc": int(meta["window_start_utc"]),
                "end_utc": int(meta["window_end_utc"]),
                "start_month": start.strftime("%Y-%m"),
                "end_month": end_inclusive.strftime("%Y-%m"),
                "label": f"{start.strftime('%b %Y')} to {end_inclusive.strftime('%b %Y')}",
                "post_count": int(meta.get("post_count", 0) or 0),
                "comment_count": int(meta.get("comment_count", 0) or 0),
            }
        )

    global_start = min(era["start_utc"] for era in eras)
    global_end = max(era["end_utc"] for era in eras)
    global_start_dt = datetime.fromtimestamp(global_start, tz=UTC)
    global_end_inclusive = datetime.fromtimestamp(global_end, tz=UTC) - timedelta(seconds=1)
    month_span = (global_end_inclusive.year - global_start_dt.year) * 12 + (
        global_end_inclusive.month - global_start_dt.month
    ) + 1
    return {
        "start_utc": global_start,
        "end_utc": global_end,
        "start_month": eras[0]["start_month"],
        "end_month": eras[-1]["end_month"],
        "label": "  &  ".join(era["label"] for era in eras),
        "month_span": month_span,
        "eras": eras,
    }


def build_activity_highlights(monthly_activity: list[dict[str, Any]]) -> dict[str, Any]:
    if not monthly_activity:
        return {}
    posts_peak = max(monthly_activity, key=lambda item: item["posts"])
    comments_peak = max(monthly_activity, key=lambda item: item["comments"])
    modeled_peak = max(monthly_activity, key=lambda item: item["modeled_documents"])
    return {
        "peak_posts_month": {
            "month_start": posts_peak["month_start"],
            "count": posts_peak["posts"],
        },
        "peak_comments_month": {
            "month_start": comments_peak["month_start"],
            "count": comments_peak["comments"],
        },
        "peak_modeled_month": {
            "month_start": modeled_peak["month_start"],
            "count": modeled_peak["modeled_documents"],
        },
    }


def build_methods_payload(
    fit_topics_checkpoint: dict[str, Any],
    fit_stance_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "notes": METHODS_NOTES,
        "sections": [
            {
                "title": "Data Acquisition",
                "items": [
                    "Arctic Shift is used instead of PRAW because the project setup cannot rely on Reddit's official API access path.",
                    "SQLite is the source of truth for posts, comments, authors, topic outputs, and app-facing cache inputs.",
                ],
            },
            {
                "title": "Topic Modeling",
                "items": [
                    (
                        f"The final topic model uses a hybrid corpus of {fit_topics_checkpoint['posts_included']:,} model-ready posts "
                        f"plus {fit_topics_checkpoint['comments_included']:,} bounded top-level comments."
                    ),
                    "Comments matter in r/fitness because daily questions, megathreads, and advice exchanges often carry the substantive discussion even when the original post is short.",
                    "The app now shows both hybrid document share and post-based share metrics. Document share reflects the actual modeling space, while post share shows how many post documents from a topic appear relative to the post corpus.",
                ],
            },
            {
                "title": "Stance Analysis",
                "items": [
                    "Stance analysis is unsupervised and only shown for validated major topics.",
                    "When a split looks weak or overlapping, the app uses cautious discussion-pattern wording instead of overstating disagreement.",
                    (
                        f"The validated stance topics in this cache are {', '.join(str(topic_id) for topic_id in fit_stance_checkpoint.get('selected_topic_ids', []))}."
                        if fit_stance_checkpoint
                        else "No stance checkpoint was found when the app cache was built."
                    ),
                ],
            },
        ],
    }


def build_reddit_url(permalink: str) -> str:
    if not permalink:
        return ""
    if permalink.startswith("http://") or permalink.startswith("https://"):
        return permalink
    return f"https://www.reddit.com{permalink}"


def format_timestamp(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%d")
