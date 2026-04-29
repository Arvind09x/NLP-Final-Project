from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from fitness_reddit_analyzer.config import (
    MAX_TOPIC_COUNT,
    METHODS_NOTES,
    MIN_TOPIC_COUNT,
    SUBREDDIT,
    TARGET_TOPIC_COUNT,
    TOPIC_MODEL_MIN_COMMENT_SHARE,
    TOPIC_MODEL_RANDOM_SEED,
    TOPIC_MODEL_TOTAL_DOC_TARGET,
    TOPIC_REPRESENTATIVE_DOCS,
    TOP_LEVEL_COMMENTS_PER_POST,
    checkpoint_path,
)
from fitness_reddit_analyzer.db import connect_db

EMPTY_TOPIC_CORPUS_COLUMNS = [
    "document_id",
    "source_type",
    "source_id",
    "post_id",
    "root_post_id",
    "title",
    "clean_text",
    "created_utc",
    "score",
]

COMMENT_ERA_FLOOR = 200


TOPIC_STOPWORDS = sorted(
    ENGLISH_STOP_WORDS.union(
        {
            "amp",
            "arent",
            "cant",
            "couldnt",
            "didnt",
            "doesnt",
            "dont",
            "fit",
            "fitness",
            "going",
            "got",
            "im",
            "isnt",
            "ive",
            "just",
            "know",
            "like",
            "make",
            "need",
            "people",
            "really",
            "rfitness",
            "thats",
            "theres",
            "theyre",
            "ve",
            "want",
            "wasnt",
            "wont",
            "youre",
        }
    )
)


@dataclass
class TopicCheckpoint:
    stage: str
    generated_at_utc: int
    subreddit: str
    posts_included: int
    comments_included: int
    total_modeled_documents: int
    eligible_top_level_comments: int
    capped_comment_limit: int
    final_topic_count: int
    assignment_source: str
    topic_metric: str
    status: str


def run() -> None:
    checkpoint_file = checkpoint_path("fit_topics")
    ensure_prepare_features_complete()
    documents = load_topic_corpus()
    if documents.empty:
        raise RuntimeError("No modeling documents were selected for fit_topics.")
    topic_model, topics, confidences = fit_topic_model(documents)
    topic_records, topic_time_series, document_topics = build_topic_outputs(
        documents=documents,
        topic_model=topic_model,
        topics=topics,
        confidences=confidences,
    )
    checkpoint = persist_topic_outputs(documents, topic_records, topic_time_series, document_topics, checkpoint_file)
    print("fit_topics complete:", json.dumps(asdict(checkpoint), ensure_ascii=True))


def ensure_prepare_features_complete() -> None:
    with connect_db() as connection:
        checkpoint = connection.execute(
            "SELECT status FROM pipeline_checkpoints WHERE stage_name = 'prepare_features'"
        ).fetchone()
    if checkpoint is None or checkpoint["status"] != "completed":
        raise RuntimeError("Run prepare_features successfully before fit_topics.")


def load_topic_corpus() -> pd.DataFrame:
    all_posts = load_modeled_posts()
    min_comment_slots = min(
        TOPIC_MODEL_TOTAL_DOC_TARGET,
        max(int(TOPIC_MODEL_TOTAL_DOC_TARGET * TOPIC_MODEL_MIN_COMMENT_SHARE), 0),
    )
    max_post_slots = max(TOPIC_MODEL_TOTAL_DOC_TARGET - min_comment_slots, 0)
    if len(all_posts) > max_post_slots:
        posts = all_posts.sample(n=max_post_slots, random_state=TOPIC_MODEL_RANDOM_SEED)
    else:
        posts = all_posts

    comment_budget = max(TOPIC_MODEL_TOTAL_DOC_TARGET - len(posts), 0)
    era_windows = load_era_windows()
    if len(era_windows) <= 1:
        comments = load_modeled_comments(comment_budget)
    else:
        eligible_comments = count_eligible_comments_per_era(era_windows)
        comment_caps = allocate_comment_caps(
            eligible_comments,
            comment_budget,
            minimum_floor=COMMENT_ERA_FLOOR,
        )
        comment_frames = [
            load_modeled_comments_for_era(start_utc, end_utc, cap)
            for (start_utc, end_utc), cap in comment_caps.items()
            if cap > 0
        ]
        comments = pd.concat(comment_frames, ignore_index=True) if comment_frames else empty_topic_corpus_frame()
    documents = pd.concat([posts, comments], ignore_index=True)
    if documents.empty:
        return documents
    documents["month_start"] = (
        pd.to_datetime(documents["created_utc"], unit="s", utc=True).dt.strftime("%Y-%m-01")
    )
    documents["clean_text"] = documents["clean_text"].astype(str)
    documents = documents.sort_values(["created_utc", "document_id"], kind="stable").reset_index(drop=True)
    return documents


def load_modeled_posts() -> pd.DataFrame:
    with connect_db() as connection:
        frame = pd.read_sql_query(
            """
            SELECT
                d.document_id,
                d.source_type,
                d.source_id,
                p.post_id,
                p.post_id AS root_post_id,
                p.title,
                d.clean_text,
                d.created_utc,
                COALESCE(p.score, 0) AS score
            FROM documents d
            JOIN posts p ON p.post_id = d.source_id
            WHERE d.source_type = 'post'
              AND d.include_in_modeling = 1
              AND LOWER(d.subreddit) = LOWER(?)
            ORDER BY d.created_utc ASC, d.document_id ASC
            """,
            connection,
            params=(SUBREDDIT,),
        )
    return frame


def load_modeled_comments(comment_cap: int) -> pd.DataFrame:
    return load_modeled_comments_for_era(start_utc=None, end_utc=None, era_cap=comment_cap)


def load_era_windows() -> list[tuple[int, int]]:
    with connect_db() as connection:
        rows = connection.execute(
            """
            SELECT window_start_utc, window_end_utc
            FROM subreddit_meta
            WHERE LOWER(subreddit) = LOWER(?)
            ORDER BY window_start_utc ASC
            """,
            (SUBREDDIT,),
        ).fetchall()
    return [(int(row["window_start_utc"]), int(row["window_end_utc"])) for row in rows]


def count_eligible_comments_per_era(era_windows: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    with connect_db() as connection:
        for start_utc, end_utc in era_windows:
            counts[(start_utc, end_utc)] = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM documents d
                    JOIN comments c ON c.comment_id = d.source_id
                    WHERE d.source_type = 'comment'
                      AND d.include_in_modeling = 1
                      AND LOWER(d.subreddit) = LOWER(?)
                      AND d.parent_id LIKE 't3_%'
                      AND d.created_utc >= ?
                      AND d.created_utc < ?
                    """,
                    (SUBREDDIT, start_utc, end_utc),
                ).fetchone()[0]
            )
    return counts


def allocate_comment_caps(
    eligible_comments: dict[tuple[int, int], int],
    comment_budget: int,
    *,
    minimum_floor: int,
) -> dict[tuple[int, int], int]:
    caps = {window: 0 for window in eligible_comments}
    active_windows = [window for window, eligible in eligible_comments.items() if eligible > 0]
    if comment_budget <= 0 or not active_windows:
        return caps

    desired_floors = {
        window: min(eligible_comments[window], minimum_floor)
        for window in active_windows
    }
    floor_budget = min(comment_budget, sum(desired_floors.values()))
    floor_allocations = proportional_allocation(desired_floors, floor_budget)
    for window, allocation in floor_allocations.items():
        caps[window] += allocation

    remaining_budget = comment_budget - sum(caps.values())
    remaining_capacity = {
        window: eligible_comments[window] - caps[window]
        for window in active_windows
        if eligible_comments[window] - caps[window] > 0
    }
    if remaining_budget > 0 and remaining_capacity:
        extra_allocations = proportional_allocation(remaining_capacity, remaining_budget)
        for window, allocation in extra_allocations.items():
            caps[window] += allocation
    return caps


def proportional_allocation(
    capacities: dict[tuple[int, int], int],
    budget: int,
) -> dict[tuple[int, int], int]:
    allocations = {window: 0 for window in capacities}
    active = [(window, capacity) for window, capacity in capacities.items() if capacity > 0]
    if budget <= 0 or not active:
        return allocations

    total_capacity = sum(capacity for _, capacity in active)
    if total_capacity <= budget:
        return {window: capacity for window, capacity in capacities.items()}

    remainders: list[tuple[float, int, tuple[int, int]]] = []
    used = 0
    for index, (window, capacity) in enumerate(active):
        raw_share = (budget * capacity) / total_capacity
        base = min(int(raw_share), capacity)
        allocations[window] = base
        used += base
        remainders.append((raw_share - base, index, window))

    remaining = budget - used
    for _, _, window in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if allocations[window] >= capacities[window]:
            continue
        allocations[window] += 1
        remaining -= 1

    if remaining > 0:
        for window, capacity in active:
            while remaining > 0 and allocations[window] < capacity:
                allocations[window] += 1
                remaining -= 1
            if remaining <= 0:
                break
    return allocations


def load_modeled_comments_for_era(
    start_utc: int | None,
    end_utc: int | None,
    era_cap: int,
) -> pd.DataFrame:
    if era_cap <= 0:
        return empty_topic_corpus_frame()
    time_filter_sql = ""
    params: list[object] = [SUBREDDIT]
    if start_utc is not None and end_utc is not None:
        time_filter_sql = """
                  AND d.created_utc >= ?
                  AND d.created_utc < ?
        """
        params.extend([start_utc, end_utc])
    params.extend([TOP_LEVEL_COMMENTS_PER_POST, era_cap])
    with connect_db() as connection:
        frame = pd.read_sql_query(
            f"""
            WITH ranked_comments AS (
                SELECT
                    d.document_id,
                    d.source_type,
                    d.source_id,
                    c.comment_id,
                    c.post_id,
                    c.post_id AS root_post_id,
                    p.title,
                    d.clean_text,
                    d.created_utc,
                    COALESCE(c.score, 0) AS score,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.post_id
                        ORDER BY COALESCE(c.score, 0) DESC, d.created_utc ASC, d.document_id ASC
                    ) AS within_post_rank
                FROM documents d
                JOIN comments c ON c.comment_id = d.source_id
                JOIN posts p ON p.post_id = c.post_id
                WHERE d.source_type = 'comment'
                  AND d.include_in_modeling = 1
                  AND LOWER(d.subreddit) = LOWER(?)
                  AND d.parent_id LIKE 't3_%'
                  {time_filter_sql}
            ),
            capped_comments AS (
                SELECT
                    document_id,
                    source_type,
                    source_id,
                    post_id,
                    root_post_id,
                    title,
                    clean_text,
                    created_utc,
                    score
                FROM ranked_comments
                WHERE within_post_rank <= ?
                ORDER BY score DESC, created_utc ASC, document_id ASC
                LIMIT ?
            )
            SELECT *
            FROM capped_comments
            ORDER BY created_utc ASC, document_id ASC
            """,
            connection,
            params=params,
        )
    return frame


def empty_topic_corpus_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EMPTY_TOPIC_CORPUS_COLUMNS)


def fit_topic_model(documents: pd.DataFrame) -> tuple[BERTopic, list[int], np.ndarray]:
    texts = documents["clean_text"].tolist()
    embeddings = build_embeddings(texts)
    topic_count = min(max(TARGET_TOPIC_COUNT, MIN_TOPIC_COUNT), MAX_TOPIC_COUNT, len(documents))
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=KMeans(
            n_clusters=topic_count,
            n_init=20,
            random_state=TOPIC_MODEL_RANDOM_SEED,
        ),
        vectorizer_model=CountVectorizer(
            stop_words=TOPIC_STOPWORDS,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
        ),
        nr_topics=None,
        top_n_words=10,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    confidences = assignment_confidences(topic_model, embeddings)
    final_topic_count = count_topics(topics)
    if not MIN_TOPIC_COUNT <= final_topic_count <= MAX_TOPIC_COUNT:
        raise RuntimeError(
            f"fit_topics produced {final_topic_count} topics, outside the required {MIN_TOPIC_COUNT}-{MAX_TOPIC_COUNT} range."
        )
    return topic_model, list(topics), confidences


def build_embeddings(texts: list[str]) -> np.ndarray:
    tfidf = TfidfVectorizer(
        stop_words=TOPIC_STOPWORDS,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.6,
        max_features=50_000,
        sublinear_tf=True,
    )
    matrix = tfidf.fit_transform(texts)
    n_features = matrix.shape[1]
    if n_features < 2:
        raise RuntimeError("Not enough vocabulary to build topic embeddings.")
    n_components = min(100, n_features - 1, len(texts) - 1)
    if n_components < 2:
        n_components = min(n_features, len(texts))
    reducer = make_pipeline(
        TruncatedSVD(n_components=n_components, random_state=TOPIC_MODEL_RANDOM_SEED),
        Normalizer(copy=False),
    )
    return reducer.fit_transform(matrix)


def assignment_confidences(topic_model: BERTopic, embeddings: np.ndarray) -> np.ndarray:
    cluster_model = topic_model.hdbscan_model
    if not hasattr(cluster_model, "transform"):
        return np.ones(len(embeddings), dtype=float)
    distances = cluster_model.transform(embeddings)
    min_distances = distances.min(axis=1)
    confidences = 1.0 / (1.0 + min_distances)
    return confidences.astype(float)


def build_topic_outputs(
    *,
    documents: pd.DataFrame,
    topic_model: BERTopic,
    topics: list[int],
    confidences: np.ndarray,
) -> tuple[list[dict], list[dict], list[dict]]:
    frame = documents.copy()
    frame["topic_id"] = topics
    frame["assignment_confidence"] = confidences
    frame = frame[frame["topic_id"] >= 0].copy()
    if frame.empty:
        raise RuntimeError("All documents were assigned to outlier topics.")

    total_documents = len(frame)
    months = sorted(frame["month_start"].unique().tolist())
    topic_records: list[dict] = []
    document_topics: list[dict] = []
    topic_time_series: list[dict] = []

    topic_info = topic_model.get_topic_info()
    valid_topic_ids = [int(topic_id) for topic_id in topic_info["Topic"].tolist() if int(topic_id) >= 0]

    for topic_id in valid_topic_ids:
        topic_docs = frame[frame["topic_id"] == topic_id].copy()
        topic_docs = topic_docs.sort_values(
            ["assignment_confidence", "score", "created_utc", "document_id"],
            ascending=[False, False, True, True],
            kind="stable",
        )
        keywords = topic_model.get_topic(topic_id) or []
        top_keywords = [
            {"term": str(term), "weight": round(float(weight), 6)}
            for term, weight in keywords[:10]
        ]
        label_auto = build_topic_label(top_keywords)
        document_share = float(len(topic_docs) / total_documents)
        representative_docs = [
            {
                "document_id": row.document_id,
                "source_type": row.source_type,
                "source_id": row.source_id,
                "post_id": row.root_post_id,
                "title": row.title,
                "excerpt": row.clean_text[:280],
            }
            for row in topic_docs.head(TOPIC_REPRESENTATIVE_DOCS).itertuples(index=False)
        ]
        month_counts = (
            topic_docs.groupby("month_start")["document_id"]
            .count()
            .reindex(months, fill_value=0)
            .astype(int)
        )
        active_counts = month_counts[month_counts > 0]
        month_coverage_ratio = float((month_counts > 0).mean())
        peak_to_median_ratio = float(
            active_counts.max() / np.median(active_counts) if len(active_counts) else 1.0
        )
        topic_records.append(
            {
                "topic_id": topic_id,
                "topic_label": label_auto,
                "topic_label_auto": label_auto,
                "top_keywords_json": json.dumps(top_keywords, ensure_ascii=True),
                # Stored in the legacy share_of_posts column, but this is document share for the hybrid corpus.
                "share_of_posts": document_share,
                "topic_type": classify_topic_type(month_coverage_ratio, peak_to_median_ratio),
                "month_coverage_ratio": month_coverage_ratio,
                "peak_to_median_ratio": peak_to_median_ratio,
                "representative_posts_json": json.dumps(representative_docs, ensure_ascii=True),
                "notes": build_topic_notes(),
            }
        )
        for row in topic_docs.itertuples(index=False):
            document_topics.append(
                {
                    "document_id": row.document_id,
                    "topic_id": int(row.topic_id),
                    "assignment_source": "bertopic_hybrid_top_level_comments_v1",
                    "assignment_confidence": round(float(row.assignment_confidence), 6),
                    # Major-topic status is finalized after validation in fit_stance.
                    "is_major_topic": 0,
                }
            )

    monthly_totals = frame.groupby("month_start")["document_id"].count().to_dict()
    for topic_record in topic_records:
        topic_id = int(topic_record["topic_id"])
        topic_month_counts = (
            frame[frame["topic_id"] == topic_id]
            .groupby("month_start")["document_id"]
            .count()
            .to_dict()
        )
        for month in months:
            document_count = int(topic_month_counts.get(month, 0))
            month_total = int(monthly_totals[month])
            topic_time_series.append(
                {
                    "topic_id": topic_id,
                    "month_start": month,
                    # Stored in the legacy post_count/post_share columns, but these are document-level metrics.
                    "post_count": document_count,
                    "post_share": float(document_count / month_total) if month_total else 0.0,
                }
            )

    return topic_records, topic_time_series, document_topics


def count_topics(topics: list[int]) -> int:
    return len({int(topic_id) for topic_id in topics if int(topic_id) >= 0})


def build_topic_label(top_keywords: list[dict]) -> str:
    words = [item["term"] for item in top_keywords[:3] if item["term"]]
    if not words:
        return "Unlabeled Topic"
    return " / ".join(word.replace("_", " ").title() for word in words)


def classify_topic_type(month_coverage_ratio: float, peak_to_median_ratio: float) -> str:
    if month_coverage_ratio >= 0.75 and peak_to_median_ratio <= 1.75:
        return "persistent"
    if peak_to_median_ratio >= 2.5:
        return "trending"
    return "mixed"


def build_topic_notes() -> str:
    return " ".join(METHODS_NOTES)


def persist_topic_outputs(
    documents: pd.DataFrame,
    topic_records: list[dict],
    topic_time_series: list[dict],
    document_topics: list[dict],
    checkpoint_file: Path,
) -> TopicCheckpoint:
    now_utc = int(datetime.now(tz=UTC).timestamp())
    checkpoint = TopicCheckpoint(
        stage="fit_topics",
        generated_at_utc=now_utc,
        subreddit=SUBREDDIT,
        posts_included=int((documents["source_type"] == "post").sum()),
        comments_included=int((documents["source_type"] == "comment").sum()),
        total_modeled_documents=int(len(documents)),
        eligible_top_level_comments=count_eligible_top_level_comments(),
        capped_comment_limit=max(TOPIC_MODEL_TOTAL_DOC_TARGET - int((documents["source_type"] == "post").sum()), 0),
        final_topic_count=len(topic_records),
        assignment_source="bertopic_hybrid_top_level_comments_v1",
        topic_metric="document_share",
        status="completed",
    )
    checkpoint_file.write_text(json.dumps(asdict(checkpoint), indent=2), encoding="utf-8")
    with connect_db() as connection:
        connection.execute("DELETE FROM topic_argument_summaries")
        connection.execute("DELETE FROM comment_stances")
        connection.execute("DELETE FROM document_topics")
        connection.execute("DELETE FROM topic_time_series")
        connection.execute("DELETE FROM topic_definitions")
        connection.executemany(
            """
            INSERT INTO topic_definitions (
                topic_id,
                topic_label,
                topic_label_auto,
                top_keywords_json,
                share_of_posts,
                topic_type,
                month_coverage_ratio,
                peak_to_median_ratio,
                representative_posts_json,
                notes,
                created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["topic_id"],
                    row["topic_label"],
                    row["topic_label_auto"],
                    row["top_keywords_json"],
                    row["share_of_posts"],
                    row["topic_type"],
                    row["month_coverage_ratio"],
                    row["peak_to_median_ratio"],
                    row["representative_posts_json"],
                    row["notes"],
                    now_utc,
                )
                for row in topic_records
            ],
        )
        connection.executemany(
            """
            INSERT INTO document_topics (
                document_id,
                topic_id,
                assignment_source,
                assignment_confidence,
                is_major_topic
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    row["document_id"],
                    row["topic_id"],
                    row["assignment_source"],
                    row["assignment_confidence"],
                    row["is_major_topic"],
                )
                for row in document_topics
            ],
        )
        connection.executemany(
            """
            INSERT INTO topic_time_series (
                topic_id,
                month_start,
                post_count,
                post_share
            ) VALUES (?, ?, ?, ?)
            """,
            [
                (
                    row["topic_id"],
                    row["month_start"],
                    row["post_count"],
                    row["post_share"],
                )
                for row in topic_time_series
            ],
        )
        connection.execute(
            """
            INSERT INTO pipeline_checkpoints (stage_name, status, payload_json, updated_at_utc)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(stage_name) DO UPDATE SET
                status = excluded.status,
                payload_json = excluded.payload_json,
                updated_at_utc = excluded.updated_at_utc
            """,
            ("fit_topics", "completed", json.dumps(asdict(checkpoint), ensure_ascii=True), now_utc),
        )
    return checkpoint


def count_eligible_top_level_comments() -> int:
    with connect_db() as connection:
        return int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM documents d
                JOIN comments c ON c.comment_id = d.source_id
                WHERE d.source_type = 'comment'
                  AND d.include_in_modeling = 1
                  AND LOWER(d.subreddit) = LOWER(?)
                  AND d.parent_id LIKE 't3_%'
                """,
                (SUBREDDIT,),
            ).fetchone()[0]
        )
