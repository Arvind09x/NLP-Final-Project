from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from fitness_reddit_analyzer.config import (
    EXCLUDED_STANCE_TOPIC_IDS,
    OPTIONAL_STANCE_TOPIC_IDS,
    STANCE_INCLUDE_OPTIONAL_TOPIC_0,
    STANCE_MAX_CENTROID_SIMILARITY,
    STANCE_MIN_MINORITY_COMMENTS,
    STANCE_MIN_MINORITY_SHARE,
    STANCE_MIN_SILHOUETTE,
    STANCE_MIN_TOPIC_COMMENTS,
    STANCE_TOPIC_SHARE_THRESHOLD,
    SUBREDDIT,
    TOPIC_MODEL_RANDOM_SEED,
    VALIDATED_STANCE_TOPIC_IDS,
    checkpoint_path,
)
from fitness_reddit_analyzer.db import connect_db


TOKEN_RE = re.compile(r"[a-z][a-z0-9_]+")
GENERIC_EXTRACTIVE_TERMS = {
    "fitness",
    "question",
    "questions",
    "people",
    "person",
    "thing",
    "things",
    "time",
    "day",
    "days",
    "weeks",
    "months",
    "years",
    "today",
    "did",
    "doing",
    "really",
    "just",
    "getting",
    "make",
    "made",
    "like",
    "ve",
    "dont",
    "doesnt",
    "didnt",
    "ive",
    "im",
    "youre",
    "theyre",
    "thats",
    "cant",
    "wont",
    "gym",
}
TOPIC_INTERPRETATIONS = {
    0: "broad catch-all fitness discussion",
    1: "weight loss, calories, protein, and body composition",
    2: "gym experiences, etiquette, and recurring community rants",
    5: "bench press, sets, reps, and strength-programming choices",
    6: "physique posts, posting rules, and community feedback",
}


@dataclass
class TopicSelection:
    topic_id: int
    topic_label: str
    share_of_posts: float
    reason: str


@dataclass
class TopicOutcome:
    topic_id: int
    topic_label: str
    interpretation: str
    analyzed_comments: int
    dominant_comments: int
    minority_comments: int
    outcome: str
    minority_share: float
    silhouette_score: float
    centroid_similarity: float


@dataclass
class StanceCheckpoint:
    stage: str
    generated_at_utc: int
    subreddit: str
    selected_topic_ids: list[int]
    selected_topic_labels: list[str]
    skipped_topics: list[dict]
    analyzed_topics: list[dict]
    comment_stances_written: int
    topic_summaries_written: int
    status: str


def run() -> None:
    checkpoint_file = checkpoint_path("fit_stance")
    ensure_topics_complete()
    selected_topics, skipped_topics = determine_stance_topics()

    comment_stance_rows: list[tuple] = []
    summary_rows: list[tuple] = []
    analyzed_topics: list[TopicOutcome] = []
    final_major_topic_ids: list[int] = []

    for topic in selected_topics:
        comments = load_candidate_comments(topic.topic_id)
        if len(comments) < STANCE_MIN_TOPIC_COMMENTS:
            skipped_topics.append(
                {
                    "topic_id": topic.topic_id,
                    "topic_label": topic.topic_label,
                    "reason": (
                        f"selected by validation but skipped because only {len(comments)} "
                        f"candidate comments were available (minimum {STANCE_MIN_TOPIC_COMMENTS})"
                    ),
                }
            )
            continue

        result = analyze_topic(topic, comments)
        analyzed_topics.append(result["topic_outcome"])
        comment_stance_rows.extend(result["comment_stances"])
        summary_rows.extend(result["topic_summaries"])
        final_major_topic_ids.append(topic.topic_id)

    checkpoint = persist_outputs(
        final_major_topic_ids=final_major_topic_ids,
        comment_stance_rows=comment_stance_rows,
        summary_rows=summary_rows,
        analyzed_topics=analyzed_topics,
        skipped_topics=skipped_topics,
        checkpoint_file=checkpoint_file,
    )
    print("fit_stance complete:", json.dumps(asdict(checkpoint), ensure_ascii=True))


def ensure_topics_complete() -> None:
    with connect_db() as connection:
        checkpoint = connection.execute(
            "SELECT status FROM pipeline_checkpoints WHERE stage_name = 'fit_topics'"
        ).fetchone()
        topic_count = connection.execute("SELECT COUNT(*) FROM topic_definitions").fetchone()[0]
    if checkpoint is None or checkpoint["status"] != "completed":
        raise RuntimeError("Run fit_topics successfully before fit_stance.")
    if int(topic_count) == 0:
        raise RuntimeError("No topic definitions found. Run fit_topics before fit_stance.")


def determine_stance_topics() -> tuple[list[TopicSelection], list[dict]]:
    with connect_db() as connection:
        rows = connection.execute(
            """
            SELECT topic_id, topic_label, share_of_posts, top_keywords_json
            FROM topic_definitions
            ORDER BY topic_id
            """
        ).fetchall()

    selected: list[TopicSelection] = []
    skipped: list[dict] = []
    validated_ids = set(VALIDATED_STANCE_TOPIC_IDS)
    optional_ids = set(OPTIONAL_STANCE_TOPIC_IDS)
    excluded_ids = set(EXCLUDED_STANCE_TOPIC_IDS)

    for row in rows:
        topic_id = int(row["topic_id"])
        share = float(row["share_of_posts"] or 0.0)
        topic_label = str(row["topic_label"])

        if share < STANCE_TOPIC_SHARE_THRESHOLD:
            skipped.append(
                {
                    "topic_id": topic_id,
                    "topic_label": topic_label,
                    "reason": (
                        f"share_of_posts={share:.4f} is below the stance threshold "
                        f"{STANCE_TOPIC_SHARE_THRESHOLD:.2f}"
                    ),
                }
            )
            continue

        if topic_id in excluded_ids:
            skipped.append(
                {
                    "topic_id": topic_id,
                    "topic_label": topic_label,
                    "reason": infer_validated_skip_reason(topic_id),
                }
            )
            continue

        if topic_id in optional_ids and not STANCE_INCLUDE_OPTIONAL_TOPIC_0:
            skipped.append(
                {
                    "topic_id": topic_id,
                    "topic_label": topic_label,
                    "reason": "broad catch-all topic kept optional and excluded by default",
                }
            )
            continue

        if topic_id in validated_ids or (topic_id in optional_ids and STANCE_INCLUDE_OPTIONAL_TOPIC_0):
            selected.append(
                TopicSelection(
                    topic_id=topic_id,
                    topic_label=topic_label,
                    share_of_posts=share,
                    reason="selected by validated stance gating",
                )
            )
            continue

        skipped.append(
            {
                "topic_id": topic_id,
                "topic_label": topic_label,
                "reason": "not part of the validated default stance topic set",
            }
        )

    return selected, skipped


def infer_validated_skip_reason(topic_id: int) -> str:
    if topic_id == 4:
        return "validated exclusion: moderation/process/template topic"
    if topic_id == 5:
        return "validated exclusion: process/community-thread topic"
    if topic_id == 3:
        return "validated exclusion: anecdote/community-only topic"
    if topic_id == 6:
        return "validated exclusion: moderation/process-heavy physique-post topic"
    if topic_id == 7:
        return "validated exclusion: low-volume anecdote/community-only topic"
    return "validated exclusion: not stance-worthy"


def load_candidate_comments(topic_id: int) -> pd.DataFrame:
    with connect_db() as connection:
        frame = pd.read_sql_query(
            """
            SELECT
                c.comment_id,
                c.post_id,
                c.score,
                c.clean_text,
                c.created_utc,
                dt.assignment_confidence
            FROM document_topics dt
            JOIN documents d ON d.document_id = dt.document_id
            JOIN comments c ON c.comment_id = d.source_id
            WHERE LOWER(d.subreddit) = LOWER(?)
              AND d.source_type = 'comment'
              AND d.include_in_modeling = 1
              AND dt.topic_id = ?
            ORDER BY dt.assignment_confidence DESC, c.score DESC, c.comment_id ASC
            """,
            connection,
            params=(SUBREDDIT, topic_id),
        )
    if frame.empty:
        return frame
    frame["clean_text"] = frame["clean_text"].fillna("").astype(str)
    frame["score"] = frame["score"].fillna(0).astype(int)
    frame["assignment_confidence"] = frame["assignment_confidence"].fillna(0.0).astype(float)
    return frame


def analyze_topic(topic: TopicSelection, comments: pd.DataFrame) -> dict:
    interpretation = TOPIC_INTERPRETATIONS.get(topic.topic_id, topic.topic_label)
    topic_terms = extract_topic_terms(topic.topic_label)
    vectorizer = build_vectorizer(len(comments), topic_terms)
    matrix = vectorizer.fit_transform(comments["clean_text"].tolist())
    if matrix.shape[1] < 2:
        raise RuntimeError(f"Topic {topic.topic_id} does not have enough text variation for stance clustering.")

    clustering = KMeans(n_clusters=2, n_init=20, random_state=TOPIC_MODEL_RANDOM_SEED)
    cluster_ids = clustering.fit_predict(matrix)
    distances = clustering.transform(matrix)
    distance_margin = np.abs(distances[:, 0] - distances[:, 1]) / np.maximum(distances.sum(axis=1), 1e-9)

    counts = np.bincount(cluster_ids, minlength=2)
    dominant_cluster = int(np.argmax(counts))
    minority_cluster = int(1 - dominant_cluster)
    total_comments = int(counts.sum())
    minority_count = int(counts[minority_cluster])
    minority_share = float(minority_count / total_comments) if total_comments else 0.0
    silhouette = compute_silhouette(matrix, cluster_ids)
    centroid_similarity = compute_centroid_similarity(clustering.cluster_centers_)

    weak_split = (
        minority_count < max(STANCE_MIN_MINORITY_COMMENTS, int(math.ceil(total_comments * STANCE_MIN_MINORITY_SHARE)))
        or minority_share < STANCE_MIN_MINORITY_SHARE
        or silhouette < STANCE_MIN_SILHOUETTE
        or centroid_similarity > STANCE_MAX_CENTROID_SIMILARITY
    )

    label_map = {
        dominant_cluster: "dominant_position",
        minority_cluster: "opposing_or_caveat_position",
    }
    outcome = "weak_split" if weak_split else "stance_split"

    comments = comments.copy()
    comments["cluster_id"] = cluster_ids
    comments["prototype_distance"] = distances[np.arange(len(comments)), cluster_ids]
    comments["stance_label"] = comments["cluster_id"].map(label_map)
    comments["stance_confidence"] = np.clip(
        0.35 + (0.35 * comments["assignment_confidence"].to_numpy()) + (0.30 * distance_margin),
        0.0,
        0.999,
    )

    topic_summaries = build_topic_summaries(
        topic=topic,
        interpretation=interpretation,
        comments=comments,
        vectorizer=vectorizer,
        matrix=matrix,
        label_map=label_map,
        counts=counts,
        outcome=outcome,
        minority_share=minority_share,
    )

    comment_rows = [
        (
            str(row.comment_id),
            int(topic.topic_id),
            str(row.stance_label),
            round(float(row.stance_confidence), 6),
            int(row.cluster_id),
            round(float(row.prototype_distance), 6),
            1,
        )
        for row in comments.itertuples(index=False)
    ]

    topic_outcome = TopicOutcome(
        topic_id=topic.topic_id,
        topic_label=topic.topic_label,
        interpretation=interpretation,
        analyzed_comments=len(comments),
        dominant_comments=int(counts[dominant_cluster]),
        minority_comments=int(counts[minority_cluster]),
        outcome=outcome,
        minority_share=round(minority_share, 4),
        silhouette_score=round(silhouette, 4),
        centroid_similarity=round(centroid_similarity, 4),
    )

    return {
        "topic_outcome": topic_outcome,
        "comment_stances": comment_rows,
        "topic_summaries": topic_summaries,
    }


def build_vectorizer(comment_count: int, topic_terms: set[str]) -> TfidfVectorizer:
    min_df = 2
    if comment_count >= 150:
        min_df = 3
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.85,
        max_features=2500,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]+\b",
        preprocessor=None,
        tokenizer=None,
    )


def compute_silhouette(matrix, cluster_ids: np.ndarray) -> float:
    if len(set(cluster_ids.tolist())) < 2:
        return 0.0
    if min(np.bincount(cluster_ids, minlength=2)) < 2:
        return 0.0
    return float(silhouette_score(matrix, cluster_ids, metric="cosine"))


def compute_centroid_similarity(centers: np.ndarray) -> float:
    normalized = normalize(centers)
    return float(np.clip(normalized[0] @ normalized[1], -1.0, 1.0))


def build_topic_summaries(
    topic: TopicSelection,
    interpretation: str,
    comments: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix,
    label_map: dict[int, str],
    counts: np.ndarray,
    outcome: str,
    minority_share: float,
) -> list[tuple]:
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    cluster_feature_strength = []
    for cluster_id in (0, 1):
        cluster_matrix = matrix[comments["cluster_id"].to_numpy() == cluster_id]
        mean_vector = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        cluster_feature_strength.append(mean_vector)

    now_utc = int(datetime.now(tz=UTC).timestamp())
    topic_terms = extract_topic_terms(topic.topic_label)
    summaries: list[tuple] = []

    for cluster_id in (0, 1):
        stance_label = label_map[cluster_id]
        other_cluster = 1 - cluster_id
        distinctive_terms = top_distinctive_terms(
            feature_names=feature_names,
            cluster_vector=cluster_feature_strength[cluster_id],
            other_vector=cluster_feature_strength[other_cluster],
            excluded_terms=topic_terms | GENERIC_EXTRACTIVE_TERMS,
        )
        cluster_comments = comments[comments["cluster_id"] == cluster_id].copy()
        cluster_comments["representative_rank"] = (
            cluster_comments["assignment_confidence"] * 0.45
            + normalize_series(-cluster_comments["prototype_distance"]) * 0.40
            + normalize_series(cluster_comments["score"]) * 0.15
        )
        cluster_comments = cluster_comments.sort_values(
            ["representative_rank", "assignment_confidence", "score", "comment_id"],
            ascending=[False, False, False, True],
            kind="stable",
        )
        representative_ids = cluster_comments["comment_id"].head(5).tolist()
        summary_text = build_summary_text(
            topic_id=topic.topic_id,
            interpretation=interpretation,
            stance_label=stance_label,
            comment_count=int(counts[cluster_id]),
            total_comments=int(counts.sum()),
            minority_share=minority_share,
            distinctive_terms=distinctive_terms,
            representative_snippets=build_representative_snippets(cluster_comments),
            outcome=outcome,
        )
        summaries.append(
            (
                int(topic.topic_id),
                stance_label,
                summary_text,
                json.dumps(representative_ids, ensure_ascii=True),
                now_utc,
            )
        )

    return summaries


def top_distinctive_terms(
    feature_names: np.ndarray,
    cluster_vector: np.ndarray,
    other_vector: np.ndarray,
    excluded_terms: set[str],
    limit: int = 6,
) -> list[str]:
    delta = cluster_vector - other_vector
    ranked_indices = np.argsort(delta)[::-1]
    chosen: list[str] = []
    for index in ranked_indices:
        term = str(feature_names[index])
        if delta[index] <= 0:
            break
        if should_skip_term(term, excluded_terms):
            continue
        chosen.append(term)
        if len(chosen) >= limit:
            break
    if chosen:
        return chosen
    fallback_indices = np.argsort(cluster_vector)[::-1]
    for index in fallback_indices:
        term = str(feature_names[index])
        if should_skip_term(term, excluded_terms):
            continue
        chosen.append(term)
        if len(chosen) >= min(limit, 4):
            break
    return chosen


def should_skip_term(term: str, excluded_terms: set[str]) -> bool:
    tokens = [token for token in term.split() if token]
    if not tokens:
        return True
    if any(token in excluded_terms for token in tokens):
        return True
    if all(not any(char.isalpha() for char in token) for token in tokens):
        return True
    if term in excluded_terms:
        return True
    return False


def build_summary_text(
    topic_id: int,
    interpretation: str,
    stance_label: str,
    comment_count: int,
    total_comments: int,
    minority_share: float,
    distinctive_terms: list[str],
    representative_snippets: list[str],
    outcome: str,
) -> str:
    share_text = f"{comment_count}/{total_comments} comments"
    term_text = ", ".join(distinctive_terms[:5]) if distinctive_terms else "no strongly distinctive phrases"
    sample_text = " ".join(f"'{snippet}'" for snippet in representative_snippets[:2])
    if not sample_text:
        sample_text = "No concise representative snippet survived cleaning."

    if stance_label == "dominant_position":
        if outcome == "stance_split":
            return (
                f"Dominant position for topic {topic_id} ({interpretation}), covering {share_text}. "
                f"Key argument signals: {term_text}. Representative evidence: {sample_text}"
            )
        return (
            f"Dominant discussion position for topic {topic_id} ({interpretation}), covering {share_text}. "
            f"The opposing/caveat side exists, but this split is weak or overlapping overall "
            f"(smaller-side share {minority_share:.1%}). Key argument signals: {term_text}. "
            f"Representative evidence: {sample_text}"
        )

    if outcome == "stance_split":
        return (
            f"Opposing or caveat position for topic {topic_id} ({interpretation}), covering {share_text}. "
            f"This side differs from the dominant cluster rather than merely repeating it. "
            f"Key argument signals: {term_text}. Representative evidence: {sample_text}"
        )
    return (
        f"Opposing or caveat position for topic {topic_id} ({interpretation}), covering {share_text}. "
        f"Because the split is weak or overlapping, treat this as a caveat/alternative argument rather than "
        f"a clean opposition camp. Key argument signals: {term_text}. Representative evidence: {sample_text}"
    )


def build_representative_snippets(cluster_comments: pd.DataFrame, limit: int = 2) -> list[str]:
    snippets: list[str] = []
    for text in cluster_comments["clean_text"].astype(str).tolist():
        snippet = clean_summary_snippet(text)
        if not snippet:
            continue
        snippets.append(snippet)
        if len(snippets) >= limit:
            break
    return snippets


def clean_summary_snippet(text: str, max_chars: int = 180) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return ""
    sentence_match = re.search(r"^(.{40,}?[.!?])(?:\s|$)", collapsed)
    snippet = sentence_match.group(1) if sentence_match else collapsed
    snippet = snippet[:max_chars].rsplit(" ", 1)[0].strip()
    return snippet.strip("\"' ")


def extract_topic_terms(text: str) -> set[str]:
    terms = {match.group(0) for match in TOKEN_RE.finditer(text.lower())}
    terms |= {term.rstrip("s") for term in terms if len(term) > 3}
    return terms


def normalize_series(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    spread = values.max() - values.min()
    if spread <= 1e-9:
        return pd.Series(np.ones(len(values)), index=series.index, dtype=float)
    return (values - values.min()) / spread


def persist_outputs(
    final_major_topic_ids: list[int],
    comment_stance_rows: list[tuple],
    summary_rows: list[tuple],
    analyzed_topics: list[TopicOutcome],
    skipped_topics: list[dict],
    checkpoint_file,
) -> StanceCheckpoint:
    now_utc = int(datetime.now(tz=UTC).timestamp())
    selected_topic_labels = [
        f"{topic.topic_id}:{topic.topic_label}"
        for topic in analyzed_topics
    ]

    with connect_db() as connection:
        connection.execute("DELETE FROM comment_stances")
        connection.execute("DELETE FROM topic_argument_summaries")
        if final_major_topic_ids:
            placeholders = ", ".join("?" for _ in final_major_topic_ids)
            connection.execute("UPDATE document_topics SET is_major_topic = 0")
            connection.execute(
                f"UPDATE document_topics SET is_major_topic = 1 WHERE topic_id IN ({placeholders})",
                tuple(final_major_topic_ids),
            )
        else:
            connection.execute("UPDATE document_topics SET is_major_topic = 0")

        connection.executemany(
            """
            INSERT INTO comment_stances (
                comment_id,
                topic_id,
                stance_label,
                stance_confidence,
                cluster_id,
                prototype_distance,
                is_substantive,
                created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [row + (now_utc,) for row in comment_stance_rows],
        )
        connection.executemany(
            """
            INSERT INTO topic_argument_summaries (
                topic_id,
                stance_label,
                summary_text,
                representative_comment_ids_json,
                generated_at_utc
            ) VALUES (?, ?, ?, ?, ?)
            """,
            summary_rows,
        )

        checkpoint = StanceCheckpoint(
            stage="fit_stance",
            generated_at_utc=now_utc,
            subreddit=SUBREDDIT,
            selected_topic_ids=list(final_major_topic_ids),
            selected_topic_labels=selected_topic_labels,
            skipped_topics=skipped_topics,
            analyzed_topics=[asdict(topic) for topic in analyzed_topics],
            comment_stances_written=len(comment_stance_rows),
            topic_summaries_written=len(summary_rows),
            status="completed",
        )
        payload = json.dumps(asdict(checkpoint), ensure_ascii=True)
        checkpoint_file.write_text(json.dumps(asdict(checkpoint), indent=2), encoding="utf-8")
        connection.execute(
            """
            INSERT INTO pipeline_checkpoints (stage_name, status, payload_json, updated_at_utc)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(stage_name) DO UPDATE SET
                status = excluded.status,
                payload_json = excluded.payload_json,
                updated_at_utc = excluded.updated_at_utc
            """,
            ("fit_stance", "completed", payload, now_utc),
        )

    return checkpoint
