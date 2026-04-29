from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_CORPUS_RULE = (
    "Select the latest subreddit_meta window whose observed documents include "
    "posts and comments for the window range and whose provenance looks fully "
    "trusted for a corpus freeze. Windows with observed data but stale summary "
    "counts or a running comment-ingestion checkpoint remain visible for "
    "exploration, but are not eligible as the default RAG corpus."
)

DEFAULT_MANIFEST_FILENAME = "corpus_manifest_v1.json"
DEFAULT_CHUNK_MANIFEST_FILENAME = "chunk_manifest_v1.json"
DEFAULT_CHUNK_ARTIFACT_FILENAME = "default_rag_chunks_v1.jsonl"
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_BATCH_SIZE = 128
DEFAULT_EXPECTED_FROZEN_CHUNK_COUNT = 313615
DEFAULT_EMBEDDING_ARTIFACT_FILENAME = "default_rag_embeddings_v1.sqlite"
DEFAULT_EMBEDDING_MANIFEST_FILENAME = "embedding_manifest_v1.json"
DEFAULT_FAISS_INDEX_FILENAME = "default_rag_dense_v1.faiss"
DEFAULT_EVAL_FILENAME = "rag_eval_v1.jsonl"
DEFAULT_EVAL_MANIFEST_FILENAME = "rag_eval_v1_manifest.json"
DEFAULT_RUNS_DIRNAME = "runs"
DEFAULT_EVAL_RUNS_DIRNAME = "eval_runs"
DEFAULT_INDIAN_LANGUAGE_DIRNAME = "indian_language"
DEFAULT_INDIAN_LANGUAGE_RUNS_DIRNAME = "indian_language_runs"
DEFAULT_HINDI_EVAL_FILENAME = "hindi_translation_eval_v1.jsonl"
DEFAULT_HINDI_EVAL_MANIFEST_FILENAME = "hindi_translation_eval_v1_manifest.json"
DEFAULT_HINDI_MANUAL_REVIEW_TEMPLATE_FILENAME = "hindi_manual_review_template_v1.csv"
DEFAULT_DENSE_TOP_K = 8
DEFAULT_LEXICAL_TOP_K = 8
DEFAULT_HYBRID_FINAL_TOP_K = 5
DEFAULT_RRF_CONSTANT = 60
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_LLM_MAX_TOKENS = 700
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_MULTILINGUAL_BERT_SCORE_MODEL = "bert-base-multilingual-cased"
DEFAULT_FACTUAL_DENSE_TOP_K = 8
DEFAULT_FACTUAL_LEXICAL_TOP_K = 8
DEFAULT_FACTUAL_HYBRID_FINAL_TOP_K = 7
DEFAULT_OPINION_DENSE_TOP_K = 14
DEFAULT_OPINION_LEXICAL_TOP_K = 14
DEFAULT_OPINION_HYBRID_FINAL_TOP_K = 7
DEFAULT_ADVERSARIAL_DENSE_TOP_K = 4
DEFAULT_ADVERSARIAL_LEXICAL_TOP_K = 4
DEFAULT_ADVERSARIAL_HYBRID_FINAL_TOP_K = 3
DEFAULT_ADVERSARIAL_MIN_DENSE_SCORE = 0.3
DEFAULT_ADVERSARIAL_MIN_RRF_SCORE = 0.02

FACTUAL_QUERY_TYPE = "factual"
OPINION_SUMMARY_QUERY_TYPE = "opinion-summary"
ADVERSARIAL_NO_ANSWER_QUERY_TYPE = "adversarial/no-answer"


@dataclass(frozen=True)
class RetrievalDefaults:
    dense_top_k: int
    lexical_top_k: int
    hybrid_final_top_k: int
    rrf_constant: int


@dataclass(frozen=True)
class QueryClassRetrievalProfile:
    query_type: str
    retrieval_mode: str
    dense_top_k: int
    lexical_top_k: int
    hybrid_final_top_k: int
    rrf_constant: int
    min_dense_score: float | None = None
    min_rrf_score: float | None = None
    guardrail_notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Part2Paths:
    repo_root: Path
    part2_root: Path
    src_root: Path
    scripts_dir: Path
    tests_dir: Path
    data_dir: Path
    indian_language_dir: Path
    chunks_dir: Path
    embeddings_dir: Path
    manifests_dir: Path
    eval_dir: Path
    indices_dir: Path
    runs_dir: Path
    eval_runs_dir: Path
    indian_language_runs_dir: Path
    reports_dir: Path
    part1_db_path: Path


def get_paths() -> Part2Paths:
    part2_root = Path(__file__).resolve().parents[2]
    repo_root = part2_root.parent
    data_dir = part2_root / "data"
    return Part2Paths(
        repo_root=repo_root,
        part2_root=part2_root,
        src_root=part2_root / "src",
        scripts_dir=part2_root / "scripts",
        tests_dir=part2_root / "tests",
        data_dir=data_dir,
        indian_language_dir=data_dir / DEFAULT_INDIAN_LANGUAGE_DIRNAME,
        chunks_dir=data_dir / "chunks",
        embeddings_dir=data_dir / "embeddings",
        manifests_dir=data_dir / "manifests",
        eval_dir=data_dir / "eval",
        indices_dir=data_dir / "indices",
        runs_dir=data_dir / DEFAULT_RUNS_DIRNAME,
        eval_runs_dir=data_dir / DEFAULT_EVAL_RUNS_DIRNAME,
        indian_language_runs_dir=data_dir / DEFAULT_INDIAN_LANGUAGE_RUNS_DIRNAME,
        reports_dir=part2_root / "reports",
        part1_db_path=repo_root / "Part1" / "data" / "fitness_part1.sqlite",
    )


def get_default_manifest_path() -> Path:
    return get_paths().manifests_dir / DEFAULT_MANIFEST_FILENAME


def get_default_chunk_manifest_path() -> Path:
    return get_paths().manifests_dir / DEFAULT_CHUNK_MANIFEST_FILENAME


def get_default_chunk_artifact_path() -> Path:
    return get_paths().chunks_dir / DEFAULT_CHUNK_ARTIFACT_FILENAME


def get_default_embedding_artifact_path() -> Path:
    return get_paths().embeddings_dir / DEFAULT_EMBEDDING_ARTIFACT_FILENAME


def get_default_embedding_manifest_path() -> Path:
    return get_paths().manifests_dir / DEFAULT_EMBEDDING_MANIFEST_FILENAME


def get_default_faiss_index_path() -> Path:
    return get_paths().indices_dir / DEFAULT_FAISS_INDEX_FILENAME


def get_default_eval_path() -> Path:
    return get_paths().eval_dir / DEFAULT_EVAL_FILENAME


def get_default_eval_manifest_path() -> Path:
    return get_paths().eval_dir / DEFAULT_EVAL_MANIFEST_FILENAME


def get_default_hindi_eval_path() -> Path:
    return get_paths().indian_language_dir / DEFAULT_HINDI_EVAL_FILENAME


def get_default_hindi_eval_manifest_path() -> Path:
    return get_paths().indian_language_dir / DEFAULT_HINDI_EVAL_MANIFEST_FILENAME


def get_default_hindi_manual_review_template_path() -> Path:
    return (
        get_paths().indian_language_dir
        / DEFAULT_HINDI_MANUAL_REVIEW_TEMPLATE_FILENAME
    )


def get_default_runs_dir() -> Path:
    return get_paths().runs_dir


def get_default_eval_runs_dir() -> Path:
    return get_paths().eval_runs_dir


def get_default_indian_language_runs_dir() -> Path:
    return get_paths().indian_language_runs_dir


def get_default_part1_db_path() -> Path:
    return get_paths().part1_db_path


def get_retrieval_defaults() -> RetrievalDefaults:
    return RetrievalDefaults(
        dense_top_k=DEFAULT_DENSE_TOP_K,
        lexical_top_k=DEFAULT_LEXICAL_TOP_K,
        hybrid_final_top_k=DEFAULT_HYBRID_FINAL_TOP_K,
        rrf_constant=DEFAULT_RRF_CONSTANT,
    )


def get_default_query_abbreviation_map() -> dict[str, str]:
    return {
        "ppl": "push pull legs",
        "5x5": "five by five strength program",
        "1rm": "one rep max",
        "pr": "personal record",
        "bw": "bodyweight",
        "rpe": "rate of perceived exertion",
        "tdee": "total daily energy expenditure",
        "bmr": "basal metabolic rate",
    }


def get_query_class_retrieval_profiles() -> dict[str, QueryClassRetrievalProfile]:
    return {
        FACTUAL_QUERY_TYPE: QueryClassRetrievalProfile(
            query_type=FACTUAL_QUERY_TYPE,
            retrieval_mode="hybrid",
            dense_top_k=DEFAULT_FACTUAL_DENSE_TOP_K,
            lexical_top_k=DEFAULT_FACTUAL_LEXICAL_TOP_K,
            hybrid_final_top_k=DEFAULT_FACTUAL_HYBRID_FINAL_TOP_K,
            rrf_constant=DEFAULT_RRF_CONSTANT,
            guardrail_notes=("balanced_hybrid_defaults",),
        ),
        OPINION_SUMMARY_QUERY_TYPE: QueryClassRetrievalProfile(
            query_type=OPINION_SUMMARY_QUERY_TYPE,
            retrieval_mode="hybrid",
            dense_top_k=DEFAULT_OPINION_DENSE_TOP_K,
            lexical_top_k=DEFAULT_OPINION_LEXICAL_TOP_K,
            hybrid_final_top_k=DEFAULT_OPINION_HYBRID_FINAL_TOP_K,
            rrf_constant=DEFAULT_RRF_CONSTANT,
            guardrail_notes=("larger_candidate_pools_for_broader_sentiment_coverage",),
        ),
        ADVERSARIAL_NO_ANSWER_QUERY_TYPE: QueryClassRetrievalProfile(
            query_type=ADVERSARIAL_NO_ANSWER_QUERY_TYPE,
            retrieval_mode="hybrid",
            dense_top_k=DEFAULT_ADVERSARIAL_DENSE_TOP_K,
            lexical_top_k=DEFAULT_ADVERSARIAL_LEXICAL_TOP_K,
            hybrid_final_top_k=DEFAULT_ADVERSARIAL_HYBRID_FINAL_TOP_K,
            rrf_constant=DEFAULT_RRF_CONSTANT,
            min_dense_score=DEFAULT_ADVERSARIAL_MIN_DENSE_SCORE,
            min_rrf_score=DEFAULT_ADVERSARIAL_MIN_RRF_SCORE,
            guardrail_notes=("smaller_candidate_pool", "prepare_for_future_abstention_guardrails"),
        ),
    }


def get_query_class_retrieval_profile(query_type: str) -> QueryClassRetrievalProfile:
    profiles = get_query_class_retrieval_profiles()
    try:
        return profiles[query_type]
    except KeyError as exc:
        supported = ", ".join(sorted(profiles))
        raise KeyError(
            f"Unsupported query_type={query_type!r}. Supported query types: {supported}"
        ) from exc
