from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from part2_rag.config import (
    ADVERSARIAL_NO_ANSWER_QUERY_TYPE,
    FACTUAL_QUERY_TYPE,
    OPINION_SUMMARY_QUERY_TYPE,
    QueryClassRetrievalProfile,
    get_default_query_abbreviation_map,
    get_query_class_retrieval_profile,
)
from part2_rag.embedding_index import EmbeddingIndexBuildError, load_embedding_manifest, load_json_file
from part2_rag.retrieval import (
    RetrievalConfig,
    RetrievalError,
    RetrievalResult,
    get_default_retrieval_config,
    retrieve,
)


WHITESPACE_PATTERN = re.compile(r"\s+")

OPINION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("opinion:what_do_people_think", re.compile(r"\bwhat do people think\b")),
    ("opinion:what_does_subreddit_say", re.compile(r"\bwhat does (?:this )?subreddit say\b")),
    ("opinion:community_consensus", re.compile(r"\b(?:community|general|overall)\s+consensus\b")),
    ("opinion:community_opinion", re.compile(r"\b(?:community|people|reddit|subreddit)\s+(?:opinion|opinions|view|views|take|takes)\b")),
    ("opinion:summary_signal", re.compile(r"\b(?:summari[sz]e|overview|common advice|common theme|pros and cons)\b")),
    ("opinion:experience_signal", re.compile(r"\b(?:experiences|experience with|anecdot(?:e|es)|success stories)\b")),
    ("opinion:is_it_worth_it", re.compile(r"\bis [a-z0-9\s-]+ worth it\b")),
)

ADVERSARIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("adversarial:prompt_injection", re.compile(r"\b(?:ignore|disregard|override)\b.*\b(?:instructions|system prompt|rules)\b")),
    ("adversarial:unsupported_source_request", re.compile(r"\b(?:this subreddit|r/fitness|reddit)\b.*\b(?:quantum mechanics|stock market|election|politics|weather|python programming|javascript|cooking|world war)\b")),
    ("adversarial:out_of_domain_topic", re.compile(r"\b(?:quantum mechanics|stock market|election|politics|weather|python programming|javascript|cooking|world war)\b")),
    ("adversarial:impossible_system_request", re.compile(r"\b(?:show|reveal|print)\b.*\b(?:system prompt|hidden instructions)\b")),
)

FITNESS_DOMAIN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fitness:training", re.compile(r"\b(?:workout|routine|program|lift|lifting|strength|hypertrophy|cardio|cut|bulk|recomp)\b")),
    ("fitness:nutrition", re.compile(r"\b(?:protein|calorie|calories|diet|macros|tdee|bmr|cutting|bulking)\b")),
    (
        "fitness:metrics",
        re.compile(
            r"\b(?:1rm|one rep max|pr|personal record|rpe|rate of perceived exertion|bodyweight|bw|sets|reps)\b"
        ),
    ),
)

DOMAIN_CONCEPT_EXPANSIONS: tuple[tuple[str, re.Pattern[str], tuple[str, ...]], ...] = (
    (
        "concept:maintenance_calories",
        re.compile(r"\bmaintenance calories?\b"),
        (
            "tdee",
            "bmr",
            "calorie calculator",
            "track weight",
            "recalibrate",
        ),
    ),
    (
        "concept:warmup",
        re.compile(r"\bwarm\s*up\b|\bwarmup\b"),
        (
            "easy sets",
            "lighter sets",
            "working set",
            "first working set",
            "ramp up",
        ),
    ),
    (
        "concept:cooldown",
        re.compile(r"\bcool\s*down\b|\bcooldown\b"),
        (
            "cool down optional",
            "cool down unnecessary",
        ),
    ),
    (
        "concept:fat_loss_muscle_retention",
        re.compile(
            r"\blose fat\b.*\blos(?:e|ing)(?:\s+much)?\s+muscle\b|\blos(?:e|ing)(?:\s+much)?\s+muscle\b.*\blose fat\b"
        ),
        (
            "calorie deficit",
            "high protein",
            "continue lifting",
            "strength training",
            "cutting",
        ),
    ),
    (
        "concept:body_recomposition",
        re.compile(r"\bbody recomposition\b|\brecomp(?:ing)?\b"),
        (
            "recomp",
            "recomping",
        ),
    ),
)
BODY_RECOMPOSITION_OPINION_TAIL_TERMS = (
    "bulk cut",
    "maintenance calories",
    "beginners",
)


def _contains_expansion_term(text: str, term: str) -> bool:
    normalized_term = term.lower().strip()
    if not normalized_term:
        return True
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])")
    return pattern.search(text) is not None


class QueryClassificationError(ValueError):
    """Raised when query classification inputs are invalid."""


@dataclass(frozen=True)
class QueryClassificationResult:
    query: str
    query_type: str
    confidence: float
    matched_rules: tuple[str, ...]
    normalized_query: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryRoutingResult:
    classification: QueryClassificationResult
    retrieval_mode_used: str
    effective_retrieval_config: dict[str, Any]
    retrieval_results: tuple[RetrievalResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.to_dict(),
            "retrieval_mode_used": self.retrieval_mode_used,
            "effective_retrieval_config": dict(self.effective_retrieval_config),
            "retrieval_results": [result.to_dict() for result in self.retrieval_results],
        }


def normalize_query_text(query: str) -> str:
    normalized = WHITESPACE_PATTERN.sub(" ", query.strip().lower())
    if not normalized:
        raise QueryClassificationError("Query text must not be empty.")
    return normalized


def expand_query_abbreviations(
    query: str,
    abbreviation_map: Mapping[str, str] | None = None,
) -> tuple[str, tuple[str, ...]]:
    normalized = normalize_query_text(query)
    expansions = abbreviation_map or get_default_query_abbreviation_map()

    expanded = normalized
    matched_rules: list[str] = []
    for abbreviation, replacement in expansions.items():
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(abbreviation.lower())}(?![a-z0-9])")
        if pattern.search(expanded) is None:
            continue
        expanded = pattern.sub(replacement.lower(), expanded)
        matched_rules.append(f"abbrev:{abbreviation.lower()}->{replacement.lower()}")
    expanded = WHITESPACE_PATTERN.sub(" ", expanded).strip()
    return expanded, tuple(matched_rules)


def expand_query_domain_concepts(query: str) -> tuple[str, tuple[str, ...]]:
    expanded = normalize_query_text(query)
    matched_rules: list[str] = []

    for rule_name, pattern, expansion_terms in DOMAIN_CONCEPT_EXPANSIONS:
        if pattern.search(expanded) is None:
            continue
        if (
            rule_name == "concept:maintenance_calories"
            and re.search(r"\bbody recomposition\b|\brecomp(?:ing)?\b", expanded) is not None
        ):
            continue

        terms_to_append = [
            term.lower()
            for term in expansion_terms
            if not _contains_expansion_term(expanded, term)
        ]
        if not terms_to_append:
            continue
        expanded = f"{expanded} {' '.join(terms_to_append)}"
        matched_rules.append(rule_name)

    expanded = WHITESPACE_PATTERN.sub(" ", expanded).strip()
    return expanded, tuple(matched_rules)


def _apply_query_type_specific_expansions(
    normalized_query: str,
    *,
    query_type: str,
) -> tuple[str, tuple[str, ...]]:
    if (
        query_type != OPINION_SUMMARY_QUERY_TYPE
        or re.search(r"\bbody recomposition\b|\brecomp(?:ing)?\b", normalized_query) is None
    ):
        return normalized_query, ()

    terms_to_append = [
        term.lower()
        for term in BODY_RECOMPOSITION_OPINION_TAIL_TERMS
        if not _contains_expansion_term(normalized_query, term)
    ]
    if not terms_to_append:
        return normalized_query, ()
    expanded_query = f"{normalized_query} {' '.join(terms_to_append)}"
    expanded_query = WHITESPACE_PATTERN.sub(" ", expanded_query).strip()
    return expanded_query, ("concept:body_recomposition_opinion_tail",)


def classify_query(
    query: str,
    *,
    abbreviation_map: Mapping[str, str] | None = None,
) -> QueryClassificationResult:
    normalized_query, abbreviation_rules = expand_query_abbreviations(
        query,
        abbreviation_map=abbreviation_map,
    )
    normalized_query, concept_rules = expand_query_domain_concepts(normalized_query)
    matched_rules = list(abbreviation_rules) + list(concept_rules)

    opinion_rules = [name for name, pattern in OPINION_PATTERNS if pattern.search(normalized_query)]
    adversarial_rules = [
        name for name, pattern in ADVERSARIAL_PATTERNS if pattern.search(normalized_query)
    ]
    fitness_rules = [
        name for name, pattern in FITNESS_DOMAIN_PATTERNS if pattern.search(normalized_query)
    ]

    if adversarial_rules:
        matched_rules.extend(adversarial_rules)
        if not fitness_rules:
            matched_rules.append("adversarial:no_clear_fitness_signal")
        confidence = 0.86 if len(adversarial_rules) >= 2 or not fitness_rules else 0.72
        normalized_query, query_type_rules = _apply_query_type_specific_expansions(
            normalized_query,
            query_type=ADVERSARIAL_NO_ANSWER_QUERY_TYPE,
        )
        matched_rules.extend(query_type_rules)
        return QueryClassificationResult(
            query=query,
            query_type=ADVERSARIAL_NO_ANSWER_QUERY_TYPE,
            confidence=confidence,
            matched_rules=tuple(matched_rules),
            normalized_query=normalized_query,
        )

    if opinion_rules:
        matched_rules.extend(opinion_rules)
        if fitness_rules:
            matched_rules.extend(fitness_rules)
        confidence = 0.82 if len(opinion_rules) >= 2 else 0.72
        normalized_query, query_type_rules = _apply_query_type_specific_expansions(
            normalized_query,
            query_type=OPINION_SUMMARY_QUERY_TYPE,
        )
        matched_rules.extend(query_type_rules)
        return QueryClassificationResult(
            query=query,
            query_type=OPINION_SUMMARY_QUERY_TYPE,
            confidence=confidence,
            matched_rules=tuple(matched_rules),
            normalized_query=normalized_query,
        )

    matched_rules.extend(fitness_rules or ("fallback:default_factual",))
    confidence = 0.64 if fitness_rules else 0.55
    normalized_query, query_type_rules = _apply_query_type_specific_expansions(
        normalized_query,
        query_type=FACTUAL_QUERY_TYPE,
    )
    matched_rules.extend(query_type_rules)
    return QueryClassificationResult(
        query=query,
        query_type=FACTUAL_QUERY_TYPE,
        confidence=confidence,
        matched_rules=tuple(matched_rules),
        normalized_query=normalized_query,
    )


def build_effective_retrieval_config(
    query_type: str,
    *,
    base_config: RetrievalConfig | None = None,
) -> tuple[str, QueryClassRetrievalProfile, RetrievalConfig]:
    profile = get_query_class_retrieval_profile(query_type)
    source = base_config or get_default_retrieval_config()
    effective_config = RetrievalConfig(
        dense_top_k=profile.dense_top_k,
        lexical_top_k=profile.lexical_top_k,
        hybrid_final_top_k=profile.hybrid_final_top_k,
        rrf_constant=profile.rrf_constant,
        faiss_index_path=source.faiss_index_path,
        embedding_store_path=source.embedding_store_path,
        chunk_artifact_path=source.chunk_artifact_path,
        corpus_manifest_path=source.corpus_manifest_path,
        chunk_manifest_path=source.chunk_manifest_path,
        embedding_manifest_path=source.embedding_manifest_path,
        part1_db_path=source.part1_db_path,
    )
    return profile.retrieval_mode, profile, effective_config


def route_and_retrieve(
    query: str,
    *,
    base_config: RetrievalConfig | None = None,
    abbreviation_map: Mapping[str, str] | None = None,
    retrieval_fn: Callable[[str], Sequence[RetrievalResult]] | None = None,
) -> QueryRoutingResult:
    classification = classify_query(query, abbreviation_map=abbreviation_map)
    retrieval_mode, profile, effective_config = build_effective_retrieval_config(
        classification.query_type,
        base_config=base_config,
    )

    if retrieval_fn is None:
        results = retrieve(
            classification.normalized_query,
            mode=retrieval_mode,
            config=effective_config,
        )
    else:
        results = retrieval_fn(classification.normalized_query)

    effective_retrieval_config = {
        "query_type": profile.query_type,
        "retrieval_mode": profile.retrieval_mode,
        "dense_top_k": effective_config.dense_top_k,
        "lexical_top_k": effective_config.lexical_top_k,
        "hybrid_final_top_k": effective_config.hybrid_final_top_k,
        "rrf_constant": effective_config.rrf_constant,
        "min_dense_score": profile.min_dense_score,
        "min_rrf_score": profile.min_rrf_score,
        "guardrail_notes": list(profile.guardrail_notes),
    }
    return QueryRoutingResult(
        classification=classification,
        retrieval_mode_used=retrieval_mode,
        effective_retrieval_config=effective_retrieval_config,
        retrieval_results=tuple(results),
    )


def format_classification_and_retrieval(
    routed: QueryRoutingResult,
) -> str:
    lines = [
        "Retrieval-only QA debug",
        "",
        "classification:",
        f"  query: {routed.classification.query}",
        f"  normalized_query: {routed.classification.normalized_query}",
        f"  query_type: {routed.classification.query_type}",
        f"  confidence: {routed.classification.confidence:.2f}",
        "  matched_rules: "
        + (", ".join(routed.classification.matched_rules) or "(none)"),
        "",
        "effective_retrieval_settings:",
    ]
    for key, value in routed.effective_retrieval_config.items():
        lines.append(f"  {key}: {value}")

    if not routed.retrieval_results:
        lines.extend(["", "top_retrieved_chunks:", "  (none)"])
        return "\n".join(lines)

    lines.extend(["", "top_retrieved_chunks:"])
    for result in routed.retrieval_results:
        lines.append(
            f"  [{result.rank}] chunk_id={result.chunk_id} | document_id={result.document_id} "
            f"| source_type={result.source_type} | chunk_index={result.chunk_index}"
        )
        lines.append(
            f"      retrieval_source={result.retrieval_source} | score={result.score:.6f} "
            f"| title={result.title!r}"
        )
        lines.append(
            f"      dense_rank={result.dense_rank} | dense_score={result.dense_score} "
            f"| lexical_rank={result.lexical_rank} | lexical_score={result.lexical_score} "
            f"| rrf_score={result.rrf_score}"
        )
        lines.append(f"      snippet: {result.snippet}")
    return "\n".join(lines)


def build_classification_and_retrieval_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify a query, expand abbreviations, select class-specific retrieval defaults, and run retrieval."
    )
    parser.add_argument("--query", required=True, help="Query text to classify and retrieve against.")
    parser.add_argument("--faiss-index-path", type=Path, default=None)
    parser.add_argument("--embedding-store-path", type=Path, default=None)
    parser.add_argument("--chunk-artifact-path", type=Path, default=None)
    parser.add_argument("--embedding-manifest-path", type=Path, default=None)
    parser.add_argument("--chunk-manifest-path", type=Path, default=None)
    parser.add_argument("--corpus-manifest-path", type=Path, default=None)
    parser.add_argument("--part1-db-path", type=Path, default=None)
    return parser


def _config_from_optional_args(args: argparse.Namespace) -> RetrievalConfig:
    defaults = get_default_retrieval_config()
    return RetrievalConfig(
        dense_top_k=defaults.dense_top_k,
        lexical_top_k=defaults.lexical_top_k,
        hybrid_final_top_k=defaults.hybrid_final_top_k,
        rrf_constant=defaults.rrf_constant,
        faiss_index_path=(args.faiss_index_path or defaults.faiss_index_path).resolve(),
        embedding_store_path=(args.embedding_store_path or defaults.embedding_store_path).resolve(),
        chunk_artifact_path=(args.chunk_artifact_path or defaults.chunk_artifact_path).resolve(),
        corpus_manifest_path=(args.corpus_manifest_path or defaults.corpus_manifest_path).resolve(),
        chunk_manifest_path=(args.chunk_manifest_path or defaults.chunk_manifest_path).resolve(),
        embedding_manifest_path=(
            args.embedding_manifest_path or defaults.embedding_manifest_path
        ).resolve(),
        part1_db_path=(args.part1_db_path or defaults.part1_db_path).resolve(),
    )


def classification_and_retrieval_debug_main() -> int:
    parser = build_classification_and_retrieval_arg_parser()
    args = parser.parse_args()
    config = _config_from_optional_args(args)
    try:
        chunk_manifest = load_json_file(config.chunk_manifest_path)
        total_chunks = int(chunk_manifest["chunk_counts"]["total_chunks"])
        embedding_manifest = load_embedding_manifest(config.embedding_manifest_path)
        if embedding_manifest.embedding_count != total_chunks:
            raise RetrievalError(
                "Embedding manifest count does not match frozen chunk manifest count."
            )
        routed = route_and_retrieve(args.query, base_config=config)
    except (
        EmbeddingIndexBuildError,
        KeyError,
        TypeError,
        ValueError,
        QueryClassificationError,
        RetrievalError,
    ) as exc:
        parser.exit(status=1, message=f"Classification + retrieval failed: {exc}\n")
    print(format_classification_and_retrieval(routed))
    return 0


__all__ = [
    "classification_and_retrieval_debug_main",
    "QueryClassificationError",
    "QueryClassificationResult",
    "QueryRoutingResult",
    "build_classification_and_retrieval_arg_parser",
    "build_effective_retrieval_config",
    "classify_query",
    "expand_query_abbreviations",
    "expand_query_domain_concepts",
    "format_classification_and_retrieval",
    "normalize_query_text",
    "route_and_retrieve",
]
