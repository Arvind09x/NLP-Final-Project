from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


PART2_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PART2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part2_rag.answer_generation import (
    AnswerGenerationError,
    build_prompt,
    format_answer_generation_result,
    generate_grounded_answer,
)
from part2_rag.config import get_default_runs_dir
from part2_rag.llm_providers import (
    ProviderConfigurationError,
    ProviderInvocationError,
    get_provider_configuration_status,
)
from part2_rag.query_classification import (
    QueryClassificationError,
    format_classification_and_retrieval,
    route_and_retrieve,
)
from part2_rag.retrieval import (
    EmbeddingIndexBuildError,
    RetrievalConfig,
    RetrievalError,
    get_default_retrieval_config,
)


SUPPORTED_PROVIDERS: tuple[str, str] = ("groq", "gemini")


def build_answer_query_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify a query, retrieve grounded context, and optionally generate a cited answer."
    )
    parser.add_argument("--query", required=True, help="Query text to run through the RAG pipeline.")
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        help="LLM provider to use when generation is enabled.",
    )
    parser.add_argument(
        "--compare-providers",
        action="store_true",
        help="Run both configured providers against the same retrieved context and prompt.",
    )
    parser.add_argument("--model", default=None, help="Optional provider-specific model override.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional final-context override applied after query classification.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Print query classification and retrieval results without calling an LLM.",
    )
    parser.add_argument(
        "--save-raw-response",
        action="store_true",
        help="Persist raw provider response JSON under Part2/data/runs/.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the grounded prompt built from the retrieved context.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=get_default_runs_dir(),
        help="Directory used for saved raw response artifacts.",
    )
    return parser


def _build_base_retrieval_config(top_k: int | None) -> RetrievalConfig | None:
    if top_k is None:
        return None
    if top_k <= 0:
        raise ValueError("--top-k must be a positive integer.")

    defaults = get_default_retrieval_config()
    return RetrievalConfig(
        dense_top_k=max(defaults.dense_top_k, top_k),
        lexical_top_k=max(defaults.lexical_top_k, top_k),
        hybrid_final_top_k=top_k,
        rrf_constant=defaults.rrf_constant,
        faiss_index_path=defaults.faiss_index_path,
        embedding_store_path=defaults.embedding_store_path,
        chunk_artifact_path=defaults.chunk_artifact_path,
        corpus_manifest_path=defaults.corpus_manifest_path,
        chunk_manifest_path=defaults.chunk_manifest_path,
        embedding_manifest_path=defaults.embedding_manifest_path,
        part1_db_path=defaults.part1_db_path,
    )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not str(args.query).strip():
        parser.exit(status=1, message="Answer query failed: Query text must not be empty.\n")
    if args.compare_providers and args.model:
        parser.exit(
            status=1,
            message="Answer query failed: --model cannot be used together with --compare-providers.\n",
        )
    if not args.retrieval_only and not args.compare_providers and not args.provider:
        parser.exit(
            status=1,
            message="Answer query failed: Choose --provider or enable --compare-providers.\n",
        )


def _build_prompt_text(query: str, routed) -> str:
    return build_prompt(
        query=query,
        normalized_query=routed.classification.normalized_query,
        query_type=routed.classification.query_type,
        retrieval_results=routed.retrieval_results,
    )


def _format_prompt_debug(prompt: str) -> str:
    return "\n".join(["Prompt debug", "-------------", prompt])


def _check_provider_configuration(provider_names: Sequence[str]) -> tuple[list[str], list[str]]:
    configured: list[str] = []
    failures: list[str] = []
    for provider_name in provider_names:
        ok, message = get_provider_configuration_status(provider_name)
        if ok:
            configured.append(provider_name)
        else:
            failures.append(f"{provider_name}: {message}")
    return configured, failures


def _run_provider_comparison(
    *,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    routed,
) -> int:
    configured, failures = _check_provider_configuration(SUPPORTED_PROVIDERS)
    if failures:
        parser.exit(
            status=1,
            message=(
                "Answer query failed: Provider comparison requires both providers to be configured.\n"
                + "\n".join(f"  - {failure}" for failure in failures)
                + "\n"
            ),
        )

    if not configured:
        parser.exit(
            status=1,
            message="Answer query failed: No providers are configured for comparison.\n",
        )

    rendered_sections: list[str] = [
        "Provider comparison",
        "===================",
        format_classification_and_retrieval(routed),
    ]
    if args.show_prompt:
        rendered_sections.extend(["", _format_prompt_debug(_build_prompt_text(args.query, routed))])

    for provider_name in SUPPORTED_PROVIDERS:
        try:
            result = generate_grounded_answer(
                args.query,
                provider_name=provider_name,
                model=None,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                save_raw_response=args.save_raw_response,
                runs_dir=args.runs_dir,
                routing_result=routed,
            )
        except (
            AnswerGenerationError,
            ProviderConfigurationError,
            ProviderInvocationError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            parser.exit(
                status=1,
                message=f"Answer query failed: Provider comparison failed for {provider_name}: {exc}\n",
            )
        rendered_sections.extend(["", format_answer_generation_result(result)])

    print("\n".join(rendered_sections))
    return 0


def answer_query_main() -> int:
    parser = build_answer_query_arg_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    try:
        routed = route_and_retrieve(
            args.query,
            base_config=_build_base_retrieval_config(args.top_k),
        )
        if args.retrieval_only:
            print(format_classification_and_retrieval(routed))
            if args.show_prompt:
                print()
                print(_format_prompt_debug(_build_prompt_text(args.query, routed)))
            return 0

        if args.compare_providers:
            return _run_provider_comparison(parser=parser, args=args, routed=routed)

        result = generate_grounded_answer(
            args.query,
            provider_name=str(args.provider),
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            save_raw_response=args.save_raw_response,
            runs_dir=args.runs_dir,
            routing_result=routed,
        )
    except (
        AnswerGenerationError,
        EmbeddingIndexBuildError,
        ProviderConfigurationError,
        ProviderInvocationError,
        QueryClassificationError,
        RetrievalError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        parser.exit(status=1, message=f"Answer query failed: {exc}\n")

    if args.show_prompt:
        print(_format_prompt_debug(_build_prompt_text(args.query, routed)))
        print()
    print(format_answer_generation_result(result))
    return 0


if __name__ == "__main__":
    exit_code = answer_query_main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
