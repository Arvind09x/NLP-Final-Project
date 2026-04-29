from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PART2_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PART2_ROOT / "src"
SCRIPTS_ROOT = PART2_ROOT / "scripts"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from answer_query import answer_query_main
from part2_rag.answer_generation import AnswerGenerationResult, Citation, RetrievedSnippet
from part2_rag.llm_providers import ProviderConfigurationError
from part2_rag.query_classification import QueryClassificationResult, QueryRoutingResult
from part2_rag.retrieval import RetrievalResult


def make_cli_routing_result() -> QueryRoutingResult:
    return QueryRoutingResult(
        classification=QueryClassificationResult(
            query="what is a good ppl routine for beginners?",
            normalized_query="what is a good push pull legs routine for beginners?",
            query_type="factual",
            confidence=0.74,
            matched_rules=("abbrev:ppl->push pull legs",),
        ),
        retrieval_mode_used="hybrid",
        effective_retrieval_config={
            "query_type": "factual",
            "retrieval_mode": "hybrid",
            "dense_top_k": 8,
            "lexical_top_k": 8,
            "hybrid_final_top_k": 5,
            "rrf_constant": 60,
            "min_dense_score": None,
            "min_rrf_score": None,
            "guardrail_notes": ["balanced_hybrid_defaults"],
        },
        retrieval_results=(
            RetrievalResult(
                rank=1,
                chunk_id="chunk-1",
                document_id="doc-1",
                source_type="comment",
                chunk_index=0,
                chunk_origin="text",
                title="Beginner routine",
                created_utc=1680307200,
                score=0.91,
                retrieval_source="hybrid",
                snippet="Start with a simple push pull legs split.",
                dense_rank=1,
                dense_score=0.91,
                lexical_rank=1,
                lexical_score=0.12,
                rrf_score=0.03,
            ),
        ),
    )


class AnswerQueryCliTests(unittest.TestCase):
    def test_retrieval_only_mode_prints_classification_and_retrieval(self) -> None:
        output = io.StringIO()
        with (
            patch("answer_query.route_and_retrieve", return_value=make_cli_routing_result()),
            patch("sys.argv", [
                "answer_query.py",
                "--query",
                "what is a good ppl routine for beginners?",
                "--provider",
                "groq",
                "--retrieval-only",
            ]),
            patch("sys.stdout", output),
        ):
            exit_code = answer_query_main()

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Retrieval-only QA debug", rendered)
        self.assertIn("query_type: factual", rendered)
        self.assertIn("effective_retrieval_settings:", rendered)
        self.assertIn("chunk_id=chunk-1", rendered)
        self.assertIn("snippet: Start with a simple push pull legs split.", rendered)
        self.assertNotIn("Grounded QA result", rendered)

    def test_show_prompt_prints_prompt_in_retrieval_only_mode(self) -> None:
        output = io.StringIO()
        with (
            patch("answer_query.route_and_retrieve", return_value=make_cli_routing_result()),
            patch("sys.argv", [
                "answer_query.py",
                "--query",
                "what do people think about cutting while lifting?",
                "--provider",
                "groq",
                "--retrieval-only",
                "--show-prompt",
            ]),
            patch("sys.stdout", output),
        ):
            exit_code = answer_query_main()

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Prompt debug", rendered)
        self.assertIn("Use only the retrieved context below", rendered)
        self.assertIn("source_label: S1", rendered)
        self.assertIn("Do not cite chunk IDs or document IDs directly.", rendered)

    def test_empty_query_fails_cleanly(self) -> None:
        stderr = io.StringIO()
        with (
            patch("sys.argv", [
                "answer_query.py",
                "--query",
                "   ",
                "--provider",
                "groq",
            ]),
            patch("sys.stderr", stderr),
        ):
            with self.assertRaises(SystemExit) as exc:
                answer_query_main()

        self.assertEqual(exc.exception.code, 1)
        self.assertIn("Query text must not be empty", stderr.getvalue())

    def test_missing_provider_credentials_message_is_clear(self) -> None:
        stderr = io.StringIO()
        with (
            patch("answer_query.route_and_retrieve", return_value=make_cli_routing_result()),
            patch(
                "answer_query.generate_grounded_answer",
                side_effect=ProviderConfigurationError(
                    "Missing Groq API key. Set one of: GROQ_API_KEY"
                ),
            ),
            patch("sys.argv", [
                "answer_query.py",
                "--query",
                "what is a good ppl routine for beginners?",
                "--provider",
                "groq",
            ]),
            patch("sys.stderr", stderr),
        ):
            with self.assertRaises(SystemExit) as exc:
                answer_query_main()

        self.assertEqual(exc.exception.code, 1)
        self.assertIn("Missing Groq API key", stderr.getvalue())

    def test_compare_providers_missing_credentials_fails_cleanly(self) -> None:
        stderr = io.StringIO()
        with (
            patch("answer_query.route_and_retrieve", return_value=make_cli_routing_result()),
            patch(
                "answer_query.get_provider_configuration_status",
                side_effect=[
                    (False, "Missing Groq API key. Set one of: GROQ_API_KEY"),
                    (False, "Missing Google AI Studio / Gemini API key. Set one of: GOOGLE_API_KEY, GEMINI_API_KEY"),
                ],
            ),
            patch("sys.argv", [
                "answer_query.py",
                "--query",
                "what is a good ppl routine for beginners?",
                "--compare-providers",
            ]),
            patch("sys.stderr", stderr),
        ):
            with self.assertRaises(SystemExit) as exc:
                answer_query_main()

        self.assertEqual(exc.exception.code, 1)
        rendered = stderr.getvalue()
        self.assertIn("Provider comparison requires both providers to be configured", rendered)
        self.assertIn("groq: Missing Groq API key", rendered)
        self.assertIn("gemini: Missing Google AI Studio / Gemini API key", rendered)

    def test_compare_providers_one_missing_credential_still_fails_cleanly(self) -> None:
        stderr = io.StringIO()
        with (
            patch("answer_query.route_and_retrieve", return_value=make_cli_routing_result()),
            patch(
                "answer_query.get_provider_configuration_status",
                side_effect=[
                    (True, "configured"),
                    (False, "Missing Google AI Studio / Gemini API key. Set one of: GOOGLE_API_KEY, GEMINI_API_KEY"),
                ],
            ),
            patch("sys.argv", [
                "answer_query.py",
                "--query",
                "what is a good ppl routine for beginners?",
                "--compare-providers",
            ]),
            patch("sys.stderr", stderr),
        ):
            with self.assertRaises(SystemExit) as exc:
                answer_query_main()

        self.assertEqual(exc.exception.code, 1)
        rendered = stderr.getvalue()
        self.assertIn("Provider comparison requires both providers to be configured", rendered)
        self.assertIn("gemini: Missing Google AI Studio / Gemini API key", rendered)

    def test_saved_raw_response_path_is_displayed(self) -> None:
        output = io.StringIO()
        result = AnswerGenerationResult(
            query="what is a good ppl routine for beginners?",
            normalized_query="what is a good push pull legs routine for beginners?",
            query_type="factual",
            answer_text="Start simple.",
            citations=(
                Citation(
                    source_label="S1",
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    title="Beginner routine",
                    source_type="comment",
                    created_utc=1680307200,
                    snippet="Start with a simple push pull legs split.",
                ),
            ),
            retrieved_snippets=(
                RetrievedSnippet(
                    source_label="S1",
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    title="Beginner routine",
                    source_type="comment",
                    created_utc=1680307200,
                    snippet="Start with a simple push pull legs split.",
                    retrieval_source="hybrid",
                    score=0.91,
                ),
            ),
            insufficient_evidence=True,
            provider="groq",
            model="llama-3.1-8b-instant",
            raw_response_path=str(PART2_ROOT / "data" / "runs" / "example.json"),
        )
        with (
            patch("answer_query.route_and_retrieve", return_value=make_cli_routing_result()),
            patch("answer_query.generate_grounded_answer", return_value=result),
            patch("sys.argv", [
                "answer_query.py",
                "--query",
                "what is a good ppl routine for beginners?",
                "--provider",
                "groq",
                "--save-raw-response",
            ]),
            patch("sys.stdout", output),
        ):
            exit_code = answer_query_main()

        self.assertEqual(exit_code, 0)
        self.assertIn("raw_response_path:", output.getvalue())

    def test_cli_output_includes_source_labels(self) -> None:
        output = io.StringIO()
        result = AnswerGenerationResult(
            query="what is a good ppl routine for beginners?",
            normalized_query="what is a good push pull legs routine for beginners?",
            query_type="factual",
            answer_text="Start simple.",
            citations=(
                Citation(
                    source_label="S1",
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    title="Beginner routine",
                    source_type="comment",
                    created_utc=1680307200,
                    snippet="Start with a simple push pull legs split.",
                ),
            ),
            retrieved_snippets=(
                RetrievedSnippet(
                    source_label="S1",
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    title="Beginner routine",
                    source_type="comment",
                    created_utc=1680307200,
                    snippet="Start with a simple push pull legs split.",
                    retrieval_source="hybrid",
                    score=0.91,
                ),
            ),
            insufficient_evidence=False,
            provider="groq",
            model="llama-3.1-8b-instant",
            raw_response_path=None,
        )
        with (
            patch("answer_query.route_and_retrieve", return_value=make_cli_routing_result()),
            patch("answer_query.generate_grounded_answer", return_value=result),
            patch("sys.argv", [
                "answer_query.py",
                "--query",
                "what is a good ppl routine for beginners?",
                "--provider",
                "groq",
            ]),
            patch("sys.stdout", output),
        ):
            exit_code = answer_query_main()

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("source_label=S1", rendered)
        self.assertIn("[S1] chunk_id=chunk-1", rendered)


if __name__ == "__main__":
    unittest.main()
