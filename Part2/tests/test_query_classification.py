from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PART2_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PART2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part2_rag.config import (
    ADVERSARIAL_NO_ANSWER_QUERY_TYPE,
    FACTUAL_QUERY_TYPE,
    OPINION_SUMMARY_QUERY_TYPE,
)
from part2_rag.query_classification import (
    QueryClassificationResult,
    build_effective_retrieval_config,
    classify_query,
    expand_query_abbreviations,
    expand_query_domain_concepts,
    route_and_retrieve,
)
from part2_rag.retrieval import RetrievalResult


class QueryClassificationTests(unittest.TestCase):
    def test_factual_classification(self) -> None:
        result = classify_query("What is the best way to estimate 1RM for squats?")
        self.assertEqual(result.query_type, FACTUAL_QUERY_TYPE)
        self.assertIn("abbrev:1rm->one rep max", result.matched_rules)
        self.assertIn("one rep max", result.normalized_query)
        self.assertGreaterEqual(result.confidence, 0.6)

    def test_opinion_summary_classification(self) -> None:
        result = classify_query("What do people think about cutting while lifting?")
        self.assertEqual(result.query_type, OPINION_SUMMARY_QUERY_TYPE)
        self.assertTrue(
            any(rule.startswith("opinion:") for rule in result.matched_rules),
            msg=f"expected opinion rule in {result.matched_rules!r}",
        )

    def test_adversarial_no_answer_classification(self) -> None:
        result = classify_query("What does this subreddit say about quantum mechanics?")
        self.assertEqual(result.query_type, ADVERSARIAL_NO_ANSWER_QUERY_TYPE)
        self.assertTrue(
            any(rule.startswith("adversarial:") for rule in result.matched_rules),
            msg=f"expected adversarial rule in {result.matched_rules!r}",
        )

    def test_abbreviation_expansion(self) -> None:
        expanded, matched_rules = expand_query_abbreviations(
            "Need a ppl plan with 5x5 work and RPE targets at bw"
        )
        self.assertEqual(
            expanded,
            "need a push pull legs plan with five by five strength program work and rate of perceived exertion targets at bodyweight",
        )
        self.assertEqual(
            matched_rules,
            (
                "abbrev:ppl->push pull legs",
                "abbrev:5x5->five by five strength program",
                "abbrev:bw->bodyweight",
                "abbrev:rpe->rate of perceived exertion",
            ),
        )

    def test_unknown_query_falls_back_to_factual(self) -> None:
        result = classify_query("hello there")
        self.assertEqual(result.query_type, FACTUAL_QUERY_TYPE)
        self.assertEqual(result.matched_rules, ("fallback:default_factual",))
        self.assertAlmostEqual(result.confidence, 0.55)

    def test_domain_concept_expansion_adds_answer_shaped_terms(self) -> None:
        expanded, matched_rules = expand_query_domain_concepts(
            "what do people think about body recomposition?"
        )
        self.assertIn("recomp", expanded)
        self.assertIn("recomping", expanded)
        self.assertEqual(matched_rules, ("concept:body_recomposition",))

    def test_opinion_query_adds_body_recomposition_opinion_tail(self) -> None:
        result = classify_query("What do people think about body recomposition?")
        self.assertIn("bulk cut", result.normalized_query)
        self.assertIn("beginners", result.normalized_query)

    def test_class_specific_retrieval_config_selection(self) -> None:
        retrieval_mode, profile, effective = build_effective_retrieval_config(
            OPINION_SUMMARY_QUERY_TYPE
        )
        self.assertEqual(retrieval_mode, "hybrid")
        self.assertEqual(profile.query_type, OPINION_SUMMARY_QUERY_TYPE)
        self.assertEqual(effective.dense_top_k, 14)
        self.assertEqual(effective.lexical_top_k, 14)
        self.assertEqual(effective.hybrid_final_top_k, 7)

    def test_routing_calls_retrieval_with_normalized_query(self) -> None:
        fake_results = [
            RetrievalResult(
                rank=1,
                chunk_id="chunk-1",
                document_id="doc-1",
                source_type="comment",
                chunk_index=0,
                chunk_origin="text",
                title="Example",
                created_utc=1680307200,
                score=0.9,
                retrieval_source="hybrid",
                snippet="example snippet",
                dense_rank=1,
                dense_score=0.9,
                lexical_rank=2,
                lexical_score=0.1,
                rrf_score=0.03,
            )
        ]

        with patch("part2_rag.query_classification.retrieve", return_value=fake_results) as mocked:
            routed = route_and_retrieve("What is a good PPL routine for beginners?")

        mocked.assert_called_once()
        called_query = mocked.call_args.args[0]
        called_mode = mocked.call_args.kwargs["mode"]
        called_config = mocked.call_args.kwargs["config"]
        self.assertEqual(
            called_query,
            "what is a good push pull legs routine for beginners?",
        )
        self.assertEqual(called_mode, "hybrid")
        self.assertEqual(called_config.dense_top_k, 8)
        self.assertEqual(routed.classification.normalized_query, called_query)
        self.assertEqual(len(routed.retrieval_results), 1)

    def test_classification_result_output_shape(self) -> None:
        result = classify_query("What is TDEE?")
        self.assertIsInstance(result, QueryClassificationResult)
        payload = result.to_dict()
        self.assertEqual(
            set(payload),
            {"query", "query_type", "confidence", "matched_rules", "normalized_query"},
        )


if __name__ == "__main__":
    unittest.main()
