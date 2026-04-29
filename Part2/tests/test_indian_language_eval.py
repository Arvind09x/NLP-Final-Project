from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PART2_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PART2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part2_rag.eval_runner import BertScoreValues
from part2_rag.indian_language_eval import (
    IndianLanguageEvalError,
    build_hindi_translation_prompt,
    compute_chrf,
    load_hindi_eval_examples,
    run_hindi_eval,
)
from part2_rag.llm_providers import BaseLLMProvider, ProviderInvocationError, ProviderResponse


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_hindi_eval_fixture(temp_path: Path) -> Path:
    rows: list[dict[str, object]] = []
    for index in range(1, 21):
        rows.append(
            {
                "example_id": f"hin_eval_{index:03d}",
                "source_text": f"Example source text {index} with PPL and PR.",
                "reference_text": f"उदाहरण हिंदी आउटपुट {index} PPL और PR के साथ।",
                "task_type": "translation",
                "source_kind": "synthetic_test",
                "difficulty_tags": ["abbreviation", "code_mixed"] if index % 2 == 0 else ["question"],
                "notes": "Synthetic Hindi eval fixture row.",
            }
        )
    eval_path = temp_path / "hindi_eval.jsonl"
    write_jsonl(eval_path, rows)
    return eval_path


class FakeHindiProvider(BaseLLMProvider):
    provider_name = "fake"

    def __init__(self, provider_name: str, *, fail: bool = False) -> None:
        self.provider_name = provider_name
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def default_model(self) -> str:
        return f"{self.provider_name}-model"

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        generation_mode: str = "json",
    ) -> ProviderResponse:
        self.calls.append(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "generation_mode": generation_mode,
            }
        )
        if self.fail:
            raise ProviderInvocationError(f"{self.provider_name} exploded")
        prompt_tail = prompt.split("Source text:\n", 1)[-1].strip()
        translated = f"हिंदी अनुवाद: {prompt_tail}"
        return ProviderResponse(
            provider=self.provider_name,
            model=model,
            text=translated,
            raw_response={"prompt_tail": prompt_tail},
        )


class IndianLanguageEvalTests(unittest.TestCase):
    def test_default_hindi_eval_dataset_loads_with_minimum_size(self) -> None:
        examples = load_hindi_eval_examples()

        self.assertGreaterEqual(len(examples), 20)
        self.assertEqual(examples[0].task_type, "translation")
        self.assertTrue(examples[0].reference_text)

    def test_loader_rejects_dataset_smaller_than_twenty_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            small_path = temp_path / "small.jsonl"
            write_jsonl(
                small_path,
                [
                    {
                        "example_id": "one",
                        "source_text": "a",
                        "reference_text": "ब",
                        "task_type": "translation",
                        "source_kind": "synthetic",
                        "difficulty_tags": [],
                        "notes": "",
                    }
                ],
            )
            with self.assertRaisesRegex(IndianLanguageEvalError, "at least 20 examples"):
                load_hindi_eval_examples(small_path)

    def test_chrf_rewards_close_match(self) -> None:
        close_score = compute_chrf(
            "calorie deficit, walking से ज़्यादा महत्वपूर्ण है",
            "calorie deficit walking से ज़्यादा महत्वपूर्ण है",
        )
        far_score = compute_chrf(
            "calorie deficit, walking से ज़्यादा महत्वपूर्ण है",
            "StrongLifts 5x5 beginners के लिए है",
        )

        self.assertGreater(close_score, far_score)
        self.assertGreater(close_score, 0.5)
        self.assertLessEqual(close_score, 1.0)

    def test_translation_prompt_mentions_abbreviation_policy(self) -> None:
        prompt = build_hindi_translation_prompt("Use a TDEE calculator.")

        self.assertIn("TDEE", prompt)
        self.assertIn("Output only the Hindi translation", prompt)

    def test_hindi_eval_run_writes_results_summary_and_manual_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            eval_path = build_hindi_eval_fixture(temp_path)
            output_dir = temp_path / "runs"

            result = run_hindi_eval(
                eval_path=eval_path,
                provider_selection="both",
                output_root_dir=output_dir,
                max_examples=4,
                save_raw_response=True,
                provider_status_fn=lambda provider_name: (True, "configured"),
                provider_factory=lambda provider_name: FakeHindiProvider(
                    provider_name,
                    fail=(provider_name == "gemini"),
                ),
                bert_score_scorer=lambda reference, candidate: BertScoreValues(
                    precision=0.71,
                    recall=0.72,
                    f1=0.73,
                ),
            )

            run_dir = Path(result.run_dir)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            manual_review_text = (run_dir / "manual_review.csv").read_text(encoding="utf-8")

        self.assertEqual(result.row_count, 8)
        statuses = {(row.provider, row.status) for row in result.rows}
        self.assertIn(("groq", "success"), statuses)
        self.assertIn(("gemini", "provider_error"), statuses)
        groq_rows = [row for row in result.rows if row.provider == "groq"]
        self.assertTrue(all(row.bert_score_f1 == 0.73 for row in groq_rows))
        self.assertEqual(summary["provider_metrics"]["groq"]["success_count"], 4)
        self.assertTrue(manifest["bert_score"]["enabled"])
        self.assertIn("fluency", manual_review_text)
        self.assertIn("adequacy", manual_review_text)

    def test_hindi_eval_uses_plain_text_generation_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            eval_path = build_hindi_eval_fixture(temp_path)
            providers: dict[str, FakeHindiProvider] = {}

            def provider_factory(provider_name: str) -> FakeHindiProvider:
                provider = FakeHindiProvider(provider_name)
                providers[provider_name] = provider
                return provider

            result = run_hindi_eval(
                eval_path=eval_path,
                provider_selection="gemini",
                output_root_dir=temp_path / "runs",
                max_examples=1,
                temperature=0.2,
                max_tokens=256,
                provider_status_fn=lambda provider_name: (True, "configured"),
                provider_factory=provider_factory,
                bert_score_scorer=False,
            )

            manifest = json.loads(
                (Path(result.run_dir) / "run_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.row_count, 1)
        self.assertEqual(
            providers["gemini"].calls,
            [{"temperature": 0.2, "max_tokens": 256, "generation_mode": "text"}],
        )
        self.assertEqual(manifest["temperature"], 0.2)
        self.assertEqual(manifest["max_tokens"], 256)
        self.assertEqual(manifest["generation_mode"], "text")


if __name__ == "__main__":
    unittest.main()
