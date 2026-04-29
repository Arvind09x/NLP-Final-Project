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

from part2_rag.eval_validation import EvalValidationError, validate_eval_set
from part2_rag.retrieval import load_frozen_chunk_catalog


def write_chunk_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_eval_file(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_chunk_row(document_id: str, chunk_id: str) -> dict[str, object]:
    source_id = document_id.replace("comment_", "")
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source_type": "comment",
        "source_id": source_id,
        "post_id": "post-1",
        "parent_id": "t3_post-1",
        "link_id": "t3_post-1",
        "created_utc": 1680307200,
        "author_id": f"author_{source_id}",
        "title": "Synthetic thread",
        "chunk_text": f"support text for {document_id}",
        "chunk_index": 0,
        "token_estimate": 12,
        "chunk_origin": "text",
    }


def make_eval_row(
    *,
    question_id: str,
    question_type: str,
    expected_has_answer: bool,
    supporting_document_ids: list[str],
    supporting_chunk_ids: list[str],
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question": f"Question for {question_id}?",
        "question_type": question_type,
        "gold_answer": "Grounded answer." if expected_has_answer else "Insufficient evidence.",
        "expected_has_answer": expected_has_answer,
        "supporting_document_ids": supporting_document_ids,
        "supporting_chunk_ids": supporting_chunk_ids,
        "notes": "Synthetic eval example.",
    }


class EvalValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        load_frozen_chunk_catalog.cache_clear()

    def _build_valid_fixture(self, temp_path: Path) -> tuple[Path, Path]:
        chunk_rows: list[dict[str, object]] = []
        eval_rows: list[dict[str, object]] = []

        for index in range(1, 14):
            document_id = f"comment_doc_{index:02d}"
            chunk_id = f"{document_id}_chunk_0000_test"
            chunk_rows.append(make_chunk_row(document_id, chunk_id))
            question_type = "factual" if index <= 8 else "opinion-summary"
            eval_rows.append(
                make_eval_row(
                    question_id=f"q_{index:02d}",
                    question_type=question_type,
                    expected_has_answer=True,
                    supporting_document_ids=[document_id],
                    supporting_chunk_ids=[chunk_id],
                )
            )

        eval_rows.append(
            make_eval_row(
                question_id="q_14",
                question_type="adversarial/no-answer",
                expected_has_answer=False,
                supporting_document_ids=[],
                supporting_chunk_ids=[],
            )
        )
        eval_rows.append(
            make_eval_row(
                question_id="q_15",
                question_type="adversarial/no-answer",
                expected_has_answer=False,
                supporting_document_ids=[],
                supporting_chunk_ids=[],
            )
        )

        chunk_artifact_path = temp_path / "chunks.jsonl"
        eval_path = temp_path / "eval.jsonl"
        write_chunk_artifact(chunk_artifact_path, chunk_rows)
        write_eval_file(eval_path, eval_rows)
        return eval_path, chunk_artifact_path

    def test_validate_eval_set_accepts_valid_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            eval_path, chunk_artifact_path = self._build_valid_fixture(temp_path)

            report = validate_eval_set(
                eval_path=eval_path,
                chunk_artifact_path=chunk_artifact_path,
            )

            self.assertEqual(report.example_count, 15)
            self.assertEqual(report.counts_by_question_type["factual"], 8)
            self.assertEqual(report.counts_by_question_type["opinion-summary"], 5)
            self.assertEqual(report.counts_by_question_type["adversarial/no-answer"], 2)

    def test_duplicate_question_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            eval_path, chunk_artifact_path = self._build_valid_fixture(temp_path)
            rows = [
                json.loads(line)
                for line in eval_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rows[1]["question_id"] = rows[0]["question_id"]
            write_eval_file(eval_path, rows)

            with self.assertRaises(EvalValidationError):
                validate_eval_set(
                    eval_path=eval_path,
                    chunk_artifact_path=chunk_artifact_path,
                )

    def test_adversarial_example_must_not_expect_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            eval_path, chunk_artifact_path = self._build_valid_fixture(temp_path)
            rows = [
                json.loads(line)
                for line in eval_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rows[-1]["expected_has_answer"] = True
            write_eval_file(eval_path, rows)

            with self.assertRaises(EvalValidationError):
                validate_eval_set(
                    eval_path=eval_path,
                    chunk_artifact_path=chunk_artifact_path,
                )

    def test_answer_bearing_example_requires_existing_support_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            eval_path, chunk_artifact_path = self._build_valid_fixture(temp_path)
            rows = [
                json.loads(line)
                for line in eval_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rows[0]["supporting_chunk_ids"] = ["missing_chunk"]
            write_eval_file(eval_path, rows)

            with self.assertRaises(EvalValidationError):
                validate_eval_set(
                    eval_path=eval_path,
                    chunk_artifact_path=chunk_artifact_path,
                )


if __name__ == "__main__":
    unittest.main()
