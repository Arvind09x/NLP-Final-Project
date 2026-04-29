from __future__ import annotations

import argparse
import sys
from pathlib import Path


PART2_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PART2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part2_rag.config import get_default_eval_path, get_default_eval_runs_dir
from part2_rag.eval_runner import EvalRunnerError, run_eval


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen Part 2 RAG eval set and persist structured artifacts."
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=get_default_eval_path(),
        help="Path to the frozen eval JSONL file.",
    )
    parser.add_argument(
        "--provider",
        choices=("groq", "gemini", "both"),
        default="groq",
        help="Provider selection to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_default_eval_runs_dir(),
        help="Root directory where timestamped eval runs are written.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run classification and retrieval only without calling providers.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        default=None,
        help="Optional question_id filter. Pass multiple times to include multiple IDs.",
    )
    parser.add_argument(
        "--save-raw-response",
        action="store_true",
        help="Persist raw provider responses under the run directory.",
    )
    parser.add_argument(
        "--skip-bert-score",
        action="store_true",
        help="Do not compute BERTScore even if the optional bert-score package is installed.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        result = run_eval(
            eval_path=args.eval_path.resolve(),
            provider_selection=args.provider,
            output_root_dir=args.output_dir.resolve(),
            retrieval_only=bool(args.retrieval_only),
            max_examples=args.max_examples,
            question_ids=args.question_id,
            save_raw_response=bool(args.save_raw_response),
            bert_score_scorer=False if args.skip_bert_score else True,
        )
    except EvalRunnerError as exc:
        parser.exit(status=1, message=f"RAG eval failed: {exc}\n")

    print("RAG eval complete")
    print(f"run_id: {result.run_id}")
    print(f"run_dir: {result.run_dir}")
    print(f"rows: {result.row_count}")
    print(f"results_jsonl: {result.results_jsonl_path}")
    print(f"results_csv: {result.results_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
