from __future__ import annotations

import argparse
import sys
from pathlib import Path


PART2_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PART2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part2_rag.config import (
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_TEMPERATURE,
    get_default_hindi_eval_path,
    get_default_indian_language_runs_dir,
)
from part2_rag.indian_language_eval import (
    IndianLanguageEvalError,
    load_hindi_eval_examples,
    run_hindi_eval,
    write_manual_review_template,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Part 2 Hindi translation eval and persist structured artifacts."
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=get_default_hindi_eval_path(),
        help="Path to the Hindi eval JSONL file.",
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
        default=get_default_indian_language_runs_dir(),
        help="Root directory where timestamped Hindi eval runs are written.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional provider max output token override for smoke tests.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional provider temperature override.",
    )
    parser.add_argument(
        "--example-id",
        action="append",
        default=None,
        help="Optional example_id filter. Pass multiple times to include multiple IDs.",
    )
    parser.add_argument(
        "--save-raw-response",
        action="store_true",
        help="Persist raw provider responses under the run directory.",
    )
    parser.add_argument(
        "--skip-bert-score",
        action="store_true",
        help="Do not compute multilingual BERTScore even if bert-score is installed.",
    )
    parser.add_argument(
        "--write-manual-review-template",
        action="store_true",
        help="Write or refresh the standalone Hindi manual review template and exit.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.write_manual_review_template:
        try:
            examples = load_hindi_eval_examples(args.eval_path.resolve())
            template_path = write_manual_review_template(examples=examples)
        except IndianLanguageEvalError as exc:
            parser.exit(status=1, message=f"Hindi eval setup failed: {exc}\n")
        print(f"manual_review_template: {template_path}")
        return 0

    try:
        result = run_hindi_eval(
            eval_path=args.eval_path.resolve(),
            provider_selection=args.provider,
            output_root_dir=args.output_dir.resolve(),
            max_examples=args.max_examples,
            example_ids=args.example_id,
            save_raw_response=bool(args.save_raw_response),
            temperature=(
                DEFAULT_LLM_TEMPERATURE if args.temperature is None else args.temperature
            ),
            max_tokens=DEFAULT_LLM_MAX_TOKENS if args.max_tokens is None else args.max_tokens,
            bert_score_scorer=False if args.skip_bert_score else True,
        )
    except IndianLanguageEvalError as exc:
        parser.exit(status=1, message=f"Hindi eval failed: {exc}\n")

    print("Hindi eval complete")
    print(f"run_id: {result.run_id}")
    print(f"run_dir: {result.run_dir}")
    print(f"rows: {result.row_count}")
    print(f"results_jsonl: {result.results_jsonl_path}")
    print(f"results_csv: {result.results_csv_path}")
    print(f"summary_json: {Path(result.run_dir) / 'summary.json'}")
    print(f"summary_md: {Path(result.run_dir) / 'summary.md'}")
    print(f"manual_review_csv: {Path(result.run_dir) / 'manual_review.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
