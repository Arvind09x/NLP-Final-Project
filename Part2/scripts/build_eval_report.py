from __future__ import annotations

import argparse
import sys
from pathlib import Path


PART2_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PART2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part2_rag.config import get_default_eval_path
from part2_rag.eval_reporting import (
    EvalReportingError,
    build_manual_review_rows,
    load_manual_review_csv,
    load_eval_examples_for_reporting,
    load_eval_run,
    summarize_manual_review,
    summarize_eval_run,
    write_eval_summary_json,
    write_eval_summary_markdown,
    write_manual_review_csv,
    write_manual_review_summary_json,
    write_manual_review_summary_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a manual review sheet and summary report for a completed eval run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to a completed eval run directory containing results.jsonl and artifacts/.",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=get_default_eval_path(),
        help="Path to the frozen eval set JSONL used to join question text and gold answers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to the run directory.",
    )
    parser.add_argument(
        "--manual-review-input",
        type=Path,
        default=None,
        help=(
            "Optional completed manual review CSV to summarize. "
            "When provided, writes manual_review_summary.json and manual_review_summary.md."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir).resolve()

    try:
        run = load_eval_run(run_dir)
        eval_examples = load_eval_examples_for_reporting(args.eval_path)
        review_rows = build_manual_review_rows(run=run, eval_examples=eval_examples)
        summary = summarize_eval_run(run=run, review_rows=review_rows)
    except EvalReportingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    manual_review_path = output_dir / "manual_review.csv"
    summary_json_path = output_dir / "eval_summary.json"
    summary_markdown_path = output_dir / "eval_summary.md"

    write_manual_review_csv(manual_review_path, review_rows)
    write_eval_summary_json(summary_json_path, summary)
    write_eval_summary_markdown(summary_markdown_path, summary)

    print(f"manual_review_csv={manual_review_path}")
    print(f"eval_summary_json={summary_json_path}")
    print(f"eval_summary_md={summary_markdown_path}")

    if args.manual_review_input is not None:
        try:
            completed_review_rows = load_manual_review_csv(args.manual_review_input.resolve())
            manual_review_summary = summarize_manual_review(completed_review_rows)
        except EvalReportingError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        manual_review_summary_json_path = output_dir / "manual_review_summary.json"
        manual_review_summary_markdown_path = output_dir / "manual_review_summary.md"
        write_manual_review_summary_json(
            manual_review_summary_json_path,
            manual_review_summary,
        )
        write_manual_review_summary_markdown(
            manual_review_summary_markdown_path,
            manual_review_summary,
        )
        print(f"manual_review_summary_json={manual_review_summary_json_path}")
        print(f"manual_review_summary_md={manual_review_summary_markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
