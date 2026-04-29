from __future__ import annotations

import argparse

from fitness_reddit_analyzer.config import PIPELINE_STAGES
from fitness_reddit_analyzer.pipeline import run_stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="r/fitness NLP Project Part 1 pipeline")
    parser.add_argument("stage", choices=PIPELINE_STAGES, help="Pipeline stage to execute")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_stage(args.stage)


if __name__ == "__main__":
    main()
