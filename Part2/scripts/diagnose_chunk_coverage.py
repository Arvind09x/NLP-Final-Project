from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PART2_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PART2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part2_rag.chunk_generation import diagnose_missing_documents
from part2_rag.config import get_default_manifest_path, get_paths


def build_arg_parser() -> argparse.ArgumentParser:
    paths = get_paths()
    parser = argparse.ArgumentParser(
        description="Diagnose selected-window documents that do not produce chunks."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=paths.part1_db_path,
        help="Path to the Part 1 SQLite database.",
    )
    parser.add_argument(
        "--source-corpus-manifest",
        type=Path,
        default=get_default_manifest_path(),
        help="Path to the frozen corpus manifest JSON.",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=220,
        help="Approximate token budget per chunk.",
    )
    parser.add_argument(
        "--post-overlap-sentences",
        type=int,
        default=1,
        help="Number of overlapping sentences between adjacent post chunks.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    diagnostics = diagnose_missing_documents(
        db_path=args.db_path.resolve(),
        source_corpus_manifest_path=args.source_corpus_manifest.resolve(),
        max_chunk_tokens=args.max_chunk_tokens,
        overlap_sentences=args.post_overlap_sentences,
    )
    print(json.dumps({"counts": diagnostics.counts, "sample_ids": diagnostics.sample_ids}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
