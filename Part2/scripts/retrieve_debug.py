from __future__ import annotations

import os
import sys
from pathlib import Path


PART2_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PART2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part2_rag.retrieval import retrieval_debug_main


if __name__ == "__main__":
    exit_code = retrieval_debug_main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
