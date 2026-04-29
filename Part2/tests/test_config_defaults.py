from __future__ import annotations

import sys
import unittest
from pathlib import Path


PART2_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PART2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part2_rag.config import DEFAULT_GEMINI_MODEL, DEFAULT_GROQ_MODEL


class ConfigDefaultsTests(unittest.TestCase):
    def test_default_provider_models_match_current_project_defaults(self) -> None:
        self.assertEqual(DEFAULT_GROQ_MODEL, "llama-3.3-70b-versatile")
        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-2.5-flash")


if __name__ == "__main__":
    unittest.main()
