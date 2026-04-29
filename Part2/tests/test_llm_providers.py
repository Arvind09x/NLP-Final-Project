from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PART2_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PART2_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from part2_rag.llm_providers import (
    GeminiProvider,
    GroqProvider,
    ProviderConfigurationError,
    _parse_dotenv_line,
)


class ProviderConfigurationTests(unittest.TestCase):
    def test_groq_provider_fails_without_credentials(self) -> None:
        with patch("part2_rag.llm_providers.load_dotenv_files"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ProviderConfigurationError):
                GroqProvider()

    def test_gemini_provider_fails_without_credentials(self) -> None:
        with patch("part2_rag.llm_providers.load_dotenv_files"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ProviderConfigurationError):
                GeminiProvider()

    def test_dotenv_line_parser_supports_export_and_quotes(self) -> None:
        self.assertEqual(
            _parse_dotenv_line('export GROQ_API_KEY="abc123"'),
            ("GROQ_API_KEY", "abc123"),
        )
        self.assertEqual(
            _parse_dotenv_line("GEMINI_API_KEY='xyz789'"),
            ("GEMINI_API_KEY", "xyz789"),
        )
        self.assertIsNone(_parse_dotenv_line("# comment"))

    def test_google_genai_config_requests_json_and_disables_25_thinking(self) -> None:
        provider = GeminiProvider(api_key="test-key")

        config = provider._build_google_genai_config(
            model="gemini-2.5-flash",
            temperature=0.1,
            max_tokens=700,
        )

        self.assertEqual(config["temperature"], 0.1)
        self.assertEqual(config["max_output_tokens"], 700)
        self.assertEqual(config["response_mime_type"], "application/json")
        self.assertEqual(config["thinking_config"], {"thinking_budget": 0})

    def test_google_genai_config_leaves_non_25_thinking_default(self) -> None:
        provider = GeminiProvider(api_key="test-key")

        config = provider._build_google_genai_config(
            model="gemini-1.5-flash",
            temperature=0.1,
            max_tokens=700,
        )

        self.assertNotIn("thinking_config", config)

    def test_google_genai_text_config_omits_json_mime_type(self) -> None:
        provider = GeminiProvider(api_key="test-key")

        config = provider._build_google_genai_config(
            model="gemini-2.5-flash",
            temperature=0.1,
            max_tokens=256,
            generation_mode="text",
        )

        self.assertEqual(config["temperature"], 0.1)
        self.assertEqual(config["max_output_tokens"], 256)
        self.assertNotIn("response_mime_type", config)
        self.assertEqual(config["thinking_config"], {"thinking_budget": 0})


if __name__ == "__main__":
    unittest.main()
