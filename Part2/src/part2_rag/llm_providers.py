from __future__ import annotations

import importlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from part2_rag.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_TEMPERATURE,
)


class ProviderConfigurationError(RuntimeError):
    """Raised when a provider is missing credentials, packages, or required config."""


class ProviderInvocationError(RuntimeError):
    """Raised when a provider call fails."""


_DOTENV_LOADED = False


def _iter_dotenv_paths() -> tuple[Path, ...]:
    part2_root = Path(__file__).resolve().parents[2]
    repo_root = part2_root.parent
    return (
        Path.cwd() / ".env",
        repo_root / ".env",
        repo_root / "Part1" / ".env",
        part2_root / ".env",
    )


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_dotenv_files() -> None:
    """Load local .env files without overriding already-exported variables."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    seen_paths: set[Path] = set()
    for dotenv_path in _iter_dotenv_paths():
        resolved_path = dotenv_path.resolve()
        if resolved_path in seen_paths or not resolved_path.exists():
            continue
        seen_paths.add(resolved_path)
        for line in resolved_path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_dotenv_line(line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class ProviderResponse:
    provider: str
    model: str
    text: str
    raw_response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ProviderGenerationMode = Literal["json", "text"]


class BaseLLMProvider:
    provider_name: str = "base"

    def default_model(self) -> str:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        generation_mode: ProviderGenerationMode = "json",
    ) -> ProviderResponse:
        raise NotImplementedError


def _validate_generation_mode(generation_mode: str) -> ProviderGenerationMode:
    if generation_mode not in {"json", "text"}:
        raise ProviderConfigurationError(
            f"Unsupported generation_mode={generation_mode!r}. Supported modes: json, text"
        )
    return generation_mode  # type: ignore[return-value]


def _read_env(
    env_names: tuple[str, ...],
    *,
    required: bool,
    description: str,
) -> str | None:
    load_dotenv_files()
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    if required:
        joined = ", ".join(env_names)
        raise ProviderConfigurationError(
            f"Missing {description}. Set one of: {joined}"
        )
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "to_dict"):
        dumped = value.to_dict()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {"repr": repr(value)}


class GroqProvider(BaseLLMProvider):
    provider_name = "groq"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        groq_module: Any | None = None,
    ) -> None:
        self.api_key = api_key or _read_env(
            ("GROQ_API_KEY",),
            required=True,
            description="Groq API key",
        )
        self._groq_module = groq_module

    def default_model(self) -> str:
        return _read_env(
            ("GROQ_MODEL",),
            required=False,
            description="Groq model",
        ) or DEFAULT_GROQ_MODEL

    def _load_module(self) -> Any:
        if self._groq_module is not None:
            return self._groq_module
        try:
            return importlib.import_module("groq")
        except ModuleNotFoundError as exc:
            raise ProviderConfigurationError(
                "Groq provider requires the `groq` package. Install project dependencies first."
            ) from exc

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        generation_mode: ProviderGenerationMode = "json",
    ) -> ProviderResponse:
        _validate_generation_mode(generation_mode)
        groq_module = self._load_module()
        chosen_model = model or self.default_model()
        try:
            client = groq_module.Groq(api_key=self.api_key)
            response = client.chat.completions.create(
                model=chosen_model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            text = response.choices[0].message.content or ""
        except Exception as exc:
            raise ProviderInvocationError(f"Groq generation failed: {exc}") from exc
        return ProviderResponse(
            provider=self.provider_name,
            model=chosen_model,
            text=str(text).strip(),
            raw_response=_as_dict(response),
        )


class GeminiProvider(BaseLLMProvider):
    provider_name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        google_genai_module: Any | None = None,
        google_generativeai_module: Any | None = None,
    ) -> None:
        self.api_key = api_key or _read_env(
            ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
            required=True,
            description="Google AI Studio / Gemini API key",
        )
        self._google_genai_module = google_genai_module
        self._google_generativeai_module = google_generativeai_module

    def default_model(self) -> str:
        return _read_env(
            ("GEMINI_MODEL",),
            required=False,
            description="Gemini model",
        ) or DEFAULT_GEMINI_MODEL

    def _load_modules(self) -> tuple[str, Any]:
        if self._google_genai_module is not None:
            return ("google.genai", self._google_genai_module)
        if self._google_generativeai_module is not None:
            return ("google.generativeai", self._google_generativeai_module)
        try:
            return ("google.genai", importlib.import_module("google.genai"))
        except ModuleNotFoundError:
            try:
                return (
                    "google.generativeai",
                    importlib.import_module("google.generativeai"),
                )
            except ModuleNotFoundError as exc:
                raise ProviderConfigurationError(
                    "Gemini provider requires `google-genai` or `google-generativeai`."
                ) from exc

    def _build_google_genai_config(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        generation_mode: ProviderGenerationMode = "json",
    ) -> dict[str, Any]:
        generation_mode = _validate_generation_mode(generation_mode)
        config: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if generation_mode == "json":
            config["response_mime_type"] = "application/json"
        if model.startswith("gemini-2.5"):
            config["thinking_config"] = {"thinking_budget": 0}
        return config

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        generation_mode: ProviderGenerationMode = "json",
    ) -> ProviderResponse:
        generation_mode = _validate_generation_mode(generation_mode)
        module_name, google_module = self._load_modules()
        chosen_model = model or self.default_model()
        try:
            if module_name == "google.genai":
                client = google_module.Client(api_key=self.api_key)
                response = client.models.generate_content(
                    model=chosen_model,
                    contents=prompt,
                    config=self._build_google_genai_config(
                        model=chosen_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        generation_mode=generation_mode,
                    ),
                )
                text = getattr(response, "text", None) or ""
            else:
                google_module.configure(api_key=self.api_key)
                generation_config_kwargs: dict[str, Any] = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                if generation_mode == "json":
                    generation_config_kwargs["response_mime_type"] = "application/json"
                generation_config = google_module.types.GenerationConfig(**generation_config_kwargs)
                model_client = google_module.GenerativeModel(model_name=chosen_model)
                response = model_client.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                text = getattr(response, "text", None) or ""
        except Exception as exc:
            raise ProviderInvocationError(f"Gemini generation failed: {exc}") from exc
        return ProviderResponse(
            provider=self.provider_name,
            model=chosen_model,
            text=str(text).strip(),
            raw_response=_as_dict(response),
        )


def get_provider(provider_name: str) -> BaseLLMProvider:
    normalized = provider_name.strip().lower()
    if normalized == "groq":
        return GroqProvider()
    if normalized in {"gemini", "google", "google-ai-studio"}:
        return GeminiProvider()
    raise ProviderConfigurationError(
        f"Unsupported provider={provider_name!r}. Supported providers: gemini, groq"
    )


def get_default_generation_settings() -> dict[str, float | int]:
    return {
        "temperature": DEFAULT_LLM_TEMPERATURE,
        "max_tokens": DEFAULT_LLM_MAX_TOKENS,
    }


def get_provider_configuration_status(provider_name: str) -> tuple[bool, str]:
    try:
        provider = get_provider(provider_name)
        model = provider.default_model()
    except ProviderConfigurationError as exc:
        return False, str(exc)
    return True, f"configured (default_model={model})"


__all__ = [
    "BaseLLMProvider",
    "GeminiProvider",
    "GroqProvider",
    "ProviderConfigurationError",
    "ProviderGenerationMode",
    "ProviderInvocationError",
    "ProviderResponse",
    "get_default_generation_settings",
    "get_provider_configuration_status",
    "get_provider",
]
