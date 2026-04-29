from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from fitness_reddit_analyzer.build_app_cache import build_snapshot
from fitness_reddit_analyzer.config import APP_CACHE_PATH, DB_PATH, PROCESSED_DIR


@dataclass(frozen=True)
class CacheOption:
    key: str
    label: str
    help_text: str
    path: Path
    is_default: bool = False


CACHE_OPTIONS: tuple[CacheOption, ...] = (
    CacheOption(
        key="era1_primary",
        label="Era 1: Primary window",
        help_text="Default cache covering Apr 2023 to Apr 2024.",
        path=APP_CACHE_PATH,
        is_default=True,
    ),
    CacheOption(
        key="era2_partial",
        label="Era 2: Partial window",
        help_text="Separate isolated cache covering the partial Jun 2018 to Aug 2018 window.",
        path=PROCESSED_DIR / "app_cache_era2_partial.json",
    ),
)


def load_dashboard_snapshot(cache_key: str | None = None, *, force_refresh: bool = False) -> dict[str, Any]:
    if force_refresh:
        _load_dashboard_snapshot.cache_clear()
    return _load_dashboard_snapshot(resolve_cache_key(cache_key))


def list_available_caches() -> list[CacheOption]:
    return [option for option in CACHE_OPTIONS if option.path.exists()]


def resolve_cache_key(cache_key: str | None = None) -> str:
    available = list_available_caches()
    if cache_key and any(option.key == cache_key for option in available):
        return cache_key
    default_option = next((option for option in available if option.is_default), None)
    if default_option is not None:
        return default_option.key
    if available:
        return available[0].key
    return "live_sqlite_fallback"


def get_cache_option(cache_key: str) -> CacheOption | None:
    return next((option for option in CACHE_OPTIONS if option.key == cache_key), None)


@lru_cache(maxsize=None)
def _load_dashboard_snapshot(cache_key: str) -> dict[str, Any]:
    cache_option = get_cache_option(cache_key)
    if cache_option is not None and cache_option.path.exists():
        payload = json.loads(cache_option.path.read_text(encoding="utf-8"))
        payload.setdefault("meta", {})
        payload["meta"]["cache_source"] = "app_cache"
        payload["meta"]["cache_key"] = cache_option.key
        payload["meta"]["cache_label"] = cache_option.label
        payload["meta"]["cache_help_text"] = cache_option.help_text
        payload["meta"]["cache_path"] = str(cache_option.path)
        return payload
    if DB_PATH.exists():
        payload = build_snapshot()
        payload.setdefault("meta", {})
        payload["meta"]["cache_source"] = "live_sqlite_fallback"
        payload["meta"]["cache_key"] = "live_sqlite_fallback"
        payload["meta"]["cache_label"] = "Live SQLite fallback"
        payload["meta"]["cache_help_text"] = "Built on demand because no named app cache file was available."
        payload["meta"]["cache_path"] = str(DB_PATH)
        return payload
    return empty_snapshot()


def empty_snapshot() -> dict[str, Any]:
    return {
        "meta": {
            "cache_source": "empty",
            "cache_key": "empty",
            "cache_label": "Unavailable",
            "cache_help_text": "No cache or SQLite database is currently available.",
            "selected_window": {"label": "Unavailable"},
        },
        "stats": {
            "total_posts": 0,
            "total_comments": 0,
            "total_authors": 0,
            "modeled_authors": 0,
            "eligible_model_documents": 0,
            "modeled_documents": 0,
            "modeled_posts": 0,
            "modeled_comments": 0,
            "total_topics": 0,
            "major_topic_count": 0,
            "stance_topic_count": 0,
            "window_label": "Unavailable",
            "corpus_summary": "Pipeline outputs are not available yet.",
        },
        "overview": {
            "summary_line": "Pipeline outputs are not available yet.",
            "monthly_activity": [],
            "activity_highlights": {},
            "top_topics": [],
        },
        "topics": [],
        "topic_table": [],
        "stance": {
            "selected_topic_ids": [],
            "skipped_topics": [],
            "analyzed_topics": [],
        },
        "methods": {
            "notes": [],
            "sections": [],
        },
    }
