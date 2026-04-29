from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

import requests
from requests import Response
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from fitness_reddit_analyzer.config import (
    ARCTIC_SHIFT_BASE_URL,
    ARCTIC_SHIFT_PAGE_SIZE,
    ARCTIC_SHIFT_TIMEOUT_SECONDS,
    ARCTIC_SHIFT_USER_AGENT,
)


class ArcticShiftError(RuntimeError):
    """Raised when Arctic Shift cannot satisfy a request."""


@dataclass(frozen=True)
class SearchPage:
    items: list[dict[str, Any]]
    metadata: dict[str, Any]


class ArcticShiftClient:
    def __init__(self) -> None:
        self.headers = {"User-Agent": ARCTIC_SHIFT_USER_AGENT}

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((requests.RequestException, ArcticShiftError)),
    )
    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{ARCTIC_SHIFT_BASE_URL}{path}"
        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=ARCTIC_SHIFT_TIMEOUT_SECONDS,
            )
        except requests.RequestException:
            return self._get_with_curl(url, params)
        self._raise_for_status(response)
        payload = response.json()
        if not isinstance(payload, dict) or "data" not in payload:
            raise ArcticShiftError(f"Unexpected response payload for {url}: {json.dumps(payload)[:500]}")
        return payload

    def _get_with_curl(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urlencode([(key, value) for key, value in params.items() if value is not None], doseq=True)
        completed = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--max-time",
                str(ARCTIC_SHIFT_TIMEOUT_SECONDS),
                "-H",
                f"User-Agent: {ARCTIC_SHIFT_USER_AGENT}",
                f"{url}?{query}",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict) or "data" not in payload:
            raise ArcticShiftError(f"Unexpected curl payload for {url}: {completed.stdout[:500]}")
        return payload

    @staticmethod
    def _raise_for_status(response: Response) -> None:
        if response.status_code == 429:
            raise ArcticShiftError("Arctic Shift rate limit encountered")
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise ArcticShiftError(str(exc)) from exc

    def search_posts(
        self,
        subreddit: str,
        after: int,
        before: int,
        *,
        limit: int = ARCTIC_SHIFT_PAGE_SIZE,
    ) -> SearchPage:
        payload = self._get(
            "/api/posts/search",
            {
                "subreddit": subreddit,
                "after": after,
                "before": before,
                "limit": limit,
                "sort": "asc",
                "sort_type": "created_utc",
            },
        )
        return SearchPage(items=list(payload.get("data", [])), metadata=self._extract_metadata(payload))

    def search_comments(
        self,
        *,
        link_id: str | None = None,
        subreddit: str | None = None,
        after: int | None = None,
        before: int | None = None,
        limit: int = ARCTIC_SHIFT_PAGE_SIZE,
    ) -> SearchPage:
        params: dict[str, Any] = {"limit": limit}
        if link_id:
            params["link_id"] = link_id
        if subreddit:
            params["subreddit"] = subreddit
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before
        params["sort"] = "asc"
        params["sort_type"] = "created_utc"
        payload = self._get("/api/comments/search", params)
        return SearchPage(items=list(payload.get("data", [])), metadata=self._extract_metadata(payload))

    def metadata_only_post_count(self, subreddit: str, after: int, before: int) -> int | None:
        payload = self._get(
            "/api/posts/search",
            {
                "subreddit": subreddit,
                "after": after,
                "before": before,
                "limit": 1,
                "sort": "asc",
                "sort_type": "created_utc",
            },
        )
        return self._coerce_total_count(payload)

    @staticmethod
    def _extract_metadata(payload: dict[str, Any]) -> dict[str, Any]:
        metadata = payload.get("metadata") or payload.get("meta") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        total_count = ArcticShiftClient._coerce_total_count(payload)
        if total_count is not None and "total_results" not in metadata:
            metadata["total_results"] = total_count
        return metadata

    @staticmethod
    def _coerce_total_count(payload: dict[str, Any]) -> int | None:
        candidates = [
            payload.get("metadata", {}).get("total_results") if isinstance(payload.get("metadata"), dict) else None,
            payload.get("meta", {}).get("total_results") if isinstance(payload.get("meta"), dict) else None,
            payload.get("total_results"),
            payload.get("total"),
        ]
        for candidate in candidates:
            if isinstance(candidate, int):
                return candidate
            if isinstance(candidate, str) and candidate.isdigit():
                return int(candidate)
        return None


def utc_start_of_month(year: int, month: int) -> int:
    return int(datetime(year=year, month=month, day=1, tzinfo=UTC).timestamp())
