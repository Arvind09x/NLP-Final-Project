from __future__ import annotations

from fitness_reddit_analyzer.build_app_cache import run as run_build_app_cache
from fitness_reddit_analyzer.config import PIPELINE_STAGES, ensure_directories
from fitness_reddit_analyzer.db import initialize_database
from fitness_reddit_analyzer.fit_stance import run as run_fit_stance
from fitness_reddit_analyzer.fit_topics import run as run_fit_topics
from fitness_reddit_analyzer.ingest_comments import run as run_ingest_comments
from fitness_reddit_analyzer.ingest_posts import run as run_ingest_posts
from fitness_reddit_analyzer.prepare_features import run as run_prepare_features


def bootstrap_project() -> None:
    ensure_directories()
    initialize_database()


def run_stage(stage_name: str) -> None:
    if stage_name not in PIPELINE_STAGES:
        raise ValueError(f"Unknown stage: {stage_name}")
    bootstrap_project()
    if stage_name == "ingest_posts":
        run_ingest_posts()
        return
    if stage_name == "ingest_comments":
        run_ingest_comments()
        return
    if stage_name == "prepare_features":
        run_prepare_features()
        return
    if stage_name == "fit_topics":
        run_fit_topics()
        return
    if stage_name == "fit_stance":
        run_fit_stance()
        return
    if stage_name == "build_app_cache":
        run_build_app_cache()
        return
    print(f"Stage '{stage_name}' is scaffolded and ready for implementation.")
