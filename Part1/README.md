# NLP Project Part 1: `r/fitness`

Python-first implementation for Ashoka University NLP Project Part 1.

## Environment

```bash
pyenv local 3.12.4
pyenv exec python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run The App

```bash
./run_app.sh
```

## Project Layout

```text
src/fitness_reddit_analyzer/
  cli.py              # Pipeline entrypoint
  config.py           # Paths and shared constants
  db.py               # SQLite schema and helpers
  pipeline.py         # Pipeline stage placeholders
  app_data.py         # Streamlit data access helpers
app.py                # Streamlit frontend entrypoint
```

## Planned CLI Stages

```bash
PYTHONPATH=src python -m fitness_reddit_analyzer.cli ingest_posts
PYTHONPATH=src python -m fitness_reddit_analyzer.cli ingest_comments
PYTHONPATH=src python -m fitness_reddit_analyzer.cli prepare_features
PYTHONPATH=src python -m fitness_reddit_analyzer.cli fit_topics
PYTHONPATH=src python -m fitness_reddit_analyzer.cli fit_stance
PYTHONPATH=src python -m fitness_reddit_analyzer.cli build_app_cache
```

## Corpus Audit Utility

```bash
PYTHONPATH=src python -m fitness_reddit_analyzer.corpus_audit --source sqlite
PYTHONPATH=src python -m fitness_reddit_analyzer.corpus_audit --source arctic --start 2023-01-01 --end 2023-10-01
```

## Notes

- Arctic Shift will be used instead of PRAW.
- SQLite is the system of record for Part 1.
- The schema is embedding-ready for Part 2, but embeddings are not populated yet.
- `ingest_posts` is implemented first and includes checkpointed, idempotent post ingestion.
- `run_app.sh` sets `PYTHONPATH=src` so Streamlit can import the project package cleanly.
