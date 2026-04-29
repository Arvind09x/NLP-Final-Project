# Part 2

This step freezes the default Part 2 RAG corpus definition against the Part 1 SQLite database and validates that the source corpus is readable, populated, and aligned with the Part 2 plan. The validator reports core table counts, lists the available `subreddit_meta` windows, derives observed per-window post/comment counts from the `documents` table using end-exclusive window bounds, and separates windows that merely have observed data from windows that are trusted enough to serve as the default frozen RAG corpus. Windows with stale metadata or manually interrupted comment ingestion are still surfaced for exploration, but they are labeled as advisory-only rather than default-safe.

## Submission Reports

Current written Part 2 reports:
- [RAG evaluation report v1](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/reports/rag_eval_report_v1.md)
- [Hindi Indian-language task report v1](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/reports/hindi_task_report_v1.md)
- [Bias detection note v1](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/reports/bias_detection_note_v1.md)
- [Ethics/right-to-be-forgotten note v1](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/reports/ethics_note_v1.md)

Run the validator from the repository root:

```bash
python3 Part2/scripts/validate_corpus.py
```

The manifest is written to:

```text
Part2/data/manifests/corpus_manifest_v1.json
```

You can also override the defaults if needed:

```bash
python3 Part2/scripts/validate_corpus.py \
  --db-path /absolute/path/to/fitness_part1.sqlite \
  --manifest-path /absolute/path/to/custom_manifest.json
```

## Chunk Generation

Chunk generation reads the frozen corpus manifest, selects only the default-safe RAG window, streams matching documents from the Part 1 SQLite database, and writes granular chunk artifacts for posts and comments under `Part2/data/`. Comments stay as single chunks unless they are too long; posts are split by paragraph first and then by sentence when needed, with a `1`-sentence overlap between adjacent post chunks to preserve local context for retrieval.

The chunk source contract follows the frozen corpus manifest directly: every document in the selected default window is eligible for chunking under end-exclusive `created_utc` bounds. The chunk builder does not reuse the Part 1 `include_in_modeling` gate, because that flag was created for topic-modeling workflows rather than the frozen Part 2 RAG corpus definition.

If a document's normalized text is empty but its raw text contains URLs, the chunk builder now emits a deterministic URL-fallback chunk instead of dropping the document. Fallback chunks preserve up to `2` URLs and label their origin with `chunk_origin = "url_fallback"`.

Run the chunk builder from the repository root:

```bash
python3 Part2/scripts/build_chunks.py
```

Default outputs:

```text
Part2/data/chunks/default_rag_chunks_v1.jsonl
Part2/data/manifests/chunk_manifest_v1.json
```

The chunk manifest records:
- source corpus manifest path
- selected window
- source document selection rule
- chunk counts
- source document counts
- fallback chunk count
- fallback chunk percentage
- coverage rate
- coverage explanation
- missing-document diagnostics with per-category sample IDs
- generation timestamp
- chunking parameters
- validation summary

Resume mode is artifact-safe: if the existing chunk artifact was built from a different corpus manifest, database path, selected window, or chunking parameters, the command fails and asks you to rerun with `--no-resume` instead of mixing incompatible chunks into one JSONL.

To audit selected-window documents that failed to produce chunks, run:

```bash
python3 Part2/scripts/diagnose_chunk_coverage.py
```

That diagnostic command emits machine-readable JSON with category counts and sample document IDs for:
- `empty_raw_text`
- `deleted_or_removed`
- `normalized_to_empty`
- `tokenization_failed`
- `other`

Optional overrides:

```bash
python3 Part2/scripts/build_chunks.py \
  --source-corpus-manifest /absolute/path/to/corpus_manifest.json \
  --chunk-artifact-path /absolute/path/to/chunks.jsonl \
  --chunk-manifest-path /absolute/path/to/chunk_manifest.json \
  --max-chunk-tokens 220 \
  --post-overlap-sentences 1
```

## Embedding + FAISS Index Build

The next pipeline stage reads the frozen chunk artifact as-is, performs a strict preflight integrity check, embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`, stores vectors incrementally in a resumable SQLite artifact under `Part2/data/embeddings/`, and then builds a FAISS dense index under `Part2/data/indices/`.

The preflight is intentionally strict:
- it hashes the frozen chunk JSONL and chunk manifest
- it checks that the manifest chunk count matches the JSONL line count
- it checks for duplicate `chunk_id`s in the frozen artifact
- it checks the authoritative frozen chunk count before any embedding work begins

If the local frozen chunk artifact does not align with the authoritative state, the build fails loudly and does not attempt to regenerate, reinterpret, or filter chunks.

Run the embedding/index build from the repository root:

```bash
python3 Part2/scripts/build_embeddings_index.py
```

Default outputs:

```text
Part2/data/embeddings/default_rag_embeddings_v1.sqlite
Part2/data/manifests/embedding_manifest_v1.json
Part2/data/indices/default_rag_dense_v1.faiss
```

Optional overrides:

```bash
python3 Part2/scripts/build_embeddings_index.py \
  --chunk-artifact-path /absolute/path/to/chunks.jsonl \
  --chunk-manifest-path /absolute/path/to/chunk_manifest.json \
  --embedding-artifact-path /absolute/path/to/embeddings.sqlite \
  --embedding-manifest-path /absolute/path/to/embedding_manifest.json \
  --faiss-index-path /absolute/path/to/index.faiss \
  --model-name sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 128 \
  --expected-chunk-count 313615
```

Resume safety is artifact-strict. Resume is allowed only if all of these match the existing embedding store metadata:
- chunk artifact path and hash
- chunk manifest path and hash
- expected chunk count
- model name
- embedding dimension
- batch size
- embedding normalization setting

To force a fresh embedding store, rerun with:

```bash
python3 Part2/scripts/build_embeddings_index.py --no-resume
```

## FAISS Smoke Search

After a successful embedding/index build, run a one-query smoke search:

```bash
python3 Part2/scripts/smoke_search_faiss.py \
  --query "best beginner strength routine for fat loss"
```

The smoke script loads the saved FAISS index, embeds the query with the same model recorded in the embedding manifest, and prints top-ranked `chunk_id`s with basic chunk metadata.

## Retrieval Layer v1

Retrieval Layer v1 adds three retrieval modes on top of the frozen chunk artifact and the completed embedding/index build:
- `dense`: query embedding against the saved FAISS index
- `lexical`: document-level retrieval from Part 1 `documents_fts`, then deterministic mapping back to frozen chunks
- `hybrid`: dense plus lexical merged with Reciprocal Rank Fusion (RRF), then deduplicated at the `document_id` level

The retrieval defaults live in the Part 2 config module:
- `dense_top_k = 8`
- `lexical_top_k = 8`
- `hybrid_final_top_k = 5`
- `rrf_constant = 60`

Default retrieval artifacts:

```text
Part2/data/indices/default_rag_dense_v1.faiss
Part2/data/embeddings/default_rag_embeddings_v1.sqlite
Part2/data/chunks/default_rag_chunks_v1.jsonl
Part2/data/manifests/corpus_manifest_v1.json
Part2/data/manifests/chunk_manifest_v1.json
Part2/data/manifests/embedding_manifest_v1.json
Part1/data/fitness_part1.sqlite
```

Run the retrieval debug script from the repository root with the local Part 2 virtualenv:

```bash
Part2/.venv312/bin/python Part2/scripts/retrieve_debug.py \
  --query "best beginner strength routine for fat loss" \
  --mode dense
```

Other supported modes:

```bash
Part2/.venv312/bin/python Part2/scripts/retrieve_debug.py \
  --query "best beginner strength routine for fat loss" \
  --mode lexical

Part2/.venv312/bin/python Part2/scripts/retrieve_debug.py \
  --query "best beginner strength routine for fat loss" \
  --mode hybrid
```

The debug output prints:
- final rank
- `chunk_id`
- `document_id`
- `source_type`
- `chunk_index`
- `chunk_origin`
- `created_utc`
- `title`
- score fields for the active retrieval mode
- a normalized snippet preview

Lexical retrieval is intentionally document-level in v1 because the Part 1 FTS table indexes documents, not chunks. When a retrieved document has multiple frozen chunks, v1 maps it to the deterministic default chunk defined as the first chunk by ascending `chunk_index` and then `chunk_id`. Hybrid retrieval keeps the best dense chunk for a document when dense retrieval is available, and otherwise falls back to the lexical default chunk for that document.

## Query Classification + Class-Specific Routing

The next retrieval layer addition is deterministic query classification with query-time abbreviation expansion. The router normalizes the raw query, expands common fitness abbreviations, classifies the query into one of the planned RAG query classes, selects the matching retrieval defaults, and then calls the existing retrieval layer.

Implemented query classes:
- `factual`
- `opinion-summary`
- `adversarial/no-answer`

Included abbreviation expansions:
- `ppl` -> `push pull legs`
- `5x5` -> `five by five strength program`
- `1rm` -> `one rep max`
- `pr` -> `personal record`
- `bw` -> `bodyweight`
- `rpe` -> `rate of perceived exertion`
- `tdee` -> `total daily energy expenditure`
- `bmr` -> `basal metabolic rate`

Class-specific retrieval defaults:
- `factual`: balanced hybrid retrieval with moderate final top-k
- `opinion-summary`: larger dense and lexical candidate pools with a larger final top-k
- `adversarial/no-answer`: smaller retrieval pool plus surfaced guardrail fields for future abstention work

Run the combined debug script with the local Part 2 virtualenv:

```bash
Part2/.venv312/bin/python Part2/scripts/classify_and_retrieve.py \
  --query "what is a good ppl routine for beginners?"
```

Other examples:

```bash
Part2/.venv312/bin/python Part2/scripts/classify_and_retrieve.py \
  --query "what do people think about cutting while lifting?"

Part2/.venv312/bin/python Part2/scripts/classify_and_retrieve.py \
  --query "what does this subreddit say about quantum mechanics?"
```

The combined debug output prints:
- raw query
- normalized and expanded query
- query type
- confidence
- matched rules
- effective retrieval settings
- top retrieved chunks with metadata and snippet previews

## Evaluation Set

Frozen eval set location:

```text
Part2/data/eval/rag_eval_v1.jsonl
Part2/data/eval/rag_eval_v1_manifest.json
```

Validate the frozen eval set with:

```bash
Part2/.venv312/bin/python Part2/scripts/validate_eval_set.py
```

## Automated Evaluation Runner v1

Run a retrieval-only eval smoke:

```bash
Part2/.venv312/bin/python Part2/scripts/run_rag_eval.py \
  --provider groq \
  --retrieval-only \
  --max-examples 3
```

Run a live Groq eval smoke and persist raw provider responses:

```bash
Part2/.venv312/bin/python Part2/scripts/run_rag_eval.py \
  --provider groq \
  --max-examples 2 \
  --save-raw-response
```

Run both providers across the same filtered eval subset:

```bash
Part2/.venv312/bin/python Part2/scripts/run_rag_eval.py \
  --provider both \
  --max-examples 5
```

Filter by question ID for a one-example smoke:

```bash
Part2/.venv312/bin/python Part2/scripts/run_rag_eval.py \
  --provider gemini \
  --question-id rag_eval_014 \
  --save-raw-response
```

Default eval-run outputs are written under:

```text
Part2/data/eval_runs/YYYYMMDDTHHMMSSZ/
```

Each run includes:
- `run_manifest.json`
- `results.jsonl`
- `results.csv`
- per-question/provider artifacts under `artifacts/`
- raw provider responses under `raw_responses/` when `--save-raw-response` is used

The flat result rows include retrieval hit metrics, ROUGE-L F1, and optional BERTScore precision/recall/F1 columns. BERTScore uses the optional `bert-score` package with `distilbert-base-uncased` by default. If the package or model is unavailable in `Part2/.venv312`, eval runs continue and record empty `bert_score_*` values plus a clear `run_manifest.json` status message. To enable it locally:

```bash
Part2/.venv312/bin/python -m pip install bert-score
```

Pass `--skip-bert-score` to keep smoke runs lightweight even when the optional dependency is installed.

## Hindi Translation Eval

The Indian-language task now uses a frozen Hindi translation dataset under:

```text
Part2/data/indian_language/hindi_translation_eval_v1.jsonl
Part2/data/indian_language/hindi_translation_eval_v1_manifest.json
Part2/data/indian_language/hindi_manual_review_template_v1.csv
```

Run the Hindi eval with:

```bash
Part2/.venv312/bin/python Part2/scripts/run_hindi_eval.py \
  --provider both \
  --save-raw-response
```

For a quick smoke run:

```bash
Part2/.venv312/bin/python Part2/scripts/run_hindi_eval.py \
  --provider both \
  --max-examples 4 \
  --skip-bert-score
```

Default Hindi eval outputs are written under:

```text
Part2/data/indian_language_runs/YYYYMMDDTHHMMSSZ/
```

Each run includes:
- `run_manifest.json`
- `results.jsonl`
- `results.csv`
- `summary.json`
- `summary.md`
- `manual_review.csv`
- per-example/provider artifacts under `artifacts/`
- raw provider responses under `raw_responses/` when `--save-raw-response` is used

The automatic metrics are `chrF` and optional multilingual `BERTScore` using `bert-base-multilingual-cased`. If the multilingual model cannot be loaded, the run still completes and records empty `bert_score_*` values plus a clear status message in `run_manifest.json`.

## Eval Report Scaffold

Build the manual review sheet and summary report for a completed run with:

```bash
Part2/.venv312/bin/python Part2/scripts/build_eval_report.py \
  --run-dir Part2/data/eval_runs/20260425T093843Z
```

By default, the report files are written into the selected run directory:

```text
Part2/data/eval_runs/20260425T093843Z/manual_review.csv
Part2/data/eval_runs/20260425T093843Z/eval_summary.json
Part2/data/eval_runs/20260425T093843Z/eval_summary.md
```

You can override the output location if needed:

```bash
Part2/.venv312/bin/python Part2/scripts/build_eval_report.py \
  --run-dir Part2/data/eval_runs/20260425T093843Z \
  --output-dir Part2/reports/20260425T093843Z
```

## Prompting + LLM Provider Integration v1

Answer generation now sits on top of classification plus retrieval. The generation layer builds a query-type-aware grounded prompt from retrieved chunks only, assigns stable source labels such as `S1`, `S2`, and `S3` to the retrieved snippets, requires providers to cite only those labels, normalizes provider output into one shared answer contract, and can optionally persist raw provider responses under `Part2/data/runs/` for debugging.

Retrieval and generation are separate on purpose. Retrieval finds the most relevant frozen corpus evidence; generation sends that retrieved context to Groq or Gemini and asks for one cited answer grounded only in those snippets. `--retrieval-only` skips the LLM entirely and prints the routing plus evidence preview. Adversarial or out-of-domain queries may also bypass the provider through deterministic abstention, so those runs can return a safe no-answer result without a live model call. Use `--show-prompt` to verify that `retrieved_context` is present in the provider prompt before debugging model behavior.

Supported providers:
- `groq`
- `gemini`

Environment variables:
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- `GEMINI_MODEL`

The CLI and Streamlit app will read these from your shell environment or from a
local `.env` file placed at the repository root, `Part1/.env`, or `Part2/.env`.
The Streamlit RAG toggles mean:
- Provider: choose which LLM receives the retrieved context.
- Evidence only: run classification and retrieval, but skip Groq/Gemini.
- Save raw response: persist the prompt and provider payload in `Part2/data/runs/`.
- Show prompt: display the exact grounded prompt built from retrieved snippets.

Run retrieval-only mode without calling an LLM:

```bash
Part2/.venv312/bin/python Part2/scripts/answer_query.py \
  --query "what is a good ppl routine for beginners?" \
  --retrieval-only
```

Show the exact grounded prompt that would be used for generation:

```bash
Part2/.venv312/bin/python Part2/scripts/answer_query.py \
  --query "what do people think about cutting while lifting?" \
  --provider groq \
  --retrieval-only \
  --show-prompt
```

The same prompt inspection also works with generation enabled:

```bash
Part2/.venv312/bin/python Part2/scripts/answer_query.py \
  --query "what do people think about cutting while lifting?" \
  --provider groq \
  --show-prompt
```

Run answer generation with Groq:

```bash
Part2/.venv312/bin/python Part2/scripts/answer_query.py \
  --query "what is a good ppl routine for beginners?" \
  --provider groq
```

Run answer generation with Gemini:

```bash
Part2/.venv312/bin/python Part2/scripts/answer_query.py \
  --query "what do people think about cutting while lifting?" \
  --provider gemini
```

Compare both providers on the same retrieved context:

```bash
Part2/.venv312/bin/python Part2/scripts/answer_query.py \
  --query "what is a good ppl routine for beginners?" \
  --compare-providers
```

Persist raw provider output for debugging:

```bash
Part2/.venv312/bin/python Part2/scripts/answer_query.py \
  --query "what is a good ppl routine for beginners?" \
  --provider groq \
  --save-raw-response
```

Override the final retrieved context size cleanly:

```bash
Part2/.venv312/bin/python Part2/scripts/answer_query.py \
  --query "what is a good ppl routine for beginners?" \
  --provider groq \
  --top-k 3
```

The answer contract includes:
- `query`
- `normalized_query`
- `query_type`
- `answer_text`
- `citations`
- `retrieved_snippets`
- `insufficient_evidence`
- `provider`
- `model`
- `raw_response_path`

The provider-facing JSON schema is:

```json
{
  "answer_text": "Grounded answer text",
  "citations": ["S1", "S3"],
  "insufficient_evidence": false
}
```

Each normalized citation in the final answer result includes:
- `source_label`
- `chunk_id`
- `document_id`
- `title`
- `source_type`
- `created_utc`
- `snippet`

## Remaining Deliverables

The latest frozen RAG evaluation is documented in `Part2/reports/rag_eval_report_v1.md`, and the Hindi task scaffold/report is in `Part2/reports/hindi_task_report_v1.md`. Remaining Part 2 work is:
- run the live Hindi evaluation for Groq and Gemini if API/network access is available
- copy the resulting Hindi metrics table and edge-case findings into the final write-up
- write corpus-grounded bias and ethics notes tied to the frozen Reddit corpus and RAG outputs
