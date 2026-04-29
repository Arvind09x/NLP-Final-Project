# Part 2 Plan

## Scope Rules
- All Part 2 code, configs, outputs, reports, and tests live under `Part2/`.
- `Part1/` is treated as read-only input. Part 2 reads the SQLite corpus and cached outputs from Part 1 but does not modify them.
- RAG is the first workstream. Translation, bias, and ethics come after the RAG system is working and evaluated.

## Working Assumptions
- Default LLM endpoints for planning: `Groq` and `Google AI Studio (Gemini)`.
- Default Indian language for planning: `Hindi`.
- RAG v1 uses the primary Part 1 corpus window, not the partial 2018 window.
- Query handling is class-aware: `factual`, `opinion-summary`, and `adversarial/no-answer`.
- Delivery stays CLI-first unless Part 2 later needs a demo UI.

## Current State From Part 1
- `Part1` already contains a usable SQLite corpus with `posts`, `comments`, `documents`, `documents_fts`, and an empty `document_embeddings` table.
- Current corpus size in `Part1/data/fitness_part1.sqlite`: `40,520` posts, `434,530` comments, `475,050` unified documents.
- `documents_fts` is already populated for all documents, which is useful for lexical fallback retrieval.
- `document_embeddings` has `0` rows, so Part 2 must build its own embedding/index pipeline.
- Topic and stance outputs exist for the main window, but the 2018 comment-ingestion checkpoint is still marked `running`; do not use that partial era in the first RAG release.

## Recommended Part 2 Layout
- [ ] Create `Part2/src/part2_rag/`.
- [ ] Create `Part2/scripts/`.
- [ ] Create `Part2/tests/`.
- [ ] Create `Part2/data/` for generated indices, manifests, eval sets, and run outputs.
- [ ] Create `Part2/reports/` for tables, notes, and final writeups.
- [ ] Create `Part2/README.md` after the first runnable pipeline exists.

## Phase 1: RAG System

### 1. Corpus Freeze And Validation
- [ ] Add a config module that points to the Part 1 SQLite database and Part 2 output directories.
- [ ] Add a corpus validation script that reports post/comment/document counts and confirms the database path is readable.
- [ ] Define the exact RAG corpus slice for v1: main `r/fitness` window only.
- [ ] Write a corpus manifest file in `Part2/data/` that records the DB path, selection rule, counts, and generation timestamp.
- [ ] Add a guard that fails fast if the Part 1 DB is missing or if the selected corpus count is `0`.

### 2. Chunking Pipeline
- [ ] Define a chunk schema with at least: `chunk_id`, `document_id`, `source_type`, `source_id`, `post_id`, `parent_id`, `link_id`, `created_utc`, `author_id`, `title`, `chunk_text`, `chunk_index`, `token_estimate`.
- [ ] Keep comments as single chunks unless they exceed the max chunk size.
- [ ] Split posts by paragraph boundaries first, then by sentence boundaries if still too long.
- [ ] Set adjacent post chunk overlap to `1` sentence, targeting roughly `20-30` tokens.
- [ ] Preserve enough metadata to cite the original post/comment exactly.
- [ ] Add retrieval-time context hydration so a retrieved comment can bring along its parent post and a few high-signal sibling comments when needed.
- [ ] Build a chunk-generation script that reads from `Part1` SQLite and writes chunk artifacts under `Part2/data/`.
- [ ] Save chunks in a stable format such as `parquet` or `jsonl`.
- [ ] Add a validation script that checks chunk counts, empty-chunk rate, max length, and missing metadata rate.

### 3. Embeddings And Index Build
- [ ] Use `sentence-transformers/all-MiniLM-L6-v2` as the first embedding model.
- [ ] Implement a batched embedding job for all chunks.
- [ ] Store chunk metadata separately from vector data so the index can be rebuilt without touching the source DB.
- [ ] Build a FAISS index for dense retrieval inside `Part2/data/indices/`.
- [ ] Record the vector-store decision explicitly: keep `FAISS` by default, and note that `ChromaDB` is the fallback if metadata bookkeeping becomes the main source of implementation friction.
- [ ] Persist an embedding manifest with model name, embedding dimension, batch size, corpus hash, and build timestamp.
- [ ] Add resumability so a failed embedding job can continue without recomputing completed batches.
- [ ] Add an integrity check that embedding count equals chunk count.

### 4. Retrieval Layer
- [ ] Implement a lightweight query classifier for `factual`, `opinion-summary`, and `adversarial/no-answer` questions.
- [ ] Add a configurable abbreviation-expansion map for common fitness acronyms at query time.
- [ ] Implement dense retrieval over the FAISS index.
- [ ] Implement lexical retrieval using the existing `documents_fts` table in the Part 1 SQLite database.
- [ ] Implement a hybrid retrieval strategy that merges dense and lexical candidates.
- [ ] Use class-specific retrieval defaults so factual and opinion queries do not share the same top-k settings.
- [ ] Add deduplication at the `document_id` level before final context assembly.
- [ ] Use `Reciprocal Rank Fusion (RRF)` as the default merge-and-rank strategy for dense plus lexical candidates.
- [ ] Add an optional reranking stage that reranks the fused top candidates and trims them to the final context set.
- [ ] Define the initial reranking policy explicitly: rerank top `50`, keep top `5`, and disable it if latency is unacceptable during early debugging.
- [ ] Set and document the default retrieval parameters: top-k dense, top-k lexical, final top-k context chunks.
- [ ] Add a similarity-threshold guardrail for adversarial or unsupported questions before generation.
- [ ] Add a retrieval debug mode that prints retrieved chunk IDs, scores, and source metadata for one query.

### 5. Prompting And Answer Generation
- [ ] Define a single answer contract: answer text, cited sources, retrieved snippets, and an `insufficient_evidence` flag.
- [ ] Create a provider interface for interchangeable LLM backends.
- [ ] Implement a `Groq` adapter.
- [ ] Implement a `Google AI Studio` adapter.
- [ ] Add environment-based config for API keys, model names, and generation settings.
- [ ] Write a grounded answer prompt that uses only retrieved context.
- [ ] Require source-aware answers with explicit citations in the prompt contract.
- [ ] Instruct the model to say when evidence is missing.
- [ ] Instruct the model not to invent answers for adversarial questions.
- [ ] Add prompt variants keyed off the query class so factual and opinion-summary questions are framed differently.
- [ ] Add response normalization so both providers return the same structured output shape.
- [ ] Save raw provider responses for later debugging.

### 6. User-Facing QA Entry Point
- [ ] Build a CLI entry point for querying the RAG system end to end.
- [ ] Print the final answer plus citations and retrieved source previews.
- [ ] Add a switch to compare both providers on the same query.
- [ ] Add a switch to dump retrieval-only output without generation.
- [ ] After the CLI works, decide whether a lightweight Streamlit UI is needed for demo convenience.

### 7. Evaluation Set Construction
- [ ] Create an evaluation dataset file in `Part2/data/eval/`.
- [ ] Write at least `15` question-answer pairs based on the actual `r/fitness` corpus.
- [ ] Include factual community questions.
- [ ] Include opinion-summary questions.
- [ ] Include at least `2` adversarial no-answer questions.
- [ ] For each example, store: `question_id`, `question`, `question_type`, `gold_answer`, `expected_has_answer`, `supporting_document_ids`, `notes`.
- [ ] Review each gold answer against the corpus before freezing the set.
- [ ] Freeze the evaluation set before meaningful prompt tuning begins.

### 8. Automated Evaluation Runner
- [ ] Build a runner that executes the full RAG pipeline for every evaluation question and every provider.
- [ ] Save per-run artifacts: retrieved chunks, final prompt, model output, latency, and any provider metadata available.
- [ ] Compute `ROUGE-L`.
- [ ] Compute `BERTScore`.
- [ ] Store results in a flat table under `Part2/reports/`.
- [ ] Add a report script that aggregates metrics by provider and by question type.

### 9. Manual Faithfulness Review
- [ ] Add a manual review sheet for all evaluation outputs.
- [ ] Mark each answer as faithful or not faithful.
- [ ] Mark each adversarial answer as correctly abstained or not.
- [ ] Compute faithfulness percentage across the test set for each provider.
- [ ] Write a short error taxonomy: retrieval miss, citation miss, hallucination, incomplete summary, overgeneralization.

### 10. Comparative Report For The RAG Section
- [ ] Produce a final comparison table with `ROUGE-L`, `BERTScore`, and faithfulness percentage for both providers.
- [ ] Add a short qualitative analysis of where each model succeeds and fails on this corpus.
- [ ] Include at least a few concrete failure examples with linked run artifacts.
- [ ] Freeze the exact models used so results are reproducible.

### 11. Testing And Hardening
- [ ] Add unit tests for chunk generation.
- [ ] Add unit tests for metadata preservation and citation mapping.
- [ ] Add unit tests for abbreviation expansion.
- [ ] Add unit tests for the query classifier.
- [ ] Add unit tests for retrieval deduplication and hybrid merge behavior.
- [ ] Add unit tests for the adversarial similarity-threshold exit.
- [ ] Add smoke tests for both provider adapters with mocked responses.
- [ ] Add a regression test for adversarial no-answer handling.
- [ ] Add one end-to-end smoke test that runs retrieval plus a mocked generator.

## Phase 2: Indian Language Task
- [ ] Confirm the target language. Default: `Hindi`.
- [ ] Reuse the same provider abstraction from the RAG system.
- [ ] Choose the task format. Recommendation: `cross-lingual QA` because it reuses the English corpus and the RAG pipeline.
- [ ] Create at least `20` reference examples.
- [ ] Store source text, target prompt, reference output, and edge-case tags.
- [ ] Deliberately include code-mixed text, Reddit slang, and named entities.
- [ ] Run both providers on the full set.
- [ ] Compute `chrF`.
- [ ] Compute multilingual `BERTScore`.
- [ ] Manually score a subset for fluency and adequacy.
- [ ] Write a short error analysis focused on edge cases.

## Phase 3: Bias Detection Note
- [ ] Design a small probe set grounded in this corpus.
- [ ] Include probes for demographic assumptions, normative fitness advice, and community stereotype carryover.
- [ ] Run both providers on the probe set with and without retrieved context where useful.
- [ ] Record whether the model amplifies, softens, ignores, or accurately reflects corpus bias.
- [ ] Save evidence snippets from both the corpus and the model outputs.
- [ ] Write a short findings note with concrete examples.

## Phase 4: Ethics Note
- [ ] Audit the corpus for re-identification risk using combinations of username history, unique life details, and post text.
- [ ] Document at least a few concrete re-identification scenarios from this dataset.
- [ ] Analyse what happens when deleted content remains in the stored corpus or index.
- [ ] Explain why full right-to-be-forgotten compliance is hard in a production RAG pipeline.
- [ ] Document mitigation options: re-indexing, tombstoning, deletion logs, and retention limits.
- [ ] Write the final reflective ethics note tied to this exact project, not generic Reddit ethics.

## Immediate Next Tasks
- [ ] Create the Part 2 scaffold under `Part2/`.
- [ ] Implement the corpus validation script.
- [ ] Implement chunk generation.
- [ ] Implement embeddings plus FAISS build.
- [ ] Implement hybrid retrieval.
- [ ] Implement the query classifier and class-specific retrieval config.
- [ ] Implement the two provider adapters.
- [ ] Build the evaluation set before tuning prompts too aggressively.

## Resolved Defaults
- [ ] Keep `Groq + Google AI Studio (Gemini)` as the default provider pair.
- [ ] Keep `Hindi` as the default Indian language.
- [ ] Keep the delivery path CLI-first and add UI only if later needed.
