# Ethics Note v1

## Project Context

This project uses Reddit r/fitness data collected through Arctic Shift because PRAW was not available for the project setup. The Part 1 pipeline stores posts, comments, author metadata, topic-modeling outputs, stance-analysis outputs, and document text in a local SQLite database:

```text
Part1/data/fitness_part1.sqlite
```

Part 2 then freezes a RAG corpus from that database and creates derived artifacts:
- `307,773` selected documents from the default-safe r/fitness window
- `313,615` text chunks in `Part2/data/chunks/default_rag_chunks_v1.jsonl`
- sentence-transformer embeddings in `Part2/data/embeddings/default_rag_embeddings_v1.sqlite`
- a FAISS dense index in `Part2/data/indices/default_rag_dense_v1.faiss`
- retrieved snippets and provider outputs under `Part2/data/eval_runs/`

The project is local and academic, but the ethical issues are the same ones that would matter in a production RAG system: archived Reddit content can remain usable after the original poster expects it to be gone, and derived artifacts can continue to expose traces of the original text.

## 1. Did Anonymization Still Leave Re-Identification Risks?

Yes. The project uses pseudonymization, not true anonymization.

The SQLite schema stores an `authors` table with both `username` and `pseudonym`, plus `author_id`, deletion flags, and document links. Even if the app or reports show only pseudonyms, the local database still contains a mapping back to usernames. That means someone with database access could reconnect a pseudonym to a Reddit account.

The text itself can also be identifying. Fitness posts often contain combinations of age, gender, height, weight, injury history, training history, food budget, location hints, and rare health circumstances. A single detail may be harmless; the combination can be unique. For example, the corpus includes non-identifying examples like:
- a beginner describing obesity, daily walking, and a very large calorie deficit
- a woman in her late 30s describing difficulty losing weight
- older gym returners in their 50s asking about wrist position and health concerns
- users discussing cost-of-living constraints for high-protein diets
- users describing injury, long-Covid-like fatigue, or disability-related barriers

Those examples are paraphrased here because quoting long personal text would increase the privacy risk.

## 2. Could Username + Posting History + Content Identify Real People?

Yes. Reddit usernames are persistent social identifiers, and posting history can be very specific. A username plus r/fitness content could be cross-referenced with other subreddits, timestamps, writing style, body measurements, progress photos, medical details, or location clues.

Even without displaying usernames, re-identification can happen through the content. If a RAG answer shows a distinctive snippet about a rare injury, an exact diet constraint, or a highly specific age/weight/training combination, a motivated person could search the public web or archives for that text. Embeddings also matter: they do not show text directly, but they are derived from text and can keep deleted or sensitive content retrievable through semantic search unless deletion is propagated.

## 3. Does The System Violate The Right To Be Forgotten?

For a classroom/local project, the risk is limited by access scope, but the system is not fully compatible with a strong right-to-be-forgotten standard as currently designed.

The problem is that Arctic Shift is an archive. If a Reddit user deletes a post after it was archived and after this project ingested it, the local SQLite row, chunk JSONL, embeddings, FAISS index entry, and saved eval artifacts may still preserve that content. The Part 2 RAG system can retrieve and summarize it even if it is no longer visible in the current Reddit interface.

So the system does not intentionally violate user deletion expectations, but it can practically violate them unless deletion requests and source removals are propagated through every artifact.

## 4. What Happens If A User Deletes A Post That Remains In Our DB/Index?

In the current artifact layout, deletion from Reddit would not automatically remove the local copy. The old content could remain in:
- `documents.raw_text` and `documents.clean_text`
- `posts` or `comments`
- `documents_fts`
- chunk JSONL
- embedding SQLite rows
- FAISS vectors
- retrieved snippet artifacts in saved runs
- manual review CSVs or reports if copied there

This creates a "derived artifact" problem. Deleting a row from the raw DB is not enough. The chunk and embedding stores must also be rebuilt or selectively tombstoned, and saved RAG outputs that quote the deleted material need review.

The safest behavior after a deletion request would be:
1. Add a tombstone record for the source document ID and timestamp.
2. Remove or redact raw text in SQLite.
3. Remove the document from FTS.
4. Remove affected chunks.
5. Rebuild or update the embedding store.
6. Rebuild FAISS.
7. Delete or redact saved eval/run artifacts that contain the snippet.
8. Keep only a minimal tombstone log so the document is not accidentally re-ingested.

## 5. Is Full Compliance Realistic For Production RAG?

Full compliance is possible only if deletion is designed into the system from the start. It is not realistic if raw data, chunks, embeddings, and saved outputs are treated as independent static artifacts.

A production RAG system would need stable source IDs, provenance for every chunk and vector, deletion/tombstone logs, re-indexing jobs, and audit checks that verify a removed source cannot still be retrieved. It would also need retention limits and a policy for model outputs, logs, caches, backups, and manual review exports. Even then, perfect compliance is difficult when upstream data comes from third-party archives rather than the live platform.

For this project, the realistic standard is narrower: avoid exposing usernames, avoid quoting sensitive personal text at length, keep artifacts local, document the risk honestly, and make future deletion propagation technically possible.

## Mitigation Options

The most relevant mitigations for this project are:

| Risk | Mitigation |
|---|---|
| Username exposure | Do not show usernames in the Streamlit app, reports, retrieved snippets, or evaluation tables. Use pseudonyms only where author grouping is needed. |
| Re-identification through text | Limit source snippets shown by RAG, paraphrase sensitive examples in reports, and avoid displaying rare combinations of age, body size, medical details, and training history. |
| Raw-to-derived artifact drift | Keep raw data, chunks, embeddings, FAISS, and run artifacts linked by stable `document_id` and `chunk_id` provenance. |
| Deleted content persists | Maintain deletion/tombstone logs and rebuild or update chunks, embeddings, FTS, and FAISS after deletion requests. |
| Over-retention | Set retention limits for raw archives and saved provider outputs, especially raw responses and prompts containing snippets. |
| Unnecessary raw access | Separate raw Reddit text from derived analysis artifacts; use least-privilege access for the raw DB. |
| Sensitive RAG citations | Show short snippets, not full comments or posts; suppress snippets that contain medical, demographic, or highly identifying details. |
| Re-ingestion of removed content | Store tombstones outside the raw corpus so future Arctic Shift pulls do not silently restore deleted material. |

## Reflection

The hardest ethical tension is that the corpus is useful precisely because it contains real, situated user experiences. r/fitness advice about hunger, injury, budget protein, older training, and beginner anxiety is valuable because it is specific. That same specificity creates privacy risk.

For this submission, I treated the data as a local research corpus rather than a public product dataset. I used aggregate counts and short/paraphrased examples in the written reports, and I avoided usernames. But the underlying system still stores linkable author metadata and derived text artifacts. If this moved beyond coursework, deletion propagation, retention limits, snippet minimization, and raw-data separation would need to be engineering requirements, not optional cleanup.
