# Hindi Indian-Language Task Report v1

## Chosen Language

The chosen Indian language is `Hindi`, which matches the default already recorded in [Part2/PLAN.md](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/PLAN.md:10).

## Task Format

This section uses an `English/code-mixed fitness text -> Hindi translation` task. I kept the task translation-based instead of cross-lingual RAG QA because it cleanly isolates multilingual generation quality while still reusing the existing Groq/Gemini provider abstraction.

The frozen dataset lives at:
- [hindi_translation_eval_v1.jsonl](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/indian_language/hindi_translation_eval_v1.jsonl)
- [hindi_translation_eval_v1_manifest.json](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/indian_language/hindi_translation_eval_v1_manifest.json)
- [hindi_manual_review_template_v1.csv](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/indian_language/hindi_manual_review_template_v1.csv)

## Dataset Design

The dataset contains `20` reference examples.
- `13` are adapted from the frozen RAG evaluation questions and gold-answer summaries.
- `7` are targeted edge-case probes written in the same r/fitness vocabulary.

Included difficult cases:
- code-mixed English/Hindi phrasing
- Reddit and fitness slang such as `OP`, `natty`, `cope`, `bulk`, `cut`, and `panic-cut`
- named entities such as `StrongLifts`, `Starting Strength`, and `Reddit PPL`
- abbreviations such as `PPL`, `TDEE`, `BMR`, `PR`, `5x5`, and `recomp`

The prompt contract intentionally allows common fitness abbreviations and named programs to stay in Latin script when that sounds more natural in Hindi, which is closer to how Indian gym communities actually write.

## Metrics

Implemented metrics:
- `chrF`
- multilingual `BERTScore` using `bert-base-multilingual-cased` when the optional model is available
- graceful per-run fallback when `bert-score` is missing or the multilingual model cannot be loaded
- manual review fields for `fluency` and `adequacy`

Live results can be generated with:

```bash
Part2/.venv312/bin/python Part2/scripts/run_hindi_eval.py --provider both --save-raw-response
```

If you want a lighter smoke run first:

```bash
Part2/.venv312/bin/python Part2/scripts/run_hindi_eval.py --provider both --max-examples 4 --skip-bert-score
```

For Gemini smoke testing, the runner now supports explicit generation controls:

```bash
Part2/.venv312/bin/python Part2/scripts/run_hindi_eval.py --provider gemini --example-id hin_eval_001 --max-tokens 256 --skip-bert-score --save-raw-response
```

Current status in this workspace:
- Existing Groq run: [20260428T180014Z](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/indian_language_runs/20260428T180014Z/summary.json)
- New Gemini run after the plain-text invocation fix: [20260428T182631Z](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/indian_language_runs/20260428T182631Z/summary.json)

| Provider | Average chrF | Average BERTScore F1 | Notes |
|---|---:|---:|---|
| Groq | 0.4674 | 0.8240 | 20/20 successful rows in run `20260428T180014Z`; 19 rows had BERTScore populated and 1 row had a BERTScore scoring error. |
| Gemini | 0.5629 | 0.8638 | 20/20 successful rows in run `20260428T182631Z`; Gemini was invoked in plain-text mode for translation. |

## Manual Review

Targeted manual review was completed on `8` difficult examples for each provider, using the provider-specific [Groq review sheet](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/indian_language_runs/20260428T180014Z/manual_review.csv) and [Gemini review sheet](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/indian_language_runs/20260428T182631Z/manual_review.csv). This focused `8`-row sample per provider is consistent with the assignment guidance to manually score `5-10` outputs, and the selected rows were intentionally biased toward harder code-mixed and slang-heavy examples.

| Provider | Reviewed rows | Avg fluency | Avg adequacy | Main note |
|---|---:|---:|---:|---|
| Groq | 8 | 3.75 | 4.38 | Meaning usually survives, but the wording is often more formal or awkward than natural Hindi gym-speak. |
| Gemini | 8 | 4.62 | 4.75 | It stays closer to fluent code-mixed Hindi, with only minor clause awkwardness or slight wording softening. |

- Gemini outperformed Groq on both manual dimensions in this targeted review, especially on fluency for code-mixed fitness advice and Reddit-style phrasing.
- Groq usually preserved the core meaning, so the adequacy gap was smaller than the fluency gap, but it more often over-formalized slang and gym terminology.
- Both providers generally kept named entities and useful abbreviations such as `TDEE`, `PR`, `5x5`, `StrongLifts`, and `Reddit PPL` intact.
- Groq had the clearest miss on script/register fit in one example that came back fully in Roman-script Hinglish, while Gemini’s weaker cases were still mostly natural Devanagari Hindi with small stylistic shifts.

## Gemini Diagnosis

The English RAG answer-generation path asks providers for a structured JSON answer and parses the returned JSON contract. Gemini should keep `response_mime_type="application/json"` for that path.

The Hindi translation eval path asks for only natural Hindi text. The shared Gemini provider had been forcing JSON MIME config for all `google.genai` calls, so Hindi translation prompts were being sent through a structured-response configuration even though the prompt did not request JSON. The provider now has an explicit generation mode:
- `generation_mode="json"` preserves JSON MIME config for English/RAG structured responses.
- `generation_mode="text"` omits `response_mime_type` for Hindi translation.

Earlier Gemini Hindi artifacts showed provider/API failures rather than a clean context-length issue:
- [20260428T180543Z hin_eval_001](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/indian_language_runs/20260428T180543Z/artifacts/hin_eval_001__gemini.json) failed with `403 PERMISSION_DENIED` and message `Your project has been denied access. Please contact support.`
- A sandboxed smoke run after the code fix, [20260428T182324Z](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/indian_language_runs/20260428T182324Z/artifacts/hin_eval_001__gemini.json), failed with DNS resolution error `[Errno 8] nodename nor servname provided, or not known`.
- The same smoke with network/API access, [20260428T182343Z](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/indian_language_runs/20260428T182343Z/artifacts/hin_eval_001__gemini.json), succeeded with `max_tokens=256`.
- The code-mixed abbreviation smoke, [20260428T182412Z](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/indian_language_runs/20260428T182412Z/artifacts/hin_eval_004__gemini.json), also succeeded with `max_tokens=256`.

Conclusion: the original Hindi failure was not explained by prompt length. It was a combination of an incorrect structured-response Gemini invocation for plain translation and environment/API availability failures. With plain-text Gemini invocation and working network/API access, the full Hindi Gemini run completes.

## Edge-Case Analysis Plan

The most important error buckets to watch in the generated `summary.md` are:
- whether providers preserve `PPL`, `TDEE`, `PR`, and `5x5` instead of over-translating them
- whether code-mixed inputs become awkwardly over-formal Hindi
- whether slang like `natty`, `cope`, `bulk`, and `cut` is preserved in a way that still sounds natural to Hindi-speaking gym users
- whether named entities such as `StrongLifts` and `Starting Strength` are copied exactly

After a live run, the run directory will include:
- `results.jsonl`
- `results.csv`
- `summary.json`
- `summary.md`
- `manual_review.csv`
- `run_manifest.json`

Those artifacts are the intended source for the final metrics table and qualitative edge-case discussion in the submission write-up.
