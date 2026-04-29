# RAG Evaluation Report v1

## Pipeline State

As of automatic runs `20260425T210127Z` and `20260426T054042Z`, the Part 2 RAG pipeline has a frozen corpus, chunk artifacts, embeddings, FAISS dense retrieval, hybrid retrieval, deterministic query classification, grounded answer generation, automated eval reporting, populated BERTScore metrics, complete Groq/Gemini automatic comparison runs, and completed manual review artifacts for the earlier hardened Groq run `20260425T122319Z` and the successful Gemini run `20260426T054042Z`.

The frozen eval set contains `15` questions:
- `8` factual
- `5` opinion-summary
- `2` adversarial/no-answer

## Baseline Vs Hardened Groq

Baseline run: `20260425T093843Z`
- Retrieval hit@k: `0.6154`
- Document hit@k: `0.6154`
- Average ROUGE-L F1: `0.1341`
- Correct adversarial abstentions: `1/2`

Hardened run: `20260425T210127Z`
- Retrieval hit@k: `0.8462`
- Document hit@k: `0.8462`
- Average ROUGE-L F1: `0.1786`
- Average BERTScore precision / recall / F1: `0.7425 / 0.7448 / 0.7423`
- Correct adversarial abstentions: `2/2`

Observed change:
- Retrieval improved from `8/13` to `11/13` answer-bearing hits.
- The prompt-injection baseline issue was eliminated.
- BERTScore is now enabled in the Part 2 Python 3.12 virtualenv via `bert-score==0.3.13`, using `distilbert-base-uncased` with the evaluator defaults.

## Latest Automatic Metrics

Run: `20260425T210127Z`
- Total rows: `15`
- Status: `success=15`
- Answer-bearing rows: `13`
- Retrieval hit@k: `0.8462`
- Document hit@k: `0.8462`
- Average ROUGE-L F1: `0.1786`
- Average BERTScore precision: `0.7425`
- Average BERTScore recall: `0.7448`
- Average BERTScore F1: `0.7423`
- BERTScore populated rows: `13/13` answer-bearing success rows
- BERTScore errors: `0`
- Adversarial correct abstentions: `2/2`

Automatic report artifacts:
- [eval_summary.json](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260425T210127Z/eval_summary.json)
- [eval_summary.md](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260425T210127Z/eval_summary.md)
- [results.csv](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260425T210127Z/results.csv)

## Groq Vs Gemini Comparison

Gemini credentials were found in the supported `Part1/.env` location via `GEMINI_API_KEY` and `GEMINI_MODEL`; no API key values were printed or written to the report. After replacing the earlier blocked key, the Gemini smoke run `20260426T054001Z` succeeded with BERTScore populated, and the full Gemini run `20260426T054042Z` completed successfully.

| Provider | Model | Run used for table | Run status | ROUGE-L F1 | BERTScore F1 | Manual faithfulness |
|---|---|---:|---|---:|---:|---|
| Groq | `llama-3.3-70b-versatile` | `20260425T210127Z` | `15/15` success | `0.1786` | `0.7423` | `10/13 = 76.92%` fully faithful on completed hardened Groq review `20260425T122319Z`; latest BERTScore rerun not manually re-reviewed |
| Gemini | `gemini-2.5-flash` | `20260426T054042Z` | `15/15` success | `0.1607` | `0.7342` | `11/13 = 84.62%` fully faithful on completed Gemini review `20260426T054042Z` |

Gemini automatic report artifacts:
- [eval_summary.json](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260426T054042Z/eval_summary.json)
- [eval_summary.md](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260426T054042Z/eval_summary.md)
- [results.csv](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260426T054042Z/results.csv)
- [manual_review.csv](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260426T054042Z/manual_review.csv)
- [manual_review_completed.csv](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260426T054042Z/manual_review_completed.csv)
- [manual_review_summary.json](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260426T054042Z/manual_review_summary.json)
- [manual_review_summary.md](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260426T054042Z/manual_review_summary.md)

Gemini run details:
- Total rows: `15`
- Status: `success=15`
- Answer-bearing rows: `13`
- Retrieval hit@k: `0.8462`
- Document hit@k: `0.8462`
- Average ROUGE-L F1: `0.1607`
- Average BERTScore precision / recall / F1: `0.7200 / 0.7506 / 0.7342`
- BERTScore populated rows: `13/13` answer-bearing success rows
- Adversarial correct abstentions: `2/2`

Qualitative comparison:
- Retrieval metrics are identical because both providers use the same deterministic classification and hybrid retrieval layer.
- Groq scored higher on lexical-overlap and semantic similarity against the gold answers in this run: ROUGE-L F1 `0.1786` vs Gemini `0.1607`, and BERTScore F1 `0.7423` vs Gemini `0.7342`.
- Both providers preserve the hardened adversarial behavior on this eval set because adversarial/no-answer queries are handled through the deterministic local abstention path before provider generation.
- Gemini's completed manual review is slightly stronger than the earlier completed Groq review: `84.62%` fully faithful vs Groq's `76.92%`, with fewer partial answers (`2` vs `3`) and no citation-validity partials (`13/13` fully valid vs Groq's `12/13`).
- Gemini's advantage is mostly qualitative: it gives fuller, better-calibrated opinion summaries for StrongLifts, recomp, cutting, cardio while bulking, and deloads.

## Manual Review Metrics

Completed manual reviews are available for the earlier hardened Groq run `20260425T122319Z` and the successful Gemini run `20260426T054042Z`. The Groq manual review was not automatically transferred to the newer BERTScore rerun because live model outputs can differ between runs.

Completed review artifact:
- [Groq manual_review_completed.csv](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260425T122319Z/manual_review_completed.csv)
- [Gemini manual_review_completed.csv](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260426T054042Z/manual_review_completed.csv)

Manual review summary:
- [Groq manual_review_summary.json](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260425T122319Z/manual_review_summary.json)
- [Groq manual_review_summary.md](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260425T122319Z/manual_review_summary.md)
- [Gemini manual_review_summary.json](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260426T054042Z/manual_review_summary.json)
- [Gemini manual_review_summary.md](/Users/arvindsrinivass/Desktop/NLP-Project/Part2/data/eval_runs/20260426T054042Z/manual_review_summary.md)

Groq primary manual metrics (`20260425T122319Z`):
- Faithfulness: `10/13 = 76.92%` fully faithful, with `3/13` partial and `0/13` fully unfaithful
- Citation validity: `12/13 = 92.31%` fully valid, with `1/13` partial and `0/13` invalid
- Correct abstention: `2/2 = 100%`
- Faithfulness by question type:
- factual: `6/8 = 75.00%` fully faithful, `2/8` partial
- opinion-summary: `4/5 = 80.00%` fully faithful, `1/5` partial
- Partial answers: `3`

Gemini primary manual metrics (`20260426T054042Z`):
- Faithfulness: `11/13 = 84.62%` fully faithful, with `2/13` partial and `0/13` fully unfaithful
- Citation validity: `13/13 = 100.00%` fully valid, with `0/13` partial and `0/13` invalid
- Correct abstention: `2/2 = 100%`
- Faithfulness by question type:
- factual: `6/8 = 75.00%` fully faithful, `2/8` partial
- opinion-summary: `5/5 = 100.00%` fully faithful, `0/5` partial
- Partial answers: `2`

Interpretation:
- The hardened system is usually grounded and citation-valid when it retrieves the right neighborhood of evidence.
- The main quality distinction is now about completeness and summary calibration rather than outright hallucination.
- Gemini matched Groq on factual faithfulness and abstention safety, but was more complete on opinion summaries in this reviewed run.

## Adversarial Handling

Adversarial behavior is in a good first-release state for this eval set.
- `rag_eval_014` correctly abstains on the out-of-domain quantum mechanics query.
- `rag_eval_015` correctly refuses the prompt-injection request and does not leak hidden instructions.

## Notes And Caveats

- Gemini is configured in the codebase and a replacement key completed the full automatic eval on `2026-04-26`.
- Automated eval now has populated BERTScore precision/recall/F1 values for the latest Groq run. A one-row smoke run `20260425T205641Z` first confirmed BERTScore output on `rag_eval_001` with precision / recall / F1 of `0.7387 / 0.8156 / 0.7752`.
- Manual review metrics here treat `partial` separately instead of collapsing it into `yes` or `no`.
