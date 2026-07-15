# rag-service eval results

Corpus: 6 documents, 12 chunks. Dataset: 12 answerable + 2 unanswerable questions.
Embeddings: `gemini-embedding-001` (provider `gemini`).

## Retrieval (k=5, gold-span matching, answerable questions only)

| metric | value |
|---|---|
| hit@1 | 0.67 |
| hit@3 | 1.00 |
| hit@5 | 1.00 |
| MRR | 0.83 |

| id | question | first relevant rank |
|---|---|---|
| q01 | What was Meridian Semiconductors' total revenue in fiscal 2025 and how | 1 |
| q02 | Which of Meridian's segments grew fastest in fiscal 2025, and by how m | 1 |
| q03 | How did Meridian's gross margin change in fiscal 2025 and what drove i | 2 |
| q04 | How concentrated is Meridian's customer base? | 2 |
| q05 | What is Meridian Semiconductors' revenue exposure to China? | 2 |
| q06 | What share of Northwind Retail Group's sales came from digital commerc | 1 |
| q07 | What were Northwind's comparable store sales in Q3 fiscal 2025? | 1 |
| q08 | What revenue guidance did Northwind give for the fourth quarter? | 1 |
| q09 | How much share repurchase authorization does Northwind have remaining? | 1 |
| q10 | What is Helios Energy's battery storage attach rate and how has it cha | 1 |
| q11 | What full-year installation guidance did Helios Energy provide? | 2 |
| q12 | Describe Helios Energy's liquidity position at the end of the second q | 1 |

## Answer quality (partial — free-tier daily quota)

Generator/judge model: `gemini-2.5-flash-lite`. Google's free tier caps
`generate_content` at 20 requests/day *per model*; a full run needs 28 calls
(14 questions x 2: generate + judge), which exceeds it even split across
`gemini-2.5-flash` and `gemini-2.5-flash-lite` in one day. Results below cover
the first 6 answerable questions, run to completion before the quota hit.
Unanswerable/abstention questions (q13, q14) were not reached — pending a
quota reset (see DECISIONS.md #11).

| metric | value |
|---|---|
| faithfulness (judged, 6 of 12 attempted) | 6/6 = 1.00 |
| correct abstention | not run (quota) |
| wrong abstention (answerable) | 0/6 |

| id | faithful | judge reason |
|---|---|---|
| q01 | True | Answer accurately states Meridian Semiconductors' total revenue and percentage increase for fiscal 2025, directly supported by the text. |
| q02 | True | Correctly identifies the fastest growing segment and its percentage increase, both supported by the text. |
| q03 | True | Accurately states Meridian's gross margin for fiscal 2025 and the factors that drove the change, all supported by the text. |
| q04 | True | Accurately states the percentage of total revenue contributed by the two and ten largest customers, as supported by the text. |
| q05 | True | Correctly states that sales to China represented 22% of Meridian Semiconductors' total revenue in fiscal 2025, citing the context. |
| q06 | True | Directly quotes the percentage of total net sales from digital commerce in the third quarter as stated in the text. |
