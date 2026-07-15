# classifier-service eval results

Test split: 20 held-out snippets (train: 48). Same fixture set for both paths.

| path | model | sentiment macro F1 | topics macro F1 |
|---|---|---|---|
| Trained path (logreg on embeddings) | `logreg@all-MiniLM-L6-v2 (onnx, local)` | 0.65 | 0.79 |
| Few-shot LLM baseline | `gemini-2.5-flash-lite` | 0.95 | 0.91 |

## Trained path (logreg on embeddings) — `logreg@all-MiniLM-L6-v2 (onnx, local)`

### Sentiment (macro F1 0.65)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| bullish | 0.50 | 0.57 | 0.53 | 7 |
| bearish | 0.86 | 0.86 | 0.86 | 7 |
| neutral | 0.60 | 0.50 | 0.55 | 6 |

### Topics (macro F1 0.79)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| earnings | 1.00 | 1.00 | 1.00 | 5 |
| m&a | 0.67 | 0.67 | 0.67 | 3 |
| regulation | 0.80 | 1.00 | 0.89 | 4 |
| macro | 0.75 | 0.75 | 0.75 | 4 |
| product | 0.75 | 0.60 | 0.67 | 5 |
| other | 0.60 | 1.00 | 0.75 | 3 |

**Misses (gold vs predicted):**

| id | gold | predicted |
|---|---|---|
| t04 | bullish/m&a | bullish/other |
| t05 | bearish/m&a,regulation | bearish/m&a,regulation,product |
| t08 | bullish/regulation,product | neutral/regulation |
| t09 | neutral/regulation | bullish/regulation,macro |
| t11 | bullish/macro | bearish/macro |
| t14 | bearish/product | bullish/regulation,product,other |
| t15 | neutral/product | bullish/product |
| t16 | bullish/other | neutral/m&a,other |
| t18 | neutral/other | bullish/other |
| t19 | bullish/earnings,product | bullish/earnings |
| t20 | bearish/earnings,macro | bearish/earnings |


## Few-shot LLM baseline — `gemini-2.5-flash-lite`

### Sentiment (macro F1 0.95)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| bullish | 1.00 | 0.86 | 0.92 | 7 |
| bearish | 1.00 | 1.00 | 1.00 | 7 |
| neutral | 0.86 | 1.00 | 0.92 | 6 |

### Topics (macro F1 0.91)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| earnings | 0.83 | 1.00 | 0.91 | 5 |
| m&a | 1.00 | 1.00 | 1.00 | 3 |
| regulation | 1.00 | 1.00 | 1.00 | 4 |
| macro | 1.00 | 1.00 | 1.00 | 4 |
| product | 0.83 | 1.00 | 0.91 | 5 |
| other | 0.50 | 1.00 | 0.67 | 3 |

**Misses (gold vs predicted):**

| id | gold | predicted |
|---|---|---|
| t04 | bullish/m&a | bullish/m&a,product |
| t06 | neutral/m&a | neutral/m&a,other |
| t07 | bearish/regulation | bearish/regulation,other |
| t09 | neutral/regulation | neutral/regulation,other |
| t10 | bearish/macro | bearish/macro,earnings |
| t16 | bullish/other | neutral/other |
