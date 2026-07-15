# classifier-service

Financial news classifier: sentiment (`bullish` / `bearish` / `neutral`) plus
multi-label topics (`earnings`, `m&a`, `regulation`, `macro`, `product`,
`other`) per snippet.

Two classification paths behind one interface (`path` field on the request):

- **`embed`** (default) — logistic regression trained at startup on shared
  embeddings. Runs fully keyless with the local ONNX provider.
- **`llm`** — few-shot JSON classification through the shared LLM client.
  Needs an API key; returns 503 without one.

Analysis of disclosed information only — **not investment advice**. Stamped
on every response via the shared `disclaimer` field.

## Run it

```bash
# from root, once:
python3 -m venv .venv && source .venv/bin/activate

cd services/classifier-service
pip install -r requirements.txt   # installs packages/shared editable
cp ../../.env.example .env        # set GEMINI_API_KEY for the llm path

# one-shot demo (no server)
python api.py --demo "Northwind beat quarterly estimates and raised guidance."

# batch (JSONL with "text" field, or plain lines)
python api.py --batch snippets.jsonl --path llm

# serve (trains the embed path on the fixture train split at startup)
python api.py --port 8001
curl -X POST localhost:8001/v1/query -H 'Content-Type: application/json' \
  -d '{"texts": ["Regulators sued to block the merger."], "path": "embed"}'
```

**No API key?** `FMA_EMBEDDING_PROVIDER=local python api.py` runs the embed
path fully offline (bundled ONNX MiniLM). The `llm` path returns 503 until a
key is set.

## Configuration

All via env / `.env` (loaded from the directory you run in):

| var | default | notes |
|---|---|---|
| `FMA_LLM_PROVIDER` | `gemini` | `gemini` \| `openai` \| `anthropic` |
| `FMA_LLM_MODEL` | `gemini-2.5-flash` | llm path + eval |
| `FMA_EMBEDDING_PROVIDER` | `gemini` | `local` = keyless ONNX MiniLM |
| `FMA_EMBEDDING_MODEL` | `gemini-embedding-001` | embed path retrains at startup, so switching is safe |
| `GEMINI_API_KEY` | — | or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` |

## API contract (what the orchestrator consumes)

Defined in `packages/shared/shared/contracts.py`; this service just serves it.

- `GET /healthz`
- `GET /v1/tool-schema` → `ToolDefinition` (`classify_financial_news`) to
  register directly as an LLM tool.
- `POST /v1/query` → `ClassifyRequest {texts (1–50), path: "embed"|"llm"}` →
  `ClassifyResponse {results[], path, model, disclaimer}`. Each result:
  `{text, sentiment, topics[]}` with `Literal`-typed labels — off-vocabulary
  output is a validation error, not a silent string. 503 when the llm path
  has no configured LLM; 502 on provider error.

## Evaluation

```bash
python -m eval.run_eval          # embed path (keyless with FMA_EMBEDDING_PROVIDER=local)
python -m eval.run_eval --llm    # + few-shot LLM baseline (needs key; --limit N for quota)
```

72-item synthetic fixture dataset (`data/news.jsonl`), fixed split: 48 train /
24 test, every topic ≥4× in test, sentiment balanced 8/8/8. Both paths are
scored on the same test split so the table below is a direct
baseline-vs-trained comparison. Full per-label report: `eval/results.md`.

### Results — same 20-item test subset, both paths (`--limit 20` fits the free-tier daily cap, DECISIONS.md #9)

| path | model | sentiment macro F1 | topics macro F1 |
|---|---|---|---|
| embed (trained) | logreg@all-MiniLM-L6-v2 (onnx, local) | 0.65 | 0.79 |
| llm (few-shot) | gemini-2.5-flash-lite | **0.95** | **0.91** |

The few-shot LLM clearly beats the 48-example trained head — expected at
this dataset size (DECISIONS.md #1: the trained path earns its keep as the
keyless/offline/cheap-at-volume option, and closes the gap as training data
grows). Full 24-item run of the embed path alone: sentiment 0.67 / topics
0.77. Per-label tables and miss lists: `eval/results.md`.

## Layout

```
api.py            server + --demo / --batch modes
service.py        ClassifierService: path dispatch (DI for tests/eval)
prompts.py        system prompt + fixed few-shots (llm path)
model/            Classifier protocol, embed_path.py, llm_path.py
data/news.jsonl   72 labeled synthetic snippets, fixed train/test split
eval/             metrics.py (P/R/F1 per label), run_eval.py, results.md
tests/            21 offline tests (fake embedder + fake LLM)
```
