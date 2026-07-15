"""Eval harness for rag-service. Run from the service root:

    python -m eval.run_eval                 # retrieval metrics (works keyless with local embeddings)
    python -m eval.run_eval --answers       # + answer generation, faithfulness judge, abstention

Retrieval: hit@1/3/5 and MRR over the fixture dataset (gold-span matching).
Answers:   faithfulness (LLM judge) on answerable questions; abstention
           correctness on unanswerable ones (exact abstain phrase, deterministic).

Ingests into an ephemeral Chroma collection so eval never touches the
service's persistent store. Writes eval/results.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb

from shared.config import load_settings
from shared.contracts import RagQueryRequest
from shared.embeddings import embeddings_from_settings
from shared.llm import LLMError, llm_from_settings
from shared.logging import get_logger, setup_logging

from ingest.pipeline import ingest
from prompts import ABSTAIN_PHRASE, format_context
from retrieval.retriever import Retriever
from service import RagService

from .judge import judge_faithfulness
from .metrics import hit_at_k, mrr, relevance_at_ranks

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data" / "fixtures"
RESULTS_PATH = HERE / "results.md"
K = 5

log = get_logger("rag.eval")


def load_dataset() -> list[dict]:
    return [json.loads(line) for line in (HERE / "dataset.jsonl").read_text().splitlines() if line.strip()]


def run(with_answers: bool, limit: int | None = None) -> str:
    settings = load_settings()
    setup_logging(settings.log_level)
    embedder = embeddings_from_settings(settings)
    collection = chromadb.EphemeralClient().get_or_create_collection(
        "eval", metadata={"hnsw:space": "cosine"}
    )
    stats = ingest(DATA_DIR, collection, embedder)
    retriever = Retriever(collection, embedder)
    dataset = load_dataset()
    if limit:
        dataset = dataset[:limit]
    answerable = [ex for ex in dataset if ex["answerable"]]
    unanswerable = [ex for ex in dataset if not ex["answerable"]]

    # --- retrieval ---
    retrieved: dict[str, list] = {}
    relevance: list[list[bool]] = []
    per_question: list[dict] = []
    for ex in dataset:
        chunks = retriever.retrieve(ex["question"], top_k=K)
        retrieved[ex["id"]] = chunks
        if ex["answerable"]:
            flags = relevance_at_ranks([c.text for c in chunks], ex["gold_spans"])
            relevance.append(flags)
            rank = flags.index(True) + 1 if any(flags) else None
            per_question.append({"id": ex["id"], "question": ex["question"], "rank": rank})

    retrieval_rows = {
        "hit@1": hit_at_k(relevance, 1),
        "hit@3": hit_at_k(relevance, 3),
        "hit@5": hit_at_k(relevance, 5),
        "MRR": mrr(relevance),
    }

    # --- answers ---
    answer_section = ""
    if with_answers:
        llm = llm_from_settings(settings)  # raises LLMError if unconfigured
        rag = RagService(retriever, llm)
        faithful_count, judged = 0, []
        wrong_abstain = 0
        for ex in answerable:
            resp = rag.query(RagQueryRequest(question=ex["question"], top_k=K))
            if ABSTAIN_PHRASE in (resp.answer or ""):
                wrong_abstain += 1
                judged.append({"id": ex["id"], "faithful": None, "reason": "abstained"})
                continue
            verdict = judge_faithfulness(
                llm, ex["question"], format_context(retrieved[ex["id"]]), resp.answer or ""
            )
            faithful_count += verdict.faithful
            judged.append({"id": ex["id"], "faithful": verdict.faithful, "reason": verdict.reason})
            log.info("%s faithful=%s %s", ex["id"], verdict.faithful, verdict.reason[:80])

        correct_abstain = 0
        for ex in unanswerable:
            resp = rag.query(RagQueryRequest(question=ex["question"], top_k=K))
            correct_abstain += ABSTAIN_PHRASE in (resp.answer or "")

        attempted = len(answerable) - wrong_abstain
        answer_section = "\n".join(
            [
                "\n## Answer quality",
                f"Generator/judge model: `{llm.model}`\n",
                "| metric | value |",
                "|---|---|",
                f"| faithfulness (judged, {attempted} attempted) | {faithful_count}/{attempted} = {faithful_count / max(attempted, 1):.2f} |",
                f"| correct abstention (unanswerable, n={len(unanswerable)}) | {correct_abstain}/{len(unanswerable)} |",
                f"| wrong abstention (answerable) | {wrong_abstain}/{len(answerable)} |",
                "",
                "| id | faithful | judge reason |",
                "|---|---|---|",
                *[f"| {j['id']} | {j['faithful']} | {j['reason']} |" for j in judged],
            ]
        )

    # --- report ---
    detail = "\n".join(
        f"| {r['id']} | {r['question'][:70]} | {r['rank'] or 'miss'} |" for r in per_question
    )
    report = f"""# rag-service eval results

Corpus: {stats.documents} documents, {stats.chunks} chunks. Dataset: {len(answerable)} answerable + {len(unanswerable)} unanswerable questions.
Embeddings: `{embedder.model}` (provider `{settings.embedding_provider}`).

## Retrieval (k={K}, gold-span matching, answerable questions only)

| metric | value |
|---|---|
{chr(10).join(f"| {k} | {v:.2f} |" for k, v in retrieval_rows.items())}

| id | question | first relevant rank |
|---|---|---|
{detail}
{answer_section}
"""
    RESULTS_PATH.write_text(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", action="store_true", help="also run answer generation + judge (needs an LLM key)")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N dataset questions (smoke test / quota-limited runs)")
    args = parser.parse_args()
    try:
        print(run(with_answers=args.answers, limit=args.limit))
    except LLMError as exc:
        raise SystemExit(f"answer eval needs a configured LLM: {exc}") from exc
    print(f"written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
