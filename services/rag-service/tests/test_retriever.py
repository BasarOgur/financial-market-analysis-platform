from ingest.chunker import chunk_text
from retrieval.retriever import Retriever


def _ingest(collection, embedder, docs: dict[str, str]):
    chunks = [c for doc_id, text in docs.items() for c in chunk_text(text, doc_id)]
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embedder.embed([c.text for c in chunks]),
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.doc_id} for c in chunks],
    )


def test_retrieves_matching_chunk_first(eval_collection, fake_embedder):
    _ingest(
        eval_collection,
        fake_embedder,
        {
            "revenue": "total revenue grew eighteen percent to record levels",
            "margin": "gross margin expanded on favorable product mix",
            "risk": "customer concentration remains a material risk factor",
        },
    )
    retriever = Retriever(eval_collection, fake_embedder)
    results = retriever.retrieve("total revenue grew how much", top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id.startswith("revenue::")
    assert results[0].score >= results[1].score
    assert results[0].meta["source"] == "revenue"
