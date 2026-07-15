import pytest

from ingest.chunker import chunk_text


def test_packs_paragraphs_under_budget():
    text = "para one words here.\n\npara two words here.\n\npara three words here."
    chunks = chunk_text(text, "doc", max_words=100, overlap_words=10)
    assert len(chunks) == 1
    assert "para one" in chunks[0].text and "para three" in chunks[0].text


def test_splits_when_budget_exceeded():
    paras = "\n\n".join(f"paragraph {i} " + "word " * 30 for i in range(5))
    chunks = chunk_text(paras, "doc", max_words=70, overlap_words=10)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 70 for c in chunks)


def test_long_paragraph_sliding_window_overlaps():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, "doc", max_words=100, overlap_words=20)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-20:] == second[:20]  # overlap preserved
    assert "w0" in first and "w299" in chunks[-1].text.split()


def test_ids_and_meta():
    chunks = chunk_text("hello world", "docA", {"company": "X"})
    assert chunks[0].chunk_id == "docA::000"
    assert chunks[0].meta == {"company": "X"}


def test_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("x", "doc", max_words=10, overlap_words=10)
