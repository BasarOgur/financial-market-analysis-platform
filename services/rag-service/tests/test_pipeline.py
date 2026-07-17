import pytest

from ingest.pipeline import Document, extract_text, ingest_document


def test_extract_text_reads_txt_and_md():
    assert extract_text("notes.txt", b"hello world") == "hello world"
    assert extract_text("notes.md", b"# title\n\nbody") == "# title\n\nbody"


def test_extract_text_rejects_unsupported_extension():
    with pytest.raises(ValueError):
        extract_text("data.csv", b"a,b\n1,2")


def test_extract_text_rejects_malformed_pdf():
    with pytest.raises(ValueError):
        extract_text("broken.pdf", b"not a real pdf")


def test_ingest_document_chunks_and_upserts(eval_collection, fake_embedder):
    doc = Document(doc_id="upload-x", text="Backlog grew 40% in Q3.", meta={"source": "x.txt"})
    stats = ingest_document(doc, eval_collection, fake_embedder)
    assert stats.documents == 1
    assert stats.chunks == 1
    assert eval_collection.count() == 1
