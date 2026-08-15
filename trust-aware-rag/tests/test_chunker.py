import pytest

from src.ingestion.chunker import Chunk, chunk_document, _chunk_words
from src.ingestion.loader import Document, Page


def make_document(page_texts):
    pages = [Page(page_number=i + 1, text=t) for i, t in enumerate(page_texts)]
    return Document(document_id="doc123", document_name="test.txt", source_path="", pages=pages)


def test_chunk_words_basic_windowing():
    words = [str(i) for i in range(10)]
    groups = _chunk_words(words, size=4, overlap=1)
    # step = 3, windows: [0:4], [3:7], [6:10]
    assert groups == [
        ["0", "1", "2", "3"],
        ["3", "4", "5", "6"],
        ["6", "7", "8", "9"],
    ]


def test_chunk_words_rejects_bad_overlap():
    with pytest.raises(ValueError):
        _chunk_words(["a", "b"], size=4, overlap=4)  # overlap must be < size
    with pytest.raises(ValueError):
        _chunk_words(["a", "b"], size=4, overlap=-1)


def test_chunk_words_empty_input():
    assert _chunk_words([], size=4, overlap=1) == []


def test_chunk_document_preserves_metadata():
    doc = make_document(["word " * 20, "other " * 20])
    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2, min_chunk_chars=1)

    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.document_id == "doc123" for c in chunks)
    assert all(c.document_name == "test.txt" for c in chunks)

    page1_chunks = [c for c in chunks if c.page_number == 1]
    page2_chunks = [c for c in chunks if c.page_number == 2]
    assert page1_chunks, "expected at least one chunk from page 1"
    assert page2_chunks, "expected at least one chunk from page 2"
    assert all("word" in c.chunk_text for c in page1_chunks)
    assert all("other" in c.chunk_text for c in page2_chunks)


def test_chunk_ids_are_unique():
    doc = make_document(["word " * 50])
    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2, min_chunk_chars=1)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_min_chunk_chars_drops_tiny_trailing_chunk():
    # 12 "apple" words at size=10, overlap=0 -> windows [0:10] (~59 chars),
    # [10:12] (~11 chars, short). min_chunk_chars=30 keeps the first window
    # and drops the short trailing one.
    doc = make_document(["apple " * 12])
    chunks_keep_short = chunk_document(doc, chunk_size=10, chunk_overlap=0, min_chunk_chars=1)
    chunks_drop_short = chunk_document(doc, chunk_size=10, chunk_overlap=0, min_chunk_chars=30)
    assert len(chunks_keep_short) == 2
    assert len(chunks_drop_short) == 1
