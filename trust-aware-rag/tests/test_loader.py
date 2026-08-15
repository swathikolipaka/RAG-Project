import pymupdf as fitz
import pytest

from src.ingestion.loader import (
    UnsupportedFileTypeError,
    load_document,
)


@pytest.fixture
def sample_txt(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("Water freezes at 0 degrees Celsius.\n\nPython was released in 1991.")
    return p


@pytest.fixture
def sample_pdf(tmp_path):
    """Build a tiny 2-page PDF on the fly, so this test needs no fixture
    binary and no network access - just PyMuPDF, which we already depend on."""
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Page one: water freezes at 0 degrees Celsius.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page two: Python was first released in 1991.")
    doc.save(str(path))
    doc.close()
    return path


def test_load_txt_is_single_page(sample_txt):
    document = load_document(sample_txt)
    assert document.document_name == "sample.txt"
    assert document.num_pages == 1
    assert "0 degrees Celsius" in document.full_text
    assert "1991" in document.full_text


def test_load_pdf_preserves_page_numbers(sample_pdf):
    document = load_document(sample_pdf)
    assert document.num_pages == 2
    assert document.pages[0].page_number == 1
    assert document.pages[1].page_number == 2
    assert "0 degrees Celsius" in document.pages[0].text
    assert "1991" in document.pages[1].text


def test_document_id_is_deterministic(sample_txt):
    doc_a = load_document(sample_txt)
    doc_b = load_document(sample_txt)
    assert doc_a.document_id == doc_b.document_id


def test_unsupported_extension_raises(tmp_path):
    bad_file = tmp_path / "sample.docx"
    bad_file.write_text("not supported yet")
    with pytest.raises(UnsupportedFileTypeError):
        load_document(bad_file)
