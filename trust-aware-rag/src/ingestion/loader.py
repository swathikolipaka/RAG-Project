"""
Document loading for trust-aware-rag (Phase 1).

Supports PDF (via PyMuPDF) and TXT. DOCX is explicitly out of scope for now
(project spec section 6: "DOCX may be added later").

Every loaded document is normalized into a `Document` dataclass with
page-level text, because downstream chunking and evidence citation need to
know which page a chunk came from (section 6: each chunk must preserve
page_number).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pymupdf as fitz  # PyMuPDF; `fitz` is the legacy import name and is
# deprecated as of recent PyMuPDF releases in favor of `import pymupdf`, but
# we keep the `fitz` alias locally so the rest of this module's code doesn't
# change - only the import line does.

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


@dataclass
class Page:
    page_number: int  # 1-indexed
    text: str


@dataclass
class Document:
    document_id: str
    document_name: str
    source_path: str
    pages: List[Page] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    @property
    def num_pages(self) -> int:
        return len(self.pages)


class UnsupportedFileTypeError(ValueError):
    pass


def _clean_text(text: str) -> str:
    """Light cleaning only, per project scope (section 2: don't overcomplicate).

    - Normalize unicode (NFKC) so visually-identical characters compare equal.
    - Strip null bytes (occasionally present in poorly-encoded PDFs/TXT).
    - Collapse runs of whitespace, but preserve paragraph breaks.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    # Collapse 3+ newlines to a double newline (paragraph break), and
    # collapse runs of spaces/tabs to a single space.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _document_id(document_name: str, content_sample: str) -> str:
    """Deterministic ID from filename + a content sample, so re-ingesting the
    same file twice (e.g. after restarting the app) produces the same ID and
    the embedding cache (keyed partly off content) still hits."""
    h = hashlib.sha256()
    h.update(document_name.encode("utf-8"))
    h.update(content_sample[:2000].encode("utf-8"))
    return h.hexdigest()[:16]


def load_pdf(path: Path) -> Document:
    document_name = path.name
    pages: List[Page] = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")
            cleaned = _clean_text(raw_text)
            pages.append(Page(page_number=i, text=cleaned))

    content_sample = pages[0].text if pages else ""
    doc_id = _document_id(document_name, content_sample)
    return Document(
        document_id=doc_id,
        document_name=document_name,
        source_path=str(path),
        pages=pages,
    )


def load_txt(path: Path) -> Document:
    document_name = path.name
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    cleaned = _clean_text(raw_text)
    # TXT has no native pagination; treat the whole file as page 1.
    # (This is a documented Phase 1 simplification - see README limitations.)
    pages = [Page(page_number=1, text=cleaned)]
    doc_id = _document_id(document_name, cleaned)
    return Document(
        document_id=doc_id,
        document_name=document_name,
        source_path=str(path),
        pages=pages,
    )


def load_document(path: Path) -> Document:
    """Dispatch to the right loader based on file extension."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".txt":
        return load_txt(path)
    raise UnsupportedFileTypeError(
        f"Unsupported file type '{ext}' for {path.name}. "
        f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def load_documents(paths: List[Path]) -> List[Document]:
    return [load_document(p) for p in paths]
