"""
Chunking for trust-aware-rag (Phase 1).

Project spec section 6 asks for "chunk_size: 500 tokens approximately" and
explicitly says not to assume these parameters are optimal. To keep Phase 1
dependency-light (no tokenizer download required), we approximate "tokens"
with whitespace-delimited words, which is a standard, well-understood proxy
(typically ~0.75 tokens/word for English with BPE tokenizers, so 500 "words"
here is a conservative-sized chunk, not an exact token count). This
approximation is a documented limitation - swap in a real tokenizer
(tiktoken / the embedding model's own tokenizer) later if chunk-size
precision turns out to matter for your results.

Chunking is done PER PAGE, not across the whole document. This is a
deliberate Phase 1 simplification: it guarantees every chunk has an
unambiguous single page_number (required by section 6), at the cost of
sometimes producing a short trailing chunk per page. Cross-page chunking
(for documents with short pages) is a documented future improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.ingestion.loader import Document


@dataclass
class Chunk:
    document_id: str
    document_name: str
    page_number: int
    chunk_id: str
    chunk_text: str


def _chunk_words(words: List[str], size: int, overlap: int) -> List[List[str]]:
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk overlap must be >= 0 and < size")

    if not words:
        return []

    chunks = []
    step = size - overlap
    start = 0
    n = len(words)
    while start < n:
        end = min(start + size, n)
        chunks.append(words[start:end])
        if end == n:
            break
        start += step
    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 75,
    min_chunk_chars: int = 20,
) -> List[Chunk]:
    """Chunk every page of `document` into overlapping word-count windows."""
    chunks: List[Chunk] = []
    running_index = 0

    for page in document.pages:
        words = page.text.split()
        word_groups = _chunk_words(words, chunk_size, chunk_overlap)
        for group in word_groups:
            text = " ".join(group)
            if len(text) < min_chunk_chars:
                continue  # drop near-empty trailing fragments
            chunk_id = f"{document.document_id}_p{page.page_number}_c{running_index}"
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    document_name=document.document_name,
                    page_number=page.page_number,
                    chunk_id=chunk_id,
                    chunk_text=text,
                )
            )
            running_index += 1

    return chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 75,
    min_chunk_chars: int = 20,
) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc in documents:
        all_chunks.extend(
            chunk_document(
                doc,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_chars=min_chunk_chars,
            )
        )
    return all_chunks
