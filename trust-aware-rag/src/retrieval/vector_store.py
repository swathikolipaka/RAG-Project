"""
FAISS vector store for trust-aware-rag (Phase 1).

Uses a flat inner-product index (IndexFlatIP). Because embeddings are
L2-normalized (see src/embeddings/embedder.py, normalize=true in
config.yaml), inner product is mathematically equivalent to cosine
similarity - this avoids a separate normalization step at search time and
keeps the index type simple, per project scope (section 2: no need for
IVF/HNSW at this corpus scale; a flat index over a few thousand chunks is
still fast on CPU).

Metadata (document_id, document_name, page_number, chunk_id, chunk_text)
is stored in a parallel list, aligned by FAISS's internal vector id
(0..N-1). This is the simplest correct design for a corpus this size -
if the KB grows into the hundreds of thousands of chunks, moving metadata
into SQLite would be the natural next step.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from src.ingestion.chunker import Chunk

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"


class SearchResult:
    __slots__ = ("chunk", "score")

    def __init__(self, chunk: dict, score: float):
        self.chunk = chunk  # dict with document_id/document_name/page_number/chunk_id/chunk_text
        self.score = score  # cosine similarity, higher = more similar

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SearchResult(score={self.score:.4f}, chunk_id={self.chunk.get('chunk_id')!r})"


class VectorStore:
    def __init__(self, dim: Optional[int] = None):
        self.dim = dim
        self._index: Optional[faiss.Index] = None
        self._metadata: List[dict] = []

    # -- construction ----------------------------------------------------

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self.dim = dim
            self._index = faiss.IndexFlatIP(dim)
        elif dim != self.dim:
            raise ValueError(
                f"Embedding dim mismatch: index built with dim={self.dim}, "
                f"got vectors with dim={dim}"
            )

    def build(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Build the index from scratch (replaces any existing contents)."""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) "
                "must be the same length"
            )
        self._index = None
        self._metadata = []
        if len(chunks) == 0:
            return
        self._ensure_index(embeddings.shape[1])
        self._index.add(embeddings.astype(np.float32))
        self._metadata = [asdict(c) for c in chunks]

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Append more chunks to an existing (or empty) index."""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must be the same length")
        if len(chunks) == 0:
            return
        self._ensure_index(embeddings.shape[1])
        self._index.add(embeddings.astype(np.float32))
        self._metadata.extend(asdict(c) for c in chunks)

    def clear(self) -> None:
        self._index = None
        self._metadata = []

    # -- search ------------------------------------------------------------

    @property
    def size(self) -> int:
        return 0 if self._index is None else self._index.ntotal

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []
        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append(SearchResult(chunk=self._metadata[idx], score=float(score)))
        return results

    # -- persistence ---------------------------------------------------------

    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(index_dir / INDEX_FILENAME))
        with open(index_dir / METADATA_FILENAME, "w", encoding="utf-8") as f:
            json.dump(
                {"dim": self.dim, "metadata": self._metadata}, f, ensure_ascii=False
            )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        index_dir = Path(index_dir)
        index_path = index_dir / INDEX_FILENAME
        metadata_path = index_dir / METADATA_FILENAME

        if not metadata_path.exists():
            raise FileNotFoundError(f"No saved vector store found at {index_dir}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        store = cls(dim=payload.get("dim"))
        store._metadata = payload.get("metadata", [])
        if index_path.exists() and store._metadata:
            store._index = faiss.read_index(str(index_path))
        return store
