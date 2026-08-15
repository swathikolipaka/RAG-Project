"""
Embedding generation for trust-aware-rag (Phase 1).

Requirements from project spec section 7:
- Load the embedding model only once.
- Cache embeddings (don't regenerate for every question).
- Save/load-able so the KB can be built once and queried many times.

Design notes:
- The SentenceTransformer model is loaded lazily on first use, not at
  import time or __init__ time, and memoized on the instance. This keeps
  `import src.embeddings.embedder` cheap and, importantly, makes the class
  testable without a network connection: tests can construct an Embedder
  and inject a fake `_model` directly, never triggering a real download.
- The on-disk cache is a single pickle file mapping
  sha256(model_name + normalize_flag + text) -> vector (np.ndarray).
  This is intentionally simple (project spec section 2: prioritize working
  over clever) rather than a proper embedding database. For the corpus
  sizes this project targets (a handful of uploaded PDFs), an in-memory
  dict flushed to one pickle file is fast enough and trivial to reason
  about; swap for SQLite/LMDB if the cache grows large.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np


def _cache_key(model_name: str, normalize: bool, text: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"1" if normalize else b"0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class Embedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self._model = None  # lazy-loaded, see `model` property
        self._cache: dict[str, np.ndarray] = {}
        self._cache_path: Optional[Path] = None

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            safe_model_name = model_name.replace("/", "__")
            self._cache_path = self.cache_dir / f"{safe_model_name}.pkl"
            self._load_cache()

    # -- model loading -----------------------------------------------------

    @property
    def model(self):
        """Lazily import + load the SentenceTransformer model, once."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # local import

            self._model = SentenceTransformer(self.model_name)
        return self._model

    # -- cache I/O -----------------------------------------------------------

    def _load_cache(self) -> None:
        if self._cache_path and self._cache_path.exists():
            with open(self._cache_path, "rb") as f:
                self._cache = pickle.load(f)

    def _flush_cache(self) -> None:
        if self._cache_path is not None:
            tmp_path = self._cache_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(self._cache, f)
            tmp_path.replace(self._cache_path)  # atomic-ish on POSIX

    # -- public API ----------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return an (N, dim) array of embeddings for `texts`, using the
        cache wherever possible and only calling the model for cache misses.

        Deduplicates cache misses BOTH against the persistent cache and
        against repeats within this same call - e.g. embed_texts(["a", "b",
        "a"]) with an empty starting cache must call the model with only
        ["a", "b"], not ["a", "b", "a"]. (An earlier version of this method
        only deduped against the persistent cache and re-encoded repeated
        texts within a single call - caught by test_embedder.py.)
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        keys = [_cache_key(self.model_name, self.normalize, t) for t in texts]
        results: List[Optional[np.ndarray]] = [self._cache.get(k) for k in keys]

        # Collect unique missing keys -> one representative text each.
        missing_key_to_text: dict[str, str] = {}
        for i, k in enumerate(keys):
            if results[i] is None and k not in missing_key_to_text:
                missing_key_to_text[k] = texts[i]

        if missing_key_to_text:
            unique_keys = list(missing_key_to_text.keys())
            unique_texts = [missing_key_to_text[k] for k in unique_keys]
            fresh = self.model.encode(
                unique_texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            fresh_by_key = {}
            for k, vec in zip(unique_keys, fresh):
                vec = np.asarray(vec, dtype=np.float32)
                self._cache[k] = vec
                fresh_by_key[k] = vec
            self._flush_cache()

            for i, k in enumerate(keys):
                if results[i] is None:
                    results[i] = fresh_by_key[k]

        return np.vstack(results).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    @property
    def cache_size(self) -> int:
        return len(self._cache)
