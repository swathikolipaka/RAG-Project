"""
End-to-end Phase 1 pipeline test: load -> chunk -> embed -> index -> search,
using the REAL configured embedding model (not the fake one used in
test_embedder.py).

This test needs to download sentence-transformers/all-MiniLM-L6-v2 from
Hugging Face on first run (~80MB), so it requires network access to
huggingface.co. In network-restricted environments (including the sandbox
this repo was originally built in) it will skip with a clear reason rather
than fail, and it will run normally in GitHub Codespaces / GitHub Actions,
which have full internet access.
"""

from pathlib import Path

import pytest

from src.config import load_config
from src.embeddings.embedder import Embedder
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_document
from src.retrieval.vector_store import VectorStore

SAMPLE_DOC = Path(__file__).resolve().parent.parent / "data" / "sample" / "water_freezing.txt"


@pytest.fixture(scope="module")
def real_embedder(tmp_path_factory):
    cfg = load_config()
    cache_dir = tmp_path_factory.mktemp("embed_cache")
    embedder = Embedder(
        model_name=cfg.get("embedding.model"),
        cache_dir=cache_dir,
        batch_size=cfg.get("embedding.batch_size", 32),
        normalize=cfg.get("embedding.normalize", True),
    )
    try:
        # Touch .model to force the (network-dependent) load now, so we can
        # skip cleanly instead of failing deep inside a test.
        _ = embedder.model
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a skip gate
        pytest.skip(f"Embedding model unavailable (likely no network): {exc}")
    return embedder


def test_full_phase1_pipeline_on_sample_doc(real_embedder, tmp_path):
    cfg = load_config()

    document = load_document(SAMPLE_DOC)
    assert document.num_pages == 1

    chunks = chunk_documents(
        [document],
        chunk_size=cfg.get("chunking.size", 500),
        chunk_overlap=cfg.get("chunking.overlap", 75),
    )
    assert len(chunks) >= 1

    vectors = real_embedder.embed_texts([c.chunk_text for c in chunks])
    assert vectors.shape[0] == len(chunks)

    store = VectorStore()
    store.build(chunks, vectors)
    store.save(tmp_path)

    reloaded = VectorStore.load(tmp_path)

    query_vec = real_embedder.embed_text("At what temperature does water freeze?")
    results = reloaded.search(query_vec, top_k=1)

    assert results, "expected at least one search result"
    assert "freez" in results[0].chunk["chunk_text"].lower()
