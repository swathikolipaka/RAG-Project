import numpy as np
import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import VectorStore


def make_chunks(n):
    return [
        Chunk(
            document_id="doc1",
            document_name="doc1.txt",
            page_number=1,
            chunk_id=f"chunk_{i}",
            chunk_text=f"this is chunk number {i}",
        )
        for i in range(n)
    ]


def normalized_random_vectors(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def test_build_and_search_returns_closest_vector():
    chunks = make_chunks(5)
    vectors = normalized_random_vectors(5, dim=16)

    store = VectorStore()
    store.build(chunks, vectors)
    assert store.size == 5

    # querying with a chunk's own vector should return that chunk first,
    # with similarity ~1.0
    results = store.search(vectors[2], top_k=3)
    assert len(results) == 3
    assert results[0].chunk["chunk_id"] == "chunk_2"
    assert results[0].score == pytest.approx(1.0, abs=1e-4)
    # results should be sorted descending by score
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_on_empty_store_returns_empty_list():
    store = VectorStore()
    query = normalized_random_vectors(1, dim=16)[0]
    assert store.search(query, top_k=5) == []


def test_build_rejects_mismatched_lengths():
    chunks = make_chunks(3)
    vectors = normalized_random_vectors(2, dim=16)
    store = VectorStore()
    with pytest.raises(ValueError):
        store.build(chunks, vectors)


def test_add_appends_to_existing_index():
    store = VectorStore()
    store.build(make_chunks(2), normalized_random_vectors(2, dim=8, seed=1))
    store.add(
        [Chunk("doc2", "doc2.txt", 1, "chunk_extra", "extra chunk")],
        normalized_random_vectors(1, dim=8, seed=2),
    )
    assert store.size == 3


def test_save_and_load_round_trip(tmp_path):
    chunks = make_chunks(4)
    vectors = normalized_random_vectors(4, dim=12, seed=42)

    store = VectorStore()
    store.build(chunks, vectors)
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    assert loaded.size == store.size
    assert loaded.dim == store.dim

    results = loaded.search(vectors[1], top_k=1)
    assert results[0].chunk["chunk_id"] == "chunk_1"


def test_load_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "does_not_exist")


def test_clear_empties_store():
    store = VectorStore()
    store.build(make_chunks(2), normalized_random_vectors(2, dim=8))
    store.clear()
    assert store.size == 0
    assert store.search(normalized_random_vectors(1, dim=8)[0], top_k=5) == []
