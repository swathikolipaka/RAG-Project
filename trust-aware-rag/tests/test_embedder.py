import numpy as np
import pytest

from src.embeddings.embedder import Embedder, _cache_key


class FakeModel:
    """Stands in for a real SentenceTransformer so these tests never touch
    the network / never download model weights. Returns a deterministic,
    distinguishable vector per input text (hash-based, dim=8)."""

    def __init__(self):
        self.encode_call_count = 0
        self.last_batch = None

    def encode(self, texts, batch_size=32, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=False):
        self.encode_call_count += 1
        self.last_batch = list(texts)
        vectors = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            vec = rng.random(8).astype(np.float32)
            if normalize_embeddings:
                vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
        return np.vstack(vectors)


@pytest.fixture
def embedder(tmp_path):
    e = Embedder(model_name="fake/model", cache_dir=tmp_path, batch_size=8)
    e._model = FakeModel()  # inject fake model, skip real loading entirely
    return e


def test_embed_texts_calls_model_once_per_unique_text(embedder):
    vectors = embedder.embed_texts(["hello", "world", "hello"])
    assert vectors.shape == (3, 8)
    # "hello" appears twice but the fake model should only ever see 2 unique texts
    assert embedder._model.encode_call_count == 1
    assert sorted(embedder._model.last_batch) == ["hello", "world"]
    # cached repeat must be byte-identical to the first occurrence
    np.testing.assert_array_equal(vectors[0], vectors[2])


def test_embed_texts_second_call_hits_cache_only(embedder):
    embedder.embed_texts(["a", "b"])
    assert embedder._model.encode_call_count == 1
    embedder.embed_texts(["a", "b"])  # should be a pure cache hit
    assert embedder._model.encode_call_count == 1  # unchanged


def test_cache_persists_across_instances(tmp_path):
    e1 = Embedder(model_name="fake/model", cache_dir=tmp_path, batch_size=8)
    e1._model = FakeModel()
    v1 = e1.embed_text("persisted text")

    e2 = Embedder(model_name="fake/model", cache_dir=tmp_path, batch_size=8)
    e2._model = FakeModel()  # fresh fake model, would give a different vector if called
    v2 = e2.embed_text("persisted text")

    np.testing.assert_array_equal(v1, v2)
    assert e2._model.encode_call_count == 0  # never had to call the model


def test_embed_texts_empty_list_returns_empty_array(embedder):
    result = embedder.embed_texts([])
    assert result.shape == (0, 0)


def test_cache_key_is_sensitive_to_model_and_normalize_flag():
    k1 = _cache_key("model-a", True, "hello")
    k2 = _cache_key("model-b", True, "hello")
    k3 = _cache_key("model-a", False, "hello")
    assert len({k1, k2, k3}) == 3
