from src.config import load_config


def test_load_config_reads_expected_sections():
    cfg = load_config()
    assert cfg.get("embedding.model") == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.get("chunking.size") == 500
    assert cfg.get("chunking.overlap") == 75
    assert cfg.get("retrieval.top_k") == 5


def test_get_missing_key_returns_default():
    cfg = load_config()
    assert cfg.get("does.not.exist", default="fallback") == "fallback"
    assert cfg.get("does.not.exist") is None


def test_resolve_path_is_relative_to_repo_root():
    cfg = load_config()
    p = cfg.resolve_path("retrieval.index_dir")
    assert p.is_absolute()
    assert p.name == "index"
    assert p.parent.name == "data"
