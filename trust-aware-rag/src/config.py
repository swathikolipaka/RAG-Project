"""
Central configuration loader for trust-aware-rag.

Design goals (see project spec section 33):
- A single source of truth: config.yaml for tunable thresholds/paths,
  environment variables (.env) for secrets and deployment-specific values.
- Nothing in this file talks to the network, a model, or FAISS. It is a
  pure data-loading module so every other module can depend on it safely
  and it's trivial to unit test.

Usage:
    from src.config import load_config
    cfg = load_config()
    cfg.get("embedding.model")
    cfg.get("retrieval.top_k", default=5)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

# Repo root = two levels up from this file (src/config.py -> src -> repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


class Config:
    """Thin wrapper around a nested dict with dot-path access.

    Deliberately NOT a big validation framework - the project spec
    prioritizes "working project" over engineering polish (section 2).
    """

    def __init__(self, data: dict):
        self._data = data or {}

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Look up a value using a dotted path, e.g. 'embedding.model'."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def resolve_path(self, dotted_key: str, default: Optional[str] = None) -> Path:
        """Like get(), but resolves the result relative to the repo root.

        Config paths (e.g. 'data/index') are always written relative to the
        repo root so the project behaves the same whether you run it from
        the repo root, a Codespace, or app/streamlit_app.py's own cwd.
        """
        raw = self.get(dotted_key, default)
        if raw is None:
            raise KeyError(f"No config value (and no default) for '{dotted_key}'")
        p = Path(raw)
        return p if p.is_absolute() else (REPO_ROOT / p)

    def as_dict(self) -> dict:
        return self._data

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config({self._data!r})"


def load_config(
    config_path: Optional[Path] = None,
    env_path: Optional[Path] = None,
) -> Config:
    """Load config.yaml and .env (if present) and return a Config object.

    Missing .env is not an error (e.g. in CI, or before a contributor has
    copied .env.example -> .env) - it just means environment-variable
    overrides won't be available, which is fine for Phase 1 since nothing
    yet requires secrets.
    """
    config_path = config_path or DEFAULT_CONFIG_PATH
    env_path = env_path or DEFAULT_ENV_PATH

    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Did you mean to pass an explicit config_path?"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return Config(data)


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read an environment variable. Centralized so secrets are never read
    ad-hoc from os.environ scattered across the codebase (makes it easy to
    audit that nothing gets logged - see section 34)."""
    return os.environ.get(name, default)
