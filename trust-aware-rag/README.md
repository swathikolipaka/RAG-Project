# trust-aware-rag

Evidence-based hallucination detection and trust-aware response verification
for RAG systems.

> **Status: Phase 1 of 10 complete.** This repo currently implements
> document ingestion, chunking, embedding (with caching), and FAISS
> retrieval — nothing more. Generation, claim extraction, verification,
> hallucination detection, and trust scoring are **not implemented yet**
> (see [Roadmap](#roadmap)). Modules for those phases exist as clearly
> labeled stubs that raise `NotImplementedError` on import, so it's
> unambiguous what's real.

## What Phase 1 actually does

```
Upload PDF/TXT
     │
     ▼
Load + clean text (per page)
     │
     ▼
Chunk (configurable size/overlap, page-aware)
     │
     ▼
Embed (sentence-transformers/all-MiniLM-L6-v2, disk-cached)
     │
     ▼
FAISS index (build once, save, reload, search)
```

You can run this today via `streamlit run app/streamlit_app.py`: upload a
document, build the knowledge base, and run similarity search over it.
There is no LLM call anywhere in Phase 1 — the app only proves the
retrieval half of the pipeline works.

## Quickstart — GitHub Codespaces (recommended)

1. Open this repo in a Codespace (`Code` → `Codespaces` → `Create codespace on main`).
   The devcontainer installs dependencies automatically via `postCreateCommand`.
2. If it didn't run automatically:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. Run the tests:
   ```bash
   pytest -v
   ```
4. Run the app:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Codespaces will prompt you to open the forwarded port (8501) in a browser.

## Quickstart — local / any environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Optional but recommended: install the CPU-only torch wheel first so
# sentence-transformers doesn't pull a much larger CUDA build.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-dev.txt
cp .env.example .env   # not required for Phase 1, but future phases need it

pytest -v
streamlit run app/streamlit_app.py
```

## Configuration

All tunable values live in `config.yaml` — see the file itself, it's
commented section by section. Phase 1 only reads `embedding.*`,
`chunking.*`, `retrieval.*`, `paths.*`, and `logging.*`. The `verification`,
`hallucination`, `trust`, and `generation` sections are present in the
schema now (so later phases don't require a breaking config change) but
are not consumed by any code yet — their default values are placeholders,
not validated settings. Do not cite them as tuned.

Secrets (API keys) go in `.env` (copy from `.env.example`), never in
`config.yaml` and never committed — see `.gitignore`.

## Project structure

```
trust-aware-rag/
├── app/streamlit_app.py       # Phase 1 demo UI (upload, build KB, search)
├── src/
│   ├── config.py               # config.yaml + .env loader
│   ├── ingestion/               # loader.py, chunker.py            [Phase 1]
│   ├── embeddings/embedder.py   # cached sentence-transformers wrapper [Phase 1]
│   ├── retrieval/vector_store.py# FAISS index wrapper               [Phase 1]
│   ├── generation/llm.py        # stub                              [Phase 2]
│   ├── claim_extraction/        # stub                              [Phase 3]
│   ├── verification/            # stub                              [Phase 4]
│   ├── hallucination/           # stub                              [Phase 5]
│   └── trust_score/             # stub                              [Phase 6]
├── evaluation/                  # empty, populated from Phase 8 on
├── tests/                       # unit + one real end-to-end test
├── data/sample/                 # tiny sample doc used by the e2e test
├── .devcontainer/                # Codespaces config
└── .github/workflows/tests.yml   # CI: install, import-check, pytest
```

## Testing

```bash
pytest -v
```

Most tests are fully offline (config, loader, chunker, vector store, and
embedder-caching logic all use synthetic data or an injected fake model —
see `tests/test_embedder.py`). One test,
`tests/test_end_to_end_phase1.py`, runs the *real* embedding model
end-to-end and needs to download `all-MiniLM-L6-v2` from Hugging Face on
first run (~80MB). It will **skip with a clear message**, not fail, in any
environment without network access to `huggingface.co`; it runs normally
on GitHub Codespaces and in the GitHub Actions CI workflow, both of which
have outbound internet access.

## Known Phase 1 limitations (documented on purpose)

- **Chunk size is word-count, not true token-count.** No tokenizer
  dependency is required for Phase 1; ~500 "words" is a conservative proxy
  for ~500 tokens, not an exact figure. See `src/ingestion/chunker.py`.
- **Chunking never crosses a page boundary**, even for PDFs with very
  short pages. This guarantees an unambiguous `page_number` per chunk at
  the cost of occasional short trailing chunks per page.
- **TXT files have no native pagination** and are treated as a single
  page. Page-number citations for TXT sources will always say "page 1."
- **The embedding cache is a single pickle file per model**, fine at the
  scale of a handful of uploaded documents; would need to move to
  SQLite/LMDB if the corpus grows large.
- **DOCX is not supported** (matches the original project scope — may be
  added later).

## Roadmap

See `config.yaml` and the phase stubs in `src/` for what's next: basic RAG
generation (Phase 2), claim extraction (Phase 3), evidence verification
(Phase 4), hallucination detection (Phase 5), trust score (Phase 6), full
Streamlit UI (Phase 7), evaluation framework (Phase 8), ablation study
(Phase 9), error analysis (Phase 10).

## License

MIT — see `LICENSE`.
