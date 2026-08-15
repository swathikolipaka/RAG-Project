"""
trust-aware-rag - Phase 1 demo app.

This is intentionally NOT the full chat interface from project spec
section 20 yet - that requires generation, claim extraction, verification,
hallucination detection, and trust scoring (Phases 2-6), none of which are
implemented in this Phase 1 delivery.

What this Phase 1 app *does* let you verify end-to-end, with your own
files, in a browser: upload -> ingest -> chunk -> embed -> FAISS index ->
raw similarity search. That's the whole Phase 1 pipeline, minus the LLM.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.embeddings.embedder import Embedder  # noqa: E402
from src.ingestion.chunker import chunk_documents  # noqa: E402
from src.ingestion.loader import load_document  # noqa: E402
from src.retrieval.vector_store import VectorStore  # noqa: E402

st.set_page_config(page_title="trust-aware-rag (Phase 1)", layout="wide")

cfg = load_config()


@st.cache_resource(show_spinner=False)
def get_embedder() -> Embedder:
    return Embedder(
        model_name=cfg.get("embedding.model"),
        cache_dir=cfg.resolve_path("embedding.cache_dir"),
        batch_size=cfg.get("embedding.batch_size", 32),
        normalize=cfg.get("embedding.normalize", True),
    )


def get_store() -> VectorStore:
    if "vector_store" not in st.session_state:
        index_dir = cfg.resolve_path("retrieval.index_dir")
        try:
            st.session_state.vector_store = VectorStore.load(index_dir)
        except FileNotFoundError:
            st.session_state.vector_store = VectorStore()
    return st.session_state.vector_store


st.title("trust-aware-rag — Phase 1")
st.caption(
    "Ingestion → Chunking → Embedding → FAISS retrieval. "
    "No generation, claim extraction, verification, or trust scoring yet "
    "(Phases 2–6)."
)

with st.sidebar:
    st.header("Knowledge base")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    top_k = st.number_input(
        "top_k (search results)",
        min_value=1,
        max_value=20,
        value=int(cfg.get("retrieval.top_k", 5)),
    )

    build_clicked = st.button("Build knowledge base", type="primary")
    clear_clicked = st.button("Clear knowledge base")

    if clear_clicked:
        store = VectorStore()
        st.session_state.vector_store = store
        index_dir = cfg.resolve_path("retrieval.index_dir")
        store.save(index_dir)  # persists an empty index
        st.success("Knowledge base cleared.")

    if build_clicked:
        if not uploaded_files:
            st.warning("Upload at least one PDF or TXT file first.")
        else:
            with st.spinner("Ingesting documents..."):
                documents = []
                for uf in uploaded_files:
                    suffix = Path(uf.name).suffix
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(uf.getbuffer())
                        tmp_path = Path(tmp.name)
                    documents.append(load_document(tmp_path))

            with st.spinner("Chunking..."):
                chunks = chunk_documents(
                    documents,
                    chunk_size=cfg.get("chunking.size", 500),
                    chunk_overlap=cfg.get("chunking.overlap", 75),
                    min_chunk_chars=cfg.get("chunking.min_chunk_chars", 20),
                )

            if not chunks:
                st.warning("No text could be extracted from the uploaded file(s).")
            else:
                with st.spinner(
                    f"Embedding {len(chunks)} chunks "
                    f"(first run downloads the model, ~80MB)..."
                ):
                    embedder = get_embedder()
                    vectors = embedder.embed_texts([c.chunk_text for c in chunks])

                store = VectorStore()
                store.build(chunks, vectors)
                index_dir = cfg.resolve_path("retrieval.index_dir")
                store.save(index_dir)
                st.session_state.vector_store = store

                st.success(
                    f"Knowledge base built: {len(documents)} document(s), "
                    f"{len(chunks)} chunks."
                )

    store = get_store()
    st.metric("Chunks in index", store.size)

st.subheader("Search the knowledge base")
query = st.text_input("Query")

if query:
    store = get_store()
    if store.size == 0:
        st.info("Build the knowledge base first (see sidebar).")
    else:
        embedder = get_embedder()
        query_vec = embedder.embed_text(query)
        results = store.search(query_vec, top_k=int(top_k))

        for i, r in enumerate(results, start=1):
            with st.expander(
                f"{i}. {r.chunk['document_name']} "
                f"(page {r.chunk['page_number']}) — similarity {r.score:.3f}",
                expanded=(i == 1),
            ):
                st.write(r.chunk["chunk_text"])
                st.caption(f"chunk_id: {r.chunk['chunk_id']}")

st.divider()
st.caption(
    "This is raw retrieval only — no answer generation, no claim "
    "verification, no trust score. Those arrive in later phases."
)
