"""Vector RAG over this project's own methodology docs (README.md), used
for project-specific rationale questions ("why do you trust X over Y",
"why did the CUSUM ranking change").

Given the small corpus size (a few dozen section-based chunks), no vector
database service is used -- embeddings are computed once via a build-time
script (build_doc_index(), run as `python -m src.agent.doc_retrieval`)
and stored locally as JSON (data/generated/agent_doc_index/chunks.json).
Retrieval at query time is an in-process cosine-similarity search over
that file (numpy), no server involved. There is deliberately no live
re-indexing -- re-run the build script by hand whenever README.md's
methodology content changes; the corpus is static enough that this is a
reasonable, spec-called-for tradeoff rather than standing up
infrastructure this project doesn't need.

Chunking follows the README's own heading structure (## / ### / ####)
rather than an arbitrary fixed-length splitter, since the docs are
already well-organized into one coherent idea per heading.
"""

import json
import os
import re

import numpy as np

from src.agent import llm_backend

README_PATH = "README.md"
INDEX_DIR = "data/generated/agent_doc_index"

# BGE-family embedding models (used by both backends -- BAAI/bge-base-en-v1.5
# locally via sentence-transformers, @cf/baai/bge-base-en-v1.5 on Cloudflare)
# are trained for asymmetric retrieval: queries benefit measurably from this
# exact instruction prefix, documents should NOT have it. This replaces
# Gemini's task_type=RETRIEVAL_QUERY/RETRIEVAL_DOCUMENT parameter (the
# original backend's mechanism for the same asymmetry), applied here in the
# caller instead of the backend interface since neither BGE API takes a
# task_type argument -- the prefix IS the mechanism for this model family.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
INDEX_PATH = os.path.join(INDEX_DIR, "chunks.json")

# Only these top-level (##) sections are included in the corpus -- the
# project's own methodology/rationale writeups (data provenance, the
# honesty disclaimer, the full attrition and spend headline-results
# writeups -- which contain the leakage/structural-confound and CUSUM
# investigation sections as ### / #### subsections -- the cross-component
# partial-correlation writeup, and the explainability agent's own design).
# Deliberately excludes architecture diagrams, setup instructions,
# repository layout, and license -- not rationale/methodology content.
INCLUDED_TOP_LEVEL_SECTIONS = {
    "Data provenance and generation methodology",
    "Honesty / scope disclaimer",
    "Headline evaluation results",
    "Cross-component analysis: does attrition risk relate to spend anomalies?",
    "Explainability agent",
}

HEADING_RE = re.compile(r"^(#{2,4})\s+(.+)$", re.MULTILINE)


def _parse_markdown_sections(md_text: str) -> list:
    """Splits md_text into {level, heading, body} chunks, one per heading
    at level 2-4; body is all text up to the next heading of any of
    those levels."""
    matches = list(HEADING_RE.finditer(md_text))
    chunks = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        body = md_text[start:end].strip()
        chunks.append({"level": level, "heading": heading, "body": body})
    return chunks


def _top_level_heading_of(chunks: list, index: int) -> str:
    for j in range(index, -1, -1):
        if chunks[j]["level"] == 2:
            return chunks[j]["heading"]
    return None


def build_chunks_from_readme() -> list:
    with open(README_PATH) as f:
        md_text = f.read()
    all_chunks = _parse_markdown_sections(md_text)

    selected = []
    for i, c in enumerate(all_chunks):
        top = _top_level_heading_of(all_chunks, i)
        if top not in INCLUDED_TOP_LEVEL_SECTIONS or not c["body"]:
            continue
        section_path = top if c["heading"] == top else f"{top} > {c['heading']}"
        selected.append({"section_path": section_path, "heading": c["heading"], "text": c["body"]})
    return selected


def build_doc_index() -> int:
    """Computes and saves embeddings for every doc chunk. Run as:
    python -m src.agent.doc_retrieval"""
    chunks = build_chunks_from_readme()
    texts = [f"{c['section_path']}\n\n{c['text']}" for c in chunks]
    embeddings = llm_backend.embed_texts(texts)
    for c, e in zip(chunks, embeddings):
        c["embedding"] = e

    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        json.dump(chunks, f)
    return len(chunks)


_cached_index = None


def _load_index() -> list:
    global _cached_index
    if _cached_index is None:
        if not os.path.exists(INDEX_PATH):
            raise RuntimeError(
                f"No doc index found at {INDEX_PATH} -- run `python -m src.agent.doc_retrieval` to build it."
            )
        with open(INDEX_PATH) as f:
            _cached_index = json.load(f)
    return _cached_index


def _cosine_similarity(a: list, b: list) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(np.dot(a_arr, b_arr) / denom) if denom else 0.0


def retrieve_relevant_chunks(question: str, top_k: int = 3) -> list:
    """Returns the top_k most relevant chunks as
    [{"section_path": ..., "text": ..., "score": ...}, ...] -- real
    passages from README.md. The synthesis step must draw its rationale
    answer from these, not invent one."""
    index = _load_index()
    query_embedding = llm_backend.embed_texts([BGE_QUERY_PREFIX + question])[0]
    scored = [
        {
            "section_path": c["section_path"],
            "text": c["text"],
            "score": _cosine_similarity(query_embedding, c["embedding"]),
        }
        for c in index
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    n = build_doc_index()
    print(f"Built doc index with {n} chunks -> {INDEX_PATH}")
