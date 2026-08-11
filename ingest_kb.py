"""
Build the small local knowledge base from documents/*.txt.
Same chunk-then-embed pattern as local-rag-assistant/ingest.py, simplified.

Run directly:
    python ingest_kb.py
"""

import glob
import os

import config
import kb
import llm


def _read_documents():
    paths = sorted(glob.glob(os.path.join(config.DOCUMENTS_DIR, "*.txt")))
    docs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


def _chunk_text(text, max_chars=800):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def run_ingestion(progress_callback=None):
    docs = _read_documents()
    if not docs:
        raise RuntimeError(f"No .txt files found in {config.DOCUMENTS_DIR}")

    all_chunks = []
    for source, text in docs:
        for chunk in _chunk_text(text):
            all_chunks.append((source, chunk))

    llm.initialize(progress_callback=progress_callback, load_embedding_model=True)

    texts = [c[1] for c in all_chunks]
    embeddings = llm.embed_texts(texts)

    kb.init_db()
    kb.clear_chunks()
    kb.insert_chunks([(s, c, e) for (s, c), e in zip(all_chunks, embeddings)])

    return len(all_chunks)


if __name__ == "__main__":
    kb.init_db()
    n = run_ingestion()
    print(f"\nIndexed {n} chunks into {config.KB_DB_PATH}")
