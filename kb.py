"""
A small, self-contained knowledge base — same chunk/embed/store/retrieve
pattern as local-rag-assistant, kept minimal here since it's just one tool
this agent has access to (see tools.py: search_knowledge_base).
"""

import json
import math
import os
import sqlite3
from contextlib import contextmanager

import config


def _ensure_data_dir():
    os.makedirs(config.DATA_DIR, exist_ok=True)


@contextmanager
def _connection():
    _ensure_data_dir()
    conn = sqlite3.connect(config.KB_DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with _connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def clear_chunks():
    with _connection() as conn:
        conn.execute("DELETE FROM chunks")
        conn.commit()


def insert_chunks(rows):
    """rows: list of (source, content, embedding) tuples."""
    with _connection() as conn:
        conn.executemany(
            "INSERT INTO chunks (source, content, embedding_json) VALUES (?, ?, ?)",
            [(s, c, json.dumps(e)) for s, c, e in rows],
        )
        conn.commit()


def count_chunks() -> int:
    with _connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def get_all_chunks():
    with _connection() as conn:
        rows = conn.execute("SELECT source, content, embedding_json FROM chunks").fetchall()
    return [{"source": r[0], "content": r[1], "embedding": json.loads(r[2])} for r in rows]


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def top_chunks(query_embedding, top_k=None):
    top_k = top_k or config.KB_TOP_K
    scored = [
        {**c, "score": _cosine_similarity(query_embedding, c["embedding"])}
        for c in get_all_chunks()
    ]
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:top_k]
