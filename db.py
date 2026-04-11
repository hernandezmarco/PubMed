"""
PostgreSQL + pgvector database layer for RAG collections.
"""
import logging
import os
import time
from contextlib import contextmanager

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

_log = logging.getLogger("pubmed.db")


def _connect():
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
    )
    register_vector(conn)
    return conn


@contextmanager
def db_conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_collections (
                id           SERIAL PRIMARY KEY,
                name         TEXT NOT NULL,
                user_query   TEXT NOT NULL,
                pubmed_query TEXT NOT NULL,
                created_at   TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_articles (
                id            SERIAL PRIMARY KEY,
                collection_id INTEGER REFERENCES rag_collections(id) ON DELETE CASCADE,
                pmid          TEXT NOT NULL,
                title         TEXT NOT NULL,
                authors       TEXT,
                journal       TEXT,
                year          TEXT,
                abstract      TEXT,
                url           TEXT,
                has_full_text BOOLEAN DEFAULT FALSE,
                pmcid         TEXT
            )
        """)
        # Migrate existing tables that pre-date these columns
        cur.execute("ALTER TABLE rag_articles ADD COLUMN IF NOT EXISTS has_full_text BOOLEAN DEFAULT FALSE")
        cur.execute("ALTER TABLE rag_articles ADD COLUMN IF NOT EXISTS pmcid TEXT")

        # Shared chunk store — deduped by (pmid, chunk_index)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS article_chunks (
                id          SERIAL PRIMARY KEY,
                pmid        TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text        TEXT NOT NULL,
                embedding   vector(384),
                UNIQUE(pmid, chunk_index)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS article_chunks_emb_idx
            ON article_chunks USING hnsw (embedding vector_cosine_ops)
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id            SERIAL PRIMARY KEY,
                collection_id INTEGER NOT NULL REFERENCES rag_collections(id) ON DELETE CASCADE,
                title         TEXT NOT NULL,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id              SERIAL PRIMARY KEY,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content         TEXT NOT NULL,
                citations       JSONB,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute(
            "ALTER TABLE conversation_messages ADD COLUMN IF NOT EXISTS citations JSONB"
        )


# ── Write helpers ──────────────────────────────────────────────────────────────

def create_collection(name: str, user_query: str, pubmed_query: str) -> int:
    t0 = time.perf_counter()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO rag_collections (name, user_query, pubmed_query) VALUES (%s, %s, %s) RETURNING id",
            (name, user_query, pubmed_query),
        )
        cid = cur.fetchone()[0]
    _log.info("op=create_collection id=%d name=%r duration=%.3fs", cid, name, time.perf_counter() - t0)
    return cid


def add_article(cid: int, art: dict, *, has_full_text: bool = False, pmcid: str | None = None):
    t0 = time.perf_counter()
    with db_conn() as conn:
        conn.cursor().execute(
            """
            INSERT INTO rag_articles
                (collection_id, pmid, title, authors, journal, year, abstract, url, has_full_text, pmcid)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                cid, art["pmid"], art["title"], art["authors"],
                art["journal"], art["year"], art["abstract"],
                art["url"], has_full_text, pmcid,
            ),
        )
    _log.debug(
        "op=add_article cid=%d pmid=%s full_text=%s duration=%.3fs",
        cid, art["pmid"], has_full_text, time.perf_counter() - t0,
    )


def chunks_exist(pmid: str) -> bool:
    t0 = time.perf_counter()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM article_chunks WHERE pmid = %s LIMIT 1", (pmid,))
        exists = cur.fetchone() is not None
    _log.debug("op=chunks_exist pmid=%s exists=%s duration=%.3fs", pmid, exists, time.perf_counter() - t0)
    return exists


def save_chunks(pmid: str, chunks: list[str], embeddings: list[np.ndarray]):
    t0 = time.perf_counter()
    data = [(pmid, i, text, emb) for i, (text, emb) in enumerate(zip(chunks, embeddings))]
    with db_conn() as conn:
        cur = conn.cursor()
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO article_chunks (pmid, chunk_index, text, embedding)
            VALUES %s
            ON CONFLICT (pmid, chunk_index) DO NOTHING
            """,
            data,
        )
    _log.info("op=save_chunks pmid=%s chunks=%d duration=%.3fs", pmid, len(chunks), time.perf_counter() - t0)


# ── Read helpers ───────────────────────────────────────────────────────────────

def list_collections() -> list[dict]:
    t0 = time.perf_counter()
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT c.id, c.name, c.user_query, c.pubmed_query,
                   c.created_at::text AS created_at,
                   COUNT(DISTINCT a.id)                                    AS article_count,
                   COUNT(DISTINCT a.id) FILTER (WHERE a.has_full_text)    AS full_text_count,
                   COUNT(DISTINCT ac.id)                                   AS chunk_count
            FROM rag_collections c
            LEFT JOIN rag_articles    a  ON a.collection_id = c.id
            LEFT JOIN article_chunks  ac ON ac.pmid = a.pmid
            GROUP BY c.id
            ORDER BY c.created_at DESC
        """)
        rows = [dict(r) for r in cur.fetchall()]
    _log.info("op=list_collections rows=%d duration=%.3fs", len(rows), time.perf_counter() - t0)
    return rows


def get_collection(cid: int) -> dict | None:
    t0 = time.perf_counter()
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT c.id, c.name, c.user_query, c.pubmed_query,
                   c.created_at::text AS created_at,
                   COUNT(DISTINCT a.id)                                    AS article_count,
                   COUNT(DISTINCT a.id) FILTER (WHERE a.has_full_text)    AS full_text_count,
                   COUNT(DISTINCT ac.id)                                   AS chunk_count
            FROM rag_collections c
            LEFT JOIN rag_articles    a  ON a.collection_id = c.id
            LEFT JOIN article_chunks  ac ON ac.pmid = a.pmid
            WHERE c.id = %s
            GROUP BY c.id
        """, (cid,))
        row = cur.fetchone()
    result = dict(row) if row else None
    _log.info("op=get_collection cid=%d found=%s duration=%.3fs", cid, result is not None, time.perf_counter() - t0)
    return result


def get_collection_articles(cid: int) -> list[dict]:
    t0 = time.perf_counter()
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, pmid, title, authors, journal, year, abstract, url, has_full_text, pmcid "
            "FROM rag_articles WHERE collection_id = %s ORDER BY id",
            (cid,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    _log.info("op=get_collection_articles cid=%d rows=%d duration=%.3fs", cid, len(rows), time.perf_counter() - t0)
    return rows


def semantic_search(cid: int, embedding: np.ndarray, k: int = 5) -> list[dict]:
    """Return the top-k most relevant chunks across all articles in the collection."""
    t0 = time.perf_counter()
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT ra.pmid, ra.title, ra.authors, ra.journal, ra.year,
                   ra.abstract, ra.url, ra.has_full_text,
                   ac.text AS chunk_text, ac.chunk_index,
                   ROUND((1 - (ac.embedding <=> %s))::numeric, 3) AS similarity
            FROM article_chunks ac
            JOIN rag_articles ra ON ra.pmid = ac.pmid AND ra.collection_id = %s
            ORDER BY ac.embedding <=> %s
            LIMIT %s
            """,
            (embedding, cid, embedding, k),
        )
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            r["similarity"] = float(r["similarity"])
    elapsed = time.perf_counter() - t0
    top_sim = rows[0]["similarity"] if rows else 0.0
    _log.info(
        "op=semantic_search cid=%d k=%d hits=%d top_similarity=%.3f duration=%.3fs",
        cid, k, len(rows), top_sim, elapsed,
    )
    _log.debug("op=semantic_search pmids=%s", [r["pmid"] for r in rows])
    return rows


def delete_collection(cid: int):
    t0 = time.perf_counter()
    with db_conn() as conn:
        conn.cursor().execute("DELETE FROM rag_collections WHERE id = %s", (cid,))
    _log.info("op=delete_collection cid=%d duration=%.3fs", cid, time.perf_counter() - t0)


# ── Conversation helpers ───────────────────────────────────────────────────────

def create_conversation(cid: int, title: str) -> int:
    t0 = time.perf_counter()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversations (collection_id, title) VALUES (%s, %s) RETURNING id",
            (cid, title[:120]),
        )
        vid = cur.fetchone()[0]
    _log.info("op=create_conversation cid=%d vid=%d duration=%.3fs", cid, vid, time.perf_counter() - t0)
    return vid


def add_message(vid: int, role: str, content: str, citations: list | None = None):
    t0 = time.perf_counter()
    with db_conn() as conn:
        conn.cursor().execute(
            "INSERT INTO conversation_messages (conversation_id, role, content, citations) "
            "VALUES (%s, %s, %s, %s)",
            (vid, role, content, psycopg2.extras.Json(citations) if citations is not None else None),
        )
    _log.debug("op=add_message vid=%d role=%s duration=%.3fs", vid, role, time.perf_counter() - t0)


def list_conversations(cid: int) -> list[dict]:
    t0 = time.perf_counter()
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT v.id, v.title, v.created_at::text AS created_at,
                   COUNT(m.id) AS message_count
            FROM conversations v
            LEFT JOIN conversation_messages m ON m.conversation_id = v.id
            WHERE v.collection_id = %s
            GROUP BY v.id
            ORDER BY v.created_at DESC
            """,
            (cid,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    _log.info("op=list_conversations cid=%d rows=%d duration=%.3fs", cid, len(rows), time.perf_counter() - t0)
    return rows


def get_conversation(vid: int) -> dict | None:
    t0 = time.perf_counter()
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT v.id, v.title, v.created_at::text AS created_at,
                   c.name AS collection_name
            FROM conversations v
            JOIN rag_collections c ON c.id = v.collection_id
            WHERE v.id = %s
            """,
            (vid,),
        )
        row = cur.fetchone()
    result = dict(row) if row else None
    _log.info("op=get_conversation vid=%d found=%s duration=%.3fs", vid, result is not None, time.perf_counter() - t0)
    return result


def get_conversation_messages(vid: int) -> list[dict]:
    t0 = time.perf_counter()
    with db_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, role, content, citations, created_at::text AS created_at "
            "FROM conversation_messages WHERE conversation_id = %s ORDER BY id",
            (vid,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    _log.info("op=get_conversation_messages vid=%d rows=%d duration=%.3fs", vid, len(rows), time.perf_counter() - t0)
    return rows


def delete_conversation(vid: int):
    t0 = time.perf_counter()
    with db_conn() as conn:
        conn.cursor().execute("DELETE FROM conversations WHERE id = %s", (vid,))
    _log.info("op=delete_conversation vid=%d duration=%.3fs", vid, time.perf_counter() - t0)
